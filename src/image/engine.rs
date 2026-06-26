#![allow(clippy::cast_precision_loss)]

use crate::{
    configuration::guild::{
        DetectionActions, DetectionPolicy, TextGatePolicy, normalize_text_gate_pattern,
    },
    image::{
        matcher::Matcher,
        pipeline::PreparedOcrCrop,
        types::{ImageCandidate, ImageFingerprint, MatchOutcome},
    },
};
use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualCandidateClass {
    KnownStrong,
    KnownSuspicious,
    NoEvidence,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextGateDecision {
    Disabled,
    OcrPending,
    OcrUnavailable,
    NoOcrText,
    ConfirmedSentence,
    ConfirmedKeywords,
    PartialKeywords,
    Rejected,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextGateVerdict {
    Disabled,
    Unknown,
    Bad,
    Good,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextGateReport {
    pub readable: bool,
    pub keyword_hits: usize,
    pub keyword_threshold: usize,
    pub sentence_hit: bool,
    pub matched_keywords: Vec<String>,
    pub matched_sentences: Vec<String>,
    pub ocr_text: Option<String>,
    pub decision: TextGateDecision,
    pub verdict: TextGateVerdict,
    pub confidence: f32,
    pub error: Option<String>,
}

impl TextGateReport {
    pub(crate) fn disabled(policy: &TextGatePolicy) -> Self {
        Self::empty(
            policy.keyword_threshold,
            TextGateDecision::Disabled,
            TextGateVerdict::Disabled,
            1.0,
            None,
        )
    }

    pub(crate) fn pending(keyword_threshold: usize) -> Self {
        Self::empty(
            keyword_threshold,
            TextGateDecision::OcrPending,
            TextGateVerdict::Unknown,
            0.0,
            Some("OCR is running".to_owned()),
        )
    }

    pub(crate) fn unavailable(keyword_threshold: usize, error: &str) -> Self {
        Self::empty(
            keyword_threshold,
            TextGateDecision::OcrUnavailable,
            TextGateVerdict::Unknown,
            0.0,
            Some(truncate_error(error)),
        )
    }

    fn no_ocr_text(policy: &TextGatePolicy, text: String) -> Self {
        Self {
            ocr_text: Some(text),
            ..Self::empty(
                policy.keyword_threshold,
                TextGateDecision::NoOcrText,
                TextGateVerdict::Unknown,
                0.0,
                None,
            )
        }
    }

    fn empty(
        keyword_threshold: usize,
        decision: TextGateDecision,
        verdict: TextGateVerdict,
        confidence: f32,
        error: Option<String>,
    ) -> Self {
        Self {
            readable: false,
            keyword_hits: 0,
            keyword_threshold,
            sentence_hit: false,
            matched_keywords: Vec::new(),
            matched_sentences: Vec::new(),
            ocr_text: None,
            decision,
            verdict,
            confidence,
            error,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProgressiveDecision {
    pub class: VisualCandidateClass,
    pub outcome: Option<MatchOutcome>,
    pub text_gate: Option<TextGateReport>,
    pub ocr_requested: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize)]
pub struct ModerationIntent {
    pub candidate: ImageCandidate,
    pub image_id: String,
    pub outcome: MatchOutcome,
    pub actions: DetectionActions,
}

pub trait OcrClient: Send + Sync {
    fn read_text<'a>(&'a self, crops: &'a [PreparedOcrCrop]) -> BoxFuture<'a, Result<OcrResponse>>;
}

#[derive(Debug, Clone)]
pub struct OcrResponse {
    pub readable: bool,
    pub text: String,
}

#[allow(dead_code)]
pub struct NoopOcrClient;

impl OcrClient for NoopOcrClient {
    fn read_text<'a>(
        &'a self,
        _crops: &'a [PreparedOcrCrop],
    ) -> BoxFuture<'a, Result<OcrResponse>> {
        Box::pin(async {
            Ok(OcrResponse {
                readable: false,
                text: String::new(),
            })
        })
    }
}

pub struct UnavailableOcrClient {
    reason: String,
}

impl UnavailableOcrClient {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl OcrClient for UnavailableOcrClient {
    fn read_text<'a>(
        &'a self,
        _crops: &'a [PreparedOcrCrop],
    ) -> BoxFuture<'a, Result<OcrResponse>> {
        Box::pin(async move { Err(anyhow::anyhow!(self.reason.clone())) })
    }
}

#[derive(Debug, Clone)]
pub struct StaticOcrClient {
    text: String,
}

impl StaticOcrClient {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl OcrClient for StaticOcrClient {
    fn read_text<'a>(
        &'a self,
        _crops: &'a [PreparedOcrCrop],
    ) -> BoxFuture<'a, Result<OcrResponse>> {
        Box::pin(async move {
            Ok(OcrResponse {
                readable: !self.text.trim().is_empty(),
                text: self.text.clone(),
            })
        })
    }
}

pub trait ArtifactSink: Send + Sync {
    fn write_json<'a, T>(&'a self, name: &'a str, value: &'a T) -> BoxFuture<'a, Result<()>>
    where
        T: Serialize + Sync + ?Sized;

    fn write_bytes<'a>(&'a self, name: &'a str, bytes: &'a [u8]) -> BoxFuture<'a, Result<()>>;
}

pub struct NoopArtifactSink;

impl ArtifactSink for NoopArtifactSink {
    fn write_json<'a, T>(&'a self, _name: &'a str, _value: &'a T) -> BoxFuture<'a, Result<()>>
    where
        T: Serialize + Sync + ?Sized,
    {
        Box::pin(async { Ok(()) })
    }

    fn write_bytes<'a>(&'a self, _name: &'a str, _bytes: &'a [u8]) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

#[derive(Debug, Clone)]
pub struct DirectoryArtifactSink {
    root: PathBuf,
}

impl DirectoryArtifactSink {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(sanitize_artifact_name(name))
    }
}

impl ArtifactSink for DirectoryArtifactSink {
    fn write_json<'a, T>(&'a self, name: &'a str, value: &'a T) -> BoxFuture<'a, Result<()>>
    where
        T: Serialize + Sync + ?Sized,
    {
        Box::pin(async move {
            let root = self.root.clone();
            let path = self.path(name);
            let body = serde_json::to_vec_pretty(value).context("serializing artifact json")?;
            tokio::task::spawn_blocking(move || {
                fs::create_dir_all(&root)
                    .with_context(|| format!("creating artifact dir {}", root.display()))?;
                fs::write(&path, body).with_context(|| format!("writing {}", path.display()))
            })
            .await
            .context("artifact writer task panicked")??;
            Ok(())
        })
    }

    fn write_bytes<'a>(&'a self, name: &'a str, bytes: &'a [u8]) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let root = self.root.clone();
            let path = self.path(name);
            let bytes = bytes.to_vec();
            tokio::task::spawn_blocking(move || {
                fs::create_dir_all(&root)
                    .with_context(|| format!("creating artifact dir {}", root.display()))?;
                fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))
            })
            .await
            .context("artifact writer task panicked")??;
            Ok(())
        })
    }
}

#[allow(dead_code)]
pub trait PersistenceBackend {
    fn backend_name(&self) -> &'static str;
}

#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct InMemoryPersistenceBackend;

impl PersistenceBackend for InMemoryPersistenceBackend {
    fn backend_name(&self) -> &'static str {
        "memory"
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LocalDirectoryPersistenceBackend {
    root: PathBuf,
}

impl LocalDirectoryPersistenceBackend {
    #[allow(dead_code)]
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    #[allow(dead_code)]
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl PersistenceBackend for LocalDirectoryPersistenceBackend {
    fn backend_name(&self) -> &'static str {
        "local_directory"
    }
}

#[allow(dead_code)]
pub trait ModerationActionSink {
    fn apply<'a>(&'a self, intent: &'a ModerationIntent) -> BoxFuture<'a, Result<()>>;
}

#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct RecordingActionSink {
    pub intents: parking_lot::Mutex<Vec<ModerationIntent>>,
}

impl ModerationActionSink for RecordingActionSink {
    fn apply<'a>(&'a self, intent: &'a ModerationIntent) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.intents.lock().push(intent.clone());
            Ok(())
        })
    }
}

pub struct ProgressiveEngine<'a, O: ?Sized, A> {
    pub matcher: Option<&'a Matcher>,
    pub detection_policy: &'a DetectionPolicy,
    pub text_gate_policy: &'a TextGatePolicy,
    pub ocr: &'a O,
    pub artifacts: &'a A,
}

impl<O, A> ProgressiveEngine<'_, O, A>
where
    O: OcrClient + ?Sized,
    A: ArtifactSink,
{
    pub async fn evaluate(
        &self,
        image: &ImageFingerprint,
        ocr_crops: Option<&[PreparedOcrCrop]>,
        artifact_prefix: &str,
    ) -> Result<ProgressiveDecision> {
        let visual = self.classify_visual(image);
        self.evaluate_with_visual(visual, ocr_crops, artifact_prefix)
            .await
    }

    pub async fn evaluate_with_visual(
        &self,
        visual: VisualClassification,
        ocr_crops: Option<&[PreparedOcrCrop]>,
        artifact_prefix: &str,
    ) -> Result<ProgressiveDecision> {
        let decision = match visual {
            VisualClassification::KnownStrong(outcome) => ProgressiveDecision {
                class: VisualCandidateClass::KnownStrong,
                outcome: Some(outcome),
                text_gate: None,
                ocr_requested: false,
            },
            VisualClassification::KnownSuspicious(outcome) => {
                self.confirm_with_text_gate(
                    VisualCandidateClass::KnownSuspicious,
                    Some(outcome),
                    ocr_crops,
                )
                .await?
            }
            VisualClassification::NoEvidence => ProgressiveDecision {
                class: VisualCandidateClass::NoEvidence,
                outcome: None,
                text_gate: None,
                ocr_requested: false,
            },
        };

        self.artifacts
            .write_json(&format!("{artifact_prefix}_decision.json"), &decision)
            .await?;
        if let Some(crops) = ocr_crops {
            for crop in crops {
                self.artifacts
                    .write_bytes(
                        &format!("{artifact_prefix}_{}.{}", crop.label, crop_extension(crop)),
                        crop.bytes.as_slice(),
                    )
                    .await?;
            }
        }

        Ok(decision)
    }

    pub fn classify_visual(&self, image: &ImageFingerprint) -> VisualClassification {
        VisualClassification::from_outcome(
            self.matcher
                .and_then(|matcher| matcher.find_for_policy(image, self.detection_policy)),
        )
    }

    async fn confirm_with_text_gate(
        &self,
        class: VisualCandidateClass,
        outcome: Option<MatchOutcome>,
        ocr_crops: Option<&[PreparedOcrCrop]>,
    ) -> Result<ProgressiveDecision> {
        if !self.text_gate_policy.enabled {
            return Ok(ProgressiveDecision {
                class,
                outcome,
                text_gate: Some(TextGateReport::disabled(self.text_gate_policy)),
                ocr_requested: false,
            });
        }

        let crops = ocr_crops.unwrap_or(&[]);
        let response = match self.ocr.read_text(crops).await {
            Ok(response) => response,
            Err(source) => {
                return Ok(ProgressiveDecision {
                    class,
                    outcome,
                    text_gate: Some(TextGateReport::unavailable(
                        self.text_gate_policy.keyword_threshold,
                        &source.to_string(),
                    )),
                    ocr_requested: true,
                });
            }
        };
        let report = evaluate_text_gate(self.text_gate_policy, &response);

        Ok(ProgressiveDecision {
            class,
            outcome,
            text_gate: Some(report),
            ocr_requested: true,
        })
    }
}

#[derive(Clone)]
pub enum VisualClassification {
    KnownStrong(MatchOutcome),
    KnownSuspicious(MatchOutcome),
    NoEvidence,
}

impl VisualClassification {
    pub fn from_outcome(outcome: Option<MatchOutcome>) -> Self {
        match outcome {
            Some(outcome) if outcome.suspicious => Self::KnownSuspicious(outcome),
            Some(outcome) => Self::KnownStrong(outcome),
            None => Self::NoEvidence,
        }
    }
}

pub fn evaluate_text_gate(policy: &TextGatePolicy, response: &OcrResponse) -> TextGateReport {
    if !policy.enabled {
        return TextGateReport::disabled(policy);
    }
    if policy.keywords.is_empty() && policy.sentences.is_empty() {
        return TextGateReport::unavailable(
            policy.keyword_threshold,
            "text gate is enabled without keywords or sentences",
        );
    }
    if !response.readable || response.text.trim().is_empty() {
        return TextGateReport::no_ocr_text(policy, response.text.clone());
    }

    let normalized_text = normalize_text_gate_pattern(&response.text);
    let text_tokens = normalized_text.split_whitespace().collect::<Vec<_>>();
    let matched_sentences = policy
        .sentences
        .iter()
        .filter_map(|sentence| {
            let sentence_tokens = sentence.split_whitespace().collect::<Vec<_>>();
            (!sentence_tokens.is_empty()
                && contains_token_window(&text_tokens, &sentence_tokens, 2))
            .then_some(sentence.clone())
        })
        .collect::<Vec<_>>();
    let sentence_hit = !matched_sentences.is_empty();
    let matched_keywords = policy
        .keywords
        .iter()
        .filter_map(|keyword| {
            let limit = keyword_edit_limit(keyword, policy.keyword_max_distance);
            let keyword_tokens = keyword.split_whitespace().collect::<Vec<_>>();
            (!keyword_tokens.is_empty()
                && contains_token_window(&text_tokens, &keyword_tokens, limit))
            .then_some(keyword.clone())
        })
        .collect::<Vec<_>>();
    let keyword_hits = matched_keywords.len();
    let decision = if sentence_hit {
        TextGateDecision::ConfirmedSentence
    } else if policy.keyword_threshold > 0 && keyword_hits >= policy.keyword_threshold {
        TextGateDecision::ConfirmedKeywords
    } else if keyword_hits > 0 {
        TextGateDecision::PartialKeywords
    } else {
        TextGateDecision::Rejected
    };
    let (verdict, confidence) = text_gate_verdict(decision, keyword_hits, policy.keyword_threshold);

    TextGateReport {
        readable: true,
        keyword_hits,
        keyword_threshold: policy.keyword_threshold,
        sentence_hit,
        matched_keywords,
        matched_sentences,
        ocr_text: Some(response.text.clone()),
        decision,
        verdict,
        confidence,
        error: None,
    }
}

fn text_gate_verdict(
    decision: TextGateDecision,
    keyword_hits: usize,
    keyword_threshold: usize,
) -> (TextGateVerdict, f32) {
    match decision {
        TextGateDecision::ConfirmedSentence => (TextGateVerdict::Bad, 0.95),
        TextGateDecision::ConfirmedKeywords => (TextGateVerdict::Bad, 0.90),
        TextGateDecision::PartialKeywords => {
            let denominator = keyword_threshold.max(1) as f32;
            let confidence = (keyword_hits as f32 / denominator).clamp(0.25, 0.75);
            (TextGateVerdict::Unknown, confidence)
        }
        TextGateDecision::Rejected => (TextGateVerdict::Good, 0.75),
        TextGateDecision::OcrPending
        | TextGateDecision::NoOcrText
        | TextGateDecision::OcrUnavailable => (TextGateVerdict::Unknown, 0.0),
        TextGateDecision::Disabled => (TextGateVerdict::Disabled, 1.0),
    }
}

fn truncate_error(value: &str) -> String {
    value.chars().take(160).collect()
}

fn contains_token_window(text_tokens: &[&str], pattern_tokens: &[&str], limit: usize) -> bool {
    let pattern_token_count = pattern_tokens.len().max(1);
    if text_tokens.len() < pattern_token_count {
        return false;
    }

    if text_tokens
        .windows(pattern_token_count)
        .any(|tokens| tokens == pattern_tokens)
    {
        return true;
    }

    if limit == 0 {
        return false;
    }

    let pattern = pattern_tokens.join(" ");
    let mut window = String::with_capacity(pattern.len().saturating_add(8));
    text_tokens.windows(pattern_token_count).any(|tokens| {
        window.clear();
        for (index, token) in tokens.iter().enumerate() {
            if index > 0 {
                window.push(' ');
            }
            window.push_str(token);
        }
        levenshtein_at_most(&window, &pattern, limit)
    })
}

fn keyword_edit_limit(keyword: &str, configured_limit: u8) -> usize {
    let length = keyword
        .chars()
        .filter(|character| !character.is_whitespace())
        .count();
    let length_limit = match length {
        0..=4 => 0,
        5..=8 => 1,
        _ => 2,
    };
    usize::from(configured_limit).min(length_limit)
}

fn levenshtein_at_most(left: &str, right: &str, limit: usize) -> bool {
    if left.len().abs_diff(right.len()) > limit {
        return false;
    }
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0usize; right.len() + 1];
    for (left_index, left_byte) in left.bytes().enumerate() {
        current[0] = left_index + 1;
        let mut row_min = current[0];
        for (right_index, right_byte) in right.bytes().enumerate() {
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            let substitution = previous[right_index] + usize::from(left_byte != right_byte);
            let value = insertion.min(deletion).min(substitution);
            current[right_index + 1] = value;
            row_min = row_min.min(value);
        }
        if row_min > limit {
            return false;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()] <= limit
}

fn sanitize_artifact_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn crop_extension(crop: &PreparedOcrCrop) -> &'static str {
    match crop.mime.as_str() {
        "image/jpeg" => "jpg",
        "image/png" => "png",
        "image/webp" => "webp",
        _ => "img",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_gate_confirms_sentence_or_keywords() {
        let policy = TextGatePolicy {
            enabled: true,
            keyword_threshold: 2,
            keyword_max_distance: 1,
            keywords: vec!["airdrop".to_owned(), "claim".to_owned()],
            sentences: vec!["connect your wallet".to_owned()],
        };

        let report = evaluate_text_gate(
            &policy,
            &OcrResponse {
                readable: true,
                text: "Please connect your wal1et now".to_owned(),
            },
        );
        assert_eq!(report.verdict, TextGateVerdict::Bad);

        let report = evaluate_text_gate(
            &policy,
            &OcrResponse {
                readable: true,
                text: "claim the airdrop".to_owned(),
            },
        );
        assert_eq!(report.decision, TextGateDecision::ConfirmedKeywords);
    }

    #[test]
    fn text_gate_tolerates_bounded_keyword_ocr_errors() {
        let policy = TextGatePolicy {
            enabled: true,
            keyword_threshold: 2,
            keyword_max_distance: 1,
            keywords: vec!["airdrop".to_owned(), "claim".to_owned(), "win".to_owned()],
            sentences: Vec::new(),
        };

        let report = evaluate_text_gate(
            &policy,
            &OcrResponse {
                readable: true,
                text: "clain the alrdrop now".to_owned(),
            },
        );
        assert_eq!(report.keyword_hits, 2);
        assert_eq!(report.decision, TextGateDecision::ConfirmedKeywords);
        assert_eq!(report.verdict, TextGateVerdict::Bad);

        let report = evaluate_text_gate(
            &policy,
            &OcrResponse {
                readable: true,
                text: "w1n a prize".to_owned(),
            },
        );
        assert_eq!(report.keyword_hits, 0);
        assert_eq!(report.decision, TextGateDecision::Rejected);
    }

    #[test]
    fn text_gate_partial_keywords_are_not_bad_verdict() {
        let policy = TextGatePolicy {
            enabled: true,
            keyword_threshold: 3,
            keyword_max_distance: 1,
            keywords: vec![
                "airdrop".to_owned(),
                "claim".to_owned(),
                "wallet".to_owned(),
            ],
            sentences: Vec::new(),
        };

        let report = evaluate_text_gate(
            &policy,
            &OcrResponse {
                readable: true,
                text: "claim your reward".to_owned(),
            },
        );
        assert_eq!(report.keyword_hits, 1);
        assert_eq!(report.decision, TextGateDecision::PartialKeywords);
        assert_eq!(report.verdict, TextGateVerdict::Unknown);
    }

    #[test]
    fn empty_enabled_text_gate_does_not_clear_ocr_followup() {
        let policy = TextGatePolicy {
            enabled: true,
            keyword_threshold: 1,
            keyword_max_distance: 1,
            keywords: Vec::new(),
            sentences: Vec::new(),
        };

        let report = evaluate_text_gate(
            &policy,
            &OcrResponse {
                readable: true,
                text: "plain readable text".to_owned(),
            },
        );

        assert_eq!(report.decision, TextGateDecision::OcrUnavailable);
        assert_eq!(report.verdict, TextGateVerdict::Unknown);
    }
}
