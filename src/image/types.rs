#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{Context, Result, anyhow};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fmt::{self, Write as _},
    time::Instant,
};
use twilight_model::id::{
    Id,
    marker::{ChannelMarker, GuildMarker, MessageMarker, UserMarker},
};

const LRU_ORDER_COMPACT_MULTIPLIER: usize = 4;

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Xxh128(u128);

impl Xxh128 {
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    pub fn from_hex(value: &str) -> Option<Self> {
        if value.len() != 32 {
            return None;
        }
        u128::from_str_radix(value, 16).ok().map(Self)
    }

    pub fn short_hex(self) -> String {
        format!("{:016x}", self.0 >> 64)
    }
}

impl fmt::Display for Xxh128 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:032x}", self.0)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateKind {
    Attachment,
    EmbedImage,
    EmbedThumbnail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageCandidate {
    pub guild_id: Id<GuildMarker>,
    pub channel_id: Id<ChannelMarker>,
    pub message_id: Id<MessageMarker>,
    pub candidate_index: u16,
    pub candidates_in_message: u16,
    pub author_id: Id<UserMarker>,
    pub author_username: Option<String>,
    pub author_global_name: Option<String>,
    pub url: String,
    pub proxy_url: Option<String>,
    pub kind: CandidateKind,
    pub mime_hint: Option<String>,
    pub size_bytes: Option<u64>,
    pub metadata_width: Option<u32>,
    pub metadata_height: Option<u32>,
    pub media_flags: Option<u64>,
    #[serde(skip)]
    pub verify_only: bool,
    #[serde(skip)]
    pub enqueued_at: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFingerprint {
    pub width: u32,
    pub height: u32,
    pub mime: Option<String>,
    pub byte_xxh128: String,
    pub phash64: String,
    pub dhash64: String,
    pub visual: ImageVisualSignature,
    pub local_anchors: Vec<ImageAnchor>,
    pub local_hashes: Vec<LocalImageHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedImageFingerprint {
    pub schema: u8,
    pub source_path: String,
    pub fingerprint: ImageFingerprint,
}

#[derive(Clone, Copy)]
pub struct ImageFingerprintParts<'a> {
    pub width: u32,
    pub height: u32,
    pub byte_xxh128: &'a str,
    pub phash64: &'a str,
    pub dhash64: &'a str,
    pub visual: &'a ImageVisualSignature,
    pub local_anchors: &'a [ImageAnchor],
    pub local_hashes: &'a [LocalImageHash],
}

impl ExportedImageFingerprint {
    pub const SCHEMA: u8 = 7;

    pub fn new(source_path: impl Into<String>, fingerprint: ImageFingerprint) -> Self {
        Self {
            schema: Self::SCHEMA,
            source_path: source_path.into(),
            fingerprint,
        }
    }

    pub fn into_validated_fingerprint(self) -> Result<ImageFingerprint> {
        if self.schema != Self::SCHEMA {
            return Err(anyhow!(
                "unsupported exported fingerprint schema {}",
                self.schema
            ));
        }
        self.fingerprint.validate()?;
        Ok(self.fingerprint)
    }
}

impl ImageFingerprint {
    pub fn validate(&self) -> Result<()> {
        Self::validate_parts(ImageFingerprintParts {
            width: self.width,
            height: self.height,
            byte_xxh128: &self.byte_xxh128,
            phash64: &self.phash64,
            dhash64: &self.dhash64,
            visual: &self.visual,
            local_anchors: &self.local_anchors,
            local_hashes: &self.local_hashes,
        })
    }

    pub fn validate_parts(parts: ImageFingerprintParts<'_>) -> Result<()> {
        let ImageFingerprintParts {
            width,
            height,
            byte_xxh128,
            phash64,
            dhash64,
            visual,
            local_anchors,
            local_hashes,
        } = parts;
        anyhow::ensure!(width > 0, "fingerprint.width must be greater than 0");
        anyhow::ensure!(height > 0, "fingerprint.height must be greater than 0");
        anyhow::ensure!(
            local_anchors.len() <= 4_096,
            "fingerprint.local_anchors must contain at most 4096 anchors"
        );
        anyhow::ensure!(
            local_hashes.len() <= 5_000,
            "fingerprint.local_hashes must contain at most 5000 hashes"
        );
        validate_hex(byte_xxh128, 16, "fingerprint.byte_xxh128")?;
        validate_hex(phash64, 8, "fingerprint.phash64")?;
        validate_hex(dhash64, 8, "fingerprint.dhash64")?;
        anyhow::ensure!(
            visual.text_grid.len() == 64,
            "fingerprint.visual.text_grid must contain exactly 64 cells"
        );
        for anchor in local_anchors {
            anyhow::ensure!(
                anchor.w > 0 && anchor.h > 0,
                "fingerprint.local_anchors[].w/h must be greater than 0"
            );
            anyhow::ensure!(
                anchor.region < 64,
                "fingerprint.local_anchors[].region must be between 0 and 63"
            );
            anyhow::ensure!(
                anchor.max_distance <= 15,
                "fingerprint.local_anchors[].max_distance must be at most 15"
            );
            validate_hex(&anchor.hash, 8, "fingerprint.local_anchors[].hash")?;
            validate_hex(&anchor.hash2, 8, "fingerprint.local_anchors[].hash2")?;
        }
        for hash in local_hashes {
            anyhow::ensure!(
                hash.w > 0 && hash.h > 0,
                "fingerprint.local_hashes[].w/h must be greater than 0"
            );
            anyhow::ensure!(
                hash.region < 64,
                "fingerprint.local_hashes[].region must be between 0 and 63"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAnchor {
    pub id: String,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub pos_x: u8,
    pub pos_y: u8,
    pub hash: String,
    pub hash2: String,
    pub luma_mean: u8,
    pub luma_std: u8,
    pub edge_density: u8,
    pub kind: String,
    pub region: u32,
    pub max_distance: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalImageHash {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub pos_x: u8,
    pub pos_y: u8,
    pub region: u32,
    pub hash: u64,
    pub hash2: u64,
    pub luma_mean: u8,
    pub luma_std: u8,
    pub edge_density: u8,
    pub scale_percent: u16,
    pub rotation_degrees: i16,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ImageVisualSignature {
    pub luma_mean: u8,
    pub luma_std: u8,
    pub rgb_mean: [u8; 3],
    pub grid_luma: [u8; 16],
    pub text_grid: Vec<u8>,
}

impl Default for ImageVisualSignature {
    fn default() -> Self {
        Self {
            luma_mean: 128,
            luma_std: 0,
            rgb_mean: [128; 3],
            grid_luma: [128; 16],
            text_grid: vec![0; 64],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchOutcome {
    pub specimen_id: String,
    pub confidence: MatchConfidence,
    pub suspicious: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phash64_distance: Option<u32>,
    pub dhash64_distance: Option<u32>,
    pub local_anchor_hits: Option<usize>,
    pub local_distinct_regions: Option<usize>,
    pub local_average_distance: Option<f32>,
    pub local_geometry_model: Option<GeometryModel>,
    pub diagnostics: MatchDiagnostics,
}

impl MatchOutcome {
    pub(crate) fn label(&self) -> &'static str {
        if self.suspicious {
            "Suspicious image"
        } else {
            "Scam image match"
        }
    }

    pub(crate) fn decision_name(&self) -> &'static str {
        if self.suspicious {
            "suspicious"
        } else {
            "confirmed"
        }
    }

    pub(crate) fn local_match_details(&self) -> String {
        match (
            self.local_anchor_hits,
            self.local_distinct_regions,
            self.local_average_distance,
        ) {
            (Some(hits), Some(regions), average) => format!(
                ", local anchors hits `{hits}`, regions `{regions}`, avg distance `{}`",
                average.map_or_else(|| "n/a".to_owned(), |value| format!("{value:.2}"))
            ),
            _ => String::new(),
        }
    }

    pub(crate) fn diagnostics_summary(&self) -> String {
        let steps = self
            .diagnostics
            .steps
            .iter()
            .map(|step| {
                let status = if step.passed { "pass" } else { "fail" };
                let mut parts = vec![format!("{}:{}={status}", step.threshold, step.step)];
                if let Some(reason) = step.reason {
                    parts.push(format!("reason={reason}"));
                }
                if let Some(specimen_id) = &step.specimen_id {
                    parts.push(format!("specimen={specimen_id}"));
                }
                if let Some(count) = step.candidates_considered {
                    parts.push(format!("n={count}"));
                }
                if let Some(phash) = step.phash64_distance {
                    parts.push(format!("ph={phash}"));
                }
                if let Some(dhash) = step.dhash64_distance {
                    parts.push(format!("dh={dhash}"));
                }
                if let Some(geometry) = step.geometry_compatible {
                    parts.push(format!("geom={geometry}"));
                }
                if let Some(visual) = step.visual_compatible {
                    parts.push(format!("visual={visual}"));
                }
                if let Some(hits) = step.local_anchor_hits {
                    parts.push(format!("hits={hits}"));
                }
                if let Some(regions) = step.local_distinct_regions {
                    parts.push(format!("regions={regions}"));
                }
                if let Some(mean) = step.local_average_distance {
                    parts.push(format!("mean={mean:.2}"));
                }
                if let Some(spread) = step.local_layout_spread {
                    parts.push(format!("spread={spread:.1}"));
                }
                if let Some(residual) = step.local_mean_residual {
                    parts.push(format!("resid={residual:.1}"));
                }
                if let Some(scale) = step.local_scale {
                    parts.push(format!("scale={scale:.2}"));
                }
                if let Some(angle) = step.local_angle {
                    parts.push(format!("angle={:.1}", angle.to_degrees()));
                }
                if let Some(model) = step.local_geometry_model {
                    parts.push(format!("model={}", model.as_str()));
                }
                if let Some(signals) = step.visual_shape_signals {
                    parts.push(format!("signals={signals}"));
                }
                if let Some(score) = step.visual_shape_score {
                    parts.push(format!("shape_score={score}"));
                }
                if let Some(score) = step.match_score {
                    parts.push(format!("score={score:.1}"));
                }
                parts.join(",")
            })
            .collect::<Vec<_>>()
            .join(" | ");
        truncate_chars(
            &format!(
                "representation={:?} candidate short={} area={} aspect={:.2} luma={}/{} text={}/{} hashes={} steps=[{}]",
                self.diagnostics.representation,
                self.diagnostics.candidate_short_edge,
                self.diagnostics.candidate_area,
                self.diagnostics.candidate_aspect,
                self.diagnostics.candidate_luma_mean,
                self.diagnostics.candidate_luma_std,
                self.diagnostics.candidate_text_grid_mean,
                self.diagnostics.candidate_text_regions,
                self.diagnostics.candidate_local_hashes,
                steps
            ),
            900,
        )
    }

    pub(crate) fn tripped_gates_summary(&self) -> String {
        let gates = self
            .diagnostics
            .steps
            .iter()
            .filter(|step| step.passed)
            .map(|step| {
                let mut label = format!("{}:{}", step.threshold, step.step);
                if let Some(score) = step.match_score {
                    let _ = write!(label, " score={score:.1}");
                }
                if let Some(hits) = step.local_anchor_hits {
                    let _ = write!(label, " hits={hits}");
                }
                if let Some(regions) = step.local_distinct_regions {
                    let _ = write!(label, " regions={regions}");
                }
                if let Some(mean) = step.local_average_distance {
                    let _ = write!(label, " mean={mean:.2}");
                }
                if let Some(model) = step.local_geometry_model {
                    let _ = write!(label, " model={}", model.as_str());
                }
                label
            })
            .collect::<Vec<_>>();

        if gates.is_empty() {
            "none".to_owned()
        } else {
            truncate_chars(&gates.join(" | "), 500)
        }
    }

    pub(crate) fn score_label(&self) -> String {
        if let Some(score) = self.match_score {
            return format!("{score:.1}");
        }
        match (
            self.phash64_distance,
            self.dhash64_distance,
            self.local_average_distance,
        ) {
            (_, _, Some(mean)) => format!("mean={mean:.2}"),
            (Some(phash), Some(dhash), _) => format!("phash={phash};dhash={dhash}"),
            _ => format!("{:?}", self.confidence),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchDiagnostics {
    pub representation: FingerprintRepresentation,
    pub candidate_short_edge: u32,
    pub candidate_area: u64,
    pub candidate_aspect: f32,
    pub candidate_luma_mean: u8,
    pub candidate_luma_std: u8,
    pub candidate_text_grid_mean: u8,
    pub candidate_text_regions: usize,
    pub candidate_local_hashes: usize,
    pub steps: Vec<MatchStepDiagnostic>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintRepresentation {
    Original,
    DiscordPreview,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchStepDiagnostic {
    pub threshold: &'static str,
    pub step: &'static str,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specimen_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_considered: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phash64_distance: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dhash64_distance: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry_compatible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visual_compatible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_anchor_hits: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_distinct_regions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_average_distance: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_layout_spread: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_mean_residual: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_angle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_geometry_model: Option<GeometryModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visual_shape_signals: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visual_shape_score: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_score: Option<f32>,
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }
    let mut truncated = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

#[derive(Debug)]
pub(crate) struct LruDedupe {
    max_entries: usize,
    entries: HashMap<String, u64>,
    order: VecDeque<(String, u64)>,
    next_sequence: u64,
}

#[derive(Debug)]
pub(crate) struct HashOutcomeLruCache {
    max_entries: usize,
    entries: FxHashMap<HashOutcomeCacheKey, usize>,
    nodes: Vec<Option<HashOutcomeCacheNode>>,
    free: Vec<usize>,
    oldest: Option<usize>,
    newest: Option<usize>,
    guild_generations: FxHashMap<u64, u64>,
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
struct HashOutcomeCacheKey {
    guild_id: u64,
    generation: u64,
    xxh128: Xxh128,
}

#[derive(Debug)]
struct HashOutcomeCacheNode {
    key: HashOutcomeCacheKey,
    policy_hash: u64,
    outcome: CachedDecisionOutcome,
    older: Option<usize>,
    newer: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum CachedDecisionOutcome {
    Pass,
    Failure(String),
    Match(CachedMatchOutcome),
}

#[derive(Debug, Clone)]
pub(crate) struct CachedMatchOutcome {
    pub(crate) specimen_id: String,
    pub(crate) confidence: MatchConfidence,
    pub(crate) suspicious: bool,
    pub(crate) match_score: Option<f32>,
    pub(crate) phash64_distance: Option<u32>,
    pub(crate) dhash64_distance: Option<u32>,
    pub(crate) local_anchor_hits: Option<usize>,
    pub(crate) local_distinct_regions: Option<usize>,
    pub(crate) local_average_distance: Option<f32>,
    pub(crate) local_geometry_model: Option<GeometryModel>,
}

#[derive(Debug)]
pub(crate) struct ImagePerfTracker {
    samples: VecDeque<u128>,
    sample_sum_ms: u128,
    total_count: u64,
    total_success: u64,
    total_failure: u64,
    sample_cap: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ImagePerfSample {
    pub(crate) guild_id: Id<GuildMarker>,
    pub(crate) duration_ms: u128,
    pub(crate) success: bool,
    pub(crate) decision: ImageScanDecisionMetric,
    pub(crate) stage_timings: Option<Box<ImageStageTimingSample>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ImageStageTimingSample {
    pub(crate) total_us: u64,
    pub(crate) preview_download_us: u64,
    pub(crate) preview_fingerprint_us: u64,
    pub(crate) preview_matcher_us: u64,
    pub(crate) preview_used: bool,
    pub(crate) preview_fallback: bool,
    pub(crate) queue_wait_us: u64,
    pub(crate) download_us: u64,
    pub(crate) download_request_us: u64,
    pub(crate) download_body_us: u64,
    pub(crate) download_gate_wait_us: u64,
    pub(crate) flagged_cache_lookup_us: u64,
    pub(crate) exact_match_lookup_us: u64,
    pub(crate) singleflight_wait_us: u64,
    pub(crate) fingerprint_us: u64,
    pub(crate) fingerprint_pipeline: ImageFingerprintTimingSample,
    pub(crate) matcher_us: u64,
    pub(crate) ocr_crop_us: u64,
    pub(crate) progressive_eval_us: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ImageFingerprintTimingSample {
    pub(crate) decode: u64,
    pub(crate) thumbnail: u64,
    pub(crate) visual: u64,
    pub(crate) orientation: u64,
    pub(crate) perceptual: u64,
    pub(crate) normalize: u64,
    pub(crate) tile_scorer: u64,
    pub(crate) text_grid: u64,
    pub(crate) local_anchors: u64,
    pub(crate) local_hashes: u64,
}

#[derive(Debug, Clone)]
pub(crate) enum ImageMetricEvent {
    Processed(ImagePerfSample),
    OcrCall {
        guild_id: Id<GuildMarker>,
    },
    OcrResolved {
        guild_id: Id<GuildMarker>,
        resolution: TextGateResolutionMetric,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ImageScanDecisionMetric {
    Pass,
    ScanFailed,
    HardMatch(ImageMatchStageMetric),
    Suspicious(ImageMatchStageMetric),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ImageMatchStageMetric {
    ExactXxh128,
    Perceptual,
    LocalAnchors,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum TextGateResolutionMetric {
    Good,
    Bad,
    Unknown,
}

#[derive(Debug, Clone)]
pub(crate) struct ImagePerfSnapshot {
    pub(crate) total_count: u64,
    pub(crate) total_success: u64,
    pub(crate) total_failure: u64,
    pub(crate) sample_count: usize,
    pub(crate) min_ms: u128,
    pub(crate) max_ms: u128,
    pub(crate) avg_ms: f64,
    pub(crate) p50_ms: u128,
    pub(crate) p90_ms: u128,
    pub(crate) p95_ms: u128,
    pub(crate) p99_ms: u128,
}

impl LruDedupe {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            entries: HashMap::new(),
            order: VecDeque::new(),
            next_sequence: 0,
        }
    }

    pub(crate) fn insert_new(&mut self, key: String) -> bool {
        if self.entries.contains_key(&key) {
            self.touch(key);
            return false;
        }

        let sequence = self.next_sequence();
        self.entries.insert(key.clone(), sequence);
        self.order.push_back((key, sequence));
        self.compact_order_if_needed();
        self.enforce_cap();
        true
    }

    pub(crate) fn remove(&mut self, key: &str) -> bool {
        let removed = self.entries.remove(key).is_some();
        if removed {
            self.compact_order_if_needed();
        }
        removed
    }

    fn touch(&mut self, key: String) {
        let sequence = self.next_sequence();
        self.entries.insert(key.clone(), sequence);
        self.order.push_back((key, sequence));
        self.compact_order_if_needed();
    }

    fn next_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        sequence
    }

    fn enforce_cap(&mut self) {
        while self.entries.len() > self.max_entries {
            let Some((key, sequence)) = self.order.pop_front() else {
                self.compact_order();
                if self.order.is_empty() {
                    break;
                }
                continue;
            };
            if self
                .entries
                .get(&key)
                .is_some_and(|current| *current == sequence)
            {
                self.entries.remove(&key);
            }
        }
    }

    fn compact_order_if_needed(&mut self) {
        let max_order_len = self
            .max_entries
            .saturating_mul(LRU_ORDER_COMPACT_MULTIPLIER)
            .max(64);
        if self.order.len() > max_order_len {
            self.compact_order();
        }
    }

    fn compact_order(&mut self) {
        let mut entries = self
            .entries
            .iter()
            .map(|(key, sequence)| (key.clone(), *sequence))
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(_, sequence)| *sequence);
        self.order = entries.into();
    }
}

impl HashOutcomeLruCache {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            entries: FxHashMap::default(),
            nodes: Vec::new(),
            free: Vec::new(),
            oldest: None,
            newest: None,
            guild_generations: FxHashMap::default(),
        }
    }

    pub(crate) fn get(
        &mut self,
        guild_id: u64,
        xxh128: Xxh128,
        policy_hash: u64,
    ) -> Option<CachedDecisionOutcome> {
        let key = self.key(guild_id, xxh128);
        let index = self.entries.get(&key).copied()?;
        if self.node(index).policy_hash != policy_hash {
            self.remove_index(index);
            return None;
        }
        let outcome = self.node(index).outcome.clone();
        self.promote(index);
        Some(outcome)
    }

    pub(crate) fn insert_match(
        &mut self,
        guild_id: u64,
        xxh128: Xxh128,
        policy_hash: u64,
        outcome: &MatchOutcome,
    ) {
        self.insert_decision(
            guild_id,
            xxh128,
            policy_hash,
            CachedDecisionOutcome::Match(CachedMatchOutcome::from(outcome)),
        );
    }

    pub(crate) fn insert_pass(&mut self, guild_id: u64, xxh128: Xxh128, policy_hash: u64) {
        self.insert_decision(guild_id, xxh128, policy_hash, CachedDecisionOutcome::Pass);
    }

    pub(crate) fn insert_failure(
        &mut self,
        guild_id: u64,
        xxh128: Xxh128,
        policy_hash: u64,
        reason: impl Into<String>,
    ) {
        self.insert_decision(
            guild_id,
            xxh128,
            policy_hash,
            CachedDecisionOutcome::Failure(truncate_chars(&reason.into(), 300)),
        );
    }

    fn insert_decision(
        &mut self,
        guild_id: u64,
        xxh128: Xxh128,
        policy_hash: u64,
        outcome: CachedDecisionOutcome,
    ) {
        let key = self.key(guild_id, xxh128);
        if let Some(index) = self.entries.get(&key).copied() {
            let node = self.node_mut(index);
            node.policy_hash = policy_hash;
            node.outcome = outcome;
            self.promote(index);
            return;
        }

        if self.entries.len() >= self.max_entries
            && let Some(index) = self.oldest
        {
            self.remove_index(index);
        }

        let index = if let Some(index) = self.free.pop() {
            self.nodes[index] = Some(HashOutcomeCacheNode {
                key,
                policy_hash,
                outcome,
                older: None,
                newer: None,
            });
            index
        } else {
            let index = self.nodes.len();
            self.nodes.push(Some(HashOutcomeCacheNode {
                key,
                policy_hash,
                outcome,
                older: None,
                newer: None,
            }));
            index
        };
        self.entries.insert(key, index);
        self.link_newest(index);
    }

    pub(crate) fn clear_guild(&mut self, guild_id: u64) {
        let next = self
            .guild_generations
            .get(&guild_id)
            .copied()
            .unwrap_or(0)
            .wrapping_add(1);
        if next == 0 {
            let max_entries = self.max_entries;
            *self = Self::new(max_entries);
            self.guild_generations.insert(guild_id, 1);
        } else {
            self.guild_generations.insert(guild_id, next);
        }
    }

    fn key(&self, guild_id: u64, xxh128: Xxh128) -> HashOutcomeCacheKey {
        HashOutcomeCacheKey {
            guild_id,
            generation: self.guild_generations.get(&guild_id).copied().unwrap_or(0),
            xxh128,
        }
    }

    fn promote(&mut self, index: usize) {
        if self.newest == Some(index) {
            return;
        }
        self.detach(index);
        self.link_newest(index);
    }

    fn link_newest(&mut self, index: usize) {
        let previous = self.newest;
        {
            let node = self.node_mut(index);
            node.older = previous;
            node.newer = None;
        }
        if let Some(previous) = previous {
            self.node_mut(previous).newer = Some(index);
        } else {
            self.oldest = Some(index);
        }
        self.newest = Some(index);
    }

    fn detach(&mut self, index: usize) {
        let (older, newer) = {
            let node = self.node(index);
            (node.older, node.newer)
        };
        if let Some(older) = older {
            self.node_mut(older).newer = newer;
        } else {
            self.oldest = newer;
        }
        if let Some(newer) = newer {
            self.node_mut(newer).older = older;
        } else {
            self.newest = older;
        }
        let node = self.node_mut(index);
        node.older = None;
        node.newer = None;
    }

    fn remove_index(&mut self, index: usize) {
        let key = self.node(index).key;
        self.detach(index);
        self.entries.remove(&key);
        self.nodes[index] = None;
        self.free.push(index);
    }

    fn node(&self, index: usize) -> &HashOutcomeCacheNode {
        self.nodes[index]
            .as_ref()
            .expect("live cache index must reference a node")
    }

    fn node_mut(&mut self, index: usize) -> &mut HashOutcomeCacheNode {
        self.nodes[index]
            .as_mut()
            .expect("live cache index must reference a node")
    }
}

impl From<&MatchOutcome> for CachedMatchOutcome {
    fn from(outcome: &MatchOutcome) -> Self {
        Self {
            specimen_id: outcome.specimen_id.clone(),
            confidence: outcome.confidence,
            suspicious: outcome.suspicious,
            match_score: outcome.match_score,
            phash64_distance: outcome.phash64_distance,
            dhash64_distance: outcome.dhash64_distance,
            local_anchor_hits: outcome.local_anchor_hits,
            local_distinct_regions: outcome.local_distinct_regions,
            local_average_distance: outcome.local_average_distance,
            local_geometry_model: outcome.local_geometry_model,
        }
    }
}

impl CachedMatchOutcome {
    pub(crate) fn into_match_outcome(self) -> MatchOutcome {
        let candidate_short_edge = 0;
        let candidate_area = 0;
        MatchOutcome {
            specimen_id: self.specimen_id.clone(),
            confidence: self.confidence,
            suspicious: self.suspicious,
            match_score: self.match_score,
            phash64_distance: self.phash64_distance,
            dhash64_distance: self.dhash64_distance,
            local_anchor_hits: self.local_anchor_hits,
            local_distinct_regions: self.local_distinct_regions,
            local_average_distance: self.local_average_distance,
            local_geometry_model: self.local_geometry_model,
            diagnostics: MatchDiagnostics {
                representation: FingerprintRepresentation::Original,
                candidate_short_edge,
                candidate_area,
                candidate_aspect: 0.0,
                candidate_luma_mean: 0,
                candidate_luma_std: 0,
                candidate_text_grid_mean: 0,
                candidate_text_regions: 0,
                candidate_local_hashes: 0,
                steps: vec![MatchStepDiagnostic {
                    threshold: if self.suspicious {
                        "suspicious"
                    } else {
                        "confirmed"
                    },
                    step: "cached_hash_outcome",
                    passed: true,
                    reason: Some("lru_cache_hit"),
                    specimen_id: Some(self.specimen_id),
                    candidates_considered: Some(1),
                    phash64_distance: self.phash64_distance,
                    dhash64_distance: self.dhash64_distance,
                    geometry_compatible: None,
                    visual_compatible: None,
                    local_anchor_hits: self.local_anchor_hits,
                    local_distinct_regions: self.local_distinct_regions,
                    local_average_distance: self.local_average_distance,
                    local_layout_spread: None,
                    local_mean_residual: None,
                    local_scale: None,
                    local_angle: None,
                    local_geometry_model: self.local_geometry_model,
                    visual_shape_signals: None,
                    visual_shape_score: None,
                    match_score: self.match_score,
                }],
            },
        }
    }
}

impl ImagePerfTracker {
    pub(crate) fn new(sample_cap: usize) -> Self {
        Self {
            // Metrics are created per guild. Grow only when samples arrive so
            // an idle guild does not eagerly reserve every history buffer.
            samples: VecDeque::new(),
            sample_sum_ms: 0,
            total_count: 0,
            total_success: 0,
            total_failure: 0,
            sample_cap,
        }
    }

    pub(crate) fn record(&mut self, sample: &ImagePerfSample) {
        self.total_count += 1;
        if sample.success {
            self.total_success += 1;
        } else {
            self.total_failure += 1;
        }

        if self.sample_cap == 0 {
            return;
        }
        if self.samples.len() == self.sample_cap
            && let Some(removed) = self.samples.pop_front()
        {
            self.sample_sum_ms = self.sample_sum_ms.saturating_sub(removed);
        }
        self.samples.push_back(sample.duration_ms);
        self.sample_sum_ms += sample.duration_ms;
    }

    pub(crate) fn snapshot(&self) -> Option<ImagePerfSnapshot> {
        if self.samples.is_empty() {
            return None;
        }

        let mut sorted = self.samples.iter().copied().collect::<Vec<_>>();
        sorted.sort_unstable();
        let sample_count = sorted.len();
        Some(ImagePerfSnapshot {
            total_count: self.total_count,
            total_success: self.total_success,
            total_failure: self.total_failure,
            sample_count,
            min_ms: *sorted.first().unwrap_or(&0),
            max_ms: *sorted.last().unwrap_or(&0),
            avg_ms: self.sample_sum_ms as f64 / sample_count as f64,
            p50_ms: percentile(&sorted, 0.50),
            p90_ms: percentile(&sorted, 0.90),
            p95_ms: percentile(&sorted, 0.95),
            p99_ms: percentile(&sorted, 0.99),
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchConfidence {
    ExactXxh128,
    Perceptual,
    LocalAnchors,
    DenseLocalAnchors,
    ClusterCoherence,
    SuspiciousPerceptual,
    SuspiciousLocalAnchors,
    SuspiciousDenseLocalAnchors,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryModel {
    Similarity,
    Affine,
    Homography,
}

impl GeometryModel {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Similarity => "similarity",
            Self::Affine => "affine",
            Self::Homography => "homography",
        }
    }
}

fn validate_hex(value: &str, bytes: usize, name: &str) -> Result<()> {
    let decoded = hex::decode(value).with_context(|| format!("{name} must be hex"))?;
    if decoded.len() != bytes {
        return Err(anyhow!("{name} must be {bytes} bytes"));
    }
    Ok(())
}

fn percentile(sorted: &[u128], quantile: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_dedupe_refreshes_hits_and_evicts_oldest() {
        let mut cache = LruDedupe::new(2);
        assert!(cache.insert_new("a".to_owned()));
        assert!(cache.insert_new("b".to_owned()));
        assert!(!cache.insert_new("a".to_owned()));

        assert!(cache.insert_new("c".to_owned()));

        assert!(!cache.insert_new("a".to_owned()));
        assert!(cache.insert_new("b".to_owned()));
    }

    #[test]
    fn hash_outcome_lru_cache_refreshes_hits_and_evicts_oldest() {
        let mut cache = HashOutcomeLruCache::new(2);
        cache.insert_pass(1, Xxh128::new(10), 1);
        cache.insert_pass(1, Xxh128::new(20), 1);

        assert!(matches!(
            cache.get(1, Xxh128::new(10), 1),
            Some(CachedDecisionOutcome::Pass)
        ));

        cache.insert_pass(1, Xxh128::new(30), 1);

        assert!(cache.get(1, Xxh128::new(20), 1).is_none());
        assert!(matches!(
            cache.get(1, Xxh128::new(10), 1),
            Some(CachedDecisionOutcome::Pass)
        ));
        assert!(matches!(
            cache.get(1, Xxh128::new(30), 1),
            Some(CachedDecisionOutcome::Pass)
        ));
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn hash_outcome_lru_cache_is_global_and_guild_scoped() {
        let mut cache = HashOutcomeLruCache::new(2);
        let hash = Xxh128::new(10);
        cache.insert_pass(1, hash, 1);
        cache.insert_pass(2, hash, 1);

        assert!(cache.get(1, hash, 1).is_some());
        assert!(cache.get(2, hash, 1).is_some());

        cache.clear_guild(1);
        assert!(cache.get(1, hash, 1).is_none());
        assert!(cache.get(2, hash, 1).is_some());
        // Generation invalidation is O(1); stale storage remains bounded by the
        // global LRU cap and is reclaimed by subsequent insertions.
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn hash_outcome_lru_cache_storage_never_exceeds_global_cap() {
        let mut cache = HashOutcomeLruCache::new(32);
        for value in 0..10_000 {
            cache.insert_pass(value % 7, Xxh128::new(value.into()), value % 3);
        }

        assert_eq!(cache.entries.len(), 32);
        assert_eq!(cache.nodes.len(), 32);
        assert!(cache.free.is_empty());
    }

    #[test]
    fn hash_outcome_lru_cache_drops_policy_mismatches() {
        let mut cache = HashOutcomeLruCache::new(2);
        let hash = Xxh128::new(10);
        cache.insert_pass(1, hash, 1);

        assert!(cache.get(1, hash, 2).is_none());
        assert!(cache.get(1, hash, 1).is_none());
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn xxh128_hex_round_trips_and_shortens() {
        let hash = Xxh128::new(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
        assert_eq!(hash.to_string(), "0123456789abcdeffedcba9876543210");
        assert_eq!(hash.short_hex(), "0123456789abcdef");
        assert_eq!(Xxh128::from_hex(&hash.to_string()), Some(hash));
    }
}
