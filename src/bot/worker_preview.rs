use crate::{
    bot::{
        runtime::AppState,
        worker::{
            CandidateDecision, CandidateStageTimings, PreviewFallback, PreviewRoute,
            PreviewScanResult, SpecimenPreviewVariant,
        },
    },
    configuration::{
        app::{DownloadConfig, MatchConfig},
        guild::DetectionPolicy,
    },
    image::{
        pipeline::{
            DiscordPreviewRequest, HashMode, build_discord_preview_request, hash_downloaded_image,
        },
        types::{ImageCandidate, MatchConfidence, MatchOutcome},
    },
};
use anyhow::Result;
use std::{sync::Arc, time::Instant};
use tracing::{info, warn};

pub(crate) async fn scan_preview_candidate(
    state: &AppState,
    candidate: &ImageCandidate,
    preview: DiscordPreviewRequest,
    context: PreviewScanContext<'_>,
    started: Instant,
) -> Result<PreviewScanResult> {
    let mut timings = CandidateStageTimings::default();
    let preview_downloaded = state
        .download_image(
            &preview.url,
            candidate.mime_hint.as_deref(),
            context.download_config,
        )
        .await?;
    timings.preview_download = preview_downloaded.timings;

    // Preview scans are an early-exit optimization only. Local and dense-local
    // hashing stay on the original image path so preview misses do not spend CPU
    // on evidence that cannot make the preview result decisive.
    let hash_mode = HashMode::candidate_without_local_hashes();
    let preview_image_id = format!(
        "preview_{}",
        preview_downloaded
            .byte_xxh128
            .chars()
            .take(16)
            .collect::<String>()
    );
    let preview_xxh128 = preview_downloaded.byte_xxh128.clone();
    let fingerprint_started = Instant::now();
    let preview_fingerprint = hash_downloaded_image(
        preview_downloaded,
        context.download_config.max_decoded_pixels,
        context.match_config,
        &state.decode_gate,
        hash_mode,
    )
    .await?;
    timings.preview_fingerprint_us = fingerprint_started.elapsed().as_micros();

    let matcher_started = Instant::now();
    let preview_match = state
        .find_preview_match_for_policy(
            Arc::new(preview_fingerprint),
            context.detection_policy.clone(),
        )
        .await?;
    timings.preview_matcher_us = matcher_started.elapsed().as_micros();

    Ok(finish_preview_scan(
        preview_match,
        preview_image_id,
        preview_xxh128,
        context.policy_hash,
        timings,
        started,
    ))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PreviewScanContext<'a> {
    pub(crate) download_config: &'a DownloadConfig,
    pub(crate) match_config: &'a MatchConfig,
    pub(crate) detection_policy: &'a DetectionPolicy,
    pub(crate) policy_hash: u64,
}

pub(crate) fn merge_preview_timings(
    target: &mut CandidateStageTimings,
    preview: &CandidateStageTimings,
    reason: &'static str,
    used: bool,
) {
    target.preview_download = preview.preview_download;
    target.preview_fingerprint_us = preview.preview_fingerprint_us;
    target.preview_matcher_us = preview.preview_matcher_us;
    target.preview_used = used;
    target.preview_fallback_reason = Some(reason);
}

pub(crate) fn log_preview_original_agreement(
    candidate: &ImageCandidate,
    preview: Option<&MatchOutcome>,
    original: Option<&MatchOutcome>,
    reason: &'static str,
) {
    let preview_decision = preview.map_or("pass", MatchOutcome::decision_name);
    let original_decision = original.map_or("pass", MatchOutcome::decision_name);
    let preview_specimen = preview.map_or("none", |outcome| outcome.specimen_id.as_str());
    let original_specimen = original.map_or("none", |outcome| outcome.specimen_id.as_str());
    let same_specimen = preview_specimen == original_specimen;
    let same_decision = preview_decision == original_decision;
    info!(
        event = "preview.original_agreement",
        guild_id = candidate.guild_id.get(),
        channel_id = candidate.channel_id.get(),
        message_id = candidate.message_id.get(),
        candidate_index = candidate.candidate_index,
        preview_decision,
        original_decision,
        preview_specimen,
        original_specimen,
        same_decision,
        same_specimen,
        fallback_reason = reason,
        "Discord preview and original scan both completed before cancellation"
    );
}

pub(crate) fn warn_preview_failure(candidate: &ImageCandidate, source: &anyhow::Error) {
    warn!(
        event = "preview.match_failed",
        guild_id = candidate.guild_id.get(),
        channel_id = candidate.channel_id.get(),
        message_id = candidate.message_id.get(),
        ?source,
        "Discord preview matching failed; falling back to original image"
    );
}

pub(crate) fn choose_preview_route(
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
) -> PreviewRoute {
    build_discord_preview_request(candidate, download_config, match_config).map_or(
        PreviewRoute::OriginalOnly("preview_ineligible"),
        PreviewRoute::Precheck,
    )
}

pub(crate) async fn generate_specimen_preview_variant(
    state: &AppState,
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
) -> Option<SpecimenPreviewVariant> {
    let preview = build_discord_preview_request(candidate, download_config, match_config)?;
    let downloaded = match state
        .download_image(
            &preview.url,
            candidate.mime_hint.as_deref(),
            download_config,
        )
        .await
    {
        Ok(downloaded) => downloaded,
        Err(source) => {
            warn_specimen_preview_failure(
                candidate,
                &source,
                "specimen.preview_download_failed",
                "failed to download Discord preview for specimen; writing original-only specimen",
            );
            return None;
        }
    };
    let bytes = downloaded.bytes.clone();
    match hash_downloaded_image(
        downloaded,
        download_config.max_decoded_pixels,
        match_config,
        &state.decode_gate,
        HashMode::FullDiagnostics,
    )
    .await
    {
        Ok(fingerprint) => Some(SpecimenPreviewVariant { fingerprint, bytes }),
        Err(source) => {
            warn_specimen_preview_failure(
                candidate,
                &source,
                "specimen.preview_hash_failed",
                "failed to hash Discord preview for specimen; writing original-only specimen",
            );
            None
        }
    }
}

fn warn_specimen_preview_failure(
    candidate: &ImageCandidate,
    source: &anyhow::Error,
    event: &'static str,
    message: &'static str,
) {
    warn!(
        event = event,
        guild_id = candidate.guild_id.get(),
        channel_id = candidate.channel_id.get(),
        message_id = candidate.message_id.get(),
        ?source,
        "{message}"
    );
}

pub(crate) fn finish_preview_scan(
    preview_match: Option<MatchOutcome>,
    preview_image_id: String,
    preview_xxh128: String,
    policy_hash: u64,
    mut timings: CandidateStageTimings,
    started: Instant,
) -> PreviewScanResult {
    let Some(outcome) = preview_match else {
        return preview_fallback(timings, "preview_miss", None);
    };
    if !preview_outcome_allows_early_exit(&outcome) {
        return preview_fallback(timings, "preview_not_decisive", Some(outcome));
    }

    timings.preview_used = true;
    timings.total_us = started.elapsed().as_micros();
    PreviewScanResult::Decisive(Box::new(CandidateDecision::matched(
        preview_image_id,
        preview_xxh128,
        policy_hash,
        outcome,
        &timings,
    )))
}

fn preview_fallback(
    mut timings: CandidateStageTimings,
    reason: &'static str,
    outcome: Option<MatchOutcome>,
) -> PreviewScanResult {
    timings.preview_fallback_reason = Some(reason);
    PreviewScanResult::Fallback(Box::new(PreviewFallback {
        timings,
        reason,
        outcome,
    }))
}

pub(crate) fn preview_outcome_allows_early_exit(outcome: &MatchOutcome) -> bool {
    !outcome.suspicious
        && matches!(
            outcome.confidence,
            MatchConfidence::ExactXxh128
                | MatchConfidence::Perceptual
                | MatchConfidence::LocalAnchors
                | MatchConfidence::ClusterCoherence
                | MatchConfidence::DenseLocalAnchors
        )
}
