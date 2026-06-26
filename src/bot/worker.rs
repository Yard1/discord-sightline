use crate::{
    bot::{
        discord::{
            BotLogColor, BotLogEvent, DetectionActionResults, MemberActionResults,
            RoleRemovalOutcome, audit_row, detection_action_summary, message_jump_link,
            user_incident_label,
        },
        ledger::{SpecimenImageAttachment, SpecimenRecord},
        runtime::{
            AppState, BotState, HashProcessingClaim, ImageByteLease, ImageMetricsCommand,
            OcrCacheKey, SpecimenWriteLogContext, SpecimenWriteOutcome,
        },
        worker_logging::{
            log_scan_failure, mark_message_matched, record_nonmatching_sibling_inspection,
        },
        worker_preview::{
            PreviewScanContext, choose_preview_route, generate_specimen_preview_variant,
            log_preview_original_agreement, merge_preview_timings, scan_preview_candidate,
            warn_preview_failure,
        },
    },
    configuration::{
        app::{DownloadConfig, MatchConfig, effective_download_config},
        guild::GuildConfig,
    },
    image::{
        engine::{
            NoopArtifactSink, ProgressiveDecision, ProgressiveEngine, TextGateDecision,
            TextGateReport, TextGateVerdict, UnavailableOcrClient, VisualCandidateClass,
            VisualClassification,
        },
        matcher::confirmed_tier1_policy,
        pipeline::{
            DiscordPreviewRequest, DownloadTimings, DownloadedImage, HashMode, PipelineTimings,
            PreparedOcrCrop, complete_staged_image_fingerprint, hash_downloaded_image,
            hash_downloaded_image_tier1, log_image_result, prepare_ocr_payload_from_downloaded,
            source_ocr_payload,
        },
        types::{
            CachedDecisionOutcome, ImageCandidate, ImageFingerprint, ImageFingerprintTimingSample,
            ImageMatchStageMetric, ImageMetricEvent, ImagePerfSample, ImageScanDecisionMetric,
            ImageStageTimingSample, MatchConfidence, MatchOutcome, TextGateResolutionMetric,
        },
    },
};
use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::{
    fmt::Write as _,
    future::Future,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
use tokio::{sync::mpsc, task::JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, info, info_span, warn};
use xxhash_rust::xxh3::xxh3_64;

const MODERATION_ACTION_TIMEOUT: Duration = Duration::from_secs(15);
const WORKER_SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

pub(crate) struct CandidateDecision {
    pub(crate) image_id: String,
    pub(crate) byte_xxh128: String,
    pub(crate) policy_hash: u64,
    pub(crate) decision: ProgressiveDecision,
    pub(crate) specimen_candidate: Option<SpecimenAutoAdd>,
    pub(crate) ocr_followup: Option<OcrFollowup>,
    pub(crate) processing_guard: Option<HashProcessingGuard>,
    pub(crate) timings: CandidateStageTimings,
}

impl CandidateDecision {
    fn pass(
        image_id: impl Into<String>,
        byte_xxh128: impl Into<String>,
        policy_hash: u64,
        timings: &CandidateStageTimings,
    ) -> Self {
        Self {
            image_id: image_id.into(),
            byte_xxh128: byte_xxh128.into(),
            policy_hash,
            decision: ProgressiveDecision {
                class: VisualCandidateClass::NoEvidence,
                outcome: None,
                text_gate: None,
                ocr_requested: false,
            },
            specimen_candidate: None,
            ocr_followup: None,
            processing_guard: None,
            timings: *timings,
        }
    }

    pub(crate) fn matched(
        image_id: impl Into<String>,
        byte_xxh128: impl Into<String>,
        policy_hash: u64,
        outcome: MatchOutcome,
        timings: &CandidateStageTimings,
    ) -> Self {
        Self {
            image_id: image_id.into(),
            byte_xxh128: byte_xxh128.into(),
            policy_hash,
            decision: ProgressiveDecision {
                class: if outcome.suspicious {
                    VisualCandidateClass::KnownSuspicious
                } else {
                    VisualCandidateClass::KnownStrong
                },
                outcome: Some(outcome),
                text_gate: None,
                ocr_requested: false,
            },
            specimen_candidate: None,
            ocr_followup: None,
            processing_guard: None,
            timings: *timings,
        }
    }
}

pub(crate) struct SpecimenAutoAdd {
    fingerprint: ImageFingerprint,
    original: ImageByteLease,
    preview: Option<SpecimenPreviewVariant>,
    full_diagnostics: bool,
}

#[derive(Clone)]
pub(crate) struct SpecimenPreviewVariant {
    pub(crate) fingerprint: ImageFingerprint,
    pub(crate) bytes: bytes::Bytes,
}

pub(crate) struct OcrFollowup {
    original: Option<ImageByteLease>,
    fingerprint: ImageFingerprint,
    preview: Option<SpecimenPreviewVariant>,
    started: Instant,
}

pub(crate) struct OcrFollowupInput {
    state: AppState,
    candidate: ImageCandidate,
    image_id: String,
    outcome: MatchOutcome,
    policy_hash: u64,
    followup: OcrFollowup,
    processing_guard: Option<HashProcessingGuard>,
    log_response: Option<oneshot::Receiver<Option<BotLogRef>>>,
    hot_path_timings: CandidateStageTimings,
    trace_id: String,
}

pub(crate) struct OriginalAutoAddInput {
    state: AppState,
    candidate: ImageCandidate,
    outcome: MatchOutcome,
    policy_hash: u64,
    trace_id: String,
}

struct ScanConfigSnapshot {
    guild_config: Arc<GuildConfig>,
    policy_hash: u64,
    download_config: DownloadConfig,
    match_config: MatchConfig,
    auto_add_possible: bool,
}

pub(crate) enum PreviewScanResult {
    Decisive(Box<CandidateDecision>),
    Fallback(Box<PreviewFallback>),
}

pub(crate) struct PreviewFallback {
    pub(crate) timings: CandidateStageTimings,
    pub(crate) reason: &'static str,
    pub(crate) outcome: Option<MatchOutcome>,
}

pub(crate) enum PreviewRoute {
    OriginalOnly(&'static str),
    Precheck(DiscordPreviewRequest),
}

pub(crate) struct HashProcessingGuard {
    state: AppState,
    key: Option<OcrCacheKey>,
    complete: Option<Arc<tokio::sync::watch::Sender<bool>>>,
}

impl HashProcessingGuard {
    fn new(
        state: AppState,
        key: OcrCacheKey,
        complete: Arc<tokio::sync::watch::Sender<bool>>,
    ) -> Self {
        Self {
            state,
            key: Some(key),
            complete: Some(complete),
        }
    }

    fn finish(&mut self) {
        if let (Some(key), Some(complete)) = (self.key.take(), self.complete.take()) {
            self.state.finish_hash_processing(&key, &complete);
        }
    }
}

impl Drop for HashProcessingGuard {
    fn drop(&mut self) {
        self.finish();
    }
}

async fn wait_for_hash_processing_completion(complete: &mut tokio::sync::watch::Receiver<bool>) {
    if *complete.borrow_and_update() {
        return;
    }
    while complete.changed().await.is_ok() {
        if *complete.borrow_and_update() {
            return;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BotLogRef {
    channel_id: twilight_model::id::Id<twilight_model::id::marker::ChannelMarker>,
    message_id: twilight_model::id::Id<twilight_model::id::marker::MessageMarker>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct CandidateStageTimings {
    pub(crate) total_us: u128,
    pub(crate) preview_download: DownloadTimings,
    pub(crate) preview_fingerprint_us: u128,
    pub(crate) preview_matcher_us: u128,
    pub(crate) preview_used: bool,
    pub(crate) preview_fallback_reason: Option<&'static str>,
    pub(crate) queue_wait_us: u128,
    pub(crate) download: DownloadTimings,
    pub(crate) flagged_cache_lookup_us: u128,
    pub(crate) exact_match_lookup_us: u128,
    pub(crate) singleflight_wait_us: u128,
    pub(crate) fingerprint_us: u128,
    pub(crate) fingerprint_pipeline: PipelineTimings,
    pub(crate) matcher_us: u128,
    pub(crate) ocr_crop_us: u128,
    pub(crate) progressive_eval_us: u128,
}

pub(crate) async fn worker_loop(state: BotState, mut rx: mpsc::Receiver<ImageCandidate>) {
    let worker_count = state.config.queue.image_worker_concurrency();
    let mut tasks = JoinSet::new();

    loop {
        let candidate = tokio::select! {
            () = state.shutdown.cancelled() => break,
            candidate = rx.recv() => match candidate {
                Some(candidate) => candidate,
                None => break,
            },
        };
        while tasks.len() >= worker_count {
            join_logged_task(&mut tasks, "worker.task_failed", "image worker task failed").await;
        }

        let state = state.clone();
        tasks.spawn(async move {
            let guild_id = candidate.guild_id;
            match state.for_guild(guild_id).await {
                // This future captures the full image-processing state machine and is large.
                // Boxing keeps each spawned worker task small and avoids storing that state
                // inline in the worker-pool JoinSet.
                Ok(state) => Box::pin(process_worker_candidate(state, candidate)).await,
                Err(source) => warn!(
                    event = "guild.load_failed",
                    guild_id = guild_id.get(),
                    ?source,
                    "could not load guild runtime for image candidate"
                ),
            }
        });
    }

    drain_joinset_on_shutdown(
        &mut tasks,
        &state.shutdown,
        "worker.task_failed",
        "image worker task failed",
        "worker.shutdown_timeout",
        "image worker tasks exceeded shutdown grace; aborting",
    )
    .await;

    info!(event = "worker.stopped", "image worker stopped");
}

async fn process_worker_candidate(state: AppState, candidate: ImageCandidate) {
    let trace_id = image_trace_id(&candidate);
    let span = info_span!(
        "sightline.image",
        trace_id = %trace_id,
        guild_id = candidate.guild_id.get(),
        channel_id = candidate.channel_id.get(),
        message_id = candidate.message_id.get(),
        user_id = candidate.author_id.get(),
        kind = ?candidate.kind,
    );
    // The inner scan future carries download, decode, match, logging, and action state across
    // awaits. Boxing prevents that large state machine from inflating the outer worker future.
    Box::pin(process_worker_candidate_inner(state, candidate, trace_id).instrument(span)).await;
}

async fn process_worker_candidate_inner(
    state: AppState,
    candidate: ImageCandidate,
    trace_id: String,
) {
    let started = Instant::now();
    let queue_wait_us = candidate.enqueued_at.map_or(0, |enqueued_at| {
        started.duration_since(enqueued_at).as_micros()
    });
    let message_id = candidate.message_id.get();
    let result = Box::pin(process_candidate(&state, &candidate, queue_wait_us)).await;
    let elapsed = started.elapsed().as_millis();

    match result {
        Ok(decision) => {
            let Some(outcome) = decision.decision.outcome.clone() else {
                record_processed_metric(
                    &state,
                    candidate.guild_id,
                    elapsed,
                    true,
                    ImageScanDecisionMetric::Pass,
                    Some(&decision.timings),
                );
                log_image_result(
                    message_id,
                    candidate.kind,
                    &candidate.url,
                    true,
                    elapsed,
                    None,
                );
                let audit = audit_row(&candidate, &decision.image_id, "pass", None, &[], elapsed);
                info!(event = "decision.audit", audit, "image decision audit");
                record_nonmatching_sibling_inspection(
                    &state,
                    &candidate,
                    decision.image_id,
                    "pass".to_owned(),
                    elapsed,
                    trace_id,
                    None,
                )
                .await;
                return;
            };
            record_processed_metric(
                &state,
                candidate.guild_id,
                elapsed,
                true,
                decision_metric_for_outcome(&outcome),
                Some(&decision.timings),
            );
            // Applying an outcome includes moderation action futures and follow-up enqueue state.
            // Boxing keeps the common worker future from embedding all of that rarely-used state.
            Box::pin(apply_detection_outcome(DetectionOutcomeInput {
                state,
                candidate,
                image_id: decision.image_id,
                byte_xxh128: decision.byte_xxh128,
                outcome,
                policy_hash: decision.policy_hash,
                specimen_candidate: decision.specimen_candidate,
                ocr_followup: decision.ocr_followup,
                processing_guard: decision.processing_guard,
                progressive: decision.decision,
                timings: decision.timings,
                elapsed,
                trace_id,
            }))
            .await;
        }
        Err(source) => {
            let error = format!("{source:#}");
            record_processed_metric(
                &state,
                candidate.guild_id,
                elapsed,
                false,
                ImageScanDecisionMetric::ScanFailed,
                None,
            );
            log_image_result(
                message_id,
                candidate.kind,
                &candidate.url,
                false,
                elapsed,
                Some(&error),
            );
            log_scan_failure(&state, &candidate, &error, elapsed, &trace_id).await;
            record_nonmatching_sibling_inspection(
                &state,
                &candidate,
                "unavailable".to_owned(),
                "scan_failed".to_owned(),
                elapsed,
                trace_id,
                Some(error),
            )
            .await;
        }
    }
}

fn record_processed_metric(
    state: &AppState,
    guild_id: twilight_model::id::Id<twilight_model::id::marker::GuildMarker>,
    duration_ms: u128,
    success: bool,
    decision: ImageScanDecisionMetric,
    timings: Option<&CandidateStageTimings>,
) {
    let _ =
        state
            .image_metrics_tx
            .try_send(ImageMetricsCommand::Record(ImageMetricEvent::Processed(
                ImagePerfSample {
                    guild_id,
                    duration_ms,
                    success,
                    decision,
                    stage_timings: timings
                        .map(|timings| Box::new(image_stage_timing_sample(timings))),
                },
            )));
}

fn image_stage_timing_sample(timings: &CandidateStageTimings) -> ImageStageTimingSample {
    ImageStageTimingSample {
        total_us: metric_us(timings.total_us),
        preview_download_us: metric_us(timings.preview_download.total_us),
        preview_fingerprint_us: metric_us(timings.preview_fingerprint_us),
        preview_matcher_us: metric_us(timings.preview_matcher_us),
        preview_used: timings.preview_used,
        preview_fallback: timings.preview_fallback_reason.is_some(),
        queue_wait_us: metric_us(timings.queue_wait_us),
        download_us: metric_us(timings.download.total_us),
        download_request_us: metric_us(timings.download.request_us),
        download_body_us: metric_us(timings.download.body_us),
        download_gate_wait_us: metric_us(timings.download.gate_wait_us),
        flagged_cache_lookup_us: metric_us(timings.flagged_cache_lookup_us),
        exact_match_lookup_us: metric_us(timings.exact_match_lookup_us),
        singleflight_wait_us: metric_us(timings.singleflight_wait_us),
        fingerprint_us: metric_us(timings.fingerprint_us),
        fingerprint_pipeline: image_fingerprint_timing_sample(timings.fingerprint_pipeline),
        matcher_us: metric_us(timings.matcher_us),
        ocr_crop_us: metric_us(timings.ocr_crop_us),
        progressive_eval_us: metric_us(timings.progressive_eval_us),
    }
}

fn image_fingerprint_timing_sample(timings: PipelineTimings) -> ImageFingerprintTimingSample {
    ImageFingerprintTimingSample {
        decode: metric_us(timings.decode_us),
        thumbnail: metric_us(timings.whole_thumbnail_us),
        visual: metric_us(timings.visual_signature_us),
        orientation: metric_us(timings.orientation_us),
        perceptual: metric_us(timings.perceptual_hashes_us),
        normalize: metric_us(timings.normalize_luma_us),
        tile_scorer: metric_us(timings.base_tile_scorer_us),
        text_grid: metric_us(timings.text_grid_us),
        local_anchors: metric_us(timings.local_anchors_us),
        local_hashes: metric_us(timings.local_hashes_us),
    }
}

fn metric_us(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn decision_metric_for_outcome(outcome: &MatchOutcome) -> ImageScanDecisionMetric {
    let stage = match_stage_for_confidence(outcome.confidence);
    if outcome.suspicious {
        ImageScanDecisionMetric::Suspicious(stage)
    } else {
        ImageScanDecisionMetric::HardMatch(stage)
    }
}

fn match_stage_for_confidence(confidence: MatchConfidence) -> ImageMatchStageMetric {
    match confidence {
        MatchConfidence::ExactXxh128 => ImageMatchStageMetric::ExactXxh128,
        MatchConfidence::Perceptual | MatchConfidence::SuspiciousPerceptual => {
            ImageMatchStageMetric::Perceptual
        }
        MatchConfidence::LocalAnchors
        | MatchConfidence::ClusterCoherence
        | MatchConfidence::SuspiciousLocalAnchors
        | MatchConfidence::DenseLocalAnchors
        | MatchConfidence::SuspiciousDenseLocalAnchors => ImageMatchStageMetric::LocalAnchors,
    }
}

fn record_ocr_resolution_metric(
    state: &AppState,
    guild_id: twilight_model::id::Id<twilight_model::id::marker::GuildMarker>,
    report: &TextGateReport,
) {
    let resolution = match report.verdict {
        TextGateVerdict::Good => TextGateResolutionMetric::Good,
        TextGateVerdict::Bad => TextGateResolutionMetric::Bad,
        TextGateVerdict::Unknown | TextGateVerdict::Disabled => TextGateResolutionMetric::Unknown,
    };
    let _ = state.image_metrics_tx.try_send(ImageMetricsCommand::Record(
        ImageMetricEvent::OcrResolved {
            guild_id,
            resolution,
        },
    ));
}

struct DetectionOutcomeInput {
    state: AppState,
    candidate: ImageCandidate,
    image_id: String,
    byte_xxh128: String,
    outcome: MatchOutcome,
    policy_hash: u64,
    specimen_candidate: Option<SpecimenAutoAdd>,
    ocr_followup: Option<OcrFollowup>,
    pub(crate) processing_guard: Option<HashProcessingGuard>,
    progressive: ProgressiveDecision,
    timings: CandidateStageTimings,
    elapsed: u128,
    trace_id: String,
}

#[expect(
    clippy::too_many_lines,
    reason = "orchestrates moderation actions, logging enqueue, and OCR deferral in one transaction"
)]
async fn apply_detection_outcome(input: DetectionOutcomeInput) {
    let DetectionOutcomeInput {
        state,
        candidate,
        image_id,
        byte_xxh128,
        outcome,
        policy_hash,
        specimen_candidate,
        ocr_followup,
        processing_guard,
        progressive,
        timings,
        elapsed,
        trace_id,
    } = input;

    if candidate.guild_id != state.guild_id() {
        warn!(
            event = "moderation.foreign_guild_blocked",
            configured_guild_id = state.guild_id().get(),
            candidate_guild_id = candidate.guild_id.get(),
            channel_id = candidate.channel_id.get(),
            message_id = candidate.message_id.get(),
            "blocked moderation action outside configured guild"
        );
        return;
    }

    state.record_specimen_hit(&outcome.specimen_id);

    let runtime_config = state.active_config_arc();
    let mut effective_outcome = outcome.clone();
    let ocr_promoted_to_confirmed =
        outcome.suspicious && text_gate_confirms_bad(progressive.text_gate.as_ref());
    if ocr_promoted_to_confirmed {
        effective_outcome.suspicious = false;
    }
    if ocr_followup.is_none() {
        state.hash_outcome_cache.lock().insert_match(
            byte_xxh128.clone(),
            policy_hash,
            &effective_outcome,
        );
    }
    let detection_rule = if effective_outcome.suspicious {
        runtime_config.detection_policy.suspicious.clone()
    } else {
        runtime_config.detection_policy.confirmed.clone()
    };
    let detection_actions = if candidate.verify_only {
        crate::configuration::guild::DetectionActions::default()
    } else {
        detection_rule.actions
    };
    let specimen_candidate = if candidate.verify_only {
        None
    } else {
        specimen_candidate
    };

    if state.safe_mode.load(Ordering::Acquire) {
        info!(
            event = "moderation.safe_mode_match",
            guild_id = candidate.guild_id.get(),
            channel_id = candidate.channel_id.get(),
            message_id = candidate.message_id.get(),
            user_id = candidate.author_id.get(),
            specimen_id = %outcome.specimen_id,
            suspicious = effective_outcome.suspicious,
            original_suspicious = outcome.suspicious,
            ocr_promoted_to_confirmed,
            process_ms = elapsed,
            trace_id = %trace_id,
            diagnostics = %serde_json::to_string(&outcome.diagnostics).unwrap_or_else(|_| "unavailable".to_owned()),
            "safe-mode detection completed"
        );
        enqueue_detection_followup(DetectionFollowup {
            state,
            candidate,
            image_id,
            outcome: effective_outcome,
            policy_hash,
            progressive,
            detection_actions,
            action_results: DetectionActionResults::default(),
            specimen_candidate: None,
            timings,
            elapsed,
            safe_mode: true,
            ocr_promoted_to_confirmed,
            trace_id,
            actions_deferred: false,
            respond_to: None,
        })
        .await;
        return;
    }

    let actions_deferred_for_ocr = ocr_followup.is_some() && effective_outcome.suspicious;
    let action_results = if actions_deferred_for_ocr {
        DetectionActionResults::default()
    } else {
        apply_moderation_actions(&state, &candidate, &detection_actions).await
    };
    if !candidate.verify_only {
        mark_message_matched(&state, &candidate).await;
    }

    info!(
        event = "moderation.match_action",
        guild_id = candidate.guild_id.get(),
        channel_id = candidate.channel_id.get(),
        message_id = candidate.message_id.get(),
        user_id = candidate.author_id.get(),
        specimen_id = %effective_outcome.specimen_id,
        suspicious = effective_outcome.suspicious,
        original_suspicious = outcome.suspicious,
        ocr_promoted_to_confirmed,
        deleted = action_results.deleted,
        roles_remove_attempted = action_results.role_removal.attempted,
        roles_removed = action_results.role_removal.removed,
        roles_remove_failed = action_results.role_removal.failed,
        timed_out = action_results.member.timed_out,
        timeout_seconds = detection_actions.timeout_seconds,
        banned = action_results.member.banned,
        kicked = action_results.member.kicked,
        process_ms = elapsed,
        trace_id = %trace_id,
        timings = %stage_timing_summary(&timings),
        visual_class = ?progressive.class,
        ocr_requested = progressive.ocr_requested,
        diagnostics = %serde_json::to_string(&outcome.diagnostics).unwrap_or_else(|_| "unavailable".to_owned()),
        "moderation action completed"
    );

    let (respond_to, response) = if ocr_followup.is_some() {
        let (respond_to, response) = oneshot::channel();
        (Some(respond_to), Some(response))
    } else {
        (None, None)
    };
    let ocr_state = state.clone();
    let ocr_candidate = candidate.clone();
    let ocr_image_id = image_id.clone();
    let ocr_outcome = effective_outcome.clone();
    let ocr_trace_id = trace_id.clone();
    enqueue_detection_followup(DetectionFollowup {
        state,
        candidate,
        image_id,
        outcome: effective_outcome,
        policy_hash,
        progressive,
        detection_actions: if actions_deferred_for_ocr {
            crate::configuration::guild::DetectionActions::default()
        } else {
            detection_actions
        },
        action_results,
        specimen_candidate: if actions_deferred_for_ocr {
            None
        } else {
            specimen_candidate
        },
        timings,
        elapsed,
        safe_mode: false,
        ocr_promoted_to_confirmed,
        trace_id,
        actions_deferred: actions_deferred_for_ocr,
        respond_to,
    })
    .await;
    if let (Some(ocr_followup), Some(response)) = (ocr_followup, response) {
        let input = OcrFollowupInput {
            state: ocr_state.clone(),
            candidate: ocr_candidate,
            image_id: ocr_image_id,
            outcome: ocr_outcome,
            policy_hash,
            followup: ocr_followup,
            processing_guard,
            log_response: Some(response),
            hot_path_timings: timings,
            trace_id: ocr_trace_id,
        };
        if let Err(source) = ocr_state.bot.ocr_followup_tx.send(input).await {
            warn!(
                event = "ocr_followup.enqueue_failed",
                ?source,
                "failed to enqueue OCR follow-up"
            );
        }
    }
}

pub(crate) async fn ocr_followup_loop(
    shutdown: CancellationToken,
    mut rx: mpsc::Receiver<OcrFollowupInput>,
) {
    let mut tasks = JoinSet::new();
    loop {
        let input = tokio::select! {
            () = shutdown.cancelled() => break,
            input = rx.recv() => match input {
                Some(input) => input,
                None => break,
            },
        };
        let concurrency = input.state.config.queue.ocr_concurrency;
        while tasks.len() >= concurrency {
            join_logged_task(
                &mut tasks,
                "ocr_followup.task_failed",
                "OCR follow-up task failed",
            )
            .await;
        }
        tasks.spawn(
            async move { run_ocr_followup(input).await }.instrument(info_span!("ocr_followup")),
        );
    }
    drain_joinset_on_shutdown(
        &mut tasks,
        &shutdown,
        "ocr_followup.task_failed",
        "OCR follow-up task failed",
        "ocr_followup.shutdown_timeout",
        "OCR follow-up tasks exceeded shutdown grace; aborting",
    )
    .await;
    info!(
        event = "ocr_followup.stopped",
        "OCR follow-up worker stopped"
    );
}

pub(crate) async fn original_auto_add_loop(
    shutdown: CancellationToken,
    mut rx: mpsc::Receiver<OriginalAutoAddInput>,
) {
    let mut tasks = JoinSet::new();
    loop {
        let input = tokio::select! {
            () = shutdown.cancelled() => break,
            input = rx.recv() => match input {
                Some(input) => input,
                None => break,
            },
        };
        let concurrency = input.state.config.queue.original_auto_add_concurrency();
        while tasks.len() >= concurrency {
            join_logged_task(
                &mut tasks,
                "original_auto_add.task_failed",
                "original auto-add task failed",
            )
            .await;
        }
        tasks.spawn(
            async move { run_original_auto_add(input).await }
                .instrument(info_span!("original_auto_add")),
        );
    }
    drain_joinset_on_shutdown(
        &mut tasks,
        &shutdown,
        "original_auto_add.task_failed",
        "original auto-add task failed",
        "original_auto_add.shutdown_timeout",
        "original auto-add tasks exceeded shutdown grace; aborting",
    )
    .await;
    info!(
        event = "original_auto_add.stopped",
        "original auto-add worker stopped"
    );
}

async fn run_original_auto_add(input: OriginalAutoAddInput) {
    let OriginalAutoAddInput {
        state,
        candidate,
        outcome,
        policy_hash,
        trace_id,
    } = input;
    let started = Instant::now();
    if state.safe_mode.load(Ordering::Acquire) {
        info!(
            event = "specimen.auto_add_original_skipped",
            guild_id = candidate.guild_id.get(),
            channel_id = candidate.channel_id.get(),
            message_id = candidate.message_id.get(),
            trace_id = %trace_id,
            reason = "safe_mode",
            "skipped deferred original auto-add"
        );
        return;
    }

    let config = state.active_config_arc();
    let download_config = effective_download_config(&state.config.download, &config.scan_policy);
    if let Some(reason) = metadata_safety_rejection_reason(&candidate, &download_config) {
        warn!(
            event = "specimen.auto_add_original_skipped",
            guild_id = candidate.guild_id.get(),
            channel_id = candidate.channel_id.get(),
            message_id = candidate.message_id.get(),
            trace_id = %trace_id,
            reason,
            "skipped deferred original auto-add before download"
        );
        return;
    }

    let match_config = config
        .detection_hyperparameters
        .effective_match_config(&state.config.matching);
    let Some(downloaded) =
        download_original_for_auto_add(&state, &candidate, &download_config, &trace_id).await
    else {
        return;
    };
    let Some((fingerprint, bytes)) = hash_original_for_auto_add(
        &state,
        &candidate,
        downloaded,
        &download_config,
        &match_config,
        &trace_id,
    )
    .await
    else {
        return;
    };
    let Some(original) = retain_image_bytes(&state, &candidate, &fingerprint.byte_xxh128, bytes)
    else {
        return;
    };
    let preview =
        generate_specimen_preview_variant(&state, &candidate, &download_config, &match_config)
            .await;
    let specimen_candidate = SpecimenAutoAdd {
        fingerprint,
        original,
        preview,
        full_diagnostics: true,
    };
    let image_processing_ms = started.elapsed().as_millis();
    let label = add_matched_candidate_to_specimens_after_action(
        &state,
        &candidate,
        Some(specimen_candidate),
        &outcome,
        policy_hash,
        Some(image_processing_ms),
    )
    .await;
    info!(
        event = "specimen.auto_add_original_completed",
        guild_id = candidate.guild_id.get(),
        channel_id = candidate.channel_id.get(),
        message_id = candidate.message_id.get(),
        trace_id = %trace_id,
        result = label.as_deref().unwrap_or("none"),
        process_ms = image_processing_ms,
        "completed deferred original auto-add"
    );
}

async fn download_original_for_auto_add(
    state: &AppState,
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
    trace_id: &str,
) -> Option<DownloadedImage> {
    match state
        .download_image(
            &candidate.url,
            candidate.mime_hint.as_deref(),
            download_config,
        )
        .await
    {
        Ok(downloaded) => Some(downloaded),
        Err(source) => {
            warn!(
                event = "specimen.auto_add_original_download_failed",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                trace_id,
                ?source,
                "failed to download original image for deferred auto-add"
            );
            None
        }
    }
}

async fn download_original_for_ocr(
    state: &AppState,
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
    image_id: &str,
) -> Option<DownloadedImage> {
    match state
        .download_image(
            &candidate.url,
            candidate.mime_hint.as_deref(),
            download_config,
        )
        .await
    {
        Ok(downloaded) => Some(downloaded),
        Err(source) => {
            warn!(
                event = "ocr.original_redownload_failed",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_id,
                ?source,
                "failed to redownload original image for OCR follow-up"
            );
            None
        }
    }
}

async fn hash_original_for_auto_add(
    state: &AppState,
    candidate: &ImageCandidate,
    downloaded: DownloadedImage,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
    trace_id: &str,
) -> Option<(ImageFingerprint, bytes::Bytes)> {
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
        Ok(fingerprint) => Some((fingerprint, bytes)),
        Err(source) => {
            warn!(
                event = "specimen.auto_add_original_hash_failed",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                trace_id,
                ?source,
                "failed to hash original image for deferred auto-add"
            );
            None
        }
    }
}

fn retain_image_bytes(
    state: &AppState,
    candidate: &ImageCandidate,
    byte_xxh128: &str,
    bytes: bytes::Bytes,
) -> Option<ImageByteLease> {
    let len = bytes.len();
    let retained = state
        .bot
        .image_byte_store
        .insert(byte_xxh128.to_owned(), bytes);
    if retained.is_none() {
        warn!(
            event = "image_byte_store.full",
            guild_id = candidate.guild_id.get(),
            channel_id = candidate.channel_id.get(),
            message_id = candidate.message_id.get(),
            image_id = %byte_xxh128,
            bytes = len,
            "could not retain original image bytes for follow-up work"
        );
    }
    retained
}

#[expect(
    clippy::too_many_lines,
    reason = "keeps OCR resolution, action application, cache update, and log edit ordering explicit"
)]
async fn run_ocr_followup(input: OcrFollowupInput) {
    let OcrFollowupInput {
        state,
        candidate,
        image_id,
        outcome,
        policy_hash,
        followup,
        mut processing_guard,
        log_response,
        hot_path_timings,
        trace_id,
    } = input;
    let byte_xxh128 = followup.fingerprint.byte_xxh128.clone();
    let ocr_key = OcrCacheKey {
        byte_xxh128: byte_xxh128.clone(),
        policy_hash,
    };
    let cell = state.ocr_singleflight_cell(byte_xxh128.clone(), policy_hash);
    let report = cell
        .get_or_init(|| async {
            run_single_ocr_followup(&state, &candidate, &image_id, &outcome, &followup).await
        })
        .await
        .clone();
    record_ocr_resolution_metric(&state, candidate.guild_id, &report);

    let mut final_outcome = outcome.clone();
    let ocr_promoted_to_confirmed = text_gate_confirms_bad(Some(&report));
    if ocr_promoted_to_confirmed {
        final_outcome.suspicious = false;
    }

    if matches!(report.verdict, TextGateVerdict::Good) {
        state
            .hash_outcome_cache
            .lock()
            .insert_pass(byte_xxh128, policy_hash);
    } else {
        state
            .hash_outcome_cache
            .lock()
            .insert_match(byte_xxh128, policy_hash, &final_outcome);
    }
    state.ocr_singleflight.remove(&ocr_key);
    if let Some(guard) = processing_guard.as_mut() {
        guard.finish();
    }
    let current_policy_hash = state.detection_policy_hash();
    let current_config = state.active_config_arc();
    let policy_changed = current_policy_hash != policy_hash;
    let guild_disabled = !current_config.enabled;
    let safe_mode = state.safe_mode.load(Ordering::Acquire);
    let final_actions = if safe_mode
        || guild_disabled
        || policy_changed
        || candidate.verify_only
        || matches!(report.verdict, TextGateVerdict::Good)
    {
        None
    } else if ocr_promoted_to_confirmed {
        Some(current_config.detection_policy.confirmed.actions.clone())
    } else if final_outcome.suspicious {
        Some(current_config.detection_policy.suspicious.actions.clone())
    } else {
        None
    };
    let final_action_results = if let Some(actions) = final_actions.as_ref() {
        apply_moderation_actions(&state, &candidate, actions).await
    } else {
        DetectionActionResults::default()
    };
    let specimen_action = if let Some(actions) = final_actions.as_ref() {
        if actions.add_to_specimens {
            let specimen_candidate = followup.original.clone().map(|original| SpecimenAutoAdd {
                fingerprint: followup.fingerprint.clone(),
                original,
                preview: followup.preview.clone(),
                full_diagnostics: false,
            });
            add_or_enqueue_matched_candidate_to_specimens_after_action(SpecimenAfterActionInput {
                state: &state,
                candidate: &candidate,
                specimen_candidate,
                outcome: &final_outcome,
                policy_hash,
                image_processing_ms: Some(us_to_ms(hot_path_timings.total_us)),
                redownload_if_missing: followup.original.is_none(),
                trace_id: &trace_id,
            })
            .await
        } else {
            None
        }
    } else {
        None
    };

    let progressive = ProgressiveDecision {
        class: if final_outcome.suspicious {
            VisualCandidateClass::KnownSuspicious
        } else {
            VisualCandidateClass::KnownStrong
        },
        outcome: Some(final_outcome.clone()),
        text_gate: Some(report),
        ocr_requested: true,
    };
    let message_link = message_jump_link(
        candidate.guild_id,
        candidate.channel_id,
        candidate.message_id,
    );
    let specimen_link = state
        .specimen_ledger_link(&final_outcome.specimen_id)
        .unwrap_or_else(|| "unavailable".to_owned());
    let mut actions_taken = Vec::new();
    if let Some(actions) = final_actions.as_ref() {
        actions_taken.extend(detection_action_summary(
            actions,
            final_action_results,
            specimen_action.as_deref(),
        ));
    }
    if ocr_promoted_to_confirmed {
        actions_taken.push("`ocr_gate=promoted_to_confirmed`".to_owned());
    } else if progressive
        .text_gate
        .as_ref()
        .is_some_and(|report| matches!(report.verdict, TextGateVerdict::Good))
    {
        actions_taken.push("`ocr_gate=cleared`".to_owned());
    }
    if safe_mode {
        actions_taken.push("`actions=skipped_safe_mode`".to_owned());
    }
    if guild_disabled {
        actions_taken.push("`actions=skipped_guild_disabled`".to_owned());
    }
    if policy_changed {
        actions_taken.push("`actions=skipped_policy_changed`".to_owned());
    }
    if candidate.verify_only {
        actions_taken.push("`actions=skipped_manual_verify`".to_owned());
    }
    let log_ref = if let Some(response) = log_response {
        match tokio::time::timeout(Duration::from_secs(10), response).await {
            Ok(Ok(log_ref)) => log_ref,
            Ok(Err(source)) => {
                warn!(
                    event = "ocr_followup.log_wait_failed",
                    ?source,
                    "could not wait for pending OCR log"
                );
                None
            }
            Err(_) => {
                warn!(
                    event = "ocr_followup.log_wait_timeout",
                    "timed out waiting for pending OCR log"
                );
                None
            }
        }
    } else {
        None
    };
    if let Some(log_ref) = log_ref {
        state
            .edit_bot_log(
                log_ref.channel_id,
                log_ref.message_id,
                detection_bot_log(&DetectionLogInput {
                    candidate: &candidate,
                    image_id: &image_id,
                    outcome: &final_outcome,
                    progressive: &progressive,
                    timings: &hot_path_timings,
                    message_link: &message_link,
                    specimen_link: &specimen_link,
                    actions_taken: &actions_taken,
                    elapsed: followup.started.elapsed().as_millis(),
                    safe_mode: false,
                    ocr_promoted_to_confirmed,
                    trace_id: &trace_id,
                }),
            )
            .await;
    }
}

async fn run_single_ocr_followup(
    state: &AppState,
    candidate: &ImageCandidate,
    image_id: &str,
    outcome: &MatchOutcome,
    followup: &OcrFollowup,
) -> TextGateReport {
    let config = state.active_config_arc();
    let download_config = effective_download_config(&state.config.download, &config.scan_policy);
    let match_config = config
        .detection_hyperparameters
        .effective_match_config(&state.config.matching);
    let ocr_crop_started = Instant::now();
    let ocr_crops = prepare_ocr_crops_for_followup(
        state,
        candidate,
        image_id,
        followup,
        &download_config,
        &match_config,
    )
    .await;
    let missing_ocr = UnavailableOcrClient::new("OCR_SPACE_API_KEY is not configured");
    let ocr: &dyn crate::image::engine::OcrClient = state.ocr_space.as_deref().map_or(
        &missing_ocr as &dyn crate::image::engine::OcrClient,
        |client| client as &dyn crate::image::engine::OcrClient,
    );
    let engine = ProgressiveEngine {
        matcher: None,
        detection_policy: &config.detection_policy,
        text_gate_policy: &config.text_gate_policy,
        ocr,
        artifacts: &NoopArtifactSink,
    };
    let artifact_id = format!("{}_{}", candidate.message_id.get(), image_id);
    let evaluate = engine.evaluate_with_visual(
        VisualClassification::KnownSuspicious(outcome.clone()),
        Some(ocr_crops.as_slice()),
        &artifact_id,
    );
    let _ =
        state
            .image_metrics_tx
            .try_send(ImageMetricsCommand::Record(ImageMetricEvent::OcrCall {
                guild_id: candidate.guild_id,
            }));
    let ocr_timeout = Duration::from_secs(state.config.ocr_space.total_timeout_seconds);
    let keyword_threshold = config.text_gate_policy.keyword_threshold;
    match tokio::time::timeout(ocr_timeout, evaluate).await {
        Ok(Ok(decision)) => decision
            .text_gate
            .unwrap_or_else(|| TextGateReport::pending(keyword_threshold)),
        Ok(Err(source)) => TextGateReport::unavailable(keyword_threshold, &source.to_string()),
        Err(_) => {
            warn!(
                event = "ocr.deadline_exceeded",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_id,
                timeout_ms = ocr_timeout.as_millis(),
                prepare_ms = ocr_crop_started.elapsed().as_millis(),
                "OCR deadline exceeded"
            );
            TextGateReport::unavailable(keyword_threshold, "OCR deadline exceeded")
        }
    }
}

async fn prepare_ocr_crops_for_followup(
    state: &AppState,
    candidate: &ImageCandidate,
    image_id: &str,
    followup: &OcrFollowup,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
) -> Vec<PreparedOcrCrop> {
    let Some(original) = followup.original.as_ref() else {
        return prepare_ocr_crops_by_redownload(
            state,
            candidate,
            image_id,
            followup,
            download_config,
            match_config,
        )
        .await;
    };
    let bytes = original.bytes();
    if let Some(payload) = source_ocr_payload(
        bytes.as_ref(),
        followup.fingerprint.mime.as_deref(),
        followup.fingerprint.width,
        followup.fingerprint.height,
    ) {
        return vec![payload];
    }

    match prepare_ocr_payload_from_downloaded(
        bytes,
        download_config.max_decoded_pixels,
        match_config,
        &state.decode_gate,
    )
    .await
    {
        Ok(payload) => vec![payload],
        Err(source) => {
            warn!(
                event = "ocr.crop_failed",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_id,
                ?source,
                "failed to prepare OCR crop"
            );
            Vec::new()
        }
    }
}

async fn prepare_ocr_crops_by_redownload(
    state: &AppState,
    candidate: &ImageCandidate,
    image_id: &str,
    followup: &OcrFollowup,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
) -> Vec<PreparedOcrCrop> {
    let Some(downloaded) =
        download_original_for_ocr(state, candidate, download_config, image_id).await
    else {
        return Vec::new();
    };
    let mime = downloaded.mime;
    let bytes = downloaded.bytes;
    if let Some(payload) = source_ocr_payload(
        bytes.as_ref(),
        mime.as_deref().or(followup.fingerprint.mime.as_deref()),
        followup.fingerprint.width,
        followup.fingerprint.height,
    ) {
        return vec![payload];
    }
    match prepare_ocr_payload_from_downloaded(
        bytes,
        download_config.max_decoded_pixels,
        match_config,
        &state.decode_gate,
    )
    .await
    {
        Ok(payload) => vec![payload],
        Err(source) => {
            warn!(
                event = "ocr.crop_failed",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_id,
                ?source,
                "failed to prepare OCR crop after redownloading original"
            );
            Vec::new()
        }
    }
}

async fn apply_moderation_actions(
    state: &AppState,
    candidate: &ImageCandidate,
    detection_actions: &crate::configuration::guild::DetectionActions,
) -> DetectionActionResults {
    let runtime_config = state.active_config_arc();
    let delete_action = async {
        if detection_actions.delete_message {
            moderation_action_or_default(
                "delete_message",
                state
                    .discord_effects
                    .delete_message(candidate.channel_id, candidate.message_id),
            )
            .await
        } else {
            false
        }
    };

    let member_actions = async {
        let timed_out = if detection_actions.timeout_user {
            moderation_action_or_default(
                "timeout_user",
                state.discord_effects.timeout_user(
                    candidate.guild_id,
                    candidate.author_id,
                    detection_actions.timeout_seconds,
                    "Sightline scam image match".to_owned(),
                ),
            )
            .await
        } else {
            false
        };

        let role_removal = if detection_actions.remove_user_roles {
            let removable_roles = runtime_config
                .verified_role_id()
                .into_iter()
                .collect::<Vec<_>>();
            moderation_action_or_default(
                "remove_user_roles",
                state.discord_effects.remove_member_roles(
                    candidate.guild_id,
                    candidate.author_id,
                    removable_roles,
                ),
            )
            .await
        } else {
            RoleRemovalOutcome::default()
        };

        let banned = if detection_actions.ban_user {
            moderation_action_or_default(
                "ban_user",
                state.discord_effects.ban_user(
                    candidate.guild_id,
                    candidate.author_id,
                    detection_actions.ban_delete_message_seconds,
                    "Sightline scam image match".to_owned(),
                ),
            )
            .await
        } else {
            false
        };

        let kicked = if detection_actions.kick_user {
            moderation_action_or_default(
                "kick_user",
                state.discord_effects.kick_user(
                    candidate.guild_id,
                    candidate.author_id,
                    "Sightline scam image match".to_owned(),
                ),
            )
            .await
        } else {
            false
        };

        (role_removal, timed_out, banned, kicked)
    };

    let (deleted, (role_removal, timed_out, banned, kicked)) =
        tokio::join!(delete_action, member_actions);
    DetectionActionResults {
        deleted,
        role_removal,
        member: MemberActionResults {
            timed_out,
            banned,
            kicked,
        },
    }
}

async fn moderation_action_or_default<T, F>(action: &'static str, future: F) -> T
where
    T: Default,
    F: Future<Output = T>,
{
    if let Ok(result) = tokio::time::timeout(MODERATION_ACTION_TIMEOUT, future).await {
        result
    } else {
        warn!(
            event = "moderation.action_timeout",
            action,
            timeout_ms = MODERATION_ACTION_TIMEOUT.as_millis(),
            "moderation action timed out; continuing to log detection"
        );
        T::default()
    }
}

pub(crate) struct DetectionFollowup {
    state: AppState,
    candidate: ImageCandidate,
    image_id: String,
    outcome: MatchOutcome,
    policy_hash: u64,
    progressive: ProgressiveDecision,
    detection_actions: crate::configuration::guild::DetectionActions,
    action_results: DetectionActionResults,
    specimen_candidate: Option<SpecimenAutoAdd>,
    timings: CandidateStageTimings,
    elapsed: u128,
    safe_mode: bool,
    ocr_promoted_to_confirmed: bool,
    trace_id: String,
    actions_deferred: bool,
    respond_to: Option<oneshot::Sender<Option<BotLogRef>>>,
}

pub(crate) async fn detection_followup_loop(
    shutdown: CancellationToken,
    mut rx: mpsc::Receiver<DetectionFollowup>,
) {
    let mut tasks = JoinSet::new();
    loop {
        let input = tokio::select! {
            () = shutdown.cancelled() => break,
            input = rx.recv() => match input {
                Some(input) => input,
                None => break,
            },
        };
        let concurrency = input.state.config.queue.detection_followup_concurrency();
        while tasks.len() >= concurrency {
            join_logged_task(
                &mut tasks,
                "detection_followup.task_failed",
                "detection follow-up task failed",
            )
            .await;
        }
        tasks.spawn(
            async move { detection_followup(input).await }
                .instrument(info_span!("detection_followup")),
        );
    }
    drain_joinset_on_shutdown(
        &mut tasks,
        &shutdown,
        "detection_followup.task_failed",
        "detection follow-up task failed",
        "detection_followup.shutdown_timeout",
        "detection follow-up tasks exceeded shutdown grace; aborting",
    )
    .await;
    info!(
        event = "detection_followup.stopped",
        "detection follow-up worker stopped"
    );
}

async fn join_logged_task(tasks: &mut JoinSet<()>, event: &'static str, message: &'static str) {
    if let Some(result) = tasks.join_next().await
        && let Err(source) = result
    {
        warn!(event, ?source, message);
    }
}

async fn drain_joinset_on_shutdown(
    tasks: &mut JoinSet<()>,
    shutdown: &CancellationToken,
    failed_event: &'static str,
    failed_message: &'static str,
    timeout_event: &'static str,
    timeout_message: &'static str,
) {
    if shutdown.is_cancelled() {
        let timed_out = tokio::time::timeout(WORKER_SHUTDOWN_GRACE, async {
            while !tasks.is_empty() {
                join_logged_task(tasks, failed_event, failed_message).await;
            }
        })
        .await
        .is_err();
        if timed_out {
            warn!(
                event = timeout_event,
                grace_ms = WORKER_SHUTDOWN_GRACE.as_millis(),
                timeout_message
            );
            tasks.abort_all();
        }
    }

    while !tasks.is_empty() {
        join_logged_task(tasks, failed_event, failed_message).await;
    }
}

async fn enqueue_detection_followup(input: DetectionFollowup) {
    let state = input.state.clone();
    if let Err(source) = state.bot.detection_followup_tx.send(input).await {
        warn!(
            event = "detection_followup.enqueue_failed",
            ?source,
            "failed to enqueue detection follow-up"
        );
    }
}

struct SpecimenAfterActionInput<'a> {
    state: &'a AppState,
    candidate: &'a ImageCandidate,
    specimen_candidate: Option<SpecimenAutoAdd>,
    outcome: &'a MatchOutcome,
    policy_hash: u64,
    image_processing_ms: Option<u128>,
    redownload_if_missing: bool,
    trace_id: &'a str,
}

async fn add_or_enqueue_matched_candidate_to_specimens_after_action(
    input: SpecimenAfterActionInput<'_>,
) -> Option<String> {
    if let Some(specimen_candidate) = input.specimen_candidate {
        return add_matched_candidate_to_specimens_after_action(
            input.state,
            input.candidate,
            Some(specimen_candidate),
            input.outcome,
            input.policy_hash,
            input.image_processing_ms,
        )
        .await;
    }
    if input.redownload_if_missing {
        return Some(enqueue_original_auto_add(
            input.state,
            input.candidate,
            input.outcome,
            input.policy_hash,
            input.trace_id,
        ));
    }
    if matches!(input.outcome.confidence, MatchConfidence::ExactXxh128) {
        Some("skipped_duplicate".to_owned())
    } else {
        Some("bytes_unavailable".to_owned())
    }
}

fn enqueue_original_auto_add(
    state: &AppState,
    candidate: &ImageCandidate,
    outcome: &MatchOutcome,
    policy_hash: u64,
    trace_id: &str,
) -> String {
    let input = OriginalAutoAddInput {
        state: state.clone(),
        candidate: candidate.clone(),
        outcome: outcome.clone(),
        policy_hash,
        trace_id: trace_id.to_owned(),
    };
    match state.bot.original_auto_add_tx.try_send(input) {
        Ok(()) => "deferred_original".to_owned(),
        Err(mpsc::error::TrySendError::Full(_)) => {
            warn!(
                event = "specimen.auto_add_original_queue_full",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                trace_id,
                "could not enqueue deferred original auto-add"
            );
            "deferred_queue_full".to_owned()
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            warn!(
                event = "specimen.auto_add_original_queue_closed",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                trace_id,
                "could not enqueue deferred original auto-add"
            );
            "deferred_queue_closed".to_owned()
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "assembles final detection audit/log payload after moderation side effects complete"
)]
async fn detection_followup(input: DetectionFollowup) {
    let specimen_action = if !input.safe_mode && input.detection_actions.add_to_specimens {
        add_or_enqueue_matched_candidate_to_specimens_after_action(SpecimenAfterActionInput {
            state: &input.state,
            candidate: &input.candidate,
            specimen_candidate: input.specimen_candidate,
            outcome: &input.outcome,
            policy_hash: input.policy_hash,
            image_processing_ms: Some(us_to_ms(input.timings.total_us)),
            redownload_if_missing: input.timings.preview_used
                || !matches!(input.outcome.confidence, MatchConfidence::ExactXxh128),
            trace_id: &input.trace_id,
        })
        .await
    } else {
        None
    };
    let specimen_add_label = specimen_action.as_deref();
    let mut actions_taken = detection_action_summary(
        &input.detection_actions,
        input.action_results,
        specimen_add_label,
    );
    if input.ocr_promoted_to_confirmed {
        actions_taken.push("`ocr_gate=promoted_to_confirmed`".to_owned());
    }
    if input.actions_deferred {
        actions_taken.push("`actions=deferred_for_ocr`".to_owned());
    }
    if input.safe_mode {
        actions_taken.clear();
    }
    if input.candidate.verify_only {
        actions_taken.push("`actions=skipped_manual_verify`".to_owned());
    }
    let audit = audit_row(
        &input.candidate,
        &input.image_id,
        input.outcome.decision_name(),
        Some(&input.outcome),
        &actions_taken,
        input.elapsed,
    );
    info!(event = "decision.audit", audit, "image decision audit");
    let message_link = message_jump_link(
        input.candidate.guild_id,
        input.candidate.channel_id,
        input.candidate.message_id,
    );
    let specimen_link = input
        .state
        .specimen_ledger_link(&input.outcome.specimen_id)
        .unwrap_or_else(|| "unavailable".to_owned());
    let label = if input.safe_mode {
        format!("Safe-mode {}", input.outcome.label())
    } else {
        input.outcome.label().to_owned()
    };
    let details = format!(
        "{}: target user {}. Message: {}. Image: {}\nTrace ID: `{}`.\nMatched specimen: {}. Confidence: `{:?}`, class `{:?}`, OCR `{}`, text gate {}{} {}{}.\nProcessed: `{}`ms. Stage timings: `{}`. Actions taken: {}.\nGates tripped: `{}`\nDiagnostics: `{}`\nAudit: `{}`",
        label,
        user_incident_label(&input.candidate),
        message_link,
        input.candidate.url,
        input.trace_id,
        specimen_reference_label(&input.outcome, &specimen_link),
        input.outcome.confidence,
        input.progressive.class,
        input.progressive.ocr_requested,
        text_gate_summary(&input.progressive),
        if input.ocr_promoted_to_confirmed {
            ", OCR confirmed bad text and promoted this suspicious image to a confirmed match."
        } else {
            "."
        },
        perceptual_distance_label(&input.outcome),
        input.outcome.local_match_details(),
        input.elapsed,
        stage_timing_summary(&input.timings),
        if actions_taken.is_empty() {
            "none".to_owned()
        } else {
            actions_taken.join(", ")
        },
        input.outcome.tripped_gates_summary(),
        input.outcome.diagnostics_summary(),
        audit
    );
    let log_channel_id = input.state.bot_log_channel_id();
    let message_id = input
        .state
        .post_bot_log(
            detection_bot_log(&DetectionLogInput {
                candidate: &input.candidate,
                image_id: &input.image_id,
                outcome: &input.outcome,
                progressive: &input.progressive,
                timings: &input.timings,
                message_link: &message_link,
                specimen_link: &specimen_link,
                actions_taken: &actions_taken,
                elapsed: input.elapsed,
                safe_mode: input.safe_mode,
                ocr_promoted_to_confirmed: input.ocr_promoted_to_confirmed,
                trace_id: &input.trace_id,
            })
            .text_attachment(
                format!(
                    "sightline-raw-{}-{}.txt",
                    input.candidate.message_id.get(),
                    input.image_id
                ),
                details,
            ),
        )
        .await;
    let log_ref = log_channel_id
        .zip(message_id)
        .map(|(channel_id, message_id)| BotLogRef {
            channel_id,
            message_id,
        });
    if let Some(respond_to) = input.respond_to {
        let _ = respond_to.send(log_ref);
    }
}

struct DetectionLogInput<'a> {
    candidate: &'a ImageCandidate,
    image_id: &'a str,
    outcome: &'a MatchOutcome,
    progressive: &'a ProgressiveDecision,
    timings: &'a CandidateStageTimings,
    message_link: &'a str,
    specimen_link: &'a str,
    actions_taken: &'a [String],
    elapsed: u128,
    safe_mode: bool,
    ocr_promoted_to_confirmed: bool,
    trace_id: &'a str,
}

#[derive(Debug, Clone, Copy)]
enum DetectionLogKind {
    Confirmed,
    Suspicious,
    Benign,
}

impl DetectionLogKind {
    fn for_input(input: &DetectionLogInput<'_>) -> Self {
        if text_gate_cleared(input.progressive.text_gate.as_ref()) {
            Self::Benign
        } else if input.outcome.suspicious {
            Self::Suspicious
        } else {
            Self::Confirmed
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Confirmed => "Scam image match",
            Self::Suspicious => "Suspicious image",
            Self::Benign => "Benign image",
        }
    }

    fn color(self, safe_mode: bool) -> BotLogColor {
        match self {
            Self::Benign => BotLogColor::Success,
            Self::Suspicious => BotLogColor::Warning,
            Self::Confirmed if safe_mode => BotLogColor::Warning,
            Self::Confirmed => BotLogColor::Danger,
        }
    }

    fn apply_copy(self, event: BotLogEvent) -> BotLogEvent {
        match self {
            Self::Confirmed => event.confirmed_detection_copy(),
            Self::Suspicious => event.suspicious_detection_copy(),
            Self::Benign => event.benign_detection_copy(),
        }
    }
}

fn detection_bot_log(input: &DetectionLogInput<'_>) -> BotLogEvent {
    let kind = DetectionLogKind::for_input(input);
    let title = kind.title();
    let actions = if input.actions_taken.is_empty() {
        "none".to_owned()
    } else {
        input.actions_taken.join(", ")
    };
    let mut pipeline = vec![
        format!("visual class `{:?}`", input.progressive.class),
        format!("OCR requested `{}`", input.progressive.ocr_requested),
        format!(
            "OCR policy {}",
            ocr_policy_label(input.outcome, input.progressive)
        ),
    ];
    if input.ocr_promoted_to_confirmed {
        pipeline.push("OCR text gate promoted this image to confirmed".to_owned());
    } else if text_gate_cleared(input.progressive.text_gate.as_ref()) {
        pipeline.push("OCR text gate cleared this image".to_owned());
    } else if input.progressive.ocr_requested {
        pipeline.push("OCR text gate did not confirm bad text".to_owned());
    }
    if input.safe_mode {
        pipeline.push("safe mode skipped destructive actions".to_owned());
    }

    let mut event = kind
        .apply_copy(BotLogEvent::new(
            title,
            format!(
                "{title} for {} in <#{}>.",
                user_incident_label(input.candidate),
                input.candidate.channel_id.get()
            ),
        ))
        .color(kind.color(input.safe_mode))
        .image_url(input.candidate.url.clone())
        .field("Target user", user_incident_label(input.candidate), false)
        .field("Source message", input.message_link.to_owned(), false)
        .field(
            "Candidate image",
            format!("`{}`\n{}", input.image_id, input.candidate.url),
            false,
        )
        .field(
            "Matched specimen",
            specimen_reference_label(input.outcome, input.specimen_link),
            false,
        )
        .field(
            "Confidence",
            format!(
                "`{:?}` / `{}`\n{}{}",
                input.outcome.confidence,
                input.outcome.decision_name(),
                perceptual_distance_label(input.outcome),
                input.outcome.local_match_details()
            ),
            false,
        )
        .field("Pipeline", pipeline.join("\n"), false)
        .field("Trace ID", format!("`{}`", input.trace_id), false)
        .field(
            "Hot path timings",
            stage_timing_summary(input.timings),
            false,
        )
        .field(
            "Gates tripped",
            input.outcome.tripped_gates_summary(),
            false,
        );

    event = match input.progressive.text_gate.as_ref() {
        Some(text_gate) => event
            .field("OCR text gate", text_gate_match_summary(text_gate), false)
            .json_attachment(
                format!(
                    "sightline-ocr-{}-{}.json",
                    input.candidate.message_id.get(),
                    input.image_id
                ),
                &OcrEvidence::from_input(input, text_gate),
            ),
        None => event.field("OCR text gate", "`not_run`", false),
    };

    event.field("Actions taken", actions, false).field(
        "Processing time",
        format!("`{}` ms", input.elapsed),
        true,
    )
}

fn specimen_reference_label(outcome: &MatchOutcome, specimen_link: &str) -> String {
    if outcome.specimen_id == "none" {
        "not specimen-based".to_owned()
    } else {
        format!("`{}`\n{}", outcome.specimen_id, specimen_link)
    }
}

fn perceptual_distance_label(outcome: &MatchOutcome) -> String {
    let mut parts = Vec::new();
    if let Some(phash) = outcome.phash64_distance {
        parts.push(format!("pHash distance `{phash}`"));
    }
    if let Some(dhash) = outcome.dhash64_distance {
        parts.push(format!("dHash distance `{dhash}`"));
    }
    if parts.is_empty() {
        "perceptual distance `unavailable`".to_owned()
    } else {
        parts.join(", ")
    }
}

fn stage_timing_summary(timings: &CandidateStageTimings) -> String {
    format!(
        "total={}ms, queue_wait={}ms, preview_download={}ms(request={}ms, body={}ms, gate={}ms, bytes={}, cache={:?}, age={}s), preview_fingerprint={}ms, preview_matcher={}ms, preview_used={}, preview_fallback={}, download={}ms(request={}ms, body={}ms, gate={}ms, xxh128={}us, bytes={}, cache={:?}, age={}s), flagged_lookup={}us, exact_lookup={}us, singleflight_wait={}ms, fingerprint={}ms(decode={}ms, thumb={}ms, visual={}ms, orient={}ms, perceptual={}ms, normalize={}ms, tiles={}ms, text_grid={}ms, anchors={}ms, dense={}ms), matcher={}ms, ocr_crop={}ms, progressive={}ms",
        us_to_ms(timings.total_us),
        us_to_ms(timings.queue_wait_us),
        us_to_ms(timings.preview_download.total_us),
        us_to_ms(timings.preview_download.request_us),
        us_to_ms(timings.preview_download.body_us),
        us_to_ms(timings.preview_download.gate_wait_us),
        timings.preview_download.bytes,
        timings.preview_download.cdn_cache_status,
        cdn_age_label(timings.preview_download.cdn_age_seconds),
        us_to_ms(timings.preview_fingerprint_us),
        us_to_ms(timings.preview_matcher_us),
        timings.preview_used,
        timings.preview_fallback_reason.unwrap_or("none"),
        us_to_ms(timings.download.total_us),
        us_to_ms(timings.download.request_us),
        us_to_ms(timings.download.body_us),
        us_to_ms(timings.download.gate_wait_us),
        timings.download.xxh128_us,
        timings.download.bytes,
        timings.download.cdn_cache_status,
        cdn_age_label(timings.download.cdn_age_seconds),
        timings.flagged_cache_lookup_us,
        timings.exact_match_lookup_us,
        us_to_ms(timings.singleflight_wait_us),
        us_to_ms(timings.fingerprint_us),
        us_to_ms(timings.fingerprint_pipeline.decode_us),
        us_to_ms(timings.fingerprint_pipeline.whole_thumbnail_us),
        us_to_ms(timings.fingerprint_pipeline.visual_signature_us),
        us_to_ms(timings.fingerprint_pipeline.orientation_us),
        us_to_ms(timings.fingerprint_pipeline.perceptual_hashes_us),
        us_to_ms(timings.fingerprint_pipeline.normalize_luma_us),
        us_to_ms(timings.fingerprint_pipeline.base_tile_scorer_us),
        us_to_ms(timings.fingerprint_pipeline.text_grid_us),
        us_to_ms(timings.fingerprint_pipeline.local_anchors_us),
        us_to_ms(timings.fingerprint_pipeline.local_hashes_us),
        us_to_ms(timings.matcher_us),
        us_to_ms(timings.ocr_crop_us),
        us_to_ms(timings.progressive_eval_us)
    )
}

fn cdn_age_label(age_seconds: Option<u32>) -> String {
    age_seconds.map_or_else(|| "unknown".to_owned(), |age| age.to_string())
}

fn us_to_ms(value: u128) -> u128 {
    value / 1_000
}

fn ocr_eligible_for_suspicious_outcome(outcome: &MatchOutcome) -> bool {
    if !outcome.suspicious {
        return false;
    }
    match outcome.confidence {
        MatchConfidence::SuspiciousPerceptual
        | MatchConfidence::SuspiciousLocalAnchors
        | MatchConfidence::SuspiciousDenseLocalAnchors
        | MatchConfidence::ClusterCoherence => true,
        MatchConfidence::ExactXxh128
        | MatchConfidence::Perceptual
        | MatchConfidence::LocalAnchors
        | MatchConfidence::DenseLocalAnchors => false,
    }
}

fn ocr_policy_label(outcome: &MatchOutcome, progressive: &ProgressiveDecision) -> String {
    if progressive.ocr_requested {
        return match outcome.confidence {
            MatchConfidence::SuspiciousPerceptual
            | MatchConfidence::SuspiciousLocalAnchors
            | MatchConfidence::SuspiciousDenseLocalAnchors
            | MatchConfidence::ClusterCoherence => {
                "requested: specimen-based suspicious match".to_owned()
            }
            _ => "requested".to_owned(),
        };
    }
    "not requested".to_owned()
}

fn text_gate_confirms_bad(report: Option<&TextGateReport>) -> bool {
    report.is_some_and(|report| {
        matches!(
            report.decision,
            TextGateDecision::ConfirmedSentence | TextGateDecision::ConfirmedKeywords
        ) && matches!(report.verdict, TextGateVerdict::Bad)
    })
}

fn text_gate_cleared(report: Option<&TextGateReport>) -> bool {
    report.is_some_and(|report| matches!(report.verdict, TextGateVerdict::Good))
}

#[derive(Serialize)]
struct OcrEvidence<'a> {
    guild_id: u64,
    channel_id: u64,
    message_id: u64,
    user_id: u64,
    image_id: &'a str,
    image_url: &'a str,
    matched_specimen_id: &'a str,
    decision: &'a str,
    ocr_requested: bool,
    text_gate: &'a TextGateReport,
    trace_id: &'a str,
}

impl<'a> OcrEvidence<'a> {
    fn from_input(input: &'a DetectionLogInput<'a>, text_gate: &'a TextGateReport) -> Self {
        Self {
            guild_id: input.candidate.guild_id.get(),
            channel_id: input.candidate.channel_id.get(),
            message_id: input.candidate.message_id.get(),
            user_id: input.candidate.author_id.get(),
            image_id: input.image_id,
            image_url: &input.candidate.url,
            matched_specimen_id: &input.outcome.specimen_id,
            decision: input.outcome.decision_name(),
            ocr_requested: input.progressive.ocr_requested,
            text_gate,
            trace_id: input.trace_id,
        }
    }
}

fn image_trace_id(candidate: &ImageCandidate) -> String {
    format!(
        "{:016x}{:016x}",
        candidate.message_id.get(),
        xxh3_64(candidate.url.as_bytes())
    )
}

fn cached_decision_to_candidate(
    cached: CachedDecisionOutcome,
    image_id: String,
    byte_xxh128: String,
    policy_hash: u64,
    timings: &CandidateStageTimings,
) -> Result<CandidateDecision> {
    match cached {
        CachedDecisionOutcome::Pass => {
            info!(
                event = "hash_outcome_cache.hit",
                image_id = %image_id,
                decision = "pass",
                "reused cached pass image outcome"
            );
            Ok(CandidateDecision::pass(
                image_id,
                byte_xxh128,
                policy_hash,
                timings,
            ))
        }
        CachedDecisionOutcome::Failure(reason) => {
            info!(
                event = "hash_outcome_cache.hit",
                image_id = %image_id,
                decision = "scan_failed",
                reason = %reason,
                "reused cached failed image outcome"
            );
            Err(anyhow!("cached scan failure: {reason}"))
        }
        CachedDecisionOutcome::Match(outcome) => {
            let outcome = outcome.into_match_outcome();
            info!(
                event = "hash_outcome_cache.hit",
                image_id = %image_id,
                specimen_id = %outcome.specimen_id,
                suspicious = outcome.suspicious,
                "reused cached match image outcome"
            );
            Ok(CandidateDecision::matched(
                image_id,
                byte_xxh128,
                policy_hash,
                outcome,
                timings,
            ))
        }
    }
}

async fn process_candidate(
    state: &AppState,
    candidate: &ImageCandidate,
    queue_wait_us: u128,
) -> Result<CandidateDecision> {
    let started = Instant::now();
    if candidate.guild_id != state.guild_id() {
        return Err(anyhow!(
            "candidate guild {} does not match configured guild {}",
            candidate.guild_id.get(),
            state.guild_id().get()
        ));
    }

    let guild_config = state.active_config_arc();
    let policy_hash = state.detection_policy_hash();
    let download_config =
        effective_download_config(&state.config.download, &guild_config.scan_policy);
    let match_config = guild_config
        .detection_hyperparameters
        .effective_match_config(&state.config.matching);
    let auto_add_possible = guild_config
        .detection_policy
        .confirmed
        .actions
        .add_to_specimens
        || guild_config
            .detection_policy
            .suspicious
            .actions
            .add_to_specimens;
    let snapshot = ScanConfigSnapshot {
        guild_config,
        policy_hash,
        download_config,
        match_config,
        auto_add_possible,
    };
    let mut timings = CandidateStageTimings {
        queue_wait_us,
        ..CandidateStageTimings::default()
    };
    if let Some(reason) = metadata_safety_rejection_reason(candidate, &snapshot.download_config) {
        timings.preview_fallback_reason.get_or_insert(reason);
        timings.total_us = started.elapsed().as_micros();
        return Ok(CandidateDecision::pass(
            reason,
            String::new(),
            policy_hash,
            &timings,
        ));
    }
    let preview =
        match choose_preview_route(candidate, &snapshot.download_config, &snapshot.match_config) {
            PreviewRoute::OriginalOnly(reason) => {
                timings.preview_fallback_reason = Some(reason);
                return process_original_candidate(state, candidate, started, &timings, &snapshot)
                    .await;
            }
            PreviewRoute::Precheck(preview) => preview,
        };

    process_candidate_with_preview_precheck(state, candidate, started, &timings, &snapshot, preview)
        .await
}

async fn process_candidate_with_preview_precheck(
    state: &AppState,
    candidate: &ImageCandidate,
    started: Instant,
    timings: &CandidateStageTimings,
    snapshot: &ScanConfigSnapshot,
    preview: DiscordPreviewRequest,
) -> Result<CandidateDecision> {
    let preview_scan = scan_preview_candidate(
        state,
        candidate,
        preview,
        PreviewScanContext {
            download_config: &snapshot.download_config,
            match_config: &snapshot.match_config,
            detection_policy: &snapshot.guild_config.detection_policy,
            policy_hash: snapshot.policy_hash,
        },
        started,
    );
    let original_scan = process_original_candidate(state, candidate, started, timings, snapshot);
    tokio::pin!(preview_scan);
    tokio::pin!(original_scan);

    tokio::select! {
        biased;
        original = &mut original_scan => {
            match original {
                Ok(mut decision) => {
                    decision.timings.preview_fallback_reason.get_or_insert("original_won_race");
                    info!(
                        event = "preview.original_race_won",
                        guild_id = candidate.guild_id.get(),
                        channel_id = candidate.channel_id.get(),
                        message_id = candidate.message_id.get(),
                        candidate_index = candidate.candidate_index,
                        "original image scan completed before preview scan"
                    );
                    Ok(decision)
                }
                Err(original_source) => match preview_scan.await {
                    Ok(PreviewScanResult::Decisive(decision)) => Ok(*decision),
                    Ok(PreviewScanResult::Fallback(fallback)) => {
                        warn!(
                            event = "preview.original_failed_preview_fallback",
                            guild_id = candidate.guild_id.get(),
                            channel_id = candidate.channel_id.get(),
                            message_id = candidate.message_id.get(),
                            reason = fallback.reason,
                            ?original_source,
                            "original image scan failed and preview was not decisive"
                        );
                        Err(original_source)
                    }
                    Err(preview_source) => {
                        warn_preview_failure(candidate, &preview_source);
                        Err(original_source)
                    }
                },
            }
        }
        preview = &mut preview_scan => {
            match preview {
                Ok(PreviewScanResult::Decisive(decision)) => Ok(*decision),
                Ok(PreviewScanResult::Fallback(fallback)) => {
                    let mut original = original_scan.await?;
                    merge_preview_timings(
                        &mut original.timings,
                        &fallback.timings,
                        fallback.reason,
                        false,
                    );
                    log_preview_original_agreement(
                        candidate,
                        fallback.outcome.as_ref(),
                        original.decision.outcome.as_ref(),
                        fallback.reason,
                    );
                    Ok(original)
                }
                Err(source) => {
                    warn_preview_failure(candidate, &source);
                    let mut original = original_scan.await?;
                    original.timings.preview_fallback_reason = Some("preview_failed");
                    Ok(original)
                }
            }
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "linear image-processing pipeline with cache, singleflight, decode, match, OCR, and auto-add decisions"
)]
async fn process_original_candidate(
    state: &AppState,
    candidate: &ImageCandidate,
    started: Instant,
    timings: &CandidateStageTimings,
    snapshot: &ScanConfigSnapshot,
) -> Result<CandidateDecision> {
    let mut timings = *timings;
    let guild_config = &snapshot.guild_config;
    let policy_hash = snapshot.policy_hash;
    let download_config = &snapshot.download_config;
    let match_config = &snapshot.match_config;
    let auto_add_possible = snapshot.auto_add_possible;

    let downloaded = state
        .download_image(
            &candidate.url,
            candidate.mime_hint.as_deref(),
            download_config,
        )
        .await?;
    timings.download = downloaded.timings;

    let image_id = downloaded.byte_xxh128.chars().take(16).collect::<String>();
    let byte_xxh128 = downloaded.byte_xxh128.clone();

    let exact_started = Instant::now();
    if let Some(outcome) =
        state.find_exact_xxh128_for_policy(&downloaded.byte_xxh128, &guild_config.detection_policy)
    {
        timings.exact_match_lookup_us = exact_started.elapsed().as_micros();
        timings.total_us = started.elapsed().as_micros();
        state
            .hash_outcome_cache
            .lock()
            .insert_match(byte_xxh128.clone(), policy_hash, &outcome);
        info!(
            event = "exact_hash_cache.hit",
            image_id = %image_id,
            specimen_id = %outcome.specimen_id,
            suspicious = outcome.suspicious,
            "matched exact specimen before image decode"
        );
        return Ok(CandidateDecision::matched(
            image_id,
            byte_xxh128,
            policy_hash,
            outcome,
            &timings,
        ));
    }
    timings.exact_match_lookup_us = exact_started.elapsed().as_micros();

    let lookup_started = Instant::now();
    let cached_outcome = {
        let mut cache = state.hash_outcome_cache.lock();
        cache.get(&downloaded.byte_xxh128, policy_hash)
    };
    if let Some(cached) = cached_outcome {
        timings.flagged_cache_lookup_us = lookup_started.elapsed().as_micros();
        timings.total_us = started.elapsed().as_micros();
        return cached_decision_to_candidate(cached, image_id, byte_xxh128, policy_hash, &timings);
    }
    timings.flagged_cache_lookup_us = lookup_started.elapsed().as_micros();

    let mut processing = loop {
        match state.start_hash_processing(byte_xxh128.clone(), policy_hash) {
            HashProcessingClaim::Owner { key, complete } => {
                break HashProcessingGuard::new(state.clone(), key, complete);
            }
            HashProcessingClaim::Waiter(mut complete) => {
                let wait_started = Instant::now();
                wait_for_hash_processing_completion(&mut complete).await;
                timings.singleflight_wait_us = timings
                    .singleflight_wait_us
                    .saturating_add(wait_started.elapsed().as_micros());
                if let Some(cached) = state
                    .hash_outcome_cache
                    .lock()
                    .get(&byte_xxh128, policy_hash)
                {
                    if let Some(outcome) = state
                        .find_exact_xxh128_for_policy(&byte_xxh128, &guild_config.detection_policy)
                    {
                        state.hash_outcome_cache.lock().insert_match(
                            byte_xxh128.clone(),
                            policy_hash,
                            &outcome,
                        );
                        timings.total_us = started.elapsed().as_micros();
                        return Ok(CandidateDecision::matched(
                            image_id,
                            byte_xxh128,
                            policy_hash,
                            outcome,
                            &timings,
                        ));
                    }
                    timings.total_us = started.elapsed().as_micros();
                    return cached_decision_to_candidate(
                        cached,
                        image_id,
                        byte_xxh128,
                        policy_hash,
                        &timings,
                    );
                }
            }
        }
    };

    if let Some(reason) = metadata_aspect_rejection_reason(candidate, match_config) {
        timings.preview_fallback_reason.get_or_insert(reason);
        timings.total_us = started.elapsed().as_micros();
        state
            .hash_outcome_cache
            .lock()
            .insert_pass(byte_xxh128.clone(), policy_hash);
        processing.finish();
        return Ok(CandidateDecision::pass(
            reason,
            byte_xxh128,
            policy_hash,
            &timings,
        ));
    }

    let retained_original = if guild_config.text_gate_policy.enabled || auto_add_possible {
        retain_image_bytes(state, candidate, &byte_xxh128, downloaded.bytes.clone())
    } else {
        None
    };
    let hash_mode = if detection_policy_needs_local_hashes(&guild_config.detection_policy) {
        HashMode::candidate()
    } else {
        HashMode::candidate_without_local_hashes()
    };
    let fingerprint_started = Instant::now();
    let staged = match hash_downloaded_image_tier1(
        downloaded,
        download_config.max_decoded_pixels,
        match_config,
        &state.decode_gate,
    )
    .await
    {
        Ok(staged) => {
            timings.fingerprint_pipeline = staged.timings;
            staged
        }
        Err(source) => {
            state.hash_outcome_cache.lock().insert_failure(
                byte_xxh128.clone(),
                policy_hash,
                source.to_string(),
            );
            return Err(source);
        }
    };

    let tier1_match_started = Instant::now();
    let tier1_policy = confirmed_tier1_policy(&guild_config.detection_policy);
    let tier1_match = state
        .find_match_for_policy(Arc::new(staged.fingerprint.clone()), tier1_policy)
        .await?;
    timings.matcher_us = timings
        .matcher_us
        .saturating_add(tier1_match_started.elapsed().as_micros());
    if let Some(outcome) = tier1_match {
        let specimen_candidate = if auto_add_possible {
            retained_original.clone().map(|original| SpecimenAutoAdd {
                fingerprint: staged.fingerprint.clone(),
                original,
                preview: None,
                full_diagnostics: false,
            })
        } else {
            None
        };
        timings.fingerprint_us = fingerprint_started.elapsed().as_micros();
        timings.total_us = started.elapsed().as_micros();
        state
            .hash_outcome_cache
            .lock()
            .insert_match(byte_xxh128.clone(), policy_hash, &outcome);
        processing.finish();
        let mut decision =
            CandidateDecision::matched(image_id, byte_xxh128, policy_hash, outcome, &timings);
        decision.specimen_candidate = specimen_candidate;
        return Ok(decision);
    }

    let fingerprint = match complete_staged_image_fingerprint(
        staged,
        match_config,
        &state.decode_gate,
        hash_mode,
    )
    .await
    {
        Ok((fingerprint, pipeline_timings)) => {
            timings.fingerprint_pipeline = pipeline_timings;
            fingerprint
        }
        Err(source) => {
            state.hash_outcome_cache.lock().insert_failure(
                byte_xxh128.clone(),
                policy_hash,
                source.to_string(),
            );
            return Err(source);
        }
    };
    timings.fingerprint_us = fingerprint_started.elapsed().as_micros();

    let fingerprint = Arc::new(fingerprint);
    let matcher_started = Instant::now();
    let match_result = match state
        .find_match_for_policy(
            Arc::clone(&fingerprint),
            guild_config.detection_policy.clone(),
        )
        .await
    {
        Ok(match_result) => match_result,
        Err(source) => {
            return Err(source);
        }
    };
    let visual = VisualClassification::from_outcome(match_result);
    timings.matcher_us = timings
        .matcher_us
        .saturating_add(matcher_started.elapsed().as_micros());
    let noop_artifacts = NoopArtifactSink;
    let missing_ocr = UnavailableOcrClient::new("OCR_SPACE_API_KEY is not configured");
    let ocr: &dyn crate::image::engine::OcrClient = state.ocr_space.as_deref().map_or(
        &missing_ocr as &dyn crate::image::engine::OcrClient,
        |client| client as &dyn crate::image::engine::OcrClient,
    );
    let engine = ProgressiveEngine {
        matcher: None,
        detection_policy: &guild_config.detection_policy,
        text_gate_policy: &guild_config.text_gate_policy,
        ocr,
        artifacts: &noop_artifacts,
    };
    let should_prepare_ocr = guild_config.text_gate_policy.enabled
        && matches!(
            &visual,
            VisualClassification::KnownSuspicious(outcome)
                if ocr_eligible_for_suspicious_outcome(outcome)
        );
    if should_prepare_ocr {
        let VisualClassification::KnownSuspicious(outcome) = visual else {
            unreachable!("OCR is only prepared for suspicious visual matches");
        };
        if retained_original.is_none() {
            warn!(
                event = "ocr.followup_redownload",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_id = %image_id,
                "OCR requested but original image bytes could not be retained; follow-up will redownload"
            );
        }
        let progressive = ProgressiveDecision {
            class: VisualCandidateClass::KnownSuspicious,
            outcome: Some(outcome.clone()),
            text_gate: Some(TextGateReport::pending(
                guild_config.text_gate_policy.keyword_threshold,
            )),
            ocr_requested: true,
        };
        let preview = if auto_add_possible {
            generate_specimen_preview_variant(state, candidate, download_config, match_config).await
        } else {
            None
        };
        let specimen_candidate = if auto_add_possible {
            retained_original.clone().map(|original| SpecimenAutoAdd {
                fingerprint: fingerprint.as_ref().clone(),
                original,
                preview: preview.clone(),
                full_diagnostics: false,
            })
        } else {
            None
        };
        timings.total_us = started.elapsed().as_micros();
        return Ok(CandidateDecision {
            image_id,
            byte_xxh128,
            policy_hash,
            decision: progressive,
            specimen_candidate,
            ocr_followup: Some(OcrFollowup {
                original: retained_original.clone(),
                fingerprint: fingerprint.as_ref().clone(),
                preview,
                started,
            }),
            processing_guard: Some(processing),
            timings,
        });
    }
    let progressive_started = Instant::now();
    let artifact_id = format!("{}_{}", candidate.message_id.get(), image_id);
    let progressive = match visual {
        VisualClassification::KnownSuspicious(outcome) if guild_config.text_gate_policy.enabled => {
            ProgressiveDecision {
                class: VisualCandidateClass::KnownSuspicious,
                outcome: Some(outcome),
                text_gate: None,
                ocr_requested: false,
            }
        }
        visual => match engine
            .evaluate_with_visual(visual, None, &artifact_id)
            .await
        {
            Ok(progressive) => progressive,
            Err(source) => {
                return Err(source);
            }
        },
    };
    timings.progressive_eval_us = progressive_started.elapsed().as_micros();
    let preview = if progressive.outcome.is_some() && auto_add_possible {
        generate_specimen_preview_variant(state, candidate, download_config, match_config).await
    } else {
        None
    };
    let specimen_candidate = if progressive.outcome.is_some() && auto_add_possible {
        retained_original.clone().map(|original| SpecimenAutoAdd {
            fingerprint: fingerprint.as_ref().clone(),
            original,
            preview,
            full_diagnostics: false,
        })
    } else {
        None
    };
    timings.total_us = started.elapsed().as_micros();
    if progressive.outcome.is_none() {
        state
            .hash_outcome_cache
            .lock()
            .insert_pass(byte_xxh128.clone(), policy_hash);
    } else if let Some(outcome) = progressive.outcome.as_ref() {
        state
            .hash_outcome_cache
            .lock()
            .insert_match(byte_xxh128.clone(), policy_hash, outcome);
    }
    processing.finish();

    Ok(CandidateDecision {
        image_id,
        byte_xxh128,
        policy_hash,
        decision: progressive,
        specimen_candidate,
        ocr_followup: None,
        processing_guard: None,
        timings,
    })
}

fn metadata_safety_rejection_reason(
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
) -> Option<&'static str> {
    if candidate
        .size_bytes
        .is_some_and(|bytes| bytes > u64::try_from(download_config.max_bytes).unwrap_or(u64::MAX))
    {
        return Some("metadata_file_too_large");
    }
    let (Some(width), Some(height)) = (candidate.metadata_width, candidate.metadata_height) else {
        return None;
    };
    if width == 0 || height == 0 {
        return Some("metadata_invalid_dimensions");
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > download_config.max_decoded_pixels {
        return Some("metadata_pixels_too_large");
    }
    None
}

fn metadata_aspect_rejection_reason(
    candidate: &ImageCandidate,
    match_config: &MatchConfig,
) -> Option<&'static str> {
    let (Some(width), Some(height)) = (candidate.metadata_width, candidate.metadata_height) else {
        return None;
    };
    if width == 0 || height == 0 {
        return None;
    }
    let aspect = f64::from(width.max(height)) / f64::from(width.min(height));
    (aspect > f64::from(match_config.local_max_aspect_ratio)).then_some("metadata_aspect_too_large")
}

pub(crate) fn detection_policy_needs_local_hashes(
    policy: &crate::configuration::guild::DetectionPolicy,
) -> bool {
    policy.confirmed.threshold.local_anchors
        || policy.confirmed.threshold.visual_shape
        || policy.suspicious.threshold.local_anchors
        || policy.suspicious.threshold.visual_shape
}

async fn add_matched_candidate_to_specimens_after_action(
    state: &AppState,
    candidate: &ImageCandidate,
    specimen_candidate: Option<SpecimenAutoAdd>,
    outcome: &MatchOutcome,
    policy_hash: u64,
    image_processing_ms: Option<u128>,
) -> Option<String> {
    let Some(specimen_candidate) = specimen_candidate else {
        return Some("already_known".to_owned());
    };
    let byte_xxh128 = specimen_candidate.fingerprint.byte_xxh128.clone();
    match add_matched_candidate_to_specimens(
        state,
        candidate,
        specimen_candidate,
        image_processing_ms,
    )
    .await
    {
        Ok(Some(specimen_id)) => {
            let mut cache = state.hash_outcome_cache.lock();
            cache.insert_match(byte_xxh128, policy_hash, outcome);
            Some(specimen_id)
        }
        Ok(None) => Some("skipped_duplicate".to_owned()),
        Err(source) => {
            warn!(
                event = "specimen.auto_add_failed",
                guild_id = candidate.guild_id.get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_id = %byte_xxh128,
                ?source,
                "failed to add matched image as specimen"
            );
            Some("failed".to_owned())
        }
    }
}

async fn add_matched_candidate_to_specimens(
    state: &AppState,
    candidate: &ImageCandidate,
    specimen_candidate: SpecimenAutoAdd,
    image_processing_ms: Option<u128>,
) -> Result<Option<String>> {
    if state.safe_mode.load(Ordering::Acquire) {
        return Ok(None);
    }
    let SpecimenAutoAdd {
        fingerprint,
        original,
        preview,
        full_diagnostics,
    } = specimen_candidate;
    let source_byte_xxh128 = fingerprint.byte_xxh128.clone();
    let source_mime = fingerprint.mime.clone();
    if state.contains_specimen_xxh128(&source_byte_xxh128) {
        return Ok(None);
    }
    let bytes = original.bytes();
    if bytes.is_empty() {
        return Err(anyhow!("auto-add specimen bytes were not retained"));
    }
    let specimen_fingerprint = if full_diagnostics {
        fingerprint
    } else {
        let config = state.active_config_arc();
        let download_config =
            effective_download_config(&state.config.download, &config.scan_policy);
        let match_config = config
            .detection_hyperparameters
            .effective_match_config(&state.config.matching);
        hash_downloaded_image(
            crate::image::pipeline::DownloadedImage {
                bytes: bytes.clone(),
                byte_xxh128: source_byte_xxh128,
                mime: source_mime,
                timings: DownloadTimings::default(),
            },
            download_config.max_decoded_pixels,
            &match_config,
            &state.decode_gate,
            HashMode::FullDiagnostics,
        )
        .await
        .context("hashing auto-added specimen image")?
    };
    let preview_fingerprint = preview.as_ref().map(|variant| variant.fingerprint.clone());

    let mut record = SpecimenRecord::new_add(
        candidate.guild_id,
        candidate.channel_id,
        candidate.message_id,
        candidate.author_id,
        state.bot_user_id,
        specimen_fingerprint,
        preview_fingerprint,
    );
    let original_attachment = SpecimenImageAttachment::original(&record, bytes)?;
    let mut image_attachments = vec![original_attachment];
    if let Some(preview) = preview {
        match SpecimenImageAttachment::discord_preview(&record, preview.bytes) {
            Ok(attachment) => image_attachments.push(attachment),
            Err(source) => {
                record.preview = None;
                warn!(
                    event = "specimen.auto_add_preview_attachment_failed",
                    guild_id = candidate.guild_id.get(),
                    channel_id = candidate.channel_id.get(),
                    message_id = candidate.message_id.get(),
                    ?source,
                    "failed to prepare auto-added specimen preview attachment"
                );
            }
        }
    }
    let record = record.sign(&state.secrets.specimen_hmac_secret)?;
    let specimen_id = record.specimen_id.clone();

    let write = match state
        .write_specimen_record(
            record,
            image_attachments,
            auto_add_specimen_log_context(candidate, image_processing_ms),
        )
        .await?
    {
        SpecimenWriteOutcome::Added(write) => write,
        SpecimenWriteOutcome::Duplicate => return Ok(None),
    };

    info!(
        event = "specimen.auto_added",
        guild_id = candidate.guild_id.get(),
        channel_id = candidate.channel_id.get(),
        message_id = candidate.message_id.get(),
        specimen_id = %write.specimen_id,
        ledger_message_id = write.ledger_message_id.get(),
        "added matched image as specimen"
    );
    Ok(Some(specimen_id))
}

fn auto_add_specimen_log_context(
    candidate: &ImageCandidate,
    image_processing_ms: Option<u128>,
) -> SpecimenWriteLogContext {
    SpecimenWriteLogContext {
        image_url: Some(candidate.url.clone()),
        image_processing_ms,
        pre_add_match: Some("Auto-add was triggered by the current detection outcome.".to_owned()),
    }
}

fn text_gate_summary(progressive: &ProgressiveDecision) -> String {
    let Some(report) = progressive.text_gate.as_ref() else {
        return "`not_run`".to_owned();
    };
    let mut summary = format!(
        "`{:?}` verdict `{:?}` confidence `{:.2}` keywords `{}/{}`",
        report.decision,
        report.verdict,
        report.confidence,
        report.keyword_hits,
        report.keyword_threshold
    );
    if matches!(report.decision, TextGateDecision::ConfirmedSentence) {
        summary.push_str(" sentence `true`");
    }
    if !report.matched_keywords.is_empty() {
        let _ = write!(
            summary,
            " matched_keywords `{}`",
            report.matched_keywords.join(", ").replace('`', "'")
        );
    }
    if !report.matched_sentences.is_empty() {
        let _ = write!(
            summary,
            " matched_sentences `{}`",
            report.matched_sentences.join(" | ").replace('`', "'")
        );
    }
    if let Some(error) = &report.error {
        let _ = write!(summary, " error `{}`", error.replace('`', "'"));
    }
    summary
}

fn text_gate_match_summary(report: &TextGateReport) -> String {
    let keywords = if report.matched_keywords.is_empty() {
        "none".to_owned()
    } else {
        report
            .matched_keywords
            .iter()
            .map(|value| format!("`{}`", sanitize_log_value(value)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let sentences = if report.matched_sentences.is_empty() {
        "none".to_owned()
    } else {
        report
            .matched_sentences
            .iter()
            .map(|value| format!("`{}`", sanitize_log_value(value)))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "decision `{:?}`, verdict `{:?}`, confidence `{:.2}`\nkeywords: {}\nsentences: {}",
        report.decision, report.verdict, report.confidence, keywords, sentences
    )
}

fn sanitize_log_value(value: &str) -> String {
    const MAX_CHARS: usize = 900;
    let value = value.replace('`', "'").replace('\r', " ");
    if value.chars().count() <= MAX_CHARS {
        value
    } else {
        let mut truncated = value.chars().take(MAX_CHARS - 3).collect::<String>();
        truncated.push_str("...");
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::worker_preview::{
        choose_preview_route, finish_preview_scan, preview_outcome_allows_early_exit,
    };
    use crate::image::types::{
        CandidateKind, FingerprintRepresentation, MatchDiagnostics, MatchStepDiagnostic,
    };
    use twilight_model::id::Id;

    fn preview_candidate() -> ImageCandidate {
        ImageCandidate {
            guild_id: Id::new(1),
            channel_id: Id::new(2),
            message_id: Id::new(3),
            candidate_index: 1,
            candidates_in_message: 1,
            author_id: Id::new(4),
            author_username: None,
            author_global_name: None,
            url: "https://cdn.discordapp.com/attachments/1/2/image.png".to_owned(),
            proxy_url: Some("https://media.discordapp.net/attachments/1/2/image.png".to_owned()),
            kind: CandidateKind::Attachment,
            mime_hint: Some("image/png".to_owned()),
            size_bytes: Some(2 * 1024 * 1024),
            metadata_width: Some(1600),
            metadata_height: Some(2400),
            media_flags: None,
            verify_only: false,
            enqueued_at: None,
        }
    }

    fn outcome(confidence: MatchConfidence, suspicious: bool) -> MatchOutcome {
        MatchOutcome {
            specimen_id: "spm_test".to_owned(),
            confidence,
            suspicious,
            match_score: None,
            phash64_distance: None,
            dhash64_distance: None,
            local_anchor_hits: None,
            local_distinct_regions: None,
            local_average_distance: None,
            local_geometry_model: None,
            diagnostics: MatchDiagnostics {
                representation: FingerprintRepresentation::DiscordPreview,
                candidate_short_edge: 100,
                candidate_area: 10_000,
                candidate_aspect: 1.0,
                candidate_luma_mean: 128,
                candidate_luma_std: 0,
                candidate_text_grid_mean: 0,
                candidate_text_regions: 0,
                candidate_local_hashes: 0,
                steps: vec![MatchStepDiagnostic {
                    threshold: "confirmed",
                    step: "test",
                    passed: true,
                    reason: None,
                    specimen_id: Some("spm_test".to_owned()),
                    candidates_considered: None,
                    phash64_distance: None,
                    dhash64_distance: None,
                    geometry_compatible: None,
                    visual_compatible: None,
                    local_anchor_hits: None,
                    local_distinct_regions: None,
                    local_average_distance: None,
                    local_layout_spread: None,
                    local_mean_residual: None,
                    local_scale: None,
                    local_angle: None,
                    local_geometry_model: None,
                    visual_shape_signals: None,
                    visual_shape_score: None,
                    match_score: None,
                }],
            },
        }
    }

    #[test]
    fn preview_early_exit_only_allows_non_suspicious_hard_matches() {
        assert!(preview_outcome_allows_early_exit(&outcome(
            MatchConfidence::ExactXxh128,
            false,
        )));
        assert!(preview_outcome_allows_early_exit(&outcome(
            MatchConfidence::Perceptual,
            false,
        )));
        assert!(preview_outcome_allows_early_exit(&outcome(
            MatchConfidence::LocalAnchors,
            false,
        )));
        assert!(!preview_outcome_allows_early_exit(&outcome(
            MatchConfidence::Perceptual,
            true,
        )));
        assert!(!preview_outcome_allows_early_exit(&outcome(
            MatchConfidence::SuspiciousPerceptual,
            true,
        )));
        assert!(!preview_outcome_allows_early_exit(&outcome(
            MatchConfidence::SuspiciousLocalAnchors,
            true,
        )));
        assert!(!preview_outcome_allows_early_exit(&outcome(
            MatchConfidence::ClusterCoherence,
            true,
        )));
    }

    #[test]
    fn suspicious_cluster_coherence_is_ocr_eligible() {
        assert!(ocr_eligible_for_suspicious_outcome(&outcome(
            MatchConfidence::ClusterCoherence,
            true,
        )));
        assert!(!ocr_eligible_for_suspicious_outcome(&outcome(
            MatchConfidence::ClusterCoherence,
            false,
        )));
    }

    #[test]
    fn preview_scan_result_decisive_for_non_suspicious_hard_match() {
        let result = finish_preview_scan(
            Some(outcome(MatchConfidence::Perceptual, false)),
            "preview_id".to_owned(),
            "a".repeat(32),
            1,
            CandidateStageTimings::default(),
            Instant::now(),
        );

        let PreviewScanResult::Decisive(decision) = result else {
            panic!("expected decisive preview result");
        };
        assert_eq!(decision.image_id, "preview_id");
        assert!(decision.timings.preview_used);
        assert_eq!(decision.timings.preview_fallback_reason, None);
        assert!(matches!(
            decision.decision.class,
            VisualCandidateClass::KnownStrong
        ));
    }

    #[test]
    fn preview_scan_result_falls_back_on_miss() {
        let result = finish_preview_scan(
            None,
            "preview_id".to_owned(),
            "a".repeat(32),
            1,
            CandidateStageTimings::default(),
            Instant::now(),
        );

        let PreviewScanResult::Fallback(fallback) = result else {
            panic!("expected preview fallback");
        };
        assert_eq!(fallback.reason, "preview_miss");
        assert!(fallback.outcome.is_none());
        assert_eq!(
            fallback.timings.preview_fallback_reason,
            Some("preview_miss")
        );
    }

    #[test]
    fn preview_scan_result_falls_back_on_suspicious_match() {
        let result = finish_preview_scan(
            Some(outcome(MatchConfidence::SuspiciousPerceptual, true)),
            "preview_id".to_owned(),
            "a".repeat(32),
            1,
            CandidateStageTimings::default(),
            Instant::now(),
        );

        let PreviewScanResult::Fallback(fallback) = result else {
            panic!("expected preview fallback");
        };
        assert_eq!(fallback.reason, "preview_not_decisive");
        assert!(fallback.outcome.is_some_and(|outcome| outcome.suspicious));
        assert_eq!(
            fallback.timings.preview_fallback_reason,
            Some("preview_not_decisive")
        );
    }

    #[test]
    fn preview_route_prechecks_when_preview_is_eligible() {
        let route = choose_preview_route(
            &preview_candidate(),
            &DownloadConfig::default(),
            &MatchConfig::default(),
        );

        let PreviewRoute::Precheck(request) = route else {
            panic!("expected preview precheck route");
        };
        assert_eq!(request.width, 724);
        assert_eq!(request.height, 1086);
    }

    #[test]
    fn preview_route_uses_original_when_preview_is_ineligible() {
        let mut candidate = preview_candidate();
        candidate.proxy_url = None;
        let route = choose_preview_route(
            &candidate,
            &DownloadConfig::default(),
            &MatchConfig::default(),
        );

        let PreviewRoute::OriginalOnly(reason) = route else {
            panic!("expected original-only route");
        };
        assert_eq!(reason, "preview_ineligible");
    }

    #[tokio::test]
    async fn hash_processing_wait_observes_completion_sent_before_await() {
        let (sender, mut receiver) = tokio::sync::watch::channel(false);
        sender.send(true).unwrap();

        wait_for_hash_processing_completion(&mut receiver).await;

        assert!(*receiver.borrow());
    }

    #[tokio::test]
    async fn hash_processing_wait_returns_when_sender_is_dropped() {
        let (sender, mut receiver) = tokio::sync::watch::channel(false);
        drop(sender);

        wait_for_hash_processing_completion(&mut receiver).await;

        assert!(!*receiver.borrow());
    }
}
