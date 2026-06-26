use crate::{
    bot::{
        ledger::{SpecimenImageAttachment, SpecimenRecord},
        runtime::{AppState, SpecimenWriteLogContext, SpecimenWriteOutcome},
    },
    configuration::{
        app::{DownloadConfig, MatchConfig, effective_download_config},
        guild::GuildConfig,
    },
    image::{
        pipeline::{HashMode, build_discord_preview_request, hash_downloaded_image, url_log_label},
        types::{ImageCandidate, ImageFingerprint},
    },
};
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinSet;
use tracing::{info, warn};
use twilight_model::id::{Id, marker::UserMarker};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ImageImportSummary {
    pub(crate) added: usize,
    pub(crate) exact_duplicates: usize,
    pub(crate) failed: usize,
}

#[derive(Clone)]
struct SpecimenPreviewVariant {
    fingerprint: ImageFingerprint,
    bytes: bytes::Bytes,
}

struct PreparedSpecimenWrite {
    candidate: ImageCandidate,
    record: SpecimenRecord,
    image_attachments: Vec<SpecimenImageAttachment>,
    image_processing_ms: u128,
    pre_add_match: Option<String>,
}

struct ProcessedSpecimenImage {
    fingerprint: ImageFingerprint,
    bytes: bytes::Bytes,
    preview: Option<SpecimenPreviewVariant>,
    preview_fingerprint: Option<ImageFingerprint>,
}

enum SpecimenImportTaskOutcome {
    Prepared(Box<PreparedSpecimenWrite>),
    ExactDuplicate,
    Failed,
}

pub(crate) async fn import_image_candidates(
    state: &AppState,
    candidates: Vec<ImageCandidate>,
    added_by_id: Id<UserMarker>,
    guild_config: &GuildConfig,
    source: &'static str,
) -> ImageImportSummary {
    let download_config =
        effective_download_config(&state.config.download, &guild_config.scan_policy);
    let match_config = guild_config
        .detection_hyperparameters
        .effective_match_config(&state.config.matching);
    let mut tasks = JoinSet::new();

    for candidate in candidates {
        let state = state.clone();
        let download_config = download_config.clone();
        let match_config = match_config.clone();
        tasks.spawn(async move {
            process_image_candidate(
                state,
                candidate,
                added_by_id,
                download_config,
                match_config,
                source,
            )
            .await
        });
    }

    let mut summary = ImageImportSummary::default();
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(SpecimenImportTaskOutcome::Prepared(write)) => {
                // Image work runs concurrently, but persistence stays behind the per-channel
                // database writer so Discord storage and matcher updates remain serialized.
                write_prepared_specimen(state, *write, source, &mut summary).await;
            }
            Ok(SpecimenImportTaskOutcome::ExactDuplicate) => {
                summary.exact_duplicates += 1;
            }
            Ok(SpecimenImportTaskOutcome::Failed) => {
                summary.failed += 1;
            }
            Err(error) => {
                summary.failed += 1;
                warn!(
                    event = "specimen.image_ingest_task_failed",
                    source,
                    guild_id = state.guild_id().get(),
                    ?error,
                    "specimen image import task failed"
                );
            }
        }
    }

    info!(
        event = "specimen.images_ingested",
        source,
        added = summary.added,
        exact_duplicates = summary.exact_duplicates,
        failed = summary.failed,
        moderator_id = added_by_id.get(),
        "image specimen import completed"
    );

    summary
}

async fn process_image_candidate(
    state: AppState,
    candidate: ImageCandidate,
    added_by_id: Id<UserMarker>,
    download_config: DownloadConfig,
    match_config: MatchConfig,
    source: &'static str,
) -> SpecimenImportTaskOutcome {
    let started = Instant::now();
    let Some((fingerprint, bytes)) =
        download_and_hash_candidate(&state, &candidate, &download_config, &match_config, source)
            .await
    else {
        return SpecimenImportTaskOutcome::Failed;
    };

    if state.contains_specimen_xxh128(&fingerprint.byte_xxh128) {
        return SpecimenImportTaskOutcome::ExactDuplicate;
    }

    let preview =
        generate_preview_variant(&state, &candidate, &download_config, &match_config, source).await;
    let preview_fingerprint = preview.as_ref().map(|variant| variant.fingerprint.clone());
    let processed = ProcessedSpecimenImage {
        fingerprint,
        bytes,
        preview,
        preview_fingerprint,
    };
    let pre_add_match = pre_add_match_summary(&state, &processed.fingerprint).await;

    let Some(mut prepared) =
        build_prepared_specimen_write(&state, candidate, added_by_id, processed, source)
    else {
        return SpecimenImportTaskOutcome::Failed;
    };
    prepared.pre_add_match = pre_add_match;
    prepared.image_processing_ms = started.elapsed().as_millis();

    SpecimenImportTaskOutcome::Prepared(Box::new(prepared))
}

async fn download_and_hash_candidate(
    state: &AppState,
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
    source: &'static str,
) -> Option<(ImageFingerprint, bytes::Bytes)> {
    let downloaded = match state
        .download_image(
            &candidate.url,
            candidate.mime_hint.as_deref(),
            download_config,
        )
        .await
    {
        Ok(downloaded) => downloaded,
        Err(error) => {
            warn!(
                event = "specimen.image_ingest_download_failed",
                source,
                guild_id = state.guild_id().get(),
                source_message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                ?error,
                "failed to download imported specimen image"
            );
            return None;
        }
    };
    let image_bytes = downloaded.bytes.clone();
    match hash_downloaded_image(
        downloaded,
        download_config.max_decoded_pixels,
        match_config,
        &state.decode_gate,
        HashMode::Specimen,
    )
    .await
    {
        Ok(fingerprint) => Some((fingerprint, image_bytes)),
        Err(error) => {
            warn!(
                event = "specimen.image_ingest_failed",
                source,
                guild_id = state.guild_id().get(),
                channel_id = candidate.channel_id.get(),
                message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                ?error,
                "failed to process imported specimen image"
            );
            None
        }
    }
}

fn build_prepared_specimen_write(
    state: &AppState,
    candidate: ImageCandidate,
    added_by_id: Id<UserMarker>,
    processed: ProcessedSpecimenImage,
    source: &'static str,
) -> Option<PreparedSpecimenWrite> {
    let ProcessedSpecimenImage {
        fingerprint,
        bytes,
        preview,
        preview_fingerprint,
    } = processed;
    let mut record = SpecimenRecord::new_add(
        candidate.guild_id,
        candidate.channel_id,
        candidate.message_id,
        candidate.author_id,
        added_by_id,
        fingerprint,
        preview_fingerprint,
    );
    let original_attachment = match SpecimenImageAttachment::original(&record, bytes) {
        Ok(attachment) => attachment,
        Err(error) => {
            warn!(
                event = "specimen.image_ingest_attachment_failed",
                source,
                guild_id = state.guild_id().get(),
                source_message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                ?error,
                "failed to prepare imported specimen image attachment"
            );
            return None;
        }
    };
    let mut image_attachments = vec![original_attachment];

    if let Some(preview) = preview {
        match SpecimenImageAttachment::discord_preview(&record, preview.bytes) {
            Ok(attachment) => image_attachments.push(attachment),
            Err(error) => {
                record.preview = None;
                warn!(
                    event = "specimen.image_ingest_preview_attachment_failed",
                    source,
                    guild_id = state.guild_id().get(),
                    source_message_id = candidate.message_id.get(),
                    image_url = %url_log_label(&candidate.url),
                    ?error,
                    "failed to prepare imported specimen preview attachment"
                );
            }
        }
    }

    let record = match record.sign(&state.secrets.specimen_hmac_secret) {
        Ok(record) => record,
        Err(error) => {
            warn!(
                event = "specimen.image_ingest_sign_failed",
                source,
                guild_id = state.guild_id().get(),
                source_message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                ?error,
                "failed to sign imported specimen image record"
            );
            return None;
        }
    };

    Some(PreparedSpecimenWrite {
        candidate,
        record,
        image_attachments,
        image_processing_ms: 0,
        pre_add_match: None,
    })
}

async fn pre_add_match_summary(state: &AppState, fingerprint: &ImageFingerprint) -> Option<String> {
    let policy = state.active_config_arc().detection_policy.clone();
    match state
        .find_match_for_policy(Arc::new(fingerprint.clone()), policy)
        .await
    {
        Ok(Some(outcome)) => Some(format!(
            "{} `{}` via `{:?}`. Gates: `{}`",
            outcome.decision_name(),
            outcome.specimen_id,
            outcome.confidence,
            outcome.tripped_gates_summary()
        )),
        Ok(None) => Some("No existing specimen matched before this add.".to_owned()),
        Err(error) => {
            warn!(
                event = "specimen.pre_add_match_failed",
                guild_id = state.guild_id().get(),
                ?error,
                "failed to check whether specimen image already matched"
            );
            Some(format!("Pre-add match check failed: {error:#}"))
        }
    }
}

async fn write_prepared_specimen(
    state: &AppState,
    write: PreparedSpecimenWrite,
    source: &'static str,
    summary: &mut ImageImportSummary,
) {
    let PreparedSpecimenWrite {
        candidate,
        record,
        image_attachments,
        image_processing_ms,
        pre_add_match,
    } = write;
    match state
        .write_specimen_record(
            record,
            image_attachments,
            SpecimenWriteLogContext {
                image_url: Some(candidate.url.clone()),
                image_processing_ms: Some(image_processing_ms),
                pre_add_match,
            },
        )
        .await
    {
        Ok(SpecimenWriteOutcome::Added(_)) => {
            summary.added += 1;
        }
        Ok(SpecimenWriteOutcome::Duplicate) => {
            summary.exact_duplicates += 1;
        }
        Err(error) => {
            summary.failed += 1;
            warn!(
                event = "specimen.image_ingest_ledger_write_failed",
                source,
                guild_id = state.guild_id().get(),
                source_message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                ?error,
                "failed to write imported specimen ledger record"
            );
        }
    }
}

async fn generate_preview_variant(
    state: &AppState,
    candidate: &ImageCandidate,
    download_config: &DownloadConfig,
    match_config: &MatchConfig,
    source: &'static str,
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
        Err(error) => {
            warn!(
                event = "specimen.image_ingest_preview_download_failed",
                source,
                guild_id = state.guild_id().get(),
                source_message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                ?error,
                "failed to download Discord preview for imported specimen"
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
        HashMode::Specimen,
    )
    .await
    {
        Ok(fingerprint) => Some(SpecimenPreviewVariant { fingerprint, bytes }),
        Err(error) => {
            warn!(
                event = "specimen.image_ingest_preview_hash_failed",
                source,
                guild_id = state.guild_id().get(),
                source_message_id = candidate.message_id.get(),
                image_url = %url_log_label(&candidate.url),
                ?error,
                "failed to hash Discord preview for imported specimen"
            );
            None
        }
    }
}
