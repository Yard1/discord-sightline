#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use super::{
    BatchHashError, collect_image_paths, detection_policy_from_match_config, sanitize_file_stem,
};
use crate::{
    configuration::{
        app::AppConfig,
        guild::{TextGatePolicy, normalize_text_gate_pattern},
    },
    image::{
        engine::{
            DirectoryArtifactSink, NoopArtifactSink, OCR_SEQUENCE_MAX_EDIT_DISTANCE, OcrResponse,
            ProgressiveEngine, StaticOcrClient, TextGateDecision, evaluate_text_gate,
        },
        matcher,
        pipeline::{HashMode, PreparedOcrCrop, hash_image_bytes, prepare_ocr_payload_from_bytes},
        types::ExportedImageFingerprint,
    },
    ocr_space::{OcrSpaceClient, load_api_key_from_env},
};
use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::{
    fs,
    io::{self, Read as _},
    path::{Path, PathBuf},
    time::Duration,
};

const CHECK_OCR_SEQUENCE_USAGE: &str =
    "usage: discord-sightline check-ocr-sequence <sequence> [--text TEXT | --text-file PATH]";

#[derive(Debug, Clone)]
pub(super) struct InspectOptions {
    artifacts_dir: Option<PathBuf>,
    fake_ocr_text: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct OcrSpaceCliReport {
    source_path: String,
    crop: OcrSpaceCliCrop,
    response: crate::ocr_space::OcrSpaceRead,
}

#[derive(Debug, Serialize)]
pub(super) struct OcrSequenceCheckReport {
    found: bool,
    normalized_sequence: String,
    normalized_text: String,
    max_edit_distance: usize,
}

pub(super) fn check_ocr_sequence(
    sequence: &str,
    args: &[String],
) -> Result<OcrSequenceCheckReport> {
    let text = match args {
        [] => {
            let mut text = String::new();
            io::stdin()
                .read_to_string(&mut text)
                .context("reading OCR text from stdin")?;
            text
        }
        [option, text] if option == "--text" => text.clone(),
        [option, path] if option == "--text-file" => {
            fs::read_to_string(path).with_context(|| format!("reading OCR text from {path}"))?
        }
        _ => return Err(anyhow!(CHECK_OCR_SEQUENCE_USAGE)),
    };

    evaluate_ocr_sequence(&text, sequence)
}

fn evaluate_ocr_sequence(text: &str, sequence: &str) -> Result<OcrSequenceCheckReport> {
    let normalized_sequence = normalize_text_gate_pattern(sequence);
    if normalized_sequence.is_empty() {
        return Err(anyhow!(
            "sequence must contain at least one letter or number"
        ));
    }
    let normalized_text = normalize_text_gate_pattern(text);
    let policy = TextGatePolicy {
        enabled: true,
        keyword_threshold: 0,
        keyword_max_distance: 0,
        keywords: Vec::new(),
        sentences: vec![normalized_sequence.clone()],
    };
    let report = evaluate_text_gate(
        &policy,
        &OcrResponse {
            readable: !text.trim().is_empty(),
            text: text.to_owned(),
        },
    );

    Ok(OcrSequenceCheckReport {
        found: report.decision == TextGateDecision::ConfirmedSentence,
        normalized_sequence,
        normalized_text,
        max_edit_distance: OCR_SEQUENCE_MAX_EDIT_DISTANCE,
    })
}

#[derive(Debug, Serialize)]
struct OcrSpaceCliCrop {
    label: String,
    width: u32,
    height: u32,
    mime: String,
    bytes: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct InspectImageReport {
    source_path: String,
    artifact_dir: Option<String>,
    fake_ocr_text_provided: bool,
    fingerprint: ExportedImageFingerprint,
    decision: crate::image::engine::ProgressiveDecision,
    crop_count: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct OcrCropExportSummary {
    input_path: String,
    output_dir: String,
    processed: usize,
    crops_written: usize,
    failed: usize,
    files: Vec<OcrCropExportFile>,
    errors: Vec<BatchHashError>,
}

#[derive(Debug, Serialize)]
struct OcrCropExportFile {
    source_path: String,
    output_path: String,
    crop_label: String,
    source_width: u32,
    source_height: u32,
    source_bytes: u64,
    width: u32,
    height: u32,
    output_bytes: u64,
}

#[derive(Debug, Serialize)]
pub(super) struct OcrCropLeaveOneOutSummary {
    input_path: String,
    output_dir: String,
    specimen_count: usize,
    fold_count: usize,
    crops_written: usize,
    failed: usize,
    folds: Vec<OcrCropLeaveOneOutFold>,
    errors: Vec<BatchHashError>,
}

#[derive(Debug, Serialize)]
struct OcrCropLeaveOneOutFold {
    held_out_source_path: String,
    fold_dir: String,
    train_crops_written: usize,
    test_crops_written: usize,
}

pub(super) fn export_ocr_crop_batch(
    input_path: &Path,
    output_dir: &Path,
    config: &AppConfig,
) -> Result<OcrCropExportSummary> {
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "creating OCR crop output directory {}",
            output_dir.display()
        )
    })?;

    let image_paths = collect_image_paths(input_path)?;
    let mut files = Vec::new();
    let mut errors = Vec::new();
    let mut processed = 0usize;

    for path in image_paths {
        match export_ocr_crops_for_image(&path, output_dir, config) {
            Ok(mut exported) => {
                processed += 1;
                files.append(&mut exported);
            }
            Err(source) => errors.push(BatchHashError {
                source_path: path.display().to_string(),
                error: source.to_string(),
            }),
        }
    }

    Ok(OcrCropExportSummary {
        input_path: input_path.display().to_string(),
        output_dir: output_dir.display().to_string(),
        processed,
        crops_written: files.len(),
        failed: errors.len(),
        files,
        errors,
    })
}

fn export_ocr_crops_for_image(
    path: &Path,
    output_dir: &Path,
    config: &AppConfig,
) -> Result<Vec<OcrCropExportFile>> {
    let prepared = prepare_ocr_crop_export(path, config)?;
    write_prepared_ocr_crop_export(&prepared, output_dir)
}

pub(super) fn export_ocr_crop_leave_one_out(
    input_path: &Path,
    output_dir: &Path,
    config: &AppConfig,
) -> Result<OcrCropLeaveOneOutSummary> {
    let image_paths = collect_image_paths(input_path)?;
    anyhow::ensure!(
        image_paths.len() >= 2,
        "leave-one-out crop export requires at least two specimen images"
    );
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "creating leave-one-out crop output directory {}",
            output_dir.display()
        )
    })?;

    let mut prepared = Vec::with_capacity(image_paths.len());
    let mut errors = Vec::new();
    for path in &image_paths {
        match prepare_ocr_crop_export(path, config) {
            Ok(export) => prepared.push(Some(export)),
            Err(source) => {
                errors.push(BatchHashError {
                    source_path: path.display().to_string(),
                    error: source.to_string(),
                });
                prepared.push(None);
            }
        }
    }

    let mut folds = Vec::new();
    let mut crops_written = 0usize;

    for (held_out_index, held_out_path) in image_paths.iter().enumerate() {
        let fold_name = specimen_fold_name(held_out_path, held_out_index);
        let fold_dir = output_dir.join(fold_name);
        let train_dir = fold_dir.join("train");
        let test_dir = fold_dir.join("test");
        fs::create_dir_all(&train_dir)
            .with_context(|| format!("creating {}", train_dir.display()))?;
        fs::create_dir_all(&test_dir)
            .with_context(|| format!("creating {}", test_dir.display()))?;

        let mut train_crops_written = 0usize;
        let mut test_crops_written = 0usize;
        for (index, export) in prepared.iter().enumerate() {
            let Some(export) = export else {
                continue;
            };
            let target_dir = if index == held_out_index {
                &test_dir
            } else {
                &train_dir
            };
            match write_prepared_ocr_crop_export(export, target_dir) {
                Ok(exported) => {
                    let count = exported.len();
                    crops_written += count;
                    if index == held_out_index {
                        test_crops_written += count;
                    } else {
                        train_crops_written += count;
                    }
                }
                Err(source) => errors.push(BatchHashError {
                    source_path: export.source_path.display().to_string(),
                    error: format!("fold held out {}: {source}", held_out_path.display()),
                }),
            }
        }

        folds.push(OcrCropLeaveOneOutFold {
            held_out_source_path: held_out_path.display().to_string(),
            fold_dir: fold_dir.display().to_string(),
            train_crops_written,
            test_crops_written,
        });
    }

    Ok(OcrCropLeaveOneOutSummary {
        input_path: input_path.display().to_string(),
        output_dir: output_dir.display().to_string(),
        specimen_count: image_paths.len(),
        fold_count: folds.len(),
        crops_written,
        failed: errors.len(),
        folds,
        errors,
    })
}

struct PreparedOcrCropExport {
    source_path: PathBuf,
    source_width: u32,
    source_height: u32,
    source_bytes: u64,
    files: Vec<PreparedOcrCropExportFile>,
}

struct PreparedOcrCropExportFile {
    file_name: String,
    crop: PreparedOcrCrop,
}

fn prepare_ocr_crop_export(path: &Path, config: &AppConfig) -> Result<PreparedOcrCropExport> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (source_width, source_height) = image::image_dimensions(path)
        .with_context(|| format!("reading image dimensions for {}", path.display()))?;
    let crop = prepare_ocr_payload_from_bytes(
        &bytes,
        config.download.max_decoded_pixels,
        &config.matching,
    )
    .with_context(|| format!("preparing OCR crops for {}", path.display()))?;
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_file_stem)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "image".to_owned());
    let files = vec![PreparedOcrCropExportFile {
        file_name: format!("{}.{}", stem, crop_extension(&crop)),
        crop,
    }];

    Ok(PreparedOcrCropExport {
        source_path: path.to_path_buf(),
        source_width,
        source_height,
        source_bytes: bytes.len() as u64,
        files,
    })
}

fn crop_extension(crop: &PreparedOcrCrop) -> &'static str {
    match crop.mime.as_str() {
        "image/jpeg" => "jpg",
        "image/png" => "png",
        "image/webp" => "webp",
        _ => "img",
    }
}

fn write_prepared_ocr_crop_export(
    prepared: &PreparedOcrCropExport,
    output_dir: &Path,
) -> Result<Vec<OcrCropExportFile>> {
    let mut exported = Vec::new();
    for file in &prepared.files {
        let output_path = output_dir.join(&file.file_name);
        fs::write(&output_path, &file.crop.bytes)
            .with_context(|| format!("writing {}", output_path.display()))?;
        exported.push(OcrCropExportFile {
            source_path: prepared.source_path.display().to_string(),
            output_path: output_path.display().to_string(),
            crop_label: file.crop.label.clone(),
            source_width: prepared.source_width,
            source_height: prepared.source_height,
            source_bytes: prepared.source_bytes,
            width: file.crop.width,
            height: file.crop.height,
            output_bytes: file.crop.bytes.len() as u64,
        });
    }
    Ok(exported)
}
fn specimen_fold_name(path: &Path, index: usize) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_file_stem)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "image".to_owned());
    format!("{:03}_{stem}", index + 1)
}

pub(super) async fn inspect_image(
    path: &Path,
    config: &AppConfig,
    options: &InspectOptions,
) -> Result<InspectImageReport> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let fingerprint = hash_image_bytes(
        &bytes,
        None,
        config.download.max_decoded_pixels,
        &config.matching,
        HashMode::candidate(),
    )?;
    let crop = prepare_ocr_payload_from_bytes(
        &bytes,
        config.download.max_decoded_pixels,
        &config.matching,
    )?;
    let crops = vec![crop];
    let policy = detection_policy_from_match_config(&config.matching);
    let matcher = matcher::Matcher::default();
    let text_gate_policy = local_text_gate_policy(options);
    let artifact_prefix = fingerprint.byte_xxh128.chars().take(16).collect::<String>();

    let decision = if let Some(artifacts_dir) = &options.artifacts_dir {
        let ocr = StaticOcrClient::new(options.fake_ocr_text.clone().unwrap_or_default());
        let artifacts = DirectoryArtifactSink::new(artifacts_dir);
        ProgressiveEngine {
            matcher: Some(&matcher),
            detection_policy: &policy,
            text_gate_policy: &text_gate_policy,
            ocr: &ocr,
            artifacts: &artifacts,
        }
        .evaluate(&fingerprint, Some(&crops), &artifact_prefix)
        .await?
    } else {
        let ocr = StaticOcrClient::new(options.fake_ocr_text.clone().unwrap_or_default());
        let artifacts = NoopArtifactSink;
        ProgressiveEngine {
            matcher: Some(&matcher),
            detection_policy: &policy,
            text_gate_policy: &text_gate_policy,
            ocr: &ocr,
            artifacts: &artifacts,
        }
        .evaluate(&fingerprint, Some(&crops), &artifact_prefix)
        .await?
    };

    Ok(InspectImageReport {
        source_path: path.display().to_string(),
        artifact_dir: options
            .artifacts_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        fake_ocr_text_provided: options.fake_ocr_text.is_some(),
        fingerprint: ExportedImageFingerprint::new(path.display().to_string(), fingerprint),
        decision,
        crop_count: crops.len(),
    })
}

pub(super) async fn test_ocr_space(path: &Path, config: &AppConfig) -> Result<OcrSpaceCliReport> {
    let api_key = load_api_key_from_env()?
        .ok_or_else(|| anyhow!("OCR_SPACE_API_KEY is required for ocr-space"))?;
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let crop = prepare_ocr_payload_from_bytes(
        &bytes,
        config.download.max_decoded_pixels,
        &config.matching,
    )?;
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(config.ocr_space.timeout_seconds))
        .user_agent("discord-sightline/0.1")
        .build()
        .context("building OCR reqwest client")?;
    let client = OcrSpaceClient::new(http, api_key, config.ocr_space.clone())?;
    let response = client.read_crop_text(&crop).await?;

    Ok(OcrSpaceCliReport {
        source_path: path.display().to_string(),
        crop: OcrSpaceCliCrop {
            label: crop.label,
            width: crop.width,
            height: crop.height,
            mime: crop.mime,
            bytes: crop.bytes.len(),
        },
        response,
    })
}

pub(super) fn parse_inspect_options(args: &[String]) -> Result<InspectOptions> {
    let mut options = InspectOptions {
        artifacts_dir: None,
        fake_ocr_text: None,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--artifacts-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--artifacts-dir requires a value"))?;
                options.artifacts_dir = Some(PathBuf::from(value));
            }
            "--fake-ocr-text" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--fake-ocr-text requires a value"))?;
                options.fake_ocr_text = Some(value.clone());
            }
            unknown => {
                return Err(anyhow!(
                    "unknown inspect option {unknown}; usage: discord-sightline inspect-image <image-path> [--artifacts-dir DIR] [--fake-ocr-text TEXT]"
                ));
            }
        }
        index += 1;
    }
    Ok(options)
}

fn local_text_gate_policy(options: &InspectOptions) -> TextGatePolicy {
    TextGatePolicy {
        enabled: options.fake_ocr_text.is_some(),
        keyword_threshold: 1,
        keyword_max_distance: 1,
        keywords: options
            .fake_ocr_text
            .as_deref()
            .map(|text| {
                text.split_whitespace()
                    .take(8)
                    .map(normalize_text_gate_pattern)
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        sentences: options
            .fake_ocr_text
            .clone()
            .filter(|text| !text.trim().is_empty())
            .map(|text| normalize_text_gate_pattern(&text))
            .into_iter()
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_checker_uses_production_whitespace_and_ocr_tolerance() {
        let report = evaluate_ocr_sequence(
            "Please CONNECT\r\n\tyour\u{2003}wal1et now",
            "connect your wallet",
        )
        .expect("sequence check should succeed");

        assert!(report.found);
        assert_eq!(report.normalized_sequence, "connect your wallet");
        assert_eq!(report.normalized_text, "please connect your wal1et now");
        assert_eq!(report.max_edit_distance, OCR_SEQUENCE_MAX_EDIT_DISTANCE);
    }

    #[test]
    fn sequence_checker_reports_a_miss() {
        let report = evaluate_ocr_sequence("unrelated readable text", "connect your wallet")
            .expect("sequence check should succeed");

        assert!(!report.found);
    }

    #[test]
    fn sequence_checker_preserves_cyrillic_text() {
        let report = evaluate_ocr_sequence(
            "деньги поступают сразу\nна баланс",
            "ДЕНЬГИ ПОСТУПАЮТ СРАЗУ НА БАЛАНС",
        )
        .expect("sequence check should succeed");

        assert!(report.found);
        assert_eq!(
            report.normalized_sequence,
            "деньги поступают сразу на баланс"
        );
    }
}
