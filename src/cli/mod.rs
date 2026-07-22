#![allow(clippy::too_many_lines)]

mod augment;
mod benchmark;
mod ocr;
mod validation;

use crate::{
    bot::ledger::SpecimenRecord,
    configuration::app::AppConfig,
    image::{
        matcher::{self, MatchEvaluationMode, compare_fingerprints},
        pipeline::{HashMode, hash_image_bytes},
        types::{ExportedImageFingerprint, ImageFingerprint, MatchConfidence},
    },
};
use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use twilight_model::id::Id;

pub fn handle_config_free_cli() -> Result<bool> {
    let mut args = env::args().skip(1).collect::<Vec<_>>().into_iter();
    let Some(command) = args.next() else {
        return Ok(false);
    };
    if command != "check-ocr-sequence" {
        return Ok(false);
    }

    let sequence = args.next().ok_or_else(|| {
        anyhow!(
            "usage: discord-sightline check-ocr-sequence <sequence> [--text TEXT | --text-file PATH]"
        )
    })?;
    let args = args.collect::<Vec<_>>();
    let report = ocr::check_ocr_sequence(&sequence, &args)?;
    println!("{}", serde_json::to_string_pretty(&report)?);

    Ok(true)
}

pub async fn handle_standalone_cli(config: &AppConfig) -> Result<bool> {
    let mut args = env::args().skip(1).collect::<Vec<_>>().into_iter();
    let Some(command) = args.next() else {
        return Ok(false);
    };

    match command.as_str() {
        "hash-image" => {
            let path = args
                .next()
                .ok_or_else(|| anyhow!("usage: discord-sightline hash-image <image-path>"))?;
            let fingerprint = hash_image_file(&path, config)?;
            let exported = ExportedImageFingerprint::new(path.clone(), fingerprint);

            println!("{}", serde_json::to_string_pretty(&exported)?);

            Ok(true)
        }
        "hash-images" => {
            let input_path = args.next().ok_or_else(|| {
                anyhow!("usage: discord-sightline hash-images <input-file-or-dir> <output-dir>")
            })?;
            let output_dir = args.next().ok_or_else(|| {
                anyhow!("usage: discord-sightline hash-images <input-file-or-dir> <output-dir>")
            })?;
            if args.next().is_some() {
                return Err(anyhow!(
                    "usage: discord-sightline hash-images <input-file-or-dir> <output-dir>"
                ));
            }

            let summary = benchmark::hash_image_batch(
                Path::new(&input_path),
                Path::new(&output_dir),
                config,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);

            Ok(true)
        }
        "augment-images" => {
            let input_path = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline augment-images <input-file-or-dir> <output-dir> [--profile mild|geometry|full] [--jpeg-quality N] [--include-original]"
                )
            })?;
            let output_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline augment-images <input-file-or-dir> <output-dir> [--profile mild|geometry|full] [--jpeg-quality N] [--include-original]"
                )
            })?;
            let args = args.collect::<Vec<_>>();
            let options = augment::parse_augment_options(&args)?;
            let report = augment::augment_image_batch(
                Path::new(&input_path),
                Path::new(&output_dir),
                options,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            Ok(true)
        }
        "compare-images" => {
            let specimen_path = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline compare-images <specimen-image-path> <candidate-image-path>"
                )
            })?;
            let candidate_path = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline compare-images <specimen-image-path> <candidate-image-path>"
                )
            })?;
            let specimen = hash_image_file(&specimen_path, config)
                .with_context(|| format!("hashing specimen image {specimen_path}"))?;
            let candidate = hash_image_file(&candidate_path, config)
                .with_context(|| format!("hashing candidate image {candidate_path}"))?;
            let comparison = compare_fingerprints(&specimen, &candidate, &config.matching);

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "matching_config": &config.matching,
                    "specimen": ExportedImageFingerprint::new(specimen_path, specimen),
                    "candidate": ExportedImageFingerprint::new(candidate_path, candidate),
                    "comparison": comparison
                }))?
            );

            Ok(true)
        }
        "compare-image-sets" => {
            let specimen_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline compare-image-sets <specimen-fingerprint-dir> <candidate-fingerprint-dir> [--evaluate-all-stages] [--include-misses] [--exclude-same-source]"
                )
            })?;
            let candidate_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline compare-image-sets <specimen-fingerprint-dir> <candidate-fingerprint-dir> [--evaluate-all-stages] [--include-misses] [--exclude-same-source]"
                )
            })?;
            let args = args.collect::<Vec<_>>();
            let options = parse_compare_sets_options(&args)?;

            let report = compare_fingerprint_sets(
                Path::new(&specimen_dir),
                Path::new(&candidate_dir),
                config,
                options,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            Ok(true)
        }
        "validate-threshold-sweep" => {
            let positive_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline validate-threshold-sweep <positive-fingerprint-dir> <negative-fingerprint-dir-or-> --config CONFIG [--config CONFIG...] [--folds N] [--seed N] [--positive-mode matched|matched-or-suspicious] [--evaluate-all-stages]"
                )
            })?;
            let negative_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline validate-threshold-sweep <positive-fingerprint-dir> <negative-fingerprint-dir-or-> --config CONFIG [--config CONFIG...] [--folds N] [--seed N] [--positive-mode matched|matched-or-suspicious] [--evaluate-all-stages]"
                )
            })?;
            let args = args.collect::<Vec<_>>();
            let options = validation::parse_threshold_sweep_options(&args)?;
            let report = validation::validate_threshold_sweep(
                Path::new(&positive_dir),
                validation::negative_fingerprint_dir_arg(&negative_dir),
                &options,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            Ok(true)
        }
        "benchmark-matcher" => {
            let specimen_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline benchmark-matcher <specimen-fingerprint-dir> <candidate-fingerprint-dir> [--repeat N] [--warmup N] [--summary-only] [--no-pairwise]"
                )
            })?;
            let candidate_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline benchmark-matcher <specimen-fingerprint-dir> <candidate-fingerprint-dir> [--repeat N] [--warmup N] [--summary-only] [--no-pairwise]"
                )
            })?;
            let args = args.collect::<Vec<_>>();
            let options = benchmark::parse_matcher_benchmark_options(&args)?;
            let report = benchmark::benchmark_matcher(
                Path::new(&specimen_dir),
                Path::new(&candidate_dir),
                config,
                &options,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            Ok(true)
        }
        "inspect-image" => {
            let path = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline inspect-image <image-path> [--artifacts-dir DIR] [--fake-ocr-text TEXT]"
                )
            })?;
            let args = args.collect::<Vec<_>>();
            let options = ocr::parse_inspect_options(&args)?;
            let report = ocr::inspect_image(Path::new(&path), config, &options).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            Ok(true)
        }
        "ocr-space" => {
            let path = args
                .next()
                .ok_or_else(|| anyhow!("usage: discord-sightline ocr-space <image-path>"))?;
            if args.next().is_some() {
                return Err(anyhow!("usage: discord-sightline ocr-space <image-path>"));
            }
            let report = ocr::test_ocr_space(Path::new(&path), config).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            Ok(true)
        }
        "export-ocr-crops" => {
            let input_path = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline export-ocr-crops <input-file-or-dir> <output-dir>"
                )
            })?;
            let output_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline export-ocr-crops <input-file-or-dir> <output-dir>"
                )
            })?;
            if args.next().is_some() {
                return Err(anyhow!(
                    "usage: discord-sightline export-ocr-crops <input-file-or-dir> <output-dir>"
                ));
            }

            let summary =
                ocr::export_ocr_crop_batch(Path::new(&input_path), Path::new(&output_dir), config)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);

            Ok(true)
        }
        "export-ocr-crops-leave-one-out" => {
            let input_path = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline export-ocr-crops-leave-one-out <input-dir> <output-dir>"
                )
            })?;
            let output_dir = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline export-ocr-crops-leave-one-out <input-dir> <output-dir>"
                )
            })?;
            if args.next().is_some() {
                return Err(anyhow!(
                    "usage: discord-sightline export-ocr-crops-leave-one-out <input-dir> <output-dir>"
                ));
            }

            let summary = ocr::export_ocr_crop_leave_one_out(
                Path::new(&input_path),
                Path::new(&output_dir),
                config,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);

            Ok(true)
        }
        "benchmark-images" => {
            let input_path = args.next().ok_or_else(|| {
                anyhow!(
                    "usage: discord-sightline benchmark-images <input-file-or-dir> [--repeat N] [--warmup N] [--mode candidate|specimen|full-diagnostics] [--summary-only] [--max-preload-bytes N]"
                )
            })?;
            let args = args.collect::<Vec<_>>();
            let options = benchmark::parse_benchmark_options(&args)?;
            let report = benchmark::benchmark_images(Path::new(&input_path), config, &options)?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            Ok(true)
        }
        _ => Ok(false),
    }
}

fn hash_image_file(path: &str, config: &AppConfig) -> Result<ImageFingerprint> {
    let bytes = fs::read(path).with_context(|| format!("reading {path}"))?;
    hash_image_bytes(
        &bytes,
        None,
        config.download.max_decoded_pixels,
        &config.matching,
        HashMode::FullDiagnostics,
    )
}

#[derive(Debug, Serialize)]
struct BatchHashSummary {
    input_path: String,
    output_dir: String,
    processed: usize,
    failed: usize,
    files: Vec<BatchHashFile>,
    errors: Vec<BatchHashError>,
}

#[derive(Debug, Serialize)]
struct BatchHashFile {
    source_path: String,
    output_path: String,
    width: u32,
    height: u32,
    byte_xxh128: String,
    phash64: String,
    dhash64: String,
    anchor_count: usize,
    local_hash_count: usize,
}

#[derive(Debug, Serialize)]
struct BatchHashError {
    source_path: String,
    error: String,
}

#[derive(Debug, Serialize)]
struct SetComparisonReport {
    specimen_dir: String,
    candidate_dir: String,
    specimen_count: usize,
    candidate_count: usize,
    comparisons_total: usize,
    matches: Vec<SetComparisonMatch>,
    best_per_candidate: Vec<BestCandidateMatch>,
    unmatched_candidates: Vec<UnmatchedCandidate>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CompareSetsOptions {
    evaluate_all_stages: bool,
    include_misses: bool,
    exclude_same_source: bool,
}

#[derive(Debug, Serialize)]
struct SetComparisonMatch {
    specimen_file: String,
    specimen_source: String,
    candidate_file: String,
    candidate_source: String,
    matched: bool,
    suspicious: bool,
    outcome: crate::image::types::MatchOutcome,
    comparison: matcher::FingerprintComparison,
}

#[derive(Debug, Serialize)]
struct BestCandidateMatch {
    candidate_file: String,
    candidate_source: String,
    specimen_file: String,
    specimen_source: String,
    matched: bool,
    suspicious: bool,
    score: f64,
    outcome: crate::image::types::MatchOutcome,
}

#[derive(Debug, Serialize)]
struct UnmatchedCandidate {
    candidate_file: String,
    candidate_source: String,
    diagnostics: crate::image::types::MatchDiagnostics,
}

#[derive(Debug, Clone)]
struct LoadedFingerprint {
    file_path: PathBuf,
    source_path: String,
    fingerprint: ImageFingerprint,
}

fn compare_fingerprint_sets(
    specimen_dir: &Path,
    candidate_dir: &Path,
    config: &AppConfig,
    options: CompareSetsOptions,
) -> Result<SetComparisonReport> {
    let specimens = load_fingerprint_dir(specimen_dir)
        .with_context(|| format!("loading specimens from {}", specimen_dir.display()))?;
    let candidates = load_fingerprint_dir(candidate_dir)
        .with_context(|| format!("loading candidates from {}", candidate_dir.display()))?;
    let specimen_records = specimen_records_from_loaded(&specimens, "local");
    let fingerprint_matcher = matcher::Matcher::new(specimen_records);
    let policy = detection_policy_from_match_config(&config.matching);
    let evaluation_mode = if options.evaluate_all_stages {
        MatchEvaluationMode::ExhaustiveDiagnostics
    } else {
        MatchEvaluationMode::ShortCircuit
    };
    let specimen_by_id = specimens
        .iter()
        .enumerate()
        .map(|(index, specimen)| (local_specimen_id("local", index), specimen))
        .collect::<std::collections::HashMap<_, _>>();
    let mut matched_pairs = Vec::new();
    let mut best_per_candidate = Vec::new();
    let mut unmatched_candidates = Vec::new();

    for candidate in &candidates {
        let filtered_specimens;
        let filtered_matcher;
        let filtered_specimen_by_id;
        let (active_matcher, active_specimen_by_id) = if options.exclude_same_source {
            filtered_specimens = specimens
                .iter()
                .filter(|specimen| specimen.source_path != candidate.source_path)
                .cloned()
                .collect::<Vec<_>>();
            filtered_matcher =
                matcher::Matcher::new(specimen_records_from_loaded(&filtered_specimens, "local"));
            filtered_specimen_by_id = specimen_id_map(&filtered_specimens);
            (&filtered_matcher, &filtered_specimen_by_id)
        } else {
            (&fingerprint_matcher, &specimen_by_id)
        };
        let explanation = active_matcher.explain_for_policy_with_mode(
            &candidate.fingerprint,
            &policy,
            evaluation_mode,
        );
        let Some(outcome) = explanation.outcome else {
            if options.include_misses {
                unmatched_candidates.push(UnmatchedCandidate {
                    candidate_file: candidate.file_path.display().to_string(),
                    candidate_source: candidate.source_path.clone(),
                    diagnostics: explanation.diagnostics,
                });
            }
            continue;
        };
        let (specimen_file, specimen_source, comparison) =
            if let Some(specimen) = active_specimen_by_id.get(&outcome.specimen_id) {
                (
                    specimen.file_path.display().to_string(),
                    specimen.source_path.clone(),
                    compare_fingerprints(
                        &specimen.fingerprint,
                        &candidate.fingerprint,
                        &config.matching,
                    ),
                )
            } else {
                (
                    outcome.specimen_id.clone(),
                    "unknown_specimen".to_owned(),
                    compare_fingerprints(
                        &candidate.fingerprint,
                        &candidate.fingerprint,
                        &config.matching,
                    ),
                )
            };
        matched_pairs.push(SetComparisonMatch {
            specimen_file: specimen_file.clone(),
            specimen_source: specimen_source.clone(),
            candidate_file: candidate.file_path.display().to_string(),
            candidate_source: candidate.source_path.clone(),
            matched: !outcome.suspicious,
            suspicious: outcome.suspicious,
            outcome: outcome.clone(),
            comparison: comparison.clone(),
        });
        best_per_candidate.push(BestCandidateMatch {
            candidate_file: candidate.file_path.display().to_string(),
            candidate_source: candidate.source_path.clone(),
            specimen_file,
            specimen_source,
            matched: !outcome.suspicious,
            suspicious: outcome.suspicious,
            score: outcome_score(&outcome),
            outcome,
        });
    }

    Ok(SetComparisonReport {
        specimen_dir: specimen_dir.display().to_string(),
        candidate_dir: candidate_dir.display().to_string(),
        specimen_count: specimens.len(),
        candidate_count: candidates.len(),
        comparisons_total: specimens.len() * candidates.len(),
        matches: matched_pairs,
        best_per_candidate,
        unmatched_candidates,
    })
}

fn specimen_id_map(
    specimens: &[LoadedFingerprint],
) -> std::collections::HashMap<String, &LoadedFingerprint> {
    specimens
        .iter()
        .enumerate()
        .map(|(index, specimen)| (local_specimen_id("local", index), specimen))
        .collect()
}

fn specimen_records_from_loaded(
    specimens: &[LoadedFingerprint],
    prefix: &str,
) -> Vec<SpecimenRecord> {
    specimens
        .iter()
        .enumerate()
        .map(|(index, specimen)| {
            let mut record = SpecimenRecord::new_add(
                Id::new(1),
                Id::new(1),
                Id::new((index + 1) as u64),
                Id::new(1),
                Id::new(1),
                specimen.fingerprint.clone(),
                None,
            );
            record.specimen_id = local_specimen_id(prefix, index);
            record
        })
        .collect()
}

fn local_specimen_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_{index}")
}

fn detection_policy_from_match_config(
    config: &crate::configuration::app::MatchConfig,
) -> crate::configuration::guild::DetectionPolicy {
    crate::configuration::guild::DetectionPolicy::from_match_config(config)
}

fn load_fingerprint_dir(dir: &Path) -> Result<Vec<LoadedFingerprint>> {
    let mut files = collect_files_with_extension(dir, "json")?;
    files.sort();
    files
        .into_iter()
        .map(|file_path| {
            let raw = fs::read_to_string(&file_path)
                .with_context(|| format!("reading {}", file_path.display()))?;
            let record: ExportedImageFingerprint = serde_json::from_str(&raw)
                .with_context(|| format!("parsing {}", file_path.display()))?;
            let source_path = record.source_path.clone();
            let fingerprint = record
                .into_validated_fingerprint()
                .with_context(|| format!("validating {}", file_path.display()))?;
            Ok(LoadedFingerprint {
                file_path,
                source_path,
                fingerprint,
            })
        })
        .collect()
}

fn collect_image_paths(input_path: &Path) -> Result<Vec<PathBuf>> {
    const MAX_RECURSION_DEPTH: usize = 16;
    const MAX_IMAGE_FILES: usize = 10_000;

    if input_path.is_file() {
        return if is_supported_image_path(input_path) {
            Ok(vec![input_path.to_owned()])
        } else {
            Err(anyhow!(
                "{} is not a supported image file",
                input_path.display()
            ))
        };
    }

    let mut paths = Vec::new();
    collect_image_paths_inner(
        input_path,
        &mut paths,
        0,
        MAX_RECURSION_DEPTH,
        MAX_IMAGE_FILES,
    )?;
    paths.sort();
    Ok(paths)
}

fn collect_image_paths_inner(
    dir: &Path,
    paths: &mut Vec<PathBuf>,
    depth: usize,
    max_depth: usize,
    max_files: usize,
) -> Result<()> {
    if depth > max_depth {
        return Err(anyhow!(
            "image directory recursion exceeded max depth {max_depth} at {}",
            dir.display()
        ));
    }
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_image_paths_inner(&path, paths, depth + 1, max_depth, max_files)?;
        } else if metadata.is_file() && is_supported_image_path(&path) {
            if paths.len() >= max_files {
                return Err(anyhow!(
                    "image collection exceeded max file count {max_files}"
                ));
            }
            paths.push(path);
        }
    }
    Ok(())
}

fn collect_files_with_extension(dir: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.metadata()?.is_file()
            && path
                .file_name()
                .and_then(|value| value.to_str())
                .is_none_or(|value| !value.starts_with('.'))
            && path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case(extension))
        {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .is_some_and(|extension| {
            matches!(extension.as_str(), "jpg" | "jpeg" | "png" | "webp" | "gif")
        })
}

fn fingerprint_output_name(path: &Path, fingerprint: &ImageFingerprint) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_file_stem)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "image".to_owned());
    let short_hash = fingerprint.byte_xxh128.chars().take(16).collect::<String>();
    format!("{stem}_{short_hash}.sightline.json")
}

fn sanitize_file_stem(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn path_to_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow!("{} is not valid UTF-8", path.display()))
}

fn outcome_score(outcome: &crate::image::types::MatchOutcome) -> f64 {
    if let Some(score) = outcome.match_score {
        return f64::from(score);
    }
    if matches!(outcome.confidence, MatchConfidence::ExactXxh128) {
        return 10_000.0;
    }
    let perceptual = outcome
        .phash64_distance
        .zip(outcome.dhash64_distance)
        .map_or(0.0, |(phash, dhash)| 1_000.0 - f64::from(phash + dhash));
    let local = outcome.local_average_distance.map_or(0.0, |distance| {
        let anchor_hits = f64::from(
            u32::try_from(outcome.local_anchor_hits.unwrap_or_default()).unwrap_or(u32::MAX),
        );
        let distinct_regions = f64::from(
            u32::try_from(outcome.local_distinct_regions.unwrap_or_default()).unwrap_or(u32::MAX),
        );
        anchor_hits * 100.0 + distinct_regions * 10.0 - f64::from(distance)
    });
    perceptual.max(local)
}

fn parse_compare_sets_options(args: &[String]) -> Result<CompareSetsOptions> {
    let mut options = CompareSetsOptions::default();
    for arg in args {
        match arg.as_str() {
            "--evaluate-all-stages" => options.evaluate_all_stages = true,
            "--include-misses" => options.include_misses = true,
            "--exclude-same-source" => options.exclude_same_source = true,
            unknown => {
                return Err(anyhow!(
                    "unknown compare-image-sets option {unknown}; usage: discord-sightline compare-image-sets <specimen-fingerprint-dir> <candidate-fingerprint-dir> [--evaluate-all-stages] [--include-misses] [--exclude-same-source]"
                ));
            }
        }
    }
    Ok(options)
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize> {
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("{name} must be a number"))?;
    if parsed == 0 {
        return Err(anyhow!("{name} must be greater than 0"));
    }
    Ok(parsed)
}
