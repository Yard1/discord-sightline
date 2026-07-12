#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use super::{
    BatchHashError, BatchHashFile, BatchHashSummary, LoadedFingerprint, collect_image_paths,
    detection_policy_from_match_config, fingerprint_output_name, hash_image_file,
    load_fingerprint_dir, parse_positive_usize, path_to_str, specimen_records_from_loaded,
};
use crate::{
    configuration::app::AppConfig,
    image::{
        matcher::{self, compare_fingerprints},
        pipeline::{HashMode, PipelineTimings, hash_image_bytes, hash_image_bytes_with_timings},
        types::ExportedImageFingerprint,
    },
};
use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Clone)]
pub(super) struct BenchmarkOptions {
    repeat: usize,
    warmup: usize,
    mode: HashMode,
    summary_only: bool,
    max_preload_bytes: usize,
}

#[derive(Debug, Clone)]
pub(super) struct MatcherBenchmarkOptions {
    repeat: usize,
    warmup: usize,
    summary_only: bool,
    pairwise: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct MatcherBenchmarkReport {
    specimen_dir: String,
    candidate_dir: String,
    specimen_count: usize,
    candidate_count: usize,
    repeat: usize,
    warmup: usize,
    matcher_build_ms: u128,
    matcher_index: matcher::MatcherIndexStats,
    production_find_for_policy: MatcherBenchmarkPhase,
    pairwise_compare_fingerprints: Option<MatcherBenchmarkPhase>,
}

#[derive(Debug, Serialize)]
struct MatcherBenchmarkPhase {
    name: &'static str,
    attempted: usize,
    matched: usize,
    suspicious: usize,
    passed: usize,
    wall_ms: u128,
    operations_per_second: f64,
    latency_us: DurationStats,
    per_candidate: Vec<MatcherBenchmarkCandidate>,
}

#[derive(Debug, Serialize)]
struct MatcherBenchmarkCandidate {
    candidate_file: String,
    candidate_source: String,
    attempts: usize,
    matched: usize,
    suspicious: usize,
    passed: usize,
    latency_us: DurationStats,
}
#[derive(Debug, Serialize)]
pub(super) struct ImageBenchmarkReport {
    input_path: String,
    image_files: usize,
    repeat: usize,
    warmup: usize,
    mode: &'static str,
    max_preload_bytes: usize,
    preload_ms: u128,
    attempted_hashes: usize,
    successful_hashes: usize,
    failed_hashes: usize,
    wall_ms: u128,
    successful_hashes_per_second: f64,
    latency_ms: DurationStats,
    pipeline_steps_us: Vec<PipelineStepStats>,
    per_file: Vec<ImageBenchmarkFile>,
    errors: Vec<BatchHashError>,
}

struct PreparedBenchmarkImage {
    path: PathBuf,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct ImageBenchmarkFile {
    source_path: String,
    attempts: usize,
    successful: usize,
    failed: usize,
    total_ms: u128,
    latency_ms: DurationStats,
    width: Option<u32>,
    height: Option<u32>,
    byte_xxh128: Option<String>,
    anchor_count: Option<usize>,
    local_hash_count: Option<usize>,
    pipeline_steps_us: Vec<PipelineStepStats>,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DurationStats {
    count: usize,
    min: Option<u128>,
    max: Option<u128>,
    avg: Option<f64>,
    p50: Option<u128>,
    p90: Option<u128>,
    p95: Option<u128>,
    p99: Option<u128>,
}

#[derive(Debug, Clone, Serialize)]
struct PipelineStepStats {
    step: &'static str,
    timings: DurationStats,
    avg_percent_of_total: Option<f64>,
}

pub(super) fn hash_image_batch(
    input_path: &Path,
    output_dir: &Path,
    config: &AppConfig,
) -> Result<BatchHashSummary> {
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "creating fingerprint output directory {}",
            output_dir.display()
        )
    })?;

    let image_paths = collect_image_paths(input_path)?;
    let mut files = Vec::new();
    let mut errors = Vec::new();

    for path in image_paths {
        match hash_image_file(path_to_str(&path)?, config) {
            Ok(fingerprint) => {
                let output_path = output_dir.join(fingerprint_output_name(&path, &fingerprint));
                let record = ExportedImageFingerprint::new(path.display().to_string(), fingerprint);
                let body = serde_json::to_string_pretty(&record)?;
                fs::write(&output_path, body)
                    .with_context(|| format!("writing {}", output_path.display()))?;
                files.push(BatchHashFile {
                    source_path: record.source_path,
                    output_path: output_path.display().to_string(),
                    width: record.fingerprint.width,
                    height: record.fingerprint.height,
                    byte_xxh128: record.fingerprint.byte_xxh128,
                    phash64: record.fingerprint.phash64,
                    dhash64: record.fingerprint.dhash64,
                    anchor_count: record.fingerprint.local_anchors.len(),
                    local_hash_count: record.fingerprint.local_hashes.len(),
                });
            }
            Err(source) => errors.push(BatchHashError {
                source_path: path.display().to_string(),
                error: source.to_string(),
            }),
        }
    }

    Ok(BatchHashSummary {
        input_path: input_path.display().to_string(),
        output_dir: output_dir.display().to_string(),
        processed: files.len(),
        failed: errors.len(),
        files,
        errors,
    })
}

pub(super) fn benchmark_images(
    input_path: &Path,
    config: &AppConfig,
    options: &BenchmarkOptions,
) -> Result<ImageBenchmarkReport> {
    let image_paths = collect_image_paths(input_path)?;
    let preload_started = Instant::now();
    let mut prepared = Vec::new();
    let mut errors = Vec::new();
    let mut preloaded_bytes = 0usize;
    for path in image_paths {
        match fs::read(&path) {
            Ok(bytes) => {
                preloaded_bytes = preloaded_bytes.saturating_add(bytes.len());
                if preloaded_bytes > options.max_preload_bytes {
                    return Err(anyhow!(
                        "benchmark preload would exceed --max-preload-bytes ({})",
                        options.max_preload_bytes
                    ));
                }
                prepared.push(PreparedBenchmarkImage { path, bytes });
            }
            Err(source) => errors.push(BatchHashError {
                source_path: path.display().to_string(),
                error: source.to_string(),
            }),
        }
    }
    let preload_ms = preload_started.elapsed().as_millis();

    for _ in 0..options.warmup {
        for image in &prepared {
            let _ = hash_image_bytes(
                &image.bytes,
                None,
                config.download.max_decoded_pixels,
                &config.matching,
                options.mode,
            );
        }
    }

    let attempted_hashes = prepared.len() * options.repeat;
    let started = Instant::now();
    let mut successful_durations = Vec::new();
    let mut successful_pipeline_timings = Vec::new();
    let mut per_file = Vec::new();
    let mut successful_hashes = 0usize;
    let mut failed_hashes = 0usize;

    for image in &prepared {
        let mut file_durations = Vec::new();
        let mut file_pipeline_timings = Vec::new();
        let mut file_successful = 0usize;
        let mut file_failed = 0usize;
        let mut file_total_ms = 0u128;
        let mut last_error = None;
        let mut sample_fingerprint = None;

        for _ in 0..options.repeat {
            let hash_started = Instant::now();
            match hash_image_bytes_with_timings(
                &image.bytes,
                None,
                config.download.max_decoded_pixels,
                &config.matching,
                options.mode,
            ) {
                Ok((fingerprint, timings)) => {
                    let elapsed = hash_started.elapsed().as_millis();
                    file_total_ms += elapsed;
                    file_durations.push(elapsed);
                    successful_durations.push(elapsed);
                    file_pipeline_timings.push(timings);
                    successful_pipeline_timings.push(timings);
                    file_successful += 1;
                    successful_hashes += 1;
                    if sample_fingerprint.is_none() {
                        sample_fingerprint = Some(fingerprint);
                    }
                }
                Err(source) => {
                    let elapsed = hash_started.elapsed().as_millis();
                    file_total_ms += elapsed;
                    file_failed += 1;
                    failed_hashes += 1;
                    let error = source.to_string();
                    last_error = Some(error.clone());
                    errors.push(BatchHashError {
                        source_path: image.path.display().to_string(),
                        error,
                    });
                }
            }
        }

        if !options.summary_only {
            per_file.push(ImageBenchmarkFile {
                source_path: image.path.display().to_string(),
                attempts: options.repeat,
                successful: file_successful,
                failed: file_failed,
                total_ms: file_total_ms,
                latency_ms: duration_stats(&file_durations),
                width: sample_fingerprint
                    .as_ref()
                    .map(|fingerprint| fingerprint.width),
                height: sample_fingerprint
                    .as_ref()
                    .map(|fingerprint| fingerprint.height),
                byte_xxh128: sample_fingerprint
                    .as_ref()
                    .map(|fingerprint| fingerprint.byte_xxh128.clone()),
                anchor_count: sample_fingerprint
                    .as_ref()
                    .map(|fingerprint| fingerprint.local_anchors.len()),
                local_hash_count: sample_fingerprint
                    .as_ref()
                    .map(|fingerprint| fingerprint.local_hashes.len()),
                pipeline_steps_us: pipeline_step_stats(&file_pipeline_timings),
                last_error,
            });
        }
    }

    let wall_ms = started.elapsed().as_millis();
    let successful_hashes_per_second = if wall_ms == 0 {
        0.0
    } else {
        successful_hashes as f64 / (wall_ms as f64 / 1000.0)
    };

    Ok(ImageBenchmarkReport {
        input_path: input_path.display().to_string(),
        image_files: prepared.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        mode: hash_mode_name(options.mode),
        max_preload_bytes: options.max_preload_bytes,
        preload_ms,
        attempted_hashes,
        successful_hashes,
        failed_hashes,
        wall_ms,
        successful_hashes_per_second,
        latency_ms: duration_stats(&successful_durations),
        pipeline_steps_us: pipeline_step_stats(&successful_pipeline_timings),
        per_file,
        errors,
    })
}

pub(super) fn benchmark_matcher(
    specimen_dir: &Path,
    candidate_dir: &Path,
    config: &AppConfig,
    options: &MatcherBenchmarkOptions,
) -> Result<MatcherBenchmarkReport> {
    let specimens = load_fingerprint_dir(specimen_dir)
        .with_context(|| format!("loading specimens from {}", specimen_dir.display()))?;
    let candidates = load_fingerprint_dir(candidate_dir)
        .with_context(|| format!("loading candidates from {}", candidate_dir.display()))?;

    let specimen_records = specimen_records_from_loaded(&specimens, "local");
    let build_started = Instant::now();
    let matcher = matcher::Matcher::new(specimen_records);
    let matcher_build_ms = build_started.elapsed().as_millis();
    let policy = detection_policy_from_match_config(&config.matching);
    let mut matcher_scratch = matcher::MatcherScratch::default();

    for _ in 0..options.warmup {
        for candidate in &candidates {
            let _ = matcher.find_for_policy_with_scratch(
                &candidate.fingerprint,
                &policy,
                &mut matcher_scratch,
            );
        }
        if options.pairwise {
            for specimen in &specimens {
                for candidate in &candidates {
                    let _ = compare_fingerprints(
                        &specimen.fingerprint,
                        &candidate.fingerprint,
                        &config.matching,
                    );
                }
            }
        }
    }

    let production_find_for_policy =
        benchmark_production_matcher_phase(&matcher, &policy, &candidates, options);
    let pairwise_compare_fingerprints = options.pairwise.then(|| {
        benchmark_pairwise_compare_phase(&specimens, &candidates, &config.matching, options)
    });

    Ok(MatcherBenchmarkReport {
        specimen_dir: specimen_dir.display().to_string(),
        candidate_dir: candidate_dir.display().to_string(),
        specimen_count: specimens.len(),
        candidate_count: candidates.len(),
        repeat: options.repeat,
        warmup: options.warmup,
        matcher_build_ms,
        matcher_index: matcher.index_stats(),
        production_find_for_policy,
        pairwise_compare_fingerprints,
    })
}
fn benchmark_production_matcher_phase(
    matcher: &matcher::Matcher,
    policy: &crate::configuration::guild::DetectionPolicy,
    candidates: &[LoadedFingerprint],
    options: &MatcherBenchmarkOptions,
) -> MatcherBenchmarkPhase {
    let started = Instant::now();
    let mut matcher_scratch = matcher::MatcherScratch::default();
    let mut durations = Vec::with_capacity(candidates.len().saturating_mul(options.repeat));
    let mut per_candidate = Vec::new();
    let mut matched = 0usize;
    let mut suspicious = 0usize;
    let mut passed = 0usize;

    for candidate in candidates {
        let mut candidate_durations = Vec::with_capacity(options.repeat);
        let mut candidate_matched = 0usize;
        let mut candidate_suspicious = 0usize;
        let mut candidate_passed = 0usize;
        for _ in 0..options.repeat {
            let op_started = Instant::now();
            let outcome = matcher.find_for_policy_with_scratch(
                &candidate.fingerprint,
                policy,
                &mut matcher_scratch,
            );
            let elapsed = op_started.elapsed().as_micros();
            durations.push(elapsed);
            candidate_durations.push(elapsed);
            match outcome {
                Some(outcome) if outcome.suspicious => {
                    suspicious += 1;
                    candidate_suspicious += 1;
                }
                Some(_) => {
                    matched += 1;
                    candidate_matched += 1;
                }
                None => {
                    passed += 1;
                    candidate_passed += 1;
                }
            }
        }
        if !options.summary_only {
            per_candidate.push(MatcherBenchmarkCandidate {
                candidate_file: candidate.file_path.display().to_string(),
                candidate_source: candidate.source_path.clone(),
                attempts: options.repeat,
                matched: candidate_matched,
                suspicious: candidate_suspicious,
                passed: candidate_passed,
                latency_us: duration_stats(&candidate_durations),
            });
        }
    }

    matcher_benchmark_phase(MatcherBenchmarkPhaseStats {
        name: "production_find_for_policy",
        attempted: candidates.len().saturating_mul(options.repeat),
        matched,
        suspicious,
        passed,
        wall_ms: started.elapsed().as_millis(),
        durations,
        per_candidate,
    })
}

fn benchmark_pairwise_compare_phase(
    specimens: &[LoadedFingerprint],
    candidates: &[LoadedFingerprint],
    config: &crate::configuration::app::MatchConfig,
    options: &MatcherBenchmarkOptions,
) -> MatcherBenchmarkPhase {
    let attempted = specimens
        .len()
        .saturating_mul(candidates.len())
        .saturating_mul(options.repeat);
    let started = Instant::now();
    let mut durations = Vec::with_capacity(attempted);
    let mut per_candidate = Vec::new();
    let mut matched = 0usize;
    let mut suspicious = 0usize;
    let mut passed = 0usize;

    for candidate in candidates {
        let mut candidate_durations =
            Vec::with_capacity(specimens.len().saturating_mul(options.repeat));
        let mut candidate_matched = 0usize;
        let mut candidate_suspicious = 0usize;
        let mut candidate_passed = 0usize;
        for _ in 0..options.repeat {
            for specimen in specimens {
                let op_started = Instant::now();
                let comparison =
                    compare_fingerprints(&specimen.fingerprint, &candidate.fingerprint, config);
                let elapsed = op_started.elapsed().as_micros();
                durations.push(elapsed);
                candidate_durations.push(elapsed);
                if comparison.matched {
                    matched += 1;
                    candidate_matched += 1;
                } else if comparison.suspicious {
                    suspicious += 1;
                    candidate_suspicious += 1;
                } else {
                    passed += 1;
                    candidate_passed += 1;
                }
            }
        }
        if !options.summary_only {
            per_candidate.push(MatcherBenchmarkCandidate {
                candidate_file: candidate.file_path.display().to_string(),
                candidate_source: candidate.source_path.clone(),
                attempts: specimens.len().saturating_mul(options.repeat),
                matched: candidate_matched,
                suspicious: candidate_suspicious,
                passed: candidate_passed,
                latency_us: duration_stats(&candidate_durations),
            });
        }
    }

    matcher_benchmark_phase(MatcherBenchmarkPhaseStats {
        name: "pairwise_compare_fingerprints",
        attempted,
        matched,
        suspicious,
        passed,
        wall_ms: started.elapsed().as_millis(),
        durations,
        per_candidate,
    })
}

struct MatcherBenchmarkPhaseStats {
    name: &'static str,
    attempted: usize,
    matched: usize,
    suspicious: usize,
    passed: usize,
    wall_ms: u128,
    durations: Vec<u128>,
    per_candidate: Vec<MatcherBenchmarkCandidate>,
}

fn matcher_benchmark_phase(stats: MatcherBenchmarkPhaseStats) -> MatcherBenchmarkPhase {
    let MatcherBenchmarkPhaseStats {
        name,
        attempted,
        matched,
        suspicious,
        passed,
        wall_ms,
        durations,
        per_candidate,
    } = stats;
    let operations_per_second = if wall_ms == 0 {
        0.0
    } else {
        attempted as f64 / (wall_ms as f64 / 1000.0)
    };
    MatcherBenchmarkPhase {
        name,
        attempted,
        matched,
        suspicious,
        passed,
        wall_ms,
        operations_per_second,
        latency_us: duration_stats(&durations),
        per_candidate,
    }
}
pub(super) fn parse_benchmark_options(args: &[String]) -> Result<BenchmarkOptions> {
    let mut options = BenchmarkOptions {
        repeat: 1,
        warmup: 0,
        mode: HashMode::FullDiagnostics,
        summary_only: false,
        max_preload_bytes: 512 * 1024 * 1024,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--repeat requires a value"))?;
                options.repeat = parse_positive_usize(value, "--repeat")?;
            }
            "--warmup" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--warmup requires a value"))?;
                options.warmup = value
                    .parse::<usize>()
                    .with_context(|| "--warmup must be a number")?;
            }
            "--mode" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--mode requires a value"))?;
                options.mode = parse_hash_mode(value)?;
            }
            "--summary-only" => options.summary_only = true,
            "--max-preload-bytes" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--max-preload-bytes requires a value"))?;
                options.max_preload_bytes = parse_positive_usize(value, "--max-preload-bytes")?;
            }
            unknown => {
                return Err(anyhow!(
                    "unknown benchmark option {unknown}; usage: discord-sightline benchmark-images <input-file-or-dir> [--repeat N] [--warmup N] [--mode candidate|specimen|full-diagnostics] [--summary-only] [--max-preload-bytes N]"
                ));
            }
        }
        index += 1;
    }
    Ok(options)
}

pub(super) fn parse_matcher_benchmark_options(args: &[String]) -> Result<MatcherBenchmarkOptions> {
    let mut options = MatcherBenchmarkOptions {
        repeat: 1,
        warmup: 0,
        summary_only: false,
        pairwise: true,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--repeat" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--repeat requires a value"))?;
                options.repeat = parse_positive_usize(value, "--repeat")?;
            }
            "--warmup" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--warmup requires a value"))?;
                options.warmup = value
                    .parse::<usize>()
                    .with_context(|| "--warmup must be a number")?;
            }
            "--summary-only" => options.summary_only = true,
            "--no-pairwise" => options.pairwise = false,
            unknown => {
                return Err(anyhow!(
                    "unknown benchmark-matcher option {unknown}; usage: discord-sightline benchmark-matcher <specimen-fingerprint-dir> <candidate-fingerprint-dir> [--repeat N] [--warmup N] [--summary-only] [--no-pairwise]"
                ));
            }
        }
        index += 1;
    }
    Ok(options)
}
fn parse_hash_mode(value: &str) -> Result<HashMode> {
    match value {
        "candidate" => Ok(HashMode::candidate()),
        "candidate-no-local" => Ok(HashMode::candidate_without_local_hashes()),
        "specimen" => Ok(HashMode::Specimen),
        "full-diagnostics" => Ok(HashMode::FullDiagnostics),
        _ => Err(anyhow!(
            "--mode must be one of candidate, candidate-no-local, specimen, or full-diagnostics"
        )),
    }
}

fn hash_mode_name(mode: HashMode) -> &'static str {
    match mode {
        HashMode::Candidate { local_hashes: true } => "candidate",
        HashMode::Candidate {
            local_hashes: false,
        } => "candidate-no-local",
        HashMode::Specimen => "specimen",
        HashMode::FullDiagnostics => "full-diagnostics",
    }
}
fn duration_stats(values: &[u128]) -> DurationStats {
    if values.is_empty() {
        return DurationStats {
            count: 0,
            min: None,
            max: None,
            avg: None,
            p50: None,
            p90: None,
            p95: None,
            p99: None,
        };
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let sum = sorted.iter().sum::<u128>();
    DurationStats {
        count: sorted.len(),
        min: sorted.first().copied(),
        max: sorted.last().copied(),
        avg: Some(sum as f64 / sorted.len() as f64),
        p50: Some(percentile(&sorted, 0.50)),
        p90: Some(percentile(&sorted, 0.90)),
        p95: Some(percentile(&sorted, 0.95)),
        p99: Some(percentile(&sorted, 0.99)),
    }
}

fn pipeline_step_stats(samples: &[PipelineTimings]) -> Vec<PipelineStepStats> {
    let total_us = samples.iter().map(|sample| sample.total_us).sum::<u128>();
    [
        (
            "total",
            samples
                .iter()
                .map(|sample| sample.total_us)
                .collect::<Vec<_>>(),
        ),
        (
            "xxh128",
            samples
                .iter()
                .map(|sample| sample.xxh128_us)
                .collect::<Vec<_>>(),
        ),
        (
            "decode",
            samples
                .iter()
                .map(|sample| sample.decode_us)
                .collect::<Vec<_>>(),
        ),
        (
            "normalize_luma",
            samples
                .iter()
                .map(|sample| sample.normalize_luma_us)
                .collect::<Vec<_>>(),
        ),
        (
            "orientation",
            samples
                .iter()
                .map(|sample| sample.orientation_us)
                .collect::<Vec<_>>(),
        ),
        (
            "base_tile_scorer",
            samples
                .iter()
                .map(|sample| sample.base_tile_scorer_us)
                .collect::<Vec<_>>(),
        ),
        (
            "local_anchors",
            samples
                .iter()
                .map(|sample| sample.local_anchors_us)
                .collect::<Vec<_>>(),
        ),
        (
            "local_hashes",
            samples
                .iter()
                .map(|sample| sample.local_hashes_us)
                .collect::<Vec<_>>(),
        ),
        (
            "whole_thumbnail",
            samples
                .iter()
                .map(|sample| sample.whole_thumbnail_us)
                .collect::<Vec<_>>(),
        ),
        (
            "visual_signature",
            samples
                .iter()
                .map(|sample| sample.visual_signature_us)
                .collect::<Vec<_>>(),
        ),
        (
            "text_grid",
            samples
                .iter()
                .map(|sample| sample.text_grid_us)
                .collect::<Vec<_>>(),
        ),
        (
            "perceptual_hashes",
            samples
                .iter()
                .map(|sample| sample.perceptual_hashes_us)
                .collect::<Vec<_>>(),
        ),
    ]
    .into_iter()
    .map(|(step, values)| {
        let step_total = values.iter().sum::<u128>();
        PipelineStepStats {
            step,
            timings: duration_stats(&values),
            avg_percent_of_total: (total_us > 0 && step != "total")
                .then(|| step_total as f64 * 100.0 / total_us as f64),
        }
    })
    .collect()
}

fn percentile(sorted: &[u128], quantile: f64) -> u128 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}
