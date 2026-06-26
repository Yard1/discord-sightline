#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use super::{
    LoadedFingerprint, detection_policy_from_match_config, load_fingerprint_dir,
    parse_positive_usize, specimen_records_from_loaded,
};
use crate::{
    configuration::app::load_config_from_path,
    image::{
        matcher::{self, MatchEvaluationMode},
        types::ImageFingerprint,
    },
};
use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PositivePredictionMode {
    Matched,
    MatchedOrSuspicious,
}

impl PositivePredictionMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::MatchedOrSuspicious => "matched-or-suspicious",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "matched" => Ok(Self::Matched),
            "matched-or-suspicious" => Ok(Self::MatchedOrSuspicious),
            _ => Err(anyhow!(
                "--positive-mode must be one of matched, matched-or-suspicious"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ThresholdSweepOptions {
    config_paths: Vec<PathBuf>,
    folds: usize,
    seed: u64,
    positive_mode: PositivePredictionMode,
    evaluate_all_stages: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct ThresholdSweepReport {
    mode: &'static str,
    positive_fingerprint_dir: String,
    negative_fingerprint_dir: Option<String>,
    positive_count: usize,
    negative_count: usize,
    folds: usize,
    seed: u64,
    positive_mode: &'static str,
    evaluate_all_stages: bool,
    runs: Vec<ThresholdSweepRun>,
}

#[derive(Debug, Serialize)]
struct ThresholdSweepRun {
    config: ThresholdSweepConfigReport,
    tier1_subset: Tier1SubsetReport,
    kfold: ValidationSummary,
    leave_one_out: ValidationSummary,
}

#[derive(Debug, Serialize)]
struct ThresholdSweepConfigReport {
    path: String,
}

#[derive(Debug, Serialize)]
struct Tier1SubsetReport {
    checked_candidates: usize,
    tier1_confirmed_candidates: usize,
    violation_count: usize,
    violations: Vec<Tier1SubsetViolation>,
}

#[derive(Debug, Serialize)]
struct Tier1SubsetViolation {
    source: String,
    tier1_specimen_id: String,
    full_outcome: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct ValidationConfusion {
    tp: usize,
    fp: usize,
    tn: usize,
    #[serde(rename = "fn")]
    false_negative: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationMetrics {
    tp: usize,
    fp: usize,
    tn: usize,
    #[serde(rename = "fn")]
    false_negative: usize,
    total: usize,
    accuracy: Option<f64>,
    precision: Option<f64>,
    recall: Option<f64>,
    specificity: Option<f64>,
    f1: Option<f64>,
    false_positive_rate: Option<f64>,
    false_negative_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationSummary {
    positive_count: usize,
    negative_count: usize,
    aggregate: ValidationMetrics,
    fully_matched_positive_sources_count: usize,
    fully_matched_positive_sources: Vec<String>,
    suspicious_only_positive_sources_count: usize,
    suspicious_only_positive_sources: Vec<String>,
    unmatched_positive_sources_count: usize,
    unmatched_positive_sources: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fully_matched_specimens_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fully_matched_specimens: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suspicious_only_specimens_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suspicious_only_specimens: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unmatched_specimens_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unmatched_specimens: Option<Vec<String>>,
    fully_matched_false_positive_sources_count: usize,
    fully_matched_false_positive_sources: Vec<String>,
    suspicious_only_false_positive_sources_count: usize,
    suspicious_only_false_positive_sources: Vec<String>,
}

#[derive(Debug, Default)]
struct PredictionCategories {
    matched: BTreeSet<String>,
    suspicious: BTreeSet<String>,
}

#[derive(Debug, Default)]
struct AggregatePredictionSets {
    positive_matched: BTreeSet<String>,
    positive_suspicious_only: BTreeSet<String>,
    positive_unmatched: BTreeSet<String>,
    negative_matched: BTreeSet<String>,
    negative_suspicious_only: BTreeSet<String>,
}

pub(super) fn validate_threshold_sweep(
    positive_dir: &Path,
    negative_dir: Option<&Path>,
    options: &ThresholdSweepOptions,
) -> Result<ThresholdSweepReport> {
    anyhow::ensure!(
        options.folds >= 2,
        "validate-threshold-sweep requires at least two folds"
    );
    let positives = load_fingerprint_dir(positive_dir)
        .with_context(|| format!("loading positives from {}", positive_dir.display()))?;
    anyhow::ensure!(
        positives.len() >= options.folds,
        "need at least {} positive fingerprints, got {}",
        options.folds,
        positives.len()
    );
    let negatives = negative_dir
        .map(|dir| {
            load_fingerprint_dir(dir)
                .with_context(|| format!("loading negatives from {}", dir.display()))
        })
        .transpose()?
        .unwrap_or_default();

    let mut runs = Vec::with_capacity(options.config_paths.len());
    for config_path in &options.config_paths {
        let config = load_config_from_path(config_path)
            .with_context(|| format!("loading threshold config {}", config_path.display()))?;
        let policy = detection_policy_from_match_config(&config.matching);
        let evaluation_mode = if options.evaluate_all_stages {
            MatchEvaluationMode::ExhaustiveDiagnostics
        } else {
            MatchEvaluationMode::ShortCircuit
        };
        let tier1_subset = validate_tier1_subset(&positives, &negatives, &policy);
        runs.push(ThresholdSweepRun {
            config: ThresholdSweepConfigReport {
                path: config_path.display().to_string(),
            },
            tier1_subset,
            kfold: validate_kfold_in_memory(
                &positives,
                &negatives,
                &policy,
                options.folds,
                options.seed,
                options.positive_mode,
                evaluation_mode,
            ),
            leave_one_out: validate_leave_one_out_in_memory(
                &positives,
                &negatives,
                &policy,
                options.positive_mode,
                evaluation_mode,
            ),
        });
    }

    Ok(ThresholdSweepReport {
        mode: "threshold_sweep",
        positive_fingerprint_dir: positive_dir.display().to_string(),
        negative_fingerprint_dir: negative_dir.map(|dir| dir.display().to_string()),
        positive_count: positives.len(),
        negative_count: negatives.len(),
        folds: options.folds,
        seed: options.seed,
        positive_mode: options.positive_mode.as_str(),
        evaluate_all_stages: options.evaluate_all_stages,
        runs,
    })
}

fn validate_tier1_subset(
    positives: &[LoadedFingerprint],
    negatives: &[LoadedFingerprint],
    policy: &crate::configuration::guild::DetectionPolicy,
) -> Tier1SubsetReport {
    let tier1_policy = matcher::confirmed_tier1_policy(policy);
    let mut checked_candidates = 0usize;
    let mut tier1_confirmed_candidates = 0usize;
    let mut violations = Vec::new();

    for held_out_index in 0..positives.len() {
        let train_indices = (0..positives.len())
            .filter(|index| *index != held_out_index)
            .collect::<Vec<_>>();
        let active_matcher = matcher_from_indices(positives, &train_indices);
        checked_candidates += 1;
        check_tier1_subset_candidate(
            &active_matcher,
            &positives[held_out_index],
            policy,
            &tier1_policy,
            &mut tier1_confirmed_candidates,
            &mut violations,
        );
    }

    if !negatives.is_empty() {
        let train_indices = (0..positives.len()).collect::<Vec<_>>();
        let active_matcher = matcher_from_indices(positives, &train_indices);
        for negative in negatives {
            checked_candidates += 1;
            check_tier1_subset_candidate(
                &active_matcher,
                negative,
                policy,
                &tier1_policy,
                &mut tier1_confirmed_candidates,
                &mut violations,
            );
        }
    }

    Tier1SubsetReport {
        checked_candidates,
        tier1_confirmed_candidates,
        violation_count: violations.len(),
        violations,
    }
}

fn check_tier1_subset_candidate(
    active_matcher: &matcher::Matcher,
    candidate: &LoadedFingerprint,
    full_policy: &crate::configuration::guild::DetectionPolicy,
    tier1_policy: &crate::configuration::guild::DetectionPolicy,
    tier1_confirmed_candidates: &mut usize,
    violations: &mut Vec<Tier1SubsetViolation>,
) {
    let tier1_candidate = tier1_fingerprint_projection(&candidate.fingerprint);
    let Some(tier1_outcome) = active_matcher.find_for_policy(&tier1_candidate, tier1_policy) else {
        return;
    };
    if tier1_outcome.suspicious {
        return;
    }
    *tier1_confirmed_candidates += 1;

    let full_outcome = active_matcher.find_for_policy(&candidate.fingerprint, full_policy);
    if full_outcome
        .as_ref()
        .is_none_or(|outcome| outcome.suspicious)
    {
        violations.push(Tier1SubsetViolation {
            source: candidate.source_path.clone(),
            tier1_specimen_id: tier1_outcome.specimen_id,
            full_outcome: full_outcome.map(|outcome| {
                format!(
                    "{}:{}",
                    if outcome.suspicious {
                        "suspicious"
                    } else {
                        "confirmed"
                    },
                    outcome.specimen_id
                )
            }),
        });
    }
}

fn tier1_fingerprint_projection(fingerprint: &ImageFingerprint) -> ImageFingerprint {
    let mut tier1 = fingerprint.clone();
    tier1.visual.text_grid = vec![0; 64];
    tier1.local_anchors.clear();
    tier1.local_hashes.clear();
    tier1
}

fn validate_kfold_in_memory(
    positives: &[LoadedFingerprint],
    negatives: &[LoadedFingerprint],
    policy: &crate::configuration::guild::DetectionPolicy,
    folds: usize,
    seed: u64,
    positive_mode: PositivePredictionMode,
    evaluation_mode: MatchEvaluationMode,
) -> ValidationSummary {
    let positive_folds = split_index_folds(shuffled_indices(positives.len(), seed), folds);
    let negative_folds = split_index_folds(
        shuffled_indices(negatives.len(), seed ^ 0x9E37_79B9_7F4A_7C15),
        folds,
    );
    let all_positive_sources =
        sources_for_indices(positives, &(0..positives.len()).collect::<Vec<_>>());
    let all_negative_sources =
        sources_for_indices(negatives, &(0..negatives.len()).collect::<Vec<_>>());

    let mut confusion = ValidationConfusion::default();
    let mut aggregate_sets = AggregatePredictionSets::default();

    for fold_index in 0..folds {
        let train_indices = positive_folds
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != fold_index)
            .flat_map(|(_, fold)| fold.iter().copied())
            .collect::<Vec<_>>();
        let matcher = matcher_from_indices(positives, &train_indices);
        let positive_test_indices = &positive_folds[fold_index];
        let negative_test_indices = negative_folds
            .get(fold_index)
            .map_or(&[][..], std::vec::Vec::as_slice);

        let positive_sources = sources_for_indices(positives, positive_test_indices);
        let positive_categories = classify_indices(
            &matcher,
            policy,
            positives,
            positive_test_indices,
            evaluation_mode,
        );
        let positive_predictions = positive_categories.predicted_sources(positive_mode);
        for source in &positive_sources {
            if positive_predictions.contains(source) {
                confusion.tp += 1;
            } else {
                confusion.false_negative += 1;
            }
        }
        add_positive_sets(&mut aggregate_sets, &positive_sources, &positive_categories);

        let negative_sources = sources_for_indices(negatives, negative_test_indices);
        let negative_categories = classify_indices(
            &matcher,
            policy,
            negatives,
            negative_test_indices,
            evaluation_mode,
        );
        let negative_predictions = negative_categories.predicted_sources(positive_mode);
        for source in &negative_sources {
            if negative_predictions.contains(source) {
                confusion.fp += 1;
            } else {
                confusion.tn += 1;
            }
        }
        add_negative_sets(&mut aggregate_sets, &negative_sources, &negative_categories);
    }

    validation_summary(
        positives.len(),
        negatives.len(),
        confusion,
        &aggregate_sets,
        &all_positive_sources,
        &all_negative_sources,
        false,
    )
}

fn validate_leave_one_out_in_memory(
    positives: &[LoadedFingerprint],
    negatives: &[LoadedFingerprint],
    policy: &crate::configuration::guild::DetectionPolicy,
    positive_mode: PositivePredictionMode,
    evaluation_mode: MatchEvaluationMode,
) -> ValidationSummary {
    let all_positive_sources =
        sources_for_indices(positives, &(0..positives.len()).collect::<Vec<_>>());
    let all_negative_sources =
        sources_for_indices(negatives, &(0..negatives.len()).collect::<Vec<_>>());
    let mut confusion = ValidationConfusion::default();
    let mut aggregate_sets = AggregatePredictionSets::default();

    for held_out_index in 0..positives.len() {
        let train_indices = (0..positives.len())
            .filter(|index| *index != held_out_index)
            .collect::<Vec<_>>();
        let matcher = matcher_from_indices(positives, &train_indices);
        let positive_test_indices = [held_out_index];
        let negative_test_indices = (0..negatives.len()).collect::<Vec<_>>();

        let positive_sources = sources_for_indices(positives, &positive_test_indices);
        let positive_categories = classify_indices(
            &matcher,
            policy,
            positives,
            &positive_test_indices,
            evaluation_mode,
        );
        let positive_predictions = positive_categories.predicted_sources(positive_mode);
        for source in &positive_sources {
            if positive_predictions.contains(source) {
                confusion.tp += 1;
            } else {
                confusion.false_negative += 1;
            }
        }
        add_positive_sets(&mut aggregate_sets, &positive_sources, &positive_categories);

        let negative_sources = sources_for_indices(negatives, &negative_test_indices);
        let negative_categories = classify_indices(
            &matcher,
            policy,
            negatives,
            &negative_test_indices,
            evaluation_mode,
        );
        let negative_predictions = negative_categories.predicted_sources(positive_mode);
        for source in &negative_sources {
            if negative_predictions.contains(source) {
                confusion.fp += 1;
            } else {
                confusion.tn += 1;
            }
        }
        add_negative_sets(&mut aggregate_sets, &negative_sources, &negative_categories);
    }

    validation_summary(
        positives.len(),
        negatives.len(),
        confusion,
        &aggregate_sets,
        &all_positive_sources,
        &all_negative_sources,
        true,
    )
}

fn matcher_from_indices(fingerprints: &[LoadedFingerprint], indices: &[usize]) -> matcher::Matcher {
    let specimens = indices
        .iter()
        .map(|index| fingerprints[*index].clone())
        .collect::<Vec<_>>();
    matcher::Matcher::new(specimen_records_from_loaded(&specimens, "local"))
}

fn classify_indices(
    matcher: &matcher::Matcher,
    policy: &crate::configuration::guild::DetectionPolicy,
    fingerprints: &[LoadedFingerprint],
    indices: &[usize],
    evaluation_mode: MatchEvaluationMode,
) -> PredictionCategories {
    let mut categories = PredictionCategories::default();
    for index in indices {
        let candidate = &fingerprints[*index];
        if let Some(outcome) = matcher
            .explain_for_policy_with_mode(&candidate.fingerprint, policy, evaluation_mode)
            .outcome
        {
            if outcome.suspicious {
                categories.suspicious.insert(candidate.source_path.clone());
            } else {
                categories.matched.insert(candidate.source_path.clone());
            }
        }
    }
    categories
}

fn add_positive_sets(
    aggregate: &mut AggregatePredictionSets,
    sources: &BTreeSet<String>,
    categories: &PredictionCategories,
) {
    aggregate
        .positive_matched
        .extend(sources.intersection(&categories.matched).cloned());
    aggregate.positive_suspicious_only.extend(
        sources
            .intersection(&categories.suspicious)
            .filter(|&source| !categories.matched.contains(source))
            .cloned(),
    );
    aggregate.positive_unmatched.extend(
        sources
            .iter()
            .filter(|&source| {
                !categories.matched.contains(source) && !categories.suspicious.contains(source)
            })
            .cloned(),
    );
}

fn add_negative_sets(
    aggregate: &mut AggregatePredictionSets,
    sources: &BTreeSet<String>,
    categories: &PredictionCategories,
) {
    aggregate
        .negative_matched
        .extend(sources.intersection(&categories.matched).cloned());
    aggregate.negative_suspicious_only.extend(
        sources
            .intersection(&categories.suspicious)
            .filter(|&source| !categories.matched.contains(source))
            .cloned(),
    );
}

fn validation_summary(
    positive_count: usize,
    negative_count: usize,
    confusion: ValidationConfusion,
    sets: &AggregatePredictionSets,
    all_positive_sources: &BTreeSet<String>,
    _all_negative_sources: &BTreeSet<String>,
    include_specimen_aliases: bool,
) -> ValidationSummary {
    let fully_matched_positive_sources =
        sorted_set(sets.positive_matched.intersection(all_positive_sources));
    let suspicious_only_positive_sources = sorted_set(
        sets.positive_suspicious_only
            .intersection(all_positive_sources),
    );
    let unmatched_positive_sources =
        sorted_set(sets.positive_unmatched.intersection(all_positive_sources));
    let fully_matched_false_positive_sources = sorted_owned_set(&sets.negative_matched);
    let suspicious_only_false_positive_sources = sorted_owned_set(&sets.negative_suspicious_only);

    ValidationSummary {
        positive_count,
        negative_count,
        aggregate: validation_metrics(confusion),
        fully_matched_positive_sources_count: fully_matched_positive_sources.len(),
        fully_matched_positive_sources: fully_matched_positive_sources.clone(),
        suspicious_only_positive_sources_count: suspicious_only_positive_sources.len(),
        suspicious_only_positive_sources: suspicious_only_positive_sources.clone(),
        unmatched_positive_sources_count: unmatched_positive_sources.len(),
        unmatched_positive_sources: unmatched_positive_sources.clone(),
        fully_matched_specimens_count: include_specimen_aliases
            .then_some(fully_matched_positive_sources.len()),
        fully_matched_specimens: include_specimen_aliases.then_some(fully_matched_positive_sources),
        suspicious_only_specimens_count: include_specimen_aliases
            .then_some(suspicious_only_positive_sources.len()),
        suspicious_only_specimens: include_specimen_aliases
            .then_some(suspicious_only_positive_sources),
        unmatched_specimens_count: include_specimen_aliases
            .then_some(unmatched_positive_sources.len()),
        unmatched_specimens: include_specimen_aliases.then_some(unmatched_positive_sources),
        fully_matched_false_positive_sources_count: fully_matched_false_positive_sources.len(),
        fully_matched_false_positive_sources,
        suspicious_only_false_positive_sources_count: suspicious_only_false_positive_sources.len(),
        suspicious_only_false_positive_sources,
    }
}

fn validation_metrics(confusion: ValidationConfusion) -> ValidationMetrics {
    let total = confusion.tp + confusion.fp + confusion.tn + confusion.false_negative;
    let precision = safe_ratio(confusion.tp, confusion.tp + confusion.fp);
    let recall = safe_ratio(confusion.tp, confusion.tp + confusion.false_negative);
    let specificity = safe_ratio(confusion.tn, confusion.tn + confusion.fp);
    let accuracy = safe_ratio(confusion.tp + confusion.tn, total);
    let f1 = precision.zip(recall).and_then(|(precision, recall)| {
        (precision + recall > 0.0).then(|| 2.0 * precision * recall / (precision + recall))
    });
    ValidationMetrics {
        tp: confusion.tp,
        fp: confusion.fp,
        tn: confusion.tn,
        false_negative: confusion.false_negative,
        total,
        accuracy,
        precision,
        recall,
        specificity,
        f1,
        false_positive_rate: safe_ratio(confusion.fp, confusion.fp + confusion.tn),
        false_negative_rate: safe_ratio(
            confusion.false_negative,
            confusion.false_negative + confusion.tp,
        ),
    }
}

fn safe_ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn sources_for_indices(fingerprints: &[LoadedFingerprint], indices: &[usize]) -> BTreeSet<String> {
    indices
        .iter()
        .map(|index| fingerprints[*index].source_path.clone())
        .collect()
}

fn sorted_set<'a>(values: impl Iterator<Item = &'a String>) -> Vec<String> {
    values.cloned().collect()
}

fn sorted_owned_set(values: &BTreeSet<String>) -> Vec<String> {
    values.iter().cloned().collect()
}

fn split_index_folds(indices: Vec<usize>, folds: usize) -> Vec<Vec<usize>> {
    let mut buckets = vec![Vec::new(); folds];
    for (position, index) in indices.into_iter().enumerate() {
        buckets[position % folds].push(index);
    }
    buckets
}

fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices = (0..len).collect::<Vec<_>>();
    let mut state = seed ^ 0xA076_1D64_78BD_642F;
    for index in (1..indices.len()).rev() {
        let random = splitmix64_next(&mut state);
        indices.swap(index, (random as usize) % (index + 1));
    }
    indices
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

impl PredictionCategories {
    fn predicted_sources(&self, mode: PositivePredictionMode) -> BTreeSet<String> {
        match mode {
            PositivePredictionMode::Matched => self.matched.clone(),
            PositivePredictionMode::MatchedOrSuspicious => self
                .matched
                .union(&self.suspicious)
                .cloned()
                .collect::<BTreeSet<_>>(),
        }
    }
}
pub(super) fn parse_threshold_sweep_options(args: &[String]) -> Result<ThresholdSweepOptions> {
    let mut options = ThresholdSweepOptions {
        config_paths: Vec::new(),
        folds: 5,
        seed: 42,
        positive_mode: PositivePredictionMode::MatchedOrSuspicious,
        evaluate_all_stages: false,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--config" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--config requires a value"))?;
                options.config_paths.push(PathBuf::from(value));
            }
            "--folds" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--folds requires a value"))?;
                options.folds = parse_positive_usize(value, "--folds")?;
            }
            "--seed" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--seed requires a value"))?;
                options.seed = value
                    .parse::<u64>()
                    .with_context(|| "--seed must be a number")?;
            }
            "--positive-mode" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--positive-mode requires a value"))?;
                options.positive_mode = PositivePredictionMode::parse(value)?;
            }
            "--evaluate-all-stages" => {
                options.evaluate_all_stages = true;
            }
            unknown => {
                return Err(anyhow!(
                    "unknown validate-threshold-sweep option {unknown}; usage: discord-sightline validate-threshold-sweep <positive-fingerprint-dir> <negative-fingerprint-dir-or-> --config CONFIG [--config CONFIG...] [--folds N] [--seed N] [--positive-mode matched|matched-or-suspicious] [--evaluate-all-stages]"
                ));
            }
        }
        index += 1;
    }
    anyhow::ensure!(
        !options.config_paths.is_empty(),
        "validate-threshold-sweep requires at least one --config"
    );
    anyhow::ensure!(
        options.folds >= 2,
        "validate-threshold-sweep requires at least two folds"
    );
    Ok(options)
}

pub(super) fn negative_fingerprint_dir_arg(value: &str) -> Option<&Path> {
    (!matches!(value, "-" | "")).then(|| Path::new(value))
}
