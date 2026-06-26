#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    reason = "Matcher scoring and geometry code uses compact numeric representations and a few large orchestration functions; clippy reports if these suppressions stop being needed."
)]

use crate::{
    bot::ledger::SpecimenRecord,
    configuration::{
        app::MatchConfig,
        guild::{DetectionPolicy, DetectionThreshold},
    },
    image::{
        geometry::{
            Accept as GeoAccept, Correspondence, GeoCfg, GeoMatch, GeometryScratch,
            Model as GeoModel, P, passes as geo_passes, verify_geometry_with_scratch,
        },
        knn::{
            ClusterScorer, CoherenceGraph, CoherenceGraphBuilder, Decision as ClusterDecision,
            HardActReason, Match as ClusterMatch, SpecimenId, Thresholds as ClusterThresholds,
        },
        matcher_opt::{self, hamming, hex16_to_u64},
        types::{
            FingerprintRepresentation, GeometryModel, ImageAnchor, ImageFingerprint,
            ImageVisualSignature, LocalImageHash, MatchConfidence, MatchDiagnostics, MatchOutcome,
            MatchStepDiagnostic,
        },
    },
};
use im::HashMap as PersistentHashMap;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::Serialize;
use std::sync::Arc;

const DENSE_LOCAL_CANDIDATE_SCAN_CAP_PER_SCALE: usize = 512;
const DENSE_LOCAL_MAX_BUCKET_SIZE: usize = 1024;
const DENSE_LOCAL_MAX_CANDIDATE_PAIR_BUDGET: usize = 32_768;
const DENSE_LOCAL_VERIFICATION_CANDIDATES: usize = 8;
const ANCHOR_MAX_BUCKET_SIZE: usize = 1024;
const ANCHOR_MAX_CANDIDATE_PAIR_BUDGET: usize = 32_768;
const LOCAL_VERIFICATION_CANDIDATES: usize = 32;
const LOCAL_GEOMETRY_ALTERNATES_PER_ANCHOR: usize = 3;
const LOCAL_ANCHOR_CANDIDATES_PER_REFERENCE_CAP: usize = 128;
const CLUSTER_GRAPH_BUILD_FLOOR: u32 = 1;
const CLUSTER_GRAPH_MAX_SPECIMENS: usize = 512;
const CLUSTER_GRAPH_MAX_PAIR_EVALUATIONS: usize = 100_000;

fn hex32_to_u128(value: &str) -> Option<u128> {
    if value.len() != 32 {
        return None;
    }
    u128::from_str_radix(value, 16).ok()
}

#[derive(Debug, Clone, Serialize)]
pub struct FingerprintComparison {
    pub exact_xxh128: bool,
    pub phash64_distance: Option<u32>,
    pub dhash64_distance: Option<u32>,
    pub perceptual_match: bool,
    pub suspicious_perceptual_match: bool,
    pub local_anchor_match: LocalAnchorComparison,
    pub matched: bool,
    pub suspicious: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalAnchorComparison {
    pub matched: bool,
    pub suspicious: bool,
    pub hits: usize,
    pub distinct_regions: usize,
    pub average_distance: Option<f32>,
    pub layout_spread: Option<f32>,
    pub mean_residual: Option<f32>,
    pub scale: Option<f32>,
    pub angle: Option<f32>,
    pub geometry_model: Option<GeometryModel>,
}

impl LocalAnchorComparison {
    fn miss() -> Self {
        Self {
            matched: false,
            suspicious: false,
            hits: 0,
            distinct_regions: 0,
            average_distance: None,
            layout_spread: None,
            mean_residual: None,
            scale: None,
            angle: None,
            geometry_model: None,
        }
    }
}

pub fn compare_fingerprints(
    specimen: &ImageFingerprint,
    candidate: &ImageFingerprint,
    thresholds: &MatchConfig,
) -> FingerprintComparison {
    let exact_xxh128 = specimen.byte_xxh128 == candidate.byte_xxh128;
    let phash64_distance = hex16_to_u64(&specimen.phash64)
        .zip(hex16_to_u64(&candidate.phash64))
        .map(|(left, right)| hamming(left, right));
    let dhash64_distance = hex16_to_u64(&specimen.dhash64)
        .zip(hex16_to_u64(&candidate.dhash64))
        .map(|(left, right)| hamming(left, right));
    let specimen_geometry = FingerprintGeometry::from_dimensions(specimen.width, specimen.height);
    let candidate_geometry =
        FingerprintGeometry::from_dimensions(candidate.width, candidate.height);
    let perceptual_match = geometry_compatible_for_match_config(
        specimen_geometry,
        candidate_geometry,
        thresholds,
        false,
    ) && phash64_distance.zip(dhash64_distance).is_some_and(
        |(phash_distance, dhash_distance)| {
            perceptual_hash_compatible_for_match_config(
                phash_distance,
                dhash_distance,
                thresholds,
                false,
            )
        },
    );
    let suspicious_perceptual_match = !perceptual_match
        && geometry_compatible_for_match_config(
            specimen_geometry,
            candidate_geometry,
            thresholds,
            true,
        )
        && phash64_distance.zip(dhash64_distance).is_some_and(
            |(phash_distance, dhash_distance)| {
                perceptual_hash_compatible_for_match_config(
                    phash_distance,
                    dhash_distance,
                    thresholds,
                    true,
                )
            },
        );
    let raw_local_anchor_match =
        compare_local_anchors(&specimen.local_anchors, &candidate.local_hashes, thresholds);
    let local_visual_match = geometry_compatible_for_match_config(
        specimen_geometry,
        candidate_geometry,
        thresholds,
        false,
    );
    let suspicious_local_visual_match = geometry_compatible_for_match_config(
        specimen_geometry,
        candidate_geometry,
        thresholds,
        true,
    );
    let local_anchor_match = LocalAnchorComparison {
        matched: raw_local_anchor_match.matched && local_visual_match,
        suspicious: !(raw_local_anchor_match.matched && local_visual_match)
            && (raw_local_anchor_match.matched || raw_local_anchor_match.suspicious)
            && suspicious_local_visual_match,
        hits: raw_local_anchor_match.hits,
        distinct_regions: raw_local_anchor_match.distinct_regions,
        average_distance: raw_local_anchor_match.average_distance,
        layout_spread: raw_local_anchor_match.layout_spread,
        mean_residual: raw_local_anchor_match.mean_residual,
        scale: raw_local_anchor_match.scale,
        angle: raw_local_anchor_match.angle,
        geometry_model: raw_local_anchor_match.geometry_model,
    };
    let suspicious = suspicious_perceptual_match || local_anchor_match.suspicious;

    FingerprintComparison {
        exact_xxh128,
        phash64_distance,
        dhash64_distance,
        perceptual_match,
        suspicious_perceptual_match,
        matched: exact_xxh128 || perceptual_match || local_anchor_match.matched,
        local_anchor_match,
        suspicious,
    }
}

pub fn compare_local_anchors(
    anchors: &[ImageAnchor],
    candidate_hashes: &[LocalImageHash],
    thresholds: &MatchConfig,
) -> LocalAnchorComparison {
    let required_anchors = thresholds
        .local_min_anchor_hits
        .min(thresholds.local_suspicious_min_anchor_hits);
    if anchors.len() < required_anchors || candidate_hashes.is_empty() {
        return LocalAnchorComparison::miss();
    }

    let anchors = anchors
        .iter()
        .filter_map(ParsedAnchor::from_anchor)
        .collect::<Vec<_>>();
    let candidate_hashes = candidate_hashes
        .iter()
        .enumerate()
        .map(|(index, hash)| ParsedLocalHash::from_local_hash(index, hash))
        .collect::<Vec<_>>();

    let strict = LocalThresholds::from_match_config(thresholds, false);
    let matched = compare_local_anchors_with_threshold(&anchors, &candidate_hashes, strict);
    if matched.matched {
        return matched;
    }

    let suspicious_limits = LocalThresholds::from_match_config(thresholds, true);
    let suspicious =
        compare_local_anchors_with_threshold(&anchors, &candidate_hashes, suspicious_limits);
    LocalAnchorComparison {
        matched: false,
        suspicious: suspicious.matched,
        hits: suspicious.hits,
        distinct_regions: suspicious.distinct_regions,
        average_distance: suspicious.average_distance,
        layout_spread: suspicious.layout_spread,
        mean_residual: suspicious.mean_residual,
        scale: suspicious.scale,
        angle: suspicious.angle,
        geometry_model: suspicious.geometry_model,
    }
}

pub fn confirmed_tier1_policy(policy: &DetectionPolicy) -> DetectionPolicy {
    let mut tier1 = policy.clone();
    tier1.confirmed.threshold.exact_xxh128 = false;
    tier1.confirmed.threshold.local_anchors = false;
    tier1.confirmed.threshold.local_unverified_support = false;
    tier1.confirmed.threshold.visual_shape = false;
    tier1.confirmed.threshold.cluster_coherence = false;
    tier1.suspicious.threshold.exact_xxh128 = false;
    tier1.suspicious.threshold.perceptual_hash = false;
    tier1.suspicious.threshold.local_anchors = false;
    tier1.suspicious.threshold.local_unverified_support = false;
    tier1.suspicious.threshold.visual_shape = false;
    tier1.suspicious.threshold.cluster_coherence = false;
    tier1
}

#[derive(Debug, Clone)]
struct IndexedSpecimen {
    record: SpecimenRecord,
    byte_xxh128: Option<u128>,
    phash64: Option<u64>,
    dhash64: Option<u64>,
    visual: ImageVisualSignature,
    geometry: FingerprintGeometry,
    anchors: Vec<ParsedAnchor>,
    dense_local_anchors: Vec<ParsedLocalHash>,
}

#[derive(Debug, Clone)]
struct IndexedSpecimenParts {
    byte_xxh128: Option<u128>,
    phash64: Option<u64>,
    dhash64: Option<u64>,
    visual: ImageVisualSignature,
    geometry: FingerprintGeometry,
    anchors: Vec<ParsedAnchor>,
    dense_local_anchors: Vec<ParsedLocalHash>,
}

#[derive(Debug, Clone, Copy)]
struct IndexedSpecimenPartsInput<'a> {
    width: u32,
    height: u32,
    byte_xxh128: &'a str,
    phash64: &'a str,
    dhash64: &'a str,
    visual: &'a ImageVisualSignature,
    anchors: &'a [ImageAnchor],
    local_hashes: &'a [LocalImageHash],
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum MatchVariant {
    Original,
    DiscordPreview,
}

impl MatchVariant {
    const fn representation(self) -> FingerprintRepresentation {
        match self {
            Self::Original => FingerprintRepresentation::Original,
            Self::DiscordPreview => FingerprintRepresentation::DiscordPreview,
        }
    }
}

#[derive(Debug, Clone)]
struct ParsedAnchor {
    hash: u64,
    hash2: u64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    region: u32,
    max_distance: u32,
    pos_x: u8,
    pos_y: u8,
    luma_mean: u8,
    luma_std: u8,
    edge_density: u8,
}

#[derive(Debug, Clone)]
pub struct Matcher {
    specimens: Vec<IndexedSpecimen>,
    preview_specimens: Vec<IndexedSpecimen>,
    xxh128_index: HashMap<u128, usize>,
    preview_xxh128_index: HashMap<u128, usize>,
    phash_segment_index: matcher_opt::FlatSegmentIndex<usize>,
    dhash_segment_index: matcher_opt::FlatSegmentIndex<usize>,
    preview_phash_segment_index: matcher_opt::FlatSegmentIndex<usize>,
    preview_dhash_segment_index: matcher_opt::FlatSegmentIndex<usize>,
    anchor_segment_index: matcher_opt::FlatSegmentIndex<IndexedAnchorRef>,
    preview_anchor_segment_index: matcher_opt::FlatSegmentIndex<IndexedAnchorRef>,
    dense_local_segment_index: matcher_opt::FlatSegmentIndex<IndexedDenseLocalRef>,
    preview_dense_local_segment_index: matcher_opt::FlatSegmentIndex<IndexedDenseLocalRef>,
    specimen_id_index: HashMap<String, SpecimenId>,
    preview_specimen_id_index: HashMap<String, SpecimenId>,
    coherence_graph: CoherenceGraph,
    preview_coherence_graph: CoherenceGraph,
    coherence_threshold: DetectionThreshold,
}

#[derive(Debug, Clone, Default)]
pub struct ExactHashIndex {
    by_xxh128: PersistentHashMap<u128, Arc<ExactHashSpecimen>>,
    by_specimen_id: PersistentHashMap<String, u128>,
}

impl ExactHashIndex {
    pub fn new(records: &[SpecimenRecord]) -> Self {
        let mut index = Self::default();
        for record in records {
            index.add_record(record);
        }
        index
    }

    pub fn add_record(&mut self, record: &SpecimenRecord) {
        let specimen = Arc::new(ExactHashSpecimen::new(record));
        let Some(byte_xxh128) = hex32_to_u128(&record.image.byte_xxh128) else {
            return;
        };
        self.by_specimen_id
            .insert(specimen.specimen_id.clone(), byte_xxh128);
        self.by_xxh128.insert(byte_xxh128, specimen);
    }

    pub fn remove_specimen(&mut self, specimen_id: &str) -> bool {
        let Some(byte_xxh128) = self.by_specimen_id.remove(specimen_id) else {
            return false;
        };
        self.by_xxh128.remove(&byte_xxh128).is_some()
    }

    pub fn contains_byte_xxh128(&self, byte_xxh128: &str) -> bool {
        hex32_to_u128(byte_xxh128).is_some_and(|key| self.by_xxh128.contains_key(&key))
    }

    pub fn find_for_policy(
        &self,
        byte_xxh128: &str,
        policy: &DetectionPolicy,
    ) -> Option<MatchOutcome> {
        for (threshold, suspicious, name) in [
            (&policy.confirmed.threshold, false, "confirmed"),
            (&policy.suspicious.threshold, true, "suspicious"),
        ] {
            if threshold.exact_xxh128
                && let Some(outcome) = self.find(byte_xxh128, suspicious, name)
            {
                return Some(outcome);
            }
        }
        None
    }

    fn find(
        &self,
        byte_xxh128: &str,
        suspicious: bool,
        threshold_name: &'static str,
    ) -> Option<MatchOutcome> {
        let specimen = self.by_xxh128.get(&hex32_to_u128(byte_xxh128)?)?;
        let mut diagnostics = match_diagnostics_for_exact_hash_specimen(
            specimen,
            FingerprintRepresentation::Original,
        );
        diagnostics.steps.push(exact_xxh128_step(
            threshold_name,
            true,
            Some(specimen.specimen_id.clone()),
            Some(1),
            None,
        ));
        Some(exact_hash_outcome(specimen, suspicious, diagnostics))
    }
}

#[derive(Debug, Clone)]
struct ExactHashSpecimen {
    specimen_id: String,
    visual: ImageVisualSignature,
    text_grid_stats: TextGridStats,
    geometry: FingerprintGeometry,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MatchEvaluationMode {
    ShortCircuit,
    ExhaustiveDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchExplanation {
    pub outcome: Option<MatchOutcome>,
    pub diagnostics: MatchDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatcherIndexStats {
    pub specimen_count: usize,
    pub preview_specimen_count: usize,
    pub phash_buckets: BucketOccupancyStats,
    pub dhash_buckets: BucketOccupancyStats,
    pub anchor_buckets: BucketOccupancyStats,
    pub dense_local_buckets: BucketOccupancyStats,
    pub preview_phash_buckets: BucketOccupancyStats,
    pub preview_dhash_buckets: BucketOccupancyStats,
    pub preview_anchor_buckets: BucketOccupancyStats,
    pub preview_dense_local_buckets: BucketOccupancyStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct BucketOccupancyStats {
    pub bucket_count: usize,
    pub entry_count: usize,
    pub min: usize,
    pub max: usize,
    pub avg: f64,
    pub p50: usize,
    pub p90: usize,
    pub p95: usize,
    pub p99: usize,
}

#[derive(Debug, Clone, Copy)]
struct IndexedAnchorRef {
    specimen_index: usize,
    anchor_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct IndexedDenseLocalRef {
    specimen_index: usize,
    dense_local_index: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct CandidateIndexStats {
    sampled_buckets: usize,
    pair_budget_exhausted: bool,
}

#[derive(Debug, Clone, Default)]
struct CandidateSelection {
    indices: Vec<usize>,
    stats: CandidateIndexStats,
}

#[derive(Default)]
struct MatchEvaluationCache {
    original_local_selection: Option<CandidateSelection>,
    preview_local_selection: Option<CandidateSelection>,
    original_anchor_hit_filter: Option<LocalFeatureFilter>,
    preview_anchor_hit_filter: Option<LocalFeatureFilter>,
    original_anchor_hits: HashMap<usize, Vec<AnchorHit>>,
    preview_anchor_hits: HashMap<usize, Vec<AnchorHit>>,
    geometry_scratch: GeometryScratch,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct LocalFeatureFilter {
    luma: u8,
    contrast: u8,
    edge_density: u8,
    position: u8,
}

#[derive(Debug, Default)]
struct IndexedLocalMatches {
    matches: Vec<(usize, LocalAnchorComparison)>,
    support_matches: Vec<(usize, LocalSupportComparison)>,
    selected_count: usize,
    stats: CandidateIndexStats,
}

#[derive(Debug)]
struct LocalSupportComparison {
    comparison: LocalAnchorComparison,
    raw_hits: usize,
}

#[derive(Debug, Clone, Copy)]
struct ThresholdSearch<'a> {
    threshold: &'a DetectionThreshold,
    suspicious: bool,
    name: &'static str,
    variant: MatchVariant,
    visual_shape: Option<VisualShapeEvidence>,
}

#[derive(Debug, Clone, Copy)]
struct ThresholdInput<'a> {
    threshold: &'a DetectionThreshold,
    suspicious: bool,
    name: &'static str,
    variant: MatchVariant,
}

#[derive(Debug, Clone, Copy)]
enum LocalMatchStage {
    Anchors,
    DenseLocalAnchors,
}

impl LocalMatchStage {
    const fn step(self, suspicious: bool) -> &'static str {
        match (self, suspicious) {
            (Self::Anchors, _) => "local_anchors",
            (Self::DenseLocalAnchors, false) => "dense_local_anchors",
            (Self::DenseLocalAnchors, true) => "suspicious_dense_local_anchors",
        }
    }

    const fn miss_reason(self) -> &'static str {
        match self {
            Self::Anchors => "no_anchor_candidate_met_threshold",
            Self::DenseLocalAnchors => "no_dense_local_candidate_met_threshold",
        }
    }

    const fn confidence(self, suspicious: bool) -> MatchConfidence {
        match (self, suspicious) {
            (Self::Anchors, false) => MatchConfidence::LocalAnchors,
            (Self::Anchors, true) => MatchConfidence::SuspiciousLocalAnchors,
            (Self::DenseLocalAnchors, false) => MatchConfidence::DenseLocalAnchors,
            (Self::DenseLocalAnchors, true) => MatchConfidence::SuspiciousDenseLocalAnchors,
        }
    }
}

fn index_specimen(
    specimen: &IndexedSpecimen,
    specimen_index: usize,
    phash_index: &mut matcher_opt::FlatSegmentIndex<usize>,
    dhash_index: &mut matcher_opt::FlatSegmentIndex<usize>,
    anchor_index: &mut matcher_opt::FlatSegmentIndex<IndexedAnchorRef>,
    dense_local_index: &mut matcher_opt::FlatSegmentIndex<IndexedDenseLocalRef>,
) {
    if let Some(phash64) = specimen.phash64 {
        for slot in matcher_opt::hamming_segments_flat(phash64) {
            phash_index.push(slot, specimen_index);
        }
    }
    if let Some(dhash64) = specimen.dhash64 {
        for slot in matcher_opt::hamming_segments_flat(dhash64) {
            dhash_index.push(slot, specimen_index);
        }
    }
    for (anchor_position, anchor) in specimen.anchors.iter().enumerate() {
        for slot in matcher_opt::hamming_segments_flat(anchor.hash) {
            anchor_index.push(
                slot,
                IndexedAnchorRef {
                    specimen_index,
                    anchor_index: anchor_position,
                },
            );
        }
    }
    for (dense_position, dense_anchor) in specimen.dense_local_anchors.iter().enumerate() {
        if dense_anchor.rotation_degrees != 0 {
            continue;
        }
        for slot in matcher_opt::dense_local_segments_flat(dense_anchor.hash) {
            dense_local_index.push(
                slot,
                IndexedDenseLocalRef {
                    specimen_index,
                    dense_local_index: dense_position,
                },
            );
        }
    }
}

impl Default for Matcher {
    fn default() -> Self {
        Self {
            specimens: Vec::new(),
            preview_specimens: Vec::new(),
            xxh128_index: HashMap::default(),
            preview_xxh128_index: HashMap::default(),
            phash_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::HAMMING_FLAT_SLOTS,
            ),
            dhash_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::HAMMING_FLAT_SLOTS,
            ),
            preview_phash_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::HAMMING_FLAT_SLOTS,
            ),
            preview_dhash_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::HAMMING_FLAT_SLOTS,
            ),
            anchor_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::HAMMING_FLAT_SLOTS,
            ),
            preview_anchor_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::HAMMING_FLAT_SLOTS,
            ),
            dense_local_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::DENSE_LOCAL_FLAT_SLOTS,
            ),
            preview_dense_local_segment_index: matcher_opt::FlatSegmentIndex::with_slots(
                matcher_opt::DENSE_LOCAL_FLAT_SLOTS,
            ),
            specimen_id_index: HashMap::default(),
            preview_specimen_id_index: HashMap::default(),
            coherence_graph: CoherenceGraph::default(),
            preview_coherence_graph: CoherenceGraph::default(),
            coherence_threshold: DetectionPolicy::default().suspicious.threshold,
        }
    }
}

impl Matcher {
    pub fn new(specimens: Vec<SpecimenRecord>) -> Self {
        Self::new_with_policy(specimens, &DetectionPolicy::default())
    }

    pub fn new_with_policy(specimens: Vec<SpecimenRecord>, policy: &DetectionPolicy) -> Self {
        let mut matcher = Self {
            specimens: specimens.into_iter().map(IndexedSpecimen::new).collect(),
            coherence_threshold: policy.suspicious.threshold.clone(),
            ..Self::default()
        };
        matcher.rebuild_indexes();
        matcher
    }

    pub fn add_with_policy(&mut self, specimen: SpecimenRecord, policy: Option<&DetectionPolicy>) {
        if let Some(policy) = policy {
            self.set_coherence_policy(policy);
        }
        let specimen = IndexedSpecimen::new(specimen);
        let index = self.specimens.len();
        if let Some(byte_xxh128) = specimen.byte_xxh128 {
            self.xxh128_index.insert(byte_xxh128, index);
        }
        self.specimen_id_index
            .insert(specimen.record.specimen_id.clone(), index as SpecimenId);
        index_specimen(
            &specimen,
            index,
            &mut self.phash_segment_index,
            &mut self.dhash_segment_index,
            &mut self.anchor_segment_index,
            &mut self.dense_local_segment_index,
        );
        self.specimens.push(specimen);
        self.coherence_graph = append_coherence_graph(
            &self.coherence_graph,
            &self.specimens,
            index,
            &self.coherence_threshold,
        );

        if let Some(preview) = IndexedSpecimen::new_preview(&self.specimens[index].record) {
            let preview_index = self.preview_specimens.len();
            if let Some(byte_xxh128) = preview.byte_xxh128 {
                self.preview_xxh128_index.insert(byte_xxh128, preview_index);
            }
            self.preview_specimen_id_index.insert(
                preview.record.specimen_id.clone(),
                preview_index as SpecimenId,
            );
            index_specimen(
                &preview,
                preview_index,
                &mut self.preview_phash_segment_index,
                &mut self.preview_dhash_segment_index,
                &mut self.preview_anchor_segment_index,
                &mut self.preview_dense_local_segment_index,
            );
            self.preview_specimens.push(preview);
            self.preview_coherence_graph = append_coherence_graph(
                &self.preview_coherence_graph,
                &self.preview_specimens,
                preview_index,
                &self.coherence_threshold,
            );
        }
    }

    pub fn len(&self) -> usize {
        self.specimens.len()
    }

    pub fn records(&self) -> Vec<SpecimenRecord> {
        self.specimens
            .iter()
            .map(|specimen| specimen.record.clone())
            .collect()
    }

    pub fn index_stats(&self) -> MatcherIndexStats {
        MatcherIndexStats {
            specimen_count: self.specimens.len(),
            preview_specimen_count: self.preview_specimens.len(),
            phash_buckets: bucket_occupancy_stats(self.phash_segment_index.slot_lens()),
            dhash_buckets: bucket_occupancy_stats(self.dhash_segment_index.slot_lens()),
            anchor_buckets: bucket_occupancy_stats(self.anchor_segment_index.slot_lens()),
            dense_local_buckets: bucket_occupancy_stats(self.dense_local_segment_index.slot_lens()),
            preview_phash_buckets: bucket_occupancy_stats(
                self.preview_phash_segment_index.slot_lens(),
            ),
            preview_dhash_buckets: bucket_occupancy_stats(
                self.preview_dhash_segment_index.slot_lens(),
            ),
            preview_anchor_buckets: bucket_occupancy_stats(
                self.preview_anchor_segment_index.slot_lens(),
            ),
            preview_dense_local_buckets: bucket_occupancy_stats(
                self.preview_dense_local_segment_index.slot_lens(),
            ),
        }
    }

    pub fn remove_specimen(&mut self, specimen_id: &str) -> bool {
        let Some(index) = self.specimen_id_index.get(specimen_id).copied() else {
            return false;
        };
        let Ok(index) = usize::try_from(index) else {
            return false;
        };
        if index >= self.specimens.len() {
            return false;
        }
        self.specimens.remove(index);
        self.rebuild_indexes();
        true
    }

    pub fn set_coherence_policy(&mut self, policy: &DetectionPolicy) {
        let threshold = policy.suspicious.threshold.clone();
        if threshold == self.coherence_threshold {
            return;
        }
        self.coherence_threshold = threshold;
        self.coherence_graph = build_coherence_graph(&self.specimens, &self.coherence_threshold);
        self.preview_coherence_graph =
            build_coherence_graph(&self.preview_specimens, &self.coherence_threshold);
    }

    fn rebuild_indexes(&mut self) {
        self.xxh128_index.clear();
        self.preview_xxh128_index.clear();
        self.phash_segment_index.clear();
        self.dhash_segment_index.clear();
        self.anchor_segment_index.clear();
        self.preview_specimens.clear();
        self.preview_phash_segment_index.clear();
        self.preview_dhash_segment_index.clear();
        self.preview_anchor_segment_index.clear();
        self.dense_local_segment_index.clear();
        self.preview_dense_local_segment_index.clear();
        self.specimen_id_index.clear();
        self.preview_specimen_id_index.clear();

        for (index, specimen) in self.specimens.iter().enumerate() {
            if let Some(byte_xxh128) = specimen.byte_xxh128 {
                self.xxh128_index.insert(byte_xxh128, index);
            }
            self.specimen_id_index
                .insert(specimen.record.specimen_id.clone(), index as SpecimenId);
            index_specimen(
                specimen,
                index,
                &mut self.phash_segment_index,
                &mut self.dhash_segment_index,
                &mut self.anchor_segment_index,
                &mut self.dense_local_segment_index,
            );
            if let Some(preview) = IndexedSpecimen::new_preview(&specimen.record) {
                let preview_index = self.preview_specimens.len();
                if let Some(byte_xxh128) = preview.byte_xxh128 {
                    self.preview_xxh128_index.insert(byte_xxh128, preview_index);
                }
                self.preview_specimen_id_index.insert(
                    preview.record.specimen_id.clone(),
                    preview_index as SpecimenId,
                );
                index_specimen(
                    &preview,
                    preview_index,
                    &mut self.preview_phash_segment_index,
                    &mut self.preview_dhash_segment_index,
                    &mut self.preview_anchor_segment_index,
                    &mut self.preview_dense_local_segment_index,
                );
                self.preview_specimens.push(preview);
            }
        }
        self.coherence_graph = build_coherence_graph(&self.specimens, &self.coherence_threshold);
        self.preview_coherence_graph =
            build_coherence_graph(&self.preview_specimens, &self.coherence_threshold);
    }

    pub fn find_for_policy(
        &self,
        image: &ImageFingerprint,
        policy: &DetectionPolicy,
    ) -> Option<MatchOutcome> {
        self.explain_for_policy_with_mode(image, policy, MatchEvaluationMode::ShortCircuit)
            .outcome
    }

    pub fn explain_for_policy_with_mode(
        &self,
        image: &ImageFingerprint,
        policy: &DetectionPolicy,
        mode: MatchEvaluationMode,
    ) -> MatchExplanation {
        self.explain_for_policy_variant(image, policy, MatchVariant::Original, mode)
    }

    pub fn find_preview_for_policy(
        &self,
        image: &ImageFingerprint,
        policy: &DetectionPolicy,
    ) -> Option<MatchOutcome> {
        self.explain_for_policy_variant(
            image,
            policy,
            MatchVariant::DiscordPreview,
            MatchEvaluationMode::ShortCircuit,
        )
        .outcome
    }

    fn explain_for_policy_variant(
        &self,
        image: &ImageFingerprint,
        policy: &DetectionPolicy,
        variant: MatchVariant,
        mode: MatchEvaluationMode,
    ) -> MatchExplanation {
        let candidate = ParsedCandidate::from_image(image);
        let mut diagnostics = match_diagnostics_for_candidate(&candidate, variant);
        let mut cache = MatchEvaluationCache::default();
        let mut outcome = if mode == MatchEvaluationMode::ShortCircuit {
            self.find_for_threshold_parsed(
                &candidate,
                ThresholdInput {
                    threshold: &policy.confirmed.threshold,
                    suspicious: false,
                    name: "confirmed",
                    variant,
                },
                mode,
                &mut diagnostics,
                &mut cache,
            )
            .or_else(|| {
                self.find_for_threshold_parsed(
                    &candidate,
                    ThresholdInput {
                        threshold: &policy.suspicious.threshold,
                        suspicious: true,
                        name: "suspicious",
                        variant,
                    },
                    mode,
                    &mut diagnostics,
                    &mut cache,
                )
            })
        } else {
            let confirmed = self.find_for_threshold_parsed(
                &candidate,
                ThresholdInput {
                    threshold: &policy.confirmed.threshold,
                    suspicious: false,
                    name: "confirmed",
                    variant,
                },
                mode,
                &mut diagnostics,
                &mut cache,
            );
            let suspicious = self.find_for_threshold_parsed(
                &candidate,
                ThresholdInput {
                    threshold: &policy.suspicious.threshold,
                    suspicious: true,
                    name: "suspicious",
                    variant,
                },
                mode,
                &mut diagnostics,
                &mut cache,
            );
            confirmed.or(suspicious)
        };
        if let Some(outcome) = &mut outcome {
            outcome.diagnostics = diagnostics.clone();
        }
        MatchExplanation {
            outcome,
            diagnostics,
        }
    }

    fn find_for_threshold_parsed(
        &self,
        candidate: &ParsedCandidate,
        input: ThresholdInput<'_>,
        mode: MatchEvaluationMode,
        diagnostics: &mut MatchDiagnostics,
        cache: &mut MatchEvaluationCache,
    ) -> Option<MatchOutcome> {
        let threshold = input.threshold;
        let suspicious = input.suspicious;
        let threshold_name = input.name;
        let variant = input.variant;
        let visual_shape = threshold
            .visual_shape
            .then(|| visual_shape_evidence(candidate, threshold))
            .flatten();
        let search = ThresholdSearch {
            threshold,
            suspicious,
            name: threshold_name,
            variant,
            visual_shape,
        };
        let mut cluster_scorer = threshold.cluster_coherence.then(|| {
            ClusterScorer::new(ClusterThresholds::new(
                threshold.cluster_chrome_ceiling_score,
                threshold.cluster_hard_score,
                threshold.cluster_member_score,
                threshold.cluster_coverage_floor_permille,
                threshold.cluster_coherence_score,
                threshold.cluster_min_size,
            ))
        });
        if threshold.exact_xxh128 {
            if let Some(specimen) = self.exact_specimen_by_xxh128(candidate.byte_xxh128, variant) {
                diagnostics.steps.push(exact_xxh128_step(
                    threshold_name,
                    true,
                    Some(specimen.record.specimen_id.clone()),
                    Some(1),
                    None,
                ));
                let outcome = exact_outcome(specimen, suspicious, diagnostics.clone());
                return Some(outcome);
            }
            diagnostics.steps.push(exact_xxh128_step(
                threshold_name,
                false,
                None,
                Some(0),
                Some("xxh128_not_found"),
            ));
        }

        let mut stage_outcomes = Vec::new();

        if threshold.perceptual_hash {
            stage_outcomes.extend(self.find_perceptual_match(candidate, search, diagnostics));
            if mode == MatchEvaluationMode::ShortCircuit
                && !suspicious
                && let Some(outcome) = self.scored_threshold_outcome(
                    &stage_outcomes,
                    search,
                    diagnostics,
                    cluster_scorer.as_mut(),
                )
            {
                return Some(outcome);
            }
        }

        if threshold.local_anchors {
            stage_outcomes.extend(self.find_local_match(
                candidate,
                search,
                diagnostics,
                LocalMatchStage::Anchors,
                cache,
            ));
            if mode == MatchEvaluationMode::ShortCircuit
                && !suspicious
                && let Some(outcome) = self.scored_threshold_outcome(
                    &stage_outcomes,
                    search,
                    diagnostics,
                    cluster_scorer.as_mut(),
                )
            {
                return Some(outcome);
            }
        }

        if threshold.local_anchors {
            stage_outcomes.extend(self.find_local_match(
                candidate,
                search,
                diagnostics,
                LocalMatchStage::DenseLocalAnchors,
                cache,
            ));
            if mode == MatchEvaluationMode::ShortCircuit
                && !suspicious
                && let Some(outcome) = self.scored_threshold_outcome(
                    &stage_outcomes,
                    search,
                    diagnostics,
                    cluster_scorer.as_mut(),
                )
            {
                return Some(outcome);
            }
        }

        if threshold.visual_shape {
            record_visual_shape_diagnostic(candidate, search, diagnostics);
        }

        let _ = mode;
        self.scored_threshold_outcome(
            &stage_outcomes,
            search,
            diagnostics,
            cluster_scorer.as_mut(),
        )
    }

    fn find_perceptual_match(
        &self,
        candidate: &ParsedCandidate,
        search: ThresholdSearch<'_>,
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<MatchOutcome> {
        let threshold = search.threshold;
        let suspicious = search.suspicious;
        let threshold_name = search.name;
        let variant = search.variant;
        let visual_shape = search.visual_shape;
        let mut considered = 0usize;
        let mut best_step: Option<MatchStepDiagnostic> = None;
        let mut best_passed_step: Option<MatchStepDiagnostic> = None;
        let mut outcomes = Vec::new();
        for specimen in self
            .perceptual_candidates(
                candidate,
                threshold.phash64_max_distance,
                threshold.dhash64_max_distance,
                variant,
            )
            .filter_map(|index| self.variant_specimens(variant).get(index))
        {
            considered += 1;
            let (phash64_distance, dhash64_distance) = hash_distances(specimen, candidate);
            let hash_compatible = phash64_distance.zip(dhash64_distance).is_some_and(
                |(phash_distance, dhash_distance)| {
                    perceptual_hash_compatible_for_threshold(
                        phash_distance,
                        dhash_distance,
                        threshold,
                    )
                },
            );
            let geometry_compatible =
                fingerprint_geometry_compatible(specimen.geometry, candidate.geometry, threshold);
            let passed = hash_compatible && geometry_compatible;
            let visual_supported_near_miss = suspicious
                && visual_shape.is_some()
                && !passed
                && phash64_distance.zip(dhash64_distance).is_some_and(
                    |(phash_distance, dhash_distance)| {
                        perceptual_hash_visually_supported(
                            phash_distance,
                            dhash_distance,
                            hash_compatible,
                            geometry_compatible,
                            threshold,
                        )
                    },
                );
            let passed = passed || visual_supported_near_miss;
            let visual_score = (passed || visual_supported_near_miss)
                .then(|| visual_signature_score(&specimen.visual, &candidate.visual, threshold));
            let (visual_compatible, match_score) = if let Some(visual_score) = visual_score {
                (
                    Some(visual_score > 0.0),
                    phash64_distance.zip(dhash64_distance).map(|distances| {
                        stage_score_with_visual(
                            perceptual_score(distances, threshold),
                            visual_score,
                            threshold,
                        )
                    }),
                )
            } else {
                (None, None)
            };
            let mut step = MatchStepDiagnostic {
                threshold: threshold_name,
                step: "perceptual_hash",
                passed,
                reason: visual_supported_near_miss.then_some("visual_supported_near_miss"),
                specimen_id: None,
                candidates_considered: Some(considered),
                phash64_distance,
                dhash64_distance,
                geometry_compatible: Some(geometry_compatible),
                visual_compatible,
                local_anchor_hits: None,
                local_distinct_regions: None,
                local_average_distance: None,
                local_layout_spread: None,
                local_mean_residual: None,
                local_scale: None,
                local_angle: None,
                local_geometry_model: None,
                visual_shape_signals: visual_shape.map(|evidence| evidence.signals),
                visual_shape_score: visual_shape.map(|evidence| evidence.score),
                match_score,
            };

            if step.passed {
                if best_scored_step(&step, best_passed_step.as_ref()) {
                    step.specimen_id = Some(specimen.record.specimen_id.clone());
                    best_passed_step = Some(step);
                }
                outcomes.push(MatchOutcome {
                    specimen_id: specimen.record.specimen_id.clone(),
                    confidence: if suspicious {
                        MatchConfidence::SuspiciousPerceptual
                    } else {
                        MatchConfidence::Perceptual
                    },
                    suspicious,
                    match_score,
                    phash64_distance,
                    dhash64_distance,
                    local_anchor_hits: None,
                    local_distinct_regions: None,
                    local_average_distance: None,
                    local_geometry_model: None,
                    diagnostics: diagnostics_stub(diagnostics),
                });
                continue;
            }

            if best_perceptual_step(&step, best_step.as_ref()) {
                let visual_score =
                    visual_signature_score(&specimen.visual, &candidate.visual, threshold);
                step.visual_compatible = Some(visual_score > 0.0);
                step.match_score = phash64_distance.zip(dhash64_distance).map(|distances| {
                    stage_score_with_visual(
                        perceptual_score(distances, threshold),
                        visual_score,
                        threshold,
                    )
                });
                step.specimen_id = Some(specimen.record.specimen_id.clone());
                best_step = Some(step);
            }
        }
        if let Some(step) = best_passed_step {
            diagnostics.steps.push(step);
            return outcomes;
        }

        diagnostics
            .steps
            .push(best_step.unwrap_or(MatchStepDiagnostic {
                threshold: threshold_name,
                step: "perceptual_hash",
                passed: false,
                reason: Some("no_index_candidates"),
                specimen_id: None,
                candidates_considered: Some(considered),
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
            }));
        outcomes
    }

    fn find_local_match(
        &self,
        candidate: &ParsedCandidate,
        search: ThresholdSearch<'_>,
        diagnostics: &mut MatchDiagnostics,
        stage: LocalMatchStage,
        cache: &mut MatchEvaluationCache,
    ) -> Vec<MatchOutcome> {
        let threshold = search.threshold;
        let suspicious = search.suspicious;
        let threshold_name = search.name;
        let variant = search.variant;
        let visual_shape = search.visual_shape;
        let indexed = match stage {
            LocalMatchStage::Anchors => self.indexed_local_anchor_matches(
                candidate,
                threshold,
                variant,
                Some(self.cached_local_selection(candidate, variant, cache)),
                cache,
            ),
            LocalMatchStage::DenseLocalAnchors => {
                self.indexed_dense_local_anchor_matches(candidate, threshold, variant, cache)
            }
        };
        let selection_stats = indexed.stats;
        let selected_count = indexed.selected_count;
        let mut best_step: Option<MatchStepDiagnostic> = None;
        let mut best_passed_step: Option<MatchStepDiagnostic> = None;
        let mut outcomes = Vec::new();
        let considered = selected_count;
        for (specimen_index, local) in indexed.matches {
            let Some(specimen) = self.variant_specimens(variant).get(specimen_index) else {
                continue;
            };
            let geometry_compatible =
                fingerprint_geometry_compatible(specimen.geometry, candidate.geometry, threshold);
            let visual_score =
                visual_signature_score(&specimen.visual, &candidate.visual, threshold);
            let visual_compatible = visual_score > 0.0;
            let (phash64_distance, dhash64_distance) = hash_distances(specimen, candidate);
            let match_score = Some(stage_score_with_visual(
                local_anchor_score(&local, threshold),
                visual_score,
                threshold,
            ));
            let verified_local_transform = local.geometry_model.is_some();
            let passed = geometry_compatible || verified_local_transform;
            let mut step = MatchStepDiagnostic {
                threshold: threshold_name,
                step: stage.step(suspicious),
                passed,
                reason: (passed && !geometry_compatible).then_some("verified_local_geometry"),
                specimen_id: None,
                candidates_considered: Some(considered),
                phash64_distance,
                dhash64_distance,
                geometry_compatible: Some(geometry_compatible),
                visual_compatible: Some(visual_compatible),
                local_anchor_hits: Some(local.hits),
                local_distinct_regions: Some(local.distinct_regions),
                local_average_distance: local.average_distance,
                local_layout_spread: local.layout_spread,
                local_mean_residual: local.mean_residual,
                local_scale: local.scale,
                local_angle: local.angle,
                local_geometry_model: local.geometry_model,
                visual_shape_signals: visual_shape.map(|evidence| evidence.signals),
                visual_shape_score: visual_shape.map(|evidence| evidence.score),
                match_score,
            };
            if step.passed {
                if best_scored_step(&step, best_passed_step.as_ref()) {
                    step.specimen_id = Some(specimen.record.specimen_id.clone());
                    best_passed_step = Some(step);
                }
                outcomes.push(MatchOutcome {
                    specimen_id: specimen.record.specimen_id.clone(),
                    confidence: stage.confidence(suspicious),
                    suspicious,
                    match_score,
                    phash64_distance,
                    dhash64_distance,
                    local_anchor_hits: Some(local.hits),
                    local_distinct_regions: Some(local.distinct_regions),
                    local_average_distance: local.average_distance,
                    local_geometry_model: local.geometry_model,
                    diagnostics: diagnostics_stub(diagnostics),
                });
                continue;
            }
            if best_local_step(&step, best_step.as_ref()) {
                step.specimen_id = Some(specimen.record.specimen_id.clone());
                best_step = Some(step);
            }
        }
        if suspicious
            && matches!(stage, LocalMatchStage::Anchors)
            && let Some(visual_shape) = visual_shape
        {
            for (specimen_index, local) in indexed.support_matches {
                let Some(specimen) = self.variant_specimens(variant).get(specimen_index) else {
                    continue;
                };
                if !local_unverified_support_passes(specimen, candidate, &local, threshold) {
                    continue;
                }
                let local_comparison = &local.comparison;
                let visual_score =
                    visual_signature_score(&specimen.visual, &candidate.visual, threshold);
                let (phash64_distance, dhash64_distance) = hash_distances(specimen, candidate);
                let match_score = Some(stage_score_with_visual(
                    local_anchor_score(local_comparison, threshold),
                    visual_score,
                    threshold,
                ));
                let step = MatchStepDiagnostic {
                    threshold: threshold_name,
                    step: stage.step(true),
                    passed: true,
                    reason: Some("unverified_local_support"),
                    specimen_id: Some(specimen.record.specimen_id.clone()),
                    candidates_considered: Some(considered),
                    phash64_distance,
                    dhash64_distance,
                    geometry_compatible: Some(true),
                    visual_compatible: Some(visual_score > 0.0),
                    local_anchor_hits: Some(local_comparison.hits),
                    local_distinct_regions: Some(local_comparison.distinct_regions),
                    local_average_distance: local_comparison.average_distance,
                    local_layout_spread: local_comparison.layout_spread,
                    local_mean_residual: local_comparison.mean_residual,
                    local_scale: local_comparison.scale,
                    local_angle: local_comparison.angle,
                    local_geometry_model: local_comparison.geometry_model,
                    visual_shape_signals: Some(visual_shape.signals),
                    visual_shape_score: Some(visual_shape.score),
                    match_score,
                };
                if best_scored_step(&step, best_passed_step.as_ref()) {
                    best_passed_step = Some(step.clone());
                }
                outcomes.push(MatchOutcome {
                    specimen_id: specimen.record.specimen_id.clone(),
                    confidence: MatchConfidence::SuspiciousLocalAnchors,
                    suspicious: true,
                    match_score,
                    phash64_distance,
                    dhash64_distance,
                    local_anchor_hits: Some(local_comparison.hits),
                    local_distinct_regions: Some(local_comparison.distinct_regions),
                    local_average_distance: local_comparison.average_distance,
                    local_geometry_model: local_comparison.geometry_model,
                    diagnostics: diagnostics_stub(diagnostics),
                });
            }
        }
        if let Some(step) = best_passed_step {
            diagnostics.steps.push(step);
            return outcomes;
        }

        diagnostics
            .steps
            .push(best_step.unwrap_or(MatchStepDiagnostic {
                threshold: threshold_name,
                step: stage.step(suspicious),
                passed: false,
                reason: Some(local_candidate_miss_reason(
                    selection_stats,
                    selected_count,
                    stage.miss_reason(),
                )),
                specimen_id: None,
                candidates_considered: Some(considered),
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
            }));
        outcomes
    }

    fn perceptual_candidates(
        &self,
        candidate: &ParsedCandidate,
        phash_max_distance: u32,
        dhash_max_distance: u32,
        variant: MatchVariant,
    ) -> Box<dyn Iterator<Item = usize> + '_> {
        let specimens = self.variant_specimens(variant);

        if phash_max_distance as u8 >= matcher_opt::HAMMING_INDEX_SEGMENTS
            && dhash_max_distance as u8 >= matcher_opt::HAMMING_INDEX_SEGMENTS
        {
            return Box::new(0..specimens.len());
        }

        if phash_max_distance <= dhash_max_distance
            && let Some(phash64) = candidate.phash64
            && (phash_max_distance as u8) < matcher_opt::HAMMING_INDEX_SEGMENTS
        {
            let index = self.variant_phash_index(variant);
            return Box::new(Self::segment_index_candidates(
                phash64,
                index,
                specimens.len(),
            ));
        }

        if let Some(dhash64) = candidate.dhash64
            && (dhash_max_distance as u8) < matcher_opt::HAMMING_INDEX_SEGMENTS
        {
            let index = self.variant_dhash_index(variant);
            return Box::new(Self::segment_index_candidates(
                dhash64,
                index,
                specimens.len(),
            ));
        }

        let Some(phash64) = candidate.phash64 else {
            return Box::new(std::iter::empty());
        };
        let index = self.variant_phash_index(variant);
        Box::new(Self::segment_index_candidates(
            phash64,
            index,
            specimens.len(),
        ))
    }

    fn segment_index_candidates(
        hash: u64,
        index: &matcher_opt::FlatSegmentIndex<usize>,
        specimen_count: usize,
    ) -> std::vec::IntoIter<usize> {
        let mut indices = Vec::new();
        let mut seen = HashSet::default();
        for slot in matcher_opt::hamming_segments_flat(hash) {
            for &specimen_index in index.get(slot) {
                debug_assert!(specimen_index < specimen_count);
                if seen.insert(specimen_index) {
                    indices.push(specimen_index);
                }
            }
        }
        indices.into_iter()
    }

    fn indexed_local_anchor_matches(
        &self,
        candidate: &ParsedCandidate,
        threshold: &DetectionThreshold,
        variant: MatchVariant,
        selection: Option<CandidateSelection>,
        cache: &mut MatchEvaluationCache,
    ) -> IndexedLocalMatches {
        if candidate.local_hashes.is_empty()
            || threshold.min_anchor_hits == 0
            || threshold.max_mean_distance.is_nan()
        {
            return IndexedLocalMatches::default();
        }
        let limits = LocalThresholds::from_detection_threshold(threshold);
        let selection = selection.unwrap_or_else(|| {
            self.local_candidate_specimens(candidate, variant, LOCAL_VERIFICATION_CANDIDATES)
        });
        if selection.indices.is_empty() {
            return IndexedLocalMatches {
                matches: Vec::new(),
                support_matches: Vec::new(),
                selected_count: 0,
                stats: selection.stats,
            };
        }

        let selected_count = selection.indices.len();
        let mut matches = Vec::new();
        let mut support_matches = Vec::new();
        for specimen_index in selection.indices {
            let Some(specimen) = self.variant_specimens(variant).get(specimen_index) else {
                continue;
            };
            let hits = Self::cached_verified_anchor_hits(
                specimen_index,
                specimen,
                candidate,
                limits,
                variant,
                cache,
            );
            if let Some(comparison) =
                verified_local_comparison(&hits, limits, &mut cache.geometry_scratch)
            {
                matches.push((specimen_index, comparison));
            } else if threshold.local_unverified_support
                && let Some(comparison) =
                    unverified_local_support_comparison(&hits, threshold.geometry_ratio_min_margin)
            {
                support_matches.push((specimen_index, comparison));
            }
        }
        IndexedLocalMatches {
            matches,
            support_matches,
            selected_count,
            stats: selection.stats,
        }
    }

    fn cached_local_selection(
        &self,
        candidate: &ParsedCandidate,
        variant: MatchVariant,
        cache: &mut MatchEvaluationCache,
    ) -> CandidateSelection {
        let slot = match variant {
            MatchVariant::Original => &mut cache.original_local_selection,
            MatchVariant::DiscordPreview => &mut cache.preview_local_selection,
        };
        slot.get_or_insert_with(|| {
            self.local_candidate_specimens(candidate, variant, LOCAL_VERIFICATION_CANDIDATES)
        })
        .clone()
    }

    fn cached_verified_anchor_hits(
        specimen_index: usize,
        specimen: &IndexedSpecimen,
        candidate: &ParsedCandidate,
        limits: LocalThresholds,
        variant: MatchVariant,
        cache: &mut MatchEvaluationCache,
    ) -> Vec<AnchorHit> {
        let filter = LocalFeatureFilter::from(limits);
        let (active_filter, hits) = match variant {
            MatchVariant::Original => (
                &mut cache.original_anchor_hit_filter,
                &mut cache.original_anchor_hits,
            ),
            MatchVariant::DiscordPreview => (
                &mut cache.preview_anchor_hit_filter,
                &mut cache.preview_anchor_hits,
            ),
        };
        match *active_filter {
            Some(existing) if existing == filter => {}
            Some(_) => {
                return collect_verified_anchor_hits(
                    &specimen.anchors,
                    &candidate.local_hashes,
                    limits,
                );
            }
            None => {
                *active_filter = Some(filter);
            }
        }
        hits.entry(specimen_index)
            .or_insert_with(|| {
                collect_verified_anchor_hits(&specimen.anchors, &candidate.local_hashes, limits)
            })
            .clone()
    }

    fn indexed_dense_local_anchor_matches(
        &self,
        candidate: &ParsedCandidate,
        threshold: &DetectionThreshold,
        variant: MatchVariant,
        cache: &mut MatchEvaluationCache,
    ) -> IndexedLocalMatches {
        if candidate.local_hashes.is_empty()
            || threshold.min_anchor_hits == 0
            || threshold.max_mean_distance.is_nan()
        {
            return IndexedLocalMatches::default();
        }
        let limits = dense_local_anchor_thresholds(threshold);
        let candidate_hashes = candidate_dense_local_hashes(&candidate.local_hashes);
        if candidate_hashes.is_empty() {
            return IndexedLocalMatches::default();
        }
        let selection = self.dense_local_candidate_specimens(
            &candidate_hashes,
            limits,
            variant,
            DENSE_LOCAL_VERIFICATION_CANDIDATES,
        );
        if selection.indices.is_empty() {
            return IndexedLocalMatches {
                matches: Vec::new(),
                support_matches: Vec::new(),
                selected_count: 0,
                stats: selection.stats,
            };
        }

        let selected_count = selection.indices.len();
        let matches = selection
            .indices
            .into_iter()
            .filter_map(|specimen_index| {
                let specimen = self.variant_specimens(variant).get(specimen_index)?;
                let hits = collect_verified_dense_local_hits(
                    &specimen.dense_local_anchors,
                    &candidate_hashes,
                    limits,
                );
                verified_local_comparison(&hits, limits, &mut cache.geometry_scratch)
                    .map(|comparison| (specimen_index, comparison))
            })
            .collect();
        IndexedLocalMatches {
            matches,
            support_matches: Vec::new(),
            selected_count,
            stats: selection.stats,
        }
    }

    fn local_candidate_specimens(
        &self,
        candidate: &ParsedCandidate,
        variant: MatchVariant,
        limit: usize,
    ) -> CandidateSelection {
        let mut best_anchor_matches: HashMap<u64, BestReferenceHit> = HashMap::default();
        let mut seen_candidate_anchor_pairs: HashSet<(u32, usize, usize)> = HashSet::default();
        let mut specimen_votes: HashMap<usize, u32> = HashMap::default();
        let mut seen_candidate_specimen_votes: HashSet<(u32, usize)> = HashSet::default();
        let mut stats = CandidateIndexStats::default();
        let anchor_index = self.variant_anchor_index(variant);
        let specimens = self.variant_specimens(variant);
        let mut query_buckets = Vec::new();
        for (candidate_index, candidate_hash) in candidate.local_hashes.iter().enumerate() {
            if candidate_hash.rotation_degrees != 0 {
                continue;
            }
            for slot in matcher_opt::hamming_segments_flat(candidate_hash.hash) {
                let bucket_len = anchor_index.get(slot).len();
                if bucket_len > 0 {
                    query_buckets.push((bucket_len, candidate_index, slot));
                }
            }
        }
        query_buckets.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        for (bucket_len, candidate_index, slot) in query_buckets {
            let candidate_hash = &candidate.local_hashes[candidate_index];
            let bucket = anchor_index.get(slot);
            debug_assert_eq!(bucket.len(), bucket_len);
            if bucket_len > ANCHOR_MAX_BUCKET_SIZE {
                stats.sampled_buckets += 1;
            }
            for anchor_ref in capped_bucket(bucket, ANCHOR_MAX_BUCKET_SIZE) {
                record_candidate_vote(
                    &mut specimen_votes,
                    &mut seen_candidate_specimen_votes,
                    candidate_hash.id,
                    anchor_ref.specimen_index,
                );
                if seen_candidate_anchor_pairs.len() >= ANCHOR_MAX_CANDIDATE_PAIR_BUDGET {
                    stats.pair_budget_exhausted = true;
                    return CandidateSelection {
                        indices: ranked_specimen_candidates(
                            best_anchor_matches,
                            specimen_votes,
                            limit,
                        ),
                        stats,
                    };
                }
                let pair_key = (
                    candidate_hash.id,
                    anchor_ref.specimen_index,
                    anchor_ref.anchor_index,
                );
                if !seen_candidate_anchor_pairs.insert(pair_key) {
                    continue;
                }
                let Some(anchor) = specimens
                    .get(anchor_ref.specimen_index)
                    .and_then(|specimen| specimen.anchors.get(anchor_ref.anchor_index))
                else {
                    continue;
                };
                let distance = descriptor_hamming(anchor, candidate_hash);
                let best_match = best_anchor_matches
                    .entry(reference_key(
                        anchor_ref.specimen_index,
                        anchor_ref.anchor_index,
                    ))
                    .or_insert_with(BestReferenceHit::empty);
                if distance > anchor.max_distance {
                    continue;
                }
                let Some(correspondence) = anchor_correspondence(anchor, candidate_hash, distance)
                else {
                    continue;
                };
                best_match.insert(AnchorHit {
                    distance,
                    quality: anchor_hit_quality(anchor, candidate_hash, distance),
                    correspondence,
                });
            }
        }

        CandidateSelection {
            indices: ranked_specimen_candidates(best_anchor_matches, specimen_votes, limit),
            stats,
        }
    }

    fn dense_local_candidate_specimens(
        &self,
        candidate_hashes: &[&ParsedLocalHash],
        limits: LocalThresholds,
        variant: MatchVariant,
        limit: usize,
    ) -> CandidateSelection {
        let mut best_dense_matches: HashMap<u64, BestReferenceHit> = HashMap::default();
        let mut seen_candidate_dense_pairs: HashSet<(u32, usize, usize)> = HashSet::default();
        let mut specimen_votes: HashMap<usize, u32> = HashMap::default();
        let mut seen_candidate_specimen_votes: HashSet<(u32, usize)> = HashSet::default();
        let mut stats = CandidateIndexStats::default();
        let dense_local_index = self.variant_dense_local_index(variant);
        let specimens = self.variant_specimens(variant);
        let mut query_buckets = Vec::new();
        for (candidate_index, candidate_hash) in candidate_hashes.iter().enumerate() {
            if candidate_hash.rotation_degrees != 0 {
                continue;
            }
            for slot in matcher_opt::dense_local_segments_flat(candidate_hash.hash) {
                let bucket_len = dense_local_index.get(slot).len();
                if bucket_len > 0 {
                    query_buckets.push((bucket_len, candidate_index, slot));
                }
            }
        }
        query_buckets.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        for (bucket_len, candidate_index, slot) in query_buckets {
            let candidate_hash = candidate_hashes[candidate_index];
            let bucket = dense_local_index.get(slot);
            debug_assert_eq!(bucket.len(), bucket_len);
            if bucket_len > DENSE_LOCAL_MAX_BUCKET_SIZE {
                stats.sampled_buckets += 1;
            }
            for dense_ref in capped_bucket(bucket, DENSE_LOCAL_MAX_BUCKET_SIZE) {
                record_candidate_vote(
                    &mut specimen_votes,
                    &mut seen_candidate_specimen_votes,
                    candidate_hash.id,
                    dense_ref.specimen_index,
                );
                if seen_candidate_dense_pairs.len() >= DENSE_LOCAL_MAX_CANDIDATE_PAIR_BUDGET {
                    stats.pair_budget_exhausted = true;
                    return CandidateSelection {
                        indices: ranked_specimen_candidates(
                            best_dense_matches,
                            specimen_votes,
                            limit,
                        ),
                        stats,
                    };
                }
                let pair_key = (
                    candidate_hash.id,
                    dense_ref.specimen_index,
                    dense_ref.dense_local_index,
                );
                if !seen_candidate_dense_pairs.insert(pair_key) {
                    continue;
                }
                let Some(dense_anchor) =
                    specimens
                        .get(dense_ref.specimen_index)
                        .and_then(|specimen| {
                            specimen
                                .dense_local_anchors
                                .get(dense_ref.dense_local_index)
                        })
                else {
                    continue;
                };
                let distance = hamming(dense_anchor.hash, candidate_hash.hash);
                let best_match = best_dense_matches
                    .entry(reference_key(
                        dense_ref.specimen_index,
                        dense_ref.dense_local_index,
                    ))
                    .or_insert_with(BestReferenceHit::empty);
                if distance > limits.max_distance {
                    best_match.observe_distance(distance, candidate_hash.physical_id);
                    continue;
                }
                let correspondence =
                    dense_local_correspondence(dense_anchor, candidate_hash, distance);
                best_match.insert(AnchorHit {
                    distance,
                    quality: dense_local_hit_quality(dense_anchor, candidate_hash, distance),
                    correspondence,
                });
            }
        }

        CandidateSelection {
            indices: ranked_specimen_candidates(best_dense_matches, specimen_votes, limit),
            stats,
        }
    }

    fn exact_specimen_by_xxh128(
        &self,
        byte_xxh128: Option<u128>,
        variant: MatchVariant,
    ) -> Option<&IndexedSpecimen> {
        let byte_xxh128 = byte_xxh128?;
        match variant {
            MatchVariant::Original => self
                .xxh128_index
                .get(&byte_xxh128)
                .and_then(|index| self.specimens.get(*index)),
            MatchVariant::DiscordPreview => self
                .preview_xxh128_index
                .get(&byte_xxh128)
                .and_then(|index| self.preview_specimens.get(*index)),
        }
    }

    fn variant_specimens(&self, variant: MatchVariant) -> &[IndexedSpecimen] {
        match variant {
            MatchVariant::Original => &self.specimens,
            MatchVariant::DiscordPreview => &self.preview_specimens,
        }
    }

    fn variant_phash_index(&self, variant: MatchVariant) -> &matcher_opt::FlatSegmentIndex<usize> {
        match variant {
            MatchVariant::Original => &self.phash_segment_index,
            MatchVariant::DiscordPreview => &self.preview_phash_segment_index,
        }
    }

    fn variant_dhash_index(&self, variant: MatchVariant) -> &matcher_opt::FlatSegmentIndex<usize> {
        match variant {
            MatchVariant::Original => &self.dhash_segment_index,
            MatchVariant::DiscordPreview => &self.preview_dhash_segment_index,
        }
    }

    fn variant_anchor_index(
        &self,
        variant: MatchVariant,
    ) -> &matcher_opt::FlatSegmentIndex<IndexedAnchorRef> {
        match variant {
            MatchVariant::Original => &self.anchor_segment_index,
            MatchVariant::DiscordPreview => &self.preview_anchor_segment_index,
        }
    }

    fn variant_dense_local_index(
        &self,
        variant: MatchVariant,
    ) -> &matcher_opt::FlatSegmentIndex<IndexedDenseLocalRef> {
        match variant {
            MatchVariant::Original => &self.dense_local_segment_index,
            MatchVariant::DiscordPreview => &self.preview_dense_local_segment_index,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VisualShapeEvidence {
    signals: usize,
    score: u32,
}

impl IndexedSpecimen {
    fn new(record: SpecimenRecord) -> Self {
        let parts = IndexedSpecimenParts::new(IndexedSpecimenPartsInput {
            width: record.image.width,
            height: record.image.height,
            byte_xxh128: &record.image.byte_xxh128,
            phash64: &record.image.phash64,
            dhash64: &record.image.dhash64,
            visual: &record.image.visual,
            anchors: &record.anchors,
            local_hashes: &record.local_hashes,
        });
        Self::from_parts(record, parts)
    }

    fn new_preview(record: &SpecimenRecord) -> Option<Self> {
        let preview = record.preview.as_ref()?;
        let parts = IndexedSpecimenParts::new(IndexedSpecimenPartsInput {
            width: preview.width,
            height: preview.height,
            byte_xxh128: &preview.byte_xxh128,
            phash64: &preview.phash64,
            dhash64: &preview.dhash64,
            visual: &preview.visual,
            anchors: &preview.anchors,
            local_hashes: &preview.local_hashes,
        });
        Some(Self::from_parts(record.clone(), parts))
    }

    fn from_parts(record: SpecimenRecord, parts: IndexedSpecimenParts) -> Self {
        Self {
            record,
            byte_xxh128: parts.byte_xxh128,
            phash64: parts.phash64,
            dhash64: parts.dhash64,
            visual: parts.visual,
            geometry: parts.geometry,
            anchors: parts.anchors,
            dense_local_anchors: parts.dense_local_anchors,
        }
    }
}

impl IndexedSpecimenParts {
    fn new(input: IndexedSpecimenPartsInput<'_>) -> Self {
        Self {
            byte_xxh128: hex32_to_u128(input.byte_xxh128),
            phash64: hex16_to_u64(input.phash64),
            dhash64: hex16_to_u64(input.dhash64),
            visual: input.visual.clone(),
            geometry: FingerprintGeometry::from_dimensions(input.width, input.height),
            anchors: input
                .anchors
                .iter()
                .filter_map(ParsedAnchor::from_anchor)
                .collect(),
            dense_local_anchors: input
                .local_hashes
                .iter()
                .enumerate()
                .map(|(index, hash)| ParsedLocalHash::from_local_hash(index, hash))
                .collect(),
        }
    }
}

impl ExactHashSpecimen {
    fn new(record: &SpecimenRecord) -> Self {
        let visual = record.image.visual.clone();
        let text_grid_stats = text_grid_stats(&visual.text_grid);
        let geometry =
            FingerprintGeometry::from_dimensions(record.image.width, record.image.height);
        Self {
            specimen_id: record.specimen_id.clone(),
            visual,
            text_grid_stats,
            geometry,
        }
    }
}

#[derive(Debug, Clone)]
struct ParsedCandidate {
    byte_xxh128: Option<u128>,
    phash64: Option<u64>,
    dhash64: Option<u64>,
    visual: ImageVisualSignature,
    text_grid_stats: TextGridStats,
    geometry: FingerprintGeometry,
    base_local_hash_count: usize,
    local_hashes: Vec<ParsedLocalHash>,
}

#[derive(Debug, Clone, Copy)]
struct FingerprintGeometry {
    width: u32,
    height: u32,
    short_edge: u32,
    area: u64,
    aspect: f32,
}

#[derive(Debug, Clone, Copy)]
struct GeometryLimits {
    min_short_edge: u32,
    min_area: u64,
    max_aspect_ratio: f32,
    max_aspect_delta: f32,
    max_width_delta: f32,
    max_height_delta: f32,
}

impl GeometryLimits {
    fn from_detection_threshold(threshold: &DetectionThreshold) -> Self {
        Self {
            min_short_edge: threshold.geometry_min_short_edge,
            min_area: threshold.geometry_min_area,
            max_aspect_ratio: threshold.geometry_max_aspect_ratio,
            max_aspect_delta: threshold.geometry_max_aspect_delta,
            max_width_delta: threshold.geometry_max_width_delta,
            max_height_delta: threshold.geometry_max_height_delta,
        }
    }

    fn for_candidate_shape(threshold: &DetectionThreshold) -> Self {
        Self {
            max_width_delta: 0.0,
            max_height_delta: 0.0,
            ..Self::from_detection_threshold(threshold)
        }
    }

    fn from_match_config(config: &MatchConfig, suspicious: bool) -> Self {
        if suspicious {
            Self {
                min_short_edge: config.suspicious_geometry_min_short_edge,
                min_area: config.suspicious_geometry_min_area,
                max_aspect_ratio: config.suspicious_geometry_max_aspect_ratio,
                max_aspect_delta: config.suspicious_geometry_max_aspect_delta,
                max_width_delta: config.suspicious_geometry_max_width_delta,
                max_height_delta: config.suspicious_geometry_max_height_delta,
            }
        } else {
            Self {
                min_short_edge: config.geometry_min_short_edge,
                min_area: config.geometry_min_area,
                max_aspect_ratio: config.geometry_max_aspect_ratio,
                max_aspect_delta: config.geometry_max_aspect_delta,
                max_width_delta: config.geometry_max_width_delta,
                max_height_delta: config.geometry_max_height_delta,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ParsedLocalHash {
    id: u32,
    physical_id: u32,
    hash: u64,
    hash2: u64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    region: u32,
    pos_x: u8,
    pos_y: u8,
    luma_mean: u8,
    luma_std: u8,
    edge_density: u8,
    scale_percent: u16,
    rotation_degrees: i16,
}

#[derive(Debug, Clone, Copy)]
struct AnchorHit {
    distance: u32,
    quality: u32,
    correspondence: Correspondence,
}

#[derive(Clone, Copy)]
struct ScanHit<'a> {
    distance: u32,
    quality: u32,
    cand: &'a ParsedLocalHash,
}

struct BestScanHit<'a> {
    best: Option<ScanHit<'a>>,
    second_distance: u32,
}

impl<'a> BestScanHit<'a> {
    fn empty() -> Self {
        Self {
            best: None,
            second_distance: u32::from(u8::MAX),
        }
    }

    fn observe_distance(&mut self, distance: u32, candidate_id: u32) {
        if let Some(best) = self.best
            && candidate_id != best.cand.physical_id
        {
            self.second_distance = self.second_distance.min(distance);
        }
    }

    fn insert(&mut self, hit: ScanHit<'a>) {
        match self.best {
            Some(best) if hit.quality < best.quality => {
                if hit.cand.physical_id != best.cand.physical_id {
                    self.second_distance = self.second_distance.min(best.distance);
                }
                self.best = Some(hit);
            }
            Some(_) => self.observe_distance(hit.distance, hit.cand.physical_id),
            None => self.best = Some(hit),
        }
    }

    fn into_dense_local_hit(self, dense_anchor: &ParsedLocalHash) -> Option<AnchorHit> {
        let best = self.best?;
        let mut correspondence = dense_local_correspondence(dense_anchor, best.cand, best.distance);
        correspondence.second_hamming = self.second_distance.min(u32::from(u8::MAX)) as u8;
        Some(AnchorHit {
            distance: best.distance,
            quality: best.quality,
            correspondence,
        })
    }
}

#[derive(Debug, Clone)]
struct BestReferenceHit {
    best: Option<AnchorHit>,
    second_distance: u32,
}

impl BestReferenceHit {
    fn empty() -> Self {
        Self {
            best: None,
            second_distance: u32::from(u8::MAX),
        }
    }

    fn observe_distance(&mut self, distance: u32, candidate_id: u32) {
        if self
            .best
            .is_none_or(|best| candidate_id != best.correspondence.cand_id)
        {
            self.second_distance = self.second_distance.min(distance);
        }
    }

    fn insert(&mut self, hit: AnchorHit) {
        match self.best {
            Some(best) if hit.quality < best.quality => {
                if hit.correspondence.cand_id != best.correspondence.cand_id {
                    self.second_distance = self.second_distance.min(best.distance);
                }
                self.best = Some(hit);
            }
            Some(_) => self.observe_distance(hit.distance, hit.correspondence.cand_id),
            None => self.best = Some(hit),
        }
    }
}

fn ranked_specimen_candidates(
    best_anchor_matches: HashMap<u64, BestReferenceHit>,
    specimen_votes: HashMap<usize, u32>,
    limit: usize,
) -> Vec<usize> {
    let mut per_specimen: HashMap<usize, (usize, u32, u64)> = HashMap::default();
    for (specimen_index, votes) in specimen_votes {
        per_specimen.entry(specimen_index).or_default().1 = votes;
    }
    for (key, best_match) in best_anchor_matches {
        if let Some(hit) = best_match.best {
            let specimen_index = reference_key_specimen(key);
            let entry = per_specimen.entry(specimen_index).or_default();
            entry.0 += 1;
            entry.2 = entry.2.saturating_add(u64::from(hit.quality));
        }
    }

    let mut ranked = per_specimen
        .into_iter()
        .map(|(specimen_index, (hits, votes, quality_sum))| {
            (specimen_index, hits, votes, quality_sum)
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(specimen_index, _hits, _votes, _quality_sum)| specimen_index)
        .collect()
}

fn record_candidate_vote(
    specimen_votes: &mut HashMap<usize, u32>,
    seen_votes: &mut HashSet<(u32, usize)>,
    candidate_id: u32,
    specimen_index: usize,
) {
    if seen_votes.insert((candidate_id, specimen_index)) {
        let entry = specimen_votes.entry(specimen_index).or_default();
        *entry = entry.saturating_add(1);
    }
}

fn reference_key(specimen_index: usize, reference_index: usize) -> u64 {
    ((specimen_index as u64) << 32) | reference_index as u64
}

fn reference_key_specimen(key: u64) -> usize {
    (key >> 32) as usize
}

fn capped_bucket<T>(bucket: &[T], cap: usize) -> impl Iterator<Item = &T> {
    let cap = cap.max(1);
    let stride = if bucket.len() <= cap {
        1
    } else {
        bucket.len().div_ceil(cap)
    };
    bucket.iter().step_by(stride).take(cap)
}

const fn local_candidate_miss_reason(
    stats: CandidateIndexStats,
    selected_count: usize,
    default_reason: &'static str,
) -> &'static str {
    if stats.pair_budget_exhausted {
        "candidate_pair_budget_exhausted"
    } else if selected_count > 0 {
        "selected_candidates_failed_verification"
    } else if stats.sampled_buckets > 0 {
        "sampled_common_buckets_no_match"
    } else {
        default_reason
    }
}

impl ParsedCandidate {
    fn from_image(image: &ImageFingerprint) -> Self {
        Self {
            byte_xxh128: hex32_to_u128(&image.byte_xxh128),
            phash64: hex16_to_u64(&image.phash64),
            dhash64: hex16_to_u64(&image.dhash64),
            visual: image.visual.clone(),
            text_grid_stats: text_grid_stats(&image.visual.text_grid),
            geometry: FingerprintGeometry::from_dimensions(image.width, image.height),
            base_local_hash_count: image
                .local_hashes
                .iter()
                .filter(|hash| hash.rotation_degrees == 0)
                .count(),
            local_hashes: image
                .local_hashes
                .iter()
                .enumerate()
                .map(|(index, hash)| ParsedLocalHash::from_local_hash(index, hash))
                .collect(),
        }
    }
}

impl FingerprintGeometry {
    fn from_dimensions(width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let short_edge = width.min(height);
        let long_edge = width.max(height);
        Self {
            width,
            height,
            short_edge,
            area: width as u64 * height as u64,
            aspect: long_edge as f32 / short_edge as f32,
        }
    }
}

impl ParsedAnchor {
    fn from_anchor(anchor: &ImageAnchor) -> Option<Self> {
        Some(Self {
            hash: hex16_to_u64(&anchor.hash)?,
            hash2: hex16_to_u64(&anchor.hash2)?,
            x: anchor.x,
            y: anchor.y,
            w: anchor.w,
            h: anchor.h,
            region: anchor.region,
            max_distance: anchor.max_distance,
            pos_x: anchor.pos_x,
            pos_y: anchor.pos_y,
            luma_mean: anchor.luma_mean,
            luma_std: anchor.luma_std,
            edge_density: anchor.edge_density,
        })
    }
}

impl ParsedLocalHash {
    fn from_local_hash(id: usize, hash: &LocalImageHash) -> Self {
        Self {
            id: id as u32,
            physical_id: local_hash_physical_id(hash),
            hash: hash.hash,
            hash2: hash.hash2,
            x: hash.x,
            y: hash.y,
            w: hash.w,
            h: hash.h,
            region: hash.region,
            pos_x: hash.pos_x,
            pos_y: hash.pos_y,
            luma_mean: hash.luma_mean,
            luma_std: hash.luma_std,
            edge_density: hash.edge_density,
            scale_percent: hash.scale_percent,
            rotation_degrees: hash.rotation_degrees,
        }
    }
}

fn local_hash_physical_id(hash: &LocalImageHash) -> u32 {
    let x = hash.x.min(0x0fff);
    let y = hash.y.min(0x0fff);
    let scale = u32::from(hash.scale_percent.min(0x00ff));
    (scale << 24) | (x << 12) | y
}

fn visual_shape_evidence(
    candidate: &ParsedCandidate,
    threshold: &DetectionThreshold,
) -> Option<VisualShapeEvidence> {
    if !fingerprint_geometry_compatible_for_limits(
        candidate.geometry,
        candidate.geometry,
        GeometryLimits::for_candidate_shape(threshold),
    ) {
        return None;
    }

    let mut signals = 0usize;
    let mut score = 0u32;
    let text_stats = candidate.text_grid_stats;
    if candidate.visual.luma_mean > threshold.visual_shape_max_luma_mean {
        return None;
    }
    if candidate.visual.luma_std > threshold.visual_shape_max_luma_std {
        return None;
    }
    if text_stats.mean > threshold.visual_shape_max_text_grid_mean {
        return None;
    }
    if text_stats.middle_percent < threshold.visual_shape_min_middle_text_percent
        || text_stats.center_percent < threshold.visual_shape_min_center_text_percent
        || text_stats.center_percent > threshold.visual_shape_max_center_text_percent
        || text_stats.edge_percent > threshold.visual_shape_max_edge_text_percent
    {
        return None;
    }
    if rgb_channel_spread(candidate.visual.rgb_mean) > threshold.visual_shape_max_rgb_spread {
        return None;
    }
    let base_local_hash_count = candidate.base_local_hash_count;
    let sparse_dark = candidate.visual.luma_mean <= threshold.visual_shape_sparse_max_luma_mean
        && text_stats.mean <= threshold.visual_shape_sparse_max_text_grid_mean
        && base_local_hash_count >= threshold.visual_shape_sparse_min_local_hashes;

    if text_stats.regions >= threshold.visual_shape_min_text_regions
        && text_stats.mean >= threshold.visual_shape_min_text_grid_mean
    {
        signals += 1;
        score += u32::from(text_stats.mean) + text_stats.regions as u32 * 8;
    } else if !sparse_dark {
        return None;
    }
    if (threshold.visual_shape_min_luma_std..=threshold.visual_shape_max_luma_std)
        .contains(&candidate.visual.luma_std)
    {
        signals += 1;
        score += candidate.visual.luma_std as u32;
    }
    if candidate.visual.luma_mean >= threshold.visual_shape_min_luma_mean
        && candidate.visual.luma_mean <= threshold.visual_shape_max_luma_mean
    {
        signals += 1;
        score += 40;
    }
    if base_local_hash_count >= threshold.visual_shape_min_local_hashes {
        signals += 1;
        score += (base_local_hash_count as u32).min(200);
    }

    if sparse_dark {
        signals += 1;
        score += 120 + (base_local_hash_count as u32).min(200);
    }

    (signals >= threshold.visual_shape_min_signals || sparse_dark)
        .then_some(VisualShapeEvidence { signals, score })
}

fn record_visual_shape_diagnostic(
    candidate: &ParsedCandidate,
    search: ThresholdSearch<'_>,
    diagnostics: &mut MatchDiagnostics,
) {
    let threshold = search.threshold;
    let threshold_name = search.name;
    let evidence = search.visual_shape;
    diagnostics.steps.push(visual_shape_diagnostic_step(
        candidate,
        threshold,
        threshold_name,
        evidence,
    ));
}

#[derive(Debug, Clone, Copy)]
struct TextGridStats {
    mean: u8,
    regions: usize,
    middle_percent: u8,
    center_percent: u8,
    edge_percent: u8,
}

fn text_grid_stats(grid: &[u8]) -> TextGridStats {
    if grid.is_empty() {
        return TextGridStats {
            mean: 0,
            regions: 0,
            middle_percent: 0,
            center_percent: 0,
            edge_percent: 0,
        };
    }
    let sum = grid.iter().map(|value| *value as u32).sum::<u32>();
    let total = sum.max(1);
    let mut row_sums = [0u32; 8];
    let mut col_sums = [0u32; 8];
    for (index, value) in grid.iter().copied().enumerate().take(64) {
        row_sums[index / 8] += value as u32;
        col_sums[index % 8] += value as u32;
    }
    let middle_sum = row_sums[2..6].iter().sum::<u32>();
    let center_sum = col_sums[2..6].iter().sum::<u32>();
    let left_edge_sum = col_sums[0..2].iter().sum::<u32>();
    let right_edge_sum = col_sums[6..8].iter().sum::<u32>();
    TextGridStats {
        mean: (sum / grid.len() as u32).min(u8::MAX as u32) as u8,
        regions: grid.iter().filter(|value| **value >= 64).count(),
        middle_percent: percent_u8(middle_sum, total),
        center_percent: percent_u8(center_sum, total),
        edge_percent: percent_u8(left_edge_sum.max(right_edge_sum), total),
    }
}

fn percent_u8(part: u32, total: u32) -> u8 {
    part.saturating_mul(100)
        .checked_div(total)
        .unwrap_or(0)
        .min(100) as u8
}

fn rgb_channel_spread(rgb_mean: [u8; 3]) -> u8 {
    let min = rgb_mean.iter().copied().min().unwrap_or(0);
    let max = rgb_mean.iter().copied().max().unwrap_or(0);
    max.saturating_sub(min)
}

fn hash_distances(
    specimen: &IndexedSpecimen,
    candidate: &ParsedCandidate,
) -> (Option<u32>, Option<u32>) {
    (
        specimen
            .phash64
            .zip(candidate.phash64)
            .map(|(left, right)| hamming(left, right)),
        specimen
            .dhash64
            .zip(candidate.dhash64)
            .map(|(left, right)| hamming(left, right)),
    )
}

fn exact_outcome(
    specimen: &IndexedSpecimen,
    suspicious: bool,
    diagnostics: MatchDiagnostics,
) -> MatchOutcome {
    MatchOutcome {
        specimen_id: specimen.record.specimen_id.clone(),
        confidence: MatchConfidence::ExactXxh128,
        suspicious,
        match_score: Some(10_000.0),
        phash64_distance: Some(0),
        dhash64_distance: Some(0),
        local_anchor_hits: None,
        local_distinct_regions: None,
        local_average_distance: None,
        local_geometry_model: None,
        diagnostics,
    }
}

fn exact_hash_outcome(
    specimen: &ExactHashSpecimen,
    suspicious: bool,
    diagnostics: MatchDiagnostics,
) -> MatchOutcome {
    MatchOutcome {
        specimen_id: specimen.specimen_id.clone(),
        confidence: MatchConfidence::ExactXxh128,
        suspicious,
        match_score: Some(10_000.0),
        phash64_distance: Some(0),
        dhash64_distance: Some(0),
        local_anchor_hits: None,
        local_distinct_regions: None,
        local_average_distance: None,
        local_geometry_model: None,
        diagnostics,
    }
}

fn match_diagnostics_for_exact_hash_specimen(
    specimen: &ExactHashSpecimen,
    representation: FingerprintRepresentation,
) -> MatchDiagnostics {
    let text_stats = specimen.text_grid_stats;
    MatchDiagnostics {
        representation,
        candidate_short_edge: specimen.geometry.short_edge,
        candidate_area: specimen.geometry.area,
        candidate_aspect: specimen.geometry.aspect,
        candidate_luma_mean: specimen.visual.luma_mean,
        candidate_luma_std: specimen.visual.luma_std,
        candidate_text_grid_mean: text_stats.mean,
        candidate_text_regions: text_stats.regions,
        candidate_local_hashes: 0,
        steps: Vec::new(),
    }
}

fn exact_xxh128_step(
    threshold: &'static str,
    passed: bool,
    specimen_id: Option<String>,
    candidates_considered: Option<usize>,
    reason: Option<&'static str>,
) -> MatchStepDiagnostic {
    MatchStepDiagnostic {
        threshold,
        step: "exact_xxh128",
        passed,
        reason,
        specimen_id,
        candidates_considered,
        phash64_distance: passed.then_some(0),
        dhash64_distance: passed.then_some(0),
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
        match_score: passed.then_some(10_000.0),
    }
}

impl Matcher {
    fn scored_threshold_outcome(
        &self,
        stage_outcomes: &[MatchOutcome],
        search: ThresholdSearch<'_>,
        diagnostics: &mut MatchDiagnostics,
        cluster_scorer: Option<&mut ClusterScorer>,
    ) -> Option<MatchOutcome> {
        let threshold = search.threshold;
        let visual_bonus = search
            .suspicious
            .then(|| {
                search
                    .visual_shape
                    .map(|evidence| visual_shape_score(evidence, threshold))
            })
            .flatten()
            .map_or(0.0, |score| {
                (score * threshold.visual_shape_score_weight).min(threshold.visual_shape_score_cap)
            });

        let mut per_specimen: HashMap<&str, ScoredOutcome> = HashMap::default();
        for outcome in stage_outcomes {
            let Some(stage_score) = outcome.match_score else {
                continue;
            };
            let weight = stage_weight(outcome.confidence, threshold);
            let entry = per_specimen
                .entry(outcome.specimen_id.as_str())
                .or_insert_with(|| ScoredOutcome {
                    score: visual_bonus,
                    outcome: outcome.clone(),
                });
            entry.score += stage_score * weight;
            if stage_score > entry.outcome.match_score.unwrap_or(f32::MIN) {
                entry.outcome = outcome.clone();
            }
        }

        if let Some(outcome) =
            self.cluster_coherence_outcome(&per_specimen, search, diagnostics, cluster_scorer)
        {
            return Some(outcome);
        }

        let best = per_specimen
            .into_values()
            .max_by(|left, right| left.score.total_cmp(&right.score));
        best.and_then(|mut scored| {
            (scored.score >= threshold.score_threshold).then(|| {
                scored.outcome.match_score = Some(scored.score);
                scored.outcome.diagnostics = diagnostics.clone();
                scored.outcome
            })
        })
    }

    fn cluster_coherence_outcome(
        &self,
        per_specimen: &HashMap<&str, ScoredOutcome>,
        search: ThresholdSearch<'_>,
        diagnostics: &mut MatchDiagnostics,
        cluster_scorer: Option<&mut ClusterScorer>,
    ) -> Option<MatchOutcome> {
        let threshold = search.threshold;
        let suspicious = search.suspicious;
        if !suspicious || !threshold.cluster_coherence {
            return None;
        }
        let id_index = self.variant_specimen_id_index(search.variant);
        let graph = self.variant_coherence_graph(search.variant);
        if graph.num_specimens() == 0 || id_index.is_empty() {
            return None;
        }

        let mut matches = Vec::new();
        let mut best: Option<&ScoredOutcome> = None;
        for (specimen_id, scored) in per_specimen {
            if scored.score <= 0.0 {
                continue;
            }
            if let Some(id) = id_index.get(*specimen_id) {
                matches.push(ClusterMatch {
                    id: *id,
                    inliers: scored.score.round().max(0.0) as u32,
                    coverage_permille: 1_000,
                });
                if best.is_none_or(|current| scored.score > current.score) {
                    best = Some(scored);
                }
            }
        }
        if matches.is_empty() {
            return None;
        }

        let scorer = cluster_scorer?;
        let decision = scorer.score(&matches, graph);
        let (passed, reason, cluster_size) = match decision {
            ClusterDecision::HardAct(HardActReason::SingleStrongMatch) => (
                false,
                Some("single_strong_match_requires_confirmed_gate"),
                None,
            ),
            ClusterDecision::HardAct(HardActReason::CoherentCluster { size }) => {
                (true, Some("coherent_cluster"), Some(size))
            }
            ClusterDecision::NoHardAct(info) => (
                false,
                Some("no_coherent_cluster"),
                Some(info.best_cluster_size),
            ),
        };
        let best = best?;
        diagnostics.steps.push(cluster_coherence_step(
            threshold,
            search.name,
            passed,
            reason,
            best,
            matches.len(),
            cluster_size,
        ));
        if !passed {
            return None;
        }

        let mut outcome = best.outcome.clone();
        outcome.confidence = MatchConfidence::ClusterCoherence;
        outcome.suspicious = suspicious;
        outcome.match_score = Some(best.score.max(threshold.cluster_hard_score as f32));
        outcome.diagnostics = diagnostics.clone();
        Some(outcome)
    }

    fn variant_specimen_id_index(&self, variant: MatchVariant) -> &HashMap<String, SpecimenId> {
        match variant {
            MatchVariant::Original => &self.specimen_id_index,
            MatchVariant::DiscordPreview => &self.preview_specimen_id_index,
        }
    }

    fn variant_coherence_graph(&self, variant: MatchVariant) -> &CoherenceGraph {
        match variant {
            MatchVariant::Original => &self.coherence_graph,
            MatchVariant::DiscordPreview => &self.preview_coherence_graph,
        }
    }
}

#[derive(Debug, Clone)]
struct ScoredOutcome {
    score: f32,
    outcome: MatchOutcome,
}

fn build_coherence_graph(
    specimens: &[IndexedSpecimen],
    threshold: &DetectionThreshold,
) -> CoherenceGraph {
    if specimens.len() > CLUSTER_GRAPH_MAX_SPECIMENS {
        return CoherenceGraph::default();
    }
    let mut builder = CoherenceGraphBuilder::new(specimens.len(), CLUSTER_GRAPH_BUILD_FLOOR);
    let mut evaluations = 0usize;
    for left in 0..specimens.len() {
        for right in (left + 1)..specimens.len() {
            if evaluations >= CLUSTER_GRAPH_MAX_PAIR_EVALUATIONS {
                return builder.build();
            }
            evaluations += 1;
            let score =
                specimen_pair_coherence_score(&specimens[left], &specimens[right], threshold);
            if score > 0 {
                builder.add_edge(left as SpecimenId, right as SpecimenId, score);
            }
        }
    }
    builder.build()
}

fn append_coherence_graph(
    existing: &CoherenceGraph,
    specimens: &[IndexedSpecimen],
    new_index: usize,
    threshold: &DetectionThreshold,
) -> CoherenceGraph {
    if specimens.len() > CLUSTER_GRAPH_MAX_SPECIMENS {
        return CoherenceGraph::default();
    }
    let mut builder = CoherenceGraphBuilder::new(specimens.len(), CLUSTER_GRAPH_BUILD_FLOOR);
    for (left, right, coherence) in existing.undirected_edges() {
        builder.add_edge(left, right, coherence);
    }
    let Some(new_specimen) = specimens.get(new_index) else {
        return builder.build();
    };
    for (evaluations, (index, specimen)) in specimens.iter().enumerate().take(new_index).enumerate()
    {
        if evaluations >= CLUSTER_GRAPH_MAX_PAIR_EVALUATIONS {
            break;
        }
        let score = specimen_pair_coherence_score(specimen, new_specimen, threshold);
        if score > 0 {
            builder.add_edge(index as SpecimenId, new_index as SpecimenId, score);
        }
    }
    builder.build()
}

fn specimen_pair_coherence_score(
    left: &IndexedSpecimen,
    right: &IndexedSpecimen,
    threshold: &DetectionThreshold,
) -> u32 {
    specimen_pair_directional_score(left, right, threshold)
        .max(specimen_pair_directional_score(right, left, threshold))
        .round()
        .clamp(0.0, 1_000.0) as u32
}

fn specimen_pair_directional_score(
    reference: &IndexedSpecimen,
    candidate: &IndexedSpecimen,
    threshold: &DetectionThreshold,
) -> f32 {
    let geometry_compatible =
        fingerprint_geometry_compatible(reference.geometry, candidate.geometry, threshold);
    if !geometry_compatible {
        return 0.0;
    }

    let visual_score = visual_signature_score(&reference.visual, &candidate.visual, threshold);
    let mut score = 0.0;
    if threshold.perceptual_hash
        && let Some(distances) = reference
            .phash64
            .zip(candidate.phash64)
            .zip(reference.dhash64.zip(candidate.dhash64))
            .map(|((left_phash, right_phash), (left_dhash, right_dhash))| {
                (
                    hamming(left_phash, right_phash),
                    hamming(left_dhash, right_dhash),
                )
            })
        && perceptual_hash_compatible_for_threshold(distances.0, distances.1, threshold)
    {
        score += stage_score_with_visual(
            perceptual_score(distances, threshold),
            visual_score,
            threshold,
        ) * threshold.perceptual_score_weight;
    }

    if threshold.local_anchors {
        let limits = LocalThresholds::from_detection_threshold(threshold);
        let mut geometry_scratch = GeometryScratch::default();
        let anchor_hits = collect_verified_anchor_hits(
            &reference.anchors,
            &candidate.dense_local_anchors,
            limits,
        );
        if let Some(local) = verified_local_comparison(&anchor_hits, limits, &mut geometry_scratch)
        {
            score += stage_score_with_visual(
                local_anchor_score(&local, threshold),
                visual_score,
                threshold,
            ) * threshold.local_anchor_score_weight;
        }

        let dense_hits = collect_verified_dense_local_hits_from_slice(
            &reference.dense_local_anchors,
            &candidate.dense_local_anchors,
            limits,
        );
        if let Some(local) = verified_local_comparison(&dense_hits, limits, &mut geometry_scratch) {
            score += stage_score_with_visual(
                local_anchor_score(&local, threshold),
                visual_score,
                threshold,
            ) * threshold.dense_local_anchor_score_weight;
        }
    }

    score
}

fn cluster_coherence_step(
    threshold: &DetectionThreshold,
    threshold_name: &'static str,
    passed: bool,
    reason: Option<&'static str>,
    best: &ScoredOutcome,
    candidates_considered: usize,
    cluster_size: Option<u16>,
) -> MatchStepDiagnostic {
    MatchStepDiagnostic {
        threshold: threshold_name,
        step: "cluster_coherence",
        passed,
        reason,
        specimen_id: Some(best.outcome.specimen_id.clone()),
        candidates_considered: Some(candidates_considered),
        phash64_distance: best.outcome.phash64_distance,
        dhash64_distance: best.outcome.dhash64_distance,
        geometry_compatible: None,
        visual_compatible: None,
        local_anchor_hits: Some(cluster_size.map_or(0, usize::from)),
        local_distinct_regions: Some(threshold.cluster_min_size as usize),
        local_average_distance: None,
        local_layout_spread: None,
        local_mean_residual: None,
        local_scale: None,
        local_angle: None,
        local_geometry_model: None,
        visual_shape_signals: None,
        visual_shape_score: None,
        match_score: Some(best.score),
    }
}

fn stage_weight(confidence: MatchConfidence, threshold: &DetectionThreshold) -> f32 {
    match confidence {
        MatchConfidence::Perceptual | MatchConfidence::SuspiciousPerceptual => {
            threshold.perceptual_score_weight
        }
        MatchConfidence::LocalAnchors | MatchConfidence::SuspiciousLocalAnchors => {
            threshold.local_anchor_score_weight
        }
        MatchConfidence::DenseLocalAnchors | MatchConfidence::SuspiciousDenseLocalAnchors => {
            threshold.dense_local_anchor_score_weight
        }
        MatchConfidence::ExactXxh128 | MatchConfidence::ClusterCoherence => 0.0,
    }
}

fn perceptual_score(
    (phash64_distance, dhash64_distance): (u32, u32),
    threshold: &DetectionThreshold,
) -> f32 {
    let total_distance = phash64_distance + dhash64_distance;
    let max_distance = threshold.perceptual_hash_max_total_distance.max(1);
    let closeness = 1.0 - (total_distance as f32 / max_distance as f32).clamp(0.0, 1.0);
    threshold.perceptual_score_floor + closeness * (100.0 - threshold.perceptual_score_floor)
}

fn stage_score_with_visual(
    core_score: f32,
    visual_score: f32,
    threshold: &DetectionThreshold,
) -> f32 {
    let visual_weight = threshold.visual_signature_score_weight.clamp(0.0, 1.0);
    let visual_multiplier = (1.0 - visual_weight) + visual_weight * (visual_score / 100.0);
    (core_score * visual_multiplier).clamp(0.0, 100.0)
}

const fn perceptual_hash_compatible(
    phash64_distance: u32,
    dhash64_distance: u32,
    phash64_max_distance: u32,
    dhash64_max_distance: u32,
    max_total_distance: u32,
) -> bool {
    phash64_distance <= phash64_max_distance
        && dhash64_distance <= dhash64_max_distance
        && phash64_distance + dhash64_distance <= max_total_distance
}

fn perceptual_hash_compatible_for_threshold(
    phash64_distance: u32,
    dhash64_distance: u32,
    threshold: &DetectionThreshold,
) -> bool {
    perceptual_hash_compatible(
        phash64_distance,
        dhash64_distance,
        threshold.phash64_max_distance,
        threshold.dhash64_max_distance,
        threshold.perceptual_hash_max_total_distance,
    )
}

fn perceptual_hash_visually_supported(
    phash64_distance: u32,
    dhash64_distance: u32,
    hash_compatible: bool,
    geometry_compatible: bool,
    threshold: &DetectionThreshold,
) -> bool {
    let slack = threshold.perceptual_visual_support_distance_slack;
    if slack == 0 {
        return false;
    }
    if hash_compatible {
        return true;
    }
    geometry_compatible
        && perceptual_hash_compatible(
            phash64_distance,
            dhash64_distance,
            threshold.phash64_max_distance.saturating_add(slack),
            threshold.dhash64_max_distance.saturating_add(slack),
            threshold
                .perceptual_hash_max_total_distance
                .saturating_add(slack),
        )
}

fn perceptual_hash_compatible_for_match_config(
    phash64_distance: u32,
    dhash64_distance: u32,
    config: &MatchConfig,
    suspicious: bool,
) -> bool {
    if suspicious {
        perceptual_hash_compatible(
            phash64_distance,
            dhash64_distance,
            config.suspicious_phash64_max_distance,
            config.suspicious_dhash64_max_distance,
            config.suspicious_perceptual_hash_max_total_distance,
        )
    } else {
        perceptual_hash_compatible(
            phash64_distance,
            dhash64_distance,
            config.phash64_max_distance,
            config.dhash64_max_distance,
            config.perceptual_hash_max_total_distance,
        )
    }
}

fn local_anchor_score(local: &LocalAnchorComparison, threshold: &DetectionThreshold) -> f32 {
    let hit_score = ratio_score(local.hits, threshold.local_score_full_hits);
    let region_score = ratio_score(local.distinct_regions, threshold.local_score_full_regions);
    let spread_score = ratio_score_f32(
        local.layout_spread.unwrap_or(0.0),
        threshold.local_score_full_spread,
    );
    let mean_score = inverse_score(
        local
            .average_distance
            .unwrap_or(threshold.max_mean_distance),
        threshold.max_mean_distance.max(0.1),
    );
    let residual_score = inverse_score(local.mean_residual.unwrap_or(24.0), 24.0);
    (hit_score * 0.50
        + region_score * 0.18
        + spread_score * 0.14
        + mean_score * 0.10
        + residual_score * 0.08)
        .clamp(0.0, 100.0)
}

fn visual_shape_score(evidence: VisualShapeEvidence, threshold: &DetectionThreshold) -> f32 {
    ratio_score_f32(evidence.score as f32, threshold.visual_shape_score_full)
}

fn ratio_score(value: usize, full_value: usize) -> f32 {
    if full_value == 0 {
        return 0.0;
    }
    ratio_score_f32(value as f32, full_value as f32)
}

fn ratio_score_f32(value: f32, full_value: f32) -> f32 {
    if !value.is_finite() || !full_value.is_finite() || full_value <= 0.0 {
        return 0.0;
    }
    (value / full_value).clamp(0.0, 1.0) * 100.0
}

fn inverse_score(value: f32, max_value: f32) -> f32 {
    if !value.is_finite() || !max_value.is_finite() || max_value <= 0.0 {
        return 0.0;
    }
    (1.0 - (value / max_value).clamp(0.0, 1.0)) * 100.0
}

fn match_diagnostics_for_candidate(
    candidate: &ParsedCandidate,
    variant: MatchVariant,
) -> MatchDiagnostics {
    let text_stats = candidate.text_grid_stats;
    MatchDiagnostics {
        representation: variant.representation(),
        candidate_short_edge: candidate.geometry.short_edge,
        candidate_area: candidate.geometry.area,
        candidate_aspect: candidate.geometry.aspect,
        candidate_luma_mean: candidate.visual.luma_mean,
        candidate_luma_std: candidate.visual.luma_std,
        candidate_text_grid_mean: text_stats.mean,
        candidate_text_regions: text_stats.regions,
        candidate_local_hashes: candidate.base_local_hash_count,
        steps: Vec::new(),
    }
}

#[inline]
fn diagnostics_stub(template: &MatchDiagnostics) -> MatchDiagnostics {
    MatchDiagnostics {
        representation: template.representation,
        candidate_short_edge: template.candidate_short_edge,
        candidate_area: template.candidate_area,
        candidate_aspect: template.candidate_aspect,
        candidate_luma_mean: template.candidate_luma_mean,
        candidate_luma_std: template.candidate_luma_std,
        candidate_text_grid_mean: template.candidate_text_grid_mean,
        candidate_text_regions: template.candidate_text_regions,
        candidate_local_hashes: template.candidate_local_hashes,
        steps: Vec::new(),
    }
}

fn best_perceptual_step(
    candidate: &MatchStepDiagnostic,
    current: Option<&MatchStepDiagnostic>,
) -> bool {
    let candidate_score = candidate
        .phash64_distance
        .zip(candidate.dhash64_distance)
        .map_or(u32::MAX, |(phash, dhash)| phash + dhash);
    let current_score = current
        .and_then(|step| step.phash64_distance.zip(step.dhash64_distance))
        .map_or(u32::MAX, |(phash, dhash)| phash + dhash);
    candidate_score < current_score
}

fn best_scored_step(
    candidate: &MatchStepDiagnostic,
    current: Option<&MatchStepDiagnostic>,
) -> bool {
    candidate.match_score.unwrap_or(f32::MIN)
        > current
            .and_then(|step| step.match_score)
            .unwrap_or(f32::MIN)
}

fn best_local_step(candidate: &MatchStepDiagnostic, current: Option<&MatchStepDiagnostic>) -> bool {
    let candidate_score = (
        candidate.local_anchor_hits.unwrap_or_default(),
        candidate.local_distinct_regions.unwrap_or_default(),
        std::cmp::Reverse(
            candidate
                .local_average_distance
                .map_or(u32::MAX, |value| (value * 100.0) as u32),
        ),
    );
    let current_score = current.map_or((0, 0, std::cmp::Reverse(u32::MAX)), |step| {
        (
            step.local_anchor_hits.unwrap_or_default(),
            step.local_distinct_regions.unwrap_or_default(),
            std::cmp::Reverse(
                step.local_average_distance
                    .map_or(u32::MAX, |value| (value * 100.0) as u32),
            ),
        )
    });
    candidate_score > current_score
}

fn visual_shape_diagnostic_step(
    candidate: &ParsedCandidate,
    threshold: &DetectionThreshold,
    threshold_name: &'static str,
    evidence: Option<VisualShapeEvidence>,
) -> MatchStepDiagnostic {
    let text_stats = candidate.text_grid_stats;
    MatchStepDiagnostic {
        threshold: threshold_name,
        step: "visual_shape",
        passed: evidence.is_some(),
        reason: evidence
            .is_none()
            .then_some("insufficient_visual_shape_signals"),
        specimen_id: None,
        candidates_considered: Some(1),
        phash64_distance: None,
        dhash64_distance: None,
        geometry_compatible: Some(fingerprint_geometry_compatible_for_limits(
            candidate.geometry,
            candidate.geometry,
            GeometryLimits::for_candidate_shape(threshold),
        )),
        visual_compatible: None,
        local_anchor_hits: None,
        local_distinct_regions: Some(text_stats.regions),
        local_average_distance: None,
        local_layout_spread: None,
        local_mean_residual: None,
        local_scale: None,
        local_angle: None,
        local_geometry_model: None,
        visual_shape_signals: evidence.map(|evidence| evidence.signals),
        visual_shape_score: evidence.map(|evidence| evidence.score),
        match_score: evidence.map(|evidence| visual_shape_score(evidence, threshold)),
    }
}

impl From<crate::bot::ledger::SpecimenRecord> for ImageFingerprint {
    fn from(value: crate::bot::ledger::SpecimenRecord) -> Self {
        Self {
            width: value.image.width,
            height: value.image.height,
            mime: value.image.mime,
            byte_xxh128: value.image.byte_xxh128,
            phash64: value.image.phash64,
            dhash64: value.image.dhash64,
            visual: value.image.visual,
            local_anchors: value.anchors,
            local_hashes: value.local_hashes,
        }
    }
}

fn descriptor_hamming(anchor: &ParsedAnchor, candidate: &ParsedLocalHash) -> u32 {
    let first = hamming(anchor.hash, candidate.hash);
    let second = hamming(anchor.hash2, candidate.hash2);
    first.saturating_add(second).div_ceil(2)
}

fn bucket_occupancy_stats(buckets: impl Iterator<Item = usize>) -> BucketOccupancyStats {
    let mut sizes = buckets.filter(|size| *size > 0).collect::<Vec<_>>();
    if sizes.is_empty() {
        return BucketOccupancyStats {
            bucket_count: 0,
            entry_count: 0,
            min: 0,
            max: 0,
            avg: 0.0,
            p50: 0,
            p90: 0,
            p95: 0,
            p99: 0,
        };
    }
    sizes.sort_unstable();
    let entry_count = sizes.iter().sum::<usize>();
    BucketOccupancyStats {
        bucket_count: sizes.len(),
        entry_count,
        min: sizes[0],
        max: sizes[sizes.len() - 1],
        avg: entry_count as f64 / sizes.len() as f64,
        p50: percentile_usize(&sizes, 50),
        p90: percentile_usize(&sizes, 90),
        p95: percentile_usize(&sizes, 95),
        p99: percentile_usize(&sizes, 99),
    }
}

fn percentile_usize(sorted: &[usize], percentile: usize) -> usize {
    let index = sorted
        .len()
        .saturating_sub(1)
        .saturating_mul(percentile)
        .div_ceil(100);
    sorted[index.min(sorted.len() - 1)]
}

#[derive(Debug, Clone, Copy)]
struct LocalThresholds {
    min_anchor_hits: usize,
    min_distinct_regions: usize,
    max_distance: u32,
    max_mean_distance: f32,
    local_luma_candidate_max_delta: u8,
    local_contrast_candidate_max_delta: u8,
    local_edge_density_candidate_max_delta: u8,
    local_position_candidate_max_delta: u8,
    geometry_enable_affine: bool,
    geometry_enable_homography: bool,
    geometry_model_slack: f32,
    geometry_max_anisotropy: f32,
    geometry_max_perspective: f32,
    geometry_affine_min_extra_inliers: usize,
    geometry_affine_min_extra_regions: usize,
    geometry_affine_max_mean_residual: f32,
    geometry_homography_min_extra_inliers: usize,
    geometry_homography_min_extra_regions: usize,
    geometry_homography_max_mean_residual: f32,
    geometry_ratio_min_margin: u8,
    geometry_enable_prosac_fallback: bool,
    geometry_prosac_max_iters: u32,
    geometry_prosac_min_inliers: usize,
}

impl LocalThresholds {
    fn from_detection_threshold(threshold: &DetectionThreshold) -> Self {
        Self {
            min_anchor_hits: threshold.min_anchor_hits,
            min_distinct_regions: threshold.min_distinct_regions,
            max_distance: threshold.max_mean_distance.ceil().clamp(0.0, 15.0) as u32,
            max_mean_distance: threshold.max_mean_distance,
            local_luma_candidate_max_delta: threshold.local_luma_candidate_max_delta,
            local_contrast_candidate_max_delta: threshold.local_contrast_candidate_max_delta,
            local_edge_density_candidate_max_delta: threshold
                .local_edge_density_candidate_max_delta,
            local_position_candidate_max_delta: threshold.local_position_candidate_max_delta,
            geometry_enable_affine: threshold.geometry_enable_affine,
            geometry_enable_homography: threshold.geometry_enable_homography,
            geometry_model_slack: threshold.geometry_model_slack,
            geometry_max_anisotropy: threshold.geometry_max_anisotropy,
            geometry_max_perspective: threshold.geometry_max_perspective,
            geometry_affine_min_extra_inliers: threshold.geometry_affine_min_extra_inliers,
            geometry_affine_min_extra_regions: threshold.geometry_affine_min_extra_regions,
            geometry_affine_max_mean_residual: threshold.geometry_affine_max_mean_residual,
            geometry_homography_min_extra_inliers: threshold.geometry_homography_min_extra_inliers,
            geometry_homography_min_extra_regions: threshold.geometry_homography_min_extra_regions,
            geometry_homography_max_mean_residual: threshold.geometry_homography_max_mean_residual,
            geometry_ratio_min_margin: threshold.geometry_ratio_min_margin,
            geometry_enable_prosac_fallback: threshold.geometry_enable_prosac_fallback,
            geometry_prosac_max_iters: threshold.geometry_prosac_max_iters,
            geometry_prosac_min_inliers: threshold.geometry_prosac_min_inliers,
        }
    }

    fn from_match_config(config: &MatchConfig, suspicious: bool) -> Self {
        if suspicious {
            Self {
                min_anchor_hits: config.local_suspicious_min_anchor_hits,
                min_distinct_regions: config.local_suspicious_min_distinct_regions,
                max_distance: config.local_anchor_max_distance,
                max_mean_distance: config.local_suspicious_max_mean_distance,
                local_luma_candidate_max_delta: config.suspicious_local_luma_candidate_max_delta,
                local_contrast_candidate_max_delta: config
                    .suspicious_local_contrast_candidate_max_delta,
                local_edge_density_candidate_max_delta: config
                    .suspicious_local_edge_density_candidate_max_delta,
                local_position_candidate_max_delta: config
                    .suspicious_local_position_candidate_max_delta,
                geometry_enable_affine: config.geometry_enable_affine,
                geometry_enable_homography: config.geometry_enable_homography,
                geometry_model_slack: config.geometry_model_slack,
                geometry_max_anisotropy: config.geometry_max_anisotropy,
                geometry_max_perspective: config.geometry_max_perspective,
                geometry_affine_min_extra_inliers: config.geometry_affine_min_extra_inliers,
                geometry_affine_min_extra_regions: config.geometry_affine_min_extra_regions,
                geometry_affine_max_mean_residual: config.geometry_affine_max_mean_residual,
                geometry_homography_min_extra_inliers: config.geometry_homography_min_extra_inliers,
                geometry_homography_min_extra_regions: config.geometry_homography_min_extra_regions,
                geometry_homography_max_mean_residual: config.geometry_homography_max_mean_residual,
                geometry_ratio_min_margin: config.geometry_ratio_min_margin,
                geometry_enable_prosac_fallback: config.geometry_enable_prosac_fallback,
                geometry_prosac_max_iters: config.geometry_prosac_max_iters,
                geometry_prosac_min_inliers: config.geometry_prosac_min_inliers,
            }
        } else {
            Self {
                min_anchor_hits: config.local_min_anchor_hits,
                min_distinct_regions: config.local_min_distinct_regions,
                max_distance: config.local_anchor_max_distance,
                max_mean_distance: config.local_max_mean_distance,
                local_luma_candidate_max_delta: config.local_luma_candidate_max_delta,
                local_contrast_candidate_max_delta: config.local_contrast_candidate_max_delta,
                local_edge_density_candidate_max_delta: config
                    .local_edge_density_candidate_max_delta,
                local_position_candidate_max_delta: config.local_position_candidate_max_delta,
                geometry_enable_affine: config.geometry_enable_affine,
                geometry_enable_homography: config.geometry_enable_homography,
                geometry_model_slack: config.geometry_model_slack,
                geometry_max_anisotropy: config.geometry_max_anisotropy,
                geometry_max_perspective: config.geometry_max_perspective,
                geometry_affine_min_extra_inliers: config.geometry_affine_min_extra_inliers,
                geometry_affine_min_extra_regions: config.geometry_affine_min_extra_regions,
                geometry_affine_max_mean_residual: config.geometry_affine_max_mean_residual,
                geometry_homography_min_extra_inliers: config.geometry_homography_min_extra_inliers,
                geometry_homography_min_extra_regions: config.geometry_homography_min_extra_regions,
                geometry_homography_max_mean_residual: config.geometry_homography_max_mean_residual,
                geometry_ratio_min_margin: config.geometry_ratio_min_margin,
                geometry_enable_prosac_fallback: config.geometry_enable_prosac_fallback,
                geometry_prosac_max_iters: config.geometry_prosac_max_iters,
                geometry_prosac_min_inliers: config.geometry_prosac_min_inliers,
            }
        }
    }
}

impl From<LocalThresholds> for LocalFeatureFilter {
    fn from(threshold: LocalThresholds) -> Self {
        Self {
            luma: threshold.local_luma_candidate_max_delta,
            contrast: threshold.local_contrast_candidate_max_delta,
            edge_density: threshold.local_edge_density_candidate_max_delta,
            position: threshold.local_position_candidate_max_delta,
        }
    }
}

fn verified_local_comparison(
    hits: &[AnchorHit],
    threshold: LocalThresholds,
    geometry_scratch: &mut GeometryScratch,
) -> Option<LocalAnchorComparison> {
    if hits.len() < threshold.min_anchor_hits {
        return None;
    }
    let correspondences = hits
        .iter()
        .map(|hit| hit.correspondence)
        .collect::<Vec<_>>();
    let geo_cfg = GeoCfg {
        min_inliers: threshold.min_anchor_hits.clamp(2, 3),
        inlier_residual: 24.0,
        enable_affine: threshold.geometry_enable_affine,
        enable_homography: threshold.geometry_enable_homography,
        model_slack: threshold.geometry_model_slack,
        max_anisotropy: threshold.geometry_max_anisotropy,
        max_perspective: threshold.geometry_max_perspective,
        ratio_min_margin: threshold.geometry_ratio_min_margin,
        enable_prosac_fallback: threshold.geometry_enable_prosac_fallback,
        prosac_max_iters: threshold.geometry_prosac_max_iters,
        prosac_min_inliers: threshold.geometry_prosac_min_inliers,
        ..GeoCfg::default()
    };
    let geo = verify_geometry_with_scratch(&correspondences, &geo_cfg, geometry_scratch)?;
    let accept = geometry_acceptance_for_model(&geo, threshold);
    geo_passes(&geo, &accept).then(|| local_comparison_from_geo(&geo))
}

fn geometry_acceptance_for_model(geo: &GeoMatch, threshold: LocalThresholds) -> GeoAccept {
    let mut accept = GeoAccept {
        min_inliers: threshold.min_anchor_hits,
        min_regions: threshold.min_distinct_regions,
        min_spread: verified_min_spread(threshold),
        max_mean_residual: 24.0,
        max_mean_hamming: threshold.max_mean_distance,
    };
    match geo.model {
        GeoModel::Similarity => accept,
        GeoModel::Affine => {
            accept.min_inliers += threshold.geometry_affine_min_extra_inliers;
            accept.min_regions += threshold.geometry_affine_min_extra_regions;
            accept.max_mean_residual = accept
                .max_mean_residual
                .min(threshold.geometry_affine_max_mean_residual);
            accept
        }
        GeoModel::Homography => {
            accept.min_inliers += threshold.geometry_homography_min_extra_inliers;
            accept.min_regions += threshold.geometry_homography_min_extra_regions;
            accept.max_mean_residual = accept
                .max_mean_residual
                .min(threshold.geometry_homography_max_mean_residual);
            accept
        }
    }
}

fn verified_min_spread(threshold: LocalThresholds) -> f32 {
    24.0 * threshold.min_distinct_regions.saturating_sub(1).max(1) as f32
}

fn local_comparison_from_geo(geo: &GeoMatch) -> LocalAnchorComparison {
    LocalAnchorComparison {
        matched: true,
        suspicious: false,
        hits: geo.inliers.len(),
        distinct_regions: geo.region_count,
        average_distance: Some(geo.mean_hamming),
        layout_spread: Some(geo.spread),
        mean_residual: Some(geo.mean_residual),
        scale: Some(geo.scale),
        angle: Some(geo.angle),
        geometry_model: Some(geometry_model_from_geo(geo.model)),
    }
}

fn unverified_local_support_comparison(
    hits: &[AnchorHit],
    ratio_min_margin: u8,
) -> Option<LocalSupportComparison> {
    let mut raw_best_per_anchor: HashMap<(u32, u32, u16), AnchorHit> = HashMap::default();
    let mut ratio_best_per_anchor: HashMap<(u32, u32, u16), AnchorHit> = HashMap::default();
    for hit in hits {
        let key = (
            hit.correspondence.spec.x.to_bits(),
            hit.correspondence.spec.y.to_bits(),
            hit.correspondence.region,
        );
        insert_best_anchor_hit(&mut raw_best_per_anchor, key, *hit);
        if anchor_hit_passes_ratio(hit, ratio_min_margin) {
            insert_best_anchor_hit(&mut ratio_best_per_anchor, key, *hit);
        }
    }
    if raw_best_per_anchor.is_empty() || ratio_best_per_anchor.is_empty() {
        return None;
    }

    let mut regions = HashSet::default();
    let mut distance_sum = 0_u32;
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for hit in ratio_best_per_anchor.values() {
        regions.insert(hit.correspondence.region);
        distance_sum = distance_sum.saturating_add(hit.distance);
        min_x = min_x.min(hit.correspondence.cand.x);
        min_y = min_y.min(hit.correspondence.cand.y);
        max_x = max_x.max(hit.correspondence.cand.x);
        max_y = max_y.max(hit.correspondence.cand.y);
    }
    let hits = ratio_best_per_anchor.len();
    Some(LocalSupportComparison {
        comparison: LocalAnchorComparison {
            matched: false,
            suspicious: true,
            hits,
            distinct_regions: regions.len(),
            average_distance: Some(distance_sum as f32 / hits as f32),
            layout_spread: Some((max_x - min_x).min(max_y - min_y).max(0.0)),
            mean_residual: None,
            scale: None,
            angle: None,
            geometry_model: None,
        },
        raw_hits: raw_best_per_anchor.len(),
    })
}

fn insert_best_anchor_hit(
    best_hits: &mut HashMap<(u32, u32, u16), AnchorHit>,
    key: (u32, u32, u16),
    hit: AnchorHit,
) {
    let entry = best_hits.entry(key).or_insert(hit);
    if hit.distance < entry.distance
        || (hit.distance == entry.distance && hit.quality < entry.quality)
    {
        *entry = hit;
    }
}

fn anchor_hit_passes_ratio(hit: &AnchorHit, ratio_min_margin: u8) -> bool {
    ratio_min_margin == 0
        || i16::from(hit.correspondence.second_hamming) - i16::from(hit.correspondence.hamming)
            >= i16::from(ratio_min_margin)
}

fn local_unverified_support_passes(
    specimen: &IndexedSpecimen,
    candidate: &ParsedCandidate,
    local: &LocalSupportComparison,
    threshold: &DetectionThreshold,
) -> bool {
    if !threshold.local_unverified_support {
        return false;
    }
    let comparison = &local.comparison;
    if comparison.hits < threshold.local_unverified_support_min_anchor_hits
        || comparison.distinct_regions < threshold.local_unverified_support_min_distinct_regions
        || comparison
            .average_distance
            .is_none_or(|distance| distance > threshold.local_unverified_support_max_mean_distance)
    {
        return false;
    }
    if local.raw_hits == 0
        || comparison.hits.saturating_mul(1_000)
            < usize::from(threshold.local_unverified_support_min_retention_permille)
                .saturating_mul(local.raw_hits)
    {
        return false;
    }
    let (phash64_distance, dhash64_distance) = hash_distances(specimen, candidate);
    let Some((phash64_distance, dhash64_distance)) = phash64_distance.zip(dhash64_distance) else {
        return false;
    };
    if phash64_distance.saturating_add(dhash64_distance)
        > threshold.local_unverified_support_max_perceptual_total_distance
    {
        return false;
    }
    local_unverified_support_geometry_compatible(specimen.geometry, candidate.geometry, threshold)
}

fn local_unverified_support_geometry_compatible(
    specimen: FingerprintGeometry,
    candidate: FingerprintGeometry,
    threshold: &DetectionThreshold,
) -> bool {
    candidate.short_edge >= threshold.geometry_min_short_edge
        && candidate.area >= threshold.geometry_min_area
        && candidate.aspect <= threshold.geometry_max_aspect_ratio
        && (specimen.aspect - candidate.aspect).abs()
            <= threshold.local_unverified_support_max_aspect_delta
        && dimension_delta_compatible(
            specimen.width,
            candidate.width,
            threshold.local_unverified_support_max_dimension_delta,
        )
        && dimension_delta_compatible(
            specimen.height,
            candidate.height,
            threshold.local_unverified_support_max_dimension_delta,
        )
}

const fn geometry_model_from_geo(model: GeoModel) -> GeometryModel {
    match model {
        GeoModel::Similarity => GeometryModel::Similarity,
        GeoModel::Affine => GeometryModel::Affine,
        GeoModel::Homography => GeometryModel::Homography,
    }
}

fn anchor_correspondence(
    anchor: &ParsedAnchor,
    candidate_hash: &ParsedLocalHash,
    distance: u32,
) -> Option<Correspondence> {
    if candidate_hash.scale_percent == 0 {
        return None;
    }
    let scale = candidate_hash.scale_percent as f32 / 100.0;
    let spec = P {
        x: anchor.x as f32 + anchor.w as f32 * 0.5,
        y: anchor.y as f32 + anchor.h as f32 * 0.5,
    };
    let cand = P {
        x: (candidate_hash.x as f32 + candidate_hash.w as f32 * 0.5) / scale,
        y: (candidate_hash.y as f32 + candidate_hash.h as f32 * 0.5) / scale,
    };
    Some(Correspondence {
        spec,
        cand,
        cand_id: candidate_hash.physical_id,
        region: anchor.region.min(u16::MAX as u32) as u16,
        hamming: distance.min(u8::MAX as u32) as u8,
        second_hamming: u8::MAX,
    })
}

fn anchor_hit_quality(
    anchor: &ParsedAnchor,
    candidate_hash: &ParsedLocalHash,
    distance: u32,
) -> u32 {
    let position_delta = anchor
        .pos_x
        .abs_diff(candidate_hash.pos_x)
        .max(anchor.pos_y.abs_diff(candidate_hash.pos_y));
    let feature_delta = anchor
        .luma_mean
        .abs_diff(candidate_hash.luma_mean)
        .saturating_add(anchor.luma_std.abs_diff(candidate_hash.luma_std))
        .saturating_add(anchor.edge_density.abs_diff(candidate_hash.edge_density));
    distance
        .saturating_mul(4)
        .saturating_add(u32::from(position_delta))
        .saturating_add(u32::from(feature_delta / 8))
}

fn dense_local_hit_quality(
    dense_anchor: &ParsedLocalHash,
    candidate_hash: &ParsedLocalHash,
    distance: u32,
) -> u32 {
    let feature_delta = dense_anchor
        .luma_mean
        .abs_diff(candidate_hash.luma_mean)
        .saturating_add(dense_anchor.luma_std.abs_diff(candidate_hash.luma_std))
        .saturating_add(
            dense_anchor
                .edge_density
                .abs_diff(candidate_hash.edge_density),
        );
    distance
        .saturating_mul(4)
        .saturating_add(u32::from(feature_delta / 8))
}

fn dense_local_anchor_thresholds(threshold: &DetectionThreshold) -> LocalThresholds {
    LocalThresholds::from_detection_threshold(threshold)
}

fn compare_local_anchors_with_threshold(
    anchors: &[ParsedAnchor],
    candidate_hashes: &[ParsedLocalHash],
    threshold: LocalThresholds,
) -> LocalAnchorComparison {
    if anchors.len() < threshold.min_anchor_hits || candidate_hashes.is_empty() {
        return LocalAnchorComparison::miss();
    }

    let hits = collect_verified_anchor_hits(anchors, candidate_hashes, threshold);
    verified_local_comparison(&hits, threshold, &mut GeometryScratch::default())
        .unwrap_or_else(LocalAnchorComparison::miss)
}

fn candidate_dense_local_hashes(candidate_hashes: &[ParsedLocalHash]) -> Vec<&ParsedLocalHash> {
    let mut selected = Vec::with_capacity(
        candidate_hashes
            .len()
            .min(DENSE_LOCAL_CANDIDATE_SCAN_CAP_PER_SCALE),
    );
    let mut counts = Vec::<(u16, usize)>::new();
    for candidate_hash in candidate_hashes {
        if candidate_hash.rotation_degrees != 0 {
            continue;
        }
        let scale_percent = candidate_hash.scale_percent;
        let Some((_, count)) = counts
            .iter_mut()
            .find(|(existing_scale, _)| *existing_scale == scale_percent)
        else {
            counts.push((scale_percent, 1));
            selected.push(candidate_hash);
            continue;
        };
        if *count < DENSE_LOCAL_CANDIDATE_SCAN_CAP_PER_SCALE {
            *count += 1;
            selected.push(candidate_hash);
        }
    }
    selected
}

fn collect_verified_anchor_hits(
    anchors: &[ParsedAnchor],
    candidate_hashes: &[ParsedLocalHash],
    threshold: LocalThresholds,
) -> Vec<AnchorHit> {
    let mut all_hits = Vec::new();
    let mut reference_hits: Vec<ScanHit<'_>> = Vec::new();
    for anchor in anchors {
        reference_hits.clear();
        for candidate_hash in candidate_hashes
            .iter()
            .filter(|candidate_hash| candidate_hash.rotation_degrees == 0)
            .filter(|candidate_hash| local_features_compatible(anchor, candidate_hash, threshold))
        {
            if candidate_hash.scale_percent == 0 {
                continue;
            }
            let distance = descriptor_hamming(anchor, candidate_hash);
            if distance > anchor.max_distance {
                continue;
            }
            reference_hits.push(ScanHit {
                distance,
                quality: anchor_hit_quality(anchor, candidate_hash, distance),
                cand: candidate_hash,
            });
            if reference_hits.len() > LOCAL_ANCHOR_CANDIDATES_PER_REFERENCE_CAP * 2 {
                retain_best_scan_hits(
                    &mut reference_hits,
                    LOCAL_ANCHOR_CANDIDATES_PER_REFERENCE_CAP,
                );
            }
        }
        retain_best_scan_hits(
            &mut reference_hits,
            LOCAL_ANCHOR_CANDIDATES_PER_REFERENCE_CAP,
        );
        append_best_distinct_anchor_hits(
            anchor,
            &mut reference_hits,
            LOCAL_GEOMETRY_ALTERNATES_PER_ANCHOR,
            &mut all_hits,
        );
    }
    all_hits
}

fn retain_best_scan_hits(hits: &mut Vec<ScanHit<'_>>, cap: usize) {
    if hits.len() <= cap {
        return;
    }
    hits.select_nth_unstable_by(cap, |left, right| left.quality.cmp(&right.quality));
    hits.truncate(cap);
}

fn append_best_distinct_anchor_hits(
    anchor: &ParsedAnchor,
    hits: &mut [ScanHit<'_>],
    limit: usize,
    selected: &mut Vec<AnchorHit>,
) {
    let start_len = selected.len();
    hits.sort_unstable_by(|left, right| {
        left.quality
            .cmp(&right.quality)
            .then_with(|| left.distance.cmp(&right.distance))
    });

    for scan in hits.iter() {
        let cand_id = scan.cand.physical_id;
        if selected[start_len..]
            .iter()
            .any(|selected_hit| selected_hit.correspondence.cand_id == cand_id)
        {
            continue;
        }
        let second_hamming = hits
            .iter()
            .filter(|other| other.cand.physical_id != cand_id)
            .map(|other| other.distance.min(u32::from(u8::MAX)) as u8)
            .min()
            .unwrap_or(u8::MAX);
        let Some(mut correspondence) = anchor_correspondence(anchor, scan.cand, scan.distance)
        else {
            continue;
        };
        correspondence.second_hamming = second_hamming;
        selected.push(AnchorHit {
            distance: scan.distance,
            quality: scan.quality,
            correspondence,
        });
        if selected.len() - start_len >= limit {
            break;
        }
    }
}

fn collect_verified_dense_local_hits(
    dense_local_anchors: &[ParsedLocalHash],
    candidate_hashes: &[&ParsedLocalHash],
    threshold: LocalThresholds,
) -> Vec<AnchorHit> {
    dense_local_anchors
        .iter()
        .filter(|dense_anchor| dense_anchor.rotation_degrees == 0)
        .filter_map(|dense_anchor| {
            let mut best = BestScanHit::empty();
            for candidate_hash in candidate_hashes.iter().copied().filter(|candidate_hash| {
                dense_local_features_compatible(dense_anchor, candidate_hash, threshold)
            }) {
                let distance = hamming(dense_anchor.hash, candidate_hash.hash);
                if distance > threshold.max_distance {
                    best.observe_distance(distance, candidate_hash.physical_id);
                    continue;
                }
                best.insert(ScanHit {
                    distance,
                    quality: dense_local_hit_quality(dense_anchor, candidate_hash, distance),
                    cand: candidate_hash,
                });
            }
            best.into_dense_local_hit(dense_anchor)
        })
        .collect()
}

fn collect_verified_dense_local_hits_from_slice(
    dense_local_anchors: &[ParsedLocalHash],
    candidate_hashes: &[ParsedLocalHash],
    threshold: LocalThresholds,
) -> Vec<AnchorHit> {
    dense_local_anchors
        .iter()
        .filter(|dense_anchor| dense_anchor.rotation_degrees == 0)
        .filter_map(|dense_anchor| {
            let mut best = BestScanHit::empty();
            for candidate_hash in candidate_hashes.iter().filter(|candidate_hash| {
                dense_local_features_compatible(dense_anchor, candidate_hash, threshold)
            }) {
                let distance = hamming(dense_anchor.hash, candidate_hash.hash);
                if distance > threshold.max_distance {
                    best.observe_distance(distance, candidate_hash.physical_id);
                    continue;
                }
                best.insert(ScanHit {
                    distance,
                    quality: dense_local_hit_quality(dense_anchor, candidate_hash, distance),
                    cand: candidate_hash,
                });
            }
            best.into_dense_local_hit(dense_anchor)
        })
        .collect()
}

fn local_features_compatible(
    anchor: &ParsedAnchor,
    candidate: &ParsedLocalHash,
    threshold: LocalThresholds,
) -> bool {
    anchor.luma_mean.abs_diff(candidate.luma_mean) <= threshold.local_luma_candidate_max_delta
        && anchor.luma_std.abs_diff(candidate.luma_std)
            <= threshold.local_contrast_candidate_max_delta
        && anchor.edge_density.abs_diff(candidate.edge_density)
            <= threshold.local_edge_density_candidate_max_delta
        && anchor.pos_x.abs_diff(candidate.pos_x) <= threshold.local_position_candidate_max_delta
        && anchor.pos_y.abs_diff(candidate.pos_y) <= threshold.local_position_candidate_max_delta
}

fn dense_local_features_compatible(
    dense_anchor: &ParsedLocalHash,
    candidate: &ParsedLocalHash,
    threshold: LocalThresholds,
) -> bool {
    dense_anchor.luma_mean.abs_diff(candidate.luma_mean) <= threshold.local_luma_candidate_max_delta
        && dense_anchor.luma_std.abs_diff(candidate.luma_std)
            <= threshold.local_contrast_candidate_max_delta
        && dense_anchor.edge_density.abs_diff(candidate.edge_density)
            <= threshold.local_edge_density_candidate_max_delta
}

fn dense_local_correspondence(
    dense_anchor: &ParsedLocalHash,
    candidate_hash: &ParsedLocalHash,
    distance: u32,
) -> Correspondence {
    let spec_scale = dense_anchor.scale_percent.max(1) as f32 / 100.0;
    let cand_scale = candidate_hash.scale_percent.max(1) as f32 / 100.0;
    let spec = P {
        x: (dense_anchor.x as f32 + dense_anchor.w as f32 * 0.5) / spec_scale,
        y: (dense_anchor.y as f32 + dense_anchor.h as f32 * 0.5) / spec_scale,
    };
    let cand = P {
        x: (candidate_hash.x as f32 + candidate_hash.w as f32 * 0.5) / cand_scale,
        y: (candidate_hash.y as f32 + candidate_hash.h as f32 * 0.5) / cand_scale,
    };
    Correspondence {
        spec,
        cand,
        cand_id: candidate_hash.physical_id,
        region: dense_anchor.region.min(u16::MAX as u32) as u16,
        hamming: distance.min(u32::from(u8::MAX)) as u8,
        second_hamming: u8::MAX,
    }
}

fn fingerprint_geometry_compatible(
    specimen: FingerprintGeometry,
    candidate: FingerprintGeometry,
    threshold: &DetectionThreshold,
) -> bool {
    fingerprint_geometry_compatible_for_limits(
        specimen,
        candidate,
        GeometryLimits::from_detection_threshold(threshold),
    )
}

fn geometry_compatible_for_match_config(
    specimen: FingerprintGeometry,
    candidate: FingerprintGeometry,
    config: &MatchConfig,
    suspicious: bool,
) -> bool {
    fingerprint_geometry_compatible_for_limits(
        specimen,
        candidate,
        GeometryLimits::from_match_config(config, suspicious),
    )
}

fn fingerprint_geometry_compatible_for_limits(
    specimen: FingerprintGeometry,
    candidate: FingerprintGeometry,
    limits: GeometryLimits,
) -> bool {
    candidate.short_edge >= limits.min_short_edge
        && candidate.area >= limits.min_area
        && candidate.aspect <= limits.max_aspect_ratio
        && (specimen.aspect - candidate.aspect).abs() <= limits.max_aspect_delta
        && dimension_delta_compatible(specimen.width, candidate.width, limits.max_width_delta)
        && dimension_delta_compatible(specimen.height, candidate.height, limits.max_height_delta)
}

fn dimension_delta_compatible(specimen: u32, candidate: u32, max_delta: f32) -> bool {
    if !max_delta.is_finite() {
        return false;
    }
    let specimen = specimen.max(1) as f32;
    let candidate = candidate.max(1) as f32;
    ((candidate / specimen) - 1.0).abs() <= max_delta
}

fn visual_signature_score(
    specimen: &ImageVisualSignature,
    candidate: &ImageVisualSignature,
    threshold: &DetectionThreshold,
) -> f32 {
    let luma = inverse_score(
        f32::from(specimen.luma_mean.abs_diff(candidate.luma_mean)),
        f32::from(threshold.visual_luma_zero_score_delta),
    );
    let color_delta = matcher_opt::mean_abs_delta(&specimen.rgb_mean, &candidate.rgb_mean);
    let grid_delta = matcher_opt::mean_abs_delta(&specimen.grid_luma, &candidate.grid_luma);
    let text_grid_delta =
        matcher_opt::text_grid_mean_delta(&specimen.text_grid, &candidate.text_grid);
    let color = inverse_score(
        f32::from(color_delta),
        f32::from(threshold.visual_color_zero_score_delta),
    );
    let grid = inverse_score(
        f32::from(grid_delta),
        f32::from(threshold.visual_grid_luma_zero_score_delta),
    );
    let text_grid = inverse_score(
        f32::from(text_grid_delta),
        f32::from(threshold.visual_text_grid_zero_score_delta),
    );
    luma.mul_add(
        0.25,
        color.mul_add(0.20, grid.mul_add(0.25, text_grid * 0.30)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bot::ledger::{SpecimenImage, SpecimenPreview, SpecimenRecord, SpecimenSource},
        configuration::guild::DetectionPolicy,
        image::types::GeometryModel,
    };

    fn specimen(phash64: &str, dhash64: &str) -> SpecimenRecord {
        SpecimenRecord {
            schema: 1,
            kind: "specimen.add".to_owned(),
            specimen_id: "spm_test".to_owned(),
            created_at: "2026-06-24T20:13:22Z".to_owned(),
            guild_id: "1".to_owned(),
            source: SpecimenSource {
                channel_id: "2".to_owned(),
                message_id: "3".to_owned(),
                source_author_id: "4".to_owned(),
                added_by_id: "5".to_owned(),
            },
            image: SpecimenImage {
                width: 1000,
                height: 1200,
                mime: Some("image/png".to_owned()),
                byte_xxh128: "b".repeat(32),
                phash64: phash64.to_owned(),
                dhash64: dhash64.to_owned(),
                visual: ImageVisualSignature::default(),
            },
            anchors: Vec::new(),
            local_hashes: Vec::new(),
            preview: None,
            sig: Some("sig".to_owned()),
        }
    }

    fn policy(config: &MatchConfig) -> DetectionPolicy {
        DetectionPolicy::from_match_config(config)
    }

    #[test]
    fn exact_xxh128_matches_first() {
        let mut specimen = specimen("0000000000000000", "0000000000000000");
        specimen.image.byte_xxh128 = "a".repeat(32);
        let matcher = Matcher::new(vec![specimen]);
        let policy = policy(&MatchConfig::default());

        let image = ImageFingerprint {
            width: 1000,
            height: 1200,
            mime: None,
            byte_xxh128: "a".repeat(32),
            phash64: "ffffffffffffffff".to_owned(),
            dhash64: "ffffffffffffffff".to_owned(),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        };

        assert!(matches!(
            matcher.find_for_policy(&image, &policy).unwrap().confidence,
            MatchConfidence::ExactXxh128
        ));
    }

    #[test]
    fn perceptual_requires_both_hashes() {
        let matcher = Matcher::new(vec![specimen("0000000000000000", "0000000000000000")]);
        let policy = policy(&MatchConfig::default());

        let image = ImageFingerprint {
            width: 1000,
            height: 1200,
            mime: None,
            byte_xxh128: "a".repeat(32),
            phash64: "000000000000003f".to_owned(),
            dhash64: "00000000000000ff".to_owned(),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        };

        assert!(matcher.find_for_policy(&image, &policy).is_some());

        let image = ImageFingerprint {
            dhash64: "ffffffffffffffff".to_owned(),
            ..image
        };

        assert!(matcher.find_for_policy(&image, &policy).is_none());
    }

    #[test]
    fn perceptual_total_distance_allows_only_balanced_near_matches() {
        let matcher = Matcher::new(vec![specimen("0000000000000000", "0000000000000000")]);
        let candidate = ImageFingerprint {
            width: 1000,
            height: 1200,
            mime: None,
            byte_xxh128: "a".repeat(32),
            phash64: "000000000000ffff".to_owned(),
            dhash64: "00000000000000ff".to_owned(),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        };
        let permissive_total = policy(&MatchConfig {
            score_threshold: 60.0,
            phash64_max_distance: 16,
            dhash64_max_distance: 12,
            perceptual_hash_max_total_distance: 24,
            suspicious_phash64_max_distance: 15,
            ..MatchConfig::default()
        });
        let tight_total = policy(&MatchConfig {
            score_threshold: 60.0,
            phash64_max_distance: 16,
            dhash64_max_distance: 12,
            perceptual_hash_max_total_distance: 23,
            suspicious_phash64_max_distance: 15,
            ..MatchConfig::default()
        });

        let outcome = matcher
            .find_for_policy(&candidate, &permissive_total)
            .unwrap();

        assert!(matches!(outcome.confidence, MatchConfidence::Perceptual));
        assert!(matcher.find_for_policy(&candidate, &tight_total).is_none());
    }

    #[test]
    fn compare_fingerprints_reports_distances_and_threshold_match() {
        let thresholds = MatchConfig {
            phash64_max_distance: 6,
            dhash64_max_distance: 8,
            ..MatchConfig::default()
        };
        let specimen = ImageFingerprint {
            width: 1000,
            height: 1200,
            mime: None,
            byte_xxh128: "a".repeat(32),
            phash64: "0000000000000000".to_owned(),
            dhash64: "0000000000000000".to_owned(),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        };
        let candidate = ImageFingerprint {
            byte_xxh128: "b".repeat(32),
            phash64: "000000000000003f".to_owned(),
            dhash64: "00000000000000ff".to_owned(),
            ..specimen.clone()
        };

        let comparison = compare_fingerprints(&specimen, &candidate, &thresholds);

        assert!(!comparison.exact_xxh128);
        assert_eq!(comparison.phash64_distance, Some(6));
        assert_eq!(comparison.dhash64_distance, Some(8));
        assert!(comparison.perceptual_match);
        assert!(comparison.matched);
    }

    #[test]
    fn suspicious_perceptual_near_miss_is_not_confirmed() {
        let matcher = Matcher::new(vec![specimen("0000000000000000", "0000000000000000")]);
        let policy = policy(&MatchConfig {
            suspicious_dhash64_max_distance: 15,
            suspicious_perceptual_hash_max_total_distance: 30,
            suspicious_score_threshold: 60.0,
            ..MatchConfig::default()
        });

        let image = ImageFingerprint {
            width: 1000,
            height: 1200,
            mime: None,
            byte_xxh128: "a".repeat(32),
            phash64: "0000000000007fff".to_owned(),
            dhash64: "0000000000007fff".to_owned(),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        };

        let outcome = matcher.find_for_policy(&image, &policy).unwrap();
        assert!(outcome.suspicious);
        assert!(matches!(
            outcome.confidence,
            MatchConfidence::SuspiciousPerceptual
        ));
    }

    #[test]
    fn visual_shape_is_not_a_standalone_suspicious_match() {
        let matcher = Matcher::new(Vec::new());
        let policy = policy(&MatchConfig::default());
        let image = visually_suspicious_image_without_specimen_evidence();

        let explanation = matcher.explain_for_policy_with_mode(
            &image,
            &policy,
            MatchEvaluationMode::ShortCircuit,
        );

        assert!(explanation.outcome.is_none());
        assert!(
            explanation
                .diagnostics
                .steps
                .iter()
                .any(|step| step.step == "visual_shape" && step.passed)
        );
    }

    #[test]
    fn preview_variants_match_only_through_preview_matcher() {
        let mut record = specimen("ffffffffffffffff", "ffffffffffffffff");
        record.preview = Some(SpecimenPreview {
            width: 800,
            height: 1000,
            mime: Some("image/png".to_owned()),
            byte_xxh128: "c".repeat(32),
            phash64: "0000000000000000".to_owned(),
            dhash64: "0000000000000000".to_owned(),
            visual: ImageVisualSignature::default(),
            anchors: Vec::new(),
            local_hashes: Vec::new(),
        });
        let matcher = Matcher::new(vec![record]);
        let policy = policy(&MatchConfig {
            score_threshold: 150.0,
            ..MatchConfig::default()
        });
        let candidate = ImageFingerprint {
            width: 800,
            height: 1000,
            mime: None,
            byte_xxh128: "d".repeat(32),
            phash64: "0000000000000003".to_owned(),
            dhash64: "0000000000000007".to_owned(),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        };

        assert!(matcher.find_for_policy(&candidate, &policy).is_none());
        let outcome = matcher
            .find_preview_for_policy(&candidate, &policy)
            .unwrap();

        assert!(matches!(outcome.confidence, MatchConfidence::Perceptual));
        assert_eq!(
            outcome.diagnostics.representation,
            FingerprintRepresentation::DiscordPreview
        );

        let preview_exact_only = ImageFingerprint {
            width: 800,
            height: 1000,
            mime: None,
            byte_xxh128: "c".repeat(32),
            phash64: "ffffffffffffffff".to_owned(),
            dhash64: "ffffffffffffffff".to_owned(),
            visual: ImageVisualSignature::default(),
            local_anchors: Vec::new(),
            local_hashes: Vec::new(),
        };
        let outcome = matcher
            .find_preview_for_policy(&preview_exact_only, &policy)
            .unwrap();

        assert!(matches!(outcome.confidence, MatchConfidence::ExactXxh128));
        assert_eq!(
            outcome.diagnostics.representation,
            FingerprintRepresentation::DiscordPreview
        );
    }

    #[test]
    fn local_anchors_require_multiple_regions() {
        let thresholds = MatchConfig {
            phash64_max_distance: 0,
            dhash64_max_distance: 0,
            local_anchor_max_distance: 4,
            local_min_anchor_hits: 2,
            local_min_distinct_regions: 2,
            ..MatchConfig::default()
        };
        let anchors = vec![
            ImageAnchor {
                id: "a01".to_owned(),
                x: 0,
                y: 0,
                w: 64,
                h: 32,
                pos_x: 32,
                pos_y: 32,
                hash: "0000000000000000".to_owned(),
                hash2: "0000000000000000".to_owned(),
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                kind: "text_dense".to_owned(),
                region: 1,
                max_distance: 4,
            },
            ImageAnchor {
                id: "a02".to_owned(),
                x: 96,
                y: 48,
                w: 64,
                h: 32,
                pos_x: 128,
                pos_y: 96,
                hash: "00000000000000ff".to_owned(),
                hash2: "00000000000000ff".to_owned(),
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                kind: "text_dense".to_owned(),
                region: 5,
                max_distance: 4,
            },
        ];
        let candidate_hashes = vec![
            LocalImageHash {
                x: 12,
                y: 20,
                w: 64,
                h: 32,
                pos_x: 36,
                pos_y: 36,
                region: 1,
                hash: 0x0000_0000_0000_0003,
                hash2: 0x0000_0000_0000_0003,
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                scale_percent: 100,
                rotation_degrees: 0,
            },
            LocalImageHash {
                x: 108,
                y: 72,
                w: 64,
                h: 32,
                pos_x: 132,
                pos_y: 100,
                region: 5,
                hash: 0x0000_0000_0000_00fc,
                hash2: 0x0000_0000_0000_00fc,
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                scale_percent: 100,
                rotation_degrees: 0,
            },
        ];

        let comparison = compare_local_anchors(&anchors, &candidate_hashes, &thresholds);

        assert!(comparison.matched);
        assert_eq!(comparison.hits, 2);
        assert_eq!(comparison.distinct_regions, 2);
        assert_eq!(comparison.geometry_model, Some(GeometryModel::Similarity));
    }

    #[test]
    fn local_anchor_near_miss_is_suspicious() {
        let thresholds = MatchConfig {
            local_min_anchor_hits: 4,
            local_min_distinct_regions: 2,
            local_suspicious_min_anchor_hits: 3,
            local_suspicious_min_distinct_regions: 2,
            geometry_enable_affine: false,
            geometry_enable_homography: false,
            ..MatchConfig::default()
        };
        let anchors = vec![
            ImageAnchor {
                id: "a01".to_owned(),
                x: 0,
                y: 0,
                w: 64,
                h: 32,
                pos_x: 32,
                pos_y: 32,
                hash: "0000000000000000".to_owned(),
                hash2: "0000000000000000".to_owned(),
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                kind: "text_dense".to_owned(),
                region: 1,
                max_distance: 4,
            },
            ImageAnchor {
                id: "a02".to_owned(),
                x: 96,
                y: 48,
                w: 64,
                h: 32,
                pos_x: 128,
                pos_y: 96,
                hash: "00000000000000ff".to_owned(),
                hash2: "00000000000000ff".to_owned(),
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                kind: "text_dense".to_owned(),
                region: 5,
                max_distance: 4,
            },
            ImageAnchor {
                id: "a03".to_owned(),
                x: 40,
                y: 96,
                w: 64,
                h: 32,
                pos_x: 72,
                pos_y: 160,
                hash: "000000000000ff00".to_owned(),
                hash2: "000000000000ff00".to_owned(),
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                kind: "text_dense".to_owned(),
                region: 9,
                max_distance: 4,
            },
        ];
        let candidate_hashes = vec![
            LocalImageHash {
                x: 12,
                y: 20,
                w: 64,
                h: 32,
                pos_x: 36,
                pos_y: 36,
                region: 1,
                hash: 0x0000_0000_0000_0003,
                hash2: 0x0000_0000_0000_0003,
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                scale_percent: 100,
                rotation_degrees: 0,
            },
            LocalImageHash {
                x: 108,
                y: 72,
                w: 64,
                h: 32,
                pos_x: 132,
                pos_y: 100,
                region: 5,
                hash: 0x0000_0000_0000_00fc,
                hash2: 0x0000_0000_0000_00fc,
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                scale_percent: 100,
                rotation_degrees: 0,
            },
            LocalImageHash {
                x: 52,
                y: 120,
                w: 64,
                h: 32,
                pos_x: 76,
                pos_y: 164,
                region: 9,
                hash: 0x0000_0000_0000_ff03,
                hash2: 0x0000_0000_0000_ff03,
                luma_mean: 128,
                luma_std: 64,
                edge_density: 64,
                scale_percent: 100,
                rotation_degrees: 0,
            },
        ];

        let comparison = compare_local_anchors(&anchors, &candidate_hashes, &thresholds);

        assert!(!comparison.matched);
        assert!(comparison.suspicious);
        assert_eq!(comparison.hits, 3);
        assert_eq!(comparison.geometry_model, Some(GeometryModel::Similarity));
    }

    fn visually_suspicious_image_without_specimen_evidence() -> ImageFingerprint {
        ImageFingerprint {
            width: 800,
            height: 1000,
            mime: None,
            byte_xxh128: "a".repeat(32),
            phash64: "ffffffffffffffff".to_owned(),
            dhash64: "ffffffffffffffff".to_owned(),
            visual: ImageVisualSignature {
                luma_mean: 60,
                luma_std: 60,
                rgb_mean: [60, 61, 62],
                grid_luma: [60; 16],
                text_grid: vec![80; 64],
            },
            local_anchors: Vec::new(),
            local_hashes: (0..200).map(test_local_hash).collect(),
        }
    }

    #[test]
    fn suspicious_cluster_coherence_stays_ocr_backed() {
        let mut left = specimen("ffffffffffffffff", "ffffffffffffffff");
        left.specimen_id = "spm_left".to_owned();
        let mut right = specimen("eeeeeeeeeeeeeeee", "eeeeeeeeeeeeeeee");
        right.specimen_id = "spm_right".to_owned();
        let mut matcher = Matcher::new(vec![left, right]);
        let mut graph = CoherenceGraphBuilder::new(2, 0);
        graph.add_edge(0, 1, 100);
        matcher.coherence_graph = graph.build();

        let mut threshold = DetectionPolicy::default().suspicious.threshold;
        threshold.cluster_coherence = true;
        threshold.cluster_member_score = 25;
        threshold.cluster_coherence_score = 63;
        threshold.cluster_min_size = 2;
        threshold.cluster_hard_score = 63;
        let mut diagnostics = MatchDiagnostics {
            representation: FingerprintRepresentation::Original,
            candidate_short_edge: 1000,
            candidate_area: 1_200_000,
            candidate_aspect: 1000.0 / 1200.0,
            candidate_luma_mean: 0,
            candidate_luma_std: 0,
            candidate_text_grid_mean: 0,
            candidate_text_regions: 0,
            candidate_local_hashes: 0,
            steps: Vec::new(),
        };
        let mut scorer = ClusterScorer::new(ClusterThresholds::new(19, 63, 25, 0, 63, 2));
        let mut per_specimen = HashMap::default();
        per_specimen.insert(
            "spm_left",
            ScoredOutcome {
                score: 36.0,
                outcome: suspicious_local_outcome("spm_left", 36.0),
            },
        );
        per_specimen.insert(
            "spm_right",
            ScoredOutcome {
                score: 38.0,
                outcome: suspicious_local_outcome("spm_right", 38.0),
            },
        );

        let outcome = matcher
            .cluster_coherence_outcome(
                &per_specimen,
                ThresholdSearch {
                    threshold: &threshold,
                    suspicious: true,
                    name: "suspicious",
                    variant: MatchVariant::Original,
                    visual_shape: None,
                },
                &mut diagnostics,
                Some(&mut scorer),
            )
            .unwrap();

        assert!(outcome.suspicious);
        assert!(matches!(
            outcome.confidence,
            MatchConfidence::ClusterCoherence
        ));
        assert!(
            diagnostics
                .steps
                .iter()
                .any(|step| step.step == "cluster_coherence" && step.passed)
        );
    }

    fn suspicious_local_outcome(specimen_id: &str, score: f32) -> MatchOutcome {
        MatchOutcome {
            specimen_id: specimen_id.to_owned(),
            confidence: MatchConfidence::SuspiciousLocalAnchors,
            suspicious: true,
            match_score: Some(score),
            phash64_distance: Some(28),
            dhash64_distance: Some(22),
            local_anchor_hits: Some(50),
            local_distinct_regions: Some(26),
            local_average_distance: Some(11.02),
            local_geometry_model: None,
            diagnostics: MatchDiagnostics {
                representation: FingerprintRepresentation::Original,
                candidate_short_edge: 1000,
                candidate_area: 1_200_000,
                candidate_aspect: 1000.0 / 1200.0,
                candidate_luma_mean: 0,
                candidate_luma_std: 0,
                candidate_text_grid_mean: 0,
                candidate_text_regions: 0,
                candidate_local_hashes: 0,
                steps: Vec::new(),
            },
        }
    }

    fn test_local_hash(index: u32) -> LocalImageHash {
        LocalImageHash {
            x: index % 40,
            y: index / 40,
            w: 64,
            h: 32,
            pos_x: 32,
            pos_y: 32,
            region: index % 16,
            hash: u64::from(index),
            hash2: u64::from(index).rotate_left(17),
            luma_mean: 60,
            luma_std: 60,
            edge_density: 80,
            scale_percent: 100,
            rotation_degrees: 0,
        }
    }
}
