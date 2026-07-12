//! Geometry verification for local image correspondences.
//!
//! The fast path votes for a similarity transform. When enough coherent inliers
//! exist, the verifier can refine that seed into affine or homography models for
//! mildly skewed photos. Every richer model must explain at least as much as the
//! simpler model and pass shape guards before it can become the result.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

#[derive(Clone, Copy, Debug)]
pub struct P {
    pub x: f32,
    pub y: f32,
}

impl P {
    #[inline]
    fn sub(self, other: P) -> P {
        P {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    #[inline]
    fn add(self, other: P) -> P {
        P {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    /// Squared magnitude (x^2 + y^2), fused. Used for all distance comparisons;
    /// take `.sqrt()` of this only when an actual length is needed.
    #[inline]
    fn len2(self) -> f32 {
        self.x.mul_add(self.x, self.y * self.y)
    }
}

/// Complex multiply `z * p`.
#[inline]
fn cmul(z: (f32, f32), p: P) -> P {
    P {
        x: (-z.1).mul_add(p.y, z.0 * p.x),
        y: z.1.mul_add(p.x, z.0 * p.y),
    }
}

// --- 3x3 projective model (subsumes similarity and affine) -----------------
// Row-major. Similarity/affine have bottom row [0,0,1]; homography is general.
type Mat3 = [[f32; 3]; 3];

#[inline]
fn h_from_similarity(z: (f32, f32), t: P) -> Mat3 {
    [[z.0, -z.1, t.x], [z.1, z.0, t.y], [0.0, 0.0, 1.0]]
}

#[inline]
fn h_from_affine(a: [[f32; 2]; 2], b: P) -> Mat3 {
    [
        [a[0][0], a[0][1], b.x],
        [a[1][0], a[1][1], b.y],
        [0.0, 0.0, 1.0],
    ]
}

/// Apply a projective model to a point.
#[inline]
fn h_apply(transform: &Mat3, point: P) -> P {
    let projected_x =
        transform[0][0].mul_add(point.x, transform[0][1].mul_add(point.y, transform[0][2]));
    let projected_y =
        transform[1][0].mul_add(point.x, transform[1][1].mul_add(point.y, transform[1][2]));
    let denominator =
        transform[2][0].mul_add(point.x, transform[2][1].mul_add(point.y, transform[2][2]));
    let inv = 1.0 / denominator;
    P {
        x: projected_x * inv,
        y: projected_y * inv,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Correspondence {
    pub spec: P,
    pub cand: P,
    pub cand_id: u32,
    pub region: u16,
    /// Descriptor distance to the best candidate feature.
    pub hamming: u8,
    /// Distance to the next-best candidate feature, if known.
    ///
    /// `u8::MAX` means "unknown" and disables ambiguity penalties for this
    /// correspondence. Otherwise, `second_hamming - hamming` is the
    /// distinctiveness margin used by ratio filtering and quality ordering.
    pub second_hamming: u8,
}

impl Correspondence {
    /// Distinctiveness margin in bits. Larger means less ambiguous.
    #[inline]
    fn margin(self) -> i32 {
        self.second_hamming as i32 - self.hamming as i32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GeoCfg {
    pub max_correspondences: usize,
    pub min_pair_separation: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub log2_scale_bin: f32,
    pub angle_bin: f32,
    pub max_rotation: f32,
    pub inlier_residual: f32,
    pub min_inliers: usize,
    /// Try an affine model after the similarity seed succeeds.
    pub enable_affine: bool,
    /// Try a homography model after a linear seed succeeds.
    pub enable_homography: bool,
    /// Extra residual budget, in pixels, for affine and homography refinement.
    pub model_slack: f32,
    /// Maximum ratio between affine singular values. Bounds shear.
    pub max_anisotropy: f32,
    /// Maximum perspective foreshortening across the inlier bounding box.
    pub max_perspective: f32,
    /// Drop correspondences whose distinctiveness margin is below this value.
    /// `0` disables the ratio filter.
    pub ratio_min_margin: u8,
    /// Try a bounded quality-ordered homography search if the voter cannot seed.
    pub enable_prosac_fallback: bool,
    /// Hard iteration cap for the fallback search.
    pub prosac_max_iters: u32,
    /// Minimum fallback consensus size. Must be above the 4-point sample size.
    pub prosac_min_inliers: usize,
}

impl Default for GeoCfg {
    fn default() -> Self {
        Self {
            max_correspondences: 64,
            min_pair_separation: 20.0,
            scale_min: 0.5,
            scale_max: 2.0,
            log2_scale_bin: 0.12,
            angle_bin: 4f32.to_radians(),
            max_rotation: 30f32.to_radians(),
            inlier_residual: 8.0,
            min_inliers: 3,
            enable_affine: true,
            enable_homography: true,
            model_slack: 2.0,
            max_anisotropy: 1.6,
            max_perspective: 2.2,
            ratio_min_margin: 0,
            enable_prosac_fallback: true,
            prosac_max_iters: 64,
            prosac_min_inliers: 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Accept {
    pub min_inliers: usize,
    pub min_regions: usize,
    pub min_spread: f32,
    pub max_mean_residual: f32,
    pub max_mean_hamming: f32,
}

/// Model that produced the match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Similarity,
    Affine,
    Homography,
}

#[derive(Clone, Debug)]
pub struct GeoMatch {
    pub scale: f32,
    pub angle: f32,
    /// Projective model used for residuals and projection.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the recovered projective model is retained for diagnostics and geometry tests"
        )
    )]
    pub homography: Mat3,
    pub model: Model,
    pub inlier_count: usize,
    pub region_count: usize,
    pub spread: f32,
    pub mean_residual: f32,
    pub mean_hamming: f32,
    pub inliers: Vec<Correspondence>,
}

#[derive(Default)]
pub struct GeometryScratch {
    correspondences: Vec<Correspondence>,
    translations: Vec<P>,
    preliminary: Vec<Correspondence>,
    eval: Vec<Correspondence>,
}

pub fn verify_geometry_with_scratch(
    correspondences: &[Correspondence],
    cfg: &GeoCfg,
    scratch: &mut GeometryScratch,
) -> Option<GeoMatch> {
    mutual_best_into(correspondences, &mut scratch.correspondences);
    if scratch.correspondences.len() < cfg.min_inliers.max(2) {
        return None;
    }
    // Optional ambiguity filter. Disabled at margin 0.
    if cfg.ratio_min_margin > 0 {
        let m = cfg.ratio_min_margin as i32;
        scratch.correspondences.retain(|c| c.margin() >= m);
        if scratch.correspondences.len() < cfg.min_inliers.max(2) {
            return None;
        }
    }
    // Quality order used by both the voter path and fallback sampler.
    scratch
        .correspondences
        .sort_by(|a, b| b.margin().cmp(&a.margin()).then(a.hamming.cmp(&b.hamming)));
    limit_correspondences(
        &mut scratch.correspondences,
        cfg.max_correspondences,
        cfg.min_inliers.max(2),
    );

    let residual_sq = cfg.inlier_residual * cfg.inlier_residual;
    let model_residual_sq = {
        let r = cfg.inlier_residual + cfg.model_slack;
        r * r
    };

    // Primary path: similarity seed, then optional richer models.
    if let Some(m) = voter_escalate(
        &scratch.correspondences,
        cfg,
        residual_sq,
        model_residual_sq,
        &mut scratch.translations,
        &mut scratch.preliminary,
        &mut scratch.eval,
    ) {
        return Some(m);
    }

    // Fallback only runs when the voter path found no model.
    if cfg.enable_prosac_fallback
        && scratch.correspondences.len() >= 4
        && let Some((inliers, h)) = prosac_homography(&scratch.correspondences, cfg, residual_sq)
        && inliers.len() >= cfg.min_inliers
    {
        let (scale, angle, _, _) = readout(&h, centroid_spec(&inliers));
        return Some(build_match(inliers, Model::Homography, h, scale, angle));
    }
    None
}

fn limit_correspondences(correspondences: &mut Vec<Correspondence>, cap: usize, required: usize) {
    let cap = cap.max(required).min(correspondences.len());
    if correspondences.len() <= cap {
        return;
    }

    let quality_keep = cap.saturating_mul(3).div_ceil(4).max(required).min(cap);
    if quality_keep == cap {
        correspondences.truncate(cap);
        return;
    }

    let tail_keep = cap - quality_keep;
    let tail_len = correspondences.len() - quality_keep;
    let mut limited = Vec::with_capacity(cap);
    limited.extend_from_slice(&correspondences[..quality_keep]);
    for index in 0..tail_keep {
        let tail_index = quality_keep + (index * tail_len / tail_keep);
        limited.push(correspondences[tail_index]);
    }
    *correspondences = limited;
}

// Similarity voter seed, optional affine/homography refinement, then final refit.
fn voter_escalate(
    correspondences: &[Correspondence],
    cfg: &GeoCfg,
    residual_sq: f32,
    model_residual_sq: f32,
    translations: &mut Vec<P>,
    preliminary: &mut Vec<Correspondence>,
    eval: &mut Vec<Correspondence>,
) -> Option<GeoMatch> {
    let (scale, angle) = vote_scale_angle(correspondences, cfg)?;
    let z = (scale * angle.cos(), scale * angle.sin());
    translations.clear();
    translations.reserve(correspondences.len());
    translations.extend(
        correspondences
            .iter()
            .map(|corr| corr.cand.sub(cmul(z, corr.spec))),
    );
    let translation = densest(translations, cfg.inlier_residual)?;
    preliminary.clear();
    preliminary.extend(
        correspondences.iter().copied().filter(|corr| {
            corr.cand.sub(cmul(z, corr.spec)).sub(translation).len2() <= residual_sq
        }),
    );
    if preliminary.len() < cfg.min_inliers.max(2) {
        return None;
    }
    let (refined_z, refined_t) = similarity_lsq(preliminary)?;

    let sim_h = h_from_similarity(refined_z, refined_t);
    let sim_ss = evaluate_into(&sim_h, correspondences, residual_sq, eval);
    if eval.len() < cfg.min_inliers.max(2) {
        return None;
    }
    let mut best = Cand {
        inliers: eval.clone(),
        sumsq: sim_ss,
        model: Model::Similarity,
        h: sim_h,
    };

    // Affine upgrade.
    if cfg.enable_affine
        && best.inliers.len() >= 3
        && let Some((a, b)) = affine_lsq(&best.inliers)
        && affine_ok(a, cfg)
    {
        let h = h_from_affine(a, b);
        let ss = evaluate_into(&h, correspondences, model_residual_sq, eval);
        if better(eval.len(), ss, best.inliers.len(), best.sumsq) {
            best.inliers.clear();
            best.inliers.extend_from_slice(eval);
            best.sumsq = ss;
            best.model = Model::Affine;
            best.h = h;
        }
    }
    // Homography upgrade.
    if cfg.enable_homography
        && best.inliers.len() >= 4
        && !near_collinear(&best.inliers)
        && let Some(h) = homography_lsq(&best.inliers)
        && homography_ok(&h, &best.inliers, cfg)
    {
        let ss = evaluate_into(&h, correspondences, model_residual_sq, eval);
        if better(eval.len(), ss, best.inliers.len(), best.sumsq) {
            best.inliers.clear();
            best.inliers.extend_from_slice(eval);
            best.sumsq = ss;
            best.model = Model::Homography;
            best.h = h;
        }
    }

    // Refit the winning model class on its final inliers.
    let (inliers, model, h) =
        refit_class(&best, correspondences, residual_sq, model_residual_sq, eval);
    if inliers.len() < cfg.min_inliers {
        return None;
    }
    // Keep the similarity readout for diagnostics.
    let scale = refined_z
        .0
        .mul_add(refined_z.0, refined_z.1 * refined_z.1)
        .sqrt();
    let angle = refined_z.1.atan2(refined_z.0);
    Some(build_match(inliers, model, h, scale, angle))
}

// Assemble diagnostics under the final model.
fn build_match(
    inliers: Vec<Correspondence>,
    model: Model,
    h: Mat3,
    scale: f32,
    angle: f32,
) -> GeoMatch {
    let mut residual_sum = 0.0f32;
    let mut hamming_sum = 0.0f32;
    for corr in &inliers {
        residual_sum += h_apply(&h, corr.spec).sub(corr.cand).len2().sqrt();
        hamming_sum += corr.hamming as f32;
    }
    let inlier_count = inliers.len();
    let (spread, region_count) = spread_and_regions(&inliers);
    GeoMatch {
        scale,
        angle,
        homography: h,
        model,
        inlier_count,
        region_count,
        spread,
        mean_residual: residual_sum / inlier_count as f32,
        mean_hamming: hamming_sum / inlier_count as f32,
        inliers,
    }
}

pub fn passes(match_: &GeoMatch, accept: &Accept) -> bool {
    match_.inlier_count >= accept.min_inliers
        && match_.region_count >= accept.min_regions
        && match_.spread >= accept.min_spread
        && match_.mean_residual <= accept.max_mean_residual
        && match_.mean_hamming <= accept.max_mean_hamming
}

// A scored candidate model.
struct Cand {
    inliers: Vec<Correspondence>,
    sumsq: f32,
    model: Model,
    h: Mat3,
}

// Collect inliers of a model over all correspondences (squared-residual test).
fn evaluate(
    h: &Mat3,
    correspondences: &[Correspondence],
    thr_sq: f32,
) -> (Vec<Correspondence>, f32) {
    let mut inliers = Vec::with_capacity(correspondences.len());
    let sumsq = evaluate_into(h, correspondences, thr_sq, &mut inliers);
    (inliers, sumsq)
}

fn evaluate_into(
    h: &Mat3,
    correspondences: &[Correspondence],
    thr_sq: f32,
    inliers: &mut Vec<Correspondence>,
) -> f32 {
    inliers.clear();
    inliers.reserve(correspondences.len());
    let mut sumsq = 0.0f32;
    for corr in correspondences {
        let r = h_apply(h, corr.spec).sub(corr.cand).len2();
        if r <= thr_sq {
            sumsq += r;
            inliers.push(*corr);
        }
    }
    sumsq
}

fn evaluate_count(h: &Mat3, correspondences: &[Correspondence], thr_sq: f32) -> (usize, f32) {
    let mut count = 0usize;
    let mut sumsq = 0.0f32;
    for corr in correspondences {
        let r = h_apply(h, corr.spec).sub(corr.cand).len2();
        if r <= thr_sq {
            count += 1;
            sumsq += r;
        }
    }
    (count, sumsq)
}

// Prefer more inliers, then lower total residual.
#[inline]
fn better(cand_n: usize, cand_ss: f32, best_n: usize, best_ss: f32) -> bool {
    cand_n > best_n || (cand_n == best_n && cand_ss + 1e-3 < best_ss)
}

// Refit the selected model class and keep it only if it preserves coverage.
fn refit_class(
    best: &Cand,
    correspondences: &[Correspondence],
    residual_sq: f32,
    model_residual_sq: f32,
    eval: &mut Vec<Correspondence>,
) -> (Vec<Correspondence>, Model, Mat3) {
    match best.model {
        Model::Homography => {
            if let Some(h) = homography_lsq(&best.inliers) {
                evaluate_into(&h, correspondences, model_residual_sq, eval);
                if eval.len() >= best.inliers.len() {
                    return (eval.clone(), Model::Homography, h);
                }
            }
            (best.inliers.clone(), Model::Homography, best.h)
        }
        Model::Affine => {
            if let Some((a, b)) = affine_lsq(&best.inliers) {
                let h = h_from_affine(a, b);
                evaluate_into(&h, correspondences, model_residual_sq, eval);
                if eval.len() >= best.inliers.len() {
                    return (eval.clone(), Model::Affine, h);
                }
            }
            (best.inliers.clone(), Model::Affine, best.h)
        }
        Model::Similarity => {
            if let Some((z, t)) = similarity_lsq(&best.inliers) {
                let h = h_from_similarity(z, t);
                evaluate_into(&h, correspondences, residual_sq, eval);
                return (eval.clone(), Model::Similarity, h);
            }
            (best.inliers.clone(), Model::Similarity, best.h)
        }
    }
}

fn mutual_best_into(correspondences: &[Correspondence], output: &mut Vec<Correspondence>) {
    output.clear();
    output.extend_from_slice(correspondences);
    output.sort_by_key(|corr| corr.cand_id);

    let mut write = 0usize;
    let mut read = 0usize;
    while read < output.len() {
        let cand_id = output[read].cand_id;
        let mut best = output[read];
        read += 1;
        while read < output.len() && output[read].cand_id == cand_id {
            if output[read].hamming < best.hamming {
                best = output[read];
            }
            read += 1;
        }
        output[write] = best;
        write += 1;
    }
    output.truncate(write);
}

#[derive(Clone, Copy, Default)]
struct Accumulator {
    count: u32,
    log2_scale_sum: f32,
    angle_sum: f32,
}

struct VoteGrid {
    scale_min_bin: i32,
    angle_min_bin: i32,
    scale_bins: usize,
    angle_bins: usize,
    cells: Vec<Accumulator>,
}

impl VoteGrid {
    fn new(cfg: &GeoCfg) -> Option<Self> {
        if !(cfg.scale_min.is_finite()
            && cfg.scale_max.is_finite()
            && cfg.log2_scale_bin.is_finite()
            && cfg.angle_bin.is_finite()
            && cfg.max_rotation.is_finite())
            || cfg.scale_min <= 0.0
            || cfg.scale_max < cfg.scale_min
            || cfg.log2_scale_bin <= 0.0
            || cfg.angle_bin <= 0.0
            || cfg.max_rotation < 0.0
        {
            return None;
        }

        let scale_min_bin = (cfg.scale_min.log2() / cfg.log2_scale_bin).floor() as i32 - 1;
        let scale_max_bin = (cfg.scale_max.log2() / cfg.log2_scale_bin).ceil() as i32 + 1;
        let angle_min_bin = (-cfg.max_rotation / cfg.angle_bin).floor() as i32 - 1;
        let angle_max_bin = (cfg.max_rotation / cfg.angle_bin).ceil() as i32 + 1;
        let scale_bins = usize::try_from(scale_max_bin - scale_min_bin + 1).ok()?;
        let angle_bins = usize::try_from(angle_max_bin - angle_min_bin + 1).ok()?;
        let len = scale_bins.checked_mul(angle_bins)?;
        Some(Self {
            scale_min_bin,
            angle_min_bin,
            scale_bins,
            angle_bins,
            cells: vec![Accumulator::default(); len],
        })
    }

    #[inline]
    fn index(&self, scale_bin: i32, angle_bin: i32) -> Option<usize> {
        let scale = usize::try_from(scale_bin - self.scale_min_bin).ok()?;
        let angle = usize::try_from(angle_bin - self.angle_min_bin).ok()?;
        (scale < self.scale_bins && angle < self.angle_bins)
            .then_some(scale * self.angle_bins + angle)
    }

    #[inline]
    fn add(&mut self, scale_bin: i32, angle_bin: i32, log2_scale: f32, angle: f32) {
        let Some(index) = self.index(scale_bin, angle_bin) else {
            return;
        };
        let entry = &mut self.cells[index];
        entry.count += 1;
        entry.log2_scale_sum += log2_scale;
        entry.angle_sum += angle;
    }

    #[inline]
    fn get(&self, scale_bin: i32, angle_bin: i32) -> Option<&Accumulator> {
        self.index(scale_bin, angle_bin)
            .and_then(|index| self.cells.get(index))
            .filter(|acc| acc.count > 0)
    }

    fn is_empty(&self) -> bool {
        self.cells.iter().all(|acc| acc.count == 0)
    }

    fn non_empty_bins(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, acc)| acc.count > 0)
            .filter_map(|(index, _)| {
                let scale = index / self.angle_bins;
                let angle = index % self.angle_bins;
                Some((
                    self.scale_min_bin + i32::try_from(scale).ok()?,
                    self.angle_min_bin + i32::try_from(angle).ok()?,
                ))
            })
    }
}

fn vote_scale_angle(correspondences: &[Correspondence], cfg: &GeoCfg) -> Option<(f32, f32)> {
    // Squared thresholds so the inner loop never takes a square root.
    let min_separation_sq = cfg.min_pair_separation * cfg.min_pair_separation;
    let scale_min_sq = cfg.scale_min * cfg.scale_min;
    let scale_max_sq = cfg.scale_max * cfg.scale_max;

    let mut bins = VoteGrid::new(cfg)?;
    for (i, corr_i) in correspondences.iter().enumerate() {
        let spec_i = corr_i.spec;
        let cand_i = corr_i.cand;
        for corr_j in correspondences.iter().skip(i + 1) {
            let spec_delta = corr_j.spec.sub(spec_i);
            let cand_delta = corr_j.cand.sub(cand_i);
            let spec_len_sq = spec_delta.len2();
            let cand_len_sq = cand_delta.len2();
            if spec_len_sq < min_separation_sq || cand_len_sq < min_separation_sq {
                continue;
            }
            // Compare squared scale to avoid `sqrt`.
            let scale_sq = cand_len_sq / spec_len_sq;
            if scale_sq < scale_min_sq || scale_sq > scale_max_sq {
                continue;
            }
            // Signed rotation from specimen delta to candidate delta.
            let cross = spec_delta
                .x
                .mul_add(cand_delta.y, -(spec_delta.y * cand_delta.x));
            let dot = spec_delta
                .x
                .mul_add(cand_delta.x, spec_delta.y * cand_delta.y);
            let angle = cross.atan2(dot);
            if angle.abs() > cfg.max_rotation {
                continue;
            }
            // Convert squared scale into log2(scale).
            let log2_scale = 0.5 * scale_sq.log2();
            bins.add(
                (log2_scale / cfg.log2_scale_bin).round() as i32,
                (angle / cfg.angle_bin).round() as i32,
                log2_scale,
                angle,
            );
        }
    }
    if bins.is_empty() {
        return None;
    }

    let mut best_key = (0i32, 0i32);
    let mut best_support = 0u32;
    for (scale_bin, angle_bin) in bins.non_empty_bins() {
        let mut support = 0u32;
        for scale_offset in -1..=1 {
            for angle_offset in -1..=1 {
                if let Some(acc) = bins.get(scale_bin + scale_offset, angle_bin + angle_offset) {
                    support += acc.count;
                }
            }
        }
        if support > best_support {
            best_support = support;
            best_key = (scale_bin, angle_bin);
        }
    }

    let mut count = 0u32;
    let mut log2_scale_sum = 0.0f32;
    let mut angle_sum = 0.0f32;
    for scale_offset in -1..=1 {
        for angle_offset in -1..=1 {
            if let Some(acc) = bins.get(best_key.0 + scale_offset, best_key.1 + angle_offset) {
                count += acc.count;
                log2_scale_sum += acc.log2_scale_sum;
                angle_sum += acc.angle_sum;
            }
        }
    }
    (count > 0).then(|| {
        (
            (log2_scale_sum / count as f32).exp2(),
            angle_sum / count as f32,
        )
    })
}

fn densest(points: &[P], radius: f32) -> Option<P> {
    let radius_sq = radius * radius;
    let mut best = None;
    let mut best_count = 0usize;
    for &point in points {
        let mut count = 0usize;
        let mut sum = P { x: 0.0, y: 0.0 };
        for &other in points {
            if point.sub(other).len2() <= radius_sq {
                count += 1;
                sum = sum.add(other);
            }
        }
        if count > best_count {
            best_count = count;
            best = Some(P {
                x: sum.x / count as f32,
                y: sum.y / count as f32,
            });
        }
    }
    best
}

fn similarity_lsq(correspondences: &[Correspondence]) -> Option<((f32, f32), P)> {
    let count = correspondences.len() as f32;
    let mut spec_center = P { x: 0.0, y: 0.0 };
    let mut cand_center = P { x: 0.0, y: 0.0 };
    for corr in correspondences {
        spec_center = spec_center.add(corr.spec);
        cand_center = cand_center.add(corr.cand);
    }
    spec_center = P {
        x: spec_center.x / count,
        y: spec_center.y / count,
    };
    cand_center = P {
        x: cand_center.x / count,
        y: cand_center.y / count,
    };

    let mut numerator_re = 0.0f32;
    let mut numerator_im = 0.0f32;
    let mut denominator = 0.0f32;
    for corr in correspondences {
        let spec = corr.spec.sub(spec_center);
        let cand = corr.cand.sub(cand_center);
        numerator_re += cand.x.mul_add(spec.x, cand.y * spec.y);
        numerator_im += cand.y.mul_add(spec.x, -(cand.x * spec.y));
        denominator += spec.x.mul_add(spec.x, spec.y * spec.y);
    }
    if denominator < 1e-6 {
        return None;
    }

    let z = (numerator_re / denominator, numerator_im / denominator);
    let translation = cand_center.sub(cmul(z, spec_center));
    Some((z, translation))
}

// --- affine least squares (6 DOF), centered for stability -------------------
// Returns `cand = A * spec + b`.
fn affine_lsq(correspondences: &[Correspondence]) -> Option<([[f32; 2]; 2], P)> {
    let n = correspondences.len() as f32;
    let mut sc = P { x: 0.0, y: 0.0 };
    let mut cc = P { x: 0.0, y: 0.0 };
    for c in correspondences {
        sc = sc.add(c.spec);
        cc = cc.add(c.cand);
    }
    sc = P {
        x: sc.x / n,
        y: sc.y / n,
    };
    cc = P {
        x: cc.x / n,
        y: cc.y / n,
    };

    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    let (mut cxx, mut cxy, mut cyx, mut cyy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for c in correspondences {
        let s = c.spec.sub(sc);
        let d = c.cand.sub(cc);
        sxx = s.x.mul_add(s.x, sxx);
        sxy = s.x.mul_add(s.y, sxy);
        syy = s.y.mul_add(s.y, syy);
        cxx = d.x.mul_add(s.x, cxx);
        cxy = d.x.mul_add(s.y, cxy);
        cyx = d.y.mul_add(s.x, cyx);
        cyy = d.y.mul_add(s.y, cyy);
    }
    let det = sxx.mul_add(syy, -(sxy * sxy));
    if det.abs() < 1e-6 {
        return None;
    }
    let inv = 1.0 / det;
    // Inverse of the 2x2 specimen covariance matrix.
    let a00 = (cxx * syy - cxy * sxy) * inv;
    let a01 = (cxy * sxx - cxx * sxy) * inv;
    let a10 = (cyx * syy - cyy * sxy) * inv;
    let a11 = (cyy * sxx - cyx * sxy) * inv;
    let a = [[a00, a01], [a10, a11]];
    let b = P {
        x: cc.x - a00.mul_add(sc.x, a01 * sc.y),
        y: cc.y - a10.mul_add(sc.x, a11 * sc.y),
    };
    Some((a, b))
}

// Preserve orientation and bound shear.
fn affine_ok(a: [[f32; 2]; 2], cfg: &GeoCfg) -> bool {
    let det = a[0][0].mul_add(a[1][1], -(a[0][1] * a[1][0]));
    if det <= 0.0 {
        return false;
    }
    // Singular values from eigenvalues of A^T A.
    let m00 = a[0][0].mul_add(a[0][0], a[1][0] * a[1][0]);
    let m11 = a[0][1].mul_add(a[0][1], a[1][1] * a[1][1]);
    let m01 = a[0][0].mul_add(a[0][1], a[1][0] * a[1][1]);
    let tr = m00 + m11;
    let disc = (tr * tr - 4.0 * (m00 * m11 - m01 * m01)).max(0.0).sqrt();
    let l_hi = f32::midpoint(tr, disc);
    let l_lo = f32::midpoint(tr, -disc);
    if l_lo <= 1e-9 {
        return false;
    }
    let sv_hi = l_hi.sqrt();
    let sv_lo = l_lo.sqrt();
    sv_hi / sv_lo <= cfg.max_anisotropy
        && sv_lo >= cfg.scale_min * 0.75
        && sv_hi <= cfg.scale_max * 1.33
}

// --- homography via normalized DLT (8 DOF, h22 = 1) -------------------------
fn homography_lsq(correspondences: &[Correspondence]) -> Option<Mat3> {
    // Hartley normalization for specimen and candidate points.
    let (ts, sc, ss) = normalizer(correspondences, true)?;
    let (tc, cc, cs) = normalizer(correspondences, false)?;

    // Accumulate the 8x8 normal equations with h22 fixed to 1.
    let mut m = [[0.0f64; 8]; 8];
    let mut rhs = [0.0f64; 8];
    for c in correspondences {
        let x = ((c.spec.x - sc.x) * ss) as f64;
        let y = ((c.spec.y - sc.y) * ss) as f64;
        let xp = ((c.cand.x - cc.x) * cs) as f64;
        let yp = ((c.cand.y - cc.y) * cs) as f64;
        // x' equation.
        let rx = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp];
        // y' equation.
        let ry = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp];
        for a in 0..8 {
            rhs[a] += rx[a] * xp + ry[a] * yp;
            for b in 0..8 {
                m[a][b] += rx[a] * rx[b] + ry[a] * ry[b];
            }
        }
    }
    let h = solve8(&mut m, &mut rhs)?;
    // Normalized homography (h22 = 1).
    let hn: Mat3 = [
        [h[0] as f32, h[1] as f32, h[2] as f32],
        [h[3] as f32, h[4] as f32, h[5] as f32],
        [h[6] as f32, h[7] as f32, 1.0],
    ];
    // Denormalize from normalized point coordinates.
    Some(mat3_mul(&mat3_mul(&inv_norm(tc), &hn), &ts))
}

// Returns a normalizer that centers points and sets mean distance to sqrt(2).
fn normalizer(corr: &[Correspondence], spec: bool) -> Option<(Mat3, P, f32)> {
    let n = corr.len() as f32;
    let mut c = P { x: 0.0, y: 0.0 };
    for k in corr {
        let p = if spec { k.spec } else { k.cand };
        c = c.add(p);
    }
    c = P {
        x: c.x / n,
        y: c.y / n,
    };
    let mut mean_d = 0.0f32;
    for k in corr {
        let p = if spec { k.spec } else { k.cand };
        mean_d += p.sub(c).len2().sqrt();
    }
    mean_d /= n;
    if mean_d < 1e-6 {
        return None;
    }
    let s = std::f32::consts::SQRT_2 / mean_d;
    let t = [[s, 0.0, -s * c.x], [0.0, s, -s * c.y], [0.0, 0.0, 1.0]];
    Some((t, c, s))
}

#[inline]
fn inv_norm(t: Mat3) -> Mat3 {
    // Inverse of the similarity normalizer returned by `norm_points`.
    let s = t[0][0];
    let e = t[0][2];
    let f = t[1][2];
    let inv_s = 1.0 / s;
    [
        [inv_s, 0.0, -e * inv_s],
        [0.0, inv_s, -f * inv_s],
        [0.0, 0.0, 1.0],
    ]
}

fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0].mul_add(b[0][j], a[i][1].mul_add(b[1][j], a[i][2] * b[2][j]));
        }
    }
    o
}

// 8x8 dense solve, Gaussian elimination with partial pivoting (f64 internally).
fn solve8(m: &mut [[f64; 8]; 8], rhs: &mut [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        // Pivot on the largest remaining coefficient in this column.
        let mut piv = col;
        let mut best = m[col][col].abs();
        for (row_index, row) in m.iter().enumerate().skip(col + 1) {
            let value = row[col].abs();
            if value > best {
                best = value;
                piv = row_index;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            m.swap(col, piv);
            rhs.swap(col, piv);
        }
        let inv = 1.0 / m[col][col];
        let pivot_row = m[col];
        let pivot_rhs = rhs[col];
        for (row_index, row) in m.iter_mut().enumerate().skip(col + 1) {
            let factor = row[col] * inv;
            if factor != 0.0 {
                for (cell, pivot_cell) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                    *cell -= factor * *pivot_cell;
                }
                rhs[row_index] -= factor * pivot_rhs;
            }
        }
    }
    let mut x = [0.0f64; 8];
    for i in (0..8).rev() {
        let mut s = rhs[i];
        for (column, coefficient) in m[i].iter().enumerate().skip(i + 1) {
            s -= *coefficient * x[column];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

// Reject folded or excessively projective models.
fn homography_ok(h: &Mat3, inliers: &[Correspondence], cfg: &GeoCfg) -> bool {
    if h[0][0].mul_add(h[1][1], -(h[0][1] * h[1][0])) <= 0.0 {
        return false;
    }
    // The denominator must stay positive and within the configured range over
    // the inlier box.
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for c in inliers {
        min_x = min_x.min(c.spec.x);
        min_y = min_y.min(c.spec.y);
        max_x = max_x.max(c.spec.x);
        max_y = max_y.max(c.spec.y);
    }
    let corners = [
        P { x: min_x, y: min_y },
        P { x: max_x, y: min_y },
        P { x: max_x, y: max_y },
        P { x: min_x, y: max_y },
    ];
    let mut wmin = f32::MAX;
    let mut wmax = f32::MIN;
    for p in corners {
        let w = h[2][0].mul_add(p.x, h[2][1].mul_add(p.y, h[2][2]));
        wmin = wmin.min(w);
        wmax = wmax.max(w);
    }
    if wmin <= 1e-6 {
        return false;
    }
    wmax / wmin <= cfg.max_perspective
}

// Homography needs points that span a 2D area, not a single line.
fn near_collinear(inliers: &[Correspondence]) -> bool {
    let n = inliers.len() as f32;
    let mut c = P { x: 0.0, y: 0.0 };
    for k in inliers {
        c = c.add(k.spec);
    }
    c = P {
        x: c.x / n,
        y: c.y / n,
    };
    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for k in inliers {
        let d = k.spec.sub(c);
        sxx = d.x.mul_add(d.x, sxx);
        sxy = d.x.mul_add(d.y, sxy);
        syy = d.y.mul_add(d.y, syy);
    }
    let tr = sxx + syy;
    let disc = (tr * tr - 4.0 * (sxx * syy - sxy * sxy)).max(0.0).sqrt();
    let l_hi = f32::midpoint(tr, disc);
    let l_lo = f32::midpoint(tr, -disc);
    // Ratio of spread along minor vs major axis.
    l_hi <= 1e-6 || (l_lo / l_hi) < 1e-3
}

fn spread_and_regions(inliers: &[Correspondence]) -> (f32, usize) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut regions = Vec::new();
    for corr in inliers {
        min_x = min_x.min(corr.cand.x);
        min_y = min_y.min(corr.cand.y);
        max_x = max_x.max(corr.cand.x);
        max_y = max_y.max(corr.cand.y);
        if !regions.contains(&corr.region) {
            regions.push(corr.region);
        }
    }
    ((max_x - min_x).min(max_y - min_y), regions.len())
}

// --- quality-ordered fallback homography search -----------------------------
// Samples 4-point homographies from a growing prefix of the ordered matches.
fn prosac_homography(
    corr: &[Correspondence],
    cfg: &GeoCfg,
    thr_sq: f32,
) -> Option<(Vec<Correspondence>, Mat3)> {
    let n = corr.len();
    if n < 4 {
        return None;
    }
    let mut rng = seed_rng(corr);
    let mut best_inliers: Vec<Correspondence> = Vec::new();
    let mut best_ss = f32::MAX;
    let mut best_h: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let max_iters = cfg.prosac_max_iters.max(1) as usize;
    // Grow the sampling prefix from 4 up to n across the iteration budget.
    let grow_every = (max_iters / n.max(1)).max(1);
    let mut pool = 4usize.min(n);
    // Stop early when consensus covers most of the set.
    let strong = (n - n / 8).max(cfg.min_inliers + 2);

    for t in 0..max_iters {
        if t > 0 && t % grow_every == 0 && pool < n {
            pool += 1;
        }
        let s = sample4(&mut rng, pool);
        let pts = [corr[s[0]], corr[s[1]], corr[s[2]], corr[s[3]]];
        if degenerate4(&pts) {
            continue;
        }
        let Some(h) = homography_lsq(&pts) else {
            continue;
        };
        if !homography_ok(&h, &pts, cfg) {
            continue;
        }
        let (count, ss) = evaluate_count(&h, corr, thr_sq);
        if count > best_inliers.len() || (count == best_inliers.len() && ss < best_ss) {
            let (inl, _) = evaluate(&h, corr, thr_sq);
            best_ss = ss;
            best_h = h;
            best_inliers = inl;
            if best_inliers.len() >= strong {
                break;
            }
        }
    }

    // Consensus must be above the minimal sample and span multiple regions.
    let floor = cfg.prosac_min_inliers.max(cfg.min_inliers).max(5);
    if best_inliers.len() < floor || near_collinear(&best_inliers) {
        return None;
    }
    if distinct_regions(&best_inliers) < 2 {
        return None;
    }
    // Refit on the full consensus set.
    if let Some(h) = homography_lsq(&best_inliers)
        && homography_ok(&h, &best_inliers, cfg)
    {
        let (inl, _) = evaluate(&h, corr, thr_sq);
        if inl.len() >= best_inliers.len() {
            return Some((inl, h));
        }
    }
    Some((best_inliers, best_h))
}

fn distinct_regions(inliers: &[Correspondence]) -> usize {
    let mut seen: Vec<u16> = Vec::new();
    for c in inliers {
        if !seen.contains(&c.region) {
            seen.push(c.region);
        }
    }
    seen.len()
}

struct XorShift(u64);
impl XorShift {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    fn below(&mut self, m: usize) -> usize {
        (self.next_u64() % m as u64) as usize
    }
}

// Deterministic seed from the input.
fn seed_rng(corr: &[Correspondence]) -> XorShift {
    let mut s = 0x9E37_79B9_7F4A_7C15_u64;
    for c in corr {
        s ^= (c.cand_id as u64).wrapping_mul(0x0100_0000_01B3);
        s = s.rotate_left(7) ^ (c.spec.x.to_bits() as u64) ^ ((c.cand.y.to_bits() as u64) << 32);
    }
    if s == 0 {
        s = 0xDEAD_BEEF;
    }
    XorShift(s)
}

// 4 distinct indices in [0, pool).
fn sample4(rng: &mut XorShift, pool: usize) -> [usize; 4] {
    let p = pool.max(4);
    let mut idx = [0usize; 4];
    let mut k = 0;
    while k < 4 {
        let r = rng.below(p);
        if !idx[..k].contains(&r) {
            idx[k] = r;
            k += 1;
        }
    }
    idx
}

// Reject minimal sets that cannot determine a homography.
fn degenerate4(p: &[Correspondence; 4]) -> bool {
    const EPS: f32 = 1.0; // twice-triangle-area threshold
    let tri = |a: P, b: P, c: P| {
        let u = b.sub(a);
        let v = c.sub(a);
        u.x.mul_add(v.y, -(u.y * v.x)).abs()
    };
    for &(i, j, k) in &[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        if tri(p[i].spec, p[j].spec, p[k].spec) < EPS {
            return true;
        }
        if tri(p[i].cand, p[j].cand, p[k].cand) < EPS {
            return true;
        }
    }
    false
}

fn centroid_spec(inliers: &[Correspondence]) -> P {
    let n = inliers.len() as f32;
    let mut c = P { x: 0.0, y: 0.0 };
    for k in inliers {
        c = c.add(k.spec);
    }
    P {
        x: c.x / n,
        y: c.y / n,
    }
}

// Report scale and angle from the local Jacobian of the projective map.
fn readout(h: &Mat3, at: P) -> (f32, f32, (f32, f32), P) {
    let img = h_apply(h, at);
    let w = h[2][0].mul_add(at.x, h[2][1].mul_add(at.y, h[2][2]));
    let inv = 1.0 / w;
    let jxx = (h[0][0] - img.x * h[2][0]) * inv;
    let jxy = (h[0][1] - img.x * h[2][1]) * inv;
    let jyx = (h[1][0] - img.y * h[2][0]) * inv;
    let jyy = (h[1][1] - img.y * h[2][1]) * inv;
    let scale = jxx.mul_add(jyy, -(jxy * jyx)).abs().sqrt();
    let angle = (jyx - jxy).atan2(jxx + jyy);
    (
        scale,
        angle,
        (scale * angle.cos(), scale * angle.sin()),
        P {
            x: h[0][2],
            y: h[1][2],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    fn corr(spec: P, cand: P, id: u32, region: u16) -> Correspondence {
        Correspondence {
            spec,
            cand,
            cand_id: id,
            region,
            hamming: 2,
            second_hamming: u8::MAX,
        }
    }

    fn vote_scale_angle_map_reference(
        correspondences: &[Correspondence],
        cfg: &GeoCfg,
    ) -> Option<(f32, f32)> {
        let min_separation_sq = cfg.min_pair_separation * cfg.min_pair_separation;
        let scale_min_sq = cfg.scale_min * cfg.scale_min;
        let scale_max_sq = cfg.scale_max * cfg.scale_max;
        let mut bins: FxHashMap<(i32, i32), Accumulator> = FxHashMap::default();

        for (i, corr_i) in correspondences.iter().enumerate() {
            let spec_i = corr_i.spec;
            let cand_i = corr_i.cand;
            for corr_j in correspondences.iter().skip(i + 1) {
                let spec_delta = corr_j.spec.sub(spec_i);
                let cand_delta = corr_j.cand.sub(cand_i);
                let spec_len_sq = spec_delta.len2();
                let cand_len_sq = cand_delta.len2();
                if spec_len_sq < min_separation_sq || cand_len_sq < min_separation_sq {
                    continue;
                }
                let scale_sq = cand_len_sq / spec_len_sq;
                if scale_sq < scale_min_sq || scale_sq > scale_max_sq {
                    continue;
                }
                let cross = spec_delta
                    .x
                    .mul_add(cand_delta.y, -(spec_delta.y * cand_delta.x));
                let dot = spec_delta
                    .x
                    .mul_add(cand_delta.x, spec_delta.y * cand_delta.y);
                let angle = cross.atan2(dot);
                if angle.abs() > cfg.max_rotation {
                    continue;
                }
                let log2_scale = 0.5 * scale_sq.log2();
                let entry = bins
                    .entry((
                        (log2_scale / cfg.log2_scale_bin).round() as i32,
                        (angle / cfg.angle_bin).round() as i32,
                    ))
                    .or_default();
                entry.count += 1;
                entry.log2_scale_sum += log2_scale;
                entry.angle_sum += angle;
            }
        }

        let mut best_key = (0i32, 0i32);
        let mut best_support = 0u32;
        for &(scale_bin, angle_bin) in bins.keys() {
            let mut support = 0u32;
            for scale_offset in -1..=1 {
                for angle_offset in -1..=1 {
                    if let Some(acc) =
                        bins.get(&(scale_bin + scale_offset, angle_bin + angle_offset))
                    {
                        support += acc.count;
                    }
                }
            }
            if support > best_support {
                best_support = support;
                best_key = (scale_bin, angle_bin);
            }
        }

        let mut count = 0u32;
        let mut log2_scale_sum = 0.0f32;
        let mut angle_sum = 0.0f32;
        for scale_offset in -1..=1 {
            for angle_offset in -1..=1 {
                if let Some(acc) = bins.get(&(best_key.0 + scale_offset, best_key.1 + angle_offset))
                {
                    count += acc.count;
                    log2_scale_sum += acc.log2_scale_sum;
                    angle_sum += acc.angle_sum;
                }
            }
        }
        (count > 0).then(|| {
            (
                (log2_scale_sum / count as f32).exp2(),
                angle_sum / count as f32,
            )
        })
    }

    #[test]
    fn flat_vote_grid_matches_map_voter_reference() {
        let angle = 11f32.to_radians();
        let h = h_from_similarity(
            (1.17 * angle.cos(), 1.17 * angle.sin()),
            P { x: 31.0, y: -22.0 },
        );
        let mut cs = Vec::new();
        for row in 0..5 {
            for col in 0..7 {
                let index = row * 7 + col;
                let spec = P {
                    x: 80.0 + col as f32 * 43.0,
                    y: 70.0 + row as f32 * 51.0,
                };
                let mut cand = h_apply(&h, spec);
                cand.x += (index % 3) as f32 - 1.0;
                cand.y += ((index + 1) % 3) as f32 - 1.0;
                cs.push(corr(spec, cand, index as u32, (index % 8) as u16));
            }
        }
        for index in 0..8 {
            cs.push(corr(
                P {
                    x: 20.0 + index as f32 * 15.0,
                    y: 410.0,
                },
                P {
                    x: 430.0 - index as f32 * 20.0,
                    y: 30.0 + index as f32 * 9.0,
                },
                100 + index,
                10,
            ));
        }

        let cfg = GeoCfg::default();
        let expected = vote_scale_angle_map_reference(&cs, &cfg).expect("map reference votes");
        let actual = vote_scale_angle(&cs, &cfg).expect("flat grid votes");
        assert!((actual.0 - expected.0).abs() < 1e-6);
        assert!((actual.1 - expected.1).abs() < 1e-6);
    }

    #[test]
    fn recovers_similarity_with_outliers() {
        let a = 10f32.to_radians();
        let z = (0.8 * a.cos(), 0.8 * a.sin());
        let h = h_from_similarity(z, P { x: 50.0, y: 30.0 });
        let pts = [
            P { x: 100., y: 120. },
            P { x: 300., y: 140. },
            P { x: 280., y: 360. },
            P { x: 120., y: 340. },
            P { x: 210., y: 250. },
        ];
        let mut cs: Vec<_> = pts
            .iter()
            .enumerate()
            .map(|(i, &p)| corr(p, h_apply(&h, p), i as u32, i as u16))
            .collect();
        for o in 0u32..3 {
            cs.push(Correspondence {
                spec: P {
                    x: 50. + o as f32,
                    y: 60.,
                },
                cand: P {
                    x: 400. - 40. * o as f32,
                    y: 30. + 17. * o as f32,
                },
                cand_id: 100 + o,
                region: 9,
                hamming: 5,
                second_hamming: u8::MAX,
            });
        }
        let m = verify_geometry_with_scratch(
            &cs,
            &GeoCfg {
                inlier_residual: 6.0,
                ..Default::default()
            },
            &mut GeometryScratch::default(),
        )
        .expect("should verify");
        assert_eq!(m.inlier_count, 5);
        assert!((m.scale - 0.8).abs() < 0.03);
        assert!(m.region_count >= 3);
        let proj = h_apply(&m.homography, pts[0]);
        assert!(proj.sub(h_apply(&h, pts[0])).len2().sqrt() < 1.0);
    }

    #[test]
    fn rejects_incoherent_matches() {
        let cs: Vec<_> = (0u32..8)
            .map(|i| {
                let v = i as f32;
                corr(
                    P {
                        x: 30. * v,
                        y: (v * 1.7).sin() * 100. + 100.,
                    },
                    P {
                        x: 400. - 23. * v,
                        y: 50. + (v * 0.9).cos() * 120.,
                    },
                    i,
                    i as u16,
                )
            })
            .collect();
        let cfg = GeoCfg {
            inlier_residual: 6.0,
            min_inliers: 5,
            ..Default::default()
        };
        let m = verify_geometry_with_scratch(&cs, &cfg, &mut GeometryScratch::default());
        assert!(m.is_none() || m.unwrap().inlier_count < 5);
    }

    #[test]
    fn recovers_perspective_that_similarity_misses() {
        let a = 5f32.to_radians();
        let sim = h_from_similarity((0.9 * a.cos(), 0.9 * a.sin()), P { x: 20.0, y: 15.0 });
        let persp: Mat3 = [[1., 0., 0.], [0., 1., 0.], [0.0009, 0.0007, 1.]];
        let h = mat3_mul(&sim, &persp);
        let pts = [
            P { x: 90., y: 100. },
            P { x: 360., y: 110. },
            P { x: 380., y: 330. },
            P { x: 110., y: 350. },
            P { x: 240., y: 150. },
            P { x: 200., y: 300. },
            P { x: 300., y: 240. },
        ];
        let cs: Vec<_> = pts
            .iter()
            .enumerate()
            .map(|(i, &p)| corr(p, h_apply(&h, p), i as u32, (i % 4) as u16))
            .collect();
        let mut scratch = GeometryScratch::default();
        let off = verify_geometry_with_scratch(
            &cs,
            &GeoCfg {
                enable_affine: false,
                enable_homography: false,
                ..Default::default()
            },
            &mut scratch,
        )
        .unwrap();
        let on = verify_geometry_with_scratch(&cs, &GeoCfg::default(), &mut scratch).unwrap();
        // The richer model should explain the same points with lower residual.
        assert!(on.inlier_count >= off.inlier_count);
        assert!(on.mean_residual < off.mean_residual);
        assert_ne!(on.model, Model::Similarity);
    }

    #[test]
    fn affine_disabled_flags_reproduce_similarity() {
        // Disabling upgrades keeps the result in the similarity model.
        let a = 8f32.to_radians();
        let h = h_from_similarity((1.1 * a.cos(), 1.1 * a.sin()), P { x: 12.0, y: -9.0 });
        let pts = [
            P { x: 80., y: 90. },
            P { x: 320., y: 120. },
            P { x: 300., y: 300. },
            P { x: 100., y: 280. },
            P { x: 200., y: 200. },
        ];
        let cs: Vec<_> = pts
            .iter()
            .enumerate()
            .map(|(i, &p)| corr(p, h_apply(&h, p), i as u32, i as u16))
            .collect();
        let m = verify_geometry_with_scratch(
            &cs,
            &GeoCfg {
                enable_affine: false,
                enable_homography: false,
                ..Default::default()
            },
            &mut GeometryScratch::default(),
        )
        .unwrap();
        assert_eq!(m.model, Model::Similarity);
        assert_eq!(m.inlier_count, 5);
    }

    #[test]
    fn prosac_recovers_when_voter_cannot_seed() {
        // Strong perspective case recovered by fallback homography search.
        let a = 4f32.to_radians();
        let sim = h_from_similarity((0.95 * a.cos(), 0.95 * a.sin()), P { x: 10.0, y: 8.0 });
        let persp: Mat3 = [[1., 0., 0.], [0., 1., 0.], [0.0022, 0.0018, 1.]];
        let h = mat3_mul(&sim, &persp);
        let pts = [
            P { x: 80., y: 90. },
            P { x: 360., y: 100. },
            P { x: 380., y: 330. },
            P { x: 100., y: 350. },
            P { x: 230., y: 160. },
            P { x: 210., y: 300. },
            P { x: 300., y: 230. },
            P { x: 160., y: 210. },
        ];
        let mut cs: Vec<_> = pts
            .iter()
            .enumerate()
            .map(|(i, &p)| corr(p, h_apply(&h, p), i as u32, (i % 4) as u16))
            .collect();
        for o in 0u32..4 {
            cs.push(Correspondence {
                spec: P {
                    x: 40. + 11. * o as f32,
                    y: 50. + 7. * o as f32,
                },
                cand: P {
                    x: 470. - 33. * o as f32,
                    y: 40. + 19. * o as f32,
                },
                cand_id: 200 + o,
                region: 8,
                hamming: 6,
                second_hamming: u8::MAX,
            });
        }
        // Force the fallback by making the voter rotation window too small.
        let cfg = GeoCfg {
            max_rotation: 0.0001,
            enable_prosac_fallback: true,
            ..Default::default()
        };
        let m = verify_geometry_with_scratch(&cs, &cfg, &mut GeometryScratch::default())
            .expect("PROSAC should recover the homography");
        assert_eq!(m.model, Model::Homography);
        assert!(m.inlier_count >= 7);
    }

    #[test]
    fn ratio_prefilter_drops_ambiguous_matches() {
        // Distinctive inliers plus ambiguous near-tie outliers.
        let a = 6f32.to_radians();
        let h = h_from_similarity((1.0 * a.cos(), 1.0 * a.sin()), P { x: 15.0, y: 10.0 });
        let pts = [
            P { x: 90., y: 100. },
            P { x: 340., y: 120. },
            P { x: 320., y: 320. },
            P { x: 110., y: 300. },
            P { x: 220., y: 210. },
        ];
        let mut cs: Vec<_> = pts
            .iter()
            .enumerate()
            .map(|(i, &p)| Correspondence {
                spec: p,
                cand: h_apply(&h, p),
                cand_id: i as u32,
                region: i as u16,
                hamming: 4,
                second_hamming: 80,
            })
            .collect();
        for o in 0u32..6 {
            cs.push(Correspondence {
                spec: P {
                    x: 60. + 9. * o as f32,
                    y: 70.,
                },
                cand: P {
                    x: 450. - 30. * o as f32,
                    y: 60.,
                },
                cand_id: 50 + o,
                region: 7,
                hamming: 20,
                second_hamming: 22,
            });
        }
        let with = verify_geometry_with_scratch(
            &cs,
            &GeoCfg {
                ratio_min_margin: 10,
                ..Default::default()
            },
            &mut GeometryScratch::default(),
        )
        .unwrap();
        // Every surviving inlier passed the distinctiveness margin.
        assert!(with.inliers.iter().all(|c| c.margin() >= 10));
        assert_eq!(with.inlier_count, 5);
    }
}
