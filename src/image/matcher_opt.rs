//! Small performance primitives used by the image matcher.
//!
//! This module is intentionally narrow: it contains safe nightly SIMD helpers and
//! compact segment indexes used during candidate generation. The functions here
//! are tested against straightforward reference implementations in the test
//! module.

#[inline]
pub fn hamming(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

#[allow(clippy::cast_possible_truncation)]
const fn build_hex_lut() -> [u8; 256] {
    let mut lut = [0xFFu8; 256];
    let mut c = 0u32;
    while c < 256 {
        let b = c as u8;
        let v = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => 0xFF,
        };
        lut[c as usize] = v;
        c += 1;
    }
    lut
}
static HEX_LUT: [u8; 256] = build_hex_lut();

/// Parse exactly 16 hex chars to a `u64`.
///
/// Returns `None` for wrong length or any non-hex byte.
#[inline]
pub fn hex16_to_u64(value: &str) -> Option<u64> {
    let b = value.as_bytes();
    if b.len() != 16 {
        return None;
    }
    let mut out = 0u64;
    for &c in b {
        let n = HEX_LUT[c as usize];
        if n == 0xFF {
            return None;
        }
        out = (out << 4) | u64::from(n);
    }
    Some(out)
}

/// Sum of absolute differences over equal-length byte slices.
#[inline]
pub fn byte_sad(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(feature = "nightly-simd")]
    {
        byte_sad_simd(a, b)
    }

    #[cfg(not(feature = "nightly-simd"))]
    {
        byte_sad_scalar(a, b)
    }
}

#[cfg(not(feature = "nightly-simd"))]
#[inline]
fn byte_sad_scalar(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| u32::from(x.abs_diff(*y)))
        .sum()
}

#[cfg(feature = "nightly-simd")]
#[inline]
fn byte_sad_simd(a: &[u8], b: &[u8]) -> u32 {
    const LANES: usize = 32;

    use std::simd::{Simd, cmp::SimdOrd, num::SimdUint};
    debug_assert_eq!(a.len(), b.len());

    let mut acc: u32 = 0;
    let (chunks_a, remainder_a) = a.as_chunks::<LANES>();
    let (chunks_b, remainder_b) = b.as_chunks::<LANES>();
    for (xa, xb) in chunks_a.iter().zip(chunks_b) {
        let va = Simd::<u8, LANES>::from_slice(xa);
        let vb = Simd::<u8, LANES>::from_slice(xb);
        let diff = va.simd_max(vb) - va.simd_min(vb); // |a-b|, no overflow in u8
        let widened: Simd<u16, LANES> = diff.cast(); // 32*255 fits u16
        acc += u32::from(widened.reduce_sum());
    }
    acc + remainder_a
        .iter()
        .zip(remainder_b)
        .map(|(x, y)| u32::from(x.abs_diff(*y)))
        .sum::<u32>()
}

/// Mean absolute byte delta, clamped to 255.
#[inline]
pub fn mean_abs_delta(a: &[u8], b: &[u8]) -> u8 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0;
    }
    let len = u32::try_from(a.len()).unwrap_or(u32::MAX);
    let mean = (byte_sad(a, b) / len).min(u32::from(u8::MAX));
    u8::try_from(mean).unwrap_or(u8::MAX)
}

/// Mean absolute delta for the expected 8x8 text-density grid.
///
/// Returns `u8::MAX` if either input is not exactly 64 bytes.
#[inline]
pub fn text_grid_mean_delta(specimen: &[u8], candidate: &[u8]) -> u8 {
    if specimen.len() != 64 || candidate.len() != 64 {
        return u8::MAX;
    }
    mean_abs_delta(specimen, candidate)
}

/// Byte features compared by the local-feature gates for one descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorFeatures {
    pub luma_mean: u8,
    pub luma_std: u8,
    pub edge_density: u8,
    pub pos_x: u8,
    pub pos_y: u8,
}

/// Maximum absolute per-feature deltas for a compatibility scan.
///
/// Every delta is inclusive, so `u8::MAX` disables that comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureGate {
    pub luma_mean: u8,
    pub luma_std: u8,
    pub edge_density: u8,
    pub position: u8,
    /// Restrict the scan to entries flagged eligible when the lanes were built.
    pub require_eligible: bool,
}

/// Lanes processed per SIMD block; also the padding granularity.
pub const FEATURE_LANES: usize = 32;

/// Structure-of-arrays copy of the byte features of a descriptor list.
///
/// The per-descriptor gates in the matcher are five `abs_diff <= delta` checks
/// on `u8` features. Laying the features out one array per feature lets a
/// whole block of descriptors be gated with a handful of vector compares, and
/// the resulting bitmask is walked in ascending index order so callers observe
/// exactly the sequence a scalar `filter` would produce.
#[derive(Debug, Clone, Default)]
pub struct FeatureLanes {
    len: usize,
    luma_mean: Vec<u8>,
    luma_std: Vec<u8>,
    edge_density: Vec<u8>,
    pos_x: Vec<u8>,
    pos_y: Vec<u8>,
    eligible: Vec<u8>,
}

impl FeatureLanes {
    /// Builds lanes from `(features, eligible)` pairs in index order.
    pub fn from_entries(entries: impl Iterator<Item = (DescriptorFeatures, bool)>) -> Self {
        let (lower, _) = entries.size_hint();
        let capacity = lower.next_multiple_of(FEATURE_LANES);
        let mut lanes = Self {
            len: 0,
            luma_mean: Vec::with_capacity(capacity),
            luma_std: Vec::with_capacity(capacity),
            edge_density: Vec::with_capacity(capacity),
            pos_x: Vec::with_capacity(capacity),
            pos_y: Vec::with_capacity(capacity),
            eligible: Vec::with_capacity(capacity),
        };
        for (features, eligible) in entries {
            lanes.luma_mean.push(features.luma_mean);
            lanes.luma_std.push(features.luma_std);
            lanes.edge_density.push(features.edge_density);
            lanes.pos_x.push(features.pos_x);
            lanes.pos_y.push(features.pos_y);
            lanes.eligible.push(u8::from(eligible));
            lanes.len += 1;
        }
        let padded = lanes.len.next_multiple_of(FEATURE_LANES);
        lanes.luma_mean.resize(padded, 0);
        lanes.luma_std.resize(padded, 0);
        lanes.edge_density.resize(padded, 0);
        lanes.pos_x.resize(padded, 0);
        lanes.pos_y.resize(padded, 0);
        lanes.eligible.resize(padded, 0);
        lanes
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg(any(not(feature = "nightly-simd"), test))]
    #[inline]
    fn entry_compatible(&self, index: usize, query: DescriptorFeatures, gate: FeatureGate) -> bool {
        self.luma_mean[index].abs_diff(query.luma_mean) <= gate.luma_mean
            && self.luma_std[index].abs_diff(query.luma_std) <= gate.luma_std
            && self.edge_density[index].abs_diff(query.edge_density) <= gate.edge_density
            && self.pos_x[index].abs_diff(query.pos_x) <= gate.position
            && self.pos_y[index].abs_diff(query.pos_y) <= gate.position
            && (!gate.require_eligible || self.eligible[index] == 1)
    }

    /// Calls `visit` with the index of every entry within `gate`, ascending.
    #[inline]
    pub fn for_each_compatible(
        &self,
        query: DescriptorFeatures,
        gate: FeatureGate,
        visit: impl FnMut(usize),
    ) {
        #[cfg(feature = "nightly-simd")]
        {
            self.for_each_compatible_simd(query, gate, visit);
        }

        #[cfg(not(feature = "nightly-simd"))]
        {
            self.for_each_compatible_scalar(query, gate, visit);
        }
    }

    #[cfg(any(not(feature = "nightly-simd"), test))]
    fn for_each_compatible_scalar(
        &self,
        query: DescriptorFeatures,
        gate: FeatureGate,
        mut visit: impl FnMut(usize),
    ) {
        for index in 0..self.len {
            if self.entry_compatible(index, query, gate) {
                visit(index);
            }
        }
    }

    #[cfg(feature = "nightly-simd")]
    fn for_each_compatible_simd(
        &self,
        query: DescriptorFeatures,
        gate: FeatureGate,
        mut visit: impl FnMut(usize),
    ) {
        use std::simd::{
            Simd,
            cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd},
        };
        type Lanes = Simd<u8, FEATURE_LANES>;

        #[inline]
        fn within(values: Lanes, query: Lanes, delta: Lanes) -> std::simd::Mask<i8, FEATURE_LANES> {
            // |values - query| without overflow, then the inclusive gate.
            (values.simd_max(query) - values.simd_min(query)).simd_le(delta)
        }

        let query_luma_mean = Lanes::splat(query.luma_mean);
        let query_luma_std = Lanes::splat(query.luma_std);
        let query_edge_density = Lanes::splat(query.edge_density);
        let query_pos_x = Lanes::splat(query.pos_x);
        let query_pos_y = Lanes::splat(query.pos_y);
        let gate_luma_mean = Lanes::splat(gate.luma_mean);
        let gate_luma_std = Lanes::splat(gate.luma_std);
        let gate_edge_density = Lanes::splat(gate.edge_density);
        let gate_position = Lanes::splat(gate.position);
        let eligible = Lanes::splat(1);

        let (luma_mean, _) = self.luma_mean.as_chunks::<FEATURE_LANES>();
        let (luma_std, _) = self.luma_std.as_chunks::<FEATURE_LANES>();
        let (edge_density, _) = self.edge_density.as_chunks::<FEATURE_LANES>();
        let (pos_x, _) = self.pos_x.as_chunks::<FEATURE_LANES>();
        let (pos_y, _) = self.pos_y.as_chunks::<FEATURE_LANES>();
        let (eligible_flags, _) = self.eligible.as_chunks::<FEATURE_LANES>();
        debug_assert_eq!(luma_mean.len() * FEATURE_LANES, self.luma_mean.len());

        for block in 0..luma_mean.len() {
            let start = block * FEATURE_LANES;
            let mut mask = within(
                Lanes::from_array(luma_mean[block]),
                query_luma_mean,
                gate_luma_mean,
            );
            mask &= within(
                Lanes::from_array(luma_std[block]),
                query_luma_std,
                gate_luma_std,
            );
            mask &= within(
                Lanes::from_array(edge_density[block]),
                query_edge_density,
                gate_edge_density,
            );
            mask &= within(Lanes::from_array(pos_x[block]), query_pos_x, gate_position);
            mask &= within(Lanes::from_array(pos_y[block]), query_pos_y, gate_position);
            if gate.require_eligible {
                mask &= Lanes::from_array(eligible_flags[block]).simd_eq(eligible);
            }
            let mut bits = mask.to_bitmask();
            let remaining = self.len - start;
            if remaining < FEATURE_LANES {
                bits &= (1u64 << remaining) - 1;
            }
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                visit(start + lane);
            }
        }
    }
}

pub const HAMMING_INDEX_SEGMENTS: u8 = 16;
pub const HAMMING_SEGMENT_BITS: u8 = 4;
pub const HAMMING_SEGMENT_MASK: u64 = (1 << HAMMING_SEGMENT_BITS) - 1;
pub const HAMMING_FLAT_SLOTS: usize =
    (HAMMING_INDEX_SEGMENTS as usize) * (1usize << HAMMING_SEGMENT_BITS); // 256

pub const DENSE_LOCAL_INDEX_SEGMENTS: u8 = 8;
pub const DENSE_LOCAL_SEGMENT_BITS: u8 = 8;
pub const DENSE_LOCAL_SEGMENT_MASK: u64 = (1 << DENSE_LOCAL_SEGMENT_BITS) - 1;
pub const DENSE_LOCAL_FLAT_SLOTS: usize =
    (DENSE_LOCAL_INDEX_SEGMENTS as usize) * (1usize << DENSE_LOCAL_SEGMENT_BITS); // 2048

#[derive(Debug, Clone)]
pub struct FlatSegmentIndex<T> {
    slots: Box<[Vec<T>]>,
}

impl<T> FlatSegmentIndex<T> {
    /// `n` is the slot count, such as `HAMMING_FLAT_SLOTS` or `DENSE_LOCAL_FLAT_SLOTS`.
    pub fn with_slots(n: usize) -> Self {
        let slots = (0..n)
            .map(|_| Vec::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { slots }
    }

    #[inline]
    pub fn push(&mut self, slot: usize, value: T) {
        self.slots[slot].push(value);
    }

    #[inline]
    pub fn get(&self, slot: usize) -> &[T] {
        &self.slots[slot]
    }

    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
    }

    /// Bucket lengths, for occupancy stats.
    pub fn slot_lens(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().map(Vec::len)
    }
}

/// Map each 4-bit segment of `hash` to a dense slot id in [0, 256).
#[inline]
pub fn hamming_segments_flat(hash: u64) -> impl Iterator<Item = usize> {
    (0..HAMMING_INDEX_SEGMENTS).map(move |segment| {
        let shift = (HAMMING_INDEX_SEGMENTS - segment - 1) * HAMMING_SEGMENT_BITS;
        let value = ((hash >> shift) & HAMMING_SEGMENT_MASK) as usize;
        (segment as usize) * (1usize << HAMMING_SEGMENT_BITS) + value
    })
}

/// Map each 8-bit segment of `hash` to a dense slot id in [0, 2048).
#[inline]
pub fn dense_local_segments_flat(hash: u64) -> impl Iterator<Item = usize> {
    (0..DENSE_LOCAL_INDEX_SEGMENTS).map(move |segment| {
        let shift = (DENSE_LOCAL_INDEX_SEGMENTS - segment - 1) * DENSE_LOCAL_SEGMENT_BITS;
        let value = ((hash >> shift) & DENSE_LOCAL_SEGMENT_MASK) as usize;
        (segment as usize) * (1usize << DENSE_LOCAL_SEGMENT_BITS) + value
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sad_scalar(a: &[u8], b: &[u8]) -> u32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| u32::from(x.abs_diff(*y)))
            .sum()
    }

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_u8(&mut self) -> u8 {
            u8::try_from(self.next_u64() & 0xff).unwrap_or(0)
        }
    }

    fn hex_reference(value: &str) -> Option<u64> {
        if value.len() != 16 {
            return None;
        }
        let mut out = 0u64;
        for byte in value.bytes() {
            let nibble = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => return None,
            };
            out = (out << 4) | u64::from(nibble);
        }
        Some(out)
    }

    #[test]
    fn hex_matches_reference() {
        for c in [
            "0000000000000000",
            "ffffffffffffffff",
            "0123456789abcdef",
            "FEDCBA9876543210",
        ] {
            assert_eq!(hex16_to_u64(c), hex_reference(c), "{c}");
        }
        assert_eq!(hex16_to_u64("xyz"), None);
        assert_eq!(hex16_to_u64("0123456789abcde"), None);
        assert_eq!(hex16_to_u64("0123456789abcdeg"), None);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..5000 {
            let v = rng.next_u64();
            let s = format!("{v:016x}");
            assert_eq!(hex16_to_u64(&s), Some(v));
            assert_eq!(hex16_to_u64(&s), hex_reference(&s));
            assert_eq!(hex16_to_u64(&s.to_uppercase()), Some(v));
        }
    }

    #[test]
    fn sad_dispatch_matches_scalar_64() {
        let mut rng = Rng(0xa5a5_5a5a_0f0f_f0f0);
        for _ in 0..2000 {
            let a: Vec<u8> = (0..64).map(|_| rng.next_u8()).collect();
            let b: Vec<u8> = (0..64).map(|_| rng.next_u8()).collect();
            assert_eq!(byte_sad(&a, &b), sad_scalar(&a, &b));
        }
        assert_eq!(byte_sad(&[0u8; 64], &[255u8; 64]), 64 * 255);
        assert_eq!(byte_sad(&[255u8; 64], &[255u8; 64]), 0);
    }

    #[test]
    fn sad_matches_scalar_various_lengths() {
        let mut rng = Rng(0x2244_6688_aacc_eeff);
        for &len in &[0usize, 1, 3, 31, 32, 33, 63, 64, 65, 96, 100] {
            let a: Vec<u8> = (0..len).map(|_| rng.next_u8()).collect();
            let b: Vec<u8> = (0..len).map(|_| rng.next_u8()).collect();
            assert_eq!(byte_sad(&a, &b), sad_scalar(&a, &b), "len {len}");
        }
    }

    #[test]
    fn mean_abs_delta_matches_scalar_formula() {
        let scalar = |a: &[u8], b: &[u8]| -> u8 {
            let sum: u16 = a
                .iter()
                .zip(b)
                .map(|(x, y)| u16::from(x.abs_diff(*y)))
                .sum();
            let len = u16::try_from(a.len()).unwrap_or(u16::MAX);
            let mean = (sum / len).min(u16::from(u8::MAX));
            u8::try_from(mean).unwrap_or(u8::MAX)
        };
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        for &len in &[3usize, 16, 64, 100, 256] {
            for _ in 0..2000 {
                let a: Vec<u8> = (0..len).map(|_| rng.next_u8()).collect();
                let b: Vec<u8> = (0..len).map(|_| rng.next_u8()).collect();
                assert_eq!(mean_abs_delta(&a, &b), scalar(&a, &b), "len {len}");
            }
        }
    }

    fn random_features(rng: &mut Rng) -> DescriptorFeatures {
        DescriptorFeatures {
            luma_mean: rng.next_u8(),
            luma_std: rng.next_u8(),
            edge_density: rng.next_u8(),
            pos_x: rng.next_u8(),
            pos_y: rng.next_u8(),
        }
    }

    #[test]
    fn feature_lanes_visit_matches_scalar_filter_in_order() {
        let mut rng = Rng(0x0dd0_ba11_5eed_f00d);
        for &len in &[0usize, 1, 5, 31, 32, 33, 64, 65, 100, 257, 1000] {
            let entries = (0..len)
                .map(|_| (random_features(&mut rng), !rng.next_u8().is_multiple_of(4)))
                .collect::<Vec<_>>();
            let lanes = FeatureLanes::from_entries(entries.iter().copied());
            assert_eq!(lanes.len(), len);
            for round in 0..40 {
                let query = random_features(&mut rng);
                // Mix wide, narrow and disabled gates, including boundary deltas.
                let delta = |rng: &mut Rng| match round % 5 {
                    0 => u8::MAX,
                    1 => 0,
                    2 => rng.next_u8() % 16,
                    _ => rng.next_u8(),
                };
                let gate = FeatureGate {
                    luma_mean: delta(&mut rng),
                    luma_std: delta(&mut rng),
                    edge_density: delta(&mut rng),
                    position: delta(&mut rng),
                    require_eligible: round % 2 == 0,
                };
                let expected = entries
                    .iter()
                    .enumerate()
                    .filter(|(_, (features, eligible))| {
                        features.luma_mean.abs_diff(query.luma_mean) <= gate.luma_mean
                            && features.luma_std.abs_diff(query.luma_std) <= gate.luma_std
                            && features.edge_density.abs_diff(query.edge_density)
                                <= gate.edge_density
                            && features.pos_x.abs_diff(query.pos_x) <= gate.position
                            && features.pos_y.abs_diff(query.pos_y) <= gate.position
                            && (!gate.require_eligible || *eligible)
                    })
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                let mut visited = Vec::new();
                lanes.for_each_compatible(query, gate, |index| visited.push(index));
                assert_eq!(visited, expected, "len {len} round {round}");
                let mut scalar = Vec::new();
                lanes.for_each_compatible_scalar(query, gate, |index| scalar.push(index));
                assert_eq!(scalar, expected, "scalar len {len} round {round}");
            }
        }
    }

    #[test]
    fn text_grid_mean_delta_matches_original_semantics() {
        let original = |s: &[u8], c: &[u8]| -> u8 {
            if s.len() != 64 || c.len() != 64 {
                return u8::MAX;
            }
            let sum: u16 = s
                .iter()
                .zip(c.iter())
                .map(|(l, r)| u16::from(l.abs_diff(*r)))
                .sum();
            let mean = (sum / 64).min(u16::from(u8::MAX));
            u8::try_from(mean).unwrap_or(u8::MAX)
        };
        let mut rng = Rng(0xfeed_face_cafe_beef);
        for _ in 0..3000 {
            let s: Vec<u8> = (0..64).map(|_| rng.next_u8()).collect();
            let c: Vec<u8> = (0..64).map(|_| rng.next_u8()).collect();
            assert_eq!(text_grid_mean_delta(&s, &c), original(&s, &c));
        }
        assert_eq!(text_grid_mean_delta(&[0u8; 63], &[0u8; 64]), u8::MAX);
        assert_eq!(text_grid_mean_delta(&[0u8; 64], &[0u8; 32]), u8::MAX);
    }

    fn original_hash_segments(hash: u64) -> Vec<(u8, u8)> {
        (0..HAMMING_INDEX_SEGMENTS)
            .map(|seg| {
                let shift = (HAMMING_INDEX_SEGMENTS - seg - 1) * HAMMING_SEGMENT_BITS;
                (seg, ((hash >> shift) & HAMMING_SEGMENT_MASK) as u8)
            })
            .collect()
    }
    #[test]
    fn flat_slot_bijective_with_tuple_key() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut seen_h = HashMap::new();
        for _ in 0..4000 {
            let h = rng.next_u64();
            for (t, f) in original_hash_segments(h)
                .iter()
                .zip(hamming_segments_flat(h))
            {
                assert!(f < HAMMING_FLAT_SLOTS);
                if let Some(p) = seen_h.insert(*t, f) {
                    assert_eq!(p, f, "hamming tuple {t:?} -> two slots");
                }
            }
        }
    }

    #[test]
    fn flat_index_query_equiv_to_hashmap() {
        let mut rng = Rng(0xabcd_ef01_2345_6789);
        let n = 2000usize;
        let phashes: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();

        let mut map: HashMap<(u8, u8), Vec<usize>> = HashMap::new();
        let mut flat: FlatSegmentIndex<usize> = FlatSegmentIndex::with_slots(HAMMING_FLAT_SLOTS);
        for (i, h) in phashes.iter().enumerate() {
            for (seg, val) in original_hash_segments(*h) {
                map.entry((seg, val)).or_default().push(i);
            }
            for slot in hamming_segments_flat(*h) {
                flat.push(slot, i);
            }
        }

        for _ in 0..500 {
            let q = rng.next_u64();
            let mut want = Vec::new();
            for (seg, val) in original_hash_segments(q) {
                if let Some(b) = map.get(&(seg, val)) {
                    want.extend_from_slice(b);
                }
            }
            want.sort_unstable();
            want.dedup();
            let mut got = Vec::new();
            for slot in hamming_segments_flat(q) {
                got.extend_from_slice(flat.get(slot));
            }
            got.sort_unstable();
            got.dedup();
            assert_eq!(got, want);
        }
    }
}
