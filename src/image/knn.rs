//! Cluster-coherence scoring for suspicious specimen matches.
//!
//! This stage is only a promotion rule. It does not find candidates by itself. A query
//! first has to match stored specimens through normal visual stages. The rule then asks
//! whether several matched specimens are also close to each other in a precomputed
//! specimen-specimen graph.
//!
//! The safety invariant is that every cluster member must score above the configured
//! cross-family chrome ceiling. Coherence alone is not enough: common UI chrome can match
//! several variants from the same family. Requiring each member to clear the chrome ceiling
//! prevents those chrome-only overlaps from entering the cluster rule.

#![forbid(unsafe_code)]

pub type SpecimenId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match {
    pub id: SpecimenId,
    pub inliers: u32,
    pub coverage_permille: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    pub valley_inliers: u32,
    pub cluster_member_inliers: u32,
    pub coverage_floor_permille: u16,
    pub coherence_threshold: u32,
    pub min_cluster_size: u16,
}

impl Thresholds {
    #[must_use]
    pub fn new(
        chrome_ceiling: u32,
        valley_inliers: u32,
        cluster_member_inliers: u32,
        coverage_floor_permille: u16,
        coherence_threshold: u32,
        min_cluster_size: u16,
    ) -> Self {
        debug_assert!(
            cluster_member_inliers > chrome_ceiling,
            "cluster_member_inliers must exceed the chrome ceiling"
        );
        Self {
            valley_inliers,
            cluster_member_inliers: cluster_member_inliers.max(chrome_ceiling + 1),
            coverage_floor_permille,
            coherence_threshold,
            min_cluster_size: min_cluster_size.max(2),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardActReason {
    SingleStrongMatch,
    CoherentCluster { size: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoHardActInfo {
    pub top1_inliers: u32,
    pub member_count: u16,
    pub best_cluster_size: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    HardAct(HardActReason),
    NoHardAct(NoHardActInfo),
}

impl Decision {
    #[cfg(test)]
    #[must_use]
    pub const fn is_hard_act(self) -> bool {
        matches!(self, Self::HardAct(_))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CoherenceGraph {
    row_start: Vec<u32>,
    col: Vec<SpecimenId>,
    coh: Vec<u32>,
}

impl CoherenceGraph {
    #[must_use]
    pub fn num_specimens(&self) -> usize {
        self.row_start.len().saturating_sub(1)
    }

    #[must_use]
    pub fn coherence(&self, a: SpecimenId, b: SpecimenId) -> u32 {
        let (left_cols, _) = self.row(a);
        let (right_cols, _) = self.row(b);
        let (key, cols, row_id) = if left_cols.len() <= right_cols.len() {
            (b, left_cols, a)
        } else {
            (a, right_cols, b)
        };
        match cols.binary_search(&key) {
            Ok(index) => {
                let (_, coherences) = self.row(row_id);
                coherences[index]
            }
            Err(_) => 0,
        }
    }

    fn row(&self, specimen: SpecimenId) -> (&[SpecimenId], &[u32]) {
        let specimen = specimen as usize;
        if specimen + 1 >= self.row_start.len() {
            return (&[], &[]);
        }
        let start = self.row_start[specimen] as usize;
        let end = self.row_start[specimen + 1] as usize;
        (&self.col[start..end], &self.coh[start..end])
    }

    pub(crate) fn undirected_edges(
        &self,
    ) -> impl Iterator<Item = (SpecimenId, SpecimenId, u32)> + '_ {
        (0..self.num_specimens())
            .filter_map(|from| SpecimenId::try_from(from).ok())
            .flat_map(|from| {
                let (cols, coherences) = self.row(from);
                cols.iter()
                    .copied()
                    .zip(coherences.iter().copied())
                    .filter_map(move |(to, coherence)| (from < to).then_some((from, to, coherence)))
            })
    }
}

pub(crate) struct CoherenceGraphBuilder {
    num_specimens: usize,
    build_floor: u32,
    edges: Vec<(SpecimenId, SpecimenId, u32)>,
}

impl CoherenceGraphBuilder {
    #[must_use]
    pub fn new(num_specimens: usize, build_floor: u32) -> Self {
        Self {
            num_specimens,
            build_floor,
            edges: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, a: SpecimenId, b: SpecimenId, coherence: u32) {
        if a == b
            || coherence < self.build_floor
            || a as usize >= self.num_specimens
            || b as usize >= self.num_specimens
        {
            return;
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        self.edges.push((a, b, coherence));
    }

    #[must_use]
    pub fn build(self) -> CoherenceGraph {
        let mut directed_edges = Vec::with_capacity(self.edges.len().saturating_mul(2));
        for (a, b, coherence) in self.edges {
            directed_edges.push((a, b, coherence));
            directed_edges.push((b, a, coherence));
        }
        directed_edges.sort_unstable_by_key(|(from, to, _)| (*from, *to));

        let mut degree = vec![0_u32; self.num_specimens];
        for &(from, _, _) in &directed_edges {
            degree[from as usize] += 1;
        }

        let mut row_start = vec![0_u32; self.num_specimens + 1];
        for index in 0..self.num_specimens {
            row_start[index + 1] = row_start[index] + degree[index];
        }

        let edge_count = row_start[self.num_specimens] as usize;
        let mut col = vec![0_u32; edge_count];
        let mut coh = vec![0_u32; edge_count];
        let mut cursor = row_start[..self.num_specimens].to_vec();
        for &(from, to, coherence) in &directed_edges {
            push_directed_edge(from, to, coherence, &mut cursor, &mut col, &mut coh);
        }

        CoherenceGraph {
            row_start,
            col,
            coh,
        }
    }
}

fn push_directed_edge(
    from: SpecimenId,
    to: SpecimenId,
    coherence: u32,
    cursor: &mut [u32],
    col: &mut [SpecimenId],
    coh: &mut [u32],
) {
    let position = cursor[from as usize] as usize;
    col[position] = to;
    coh[position] = coherence;
    cursor[from as usize] += 1;
}

#[derive(Clone, Copy)]
struct Member {
    id: SpecimenId,
}

pub struct ClusterScorer {
    cfg: Thresholds,
    members: Vec<Member>,
    parent: Vec<u16>,
    rank: Vec<u8>,
    size: Vec<u16>,
}

impl ClusterScorer {
    #[must_use]
    pub fn new(cfg: Thresholds) -> Self {
        Self {
            cfg,
            members: Vec::new(),
            parent: Vec::new(),
            rank: Vec::new(),
            size: Vec::new(),
        }
    }

    #[must_use]
    pub fn score(&mut self, matches: &[Match], graph: &CoherenceGraph) -> Decision {
        self.members.clear();
        let mut top1_inliers = 0_u32;
        let mut single_strong = false;
        for item in matches {
            top1_inliers = top1_inliers.max(item.inliers);
            let coverage_ok = item.coverage_permille >= self.cfg.coverage_floor_permille;
            if item.inliers >= self.cfg.valley_inliers && coverage_ok {
                single_strong = true;
            }
            if item.inliers >= self.cfg.cluster_member_inliers
                && coverage_ok
                && self.members.len() < usize::from(u16::MAX)
            {
                self.members.push(Member { id: item.id });
            }
        }

        let member_count = self.members.len();
        let member_count_u16 = u16::try_from(member_count).unwrap_or(u16::MAX);
        if member_count_u16 < self.cfg.min_cluster_size {
            if single_strong {
                return Decision::HardAct(HardActReason::SingleStrongMatch);
            }
            return Decision::NoHardAct(NoHardActInfo {
                top1_inliers,
                member_count: member_count_u16,
                best_cluster_size: member_count_u16.min(1),
            });
        }

        self.members.sort_unstable_by_key(|member| member.id);
        self.reset_union_find(member_count);
        for left in 0..member_count {
            for right in (left + 1)..member_count {
                let left_id = self.members[left].id;
                let right_id = self.members[right].id;
                if graph.coherence(left_id, right_id) >= self.cfg.coherence_threshold {
                    self.union(left, right);
                }
            }
        }

        let best_cluster_size = self.best_cluster_size(member_count);
        if best_cluster_size >= self.cfg.min_cluster_size {
            Decision::HardAct(HardActReason::CoherentCluster {
                size: best_cluster_size,
            })
        } else if single_strong {
            Decision::HardAct(HardActReason::SingleStrongMatch)
        } else {
            Decision::NoHardAct(NoHardActInfo {
                top1_inliers,
                member_count: member_count_u16,
                best_cluster_size,
            })
        }
    }

    fn reset_union_find(&mut self, len: usize) {
        let len_u16 = u16::try_from(len).unwrap_or(u16::MAX);
        self.parent.clear();
        self.rank.clear();
        self.size.clear();
        self.parent.extend(0..len_u16);
        self.rank.resize(len, 0);
        self.size.resize(len, 1);
    }

    fn best_cluster_size(&mut self, len: usize) -> u16 {
        let mut best = 1;
        for index in 0..len {
            if self.find(index) == index {
                best = best.max(self.size[index]);
            }
        }
        best
    }

    fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] as usize != index {
            let grandparent = self.parent[self.parent[index] as usize];
            self.parent[index] = grandparent;
            index = grandparent as usize;
        }
        index
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            core::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = u16::try_from(left_root).unwrap_or(u16::MAX);
        self.size[left_root] += self.size[right_root];
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> CoherenceGraph {
        let mut builder = CoherenceGraphBuilder::new(5, 1);
        builder.add_edge(0, 1, 80);
        builder.add_edge(1, 2, 80);
        builder.add_edge(0, 2, 80);
        builder.add_edge(3, 4, 80);
        builder.add_edge(0, 3, 10);
        builder.build()
    }

    fn cfg() -> Thresholds {
        Thresholds::new(19, 63, 25, 0, 63, 2)
    }

    #[test]
    fn single_strong_match_promotes() {
        let mut scorer = ClusterScorer::new(cfg());
        let matches = [Match {
            id: 0,
            inliers: 64,
            coverage_permille: 1_000,
        }];
        assert!(scorer.score(&matches, &graph()).is_hard_act());
    }

    #[test]
    fn coherent_cluster_promotes_below_valley() {
        let mut scorer = ClusterScorer::new(cfg());
        let matches = [
            Match {
                id: 0,
                inliers: 40,
                coverage_permille: 1_000,
            },
            Match {
                id: 1,
                inliers: 35,
                coverage_permille: 1_000,
            },
        ];
        assert!(scorer.score(&matches, &graph()).is_hard_act());
    }

    #[test]
    fn chrome_band_does_not_enter_cluster_rule() {
        let mut scorer = ClusterScorer::new(cfg());
        let matches = [
            Match {
                id: 0,
                inliers: 19,
                coverage_permille: 1_000,
            },
            Match {
                id: 1,
                inliers: 18,
                coverage_permille: 1_000,
            },
        ];
        match scorer.score(&matches, &graph()) {
            Decision::NoHardAct(info) => assert_eq!(info.member_count, 0),
            other @ Decision::HardAct(_) => {
                panic!("chrome-only matches must not promote: {other:?}");
            }
        }
    }

    #[test]
    fn incoherent_members_do_not_promote() {
        let mut scorer = ClusterScorer::new(cfg());
        let matches = [
            Match {
                id: 0,
                inliers: 40,
                coverage_permille: 1_000,
            },
            Match {
                id: 3,
                inliers: 35,
                coverage_permille: 1_000,
            },
        ];
        assert!(!scorer.score(&matches, &graph()).is_hard_act());
    }
}
