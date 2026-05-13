//! Random-variable description trait and fast lookup tables for redundancy /
//! PUC computations.
//!
//! This module sits between the raw histogram/log-ratio tables computed in
//! [`crate::mvim::imeasures`] and the network-construction kernels in
//! [`crate::mcpn`] / [`crate::pucn`]. It defines:
//!
//! * [`MRVTrait`] — the abstract view of a discrete multi-variable
//!   distribution. Implementors expose per-variable histograms, marginal
//!   dimensions, mutual information `I(X_i; X_j)`, specific information `SI`
//!   and log-marginal-ratio (LMR) vectors. Default trait methods build on
//!   those primitives to compute Williams–Beer-style [`redundancy`]
//!   contributions and the PUC score introduced by Chan et al., 2017
//!   ("Gene Regulatory Network Inference from Single-Cell Data Using
//!   Multivariate Information Measures", Cell Systems 5(3)).
//! * [`UnitLMRSA`] — a single sorted/prefix-sum/rank record for one
//!   `(about, rstate)` pair, useful for ad-hoc per-state queries.
//! * [`LMRSA`] — the full table of [`UnitLMRSA`]-style records concatenated
//!   along the `dim` (state) axis, for every `by` variable. The sorted +
//!   prefix-sum + inverse-rank encoding lets [`LMRSA::minsum_wsrc`] /
//!   [`LMRSA::minsum_nosrc`] evaluate `Σ_state min(LMR(about,by_a)(state),
//!   LMR(about,by_b)(state))` in O(dim) per pair, instead of a O(dim · nvars)
//!   linear scan.
//! * [`LMRDataStructure`] / [`LMRSubsetDataStructure`] — thin wrappers around
//!   an [`LMRSA`] tied to a fixed `about` variable. The first version covers
//!   every variable in the distribution; the second restricts to a subset
//!   tracked by a `subset_map` and falls back to a binary search for any
//!   query whose source variable is outside the subset. Both are consumed by
//!   [`crate::mvim::misi::MISIRangePair`] and [`crate::mvim::misi::MISIPair`]
//!   to compute PUC several times faster than the generic trait method.

use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;
use std::rc::Rc;

use super::imeasures::redundancy;
use crate::types::{PNFloat, PNInteger};

/// Errors returned by [`MRVTrait`] queries and the LMR data-structure
/// constructors.
#[derive(Debug)]
pub enum Error {
    /// A variable identifier (`i`/`j`/`k`) is not in the implementor's range.
    InvalidIndex(usize),
    /// The `about` argument passed to a histogram / SI / LMR query is unknown.
    InvalidAbout(usize),
    /// The `by` argument paired with `about` is unknown.
    InvalidBy(usize),
    /// Catch-all for invariants that should not be reachable in practice
    /// (e.g. an [`crate::mvim::misi::LMRDSPair`] queried before
    /// `set_lmr_ds`).
    Unexpected(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidIndex(idx) => {
                write!(f, "Invalid Index : {idx}")
            }
            Error::InvalidAbout(abt) => {
                write!(f, "Invalid About : {abt}")
            }
            Error::InvalidBy(rby) => {
                write!(f, "Invalid By    : {rby}")
            }
            Error::Unexpected(str_err) => {
                write!(f, "Unexpected    : {str_err}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Abstract view of a discrete multi-variable distribution.
///
/// Implementors describe `nvariables()` random variables observed over
/// `nobservations()` samples. Each variable `i` is identified by an
/// `IntT`-typed index (`0..nvariables()`) and is summarized by a marginal
/// histogram (`get_hist`, `get_hist_dim`), pairwise mutual information
/// (`get_mi`), and the per-`(about, by)` specific information / log-marginal
/// ratio (LMR) vectors (`get_si`/`si_value`, `get_lmr`/`lmr_value`).
///
/// Default methods build redundancy and PUC quantities on top of the
/// primitives above:
///
/// * [`get_redundancies`](MRVTrait::get_redundancies) — Williams–Beer
///   redundancy contributions for each ordering of a triple `(i, j, k)`.
/// * [`mpuc`](MRVTrait::mpuc),
///   [`redundancy_updates`](MRVTrait::redundancy_updates),
///   [`accumulate_redundancies`](MRVTrait::accumulate_redundancies) — the
///   PUC computation defined by Chan et al. 2017 used by [`crate::pucn`].
/// * [`compute_puc_matrix`](MRVTrait::compute_puc_matrix) and
///   [`compute_puc_matrix_for`](MRVTrait::compute_puc_matrix_for) — assemble
///   a dense `nvars × nvars` PUC adjacency matrix.
/// * [`compute_lm_puc`](MRVTrait::compute_lm_puc) — the LMR-based PUC form
///   `(nvars-2 - minsum(i,j)/I(i,j)) + (nvars-2 - minsum(j,i)/I(i,j))`,
///   which equals the Chan et al. PUC sum but is computed via the
///   sorted/prefix-sum [`LMRSA`] tables in O(dim) instead of O(dim · nvars).
///
/// All `get_*` accessors return data wrapped in [`Result`] so implementors
/// can validate indices and raise [`Error::InvalidIndex`] /
/// [`Error::InvalidAbout`] / [`Error::InvalidBy`].
pub trait MRVTrait<IntT: 'static + PNInteger, FloatT: 'static + PNFloat> {
    /// Marginal histogram of variable `i`, length `get_hist_dim(i)`.
    fn get_hist(&self, i: IntT) -> Result<Array1<FloatT>, Error>;
    /// Number of histogram bins (`dim`) for variable `i`.
    fn get_hist_dim(&self, i: IntT) -> Result<IntT, Error>;
    /// Pairwise mutual information `I(X_i; X_j)` (symmetric).
    fn get_mi(&self, i: IntT, j: IntT) -> Result<FloatT, Error>;
    /// Specific information vector `SI_about[·]` of length
    /// `get_hist_dim(about)` describing what knowing each value of `by`
    /// reveals about `about`.
    fn get_si(&self, about: IntT, by: IntT) -> Result<Array1<FloatT>, Error>;
    /// Single entry of [`get_si`](MRVTrait::get_si) at state `rstate`.
    fn si_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error>;
    /// Log-marginal-ratio vector `LMR_about[·]` of length
    /// `get_hist_dim(about)` (the `lmr_about_x_from_ljvi` projection of
    /// the joint log-ratio table; see [`crate::mvim::imeasures`]).
    fn get_lmr(&self, i: IntT, j: IntT) -> Result<Array1<FloatT>, Error>;
    /// Single entry of [`get_lmr`](MRVTrait::get_lmr) at state `rstate`.
    fn lmr_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error>;
    /// Sample count `N` returned as `FloatT`, used as the probability
    /// normalization in redundancy/PUC sums.
    fn ndata(&self) -> FloatT;
    /// Number of observations `N` (the sample dimension).
    fn nobservations(&self) -> usize;
    /// Number of random variables in the distribution.
    fn nvariables(&self) -> usize;

    /// Williams–Beer redundancy contributions of every ordering of the
    /// triple `(i, j, k)`.
    ///
    /// Returns the tuple `(R_i, R_j, R_k)` where `R_x = redundancy(hist_x,
    /// SI(x|y), SI(x|z), Some(N))` for the cyclic permutations
    /// `(x, y, z) ∈ {(i,j,k), (j,i,k), (k,i,j)}`. Used by
    /// [`redundancy_updates`](MRVTrait::redundancy_updates) to feed the PUC
    /// scoring kernel.
    fn get_redundancies(
        &self,
        i: IntT,
        j: IntT,
        k: IntT,
    ) -> Result<(FloatT, FloatT, FloatT), Error> {
        Ok((
            redundancy(
                self.get_hist(i)?.view(),
                self.get_si(i, j)?.view(),
                self.get_si(i, k)?.view(),
                Some(self.ndata()),
            ),
            redundancy(
                self.get_hist(j)?.view(),
                self.get_si(j, i)?.view(),
                self.get_si(j, k)?.view(),
                Some(self.ndata()),
            ),
            redundancy(
                self.get_hist(k)?.view(),
                self.get_si(k, i)?.view(),
                self.get_si(k, j)?.view(),
                Some(self.ndata()),
            ),
        ))
    }

    /// Per-pair PUC contribution `mPUC(i, j; redundancy)`.
    ///
    /// Defined as `(I(X_i;X_j) - R) / I(X_i;X_j)` and clamped to `[0, ∞)`.
    /// Negative or non-finite values (e.g. when `I(X_i;X_j) = 0`) are
    /// reported as zero so they do not subtract from accumulated PUC scores.
    fn mpuc(
        &self,
        i: IntT,
        j: IntT,
        redundancy: FloatT,
    ) -> Result<FloatT, Error> {
        let mi = self.get_mi(i, j)?;
        let puc_score = (mi - redundancy) / mi;
        if puc_score.is_finite() && puc_score >= FloatT::zero() {
            Ok(puc_score)
        } else {
            Ok(FloatT::zero())
        }
    }

    /// Three PUC updates produced by a single triple `(i, j, k)`.
    ///
    /// Returns `(Δij, Δik, Δjk)` — the PUC contributions to the edges
    /// `(i,j)`, `(i,k)` and `(j,k)` respectively, each computed from the
    /// pair of redundancies attributable to that edge. Requires `i < j`.
    /// Used by [`compute_puc_matrix`](MRVTrait::compute_puc_matrix) to
    /// build the dense PUC adjacency matrix one combination at a time.
    fn redundancy_updates(
        &self,
        i: IntT,
        j: IntT,
        k: IntT,
    ) -> Result<(FloatT, FloatT, FloatT), Error> {
        assert!(i < j);
        let (ri, rj, rk) = self.get_redundancies(i, j, k)?;
        Ok((
            self.mpuc(i, j, ri)? + self.mpuc(i, j, rj)?,
            self.mpuc(i, k, ri)? + self.mpuc(i, k, rk)?,
            self.mpuc(k, k, rj)? + self.mpuc(j, k, rk)?,
        ))
    }

    /// PUC contribution to the edge `(i, j)` from a single conditioning
    /// variable `by`.
    ///
    /// Computes `mpuc(i, j, R_i) + mpuc(i, j, R_j)` where `R_i` and `R_j`
    /// are the Williams–Beer redundancies of `(i, by)` and `(j, by)` against
    /// each other (the third "leg" of the PUC triple is fixed at `by`).
    /// Asserts `i < j` and `i != by != j` — callers in
    /// [`accumulate_redundancies_for`](MRVTrait::accumulate_redundancies_for)
    /// already filter out the trivial cases.
    fn redundancy_update_for(
        &self,
        i: IntT,
        j: IntT,
        by: IntT,
    ) -> Result<FloatT, Error> {
        assert!(i < j);
        assert!(i != by);
        assert!(j != by);
        let iby_si = self.get_si(i, by)?;
        let jby_si = self.get_si(j, by)?;
        let ri = redundancy(
            self.get_hist(i)?.view(),
            self.get_si(i, j)?.view(),
            iby_si.view(),
            Some(self.ndata()),
        );
        let rj = redundancy(
            self.get_hist(j)?.view(),
            self.get_si(j, i)?.view(),
            jby_si.view(),
            Some(self.ndata()),
        );
        Ok(self.mpuc(i, j, ri)? + self.mpuc(i, j, rj)?)
    }

    /// Sum of [`redundancy_update_for`](MRVTrait::redundancy_update_for)
    /// across an explicit set of conditioning variables `by_nodes`.
    ///
    /// Skips any `bx == i` or `bx == j` so the caller may pass a list that
    /// contains the edge endpoints. Used by
    /// [`compute_puc_matrix_for`](MRVTrait::compute_puc_matrix_for) to
    /// restrict PUC accumulation to a subset of the genome.
    fn accumulate_redundancies_for(
        &self,
        i: IntT,
        j: IntT,
        //by_nodes: impl Iterator<Item = &'a i32>,
        by_nodes: &[IntT],
    ) -> Result<FloatT, Error> {
        let mut acc: FloatT = FloatT::zero();
        for bx in by_nodes {
            if *bx != i && *bx != j {
                acc += self.redundancy_update_for(i, j, *bx)?
            }
        }
        Ok(acc)
    }

    /// PUC accumulation against every variable `0..nvariables()`.
    ///
    /// Convenience wrapper over
    /// [`accumulate_redundancies_for`](MRVTrait::accumulate_redundancies_for)
    /// with the full variable range as `by_nodes`.
    fn accumulate_redundancies(&self, i: IntT, j: IntT) -> Result<FloatT, Error> {
        let nvars: usize = self.nvariables();
        let vrange: Vec<IntT> =
            (0..nvars).map(|vx| IntT::from_usize(vx).unwrap()).collect();
        self.accumulate_redundancies_for(i, j, &vrange)
    }

    /// Build the dense `nvars × nvars` PUC adjacency matrix.
    ///
    /// Iterates over all `C(nvars, 3)` triples and accumulates each triple's
    /// contribution into the upper triangle via
    /// [`redundancy_updates`](MRVTrait::redundancy_updates), then mirrors
    /// the upper triangle into the lower triangle. The diagonal stays zero.
    fn compute_puc_matrix(&self) -> Result<Array2<FloatT>, Error> {
        let nvars = self.nvariables();
        let mut puc_network: Array2<FloatT> = Array2::zeros((nvars, nvars));

        for cvec in (0..nvars).combinations(3) {
            let i = cvec[0];
            let j = cvec[1];
            let k = cvec[2];
            let (rij, rik, rjk) = self.redundancy_updates(
                IntT::from_usize(i).unwrap(),
                IntT::from_usize(j).unwrap(),
                IntT::from_usize(k).unwrap(),
            )?;
            puc_network[[i, j]] += rij;
            puc_network[[i, k]] += rik;
            puc_network[[j, k]] += rjk;
        }
        for cvec in (0..nvars).combinations(2) {
            let i = cvec[0];
            let j = cvec[1];
            puc_network[[j, i]] = puc_network[[i, j]];
        }
        Ok(puc_network)
    }

    /// Subset variant of
    /// [`compute_puc_matrix`](MRVTrait::compute_puc_matrix).
    ///
    /// Each upper-triangular entry `(i, j)` is filled with
    /// `accumulate_redundancies_for(i, j, by_nodes)`. The lower triangle
    /// stays zero (no symmetric mirroring), since callers typically only
    /// consume the upper triangle when restricting to a subset.
    fn compute_puc_matrix_for(
        &self,
        by_nodes: &[IntT],
    ) -> Result<Array2<FloatT>, Error> {
        let nvars = self.nvariables();
        let mut puc_network: Array2<FloatT> = Array2::zeros((nvars, nvars));
        for cvec in (0..nvars).combinations(2) {
            let i = cvec[0];
            let j = cvec[1];
            puc_network[[i, j]] = self.accumulate_redundancies_for(
                IntT::from_usize(i).unwrap(),
                IntT::from_usize(j).unwrap(),
                by_nodes,
            )?
        }
        Ok(puc_network)
    }

    /// Dump every per-triple redundancy into a hashmap keyed by ordered
    /// triple.
    ///
    /// For each combination `i < j < k` the three values from
    /// [`redundancy_updates`](MRVTrait::redundancy_updates) are inserted
    /// under the keys `(i, j, k)`, `(i, k, j)` and `(k, j, i)` respectively.
    /// Mainly useful for diagnostic dumps; `compute_puc_matrix*` is the
    /// production path. Returns an empty map when `nvariables() == 0`.
    fn compute_redundancies(
        &self,
    ) -> Result<HashMap<(i32, i32, i32), FloatT>, Error> {
        let mut hsmap: HashMap<(i32, i32, i32), FloatT> = HashMap::new();
        let nvars = self.nvariables() as i32;
        if nvars == 0 {
            return Ok(hsmap);
        }
        for cvec in (0..nvars).combinations(3) {
            let i = cvec[0];
            let j = cvec[1];
            let k = cvec[2];
            let (rij, rik, rjk) = self.redundancy_updates(
                IntT::from_i32(i).unwrap(),
                IntT::from_i32(j).unwrap(),
                IntT::from_i32(k).unwrap(),
            )?;
            hsmap.insert((i, j, k), rij);
            hsmap.insert((i, k, j), rik);
            hsmap.insert((k, j, i), rjk);
        }
        Ok(hsmap)
    }

    /// Per-`by` vector of `Σ_state min(LMR(about, target)(s), LMR(about,
    /// by)(s))`.
    ///
    /// Iterates over every variable except `about` and `target`, computing
    /// for each candidate `by` the elementwise minimum between
    /// `LMR(about, target)` and `LMR(about, by)` and summing across states.
    /// This function implementation takes O(nvars · dim) ; 
    /// [`LMRSA::minsum_wsrc`] / [`LMRSA::minsum_nosrc`] implements the 
    /// O(dim) variant.
    fn minsum_list(
        &self,
        about: IntT,
        target: IntT,
    ) -> Result<Array1<FloatT>, Error> {
        let nvars: usize = self.nvariables();
        let lmr_abtgt = self.get_lmr(about, target)?;
        let by_nodes: Vec<IntT> = (0..nvars)
            .map(|x| IntT::from_usize(x).unwrap())
            .filter(|x| *x != about && *x != target)
            .collect();
        let mut ms_list =
            Array1::<FloatT>::from_elem(by_nodes.len(), FloatT::zero());
        for bx in 0..by_nodes.len() {
            let lmx = self.get_lmr(about, by_nodes[bx])?;
            ms_list[bx] = std::iter::zip(lmx.iter(), lmr_abtgt.iter())
                .fold(FloatT::zero(), |acc, (lma, lmb)| acc + lma.min(*lmb));
        }
        Ok(ms_list)
    }

    /// Compute sum:
    ///  `Σ_by Σ_state min(LMR(about, target)(s), LMR(about, by)(s))`
    /// i.e., scalar reduction of [`minsum_list`](MRVTrait::minsum_list).
    fn get_lmr_minsum(&self, about: IntT, target: IntT) -> Result<FloatT, Error> {
        Ok(self.minsum_list(about, target)?.sum())
    }

    /// Scaling factor in front of `minsum / I(about, target)` inside
    /// [`compute_lm_puc`](MRVTrait::compute_lm_puc).
    ///
    /// Defaults to `nvariables() - 2` (one position per non-(about/target)
    /// variable). Subset-aware implementors override this to restrict 
    /// to only those that are part of the active subset.
    fn get_puc_factor(
        &self,
        _about: IntT,
        _target: IntT,
    ) -> Result<FloatT, Error> {
        Ok(FloatT::from_usize(self.nvariables() - 2).unwrap())
    }

    /// LMR-based PUC score for the edge `(i, j)`.
    ///
    /// Equal to
    /// `(P(i,j) - minsum(i,j)/I(i,j)) + (P(j,i) - minsum(j,i)/I(i,j))`,
    /// where `P` is [`get_puc_factor`](MRVTrait::get_puc_factor) and
    /// `minsum` is [`get_lmr_minsum`](MRVTrait::get_lmr_minsum). Returns
    /// `I(i,j)` directly when it is non-positive (so the edge contributes a
    /// non-positive but finite value rather than dividing by zero).
    fn compute_lm_puc(&self, i: IntT, j: IntT) -> Result<FloatT, Error> {
        let mij = self.get_mi(i, j)?;
        if mij <= FloatT::zero() {
            return Ok(mij);
        }
        let (pij_factor, pji_factor) =
            (self.get_puc_factor(i, j)?, self.get_puc_factor(j, i)?);
        Ok((pij_factor - (self.get_lmr_minsum(i, j)? / mij))
            + (pji_factor - (self.get_lmr_minsum(j, i)? / mij)))
    }
}

/// Sorted/prefix-sum/rank record for the LMR vector of a single
/// `(about, rstate)` pair.
///
/// Stores the LMR values of every "by" variable at one fixed state of
/// `about`. All three arrays have length `nvars` (the total number of
/// variables in the distribution):
///
/// * `sorted`  — the LMR values sorted in ascending order.
/// * `pfxsum`  — the inclusive prefix sum of `sorted`.
/// * `rank[i]` — the position of variable `i` inside `sorted`.
///
/// Together they let [`minsum_wsrc`](Self::minsum_wsrc) split
/// `Σ_by min(LMR_target, LMR_by)` into "values below the cutoff" (looked up
/// via `pfxsum`) and "values clamped to the cutoff" (counted via the rank)
/// in O(1).
pub struct UnitLMRSA<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    /// `about` variable identifier this record belongs to.
    pub about: IntT,
    /// State of `about` this record is fixed at.
    pub rstate: IntT,
    /// LMR values sorted ascending; length `nvars`.
    sorted: Array1<FloatT>,
    /// Inclusive prefix sum of `sorted`.
    pfxsum: Array1<FloatT>,
    /// Inverse lookup: `rank[v]` is the position of variable `v` in `sorted`.
    rank: Array1<IntT>,
}

impl<IntT, FloatT> UnitLMRSA<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    /// Build a [`UnitLMRSA`] from the raw LMR slice for `(about, rstate)`.
    ///
    /// `lmr[i]` is the LMR value of the candidate "by" variable `i`. The
    /// constructor sorts the slice, builds the prefix sum, and inverts the
    /// permutation to populate `rank` so subsequent queries are O(1).
    pub fn from_slice(about: IntT, rstate: IntT, lmr: &[FloatT]) -> Self {
        let nvars = lmr.len();

        let mut si_vec: Vec<(FloatT, IntT)> = lmr
            .iter()
            .enumerate()
            .map(|(vidx, y)| {
                let by_var = IntT::from_usize(vidx).unwrap();
                (*y, by_var)
            })
            .collect();
        si_vec.sort_by(|(fa, _ia), (fb, _ib)| fa.total_cmp(fb));

        let mut curr_sum = FloatT::zero();
        let mut sorted: Array1<FloatT> = Array1::from_elem(nvars, FloatT::zero());
        let mut pfxsum: Array1<FloatT> = Array1::from_elem(nvars, FloatT::zero());
        let mut rank: Array1<IntT> = Array1::from_elem(nvars, IntT::zero());
        for (ix, (svx, by_var)) in si_vec.iter().enumerate() {
            let by_idx = by_var.to_usize().unwrap();
            curr_sum += *svx;
            rank[by_idx] = IntT::from_usize(ix).unwrap();
            sorted[ix] = *svx;
            pfxsum[ix] = curr_sum;
        }
        Self {
            about,
            rstate,
            sorted,
            pfxsum,
            rank,
        }
    }

    /// O(1) computation of `Σ_{by ≠ src} min(LMR_src, LMR_by)` for this
    /// `(about, rstate)`.
    ///
    /// Splits the sum at the rank of `src_idx` in [`sorted`](Self::sorted):
    /// every value strictly below `LMR(src)` contributes itself (read off
    /// `pfxsum`), and every value at or above contributes `LMR(src)` itself
    /// (counted by `nvars - 1 - rank`).
    pub fn minsum_wsrc(&self, src_idx: usize) -> FloatT {
        let mut rdsum = FloatT::zero();
        let lmrank = self.rank[src_idx].to_usize().unwrap();
        let lmv = self.sorted[lmrank];
        let lmlow = if lmrank > 0 {
            self.pfxsum[lmrank - 1]
        } else {
            FloatT::zero()
        };
        let lmhigh =
            FloatT::from_usize(self.sorted.len() - 1 - lmrank).unwrap() * lmv;
        rdsum += lmlow + lmhigh;
        rdsum
    }
}

/// Sorted/prefix-sum/rank tables for the LMR vectors of a single `about`
/// variable across every state.
///
/// Conceptually a stack of `dim` [`UnitLMRSA`]s, one per state of `about`,
/// flattened into three `dim · nvars`-long buffers and indexed by
/// `rstate · nvars + position`. `dim` is the number of histogram bins of
/// `about` and `nvars` is the number of "by" variables tracked by the
/// table (either every variable in the distribution or a fixed subset).
///
/// * `sorted[rstate · nvars + ix]` — LMR values for state `rstate` sorted
///   ascending.
/// * `pfxsum[rstate · nvars + ix]` — inclusive prefix sum of `sorted`
///   within the same state segment.
/// * `rank[rstate · nvars + v]` — position of variable `v` in
///   the sorted segment for state `rstate`.
///
/// The two query methods produce the per-state minsum used by PUC computation:
///
/// * [`minsum_wsrc`](Self::minsum_wsrc) — when the source variable is one
///   of the tracked variables, look up its rank and split the sum in O(1)
///   per state.
/// * [`minsum_nosrc`](Self::minsum_nosrc) — when the source LMR vector is
///   not in the table, binary-search each state to find the cutoff and
///   apply the same prefix-sum trick.
pub struct LMRSA<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    /// Total length of each flat buffer (`dim * nvars`); kept for symmetry
    /// with the constructor and currently unused at query time.
    _size: usize,
    /// Number of "by" variables tracked.
    nvars: usize,
    /// Number of states of `about`.
    dim: usize,
    /// `sorted` segments concatenated across states.
    sorted: Array1<FloatT>,
    /// `pfxsum` segments concatenated across states.
    pfxsum: Array1<FloatT>,
    /// `rank` segments concatenated across states.
    rank: Array1<IntT>,
}

impl<IntT, FloatT> LMRSA<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    #![allow(clippy::needless_range_loop)]
    /// Build the full-distribution [`LMRSA`] for a given `about` variable.
    ///
    /// Calls `pidata.get_lmr(about, by)` for every `by ∈ 0..nvariables()`
    /// and packs the values into `siv_lst[rstate][vidx]` before delegating
    /// the sort/prefix-sum/rank construction to
    /// [`from_siv_list`](Self::from_siv_list).
    pub fn from_mrv_trait(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
    ) -> Result<Self, Error> {
        let nvars = pidata.nvariables();
        let dim = pidata.get_hist_dim(about)?.to_usize().unwrap();
        let size = dim * nvars;
        let mut siv_lst: Vec<Vec<(FloatT, IntT)>> =
            vec![vec![(FloatT::zero(), IntT::zero()); nvars]; dim];
        //
        for vidx in 0..nvars {
            let by_var = IntT::from_usize(vidx).unwrap();
            let lmr_ax = pidata.get_lmr(about, by_var)?;
            for rstate in 0..dim {
                siv_lst[rstate][vidx] = (lmr_ax[rstate], by_var)
            }
        }
        Self::from_siv_list(&mut siv_lst, size, nvars, dim)
    }

    /// Sort each per-state vector and assemble the segmented `sorted`,
    /// `pfxsum` and `rank` arrays.
    ///
    /// `siv_list[rstate]` is the unsorted vector of `(lmr_value, by_var)`
    /// pairs for state `rstate`. The function sorts each entry in place,
    /// then walks it once to populate the three flat buffers used by
    /// [`minsum_wsrc`](Self::minsum_wsrc) / [`minsum_nosrc`](Self::minsum_nosrc).
    pub fn from_siv_list(
        siv_list: &mut [Vec<(FloatT, IntT)>],
        size: usize,
        nvars: usize,
        dim: usize,
    ) -> Result<Self, Error> {
        for si_vec in siv_list.iter_mut() {
            si_vec.sort_by(|(fa, _ia), (fb, _ib)| fa.total_cmp(fb));
        }
        let mut sorted: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut pfxsum: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut rank: Array1<IntT> = Array1::from_elem(size, IntT::zero());
        for (rstate, si_vec) in siv_list.iter().enumerate() {
            let rsbegin = rstate * nvars;
            let mut curr_sum = FloatT::zero();
            for (ix, (svx, by_var)) in si_vec.iter().enumerate() {
                let by_idx = by_var.to_usize().unwrap();
                curr_sum += *svx;
                rank[rsbegin + by_idx] = IntT::from_usize(ix).unwrap();
                sorted[rsbegin + ix] = *svx;
                pfxsum[rsbegin + ix] = curr_sum;
            }
        }

        Ok(LMRSA {
            _size: size,
            nvars,
            dim,
            sorted,
            pfxsum,
            rank,
        })
    }

    /// Build an [`LMRSA`] directly from a flat LMR buffer.
    ///
    /// `lmr` is laid out as `nvars` consecutive segments of length `dim`,
    /// where segment `vidx` holds the LMR vector `LMR(about, vidx)`. Used
    /// by callers that already loaded the LMR table from disk and need to
    /// reorganize it for fast minsum queries.
    pub fn from_lmr_slice(
        dim: IntT,
        nvars: usize,
        lmr: &[FloatT],
    ) -> Result<Self, Error> {
        let size = lmr.len();
        let dim = dim.to_usize().unwrap();
        assert!(lmr.len() == dim * nvars);
        let mut siv_lst: Vec<Vec<(FloatT, IntT)>> =
            vec![vec![(FloatT::zero(), IntT::zero()); nvars]; dim];
        for vidx in 0..nvars {
            let by_var = IntT::from_usize(vidx).unwrap();
            let lmr_ax = &lmr[(vidx * dim)..((vidx + 1) * dim)];
            for rstate in 0..dim {
                siv_lst[rstate][vidx] = (lmr_ax[rstate], by_var)
            }
        }

        Self::from_siv_list(&mut siv_lst, size, nvars, dim)
    }

    /// Build a subset-restricted [`LMRSA`] for `about`.
    ///
    /// `subset_map` maps each external variable id to its position in the
    /// per-state vectors (so `subset_map.len() == nvars`). For every entry
    /// the constructor pulls `LMR(about, by_var)` from `pidata`, sorts each
    /// state's segment, and records the inverse permutation back into the
    /// subset positions so [`minsum_wsrc`](Self::minsum_wsrc) can later
    /// look up by the subset's local index.
    pub fn from_subset_map(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
        subset_map: &HashMap<IntT, usize>,
    ) -> Result<Self, Error> {
        let nvars = subset_map.len();
        let dim = pidata.get_hist_dim(about)?.to_usize().unwrap();
        let size = dim * subset_map.len();
        let mut siv_lst: Vec<Vec<(FloatT, IntT)>> =
            vec![vec![(FloatT::zero(), IntT::zero()); nvars]; dim];
        //
        for (by_var, by_idx) in subset_map.iter() {
            let lmr_ax = pidata.get_lmr(about, *by_var)?;
            for rstate in 0..dim {
                siv_lst[rstate][*by_idx] = (lmr_ax[rstate], *by_var)
            }
        }
        for si_vec in siv_lst.iter_mut() {
            si_vec.sort_by(|(fa, _ia), (fb, _ib)| fa.total_cmp(fb));
        }
        let mut sorted: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut pfxsum: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut rank: Array1<IntT> = Array1::from_elem(size, IntT::zero());
        for (rstate, si_vec) in siv_lst.iter().enumerate() {
            let rsbegin = rstate * nvars;
            let mut curr_sum = FloatT::zero();
            for (ix, (svx, by_var)) in si_vec.iter().enumerate() {
                let by_idx = subset_map[by_var];
                curr_sum += *svx;
                rank[rsbegin + by_idx] = IntT::from_usize(ix).unwrap();
                sorted[rsbegin + ix] = *svx;
                pfxsum[rsbegin + ix] = curr_sum;
            }
        }
        Ok(LMRSA {
            _size: size,
            nvars,
            dim,
            sorted,
            pfxsum,
            rank,
        })
    }

    /// O(dim) minsum when the source variable is part of the table.
    ///
    /// Sums the per-state contribution of [`UnitLMRSA::minsum_wsrc`] across
    /// every state of `about`. `src_idx` is the position of the source
    /// variable inside the table (subset-local index for subset-built
    /// tables).
    pub fn minsum_wsrc(&self, src_idx: usize) -> FloatT {
        let mut rdsum = FloatT::zero();
        for rstate in 0..self.dim {
            let rsbegin = rstate * self.nvars;
            let lmrank = self.rank[rsbegin + src_idx].to_usize().unwrap();
            let lmv = self.sorted[rsbegin + lmrank];
            let lmlow = if lmrank > 0 {
                self.pfxsum[rsbegin + lmrank - 1]
            } else {
                FloatT::zero()
            };
            let lmhigh =
                FloatT::from_usize(self.nvars - 1 - lmrank).unwrap() * lmv;
            rdsum += lmlow + lmhigh;
        }
        rdsum
    }

    /// O(dim · log nvars) minsum when the source LMR vector is not in the
    /// table.
    ///
    /// `lmd` is the externally supplied `LMR(about, src)` vector of length
    /// `dim`. For each state the function binary-searches the sorted
    /// segment to find where `lmd[rstate]` would be inserted, then applies
    /// the same prefix-sum / "tail count" decomposition as
    /// [`minsum_wsrc`](Self::minsum_wsrc). Used by
    /// [`LMRSubsetDataStructure::minsum`] when the requested source is
    /// outside the active subset.
    pub fn minsum_nosrc(&self, lmd: ArrayView1<FloatT>) -> FloatT {
        assert!(lmd.len() == self.dim);
        let mut rdsum = FloatT::zero();
        for (rstate, lmv) in lmd.iter().enumerate() {
            let rsbegin = rstate * self.nvars;
            let rsend = rsbegin + self.nvars;
            let lmrank = match self
                .sorted
                .slice(ndarray::s![rsbegin..rsend])
                .as_slice()
                .unwrap()
                .binary_search_by(|x| x.total_cmp(lmv))
            {
                Ok(pos) => pos,
                Err(pos) => pos,
            };
            let lmlow = if lmrank > 0 {
                self.pfxsum[rsbegin + lmrank - 1]
            } else {
                FloatT::zero()
            };
            let lmhigh = FloatT::from_usize(self.nvars - lmrank).unwrap() * *lmv;
            rdsum += lmlow + lmhigh;
        }
        rdsum
    }
}

/// Wrapper bundling an [`LMRSA`] with the `about` variable it was built
/// for, sized for the entire variable set.
///
/// Consumed by [`crate::mvim::misi::MISIRangePair`] and
/// [`crate::mvim::misi::MISIPair`] to compute PUC orders of magnitude faster
/// than [`MRVTrait::compute_lm_puc`]'s default linear scan.
pub struct LMRDataStructure<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    /// Variable this table is centered on; arguments to [`minsum`] are
    /// interpreted as `LMR(about, src_var)`.
    about: IntT,
    /// Total number of variables in the underlying distribution.
    nvars: usize,
    /// Sorted/prefix-sum/rank tables, one segment per state of `about`.
    lmr: LMRSA<IntT, FloatT>,
}

impl<IntT, FloatT> LMRDataStructure<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    /// Build an [`LMRDataStructure`] for `about` covering every variable
    /// in `pidata`.
    pub fn new(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
    ) -> Result<Self, Error> {
        Ok(LMRDataStructure {
            about,
            nvars: pidata.nvariables(),
            lmr: LMRSA::from_mrv_trait(pidata, about)?,
        })
    }

    /// `about` variable this table was built for.
    pub fn get_about(&self) -> IntT {
        self.about
    }

    /// PUC scaling factor `nvars - 2` — the count of "by" variables that
    /// can sit between an edge's two endpoints in the full distribution.
    /// `_target` is unused but accepted to match the subset-aware signature.
    pub fn mi_factor(&self, _target: IntT) -> usize {
        self.nvars - 2
    }

    /// Aggregate minsum `Σ_state Σ_{by} min(LMR(about, src), LMR(about, by))`.
    ///
    /// Forwards to [`LMRSA::minsum_wsrc`] using `src_var` as the table
    /// index (since the table covers every variable, `src_var` is its own
    /// position).
    pub fn minsum(&self, src_var: IntT) -> Result<FloatT, Error> {
        Ok(self.lmr.minsum_wsrc(src_var.to_usize().unwrap()))
    }
}

/// Subset-restricted variant of [`LMRDataStructure`].
///
/// Stores LMR tables only for the variables in `subset_map`. Queries whose
/// `src_var` is in the subset are O(dim) lookups via
/// [`LMRSA::minsum_wsrc`]; queries for `src_var` outside the subset fall
/// back to [`LMRSA::minsum_nosrc`] after fetching the relevant LMR vector
/// from the parent [`MRVTrait`].
pub struct LMRSubsetDataStructure<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    /// External-id → subset-local-index map shared across the pair of
    /// tables built for one MISI record.
    subset_map: Rc<HashMap<IntT, usize>>,
    /// Variable this table is centered on.
    about: IntT,
    /// Number of subset variables tracked (`subset_map.len()`).
    nvars: usize,
    /// Sorted/prefix-sum/rank tables, one segment per state of `about`,
    /// covering only the subset variables.
    lmr: LMRSA<IntT, FloatT>,
}

impl<IntT, FloatT> LMRSubsetDataStructure<IntT, FloatT>
where
    IntT: 'static + PNInteger,
    FloatT: 'static + PNFloat,
{
    /// Build a subset-restricted table for `about` keyed by `subset_map`.
    pub fn new(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
        subset_map: Rc<HashMap<IntT, usize>>,
    ) -> Result<Self, Error> {
        Ok(LMRSubsetDataStructure {
            about,
            nvars: subset_map.len(),
            lmr: LMRSA::from_subset_map(pidata, about, &subset_map)?,
            subset_map,
        })
    }

    /// `about` variable this table was built for.
    pub fn get_about(&self) -> IntT {
        self.about
    }

    /// PUC scaling factor for the subset.
    ///
    /// Returns `nvars - (1 if about is in the subset else 0) - (1 if target
    /// is in the subset else 0)` so the count matches the number of
    /// non-endpoint subset variables used in the underlying minsum.
    pub fn mi_factor(&self, target: IntT) -> usize {
        self.nvars
            - (if self.subset_map.contains_key(&self.about) {
                1
            } else {
                0
            } + if self.subset_map.contains_key(&target) {
                1
            } else {
                0
            })
    }

    /// Aggregate minsum that handles both in-subset and out-of-subset
    /// sources transparently.
    ///
    /// * If `src_var` is in `subset_map`, dispatch to
    ///   [`LMRSA::minsum_wsrc`] with the subset-local index.
    /// * Otherwise pull `LMR(about, src_var)` from `pidata` and route
    ///   through [`LMRSA::minsum_nosrc`].
    pub fn minsum(
        &self,
        pidata: &impl MRVTrait<IntT, FloatT>,
        src_var: IntT,
    ) -> Result<FloatT, Error> {
        if self.subset_map.contains_key(&src_var) {
            Ok(self.lmr.minsum_wsrc(self.subset_map[&src_var]))
        } else {
            let lmd = pidata.get_lmr(self.about, src_var)?;
            Ok(self.lmr.minsum_nosrc(lmd.view()))
        }
    }
}

#[cfg(test)]
mod tests {
    // Some Tests in misi.rs
}
