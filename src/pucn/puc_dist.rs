//! Distributed PUC workflow using a  Flattened-LMR method.
//!
//! Where [`crate::pucn::puc`] computes PUC scores from the
//! [`MISIRangePair`](crate::mvim::misi::MISIRangePair) view (one
//! per rectangular pair range), this module operates on the flat
//! `lmr` / `mi` arrays produced by the MI/SI workflows and lifted
//! across MPI ranks via [`LMRSA`]. The workflow
//! ([`PUCDistWorkflow`]) is sample-free.
//!
//! The module groups three layers:
//!
//! * [`IndexLU`] — indexing/lookup over the `data/lmr`,
//!   `data/mi`, `data/si_start` and `data/hist_dim` arrays. Provides
//!   `(about, by) ↔ flat_index` mappings.
//! * [`MIPairIterator`] / [`LMRPairIterator`] — companion iterators
//!   that walk an MI-index range or an LMR-index range and yield the
//!   matching `(x, y)` / `(about, by)` variable pairs.
//! * [`DistLMR`] — distributed view of the flat MI / LMR arrays
//!   parameterised by a [`DistMode`] (currently
//!   [`DistMode::VarUniform`] is the supported mode). Holds the
//!   per-rank [`InterleavedDist`] / [`ArbitDist`] descriptors and
//!   the local slabs of `mi` and `lmr` data.
//! * [`DistLMRMinSum`] — per-rank `(x, y, minsum)` table built from
//!   a [`DistLMR`] using [`LMRSA::minsum_wsrc`]; the result is
//!   re-dist with paired `all2all` exchanges so each rank ends
//!   up owning the records for the MI pairs it stores.
//! * [`PUCDistWorkflow`] / [`PUCDistWorkFlowHelper`] — the public
//!   driver and stateless helper namespace that wire the layers
//!   together and save the final [`PUCResults`] as HDF5 file.

use super::{WorkflowArgs, puc::PUCRTrait, puc::PUCResults};
use crate::{
    comm::CommIfx,
    cond_info,
    h5::{io, mpio},
    map_with_result_to_tuple,
    mvim::rv::LMRSA,
    types::{PNFloat, PNInteger},
    util::{exc_prefix_sum, triu_index_to_pair, triu_pair_to_index},
};

use anyhow::Result;
use hdf5::{File, H5Type};
use itertools::Itertools;
use mpi::{datatype::DatatypeRef, traits::Equivalence};
use ndarray::{Array1, Array2};
use num::ToPrimitive;
use sope::{
    collective::{all2all_vec, all2allv_vec, allgather_one},
    gather_debug,
    partition::{ArbitDist, Dist, InterleavedDist},
    timer::SectionTimer,
    traits::GEquivalence,
};
use std::{
    fmt::Display,
    iter::zip,
    marker::PhantomData,
    ops::{Div, Range},
};

/// Index lookups for the SI/LMR distributed PUC scheme.
///
/// Holds the LMR start offsets and histogram dimensions loaded from
/// the MISI HDF5 file, plus the global counts (`nvars`,
/// `nlmr`, `npairs`). Provides `(about, by) ↔ flat_index` conversions
/// for both the LMR layout (`nvars x nvars` grid keyed by
/// `lmr_start_idx`) and the MI layout (upper-triangular,
/// `npairs = nvars * (nvars - 1) / 2`).
pub struct IndexLU {
    /// LMR/SI start offset (length `nvars`), shared by
    /// the `data/si_start` and `data/lmr_start` HDF5 datasets.
    lmr_start: Vec<usize>,
    /// Histogram dimension (length `nvars`). Used to
    /// stride within the `lmr` slab for one `about`.
    hist_dim: Vec<usize>,
    /// Total number of variables.
    nvars: usize,
    /// Total length of the flat `lmr` / `si` arrays.
    nlmr: usize,
    /// Number of upper-triangular pairs (length of the flat `mi` array).
    npairs: usize,
}

impl Display for IndexLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{nvars : {}, nsi: {}, npairs: {}, starts: {}, hist: {}}}",
            self.nvars,
            self.nlmr,
            self.npairs,
            self.lmr_start.len(),
            self.hist_dim.len(),
        )
    }
}

impl IndexLU {
    /// Build an [`IndexLU`] by reading the dimension / offset
    /// metadata from `misi_data_file`.
    ///
    /// Loads the `data/hist_dim` and `data/si_start` datasets 
    /// (LMR and SI have same starts/sizes) plus the `nvars`,
    /// `npairs` and `nsi` scalar attributes.
    /// `SizeT` is the on-disk type of the offset / count datasets and `IntT`
    /// of `hist_dim`.
    pub fn new<SizeT, IntT>(misi_data_file: &str) -> Result<Self>
    where
        SizeT: ToPrimitive + H5Type,
        IntT: ToPrimitive + H5Type,
    {
        let fptr = File::open(misi_data_file)?;
        let data_g = fptr.group("data")?;
        let hist_dim: Vec<usize> = data_g
            .dataset("hist_dim")?
            .read_1d::<IntT>()?
            .map(|x| x.to_usize().unwrap())
            .into_iter()
            .collect();
        // NOTE:: SI and LMR have the same starts
        let lmr_start: Vec<usize> = data_g
            .dataset("si_start")?
            .read_1d::<SizeT>()?
            .map(|x| x.to_usize().unwrap())
            .into_iter()
            .collect();
        // NOTE:: SI and LMR have the same size
        let (nvars, npairs, nlmr) = map_with_result_to_tuple![
            |x| io::read_scalar_attr::<SizeT>(&data_g, x) ;
            "nvars", "npairs", "nsi"
        ];
        Ok(Self {
            lmr_start,
            hist_dim,
            nvars: nvars.to_usize().unwrap(),
            nlmr: nlmr.to_usize().unwrap(),
            npairs: npairs.to_usize().unwrap(),
        })
    }

    /// Offset of the first element of the LMR slab for `(v_about, v_by)`: 
    /// `lmr_start[v_about] + v_by * hist_dim[v_about]`.
    fn lmr_start_idx(&self, (v_about, v_by): (usize, usize)) -> usize {
        self.lmr_start[v_about] + v_by * self.hist_dim[v_about]
    }

    /// Offset of the one-past-the-end element of the LMR slab
    /// for `(v_about, v_by)`.
    fn lmr_end_idx(&self, (v_about, v_by): (usize, usize)) -> usize {
        self.lmr_start[v_about] + (v_by + 1) * self.hist_dim[v_about]
    }

    /// Half-open LMR offset range covering every `(v_about, *)` slab.
    fn lmr_var_bounds(&self, v_about: usize) -> Range<usize> {
        self.lmr_start_idx((v_about, 0))
            ..self.lmr_end_idx((v_about, self.nvars - 1))
    }

    /// Reverse of [`Self::lmr_start_idx`] in the `about`
    /// coordinate. Returns the variable whose LMR slab contains the
    /// flat index `idx`.
    fn lmr_about(&self, idx: usize) -> usize {
        let var = self.lmr_start.partition_point(|&x| x <= idx);
        if var == 0 {
            0
        } else if var >= self.lmr_start.len() {
            self.lmr_start.len() - 1
        } else {
            assert!(idx >= self.lmr_start[var - 1]);
            var - 1
        }
    }

    /// Reverse of [`Self::lmr_start_idx`] in the `by` coordinate.
    /// returns the conditioning variable for `idx`, given the owning
    /// `about` resolved by [`Self::lmr_about`].
    fn lmr_by(&self, idx: usize, about: usize) -> usize {
        assert!(about < self.lmr_start.len());
        assert!(idx >= self.lmr_start[about]);
        let (abt_start, abt_dim) = (self.lmr_start[about], self.hist_dim[about]);
        let si_surplus = idx - abt_start;
        // NOTE:: this can be div or div_ceil? depending on how it is owned
        si_surplus.div(abt_dim).min(self.lmr_start.len() - 1)
    }

    /// Combined reverse lookup: `idx -> (about, by)` over the LMR layout.
    fn lmr_index2pair(&self, idx: usize) -> (usize, usize) {
        let sabt = self.lmr_about(idx);
        (sabt, self.lmr_by(idx, sabt))
    }

    /// Reverse lookup over the upper-triangular MI layout
    /// (forwards to [`triu_index_to_pair`]).
    fn mi_index2pair(&self, idx: usize) -> (usize, usize) {
        triu_index_to_pair(self.nvars, idx)
    }

    /// Successor in row-major LMR order; wraps from
    /// `(about, nvars - 1)` to `(about + 1, 0)`.
    fn lmr_next_to(&self, (about, by): (usize, usize)) -> (usize, usize) {
        if by == self.nvars - 1 {
            (about + 1, 0)
        } else {
            (about, by + 1)
        }
    }

    /// Successor in row-major upper-triangular MI order; wraps from
    /// `(x, nvars - 1)` to `(x + 1, x + 2)`.
    fn mi_next_to(&self, (x, y): (usize, usize)) -> (usize, usize) {
        if y == self.nvars - 1 {
            (x + 1, x + 2)
        } else {
            (x, y + 1)
        }
    }

    /// Forward lookup over the upper-triangular MI layout; the
    /// arguments may be in either order.
    pub fn mi_pair2index(&self, (x, y): (usize, usize)) -> usize {
        let (x, y) = if x < y { (x, y) } else { (y, x) };
        triu_pair_to_index(self.nvars, x, y)
    }

    /// Histogram dimension for variable `x`.
    pub fn dim(&self, x: usize) -> usize {
        self.hist_dim[x]
    }
}

/// Iterator over the upper-triangular MI pairs covered by the
/// `mi_range` flat-index window.
///
/// On the first step seeds the cursor with the pair corresponding to 
/// `mi_range.start` and then advances with [`IndexLU::mi_next_to`].
pub struct MIPairIterator<'a> {
    /// Index lookup the iterator walks over.
    ixlu: &'a IndexLU,
    /// Half-open range of flat MI indices to traverse.
    mi_range: Range<usize>,
    /// Cached current `(x, y)` pair.
    c_item: (usize, usize),
    /// Flat-index cursor.
    counter: usize,
}

impl<'a> MIPairIterator<'a> {
    /// Build an iterator over the MI pairs whose flat indices fall in
    /// `mi_range`.
    pub fn new(ixlu: &'a IndexLU, mi_range: Range<usize>) -> Self {
        Self {
            counter: mi_range.start,
            c_item: (0, 0),
            mi_range,
            ixlu,
        }
    }
}

impl<'a> Iterator for MIPairIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.mi_range.end {
            self.c_item = (0, 0);
            None
        } else if self.counter == self.mi_range.start {
            self.c_item = self.ixlu.mi_index2pair(self.mi_range.start);
            self.counter += 1;
            Some(self.c_item)
        } else {
            self.c_item = self.ixlu.mi_next_to(self.c_item);
            self.counter += 1;
            Some(self.c_item)
        }
    }
}

/// Iterator over `(about, by)` LMR pairs covered by the `lmr_range`
/// index window.
///
/// This iterator calls [`IndexLU::lmr_index2pair`] on every
/// step so consecutive flat indices that resolve to the same pair
/// emit the pair multiple times.
pub struct LMRPairIterator<'a> {
    /// Index lookup the iterator walks over.
    ixlu: &'a IndexLU,
    /// Half-open range of flat LMR indices to traverse.
    lmr_range: Range<usize>,
    /// Cached current `(about, by)` pair.
    c_item: (usize, usize),
    /// Flat-index cursor.
    c_index: usize,
}

impl<'a> LMRPairIterator<'a> {
    /// Build an iterator over the LMR pairs whose flat indices fall
    /// in `lmr_range`.
    pub fn new(ixlu: &'a IndexLU, lmr_range: Range<usize>) -> Self {
        Self {
            c_index: lmr_range.start,
            c_item: (0, 0),
            lmr_range,
            ixlu,
        }
    }
}

impl<'a> Iterator for LMRPairIterator<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.c_index >= self.lmr_range.end {
            self.c_item = (0, 0);
            None
        } else if self.c_index >= self.lmr_range.start {
            self.c_item = self.ixlu.lmr_index2pair(self.c_index);
            self.c_index += 1;
            Some(self.c_item)
        } else {
            self.c_index = self.lmr_range.end;
            self.c_item = (0, 0);
            None
        }
    }
}

/// Distribution scheme for [`DistLMR`].
#[derive(PartialEq, Clone)]
pub enum DistMode {
    /// Uniformly partition the `nlmr` flatttend LMR  across ranks.
    /// NOTE:: fully supported yet. 
    /// Currently only works with  [`DistLMR::si_local_range_for`] reverse path 
    /// TODO: the rest of the pipeline.
    LMRUniform,
    /// Uniformly partition the `nvars` variables across ranks; each
    /// rank then owns the entire LMR slab for its variable block.
    /// This is the mode chosen by [`PUCDistWorkflow::run`].
    VarUniform,
}

/// Distributed view of the flat MI and LMR arrays.
///
/// Produced by the MISI workflow, it combines:
///
/// * the [`IndexLU`] metadata and a [`DistMode`] selector;
/// * three partition descriptors —
///   [`InterleavedDist`] over variables (`var_dist`),
///   [`InterleavedDist`] over MI pairs (`mi_dist`),
///   [`ArbitDist`] over LMR slots (`lmr_dist`); and
/// * the local slabs of `mi` and `lmr` data backed by parallel HDF5 reads.
///
/// NOTE::  `si` is intentionally not loaded here — only the LMR vectors are 
/// needed for the distributed PUC computation.
pub struct DistLMR<FloatT> {
    /// Index lookup over the flat layout.
    ixlu: IndexLU, // index lookup
    /// LMR partition scheme (sized from local slab size).
    lmr_dist: ArbitDist,
    /// MI partition scheme (uniform-block over `npairs`).
    mi_dist: InterleavedDist,
    /// Per-rank variable partition (uniform-block over `nvars`).
    var_dist: InterleavedDist,
    //si: Array1<FloatT>, // NOTE:: SI not needed for now
    /// Local slab of the flat `data/lmr` array.
    lmr: Array1<FloatT>,
    /// Local slab of the flat `data/mi` array.
    mi: Array1<FloatT>,
    /// Active distribution scheme.
    mode: DistMode,
}

impl<FloatT> Display for DistLMR<FloatT>
where
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        write!(
            f,
            "{{lmr_bounds: {:?}, lmr_dist: {:?}, lmr: {}}}, ",
            self.si_pair_bounds(),
            self.lmr_dist.range(),
            self.lmr.len(),
        )?;
        write!(
            f,
            "{{mi_bounds: {:?},  mi_dist: {:?}, mi: {}}}",
            self.mi_pair_bounds(),
            self.mi_dist.range(),
            self.mi.len(),
        )?;
        write!(f, "}}")
    }
}

impl<FloatT> DistLMR<FloatT>
where
    FloatT: 'static + PNFloat + H5Type,
{
    /// `(about, by)` pair of the first SI/LMR slab owned by process ranked
    /// `rank` for the given [`DistMode`].
    pub fn si_start_for(
        ixlu: &IndexLU,
        nproc: i32,
        rank: i32,
        mode: DistMode,
    ) -> (usize, usize) {
        match mode {
            DistMode::LMRUniform => {
                let si_begin = ixlu.lmr_index2pair(InterleavedDist::block_start(
                    ixlu.nlmr, nproc, rank,
                ));
                if rank > 0 {
                    ixlu.lmr_next_to(si_begin)
                } else {
                    si_begin
                }
            }
            DistMode::VarUniform => {
                (InterleavedDist::block_start(ixlu.nvars, nproc, rank), 0)
            }
        }
    }

    /// Companion to [`Self::si_start_for`]: returns the last
    /// `(about, by)` pair owned by `rank` (inclusive) under the
    /// given [`DistMode`].
    pub fn si_end_for(
        ixlu: &IndexLU,
        nproc: i32,
        rank: i32,
        mode: DistMode,
    ) -> (usize, usize) {
        match mode {
            DistMode::LMRUniform => ixlu.lmr_index2pair(
                InterleavedDist::block_end(ixlu.nlmr, nproc, rank),
            ),
            DistMode::VarUniform => (
                InterleavedDist::block_end(ixlu.nvars, nproc, rank) - 1,
                ixlu.nvars - 1,
            ),
        }
    }

    /// Bundle [`Self::si_start_for`] and [`Self::si_end_for`] into
    /// a `(start_pair, end_pair)` tuple.
    pub fn si_bounds_for(
        ixlu: &IndexLU,
        nproc: i32,
        rank: i32,
        mode: DistMode,
    ) -> ((usize, usize), (usize, usize)) {
        (
            Self::si_start_for(ixlu, nproc, rank, mode.clone()),
            Self::si_end_for(ixlu, nproc, rank, mode.clone()),
        )
    }

    /// Construct a [`DistLMR`] from a pre-computed [`IndexLU`] and
    /// the local LMR block size.
    ///
    /// All-gathers `local_size` for `lmr` [`Dist`], then runs collective 
    /// parallel-IO reads ([`crate::h5::mpio::block_read1d`])
    /// for `data/mi` & `data/lmr`.
    pub fn from_ixlu(
        local_size: usize,
        ixlu: IndexLU,
        mode: DistMode,
        cx: &CommIfx,
        h5f: &str,
    ) -> Result<Self> {
        //
        let sizes = allgather_one(&local_size, cx.comm())?;
        let mi_dist = InterleavedDist::new(ixlu.npairs, cx.size, cx.rank);
        let lmr_dist = ArbitDist::new(ixlu.nlmr, cx.size, cx.rank, sizes);
        let mi = mpio::block_read1d(cx, h5f, "data/mi", Some(&mi_dist))?;
        //let si = mpio::block_read1d(cx, h5f, "data/si", Some(&si_dist))?;
        let lmr = mpio::block_read1d(cx, h5f, "data/lmr", Some(&lmr_dist))?;

        Ok(Self {
            lmr_dist,
            lmr,
            //si, // NOTE:: SI not needed for now
            mi_dist,
            mi,
            var_dist: InterleavedDist::new(ixlu.nvars, cx.size, cx.rank),
            mode,
            ixlu,
        })
    }

    /// Build the [`IndexLU`] from the file at path `h5f`, and use
    /// [`Self::from_ixlu`] to build the rest.
    pub fn with_mode<SizeT, IntT>(
        cx: &CommIfx,
        h5f: &str,
        mode: DistMode,
    ) -> Result<Self>
    where
        SizeT: 'static + PNInteger + H5Type,
        IntT: PNInteger + H5Type,
    {
        let ixlu = IndexLU::new::<SizeT, IntT>(h5f)?;
        let (si_begin, si_end) =
            Self::si_bounds_for(&ixlu, cx.size, cx.rank, mode.clone());
        let idx_begin = ixlu.lmr_start_idx(si_begin);
        let idx_end = ixlu.lmr_end_idx(si_end);
        Self::from_ixlu(idx_end - idx_begin, ixlu, mode, cx, h5f)
    }

    /// Total number of variables (forwarded from [`IndexLU`]).
    pub fn nvars(&self) -> usize {
        self.ixlu.nvars
    }

    /// Histogram dimension of variable `x` (forwarded from [`IndexLU`]).
    pub fn dim(&self, x: usize) -> usize {
        self.ixlu.dim(x)
    }

    /// First `(about, by)` pair owned by this rank in the active [`DistMode`].
    pub fn si_first_pair(&self) -> (usize, usize) {
        Self::si_start_for(
            &self.ixlu,
            self.lmr_dist.comm_size() as i32,
            self.lmr_dist.comm_rank() as i32,
            self.mode.clone(),
        )
    }

    /// Last `(about, by)` pair (inclusive) owned by this rank.
    pub fn si_last_pair(&self) -> (usize, usize) {
        Self::si_end_for(
            &self.ixlu,
            self.lmr_dist.comm_size() as i32,
            self.lmr_dist.comm_rank() as i32,
            self.mode.clone(),
        )
    }

    /// `(first_pair, last_pair)` SI/LMR bounds owned by this rank.
    pub fn si_pair_bounds(&self) -> ((usize, usize), (usize, usize)) {
        Self::si_bounds_for(
            &self.ixlu,
            self.lmr_dist.comm_size() as i32,
            self.lmr_dist.comm_rank() as i32,
            self.mode.clone(),
        )
    }

    /// First MI pair owned by this rank.
    pub fn mi_first_pair(&self) -> (usize, usize) {
        self.ixlu.mi_index2pair(self.mi_dist.start())
    }

    /// Last MI pair (inclusive) owned by this rank.
    pub fn mi_last_pair(&self) -> (usize, usize) {
        self.ixlu.mi_index2pair(self.mi_dist.end() - 1)
    }

    /// `(first_pair, last_pair)` MI bounds owned by this rank.
    pub fn mi_pair_bounds(&self) -> ((usize, usize), (usize, usize)) {
        (self.mi_first_pair(), self.mi_last_pair())
    }

    /// Local range (relative to this rank's `lmr` slab) covering the
    /// LMR entries corresponding to variable `v_about`.
    ///
    /// NOTE:: Only implemented for [`DistMode::VarUniform`]; calling this
    /// in [`DistMode::LMRUniform`] panics with `todo!`.
    pub fn si_local_range_for(&self, v_about: usize) -> Range<usize> {
        match self.mode {
            DistMode::VarUniform => {
                let si_gstart = self.lmr_dist.start();
                let si_gend = self.lmr_dist.end();
                let v_gidx = self.ixlu.lmr_start_idx((v_about, 0));
                assert!(si_gstart <= v_gidx);
                assert!(v_gidx < si_gend);
                let v_start_idx = v_gidx - si_gstart;
                let v_size = self.dim(v_about) * self.nvars();
                let v_end_idx = v_start_idx + v_size;
                //let v_end_idx = v_end_idx.min(self.si.len());
                v_start_idx..v_end_idx
            }
            DistMode::LMRUniform => {
                // TODO:: Need to sort, block segment prefix sum
                todo!("For non over lapping")
            }
        }
    }

    /// Translate a global `(x, y)` MI pair into its offset inside
    /// this rank's `mi` slab. Asserts ownership.
    pub fn mi_local_index(&self, (x, y): (usize, usize)) -> usize {
        let mi_start = self.mi_dist.start();
        let m_gidx = self.ixlu.mi_pair2index((x, y));
        assert!(mi_start <= m_gidx);
        assert!(m_gidx < self.mi_dist.end());
        m_gidx - mi_start
    }

    /// Owner rank of `(x, y)` under the MI partition
    /// ([`InterleavedDist::owner`]).
    pub fn mi_pair2owner(&self, (x, y): (usize, usize)) -> i32 {
        let midx = self.ixlu.mi_pair2index((x, y));
        self.mi_dist.owner(midx)
    }

    /// Per-destination send counts for the SI → MI re-shuffle:
    /// `pvec[r]` is the number of off-diagonal local LMR slots whose
    /// MI pair is owned by rank `r`.
    pub fn si2mi_counts(&self) -> Vec<usize> {
        // TODO:: verify if it is the same for both modes
        let mut pvec = vec![0; self.lmr_dist.comm_size() as usize];
        for (x, y) in LMRPairIterator::new(&self.ixlu, self.lmr_dist.range()) {
            if x == y {
                continue;
            }
            let mown = self.mi_pair2owner((x, y)) as usize;
            pvec[mown] += 1;
        }
        pvec
    }

    /// Like [`Self::si2mi_counts`] but counts each `(x, y)` pair
    /// only once (using `Itertools::unique`), so the result matches
    /// the number of distinct minsum entries the rank will produce.
    pub fn si2mi_unique_counts(&self) -> Vec<usize> {
        // TODO:: verify if it is the same for both modes
        let mut pvec = vec![0; self.lmr_dist.comm_size() as usize];
        let siter = LMRPairIterator::new(&self.ixlu, self.lmr_dist.range());
        for (x, y) in siter.unique() {
            if x == y {
                continue;
            }
            let mown = self.mi_pair2owner((x, y)) as usize;
            pvec[mown] += 1;
        }
        pvec
    }

    /// Borrow the lice slice of LMR slab corresponding to variable `v_about`,
    /// range identified by [`Self::si_local_range_for`].
    pub fn local_lmr_slice_for(&self, v_about: usize) -> &[FloatT] {
        let v_range = self.si_local_range_for(v_about);
        &self.lmr.as_slice().unwrap_or_default()[v_range]
    }
}

/// Entry struct used for exchange during the SI → MI re-shuffle in
/// [`DistLMRMinSum`].
///
/// Each entry encodes the pair `(x, y)` (with `x < y`), its precomputed
/// minsum value, and an `ind` flag that records whether the original
/// `about < by` orientation so the consumer can preserve
/// direction-specific information.
#[derive(GEquivalence, Clone, Default)]
struct LMREntry<IntT, FloatT> {
    /// First (smaller) variable index of the pair.
    x: IntT,
    /// Second (larger) variable index of the pair.
    y: IntT,
    /// Local minsum value contributed by this rank.
    msum: FloatT,
    /// `true` when the entry was generated from a `v_about < v_by`
    /// pairing, `false` otherwise.
    ind: bool,
}

/// Tuple of `(entries, send_counts)` 
/// with send_counts is a vector with per-rank destination counts.
type LMREntryList<IntT, FloatT> = (Vec<LMREntry<IntT, FloatT>>, Vec<usize>);

/// Distributed `(x, y, minsum)` table produced from a [`DistLMR`].
///
/// For every `(v_about, v_by)` pair that a process owns 
/// this object is  expected to include the entries for the MI pairs it stores.
pub struct DistLMRMinSum<'a, IntT, FloatT> {
    /// Borrow of the underlying [`DistLMR`].
    dist: &'a DistLMR<FloatT>,
    /// Smaller index of each owned canonicalised pair.
    pair_x: Array1<IntT>,
    /// Larger index of each owned canonicalised pair.
    pair_y: Array1<IntT>,
    /// Minsum value matching the pair at the same offset.
    minsum: Array1<FloatT>,
    /// Direction indicator copied from [`LMREntry::ind`].
    minsum_ind: Array1<bool>,
}

impl<'a, IntT, FloatT> DistLMRMinSum<'a, IntT, FloatT>
where
    IntT: PNInteger
        + H5Type
        + Default
        + Clone
        + Equivalence<Out = DatatypeRef<'static>>,
    FloatT: 'static
        + PNFloat
        + H5Type
        + Default
        + Clone
        + Equivalence<Out = DatatypeRef<'static>>,
{
    /// Build the local pre-shuffle [`LMREntry`] vector and its
    /// per-destination send counts under given mode.
    ///
    /// For every owned `v_about`,  constructs an [`LMRSA`], and computes
    /// [`LMRSA::minsum_wsrc`] for each `v_by != v_about`. The
    /// resulting entries are placed into the destination-grouped
    /// `entries` vector at rank-specific positions.
    fn init_unif_vars(
        cx: &CommIfx,
        dist: &'a DistLMR<FloatT>,
    ) -> Result<LMREntryList<IntT, FloatT>> {
        assert!(dist.mode == DistMode::VarUniform);
        let s_timer = SectionTimer::from_comm(cx.comm(), ",");
        let var_ints: Vec<IntT> = (0..dist.nvars())
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        //  1. allocate minsum, indicator vec with capacity
        let counts = dist.si2mi_unique_counts();
        let mut offsets: Vec<usize> =
            exc_prefix_sum(counts.clone().into_iter(), 1);
        let size = counts.iter().sum::<usize>();
        let mut nctx = 0;
        let mut entries = vec![LMREntry::<IntT, FloatT>::default(); size];
        for v_about in dist.var_dist.range() {
            // 1. lmr slice corresponding to variable vi
            let lslice = dist.local_lmr_slice_for(v_about);
            let hdim = dist.dim(v_about);
            // 2. Build LMRSA
            let lmr_sa = LMRSA::<IntT, FloatT>::from_lmr_slice(
                IntT::from_usize(hdim).unwrap_or_default(),
                dist.nvars(),
                lslice,
            )?;
            // 3. Minsum values and push into arrays
            for v_by in 0..dist.nvars() {
                if v_by == v_about {
                    continue;
                }
                let msum = lmr_sa.minsum_wsrc(v_by);
                // 4. Place as the specific offests.
                let mown = dist.mi_pair2owner((v_about, v_by)) as usize;
                let loc = offsets[mown];
                entries[loc].msum = msum;
                if v_about < v_by {
                    entries[loc].x = var_ints[v_about];
                    entries[loc].y = var_ints[v_by];
                    entries[loc].ind = true;
                } else {
                    entries[loc].x = var_ints[v_by];
                    entries[loc].y = var_ints[v_about];
                    entries[loc].ind = false;
                }
                offsets[mown] += 1;
                nctx += 1;
            }
        }
        s_timer.info_section("Dist PUC::LMR Minsum:: LMR Build");
        s_timer.reset();
        gather_debug!(cx.comm(); "COUNTS {:?}", counts);
        gather_debug!(cx.comm(); "SX {} {}", nctx, size);
        debug_assert!(nctx == size);
        debug_assert!(itertools::all(entries.iter(), |ety| ety.x < ety.y));
        Ok((entries, counts))
    }

    /// Build a [`DistLMRMinSum`] under [`DistMode::VarUniform`] mode.
    ///
    /// Calls [`Self::init_unif_vars`] to construct the local entry
    /// list, exchanges the entries across ranks,
    /// then unpacks the received entries vector into the parallel
    /// `pair_x` / `pair_y` / `minsum` / `minsum_ind` arrays.
    pub fn from_unif_vars(
        cx: &CommIfx,
        dist: &'a DistLMR<FloatT>,
    ) -> Result<Self> {
        let (entries, counts) = Self::init_unif_vars(cx, dist)?;
        let entries = if cx.size > 1 {
            let s_timer = SectionTimer::from_comm(cx.comm(), ",");
            let recv_counts = all2all_vec(&counts, cx.comm())?;
            gather_debug!(cx.comm(); "RECV_COUNTS {:?}", recv_counts);
            let entries =
                all2allv_vec(&entries, &counts, &recv_counts, cx.comm())?;
            gather_debug!(cx.comm(); "ENTRIES {:?}", entries.len());
            debug_assert!(itertools::all(entries.iter(), |ety| ety.x < ety.y));
            s_timer.info_section("Dist PUC::LMR Minsum:: LMR All2All");
            entries
        } else {
            entries
        };
        let size = entries.len();
        let pair_x = Array1::from_shape_fn(size, |i| entries[i].x);
        let pair_y = Array1::from_shape_fn(size, |i| entries[i].y);
        let minsum = Array1::from_shape_fn(size, |i| entries[i].msum);
        let minsum_ind = Array1::from_shape_fn(size, |i| entries[i].ind);

        Ok(Self {
            dist,
            pair_x,
            pair_y,
            minsum,
            minsum_ind,
        })
    }
}

/// Driver for the flattened distributed PUC workflow
/// ([`crate::pucn::RunMode::PUCLMRDist`]).
///
/// Borrows the MPI context and parsed configuration.
pub struct PUCDistWorkflow<'a> {
    /// MPI communicator wrapper used by every collective call.
    pub mpi_ifx: &'a CommIfx,
    /// Parsed workflow configuration.
    pub args: &'a WorkflowArgs,
}

/// Stateless namespace for the per-step helpers used by
/// [`PUCDistWorkflow`].
///
/// Carries a [`PhantomData`] marker so the three numeric type
/// parameters are bound once at the call site
/// (`PUCDistWorkFlowHelper::<i64, i32, f32>::var_uniform_dist(...)`).
struct PUCDistWorkFlowHelper<SizeT, IntT, FloatT> {
    /// Phantom marker for the three numeric type parameters.
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> PUCDistWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger
        + H5Type
        + Default
        + Clone
        + Equivalence<Out = DatatypeRef<'static>>,
    FloatT: 'static
        + PNFloat
        + H5Type
        + Default
        + Clone
        + Equivalence<Out = DatatypeRef<'static>>,
{
    /// Build the [`DistLMR`] distribution from the workflow's MISI HDF5 file.
    fn var_uniform_dist(wf: &PUCDistWorkflow) -> Result<DistLMR<FloatT>> {
        DistLMR::<FloatT>::with_mode::<SizeT, IntT>(
            wf.mpi_ifx,
            &wf.args.misi_data_file,
            DistMode::VarUniform,
        )
    }

    /// Build the per-rank [`DistLMRMinSum`] table.
    fn var_uniform_lmr_minsum<'a>(
        wf: &PUCDistWorkflow,
        dist: &'a DistLMR<FloatT>,
    ) -> Result<DistLMRMinSum<'a, IntT, FloatT>> {
        DistLMRMinSum::<'a, IntT, FloatT>::from_unif_vars(wf.mpi_ifx, dist)
    }

    /// Generate the local MI pair indices owned by this rank as
    /// an `nlocal x 2` index matrix.
    ///
    /// Uses per-rank MI range with [`MIPairIterator`] to build 2-D array.
    fn generate_pairs(dist: &DistLMR<FloatT>) -> Array2<IntT> {
        let mut r_pairs = Array2::<IntT>::zeros((dist.mi.len(), 2));
        let (s_vec, t_vec): (Vec<_>, Vec<_>) =
            MIPairIterator::new(&dist.ixlu, dist.mi_dist.range())
                .map(|(a, b)| {
                    (
                        IntT::from_usize(a).unwrap_or_default(),
                        IntT::from_usize(b).unwrap_or_default(),
                    )
                })
                .collect();
        r_pairs
            .slice_mut(ndarray::s![.., 0])
            .assign(&Array1::from_vec(s_vec));
        r_pairs
            .slice_mut(ndarray::s![.., 1])
            .assign(&Array1::from_vec(t_vec));

        r_pairs
    }

    /// Combine the local `mi` slab with the [`DistLMRMinSum`] table
    /// to produce the per-rank [`PUCResults`].
    fn compute_puc(
        dist: &DistLMR<FloatT>,
        dlmr_sum: &DistLMRMinSum<IntT, FloatT>,
    ) -> PUCResults<IntT, FloatT> {
        let r_pindex = Self::generate_pairs(dist);
        let mut r_pucs = Array1::from_vec(vec![FloatT::zero(); r_pindex.nrows()]);
        for ((x, y), minsum) in
            zip(dlmr_sum.pair_x.iter(), dlmr_sum.pair_y.iter())
                .zip(dlmr_sum.minsum.iter())
        {
            let midx = dist
                .mi_local_index((x.to_usize().unwrap(), y.to_usize().unwrap()));
            let mi = dist.mi[midx];
            let puc_update =
                FloatT::from_usize(dist.nvars() - 2).unwrap() - (*minsum / mi);
            r_pucs[midx] += puc_update;
        }
        PUCResults::new(r_pindex, r_pucs)
    }
}

impl<'a> PUCDistWorkflow<'a> {
    /// Run the flattened distributed PUC pipeline.
    ///
    /// 1. Build the [`DistMode::VarUniform`] [`DistLMR`] over
    ///    [`WorkflowArgs::misi_data_file`].
    /// 2. Compute and re-shuffle the per-rank [`DistLMRMinSum`].
    /// 3. Combine the MI and minsum data into the local [`PUCResults`].
    /// 4. Save the results collectively to
    ///    [`WorkflowArgs::puc_file`].
    pub fn run(&self) -> Result<()> {
        type HelperT = PUCDistWorkFlowHelper<i64, i32, f32>;
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        let vu_dist = HelperT::var_uniform_dist(self)?;
        s_timer.info_section("Dist PUC::Build");
        gather_debug!(self.mpi_ifx.comm(); "LMR Distr. {}", vu_dist);
        cond_info!(self.mpi_ifx.is_root(); "Bounds {}", vu_dist.ixlu);
        s_timer.reset();
        let dist_minsum = HelperT::var_uniform_lmr_minsum(self, &vu_dist)?;
        s_timer.info_section("Dist PUC::LMR Minsum");
        gather_debug!(
            self.mpi_ifx.comm();
            "Dist Minsum ({} {} {} {:?}) ({} {} {} {})",
            vu_dist.mi.len(),
            vu_dist.mi[0],
            vu_dist.mi_dist.start(),
            vu_dist.mi_dist.range(),
            dist_minsum.pair_x[0],
            dist_minsum.pair_y[0],
            vu_dist.ixlu.mi_pair2index((
                dist_minsum.pair_x[0].to_usize().unwrap(),
                dist_minsum.pair_y[0].to_usize().unwrap(),
            )),
            dist_minsum.minsum[0],
        );
        s_timer.reset();
        let puc_results = HelperT::compute_puc(&vu_dist, &dist_minsum);
        s_timer.info_section("Dist PUC::Compute PUC");
        s_timer.reset();
        gather_debug!(
            self.mpi_ifx.comm();
            "{} {:?} {}",
            puc_results.len(),
            puc_results.index.slice(ndarray::s![0, ..]),
            puc_results.val[0],
        );

        puc_results.save(self.mpi_ifx, &self.args.puc_file)?;
        s_timer.info_section("Dist PUC::Save Results");
        Ok(())
    }
}
