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
use mpi::traits::Equivalence;
use ndarray::{Array1, Array2};
use num::ToPrimitive;
use sope::{
    collective::{all2all_vec, all2allv_vec, allgather_one},
    gather_debug,
    partition::{ArbitDist, Dist, InterleavedDist},
    timer::SectionTimer,
};
use std::{
    fmt::Display,
    iter::zip,
    marker::PhantomData,
    ops::{Div, Range},
};

// Index lookups for SI an LMR distributed scheme :
//   Mapping between (about, by) pair and flat array index
//   Next functions for iterations
pub struct IndexLU {
    lmr_start: Vec<usize>,
    hist_dim: Vec<usize>,
    nvars: usize,
    nlmr: usize,
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
        let lmr_start: Vec<usize> = data_g
            .dataset("si_start")? // NOTE:: SI and LMR have the same starts
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

    fn lmr_start_idx(&self, (v_about, v_by): (usize, usize)) -> usize {
        self.lmr_start[v_about] + v_by * self.hist_dim[v_about]
    }

    fn lmr_end_idx(&self, (v_about, v_by): (usize, usize)) -> usize {
        self.lmr_start[v_about] + (v_by + 1) * self.hist_dim[v_about]
    }

    fn lmr_var_bounds(&self, v_about: usize) -> Range<usize> {
        self.lmr_start_idx((v_about, 0))
            ..self.lmr_end_idx((v_about, self.nvars - 1))
    }

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

    fn lmr_by(&self, idx: usize, about: usize) -> usize {
        assert!(about < self.lmr_start.len());
        assert!(idx >= self.lmr_start[about]);
        let (abt_start, abt_dim) = (self.lmr_start[about], self.hist_dim[about]);
        let si_surplus = idx - abt_start;
        // NOTE:: this can be div or div_ceil? depending on how it is owned
        si_surplus.div(abt_dim).min(self.lmr_start.len() - 1)
    }

    fn lmr_index2pair(&self, idx: usize) -> (usize, usize) {
        let sabt = self.lmr_about(idx);
        (sabt, self.lmr_by(idx, sabt))
    }

    fn mi_index2pair(&self, idx: usize) -> (usize, usize) {
        triu_index_to_pair(self.nvars, idx)
    }

    fn lmr_next_to(&self, (about, by): (usize, usize)) -> (usize, usize) {
        if by == self.nvars - 1 {
            (about + 1, 0)
        } else {
            (about, by + 1)
        }
    }

    fn mi_next_to(&self, (x, y): (usize, usize)) -> (usize, usize) {
        if y == self.nvars - 1 {
            (x + 1, x + 2)
        } else {
            (x, y + 1)
        }
    }

    pub fn mi_pair2index(&self, (x, y): (usize, usize)) -> usize {
        let (x, y) = if x < y { (x, y) } else { (y, x) };
        triu_pair_to_index(self.nvars, x, y)
    }

    pub fn dim(&self, x: usize) -> usize {
        self.hist_dim[x]
    }
}

pub struct MIPairIterator<'a> {
    ixlu: &'a IndexLU,
    mi_range: Range<usize>,
    c_item: (usize, usize),
    counter: usize,
}

impl<'a> MIPairIterator<'a> {
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

pub struct LMRPairIterator<'a> {
    ixlu: &'a IndexLU,
    lmr_range: Range<usize>,
    c_item: (usize, usize),
    c_index: usize,
}

impl<'a> LMRPairIterator<'a> {
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

#[derive(PartialEq, Clone)]
pub enum DistMode {
    LMRUniform,
    VarUniform,
}

// Distributed for MI and LMR
//  - Local starting and ending pairs
//  - Distribution scheme
//  - Distributed si, lmr and mi
pub struct DistLMR<FloatT> {
    ixlu: IndexLU, // index lookup
    lmr_dist: ArbitDist,
    mi_dist: InterleavedDist,
    var_dist: InterleavedDist,
    //si: Array1<FloatT>, // NOTE:: SI not needed for now
    lmr: Array1<FloatT>,
    mi: Array1<FloatT>,
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

    pub fn nvars(&self) -> usize {
        self.ixlu.nvars
    }

    pub fn dim(&self, x: usize) -> usize {
        self.ixlu.dim(x)
    }

    pub fn si_first_pair(&self) -> (usize, usize) {
        Self::si_start_for(
            &self.ixlu,
            self.lmr_dist.comm_size() as i32,
            self.lmr_dist.comm_rank() as i32,
            self.mode.clone(),
        )
    }

    pub fn si_last_pair(&self) -> (usize, usize) {
        Self::si_end_for(
            &self.ixlu,
            self.lmr_dist.comm_size() as i32,
            self.lmr_dist.comm_rank() as i32,
            self.mode.clone(),
        )
    }

    pub fn si_pair_bounds(&self) -> ((usize, usize), (usize, usize)) {
        Self::si_bounds_for(
            &self.ixlu,
            self.lmr_dist.comm_size() as i32,
            self.lmr_dist.comm_rank() as i32,
            self.mode.clone(),
        )
    }

    pub fn mi_first_pair(&self) -> (usize, usize) {
        self.ixlu.mi_index2pair(self.mi_dist.start())
    }

    pub fn mi_last_pair(&self) -> (usize, usize) {
        self.ixlu.mi_index2pair(self.mi_dist.end() - 1)
    }

    pub fn mi_pair_bounds(&self) -> ((usize, usize), (usize, usize)) {
        (self.mi_first_pair(), self.mi_last_pair())
    }

    // lmr/si range corresponding to the avilable region
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

    pub fn mi_local_index(&self, (x, y): (usize, usize)) -> usize {
        let mi_start = self.mi_dist.start();
        let m_gidx = self.ixlu.mi_pair2index((x, y));
        assert!(mi_start <= m_gidx);
        assert!(m_gidx < self.mi_dist.end());
        m_gidx - mi_start
    }

    pub fn mi_pair2owner(&self, (x, y): (usize, usize)) -> i32 {
        let midx = self.ixlu.mi_pair2index((x, y));
        self.mi_dist.owner(midx)
    }

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

    pub fn local_lmr_slice_for(&self, v_about: usize) -> &[FloatT] {
        let v_range = self.si_local_range_for(v_about);
        &self.lmr.as_slice().unwrap_or_default()[v_range]
    }
}

pub struct DistLMRMinSum<'a, IntT, FloatT> {
    dist: &'a DistLMR<FloatT>,
    pair_x: Array1<IntT>,
    pair_y: Array1<IntT>,
    minsum: Array1<FloatT>,
    minsum_ind: Array1<bool>,
}

impl<'a, IntT, FloatT> DistLMRMinSum<'a, IntT, FloatT>
where
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    pub fn from_unif_vars(
        cx: &CommIfx,
        dist: &'a DistLMR<FloatT>,
    ) -> Result<Self> {
        assert!(dist.mode == DistMode::VarUniform);
        let var_ints: Vec<IntT> = (0..dist.nvars())
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        //  1. allocate minsum, indicator vec with capacity
        let counts = dist.si2mi_unique_counts();
        let mut offsets: Vec<usize> =
            exc_prefix_sum(counts.clone().into_iter(), 1);
        let size = counts.iter().sum::<usize>();
        let mut px_vec = vec![IntT::zero(); size];
        let mut py_vec = vec![IntT::zero(); size];
        let mut msum_vec = vec![FloatT::zero(); size];
        let mut msum_ind = vec![false; size];
        let mut nctx = 0;
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
                msum_vec[loc] = msum;
                if v_about < v_by {
                    px_vec[loc] = var_ints[v_about];
                    py_vec[loc] = var_ints[v_by];
                    msum_ind[loc] = true;
                } else {
                    px_vec[loc] = var_ints[v_by];
                    py_vec[loc] = var_ints[v_about];
                    msum_ind[loc] = false;
                }
                offsets[mown] += 1;
                nctx += 1;
            }
        }
        gather_debug!(cx.comm(); "SX {} {}", nctx, size);
        debug_assert!(nctx == size);
        debug_assert!(itertools::all(
            zip(px_vec.iter(), py_vec.iter()),
            |(x, y)| *x < *y
        ));

        let recv_counts = all2all_vec(&counts, cx.comm())?;
        let px_vec = all2allv_vec(&px_vec, &counts, &recv_counts, cx.comm())?;
        let pair_x: Array1<IntT> = Array1::from_vec(px_vec);
        let py_vec = all2allv_vec(&py_vec, &counts, &recv_counts, cx.comm())?;
        let pair_y: Array1<IntT> = Array1::from_vec(py_vec);
        let msum_vec = all2allv_vec(&msum_vec, &counts, &recv_counts, cx.comm())?;
        let minsum: Array1<FloatT> = Array1::from_vec(msum_vec);
        let msum_ind = all2allv_vec(&msum_ind, &counts, &recv_counts, cx.comm())?;
        let minsum_ind: Array1<bool> = Array1::from_vec(msum_ind);
        debug_assert!(itertools::all(
            zip(pair_x.iter(), pair_y.iter()),
            |(x, y)| *x < *y
        ));

        Ok(Self {
            dist,
            pair_x,
            pair_y,
            minsum,
            minsum_ind,
        })
    }
}

pub struct PUCDistWorkflow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub args: &'a WorkflowArgs,
}

struct PUCDistWorkFlowHelper<SizeT, IntT, FloatT> {
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> PUCDistWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    fn var_uniform_dist(wf: &PUCDistWorkflow) -> Result<DistLMR<FloatT>> {
        DistLMR::<FloatT>::with_mode::<SizeT, IntT>(
            wf.mpi_ifx,
            &wf.args.misi_data_file,
            DistMode::VarUniform,
        )
    }

    fn var_uniform_lmr_minsum<'a>(
        wf: &PUCDistWorkflow,
        dist: &'a DistLMR<FloatT>,
    ) -> Result<DistLMRMinSum<'a, IntT, FloatT>> {
        DistLMRMinSum::<'a, IntT, FloatT>::from_unif_vars(wf.mpi_ifx, dist)
    }

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
    pub fn run(&self) -> Result<()> {
        type HelperT = PUCDistWorkFlowHelper<i64, i32, f32>;
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        let vu_dist = HelperT::var_uniform_dist(self)?;
        s_timer.info_section("Dist PUC::Build");
        gather_debug!(self.mpi_ifx.comm(); "LMR Distr. {}", vu_dist);
        cond_info!(self.mpi_ifx.is_root(); "Bounds {}", vu_dist.ixlu);
        s_timer.reset();
        let dist_minsum = HelperT::var_uniform_lmr_minsum(self, &vu_dist)?;
        s_timer.info_section("Dist PUC::LMR Minsum");
        gather_debug!(
            self.mpi_ifx.comm();
            "Dist Minsum {} {} {} {} {}",
            vu_dist.mi.len(),
            vu_dist.mi[0],
            dist_minsum.pair_x[0],
            dist_minsum.pair_y[0],
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
