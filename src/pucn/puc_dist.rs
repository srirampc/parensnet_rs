use super::WorkflowArgs;
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
use mpi::traits::Equivalence;
use ndarray::Array1;
use sope::{
    collective::{all2all_vec, all2allv_vec, allgather_one},
    gather_debug, gather_info,
    partition::{ArbitDist, Dist, InterleavedDist},
    timer::SectionTimer,
};
use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Div, Range},
    rc::Rc,
};

// Index lookups for SI an LMR arrays :
//   Mapping between (about, by) pair and flat array index
//   Next functions for iterations
pub struct SILMRIndexLU {
    si_start: Vec<usize>,
    hist_dim: Vec<usize>,
    nvars: usize,
    nsi: usize,
    npairs: usize,
}

impl Display for SILMRIndexLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{nvars : {}, nsi: {}, npairs: {}, starts: {}, hist: {}}}",
            self.nvars,
            self.nsi,
            self.npairs,
            self.si_start.len(),
            self.hist_dim.len(),
        )
    }
}

impl SILMRIndexLU {
    pub fn new<SizeT, IntT>(args: &WorkflowArgs) -> Result<Self>
    where
        SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
        IntT: PNInteger + H5Type + Default + Equivalence,
    {
        let fptr = File::open(&args.misi_data_file)?;
        let data_g = fptr.group("data")?;
        let hist_dim: Vec<usize> = data_g
            .dataset("hist_dim")?
            .read_1d::<IntT>()?
            .map(|x| x.to_usize().unwrap())
            .into_iter()
            .collect();
        let si_start: Vec<usize> = data_g
            .dataset("si_start")?
            .read_1d::<SizeT>()?
            .map(|x| x.to_usize().unwrap())
            .into_iter()
            .collect();
        let (nvars, npairs, nsi) = map_with_result_to_tuple![
            |x| io::read_scalar_attr::<SizeT>(&data_g, x) ;
            "nvars", "npairs", "nsi"
        ];
        Ok(Self {
            si_start,
            hist_dim,
            nvars: nvars.to_usize().unwrap(),
            nsi: nsi.to_usize().unwrap(),
            npairs: npairs.to_usize().unwrap(),
        })
    }

    fn start_idx(&self, (v_about, v_by): (usize, usize)) -> usize {
        self.si_start[v_about] + v_by * self.hist_dim[v_about]
    }

    fn end_idx(&self, (v_about, v_by): (usize, usize)) -> usize {
        self.si_start[v_about] + (v_by + 1) * self.hist_dim[v_about]
    }

    fn var_bounds(&self, v_about: usize) -> Range<usize> {
        self.start_idx((v_about, 0))..self.end_idx((v_about, self.nvars - 1))
    }

    fn about(&self, idx: usize) -> usize {
        let var = self.si_start.partition_point(|&x| x <= idx);
        if var == 0 {
            0
        } else if var >= self.si_start.len() {
            self.si_start.len() - 1
        } else {
            assert!(idx >= self.si_start[var - 1]);
            var - 1
        }
    }

    fn by(&self, idx: usize, about: usize) -> usize {
        assert!(about < self.si_start.len());
        assert!(idx >= self.si_start[about]);
        let (abt_start, abt_dim) = (self.si_start[about], self.hist_dim[about]);
        let si_surplus = idx - abt_start;
        // TODO:: should this be div? or div_ceil?
        si_surplus.div(abt_dim).min(self.si_start.len() - 1)
    }

    fn index2pair(&self, idx: usize) -> (usize, usize) {
        let sabt = self.about(idx);
        (sabt, self.by(idx, sabt))
    }

    fn next_to(&self, (about, by): (usize, usize)) -> (usize, usize) {
        if by == self.nvars - 1 {
            (about + 1, 0)
        } else {
            (about, by + 1)
        }
    }
}

#[derive(PartialEq)]
pub enum DistMode {
    LMRUniform,
    VarUniform,
}

// Distributed for MI and SI
//  - Local starting and ending pairs
//  - Distribution scheme
//  - Distributed si, lmr and mi
pub struct LMRDist<FloatT> {
    ixlu: Rc<SILMRIndexLU>,
    si_begin: (usize, usize), // begin about, by
    si_end: (usize, usize),   // end about, by
    mi_begin: (usize, usize), // begin i, j
    mi_end: (usize, usize),   // end i, j
    si_dist: ArbitDist,
    mi_dist: InterleavedDist,
    var_dist: InterleavedDist,
    si: Array1<FloatT>,
    lmr: Array1<FloatT>,
    mi: Array1<FloatT>,
    mode: DistMode,
}

impl<FloatT> Display for LMRDist<FloatT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        write!(
            f,
            "{{si_bounds: {:?}, si_dist: {:?}, si: {}}}, ",
            (self.si_begin, self.si_end),
            self.si_dist.range(),
            self.si.len(),
        )?;
        write!(
            f,
            "{{mi_bounds: {:?},  mi_dist: {:?}, mi: {}}}",
            (self.mi_begin, self.mi_end),
            self.mi_dist.range(),
            self.mi.len(),
        )?;
        write!(f, "}}")
    }
}

impl<FloatT> LMRDist<FloatT>
where
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    pub fn with_lmr_uniform(
        ixlu: Rc<SILMRIndexLU>,
        cx: &CommIfx,
        h5fn: &str,
    ) -> Result<Self> {
        let si_range = InterleavedDist::new(ixlu.nsi, cx.size, cx.rank).range();
        let mut si_begin = ixlu.index2pair(si_range.start);
        let si_end = ixlu.index2pair(si_range.end);
        if cx.rank > 0 {
            si_begin = ixlu.next_to(si_begin)
        }
        let idx_begin = ixlu.start_idx(si_begin);
        let idx_end = ixlu.end_idx(si_end);

        //
        let sizes = allgather_one(&(idx_end - idx_begin), cx.comm())?;
        let mdst = InterleavedDist::new(ixlu.npairs, cx.size, cx.rank);
        let sdst = ArbitDist::new(ixlu.nsi, cx.size, cx.rank, sizes);
        let mi = mpio::block_read1d(cx, h5fn, "data/mi", Some(&mdst))?;
        let si = mpio::block_read1d(cx, h5fn, "data/si", Some(&sdst))?;
        let lmr = mpio::block_read1d(cx, h5fn, "data/lmr", Some(&sdst))?;

        Ok(Self {
            ixlu: ixlu.clone(),
            si_begin,
            si_end,
            si_dist: sdst,
            si,
            lmr,
            mi_begin: triu_index_to_pair(ixlu.nvars, mdst.start()),
            mi_end: triu_index_to_pair(ixlu.nvars, mdst.end() - 1),
            mi_dist: mdst,
            mi,
            var_dist: InterleavedDist::new(ixlu.nvars, cx.size, cx.rank),
            mode: DistMode::LMRUniform,
        })
    }

    pub fn with_var_uniform(
        ixlu: Rc<SILMRIndexLU>,
        cx: &CommIfx,
        h5fn: &str,
    ) -> Result<Self> {
        let var_dist = InterleavedDist::new(ixlu.nvars, cx.size, cx.rank);
        let nv_range = var_dist.range();
        let si_begin = (nv_range.start, 0);
        let si_end = (nv_range.end - 1, ixlu.nvars - 1);
        let idx_begin = ixlu.start_idx(si_begin);
        let idx_end = ixlu.end_idx(si_end);
        //
        let sizes = allgather_one(&(idx_end - idx_begin), cx.comm())?;
        let mdst = InterleavedDist::new(ixlu.npairs, cx.size, cx.rank);
        let sdst = ArbitDist::new(ixlu.nsi, cx.size, cx.rank, sizes);
        let mi = mpio::block_read1d(cx, h5fn, "data/mi", Some(&mdst))?;
        let si = mpio::block_read1d(cx, h5fn, "data/si", Some(&sdst))?;
        let lmr = mpio::block_read1d(cx, h5fn, "data/lmr", Some(&sdst))?;

        Ok(Self {
            ixlu: ixlu.clone(),
            si_begin,
            si_end,
            si_dist: sdst,
            si,
            lmr,
            mi_begin: triu_index_to_pair(ixlu.nvars, mdst.start()),
            mi_end: triu_index_to_pair(ixlu.nvars, mdst.end() - 1),
            mi_dist: mdst,
            mi,
            var_dist,
            mode: DistMode::VarUniform,
        })
    }

    pub fn var_local_range(&self, v_about: usize) -> Range<usize> {
        match self.mode {
            DistMode::VarUniform => {
                let sidx = self.si_dist.start();
                let v_gidx = self.ixlu.start_idx((v_about, 0));
                assert!(sidx <= v_gidx);
                assert!(v_gidx < self.si_dist.end());
                let v_start_idx = v_gidx - sidx;
                let v_size = self.ixlu.hist_dim[v_about] * self.ixlu.nvars;
                let v_end_idx = v_start_idx + v_size;
                v_start_idx..v_end_idx
            }
            DistMode::LMRUniform => {
                // TODO:: Need to sort, block segment prefix sum
                todo!("For non over lapping")
            }
        }
    }

    pub fn mi_pair2owner(&self, (x, y): (usize, usize)) -> i32 {
        let (x, y) = if x < y { (x, y) } else { (y, x) };
        let midx = triu_pair_to_index(self.ixlu.nvars, x, y);
        self.mi_dist.owner(midx)
    }

    pub fn si_pair_counts(&self) -> Vec<usize> {
        match self.mode {
            DistMode::VarUniform => {
                let mut pvec = vec![0; self.si_dist.comm_size() as usize];
                for sidx in self.si_dist.range() {
                    let (x, y) = self.ixlu.index2pair(sidx);
                    if x == y {
                        continue;
                    }
                    let mown = self.mi_pair2owner((x, y)) as usize;
                    pvec[mown] += 1;
                }
                pvec
            }
            DistMode::LMRUniform => {
                // TODO:: probably the same, I think
                todo!("For non over lapping")
            }
        }
    }

    pub fn var_lmr_slice(&self, v_about: usize) -> &[FloatT] {
        let v_range = self.var_local_range(v_about);
        &self.lmr.as_slice().unwrap_or_default()[v_range]
    }
}

pub struct LMRSADist<'a, IntT, FloatT> {
    dist: &'a LMRDist<FloatT>,
    pair_x: Array1<IntT>,
    pair_y: Array1<IntT>,
    minsum: Array1<FloatT>,
    minsum_ind: Array1<bool>,
}

impl<'a, IntT, FloatT> LMRSADist<'a, IntT, FloatT>
where
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    pub fn from_unif_vars(
        cx: &CommIfx,
        dist: &'a LMRDist<FloatT>,
    ) -> Result<Self> {
        assert!(dist.mode == DistMode::VarUniform);
        let idxlu = &dist.ixlu;
        let var_ints: Vec<IntT> = (0..idxlu.nvars)
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        //  1. allocate minsum, indicator vec with capacity
        let counts = dist.si_pair_counts();
        let mut offsets: Vec<usize> =
            exc_prefix_sum(counts.clone().into_iter(), 1);
        let size = counts.iter().sum::<usize>();
        let mut px_vec = vec![IntT::zero(); size];
        let mut py_vec = vec![IntT::zero(); size];
        let mut msum_vec = vec![FloatT::zero(); size];
        let mut msum_ind = vec![false; size];
        for v_about in dist.var_dist.range() {
            // 1. lmr slice corresponding to variable vi
            let lslice = dist.var_lmr_slice(v_about);
            // 2. Build LMRSA
            let lmr_sa = LMRSA::<IntT, FloatT>::from_lmr_slice(
                IntT::from_usize(idxlu.hist_dim[v_about]).unwrap_or_default(),
                idxlu.nvars,
                lslice,
            )?;
            // 3. Minsum values and push into arrays
            for v_by in 0..idxlu.nvars {
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
            }
        }
        let recv_counts = all2all_vec(&counts, cx.comm())?;
        let px_vec = all2allv_vec(&px_vec, &counts, &recv_counts, cx.comm())?;
        let pair_x: Array1<IntT> = Array1::from_vec(px_vec);
        let py_vec = all2allv_vec(&py_vec, &counts, &recv_counts, cx.comm())?;
        let pair_y: Array1<IntT> = Array1::from_vec(py_vec);
        let msum_vec = all2allv_vec(&msum_vec, &counts, &recv_counts, cx.comm())?;
        let minsum: Array1<FloatT> = Array1::from_vec(msum_vec);
        let msum_ind = all2allv_vec(&msum_ind, &counts, &recv_counts, cx.comm())?;
        let minsum_ind: Array1<bool> = Array1::from_vec(msum_ind);

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

impl<'a, SizeT, IntT, FloatT> PUCDistWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    fn dist_for(wf: &PUCDistWorkflow) -> Result<LMRDist<FloatT>> {
        let l_bounds = Rc::new(SILMRIndexLU::new::<SizeT, IntT>(wf.args)?);
        let l_dist = LMRDist::with_var_uniform(
            l_bounds,
            wf.mpi_ifx,
            &wf.args.misi_data_file,
        )?;

        Ok(l_dist)
    }
}

impl<'a> PUCDistWorkflow<'a> {
    pub fn run(&self) -> Result<()> {
        type HelperT = PUCDistWorkFlowHelper<i64, i32, f32>;
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        let l_dist = HelperT::dist_for(self)?;
        let l_bounds = &l_dist.ixlu;
        s_timer.info_section("Dist Build");
        gather_info!(self.mpi_ifx.comm(); "D {}", l_dist);
        cond_info!(self.mpi_ifx.is_root(); "Bounds {}", l_bounds);
        s_timer.reset();
        Ok(())
    }
}
