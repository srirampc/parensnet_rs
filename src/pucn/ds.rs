use anyhow::{Ok, Result};
use hdf5::H5Type;
use mpi::traits::{Communicator, Equivalence};
use ndarray::{Array1, Array2, ArrayView1};
use num::{FromPrimitive, ToPrimitive, Zero};
use sope::{
    collective::{all2all_vec, all2allv_vec, allgather_one, allgatherv_full_vec},
    partition::{ArbitDist, Dist, InterleavedDist},
    reduction::allreduce_sum,
    timer::SectionTimer,
    util::exc_prefix_sum,
};
use std::{collections::HashMap, fmt::Debug, iter::zip, ops::Range};

use super::WorkflowArgs;
use crate::{
    comm::CommIfx,
    h5::{io, mpio},
    hist::{HSFloat, bayesian_blocks_bin_edges, histogram_1d},
    map_with_result_to_tuple,
    types::{AddFromZero, FromToPrimitive, PNInteger},
    util::{block_owner, block_range, triu_index_to_pair},
};

pub(super) struct Node<IntT, FloatT> {
    nbins: IntT,
    nhist: IntT,
    bins: Array1<FloatT>,
    hist: Array1<FloatT>,
}

impl<IntT, FloatT> Node<IntT, FloatT> {
    fn new(
        nbins: IntT,
        nhist: IntT,
        bins: Array1<FloatT>,
        hist: Array1<FloatT>,
    ) -> Self {
        Node {
            nbins,
            nhist,
            bins,
            hist,
        }
    }

    pub fn from_data(c_data: ArrayView1<FloatT>) -> Self
    where
        FloatT: 'static + HSFloat,
        IntT: AddFromZero + FromPrimitive + Clone,
    {
        let cbins = bayesian_blocks_bin_edges(c_data);
        let chist =
            histogram_1d::<FloatT, FloatT>(c_data, cbins.as_slice().unwrap());
        Self::new(
            IntT::from_usize(cbins.len()).unwrap(),
            IntT::from_usize(chist.len()).unwrap(),
            cbins,
            chist,
        )
    }
}

pub(super) struct NodeCollection<SizeT, IntT, FloatT> {
    pub bin_dim: Array1<IntT>,
    pub hist_dim: Array1<IntT>,
    pub bin_start: Array1<SizeT>,
    pub si_start: Array1<SizeT>,
    pub hist_start: Array1<SizeT>,
    pub nsi: SizeT,
    // bins/hist flattened to a histogram
    pub abins: Array1<FloatT>,
    pub ahist: Array1<FloatT>,
}

impl<SizeT, IntT, FloatT> NodeCollection<SizeT, IntT, FloatT> {
    pub fn from_nodes(
        v_nodes: &[Node<IntT, FloatT>],
        comm: &dyn Communicator,
    ) -> Result<Self>
    where
        SizeT: 'static + PNInteger + Equivalence,
        IntT: Clone + Debug + Default + ToPrimitive + Equivalence,
        FloatT: Clone + Debug + Default + Equivalence,
    {
        // bin/hist sizes
        let bin_dim: Vec<IntT> =
            v_nodes.iter().map(|x| x.nbins.clone()).collect();
        let bin_dim: Vec<IntT> = allgatherv_full_vec(&bin_dim, comm)?;
        let hist_dim: Vec<IntT> =
            v_nodes.iter().map(|x| x.nhist.clone()).collect();
        let hist_dim: Vec<IntT> = allgatherv_full_vec(&hist_dim, comm)?;

        // histogram and bin boundaries
        let vbins: Vec<FloatT> = v_nodes
            .iter()
            .flat_map(|x| x.bins.iter().cloned())
            .collect();
        let vbins: Vec<FloatT> = allgatherv_full_vec(&vbins, comm)?;

        let vhist: Vec<FloatT> = v_nodes
            .iter()
            .flat_map(|x| x.hist.iter().cloned())
            .collect();
        let vhist: Vec<FloatT> = allgatherv_full_vec(&vhist, comm)?;

        // Starting positions in the flattened arrays
        let hist_starts: Vec<SizeT> = exc_prefix_sum(
            hist_dim
                .iter()
                .map(|x| SizeT::from_i64(x.to_i64().unwrap()).unwrap()),
            SizeT::one(),
        );
        let bin_starts: Vec<SizeT> = exc_prefix_sum(
            bin_dim
                .iter()
                .map(|x| SizeT::from_i64(x.to_i64().unwrap()).unwrap()),
            SizeT::one(),
        );
        let si_start: Vec<i64> = exc_prefix_sum(
            hist_dim.iter().map(|x| x.to_i64().unwrap()),
            bin_dim.len() as i64,
        );
        let si_start: Vec<SizeT> = si_start
            .into_iter()
            .map(|x| SizeT::from_i64(x).unwrap())
            .collect();

        let nsi = hist_dim
            .iter()
            .map(|x| x.to_usize().unwrap())
            .sum::<usize>()
            * hist_dim.len();

        Ok(Self {
            bin_dim: Array1::from_vec(bin_dim),
            hist_dim: Array1::from_vec(hist_dim),
            hist_start: Array1::from_vec(hist_starts),
            bin_start: Array1::from_vec(bin_starts),
            si_start: Array1::from_vec(si_start),
            abins: Array1::from_vec(vbins),
            ahist: Array1::from_vec(vhist),
            nsi: SizeT::from_usize(nsi).unwrap(),
        })
    }

    pub fn from_h5(h5_file: &str, nvars: usize) -> Result<Self>
    where
        SizeT: H5Type + ToPrimitive,
        IntT: H5Type,
        FloatT: H5Type,
    {
        let file = hdf5::File::open(h5_file)?;
        let data_g = file.group("data")?;
        // attributes
        let (_nobs, _nvars, nsi) = map_with_result_to_tuple![
            |x| io::read_scalar_attr::<SizeT>(&data_g, x) ;
            "nobs", "nvars", "nsi"
        ];

        let (hist_start, bin_start, si_start) = map_with_result_to_tuple![
            |x| data_g.dataset(x)?.read_1d::<SizeT>();
           "hist_start", "bins_start", "si_start"
        ];

        let (hist_dim, bin_dim) = map_with_result_to_tuple![
            |x| data_g.dataset(x)?.read_1d::<IntT>();
           "hist_dim", "bins_dim"
        ];

        let (ahist, abins) = map_with_result_to_tuple![
            |x| data_g.dataset(x)?.read_1d::<FloatT>();
           "hist", "bins"
        ];

        // Restrict nvars
        let (hist_start, bin_start, si_start, hist_dim, bin_dim) =
            if nvars > 1 && nvars < _nvars.to_usize().unwrap() {
                (
                    hist_start.slice_move(ndarray::s![..nvars]),
                    bin_start.slice_move(ndarray::s![..nvars]),
                    si_start.slice_move(ndarray::s![..nvars]),
                    hist_dim.slice_move(ndarray::s![..nvars]),
                    bin_dim.slice_move(ndarray::s![..nvars]),
                )
            } else {
                (hist_start, bin_start, si_start, hist_dim, bin_dim)
            };

        Ok(Self {
            hist_dim,
            hist_start,
            bin_dim,
            bin_start,
            si_start,
            nsi,
            ahist,
            abins,
        })
    }

    pub fn hist(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        SizeT: ToPrimitive,
        IntT: ToPrimitive,
    {
        let hstart = self.hist_start[idx].to_usize().unwrap();
        let hsize = self.hist_dim[idx].to_usize().unwrap();
        let hend = hstart + hsize;
        self.ahist.slice(ndarray::s![hstart..hend])
    }

    pub fn bins(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        SizeT: ToPrimitive,
        IntT: ToPrimitive,
    {
        let bstart = self.bin_start[idx].to_usize().unwrap();
        let bsize = self.bin_dim[idx].to_usize().unwrap();
        let bend = bstart + bsize;
        self.abins.slice(ndarray::s![bstart..bend])
    }

    pub fn bin_dim_ref(&self) -> &Array1<IntT> {
        &self.bin_dim
    }

    pub fn len(&self) -> usize {
        self.hist_dim.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hist_dim.is_empty()
    }
}

pub(super) struct PairMI<IntT, FloatT> {
    pub index: usize,
    pub pair: Option<(IntT, IntT)>,
    pub xy_tab: Option<Array2<FloatT>>,
    pub mi: FloatT,
}

pub(super) struct PairMICollection<IntT, FloatT> {
    pub index: Vec<usize>,
    pub dims: Option<(Vec<IntT>, Vec<IntT>)>,
    pub xy_tab: Option<Array1<FloatT>>,
    pub mi: Array1<FloatT>,
}

impl<IntT, FloatT> PairMICollection<IntT, FloatT>
where
    IntT: Clone + Default + Debug + Equivalence + FromToPrimitive + Zero,
    FloatT: Clone + Default + Debug + H5Type + Equivalence + Zero,
{
    pub fn from_h5(
        cx: &CommIfx,
        args: &WorkflowArgs,
        h5f: &str,
        hist_dim: &[IntT],
    ) -> Result<Self> {
        assert!(args.npairs > 0);
        let mi_dist = InterleavedDist::new(args.npairs, cx.size, cx.rank);
        let mi = mpio::block_read1d(cx, h5f, "data/mi", Some(&mi_dist))?;
        let index = mi_dist.range().collect();
        let dims: (Vec<IntT>, Vec<IntT>) = mi_dist
            .range()
            .map(|x| {
                let (i, j): (usize, usize) = triu_index_to_pair(args.nvars, x);
                (hist_dim[i].clone(), hist_dim[j].clone())
            })
            .collect();
        let size: usize = zip(dims.0.iter(), dims.1.iter())
            .map(|(x, y)| (*x).to_usize().unwrap() * (*y).to_usize().unwrap())
            .sum();
        let sizes: Vec<usize> = allgather_one(&size, cx.comm())?;
        let h_dist =
            ArbitDist::new(sizes.iter().sum::<usize>(), cx.size, cx.rank, sizes);
        let xy_tab =
            mpio::block_read1d(cx, h5f, "data/pair_hist", Some(&h_dist))?;
        Ok(Self {
            index,
            dims: Some(dims),
            xy_tab: Some(xy_tab),
            mi,
        })
    }

    pub fn from_vec(vdata: &[PairMI<IntT, FloatT>]) -> Self {
        let index = vdata.iter().map(|x| x.index).collect();
        let mi = vdata.iter().map(|x| x.mi.clone()).collect();
        let tab_flag = itertools::all(vdata.iter(), |x| x.xy_tab.is_some());
        if !tab_flag {
            return Self {
                index,
                mi,
                xy_tab: None,
                dims: None,
            };
        }
        let dims = vdata
            .iter()
            .map(|x| {
                if let Some(xyt) = x.xy_tab.as_ref() {
                    let d = xyt.shape();
                    (
                        IntT::from_usize(d[0]).unwrap(),
                        IntT::from_usize(d[1]).unwrap(),
                    )
                } else {
                    (IntT::zero(), IntT::zero())
                }
            })
            .collect::<(Vec<_>, Vec<_>)>();
        let xy_tab = vdata
            .iter()
            .flat_map(|x| {
                if let Some(xyt) = x.xy_tab.as_ref() {
                    xyt.flatten().to_owned()
                } else {
                    Array1::zeros(1)
                }
            })
            .collect::<Array1<_>>();
        Self {
            index,
            mi,
            dims: Some(dims),
            xy_tab: Some(xy_tab),
        }
    }

    pub fn distribute(&self, mcx: &CommIfx, hist_dim: &[IntT]) -> Result<Self> {
        let s_timer = SectionTimer::from_comm(mcx.comm(), ",");
        let npairs: usize = allreduce_sum(&(self.index.len()), mcx.comm());
        let np = mcx.size as usize;
        let (snd_pairs, snd_tabs) = self
            .index
            .iter()
            //.zip(zip(self.dims.0.iter(), self.dims.1.iter()))
            .fold((vec![0usize; np], vec![0usize; np]), |mut sv, idx| {
                let p_own = block_owner(*idx, mcx.size, npairs) as usize;
                sv.0[p_own] += 1;
                let (ix, iy): (usize, usize) =
                    triu_index_to_pair(hist_dim.len(), *idx);
                let dx: IntT = hist_dim[ix].clone();
                let dy: IntT = hist_dim[iy].clone();
                let rdim = dx.to_usize().unwrap() * dy.to_usize().unwrap();
                sv.1[p_own] += rdim;
                sv
            });
        s_timer.info_section("PairMICollection::distribute::Preparation");
        s_timer.reset();
        let rcv_pairs = all2all_vec(&snd_pairs, mcx.comm())?;
        let rcv_tabs = all2all_vec(&snd_tabs, mcx.comm())?;

        let index =
            all2allv_vec(&self.index[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let mi = all2allv_vec(
            self.mi.as_slice().unwrap(),
            &snd_pairs,
            &rcv_pairs,
            mcx.comm(),
        )?;

        let dims = if let Some(dims) = self.dims.as_ref() {
            Some((
                all2allv_vec(&dims.0[..], &snd_pairs, &rcv_pairs, mcx.comm())?,
                all2allv_vec(&dims.1[..], &snd_pairs, &rcv_pairs, mcx.comm())?,
            ))
        } else {
            None
        };

        let xy_tab = if let Some(xy_tab) = self.xy_tab.as_ref() {
            let xy_tab = all2allv_vec(
                xy_tab.as_slice().unwrap(),
                &snd_tabs,
                &rcv_tabs,
                mcx.comm(),
            )?;
            Some(Array1::from_vec(xy_tab))
        } else {
            None
        };
        s_timer.info_section("PairMICollection::distribute::All2All");

        Ok(Self {
            index,
            dims,
            mi: Array1::from_vec(mi),
            xy_tab,
        })
    }
}

pub(super) struct OrdPairSI<IntT, FloatT> {
    pub about: IntT,
    pub by: IntT,
    pub si: Option<Array1<FloatT>>,
    pub lmr: Array1<FloatT>,
}

pub(super) struct OrdPairSICollection<IntT, FloatT> {
    pub nvars: usize,
    pub nord_pairs: usize,
    pub about: Vec<IntT>,
    pub by: Vec<IntT>,
    pub sizes: Vec<IntT>,
    pub si: Option<Array1<FloatT>>,
    pub lmr: Array1<FloatT>,
}

impl<IntT, FloatT> OrdPairSICollection<IntT, FloatT>
where
    IntT: Clone + Default + Debug + Equivalence + FromToPrimitive,
    FloatT: Clone + Default + Debug + Equivalence + Zero,
{
    pub fn from_vec(
        nvars: usize,
        vdata: &[OrdPairSI<IntT, FloatT>],
    ) -> Self {
        let about: Vec<IntT> = vdata.iter().map(|x| x.about.clone()).collect();
        let by: Vec<IntT> = vdata.iter().map(|x| x.by.clone()).collect();
        let sizes: Vec<IntT> = vdata
            .iter()
            .map(|x| IntT::from_usize(x.lmr.len()))
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let si_flag = itertools::all(vdata.iter(), |x| x.si.is_some());
        let si: Option<Array1<FloatT>> = if si_flag {
            let si = vdata
                .iter()
                .flat_map(|x| {
                    if let Some(si) = x.si.as_ref() {
                        si.to_owned()
                    } else {
                        Array1::zeros(about.len())
                    }
                })
                .collect::<Array1<_>>();
            Some(si)
        } else {
            None
        };
        let lmr = vdata
            .iter()
            .flat_map(|x| x.lmr.to_owned())
            .collect::<Array1<_>>();

        Self {
            nvars,
            nord_pairs: nvars * nvars,
            about,
            by,
            sizes,
            si,
            lmr,
        }
    }

    pub fn distribute(&self, mcx: &CommIfx) -> Result<Self> {
        let s_timer = SectionTimer::from_comm(mcx.comm(), ",");
        let np = mcx.size as usize;
        let (snd_pairs, snd_si) = self
            .sizes
            .iter()
            .zip(zip(self.about.iter(), self.by.iter()))
            .fold(
                (vec![0usize; np], vec![0usize; np]),
                |mut sv, (sz, (a, b))| {
                    let (ua, ub) = (a.to_usize().unwrap(), b.to_usize().unwrap());
                    // NOTE: using number of ordered pairs.
                    let idx = ua * self.nvars + ub;
                    let p_own =
                        block_owner(idx, mcx.size, self.nord_pairs) as usize;
                    sv.0[p_own] += 1;
                    sv.1[p_own] += sz.to_usize().unwrap();
                    sv
                },
            );
        s_timer.info_section("OrdPairSICollection::distribute::Preparation");
        s_timer.reset();
        let rcv_pairs = all2all_vec(&snd_pairs, mcx.comm())?;
        let rcv_si = all2all_vec(&snd_si, mcx.comm())?;

        let about =
            all2allv_vec(&self.about[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let by = all2allv_vec(&self.by[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let sizes =
            all2allv_vec(&self.sizes[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let si = if let Some(lsi) = self.si.as_ref() {
            Some(Array1::from_vec(all2allv_vec(
                lsi.as_slice().unwrap(),
                &snd_si,
                &rcv_si,
                mcx.comm(),
            )?))
        } else {
            None
        };
        let lmr = all2allv_vec(
            self.lmr.as_slice().unwrap(),
            &snd_si,
            &rcv_si,
            mcx.comm(),
        )?;
        s_timer.info_section("OrdPairSICollection::distribute::All2All");

        Ok(Self {
            nvars: self.nvars,
            nord_pairs: self.nord_pairs,
            about,
            by,
            sizes,
            si,
            lmr: Array1::from_vec(lmr),
        })
    }

    fn pairs_lookup(&self) -> HashMap<usize, (usize, usize)> {
        let mut p_lookup =
            HashMap::<usize, (usize, usize)>::with_capacity(self.about.len());
        // Should this be pairs ?
        let mut offset = 0;
        for (i, (a, b)) in zip(self.about.iter(), self.by.iter()).enumerate() {
            let tgt_index =
                a.to_usize().unwrap() * self.nvars + b.to_usize().unwrap();
            p_lookup.insert(tgt_index, (i, offset));
            offset += self.sizes[i].to_usize().unwrap();
        }
        assert!(self.si.as_ref().is_none_or(|x| offset == x.len()));
        assert!(offset == self.lmr.len());
        p_lookup
    }

    pub fn fill_diag(&mut self, hist_dim: &[IntT], mcx: &CommIfx) {
        //  Initialize array with the allocated ordered pairs
        let brg = block_range(mcx.rank, mcx.size, self.nord_pairs);
        let about = brg.clone().map(|idx| idx / self.nvars).collect::<Vec<_>>();
        let by = brg.clone().map(|idx| idx % self.nvars).collect::<Vec<_>>();
        let n_si: usize = about
            .iter()
            .map(|x| hist_dim[x.to_usize().unwrap()].to_usize().unwrap())
            .sum();

        // Pairs Lookup
        let lookup = self.pairs_lookup();
        let si_slice = self.si.as_ref().map(|x| x.as_slice().unwrap());
        let lmr_slice = self.lmr.as_slice().unwrap();

        let mut sizes: Vec<IntT> = vec![IntT::default(); about.len()];
        let mut si: Option<Vec<FloatT>> =
            self.si.as_ref().map(|_| vec![FloatT::default(); n_si]);
        let mut lmr: Vec<FloatT> = vec![FloatT::default(); n_si];
        let mut offset: usize = 0;
        for (i, (x, y)) in zip(about.iter(), by.iter()).enumerate() {
            let h_dim = hist_dim[x.to_usize().unwrap()].clone();
            let x_dim = h_dim.to_usize().unwrap();
            let sir = offset..(offset + x_dim);
            if x != y {
                let op_idx = x * self.nvars + y;
                if let Some(vx) = lookup.get(&op_idx) {
                    let fmr = vx.1..(vx.1 + x_dim);
                    lmr[sir.clone()].clone_from_slice(&lmr_slice[fmr.clone()]);
                    if let (Some(si_slice), Some(si)) = (si_slice, si.as_mut()) {
                        si[sir].clone_from_slice(&si_slice[fmr]);
                    }
                }
            }
            sizes[i] = h_dim;
            offset += x_dim;
        }

        self.about = about
            .into_iter()
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        self.by = by
            .into_iter()
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        self.si = si.map(|x| Array1::from_vec(x));
        self.lmr = Array1::from_vec(lmr);
        self.sizes = sizes;
    }
}

// aliases to
pub(super) type NodePair<IntT, FloatT> = (
    OrdPairSI<IntT, FloatT>,
    OrdPairSI<IntT, FloatT>,
    PairMI<IntT, FloatT>,
);

pub(super) type BatchPairs<IntT, FloatT> = (
    Vec<OrdPairSI<IntT, FloatT>>,
    Vec<OrdPairSI<IntT, FloatT>>,
    Vec<PairMI<IntT, FloatT>>,
);

pub(super) trait BPTrait<IntT, FloatT> {
    fn new(rows: Range<usize>, cols: Range<usize>) -> Self;
    fn push(&mut self, node_pair: NodePair<IntT, FloatT>);
}

impl<IntT, FloatT> BPTrait<IntT, FloatT> for BatchPairs<IntT, FloatT> {
    fn new(rows: Range<usize>, cols: Range<usize>) -> Self {
        let npairs: usize = rows
            .map(|row| cols.clone().filter(|col| row < *col).sum::<usize>())
            .sum();
        (
            Vec::with_capacity(npairs),
            Vec::with_capacity(npairs),
            Vec::with_capacity(npairs),
        )
    }

    fn push(&mut self, node_pair: NodePair<IntT, FloatT>) {
        let (s0, s1, m) = node_pair;
        self.0.push(s0);
        self.1.push(s1);
        self.2.push(m);
    }
}

pub(super) type NodePairCollection<IntT, FloatT> = (
    OrdPairSICollection<IntT, FloatT>,
    PairMICollection<IntT, FloatT>,
);
