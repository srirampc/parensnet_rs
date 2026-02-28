use anyhow::{Ok, Result};
use hdf5::H5Type;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use ndarray::{Array1, Array2, ArrayView1};
use num::{FromPrimitive, ToPrimitive};
use sope::{
    collective::{all2all_vec, all2allv_vec, allgatherv_full_vec},
    reduction::allreduce_sum,
    timer::SectionTimer,
    util::exc_prefix_sum,
};
use std::{
    collections::HashMap, fmt::Debug, iter::zip, marker::PhantomData, ops::Range,
};

use super::{WorkDistributor, WorkflowArgs};
use crate::{
    anndata::AnnData,
    comm::CommIfx,
    cond_debug, cond_info,
    h5::{io, mpio},
    hist::{HSFloat, bayesian_blocks_bin_edges, histogram_1d, histogram_2d},
    map_with_result_to_tuple,
    mvim::imeasures::{
        lmr_about_x_from_lvji, lmr_about_y_from_lvji, log_jvi_ratio,
        mi_from_ljvi, si_from_ljvi,
    },
    types::{AddFromZero, FromToPrimitive, PNFloat, PNInteger},
    util::{block_owner, block_range, triu_pair_to_index},
};

struct Node<IntT, FloatT> {
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

    fn from_data(c_data: ArrayView1<FloatT>) -> Self
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

struct NodeCollection<SizeT, IntT, FloatT> {
    bin_dim: Array1<IntT>,
    hist_dim: Array1<IntT>,
    bin_start: Array1<SizeT>,
    si_start: Array1<SizeT>,
    hist_start: Array1<SizeT>,
    nsi: SizeT,
    // bins/hist flattened to a histogram
    abins: Array1<FloatT>,
    ahist: Array1<FloatT>,
}

impl<SizeT, IntT, FloatT> NodeCollection<SizeT, IntT, FloatT> {
    fn from_nodes(
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

    fn from_h5(h5_file: &str) -> Result<Self>
    where
        SizeT: H5Type,
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

    fn hist(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        SizeT: ToPrimitive,
        IntT: ToPrimitive,
    {
        let hstart = self.hist_start[idx].to_usize().unwrap();
        let hsize = self.hist_dim[idx].to_usize().unwrap();
        let hend = hstart + hsize;
        self.ahist.slice(ndarray::s![hstart..hend])
    }

    fn bins(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        SizeT: ToPrimitive,
        IntT: ToPrimitive,
    {
        let bstart = self.bin_start[idx].to_usize().unwrap();
        let bsize = self.bin_dim[idx].to_usize().unwrap();
        let bend = bstart + bsize;
        self.abins.slice(ndarray::s![bstart..bend])
    }

    fn len(&self) -> usize {
        self.hist_dim.len()
    }

    fn is_empty(&self) -> bool {
        self.hist_dim.is_empty()
    }
}

struct PairMI<IntT, FloatT> {
    index: usize,
    pair: (IntT, IntT),
    xy_tab: Array2<FloatT>,
    mi: FloatT,
}

struct PairMICollection<IntT, FloatT> {
    index: Vec<usize>,
    dims: (Vec<IntT>, Vec<IntT>),
    xy_tab: Array1<FloatT>,
    mi: Array1<FloatT>,
}

impl<IntT, FloatT> PairMICollection<IntT, FloatT>
where
    IntT: Clone + Default + Debug + Equivalence + FromToPrimitive,
    FloatT: Clone + Default + Debug + Equivalence,
{
    fn from_vec(vdata: &[PairMI<IntT, FloatT>]) -> Self {
        let index = vdata.iter().map(|x| x.index).collect();
        let mi = vdata.iter().map(|x| x.mi.clone()).collect();
        let dims = vdata
            .iter()
            .map(|x| {
                let d = x.xy_tab.shape();
                (
                    IntT::from_usize(d[0]).unwrap(),
                    IntT::from_usize(d[1]).unwrap(),
                )
            })
            .collect::<(Vec<_>, Vec<_>)>();
        let xy_tab = vdata
            .iter()
            .flat_map(|x| x.xy_tab.flatten().to_owned())
            .collect::<Array1<_>>();
        Self {
            index,
            mi,
            dims,
            xy_tab,
        }
    }

    fn distribute(&self, mcx: &CommIfx) -> Result<Self> {
        let npairs: usize = allreduce_sum(&(self.index.len()), mcx.comm());
        let np = mcx.size as usize;
        let (snd_pairs, snd_tabs) = self
            .index
            .iter()
            .zip(zip(self.dims.0.iter(), self.dims.1.iter()))
            .fold(
                (vec![0usize; np], vec![0usize; np]),
                |mut sv, (idx, (dx, dy))| {
                    let p_own = block_owner(*idx, mcx.size, npairs) as usize;
                    sv.0[p_own] += 1;
                    let rdim = dx.to_usize().unwrap() * dy.to_usize().unwrap();
                    sv.1[p_own] += rdim;
                    sv
                },
            );
        let rcv_pairs = all2all_vec(&snd_pairs, mcx.comm())?;
        let rcv_tabs = all2all_vec(&snd_tabs, mcx.comm())?;

        let index =
            all2allv_vec(&self.index[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let dims_x =
            all2allv_vec(&self.dims.0[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let dims_y =
            all2allv_vec(&self.dims.1[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let mi = all2allv_vec(
            self.mi.as_slice().unwrap(),
            &snd_pairs,
            &rcv_pairs,
            mcx.comm(),
        )?;
        let xy_tab = all2allv_vec(
            self.xy_tab.as_slice().unwrap(),
            &snd_tabs,
            &rcv_tabs,
            mcx.comm(),
        )?;

        Ok(Self {
            index,
            dims: (dims_x, dims_y),
            mi: Array1::from_vec(mi),
            xy_tab: Array1::from_vec(xy_tab),
        })
    }
}

struct OrdPairSI<IntT, FloatT> {
    about: IntT,
    by: IntT,
    si: Array1<FloatT>,
    lmr: Array1<FloatT>,
}

struct OrdPairSICollection<IntT, FloatT> {
    nvars: usize,
    nord_pairs: usize,
    n_si: usize,
    about: Vec<IntT>,
    by: Vec<IntT>,
    sizes: Vec<IntT>,
    si: Array1<FloatT>,
    lmr: Array1<FloatT>,
}

impl<IntT, FloatT> OrdPairSICollection<IntT, FloatT>
where
    IntT: Clone + Default + Debug + Equivalence + FromToPrimitive,
    FloatT: Clone + Default + Debug + Equivalence,
{
    fn from_vec(hist_dim: &[IntT], vdata: &[OrdPairSI<IntT, FloatT>]) -> Self {
        let nvars = hist_dim.len();
        let about: Vec<IntT> = vdata.iter().map(|x| x.about.clone()).collect();
        let by: Vec<IntT> = vdata.iter().map(|x| x.by.clone()).collect();
        let sizes: Vec<IntT> = vdata
            .iter()
            .map(|x| IntT::from_usize(x.si.len()))
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let si = vdata
            .iter()
            .flat_map(|x| x.si.to_owned())
            .collect::<Array1<_>>();
        let lmr = vdata
            .iter()
            .flat_map(|x| x.lmr.to_owned())
            .collect::<Array1<_>>();
        let n_si: usize = about.iter().map(|x| x.to_usize().unwrap()).sum();
        Self {
            nvars,
            nord_pairs: nvars * nvars,
            n_si,
            about,
            by,
            sizes,
            si,
            lmr,
        }
    }

    fn distribute(&self, mcx: &CommIfx) -> Result<Self> {
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
        let rcv_pairs = all2all_vec(&snd_pairs, mcx.comm())?;
        let rcv_si = all2all_vec(&snd_si, mcx.comm())?;

        let about =
            all2allv_vec(&self.about[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let by = all2allv_vec(&self.by[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let sizes =
            all2allv_vec(&self.sizes[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let si = all2allv_vec(
            self.si.as_slice().unwrap(),
            &snd_si,
            &rcv_si,
            mcx.comm(),
        )?;
        let lmr = all2allv_vec(
            self.lmr.as_slice().unwrap(),
            &snd_si,
            &rcv_si,
            mcx.comm(),
        )?;
        let n_si: usize = about.iter().map(|x| x.to_usize().unwrap()).sum();

        Ok(Self {
            nvars: self.nvars,
            nord_pairs: self.nord_pairs,
            n_si,
            about,
            by,
            sizes,
            si: Array1::from_vec(si),
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
        assert!(offset == self.si.len());
        assert!(offset == self.lmr.len());
        p_lookup
    }

    fn fill_diag(&mut self, hist_dim: &[IntT], mcx: &CommIfx) {
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
        let si_slice = self.si.as_slice().unwrap();
        let lmr_slice = self.lmr.as_slice().unwrap();

        let mut sizes: Vec<IntT> = vec![IntT::default(); about.len()];
        let mut si: Vec<FloatT> = vec![FloatT::default(); n_si];
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
                    let s_in = &si_slice[fmr.clone()];
                    si[sir.clone()].clone_from_slice(s_in);
                    lmr[sir].clone_from_slice(&lmr_slice[fmr]);
                }
            }
            sizes[i] = h_dim;
            offset += x_dim;
        }

        // TODO::
        self.n_si = n_si;
        self.about = about
            .into_iter()
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        self.by = by
            .into_iter()
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        self.si = Array1::from_vec(si);
        self.lmr = Array1::from_vec(lmr);
        self.sizes = sizes;
    }
}

// aliases to
type NodePair<IntT, FloatT> = (
    OrdPairSI<IntT, FloatT>,
    OrdPairSI<IntT, FloatT>,
    PairMI<IntT, FloatT>,
);

type BatchPairs<IntT, FloatT> = (
    Vec<OrdPairSI<IntT, FloatT>>,
    Vec<OrdPairSI<IntT, FloatT>>,
    Vec<PairMI<IntT, FloatT>>,
);

trait BPTrait<IntT, FloatT> {
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

pub struct MISIWorkFlow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
    pub adata: &'a AnnData,
}

struct MISIWorkFlowHelper<SizeT, IntT, FloatT> {
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

type NodePairCollection<IntT, FloatT> = (
    OrdPairSICollection<IntT, FloatT>,
    PairMICollection<IntT, FloatT>,
);

impl<SizeT, IntT, FloatT> MISIWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    fn construct_nodes(
        wf: &MISIWorkFlow,
        rank: i32,
    ) -> Result<NodeCollection<SizeT, IntT, FloatT>> {
        let columns = wf.wdistr.var_dist[rank as usize].clone();
        let rdata = wf.adata.read_range_data_around::<FloatT>(
            columns.clone(),
            wf.args.nroundup,
        )?;
        let var_data: Vec<Node<IntT, FloatT>> = columns
            .into_iter()
            .enumerate()
            .map(|(i, _cx)| Node::<IntT, FloatT>::from_data(rdata.column(i)))
            .collect();
        log::debug!("At Rank {} Built {} nodes", wf.mpi_ifx.rank, var_data.len());
        let nodes = NodeCollection::<SizeT, IntT, FloatT>::from_nodes(
            &var_data,
            wf.mpi_ifx.comm(),
        )?;
        if log::log_enabled!(log::Level::Debug) {
            wf.mpi_ifx.comm().barrier();
            cond_debug!(wf.mpi_ifx.is_root(); "Collected {} Nodes", nodes.len());
        }
        Ok(nodes)
    }

    fn init_node_pair(
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
        indices: (usize, usize),
        rcdata: (ArrayView1<FloatT>, ArrayView1<FloatT>),
        args: &WorkflowArgs,
    ) -> NodePair<IntT, FloatT> {
        let nobs = FloatT::from_usize(args.nobs).unwrap();
        let tbase = args.tbase.clone();
        let (x_idx, y_idx) = indices;
        let (x_data, y_data) = rcdata;
        let (x_hist, y_hist) = (nodes.hist(x_idx), nodes.hist(y_idx));
        let (x_bins, y_bins) = (nodes.bins(x_idx), nodes.bins(y_idx));
        let xy_tab = histogram_2d(
            x_data,
            x_bins.as_slice().unwrap(),
            y_data,
            y_bins.as_slice().unwrap(),
        );

        let ljvi_ratio =
            log_jvi_ratio(xy_tab.view(), x_hist, y_hist, tbase, nobs);
        let (x_lmr, y_lmr) = (
            lmr_about_x_from_lvji(ljvi_ratio.view(), xy_tab.view(), Some(nobs)),
            lmr_about_y_from_lvji(ljvi_ratio.view(), xy_tab.view(), Some(nobs)),
        );
        let (x_si, y_si) = si_from_ljvi(
            ljvi_ratio.view(),
            xy_tab.view(),
            x_hist.view(),
            y_hist.view(),
        );
        let mi = mi_from_ljvi(ljvi_ratio.view(), xy_tab.view(), Some(nobs));
        let index = triu_pair_to_index(args.nvars, x_idx, y_idx);
        let pair = (
            IntT::from_usize(x_idx).unwrap(),
            IntT::from_usize(y_idx).unwrap(),
        );

        (
            OrdPairSI {
                about: pair.0,
                by: pair.1,
                si: x_si,
                lmr: x_lmr,
            },
            OrdPairSI {
                about: pair.1,
                by: pair.0,
                si: y_si,
                lmr: y_lmr,
            },
            PairMI::<IntT, FloatT> {
                index,
                pair,
                xy_tab,
                mi,
            },
        )
    }

    fn construct_batch_pairs(
        wf: &MISIWorkFlow,
        rank: i32,
        bidx: usize,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<BatchPairs<IntT, FloatT>> {
        let (rows, cols) = wf.wdistr.pairs_2d().batch_range(bidx, rank);
        let (row_data, col_data) = (
            wf.adata.read_range_data_around::<FloatT>(
                rows.clone(),
                wf.args.nroundup,
            )?,
            wf.adata.read_range_data_around::<FloatT>(
                cols.clone(),
                wf.args.nroundup,
            )?,
        );
        let mut batch_pairs =
            BatchPairs::<IntT, FloatT>::new(rows.clone(), cols.clone());
        for (i, rx) in rows.clone().enumerate() {
            let r_data = row_data.column(i);
            for (j, cx) in cols.clone().enumerate() {
                if rx < cx {
                    let c_data = col_data.column(j);
                    batch_pairs.push(Self::init_node_pair(
                        nodes,
                        (rx, cx),
                        (r_data, c_data),
                        wf.args,
                    ));
                }
            }
        }

        Ok(batch_pairs)
    }

    fn construct_node_pairs(
        wf: &MISIWorkFlow,
        rank: i32,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<NodePairCollection<IntT, FloatT>> {
        let nbatches = wf.wdistr.pairs_2d().num_batches();

        let (v_si, v_siy, v_pmi) = (0..nbatches)
            .map(|bidx| Self::construct_batch_pairs(wf, rank, bidx, nodes))
            .collect::<Result<(Vec<_>, Vec<_>, Vec<_>)>>()?;

        // flatten all vec of vec
        let mut v_pmi: Vec<PairMI<IntT, FloatT>> =
            v_pmi.into_iter().flatten().collect();
        v_pmi.sort_by_key(|x| x.pair);

        let mut v_si: Vec<OrdPairSI<IntT, FloatT>> =
            v_si.into_iter().flatten().collect();
        let v_siy: Vec<OrdPairSI<IntT, FloatT>> =
            v_siy.into_iter().flatten().collect();
        v_si.extend(v_siy);
        v_si.sort_by_key(|x| (x.about, x.by));
        log::debug!(
            "At Rank {} :: Built MI {} and SI {}",
            wf.mpi_ifx.rank,
            v_pmi.len(),
            v_si.len()
        );
        if log::log_enabled!(log::Level::Debug) {
            wf.mpi_ifx.comm().barrier();
            let n_mi = allreduce_sum(&(v_pmi.len()), wf.mpi_ifx.comm());
            let n_si = allreduce_sum(&(v_si.len()), wf.mpi_ifx.comm());
            cond_debug!(
                wf.mpi_ifx.is_root(); "Built MI : {} & SI : {} ", n_mi, n_si
            );
        }

        let v_si = OrdPairSICollection::<IntT, FloatT>::from_vec(
            nodes.hist_dim.as_slice().unwrap(),
            &v_si,
        );
        let v_pmi = PairMICollection::<IntT, FloatT>::from_vec(&v_pmi);
        if log::log_enabled!(log::Level::Debug) {
            wf.mpi_ifx.comm().barrier();
            let n_mi = allreduce_sum(&(v_pmi.index.len()), wf.mpi_ifx.comm());
            let n_si = allreduce_sum(&(v_si.about.len()), wf.mpi_ifx.comm());
            cond_debug!(
                wf.mpi_ifx.is_root(); "Collected {} MI & {} SI", n_mi, n_si
            );
        }
        Ok((v_si, v_pmi))
    }

    fn write_nodes_h5(
        wf: &MISIWorkFlow,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<()> {
        let hfptr = io::create_file(&wf.args.misi_data_file)?;
        let data_group = hfptr.create_group("data")?;
        data_group
            .new_attr::<SizeT>()
            .create("nvars")?
            .write_scalar(&(SizeT::from_usize(wf.adata.nvars).unwrap()))?;
        data_group
            .new_attr::<SizeT>()
            .create("nobs")?
            .write_scalar(&(SizeT::from_usize(wf.adata.nobs).unwrap()))?;
        data_group
            .new_attr::<SizeT>()
            .create("npairs")?
            .write_scalar(&(SizeT::from_usize(wf.adata.npairs).unwrap()))?;
        data_group
            .new_attr::<SizeT>()
            .create("nsi")?
            .write_scalar(&nodes.nsi)?;

        io::write_1d(&data_group, "hist_dim", &nodes.hist_dim)?;
        io::write_1d(&data_group, "hist_start", &nodes.hist_start)?;
        io::write_1d(&data_group, "hist", &nodes.ahist)?;

        io::write_1d(&data_group, "bins_dim", &nodes.bin_dim)?;
        io::write_1d(&data_group, "bins_start", &nodes.bin_start)?;
        io::write_1d(&data_group, "bins", &nodes.abins)?;

        io::write_1d(&data_group, "si_start", &nodes.si_start)?;
        Ok(())
    }

    fn write_node_pairs(
        w: &MISIWorkFlow,
        npairs_si: &OrdPairSICollection<IntT, FloatT>,
        npairs_mi: &PairMICollection<IntT, FloatT>,
    ) -> Result<()> {
        let h5_fptr = mpio::open_file_rw(w.mpi_ifx, &w.args.misi_data_file)?;
        let data_group = h5_fptr.group("data")?;
        //TODO::
        mpio::block_write1d(w.mpi_ifx, &data_group, "mi", &npairs_mi.mi)?;
        mpio::block_write1d(w.mpi_ifx, &data_group, "si", &npairs_si.si)?;
        mpio::block_write1d(w.mpi_ifx, &data_group, "lmr", &npairs_si.lmr)?;
        Ok(())
    }
}

impl<'a> MISIWorkFlow<'a> {
    pub fn run(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        cond_info!(self.mpi_ifx.is_root(); "Starting MISIWorkFlow::Run");

        let nodes = HelperT::construct_nodes(self, self.mpi_ifx.rank)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Construct Nodes");
            cond_info!(
                self.mpi_ifx.is_root();
                "Nodes Constructed: {} w. {}", nodes.len(), nodes.bin_dim.len()
            );
            s_timer.reset();
        }

        let (npairs_si, npairs_mi) =
            HelperT::construct_node_pairs(self, self.mpi_ifx.rank, &nodes)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Construct Node Pairs");
            let n_mi =
                allreduce_sum(&(npairs_mi.index.len()), self.mpi_ifx.comm());
            let n_si =
                allreduce_sum(&(npairs_si.about.len()), self.mpi_ifx.comm());
            self.mpi_ifx.comm().barrier();
            cond_info!(
                self.mpi_ifx.is_root(); "MI Size: {} SI Size: {}", n_mi, n_si
            );
            s_timer.reset();
        }

        let npairs_mi = npairs_mi.distribute(self.mpi_ifx)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Node Pairs MI Distribution");
            let n_mi =
                allreduce_sum(&(npairs_mi.index.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root(); "Distributed MI: {} ", n_mi
            );
            s_timer.reset();
        }

        let mut npairs_si = npairs_si.distribute(self.mpi_ifx)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Node Pairs SI Distribution");
            let n_si =
                allreduce_sum(&(npairs_si.about.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root(); "Distributed SI: {}", n_si
            );
            s_timer.reset();
        }

        npairs_si.fill_diag(nodes.hist_dim.as_slice().unwrap(), self.mpi_ifx);
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Serialize Peparation");
            let n_si =
                allreduce_sum(&(npairs_si.about.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root(); "Update SI for Serialization: {}", n_si
            );
            s_timer.reset();
        }

        if self.mpi_ifx.rank == 0 {
            HelperT::write_nodes_h5(self, &nodes)?;
        }
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Write Nodes");
            cond_info!(self.mpi_ifx.is_root(); "Completed Writing Nodes");
            s_timer.reset();
        }

        self.mpi_ifx.comm().barrier();
        HelperT::write_node_pairs(self, &npairs_si, &npairs_mi)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Write Node Pairs");
            cond_info!(self.mpi_ifx.is_root(); "Finished MISIWorkFlow::Run");
        }
        Ok(())
    }

    pub fn run_flat(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        // 1. Flat loading of  all the nodes
        let _nodes =
            NodeCollection::<i64, i32, f32>::from_h5(&self.args.misi_data_file)?;
        // 2. Load ljvi start in distributed manner
        let _jv_start: Array1<i64> = mpio::block_read1d(
            self.mpi_ifx,
            &self.args.misi_data_file,
            "jvi_start",
            None,
        )?;
        // TODO::
        Ok(())
    }
}
