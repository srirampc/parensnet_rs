use anyhow::{Ok, Result};
use hdf5::H5Type;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use ndarray::{Array1, Array2, ArrayView1};
use num::{FromPrimitive, ToPrimitive};
use sope::{
    collective::{all2all_vec, all2allv_vec, allgatherv_full_vec},
    cond_info,
    reduction::allreduce_sum,
    util::exc_prefix_sum,
};
use std::{fmt::Debug, iter::zip, marker::PhantomData};

use super::{WorkDistributor, WorkflowArgs};
use crate::{
    anndata::AnnData,
    comm::CommIfx,
    h5::{io, mpio},
    hist::{HSFloat, bayesian_blocks_bin_edges, histogram_1d, histogram_2d},
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
        let bin_sizes: Vec<IntT> =
            v_nodes.iter().map(|x| x.nbins.clone()).collect();
        let bin_sizes: Vec<IntT> = allgatherv_full_vec(&bin_sizes, comm)?;
        let hist_sizes: Vec<IntT> =
            v_nodes.iter().map(|x| x.nhist.clone()).collect();
        let hist_sizes: Vec<IntT> = allgatherv_full_vec(&hist_sizes, comm)?;

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
            hist_sizes
                .iter()
                .map(|x| SizeT::from_i64(x.to_i64().unwrap()).unwrap()),
            SizeT::one(),
        );
        let bin_starts: Vec<SizeT> = exc_prefix_sum(
            bin_sizes
                .iter()
                .map(|x| SizeT::from_i64(x.to_i64().unwrap()).unwrap()),
            SizeT::one(),
        );

        let nsi = hist_sizes
            .iter()
            .map(|x| x.to_usize().unwrap())
            .sum::<usize>()
            * hist_sizes.len();

        Ok(Self {
            bin_dim: Array1::from_vec(bin_sizes),
            hist_dim: Array1::from_vec(hist_sizes),
            hist_start: Array1::from_vec(hist_starts),
            bin_start: Array1::from_vec(bin_starts),
            abins: Array1::from_vec(vbins),
            ahist: Array1::from_vec(vhist),
            nsi: SizeT::from_usize(nsi).unwrap(),
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

struct OrdPairSICollection<SizeT, IntT, FloatT> {
    nvars: usize,
    nord_pairs: usize,
    si_start: Array1<SizeT>,
    about: Vec<IntT>,
    by: Vec<IntT>,
    sizes: Vec<IntT>,
    si: Array1<FloatT>,
    lmr: Array1<FloatT>,
}

impl<SizeT, IntT, FloatT> OrdPairSICollection<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger,
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
        let si_start: Vec<i64> = exc_prefix_sum(
            hist_dim.iter().map(|x| x.to_i64().unwrap()),
            nvars as i64,
        );
        let si_start: Vec<SizeT> = si_start
            .into_iter()
            .map(|x| SizeT::from_i64(x).unwrap())
            .collect();

        Self {
            nvars,
            si_start: Array1::from_vec(si_start),
            nord_pairs: nvars * nvars,
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

        Ok(Self {
            nvars: self.nvars,
            nord_pairs: self.nord_pairs,
            si_start: self.si_start.clone(),
            about,
            by,
            sizes,
            si: Array1::from_vec(si),
            lmr: Array1::from_vec(lmr),
        })
    }

    fn prepare_serialize(&mut self, mcx: &CommIfx) -> Result<()> {
        let _prange = block_range(mcx.rank, mcx.size, self.nord_pairs);

        //let ((s_x, s_y), (t_x, t_y)) = (
        //    (prange.start / self.nvars, prange.start % self.nvars),
        //    (prange.end / self.nvars, prange.end % self.nvars),
        //);

        // TODO::
        //  1. Initialize array with the allocated pairs
        todo!("Implement Serialization");
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

pub struct MISIWorkFlow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
    pub adata: &'a AnnData,
}

struct MISIWorkFlowHelper<SizeT, IntT, FloatT> {
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

type NodePairCollection<SizeT, IntT, FloatT> = (
    OrdPairSICollection<SizeT, IntT, FloatT>,
    PairMICollection<IntT, FloatT>,
);

impl<SizeT, IntT, FloatT> MISIWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + Default + Equivalence,
    IntT: PNInteger + Default + Equivalence,
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
        log::info!("At Rank {} Built {} nodes", wf.mpi_ifx.rank, var_data.len());
        let nodes = NodeCollection::<SizeT, IntT, FloatT>::from_nodes(
            &var_data,
            wf.mpi_ifx.comm(),
        )?;
        if log::log_enabled!(log::Level::Info) {
            wf.mpi_ifx.comm().barrier();
            cond_info!(wf.mpi_ifx.is_root(); "Built {} Nodes", nodes.len());
        }
        Ok(nodes)
    }

    fn init_node_pairs(
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
        let index = triu_pair_to_index(args.npairs, x_idx, y_idx);
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
        let r_col_data = &col_data;

        let node_pairs = rows
            .clone()
            .enumerate()
            .flat_map(|(i, rx)| {
                let r_data = row_data.column(i);
                cols.clone().enumerate().flat_map(move |(j, cx)| {
                    if i < j {
                        let c_data = r_col_data.column(j);
                        let npair = Self::init_node_pairs(
                            nodes,
                            (rx, cx),
                            (r_data, c_data),
                            wf.args,
                        );
                        Some(npair)
                    } else {
                        None
                    }
                })
            })
            .collect::<(Vec<_>, Vec<_>, Vec<_>)>();
        Ok(node_pairs)
    }

    fn construct_node_pairs(
        wf: &MISIWorkFlow,
        rank: i32,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<NodePairCollection<SizeT, IntT, FloatT>> {
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

        let v_si = OrdPairSICollection::<SizeT, IntT, FloatT>::from_vec(
            nodes.hist_dim.as_slice().unwrap(),
            &v_si,
        );
        let v_pmi = PairMICollection::<IntT, FloatT>::from_vec(&v_pmi);
        Ok((v_si, v_pmi))
    }
}

impl<'a> MISIWorkFlow<'a> {
    fn write_nodes_h5(
        &self,
        nodes: &NodeCollection<i64, i32, f32>,
        si_start: &Array1<i64>,
    ) -> Result<()> {
        let hfptr = io::create_file(&self.args.misi_data_file)?;
        let data_group = hfptr.create_group("data")?;
        data_group
            .new_attr::<i64>()
            .create("nvars")?
            .write_scalar(&(self.adata.nvars as i64))?;
        data_group
            .new_attr::<i64>()
            .create("nobs")?
            .write_scalar(&(self.adata.nobs as i64))?;
        data_group
            .new_attr::<i64>()
            .create("npairs")?
            .write_scalar(&(self.adata.npairs as i64))?;
        data_group
            .new_attr::<i64>()
            .create("nsi")?
            .write_scalar(&nodes.nsi)?;

        io::write_1d(&data_group, "hist_dim", &nodes.hist_dim)?;
        io::write_1d(&data_group, "hist_start", &nodes.hist_start)?;
        io::write_1d(&data_group, "hist", &nodes.ahist)?;

        io::write_1d(&data_group, "bins_dim", &nodes.bin_dim)?;
        io::write_1d(&data_group, "bins_start", &nodes.bin_start)?;
        io::write_1d(&data_group, "bins", &nodes.abins)?;

        io::write_1d(&data_group, "si_start", si_start)?;
        Ok(())
    }

    fn write_node_pairs(
        &self,
        npairs_si: &OrdPairSICollection<i64, i32, f32>,
        npairs_mi: &PairMICollection<i32, f32>,
    ) -> Result<()> {
        let h5_fptr =
            mpio::open_file_rw(self.mpi_ifx, &self.args.misi_data_file)?;
        let data_group = h5_fptr.group("data")?;
        //TODO::
        mpio::block_write1d(self.mpi_ifx, &data_group, "mi", &npairs_mi.mi)?;
        mpio::block_write1d(self.mpi_ifx, &data_group, "si", &npairs_si.si)?;
        mpio::block_write1d(self.mpi_ifx, &data_group, "lmr", &npairs_si.lmr)?;
        //
        Ok(())
    }

    pub fn run(&self) -> Result<()> {
        let nodes = MISIWorkFlowHelper::<i64, i32, f32>::construct_nodes(
            self,
            self.mpi_ifx.rank,
        )?;
        let (npairs_si, npairs_mi) =
            MISIWorkFlowHelper::<i64, i32, f32>::construct_node_pairs(
                self,
                self.mpi_ifx.rank,
                &nodes,
            )?;
        //let hist_dim: Vec<usize> = nodes
        //    .hist_dim
        //    .to_vec()
        //    .iter()
        //    .map(|x| *x as usize)
        //    .collect();
        let npairs_mi = npairs_mi.distribute(self.mpi_ifx)?;
        let mut npairs_si = npairs_si.distribute(self.mpi_ifx)?;
        npairs_si.prepare_serialize(self.mpi_ifx)?;

        //println!("Done {}", node_pairs.len());
        if self.mpi_ifx.rank == 0 {
            self.write_nodes_h5(&nodes, &npairs_si.si_start)?;
        }
        self.mpi_ifx.comm().barrier();
        self.write_node_pairs(&npairs_si, &npairs_mi)?;
        Ok(())
    }
}
