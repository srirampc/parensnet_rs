use anyhow::{Ok, Result};
use hdf5::H5Type;
use mpi::traits::{CommunicatorCollectives, Equivalence};
use ndarray::{Array2, ArrayView1};
use sope::{
    collective::gather_strings, reduction::allreduce_sum, timer::SectionTimer,
};
use std::{iter::zip, marker::PhantomData};

use super::{
    MISIWorkFlowTrait, WorkflowArgs,
    ds::{
        BPTrait, BatchPairs, Node, NodeCollection, NodePair, NodePairCollection,
        OrdPairSI, OrdPairSICollection, PairMI, PairMICollection,
    },
};
use crate::{
    comm::CommIfx,
    cond_debug, cond_info,
    h5::{io, mpio},
    hist::histogram_2d,
    mvim::imeasures::{
        lmr_about_x_from_lvji, lmr_about_y_from_lvji, log_jvi_ratio,
        mi_from_ljvi, si_from_ljvi,
    },
    types::{PNFloat, PNInteger},
    util::{triu_index_to_pair, triu_pair_to_index},
};

pub(super) struct MISIWorkFlowHelper<SizeT, IntT, FloatT> {
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> MISIWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    pub fn construct_nodes(
        wf: &dyn MISIWorkFlowTrait,
        rank: i32,
    ) -> Result<NodeCollection<SizeT, IntT, FloatT>> {
        let s_timer = SectionTimer::from_comm(wf.comm_ifx().comm(), ",");
        let columns = wf.wf_dist().var_dist[rank as usize].clone();
        wf.io_timer().reset();
        let rdata = wf.ann_data().par_read_range_data_around::<FloatT>(
            columns.clone(),
            wf.wf_args().nroundup,
            wf.comm_ifx(),
        )?;
        wf.io_timer().add_elapsed();
        let var_data: Vec<Node<IntT, FloatT>> = columns
            .into_iter()
            .enumerate()
            .map(|(i, _cx)| Node::<IntT, FloatT>::from_data(rdata.column(i)))
            .collect();
        log::debug!(
            "At Rank {} Built {} nodes",
            wf.comm_ifx().rank,
            var_data.len()
        );
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Local Nodes Construction ");
            let n_nodes = allreduce_sum(&(var_data.len()), wf.comm_ifx().comm());
            cond_info!(
                wf.comm_ifx().is_root(); "Total nodes: {} ", n_nodes
            );
            s_timer.reset();
        }
        let nodes = NodeCollection::<SizeT, IntT, FloatT>::from_nodes(
            &var_data,
            wf.comm_ifx().comm(),
        )?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Nodes Collection");
            cond_info!(
                wf.comm_ifx().is_root(); "Collected nodes: {} at root", nodes.len()
            );
            s_timer.reset();
        }
        if log::log_enabled!(log::Level::Debug) {
            wf.comm_ifx().comm().barrier();
            cond_debug!(wf.comm_ifx().is_root(); "Collected {} Nodes", nodes.len());
        }
        Ok(nodes)
    }

    pub(super) fn compute_hist_node_pair(
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
        indices: (usize, usize),
        rcdata: (ArrayView1<FloatT>, ArrayView1<FloatT>),
    ) -> Array2<FloatT> {
        let (x_idx, y_idx) = indices;
        let (x_data, y_data) = rcdata;
        let (x_bins, y_bins) = (nodes.bins(x_idx), nodes.bins(y_idx));
        histogram_2d(
            x_data,
            x_bins.as_slice().unwrap(),
            y_data,
            y_bins.as_slice().unwrap(),
        )
    }

    fn compute_lmr_node_pair(
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
        indices: (usize, usize),
        args: &WorkflowArgs,
        xy_tab: Array2<FloatT>,
    ) -> NodePair<IntT, FloatT> {
        let (x_idx, y_idx) = indices;
        let (x_hist, y_hist) = (nodes.hist(x_idx), nodes.hist(y_idx));
        let tbase = args.tbase.clone();
        let nobs = FloatT::from_usize(args.nobs).unwrap();
        let ljvi_ratio =
            log_jvi_ratio(xy_tab.view(), x_hist, y_hist, tbase, nobs);
        let (x_lmr, y_lmr) = (
            lmr_about_x_from_lvji(ljvi_ratio.view(), xy_tab.view(), Some(nobs)),
            lmr_about_y_from_lvji(ljvi_ratio.view(), xy_tab.view(), Some(nobs)),
        );
        let pair = (
            IntT::from_usize(x_idx).unwrap(),
            IntT::from_usize(y_idx).unwrap(),
        );
        let mi = mi_from_ljvi(ljvi_ratio.view(), xy_tab.view(), Some(nobs));
        let index = triu_pair_to_index(args.nvars, x_idx, y_idx);

        if args.lmr_only {
            return (
                OrdPairSI {
                    about: pair.0,
                    by: pair.1,
                    si: None,
                    lmr: x_lmr,
                },
                OrdPairSI {
                    about: pair.1,
                    by: pair.0,
                    si: None,
                    lmr: y_lmr,
                },
                PairMI::<IntT, FloatT> {
                    index,
                    pair: None,
                    xy_tab: None,
                    mi,
                },
            );
        }
        let (x_si, y_si) = si_from_ljvi(
            ljvi_ratio.view(),
            xy_tab.view(),
            x_hist.view(),
            y_hist.view(),
        );
        (
            OrdPairSI {
                about: pair.0,
                by: pair.1,
                si: Some(x_si),
                lmr: x_lmr,
            },
            OrdPairSI {
                about: pair.1,
                by: pair.0,
                si: Some(y_si),
                lmr: y_lmr,
            },
            PairMI::<IntT, FloatT> {
                index,
                pair: Some(pair),
                xy_tab: Some(xy_tab),
                mi,
            },
        )
    }

    pub fn build_node_pair(
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
        indices: (usize, usize),
        rcdata: (ArrayView1<FloatT>, ArrayView1<FloatT>),
        args: &WorkflowArgs,
    ) -> NodePair<IntT, FloatT> {
        let xy_tab = Self::compute_hist_node_pair(nodes, indices, rcdata);
        Self::compute_lmr_node_pair(nodes, indices, args, xy_tab)
    }

    pub(super) fn load_batch_data(
        wf: &dyn MISIWorkFlowTrait,
        rank: i32,
        bidx: usize,
    ) -> Result<(Array2<FloatT>, Array2<FloatT>)> {
        let (rows, cols) = wf.wf_dist().pairs_2d().batch_range(bidx, rank);
        wf.io_timer().reset();
        let block_data = (
            wf.ann_data().par_rmajor_read_range_data_around::<FloatT>(
                rows.clone(),
                wf.wf_args().nroundup,
                wf.comm_ifx(),
            )?,
            wf.ann_data().par_rmajor_read_range_data_around::<FloatT>(
                cols.clone(),
                wf.wf_args().nroundup,
                wf.comm_ifx(),
            )?,
        );
        wf.io_timer().add_elapsed();
        Ok(block_data)
    }

    fn construct_hist_node_pairs_for_batch(
        wf: &dyn MISIWorkFlowTrait,
        rank: i32,
        bidx: usize,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<Vec<PairMI<IntT, FloatT>>> {
        let (row_data, col_data) = Self::load_batch_data(wf, rank, bidx)?;
        if log::log_enabled!(log::Level::Debug) {
            let n_hist = allreduce_sum(&(row_data.len()), wf.comm_ifx().comm());
            cond_debug!(
                wf.comm_ifx().is_root(); "Loaded Batch {} with : {} ", bidx, n_hist
            );
        }
        let (rows, cols) = wf.wf_dist().pairs_2d().batch_range(bidx, rank);
        let mut v_hist: Vec<PairMI<IntT, FloatT>> = Vec::new();
        for (i, rx) in rows.clone().enumerate() {
            let r_data = row_data.column(i);
            for (j, cx) in cols.clone().enumerate() {
                if rx < cx {
                    let c_data = col_data.column(j);
                    let index = triu_pair_to_index(wf.wf_args().nvars, rx, cx);
                    let hist = Self::compute_hist_node_pair(
                        nodes,
                        (rx, cx),
                        (r_data, c_data),
                    );
                    v_hist.push(PairMI::<IntT, FloatT> {
                        mi: FloatT::zero(),
                        index,
                        pair: Some((
                            IntT::from_usize(rx).unwrap(),
                            IntT::from_usize(cx).unwrap(),
                        )),
                        xy_tab: Some(hist),
                    });
                }
            }
        }
        if log::log_enabled!(log::Level::Debug) {
            // wf.mpi_ifx.comm().barrier();
            let n_hist = allreduce_sum(&(v_hist.len()), wf.comm_ifx().comm());
            cond_debug!(
                wf.comm_ifx().is_root(); "Built Batch {} with : {} ", bidx, n_hist
            );
        }
        Ok(v_hist)
    }

    fn construct_node_pairs_for_batch(
        wf: &dyn MISIWorkFlowTrait,
        rank: i32,
        bidx: usize,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<BatchPairs<IntT, FloatT>> {
        let s_timer = SectionTimer::from_comm(wf.comm_ifx().comm(), ",");
        if log::log_enabled!(log::Level::Debug) {
            s_timer.reset();
        }
        let (rows, cols) = wf.wf_dist().pairs_2d().batch_range(bidx, rank);
        let (row_data, col_data) = Self::load_batch_data(wf, rank, bidx)?;
        if log::log_enabled!(log::Level::Debug) {
            s_timer.info_section("Local Batch Loading ");
            if wf.detailed_log() {
                let v_str = gather_strings(
                    format!("{}:({:?},{:?})", rank, &rows, &cols).to_string(),
                    0,
                    wf.comm_ifx().comm(),
                )?;
                if let Some(v_str) = v_str {
                    log::debug!("Loaded Batch Data {} {:?}", bidx, v_str);
                }
            }
            s_timer.reset();
        }
        let mut batch_pairs =
            BatchPairs::<IntT, FloatT>::new(rows.clone(), cols.clone());
        for (i, rx) in rows.clone().enumerate() {
            let r_data = row_data.column(i);
            for (j, cx) in cols.clone().enumerate() {
                if rx < cx {
                    let c_data = col_data.column(j);
                    batch_pairs.push(Self::build_node_pair(
                        nodes,
                        (rx, cx),
                        (r_data, c_data),
                        wf.wf_args(),
                    ));
                }
            }
        }
        if log::log_enabled!(log::Level::Debug) {
            s_timer.info_section("Local Batch Compute ");
            // wf.mpi_ifx.comm().barrier();
            let n_mi =
                allreduce_sum(&(batch_pairs.2.len()), wf.comm_ifx().comm());
            let n_si =
                allreduce_sum(&(batch_pairs.0.len()), wf.comm_ifx().comm());
            cond_debug!(
                wf.comm_ifx().is_root();
                "Built Batch {} MI : {}; SI: {} ",
                bidx, n_mi, n_si,
            );
        }
        Ok(batch_pairs)
    }

    pub fn construct_hist_node_pairs(
        wf: &dyn MISIWorkFlowTrait,
        rank: i32,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<PairMICollection<IntT, FloatT>> {
        let nbatches = wf.wf_dist().pairs_2d().num_batches();

        let v_hist = (0..nbatches)
            .map(|bidx| {
                Self::construct_hist_node_pairs_for_batch(wf, rank, bidx, nodes)
            })
            .collect::<Result<Vec<_>>>()?;

        // flatten
        let mut v_hist: Vec<_> = v_hist.into_iter().flatten().collect();
        v_hist.sort_by_key(|x| x.index);
        if log::log_enabled!(log::Level::Info) {
            wf.comm_ifx().comm().barrier();
            let n_hist = allreduce_sum(&(v_hist.len()), wf.comm_ifx().comm());
            cond_info!(
                wf.comm_ifx().is_root(); "Built Hist : {} ", n_hist
            );
        }

        let v_hist = PairMICollection::<IntT, FloatT>::from_vec(&v_hist);
        Ok(v_hist)
    }

    pub fn construct_node_pairs_in_batches(
        wf: &dyn MISIWorkFlowTrait,
        rank: i32,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
    ) -> Result<NodePairCollection<IntT, FloatT>> {
        let nbatches = wf.wf_dist().pairs_2d().num_batches();

        let (v_si, v_siy, v_pmi) = (0..nbatches)
            .map(|bidx| {
                Self::construct_node_pairs_for_batch(wf, rank, bidx, nodes)
            })
            .collect::<Result<(Vec<_>, Vec<_>, Vec<_>)>>()?;

        // flatten all vec of vec
        let v_pmi: Vec<PairMI<IntT, FloatT>> =
            v_pmi.into_iter().flatten().collect();

        let v_si: Vec<OrdPairSI<IntT, FloatT>> =
            v_si.into_iter().flatten().collect();
        let v_siy: Vec<OrdPairSI<IntT, FloatT>> =
            v_siy.into_iter().flatten().collect();
        Self::construct_node_pairs_collection(
            wf.comm_ifx(),
            nodes,
            v_pmi,
            v_si,
            v_siy,
        )
    }

    fn construct_node_pairs_collection(
        mpi_ifx: &CommIfx,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
        mut v_pmi: Vec<PairMI<IntT, FloatT>>,
        mut v_si: Vec<OrdPairSI<IntT, FloatT>>,
        v_siy: Vec<OrdPairSI<IntT, FloatT>>,
    ) -> Result<NodePairCollection<IntT, FloatT>> {
        v_pmi.sort_by_key(|x| x.pair);
        v_si.extend(v_siy);
        v_si.sort_by_key(|x| (x.about, x.by));
        log::debug!(
            "At Rank {} :: Built MI {} and SI {}",
            mpi_ifx.rank,
            v_pmi.len(),
            v_si.len()
        );
        if log::log_enabled!(log::Level::Debug) {
            mpi_ifx.comm().barrier();
            let n_mi = allreduce_sum(&(v_pmi.len()), mpi_ifx.comm());
            let n_si = allreduce_sum(&(v_si.len()), mpi_ifx.comm());
            cond_debug!(
                mpi_ifx.is_root(); "Built MI : {} & SI : {} ", n_mi, n_si
            );
        }

        let v_si =
            OrdPairSICollection::<IntT, FloatT>::from_vec(nodes.len(), &v_si);
        let v_pmi = PairMICollection::<IntT, FloatT>::from_vec(&v_pmi);
        if log::log_enabled!(log::Level::Debug) {
            mpi_ifx.comm().barrier();
            let n_mi = allreduce_sum(&(v_pmi.index.len()), mpi_ifx.comm());
            let n_si = allreduce_sum(&(v_si.about.len()), mpi_ifx.comm());
            cond_debug!(
                mpi_ifx.is_root(); "Collected {} MI & {} SI", n_mi, n_si
            );
        }
        Ok((v_si, v_pmi))
    }

    pub fn construct_lmr_node_pairs(
        wf: &dyn MISIWorkFlowTrait,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
        hist_pairs: PairMICollection<IntT, FloatT>,
    ) -> Result<Option<NodePairCollection<IntT, FloatT>>> {
        if hist_pairs.xy_tab.is_none() {
            return Ok(None);
        }

        let idx2dim = |idx: usize| {
            let (dx, dy): (usize, usize) = triu_index_to_pair(nodes.len(), idx);
            let dim_x = nodes.hist_dim[dx];
            let dim_y = nodes.hist_dim[dy];
            (dim_x, dim_y)
        };
        let dims_itr = hist_pairs.index.iter().map(|idx| idx2dim(*idx));
        let offset_itr =
            dims_itr.clone().scan(0usize, |state, (dim_x, dim_y)| {
                let cx = *state;
                //let (dx, dy): (usize, usize) =
                //    triu_index_to_pair(nodes.len(), *idx);
                //let dim_x = nodes.hist_dim[dx].clone();
                //let dim_y = nodes.hist_dim[dy].clone();
                *state += dim_x.to_usize().unwrap() * dim_y.to_usize().unwrap();
                Some(cx)
            });
        if let Some(xy_tab) = hist_pairs.xy_tab.as_ref() {
            let sl_xytab = xy_tab.as_slice().unwrap();
            let (v_si, v_siy, v_pmi) = zip(hist_pairs.index.iter(), offset_itr)
                .zip(dims_itr)
                .map(|((idx, offset), (x_dim, y_dim))| {
                    let (i, j) = triu_index_to_pair(wf.wf_args().nvars, *idx);
                    let (x_dim, y_dim) =
                        (x_dim.to_usize().unwrap(), y_dim.to_usize().unwrap());
                    let xy_dim = x_dim * y_dim;
                    let xyt_vec = sl_xytab[offset..(offset + xy_dim)].to_vec();
                    let xy_tab = Array2::from_shape_vec((x_dim, y_dim), xyt_vec)?;
                    Ok(Self::compute_lmr_node_pair(
                        nodes,
                        (i, j),
                        wf.wf_args(),
                        xy_tab,
                    ))
                })
                .collect::<Result<(Vec<_>, Vec<_>, Vec<_>)>>()?;
            Ok(Some(Self::construct_node_pairs_collection(
                wf.comm_ifx(),
                nodes,
                v_pmi,
                v_si,
                v_siy,
            )?))
        } else {
            Ok(None)
        }
    }

    pub fn write_nodes_h5(
        wf: &dyn MISIWorkFlowTrait,
        nodes: &NodeCollection<SizeT, IntT, FloatT>,
        misi_data_file: &str,
    ) -> Result<()> {
        let hfptr = io::create_file(misi_data_file)?;
        let data_group = hfptr.create_group("data")?;
        data_group
            .new_attr::<SizeT>()
            .create("nvars")?
            .write_scalar(&(SizeT::from_usize(wf.wf_args().nvars).unwrap()))?;
        data_group
            .new_attr::<SizeT>()
            .create("nobs")?
            .write_scalar(&(SizeT::from_usize(wf.wf_args().nobs).unwrap()))?;
        data_group
            .new_attr::<SizeT>()
            .create("npairs")?
            .write_scalar(&(SizeT::from_usize(wf.ann_data().npairs).unwrap()))?;
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

    pub fn write_hist_pairs(
        w: &dyn MISIWorkFlowTrait,
        hist_pairs: &PairMICollection<IntT, FloatT>,
        hist_data_file: &str,
    ) -> Result<()> {
        // let rank_out_file = format!(
        //     "{}.{:04}.{}",
        //     hist_data_file.strip_suffix(".h5").unwrap_or_default(),
        //     w.comm_ifx().rank,
        //     "h5"
        // );
        // let h5_fptr = io::create_file(&rank_out_file)?;
        // let data_group = h5_fptr.create_group("data")?;
        // if let Some(xy_tab) = hist_pairs.xy_tab.as_ref() {
        //     io::write_1d(&data_group, "pair_hist", xy_tab)?;
        // }
        // io::write_1d(&data_group, "mi", &hist_pairs.mi)?;
        // h5_fptr.close()?;

        let h5_fptr =
            mpio::create_file(w.comm_ifx(), hist_data_file)?;
        let data_group = h5_fptr.create_group("data")?;
        if let Some(xy_tab) = hist_pairs.xy_tab.as_ref() {
            mpio::block_write1d(w.comm_ifx(), &data_group, "hist", xy_tab)?;
        }
        mpio::block_write1d(w.comm_ifx(), &data_group, "mi", &hist_pairs.mi)?;
        Ok(())
    }

    pub fn write_node_pairs(
        w: &dyn MISIWorkFlowTrait,
        npairs_si: &OrdPairSICollection<IntT, FloatT>,
        npairs_mi: &PairMICollection<IntT, FloatT>,
    ) -> Result<()> {
        let h5_fptr =
            mpio::open_file_rw(w.comm_ifx(), &w.wf_args().misi_data_file)?;
        let data_group = h5_fptr.group("data")?;
        //TODO::
        mpio::block_write1d(w.comm_ifx(), &data_group, "mi", &npairs_mi.mi)?;
        if let Some(rsi) = npairs_si.si.as_ref() {
            mpio::block_write1d(w.comm_ifx(), &data_group, "si", rsi)?;
        }
        mpio::block_write1d(w.comm_ifx(), &data_group, "lmr", &npairs_si.lmr)?;
        Ok(())
    }
}
