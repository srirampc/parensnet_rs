use anyhow::{Ok, Result};
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use ndarray::{Array1, Array2, ArrayView1};
use num::{FromPrimitive, ToPrimitive};
use sope::{collective::allgatherv_full_vec, cond_info, util::exc_prefix_sum};
use std::fmt::Debug;

use super::{WorkDistributor, WorkflowArgs};
use crate::{
    anndata::AnnData,
    comm::CommIfx,
    h5::io,
    hist::{bayesian_blocks_bin_edges, histogram_1d, histogram_2d},
    mvim::{
        imeasures::{
            lmr_about_x_from_lvji, lmr_about_y_from_lvji, log_jvi_ratio,
            mi_from_ljvi, si_from_ljvi,
        },
        rv::{MRVFloat, MRVInteger},
    },
    types::{AddFromZero, AssignOps, DbgDisplay, OrderedFloat},
    util::triu_pair_to_index,
};

pub struct MISIWorkFlow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
    pub adata: &'a AnnData,
}

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
        FloatT: 'static + OrderedFloat + FromPrimitive + AssignOps + DbgDisplay,
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

struct NodeCollection<IntT, FloatT> {
    bin_sizes: Array1<IntT>,
    hist_sizes: Array1<IntT>,
    bin_starts: Array1<usize>,
    hist_starts: Array1<usize>,
    // bins/hist flattened to a histogram
    abins: Array1<FloatT>,
    ahist: Array1<FloatT>,
}

impl<IntT, FloatT> NodeCollection<IntT, FloatT> {
    fn from_nodes(
        vdata: &[Node<IntT, FloatT>],
        comm: &dyn Communicator,
    ) -> Result<Self>
    where
        IntT: Clone + Default + Debug + Equivalence + ToPrimitive,
        FloatT: Clone + Default + Debug + Equivalence,
    {
        // sizes and starts
        let bin_sizes: Vec<IntT> =
            vdata.iter().map(|x| x.nbins.clone()).collect();
        let hist_sizes: Vec<IntT> =
            vdata.iter().map(|x| x.nhist.clone()).collect();
        let bin_sizes: Vec<IntT> = allgatherv_full_vec(&bin_sizes, comm)?;
        let hist_sizes: Vec<IntT> = allgatherv_full_vec(&hist_sizes, comm)?;
        let uszhist: Vec<usize> =
            hist_sizes.iter().map(|x| x.to_usize().unwrap()).collect();
        let hist_starts: Vec<usize> = exc_prefix_sum(uszhist.into_iter(), 1);
        let uszbin: Vec<usize> =
            bin_sizes.iter().map(|x| x.to_usize().unwrap()).collect();
        let bin_starts: Vec<usize> = exc_prefix_sum(uszbin.into_iter(), 1);

        let vbins: Vec<FloatT> =
            vdata.iter().flat_map(|x| x.bins.iter().cloned()).collect();
        let vbins: Vec<FloatT> = allgatherv_full_vec(&vbins, comm)?;
        let abins = Array1::from_vec(vbins);
        let vhist: Vec<FloatT> =
            vdata.iter().flat_map(|x| x.hist.iter().cloned()).collect();
        let vhist: Vec<FloatT> = allgatherv_full_vec(&vhist, comm)?;
        let ahist = Array1::from_vec(vhist);

        Ok(NodeCollection::<IntT, FloatT> {
            bin_sizes: Array1::from_vec(bin_sizes),
            hist_sizes: Array1::from_vec(hist_sizes),
            hist_starts: Array1::from_vec(hist_starts),
            bin_starts: Array1::from_vec(bin_starts),
            abins,
            ahist,
        })
    }

    fn hist(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        IntT: ToPrimitive,
    {
        let hstart = self.hist_starts[idx];
        let hsize = self.hist_sizes[idx].to_usize().unwrap();
        let hend = hstart + hsize;
        self.ahist.slice(ndarray::s![hstart..hend])
    }

    fn bins(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        IntT: ToPrimitive,
    {
        let bstart = self.bin_starts[idx];
        let bsize = self.bin_sizes[idx].to_usize().unwrap();
        let bend = bstart + bsize;
        self.abins.slice(ndarray::s![bstart..bend])
    }

    fn len(&self) -> usize {
        self.hist_sizes.len()
    }

    fn is_empty(&self) -> bool {
        self.hist_sizes.is_empty()
    }
}

struct NodePair<IntT, FloatT> {
    pair_index: usize,
    x_idx: IntT,
    y_idx: IntT,
    xy_tab: Array2<FloatT>,
    x_si: Array1<FloatT>,
    y_si: Array1<FloatT>,
    x_lmr: Array1<FloatT>,
    y_lmr: Array1<FloatT>,
    mi: FloatT,
}

impl<IntT, FloatT> NodePair<IntT, FloatT> {
    fn from_node_collection(
        nodes: &NodeCollection<IntT, FloatT>,
        indices: (usize, usize),
        rcdata: (ArrayView1<FloatT>, ArrayView1<FloatT>),
        args: &WorkflowArgs,
    ) -> Self
    where
        IntT: Clone + Default + Debug + Equivalence + MRVInteger,
        FloatT: 'static + Default + Equivalence + MRVFloat,
    {
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
        let pair_index = triu_pair_to_index(args.npairs, x_idx, y_idx);

        Self {
            pair_index,
            x_idx: IntT::from_usize(x_idx).unwrap(),
            y_idx: IntT::from_usize(y_idx).unwrap(),
            xy_tab,
            x_lmr,
            y_lmr,
            x_si,
            y_si,
            mi,
        }
    }
}

impl<'a> MISIWorkFlow<'a> {
    fn construct_nodes(&self, rank: i32) -> Result<NodeCollection<i32, f32>> {
        let columns = self.wdistr.var_dist[rank as usize].clone();
        let rdata = self
            .adata
            .read_range_data_around::<f32>(columns.clone(), self.args.nroundup)?;
        let var_data: Vec<Node<i32, f32>> = columns
            .into_iter()
            .enumerate()
            .map(|(i, _cx)| Node::<i32, f32>::from_data(rdata.column(i)))
            .collect();
        log::info!(
            "At Rank {} Built {} nodes",
            self.mpi_ifx.rank,
            var_data.len()
        );
        let nodes = NodeCollection::<i32, f32>::from_nodes(
            &var_data,
            self.mpi_ifx.comm(),
        )?;
        if log::log_enabled!(log::Level::Info) {
            self.mpi_ifx.comm().barrier();
            cond_info!(self.mpi_ifx.is_root(); "Built {} Nodes", nodes.len());
        }
        Ok(nodes)
    }

    fn construct_batch_pairs(
        &self,
        rank: i32,
        bidx: usize,
        nodes: &NodeCollection<i32, f32>,
    ) -> Result<Vec<NodePair<i32, f32>>> {
        let (rows, cols) = self.wdistr.pairs_2d().batch_range(bidx, rank);
        let (row_data, col_data) = (
            self.adata.read_range_data_around::<f32>(
                rows.clone(),
                self.args.nroundup,
            )?,
            self.adata.read_range_data_around::<f32>(
                cols.clone(),
                self.args.nroundup,
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
                        let npair = NodePair::<i32, f32>::from_node_collection(
                            nodes,
                            (rx, cx),
                            (r_data, c_data),
                            self.args,
                        );
                        Some(npair)
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<NodePair<i32, f32>>>();
        Ok(node_pairs)
    }

    fn construct_node_pairs(
        &self,
        rank: i32,
        nodes: &NodeCollection<i32, f32>,
    ) -> Result<Vec<NodePair<i32, f32>>> {
        let nbatches = self.wdistr.pairs_2d().num_batches();
        let mut bat_results: Vec<NodePair<i32, f32>> = Vec::new();
        for bidx in 0..nbatches {
            bat_results.extend(self.construct_batch_pairs(rank, bidx, nodes)?);
        }
        bat_results.sort_by(|x, y| (x.x_idx, x.y_idx).cmp(&(y.x_idx, y.y_idx)));
        Ok(bat_results)
    }

    fn write_nodes_h5(&self, nodes: &NodeCollection<i32, f32>) -> Result<()> {
        let hfptr = io::create_file(&self.args.misi_data_file)?;
        let data_group = hfptr.create_group("data")?;
        data_group
            .new_attr::<usize>()
            .create("nvars")?
            .write_scalar(&self.adata.nvars)?;
        data_group
            .new_attr::<usize>()
            .create("nobs")?
            .write_scalar(&self.adata.nobs)?;
        data_group
            .new_attr::<usize>()
            .create("npairs")?
            .write_scalar(&self.adata.npairs)?;
        io::write_1d(&data_group, "hist_dim", &nodes.hist_sizes)?;
        io::write_1d(&data_group, "hist_start", &nodes.hist_starts)?;
        io::write_1d(&data_group, "hist", &nodes.ahist)?;

        io::write_1d(&data_group, "bins_dim", &nodes.bin_sizes)?;
        io::write_1d(&data_group, "bins_start", &nodes.bin_starts)?;
        io::write_1d(&data_group, "bins", &nodes.abins)?;
        Ok(())
    }

    pub fn run(&self) -> Result<()> {
        let nodes = self.construct_nodes(self.mpi_ifx.rank)?;
        //let node_pairs = self.construct_node_pairs(self.mpi_ifx.rank, &nodes)?;
        //println!("Done {}", node_pairs.len());
        if self.mpi_ifx.rank == 0 {
            self.write_nodes_h5(&nodes)?;
        }
        Ok(())
    }
}
