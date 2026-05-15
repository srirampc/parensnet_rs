//! Driver for the MISI workflows 
//! (computes Mutual Information / Specific Information / LMR) 
//! in [`crate::pucn`].
//!
//! [`MISIWorkFlow`] is a light-weight struct that bundles the MPI
//! communicator interface, the pair-work distributor, the parsed
//! [`WorkflowArgs`], the source AnnData, and a cumulative IO timer.
//! It implements [`MISIWorkFlowTrait`] that uses the stateless helpers in
//! [`crate::pucn::helpers::MISIWorkFlowHelper`] to build the MI/SI/LMR
//! data containers.
//!
//! The struct exposes one method per [`crate::pucn::RunMode`] it can
//! handle:
//!
//! * [`MISIWorkFlow::run`] — full pipeline ([`RunMode::MISI`](crate::pucn::RunMode::MISI)):
//!   construct nodes (or reload them from `hist_data_file`), build
//!   per-pair MI / SI / LMR vectors in batches, distribute, and
//!   save to disk.
//! * [`MISIWorkFlow::run_hist_nodes`] — only construct the
//!   per-variable [`NodeCollection`]
//!   ([`RunMode::HistNodes`](crate::pucn::RunMode::HistNodes)) and 
//!   save to disk.
//! * [`MISIWorkFlow::run_hist`] — construct nodes plus the joint
//!   histograms ([`PairMICollection`]) 
//!   ([`RunMode::HistDist`](crate::pucn::RunMode::HistDist))
//!   and save to disk.
//! * [`MISIWorkFlow::run_misi_dist`] — load nodes and joint
//!   histograms from disk, compute MI / SI / LMR, distribute, and
//!   save to disk ([`RunMode::MISIDist`](crate::pucn::RunMode::MISIDist)).
//! * [`MISIWorkFlow::run_misi_dist_from_nodes`] — load only the
//!   nodes from disk, recompute the histograms in-process, and then
//!   continue with the MI / SI / LMR step
//!   ([`RunMode::HistNodes2MISI`](crate::pucn::RunMode::HistNodes2MISI)).

use std::path::Path;

use anyhow::{Ok, Result};
use mpi::traits::CommunicatorCollectives;
use sope::{
    reduction::{all_of, allreduce_sum},
    timer::{CumulativeTimer, SectionTimer},
};

use super::{
    MISIWorkFlowTrait, WorkflowArgs,
    ds::{NodeCollection, OrdPairSICollection, PairMICollection},
    helpers::MISIWorkFlowHelper,
};
use crate::{
    anndata::AnnData, comm::CommIfx, cond_info, util::PairWorkDistributor,
};


/// Runtime context for execution of the MISI workflows.
///
/// Includes references to the MPI communicator wrapper, work distributor, 
/// parsed configuration, AnnData. Also has a cumulative IO timer for the
/// duration of one workflow run.
pub struct MISIWorkFlow<'a> {
    /// MPI communicator wrapper used by every collective call.
    pub mpi_ifx: &'a CommIfx,
    /// Pair-work distributor describing 
    /// (a) the per-rank block distribution of the variable and
    /// (b) the 2-D tiles of the pair grid.
    pub wdistr: &'a PairWorkDistributor,
    /// Parsed workflow configuration.
    pub args: &'a WorkflowArgs,
    /// Source AnnData expression file.
    pub adata: &'a AnnData,
    /// Cumulative IO timer; the helpers add elapsed read/write time
    /// to it on every parallel-IO call.
    pub io_timer: CumulativeTimer<'a>,
}

impl<'a> MISIWorkFlowTrait<'a> for MISIWorkFlow<'a> {
    fn comm_ifx(&self) -> &'a CommIfx {
        self.mpi_ifx
    }

    fn wf_dist(&self) -> &'a PairWorkDistributor {
        self.wdistr
    }

    fn wf_args(&self) -> &'a WorkflowArgs {
        self.args
    }

    fn ann_data(&self) -> &'a AnnData {
        self.adata
    }

    fn io_timer(&self) -> &CumulativeTimer<'a> {
        &self.io_timer
    }

    fn detailed_log(&self) -> bool {
        std::env::var("PARENTSNET_DETAIL_LOG").is_ok()
    }
}

impl<'a> MISIWorkFlow<'a> {
    /// Distribute and save to distk a previously computed
    /// `(NodeCollection, OrdPairSICollection, PairMICollection)`
    /// triple to the workflow's MISI HDF5 file.
    ///
    /// The MI and SI/LMR collections are first re-shuffled across
    /// ranks (skipped when running on a single rank) so that each
    /// rank holds its block of the global pair space. 
    /// The SI collection is then expanded to fill the diagonal entries 
    /// before serialization. 
    /// Rank 0 writes the [`NodeCollection`], and every rank participates in
    /// the parallel-IO pair-data write
    /// ([`MISIWorkFlowHelper::write_node_pairs`]).
    fn save_distribute(
        &self,
        nodes: NodeCollection<i64, i32, f32>,
        npairs_si: OrdPairSICollection<i32, f32>,
        npairs_mi: PairMICollection<i32, f32>,
    ) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        let npairs_mi = if self.mpi_ifx.size > 1 {
            npairs_mi
                .distribute(self.mpi_ifx, nodes.hist_dim.as_slice().unwrap())?
        } else {
            npairs_mi
        };
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Node Pairs MI Distribution");
            let n_mi =
                allreduce_sum(&(npairs_mi.index.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root(); "Distributed MI: {} ", n_mi
            );
            s_timer.reset();
        }

        let mut npairs_si = if self.mpi_ifx.size > 1 {
            npairs_si.distribute(self.mpi_ifx)?
        } else {
            npairs_si
        };
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Node Pairs SI Distribution");
            let n_sipx =
                allreduce_sum(&(npairs_si.about.len()), self.mpi_ifx.comm());
            let n_lmr =
                allreduce_sum(&(npairs_si.lmr.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root();
                "Distributed SI Pairs: {}, LMR: {}", n_sipx, n_lmr
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
            HelperT::write_nodes_h5(self, &nodes, &self.args.misi_data_file)?;
        }
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Write Nodes");
            cond_info!(
                self.mpi_ifx.is_root();
                "Completed Writing Nodes to {}", self.args.misi_data_file
            );
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

    /// Run the full MISI pipeline
    /// ([`RunMode::MISI`](crate::pucn::RunMode::MISI)).
    ///
    /// Reuses a saved [`NodeCollection`] from
    /// [`WorkflowArgs::hist_data_file`] when that file exists,
    /// otherwise constructs it via
    /// [`MISIWorkFlowHelper::construct_nodes`].  
    /// Builds the per-pair MI / SI / LMR records in batches with
    /// [`MISIWorkFlowHelper::construct_node_pairs_in_batches`],
    /// after which the results are distributed and 
    /// saved ([`Self::save_distribute`]).
    pub fn run(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        cond_info!(self.mpi_ifx.is_root(); "Starting MISIWorkFlow::Run");

        let nodes = if Path::new(&self.args.hist_data_file).exists() {
            NodeCollection::<i64, i32, f32>::from_h5(
                &self.args.hist_data_file,
                self.args.nvars,
            )?
        } else {
            HelperT::construct_nodes(self, self.mpi_ifx.rank)?
        };
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Construct Nodes");
            cond_info!(
                self.mpi_ifx.is_root();
                "Nodes Constructed: {} w. {}", nodes.len(), nodes.bin_dim.len()
            );
            s_timer.reset();
        }

        let (npairs_si, npairs_mi) = HelperT::construct_node_pairs_in_batches(
            self,
            self.mpi_ifx.rank,
            &nodes,
        )?;
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
        self.save_distribute(nodes, npairs_si, npairs_mi)?;
        self.io_timer.info_region("Total IO");
        Ok(())
    }

    /// Construct only the per-variable
    /// [`NodeCollection`] and save to disk
    /// ([`RunMode::HistNodes`](crate::pucn::RunMode::HistNodes)).
    ///
    /// Used to pre-compute the Bayesian-blocks histograms once and
    /// reuse them across subsequent runs (the file is read by
    /// [`Self::run_misi_dist`] and [`Self::run_misi_dist_from_nodes`]).
    /// NOTE:: Rank 0 writes [`WorkflowArgs::hist_data_file`].
    pub fn run_hist_nodes(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
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
        if self.mpi_ifx.rank == 0 {
            HelperT::write_nodes_h5(self, &nodes, &self.args.hist_data_file)?;
        }
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Write Nodes");
            cond_info!(self.mpi_ifx.is_root(); "Completed Writing Nodes");
            s_timer.reset();
        }
        self.io_timer.info_region("Total IO");
        Ok(())
    }

    /// Construct nodes plus the joint histograms
    /// ([`PairMICollection`]) and save them on disk to
    /// [`WorkflowArgs::hist_data_file`]
    /// ([`RunMode::HistDist`](crate::pucn::RunMode::HistDist)).
    ///
    /// Distributes the pairs across ranks for construction. 
    /// Rank 0 writes the [`NodeCollection`]; every rank participates in 
    /// the parallel-IO write of joint histograms
    /// ([`MISIWorkFlowHelper::write_hist_pairs`]).
    pub fn run_hist(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
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
        let hist_pairs =
            HelperT::construct_hist_node_pairs(self, self.mpi_ifx.rank, &nodes)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Construct Hist Pairs");
            let n_hist =
                allreduce_sum(&(hist_pairs.index.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root();
                "Node Pairs Constructed: {} ", n_hist
            );
            s_timer.reset();
        }

        let hist_pairs = if self.mpi_ifx.size > 1 {
            hist_pairs
                .distribute(self.mpi_ifx, nodes.hist_dim.as_slice().unwrap())?
        } else {
            hist_pairs
        };
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Node Pairs Distribution");
            let n_mi =
                allreduce_sum(&(hist_pairs.index.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root(); "Distributed Node Pairs: {} ", n_mi
            );
            s_timer.reset();
        }

        let write_sucess = if self.mpi_ifx.rank == 0 {
            match HelperT::write_nodes_h5(self, &nodes, &self.args.hist_data_file)
            {
                Err(err) => {
                    log::error!("Error in writing nodes : {}", err);
                    all_of(false, self.mpi_ifx.comm())
                }
                _ => all_of(true, self.mpi_ifx.comm()),
            }
        } else {
            all_of(true, self.mpi_ifx.comm())
        };
        assert!(write_sucess);
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Write Nodes");
            cond_info!(self.mpi_ifx.is_root(); "Completed Writing Nodes");
            s_timer.reset();
        }
        self.mpi_ifx.comm().barrier();
        HelperT::write_hist_pairs(self, &hist_pairs, &self.args.hist_data_file)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Write Node Pairs");
            cond_info!(self.mpi_ifx.is_root(); "Finished MISIWorkFlow::Run");
        }
        self.io_timer.info_region("Total IO");
        Ok(())
    }

    /// Construct MI / SI / LMR, then  distribute and  save to disk.
    ///
    /// Calls [`MISIWorkFlowHelper::construct_lmr_node_pairs`] on the
    /// supplied joint-histogram collection. When a vaild
    /// [`NodePairCollection`](crate::pucn::ds::NodePairCollection),
    /// result is returnted, distribute and save with
    /// [`Self::save_distribute`].
    fn run_misi_dist_from_hist(
        &self,
        nodes: NodeCollection<i64, i32, f32>,
        hist_pairs: PairMICollection<i32, f32>,
    ) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        if let Some((npairs_si, npairs_mi)) =
            HelperT::construct_lmr_node_pairs(self, &nodes, hist_pairs)?
        {
            if log::log_enabled!(log::Level::Info) {
                s_timer.info_section("Construct LMR Pairs");
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

            self.save_distribute(nodes, npairs_si, npairs_mi)
        } else {
            Ok(())
        }
    }

    /// Run the distributed MI / SI / LMR step from a fully
    /// pre-computed histogram file
    /// ([`RunMode::MISIDist`](crate::pucn::RunMode::MISIDist)).
    ///
    /// Both the [`NodeCollection`] and the
    /// [`PairMICollection`] (joint histograms) are loaded from the
    /// [`WorkflowArgs::hist_data_file`]. 
    /// Panices if [`WorkflowArgs::lmr_only`] is `true`.
    pub fn run_misi_dist(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        cond_info!(self.mpi_ifx.is_root(); "Starting MISIWorkFlow::Run MISI Dist");
        assert!(!self.args.lmr_only);
        // 1. Flat loading of  all the nodes
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        let nodes = NodeCollection::<i64, i32, f32>::from_h5(
            &self.args.hist_data_file,
            self.args.nvars,
        )?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Load Nodes");
            //cond_info!(
            //    self.mpi_ifx.is_root();
            //    "SI last : {}",
            //    nodes.si_start[nodes.si_start.len() - 1]
            //);
            cond_info!(
                self.mpi_ifx.is_root();
                "Nodes Loaded: {} w. {}", nodes.len(), nodes.bin_dim.len()
            );
            s_timer.reset();
        }
        // 2. Load node pairs
        let hist_pairs = PairMICollection::<i32, f32>::from_h5(
            self.mpi_ifx,
            self.args,
            &self.args.hist_data_file,
            nodes.hist_dim.as_slice().unwrap(),
        )?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Load Hist Pairs");
            let n_hist =
                allreduce_sum(&(hist_pairs.index.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root();
                "Hist Pairs Loaded: {} ", n_hist
            );
            s_timer.reset();
        }
        Self::run_misi_dist_from_hist(self, nodes, hist_pairs)?;
        cond_info!(self.mpi_ifx.is_root(); "Completed MISIWorkFlow::Run MISI Dist");
        self.io_timer.info_region("Total IO");
        Ok(())
    }

    /// Run the distributed MI / SI / LMR step starting from a
    /// nodes-only histogram file
    /// ([`RunMode::HistNodes2MISI`](crate::pucn::RunMode::HistNodes2MISI)).
    ///
    /// Loads only the [`NodeCollection`] from
    /// [`WorkflowArgs::hist_data_file`], rebuilds the joint histograms
    /// in-process via
    /// [`MISIWorkFlowHelper::construct_hist_node_pairs`], distributes
    /// them across ranks, and then delegates to
    /// [`Self::run_misi_dist_from_hist`]. 
    /// Useful when the
    /// nodes-only file produced by [`Self::run_hist_nodes`] is reused
    /// without persisting the (much larger) joint histograms.
    pub fn run_misi_dist_from_nodes(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        cond_info!(self.mpi_ifx.is_root(); "Starting MISIWorkFlow::Run MISI Dist from Nodes");
        // 1. Flat loading of  all the nodes
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        let nodes = NodeCollection::<i64, i32, f32>::from_h5(
            &self.args.hist_data_file,
            self.args.nvars,
        )?;

        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Load Nodes");
            cond_info!(
                self.mpi_ifx.is_root();
                "Nodes Loaded: {} w. {}", nodes.len(), nodes.bin_dim.len()
            );
            s_timer.reset();
        }
        let hist_pairs =
            HelperT::construct_hist_node_pairs(self, self.mpi_ifx.rank, &nodes)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Construct Hist Pairs");
            let n_hist =
                allreduce_sum(&(hist_pairs.index.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root();
                "Node Pairs Constructed: {} ", n_hist
            );
            s_timer.reset();
        }

        let hist_pairs = hist_pairs
            .distribute(self.mpi_ifx, nodes.hist_dim.as_slice().unwrap())?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Node Pairs Distribution");
            let n_mi =
                allreduce_sum(&(hist_pairs.index.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root(); "Distributed Node Pairs: {} ", n_mi
            );
            s_timer.reset();
        }
        Self::run_misi_dist_from_hist(self, nodes, hist_pairs)?;
        cond_info!(self.mpi_ifx.is_root(); "Complete MISIWorkFlow::Run MISI Dist from Nodes");
        self.io_timer.info_region("Total IO");
        Ok(())
    }
}
