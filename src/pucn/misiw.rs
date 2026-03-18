use std::path::Path;

use anyhow::{Ok, Result};
use mpi::traits::CommunicatorCollectives;
use sope::{
    reduction::{all_of, allreduce_sum},
    timer::SectionTimer,
};

use super::{
    MISIWorkFlowTrait, WorkDistributor, WorkflowArgs,
    ds::{NodeCollection, OrdPairSICollection, PairMICollection},
    helpers::MISIWorkFlowHelper,
};
use crate::{anndata::AnnData, comm::CommIfx, cond_info};

pub struct MISIWorkFlow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
    pub adata: &'a AnnData,
}

impl<'a> MISIWorkFlowTrait<'a> for MISIWorkFlow<'a> {
    fn comm_ifx(&self) -> &'a CommIfx {
        self.mpi_ifx
    }

    fn wf_dist(&self) -> &'a WorkDistributor {
        self.wdistr
    }

    fn wf_args(&self) -> &'a WorkflowArgs {
        self.args
    }

    fn ann_data(&self) -> &'a AnnData {
        self.adata
    }
}

impl<'a> MISIWorkFlow<'a> {
    fn save_distribute(
        &self,
        nodes: NodeCollection<i64, i32, f32>,
        npairs_si: OrdPairSICollection<i32, f32>,
        npairs_mi: PairMICollection<i32, f32>,
    ) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
        let npairs_mi = npairs_mi
            .distribute(self.mpi_ifx, nodes.hist_dim.as_slice().unwrap())?;
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

    pub fn run(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
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
        self.save_distribute(nodes, npairs_si, npairs_mi)
    }

    pub fn run_hist_nodes(&self) -> Result<()> {
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
        if self.mpi_ifx.rank == 0 {
            HelperT::write_nodes_h5(self, &nodes, &self.args.hist_data_file)?;
        }
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Write Nodes");
            cond_info!(self.mpi_ifx.is_root(); "Completed Writing Nodes");
            s_timer.reset();
        }
        Ok(())
    }

    pub fn run_hist(&self) -> Result<()> {
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
        Ok(())
    }

    fn run_misi_dist_from_hist(
        &self,
        nodes: NodeCollection<i64, i32, f32>,
        hist_pairs: PairMICollection<i32, f32>,
    ) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
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

    pub fn run_misi_dist(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        cond_info!(self.mpi_ifx.is_root(); "Starting MISIWorkFlow::Run MISI Dist");
        assert!(!self.args.lmr_only);
        // 1. Flat loading of  all the nodes
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
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
        Ok(())
    }

    pub fn run_misi_dist_from_nodes(&self) -> Result<()> {
        type HelperT = MISIWorkFlowHelper<i64, i32, f32>;
        cond_info!(self.mpi_ifx.is_root(); "Starting MISIWorkFlow::Run MISI Dist from Nodes");
        // 1. Flat loading of  all the nodes
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");
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
        Ok(())
    }
}
