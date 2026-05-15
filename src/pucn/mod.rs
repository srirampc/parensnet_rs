//! PUC / MI-SI gene-network construction workflows.
//!
//! This module exposes teh function, [`execute_workflow`],
//! which dispatches over the ordered list of stages declared in
//! [`WorkflowArgs::mode`] and runs the matching workflow struct from
//! one of the submodules.
//!
//! # Submodules
//! * [`args`] — YAML/TOML-deserializable configuration types
//!   ([`WorkflowArgs`], [`RunMode`], `LogLevel`).
//! * [`ds`] — flattened, MPI-friendly data containers
//!   ([`crate::pucn::ds::NodeCollection`],
//!   [`crate::pucn::ds::PairMICollection`],
//!   [`crate::pucn::ds::OrdPairSICollection`], etc.).
//! * [`helpers`] — stateless `MISIWorkFlowHelper` that implements the
//!   per-stage primitives (node construction, pair batching, HDF5
//!   writers) used by the workflows.
//! * [`misiw`] — the [`misiw::MISIWorkFlow`] struct and its `run_*`
//!   methods covering the [`RunMode::MISI`], [`RunMode::MISIDist`],
//!   [`RunMode::HistDist`], [`RunMode::HistNodes`] and
//!   [`RunMode::HistNodes2MISI`] stages.
//! * [`puc`] — sampled-PUC and LMR-PUC workflows
//!   ([`puc::SampledPUCWorkflow`], [`puc::LMRPUCWorkflow`]).
//! * [`puc_dist`] — distributed LMR-PUC variant
//!   ([`puc_dist::PUCDistWorkflow`]).
//!
//! # Public API
//! * [`WorkflowArgs`] / [`RunMode`] — re-exported from [`args`].
//! * [`WorkflowError`] — error type returned from the helper functions 
//!   in this module.
//! * [`execute_workflow`] — the dispatcher used by front-ends.
//! * [`generate_samples`] / [`generate_random_samples`] /
//!   [`collect_samples`] — sampling utilities shared by the sampled-
//!   PUC workflows.
//!
//! # Internal traits
//! * [`MISIWorkFlowTrait`] — view-only accessor trait implemented by
//!   [`misiw::MISIWorkFlow`] and consumed by
//!   [`crate::pucn::helpers::MISIWorkFlowHelper`].
//! * [`PUCWorkFlowTrait`] — analogous accessor trait for the PUC-only
//!   workflows in [`puc`].

#![allow(dead_code)]
mod args;
mod ds;
mod helpers;
mod misiw;
mod puc;
mod puc_dist;
pub use self::args::{RunMode, WorkflowArgs};

use crate::{
    anndata::AnnData,
    comm::CommIfx,
    util::{PairWorkDistributor, Vec2d, IdVResults},
};
use anyhow::{Result, bail};
use mpi::traits::{Communicator, Root};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use num::{FromPrimitive, Integer};
use sope::timer::CumulativeTimer;
use thiserror::Error;

/// Errors produced by the helpers in this module.
#[derive(Error, Debug)]
pub enum WorkflowError {
    /// Invalid sample config : Either `nrounds` or `nsamples` is zero.
    #[error("Invalid No. of Samples: (nvars:{0}, nrounds:{1}, nsamples:{2})")]
    InvalidSamples(usize, usize, usize),
}

/// View-only accessor trait implemented by the MI/SI workflow struct
/// ([`misiw::MISIWorkFlow`]).
pub(super) trait MISIWorkFlowTrait<'a> {
    /// MPI communicator interface (rank/size + `comm()`).
    fn comm_ifx(&self) -> &'a CommIfx;
    /// 2-D pair work distribution used to drive the per-batch loops.
    fn wf_dist(&self) -> &'a PairWorkDistributor;
    /// Parsed configuration ([`WorkflowArgs`]).
    fn wf_args(&self) -> &'a WorkflowArgs;
    /// AnnData handle providing access to the expression matrix.
    fn ann_data(&self) -> &'a AnnData;
    /// Cumulative timer used to accumulate IO time across stages.
    fn io_timer(&self) -> &CumulativeTimer<'a>;
    /// When `true`, the workflow emits per-stage debug-level logging.
    /// Defaults to `false`.
    fn detailed_log(&self) -> bool {
        false
    }
}

/// View-only accessor trait implemented by the PUC-only workflows in
/// [`puc`] / [`puc_dist`].
pub(super) trait PUCWorkFlowTrait<'a> {
    /// MPI communicator interface (rank/size + `comm()`).
    fn comm_ifx(&self) -> &'a CommIfx;
    /// Pair work distribution used to drive the per-rank loops.
    fn wf_dist(&self) -> &'a PairWorkDistributor;
    /// Parsed configuration ([`WorkflowArgs`]).
    fn wf_args(&self) -> &'a WorkflowArgs;

    /// When `true`, the workflow emits per-stage debug-level logging.
    /// Defaults to `false`.
    fn detailed_log(&self) -> bool {
        false
    }
}

/// Draw without replacement `nrounds * nsamples` variable indices from 
/// `0..nvars` and return them in row-major order
/// (`row` indexes the round, `col` indexes the sample within the round).
pub fn generate_samples(
    nvars: usize,
    nrounds: usize,
    nsamples: usize,
) -> Vec<usize> {
    let mut sid_vec = vec![0usize; nrounds * nsamples];
    let var_ids = Array1::<usize>::from_iter(0..nvars);
    for row in 0..nrounds {
        let row_samples = var_ids.sample_axis(
            Axis(0),
            nsamples,
            SamplingStrategy::WithoutReplacement,
        );
        for (col, val) in row_samples.iter().enumerate() {
            sid_vec[row * nsamples + col] = *val;
        }
    }
    sid_vec
}

/// Variant of [`generate_samples`] returning an `(nsamples, nrounds)`
/// [`Array2`] where each column is one independent
/// without-replacement draw of `nsamples` indices from `0..nvars`.
pub fn generate_random_samples(
    nvars: usize,
    nrounds: usize,
    nsamples: usize,
) -> Array2<usize> {
    let idarr = Array1::<usize>::from_iter(0..nvars);
    let mut sample_arr = Array2::<usize>::from_elem((nsamples, nrounds), 0);
    for col in 0..nrounds {
        let ridx = idarr.sample_axis(
            Axis(0),
            nsamples,
            SamplingStrategy::WithoutReplacement,
        );
        let mut sample_col = sample_arr.column_mut(col);
        sample_col.assign(&ridx);
    }
    sample_arr
}

/// Generate the per-round sample matrix on the root rank, broadcast
/// it to every rank, and wrap the result in a [`Vec2d`] of element
/// type `T`.
///
/// # Errors
/// Returns [`WorkflowError::InvalidSamples`] when `nrounds == 0` or
/// `nsamples == 0`.
pub fn collect_samples<T: Integer + Clone + FromPrimitive>(
    mcx: &CommIfx,
    nvars: usize,
    nrounds: usize,
    nsamples: usize,
) -> Result<Vec2d<T>> {
    if nrounds == 0 || nsamples == 0 {
        bail!(WorkflowError::InvalidSamples(nvars, nrounds, nsamples,));
    }
    let mut sample_arr = if mcx.is_root() {
        generate_samples(nvars, nrounds, nsamples)
    } else {
        vec![0usize; nrounds * nsamples]
    };
    let rootp = mcx.comm().process_at_rank(0);
    rootp.broadcast_into(&mut sample_arr);
    let tsamples = sample_arr
        .iter()
        .map(|val| T::from_usize(*val).unwrap())
        .collect();
    Ok(Vec2d::new(tsamples, nsamples, nrounds))
}

/// Run every stage listed in `args.mode`, in order, on the supplied
/// MPI communicator.
///
/// Builds a single [`PairWorkDistributor`] from `args.nvars` /
/// `args.npairs` and reuses it for every stage. Each [`RunMode`]
/// variant is dispatched as follows:
///
/// * [`RunMode::SamplesRanges`] → [`puc::SampledPUCWorkflow::run`].
/// * [`RunMode::PUCLMR`] → [`puc::LMRPUCWorkflow::run`].
/// * [`RunMode::PUCLMRDist`] → [`puc_dist::PUCDistWorkflow::run`].
/// * [`RunMode::MISI`] / [`RunMode::MISIDist`] /
///   [`RunMode::HistDist`] / [`RunMode::HistNodes`] /
///   [`RunMode::HistNodes2MISI`] → open the AnnData file, build a
///   [`misiw::MISIWorkFlow`], and call the matching `run_*` method
///   ([`misiw::MISIWorkFlow::run`],
///   [`misiw::MISIWorkFlow::run_misi_dist`],
///   [`misiw::MISIWorkFlow::run_hist`],
///   [`misiw::MISIWorkFlow::run_hist_nodes`], or
///   [`misiw::MISIWorkFlow::run_misi_dist_from_nodes`]).
///
/// Other [`RunMode`] variants currently fall through to a `todo!`
/// and will panic if requested.
///
/// # Errors
/// Propagates any error returned by the underlying workflow
/// (`AnnData` construction, IO, MPI collectives, ...).
pub fn execute_workflow(mpi_ifx: &CommIfx, args: &WorkflowArgs) -> Result<()> {
    // Compute Distributions
    let wdistr = PairWorkDistributor::new(
        args.nvars,
        args.npairs,
        mpi_ifx.rank,
        mpi_ifx.size,
    );

    for rmode in &args.mode {
        match rmode {
            RunMode::SamplesRanges => {
                let spuc = puc::SampledPUCWorkflow {
                    args,
                    wdistr: &wdistr,
                    mpi_ifx,
                };
                spuc.run()?;
            }
            RunMode::PUCLMR => {
                let lpuc = puc::LMRPUCWorkflow {
                    args,
                    wdistr: &wdistr,
                    mpi_ifx,
                };
                lpuc.run()?;
            }
            RunMode::PUCLMRDist => {
                let lpuc = puc_dist::PUCDistWorkflow { args, mpi_ifx };
                lpuc.run()?;
            }
            RunMode::MISI
            | RunMode::MISIDist
            | RunMode::HistDist
            | RunMode::HistNodes
            | RunMode::HistNodes2MISI => {
                let adata = AnnData::new(
                    &args.h5ad_file,
                    None,
                    args.row_major_h5_file.clone(),
                )?;
                let rmisi = misiw::MISIWorkFlow {
                    args,
                    adata: &adata,
                    wdistr: &wdistr,
                    mpi_ifx,
                    io_timer: CumulativeTimer::from_comm(mpi_ifx.comm(), ","),
                };
                match rmode {
                    RunMode::MISI => rmisi.run()?,
                    RunMode::MISIDist => rmisi.run_misi_dist()?,
                    RunMode::HistDist => rmisi.run_hist()?,
                    RunMode::HistNodes => rmisi.run_hist_nodes()?,
                    RunMode::HistNodes2MISI => {
                        rmisi.run_misi_dist_from_nodes()?
                    }
                    _ => todo!("Missing mode"),
                }
            }
            _ => todo!("Mode {:?} Not Completed Yet", rmode),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_config_file_path;
    use crate::types::{DiscretizerMethod, LogBase};
    use crate::util::read_file_to_string;
    use log::{debug, error, info};

    macro_rules! parse_wflow_cfg {
        ($name:literal) => {
            serde_saphyr::from_str::<WorkflowArgs>(
                &(read_file_to_string(test_config_file_path!($name))?),
            )
        };
    }

    #[test]
    pub fn test_parse_workflow_args() -> Result<()> {
        crate::tests::log_init();
        match parse_wflow_cfg!("/pucn/pbmc20k_500_lpuc.yml") {
            Ok(wargs) => {
                info!("Parsed successfully: {:?}", wargs);
                assert_eq!(wargs.disc_method, DiscretizerMethod::BayesianBlocks);
                assert_eq!(wargs.tbase, LogBase::Natural);
            }
            Err(e) => {
                error!("Failed to parse YAML: {}", e);
            }
        };

        match parse_wflow_cfg!("/pucn/pbmc20k_500_S6x4p.yml") {
            Ok(p_wargs) => {
                info!("Parsed successfully: {:?}", p_wargs);
                assert_eq!(p_wargs.mode, vec![RunMode::SamplesLMRRanges]);
            }
            Err(e) => {
                error!("Failed to parse YAML: {}", e);
            }
        }
        Ok(())
    }

    pub fn test_parse_workflow_distr() -> Result<()> {
        let wargs = parse_wflow_cfg!("pbmc20k_500_S6x4p.yml").unwrap();
        let wdistr = PairWorkDistributor::new(wargs.nvars, wargs.npairs, 2, 16);
        debug!("Work Distribution : {:?}", wdistr.pair_blocks());
        Ok(())
    }

    #[test]
    pub fn test_sample() {
        crate::tests::log_init();
        let samples = super::generate_random_samples(500, 4, 6);
        debug!("Samples {:?}", samples);
        let samples2 = super::generate_samples(500, 4, 6);
        debug!("Samples {:?}", samples2);
    }
}
