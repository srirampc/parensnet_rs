//! MCPNet B-spline mutual-information workflow.
//!
//! This module drives the B-spline kernels ported from
//! [MCPNet](https://github.com/AluruLab/MCPNet) over [`AnnData`] expression
//! matrices in batched, MPI-distributed fashion. Two related stages
//! are supported, selected via [`RunMode`]:
//!
//! * [`RunMode::MIBSplineWeights`] — compute the per-cell B-spline
//!   weight matrix (one row per variable) and saved it to HDF5.
//! * [`RunMode::MIBSpline`] — compute the upper-triangular pairwise
//!   mutual-information matrix from the weight matrix HDF5 file,
//!   and write the `(index, mi)` pairs to HDF5.
//!
//! # Submodules
//! * [`args`] — YAML/TOML-deserializable configuration
//!   ([`WorkflowArgs`], [`RunMode`]).
//! * [`miw`] — the [`miw::MIWorkFlow`] struct, its
//!   [`miw::MIWorkFlowHelper`], and the per-rank `run_bspline_*`
//!   methods that actually invoke
//!   [`crate::corr::mi::bspline_weights`] /
//!   [`crate::corr::mi::bspline_mi`].
//!
//! # Public surface
//! * [`WorkflowArgs`] / [`RunMode`] re-exported from [`args`].
//! * [`execute_workflow`] — entry point used by the binary
//!   front-end; iterates over [`WorkflowArgs::mode`] and dispatches
//!   each stage to a [`miw::MIWorkFlow`].

mod args;

pub use self::args::{RunMode, WorkflowArgs};
mod miw;

use anyhow::Result;
use sope::timer::CumulativeTimer;
use crate::{anndata::AnnData, comm::CommIfx, mcpn::miw::MIWorkFlow, util::PairWorkDistributor};

/// Run every stage listed in `args.mode` on the supplied MPI
/// communicator.
///
/// Builds a single [`PairWorkDistributor`] from `args.nvars` /
/// `args.npairs`. For each [`RunMode`] variant  the matching `run_*` method
/// is invoked from [`miw::MIWorkFlow`] object:
///
/// * [`RunMode::MIBSplineWeights`] →
///   [`miw::MIWorkFlow::run_bspline_weights`].
/// * [`RunMode::MIBSpline`] →
///   [`miw::MIWorkFlow::run_bspline_mi`].
///
/// # Errors
/// Propagates any error returned by [`AnnData::new`] or the
/// underlying workflow (HDF5 IO, MPI collectives, ...).
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
            RunMode::MIBSpline |  RunMode::MIBSplineWeights => {
                let adata = AnnData::new(
                    &args.h5ad_file,
                    None,
                    args.row_major_h5_file.clone(),
                )?;

                let miwf  = MIWorkFlow {
                    args,
                    adata: &adata,
                    wf_dist: &wdistr,
                    comm_ifx: mpi_ifx,
                    io_timer: CumulativeTimer::from_comm(mpi_ifx.comm(), ","),
                };

                match rmode {
                    RunMode::MIBSplineWeights => miwf.run_bspline_weights()?,
                    RunMode::MIBSpline => miwf.run_bspline_mi()?,
                }
            }
        }
    }

    Ok(())
}
