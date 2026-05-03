mod args;

pub use self::args::{RunMode, WorkflowArgs};
mod miw;

use anyhow::Result;
use sope::timer::CumulativeTimer;
use crate::{anndata::AnnData, comm::CommIfx, mcpn::miw::MIWorkFlow, util::PairWorkDistributor};
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
