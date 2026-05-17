//! `mcpgrn_cli` — MPI front-end for the MCPNet MI workflow.
//!
//! Thin command-line wrapper that parses a single YAML config file
//! into a [`WorkflowArgs`], fills in the AnnData matrix dimensions, 
//! and dispatches each stage
//! in [`parensnet_rs::mcpn::WorkflowArgs::mode`]
//! ([`parensnet_rs::mcpn::RunMode::MIBSplineWeights`] /
//! [`parensnet_rs::mcpn::RunMode::MIBSpline`]) through
//! [`parensnet_rs::mcpn::execute_workflow`].
//!
//! Designed to be launched via `mpirun`/`srun`; each rank reads the
//! same config file and participates in the collective workflow.

use anyhow::Result;
use clap::Parser;
use parensnet_rs::{
    anndata::xds_dimensions,
    comm::CommIfx,
    cond_error, cond_info,
    mcpn::{WorkflowArgs, execute_workflow},
};
use sope::reduction::any_of;
use thiserror::Error;

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML file with all the arguments
    config: std::path::PathBuf,
}

impl std::fmt::Display for CLIArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(config: {:?})", self.config.to_str())
    }
}

/// Bundles: parsed CLI args plus the initialised MPI communicator handle.
struct CLIInit {
    /// Parsed command-line arguments.
    args: CLIArgs,
    /// Initialised MPI communicator interface ([`CommIfx`]).
    mpi_ifx: CommIfx,
}

/// CLI-level errors.
#[derive(Error, Debug)]
enum Error {
    /// The config file path could not be read on at least one rank.
    #[error("Failed to read {0}")]
    InputReadError(String),
}

/// Initialise the env logger, the MPI world, and parse the CLI arguments.
fn cli_init() -> Result<CLIInit> {
    env_logger::try_init()?;
    let mpi_ifx = CommIfx::init();
    // Parse command line arguments
    match CLIArgs::try_parse() {
        Ok(args) => Ok(CLIInit { args, mpi_ifx }),
        Err(err) => {
            if mpi_ifx.rank == 0 {
                let _ = err.print();
            };
            Err(anyhow::Error::from(err))
        }
    }
}

/// Loads the YAML config, fills in the AnnData dimensions and 
/// calls [`parensnet_rs::mcpn::execute_workflow`].
///
/// Read failures are detected on every rank and combined with
/// [`any_of`] so the program aborts collectively when at least one
/// rank could not open the file.
fn run(clid: CLIInit) -> Result<()> {
    let mcx = &clid.mpi_ifx;
    cond_info!(mcx.is_root(); "Command Line Arguments : {}", clid.args);
    // Read input fail
    let rstr = std::fs::read_to_string(&clid.args.config);
    if rstr.is_err() {
        log::error!(
            "RANK {} :: Failed to read input: {:?}",
            mcx.rank,
            rstr.as_ref().err()
        );
    }
    if any_of(rstr.is_err(), clid.mpi_ifx.comm()) {
        let errv =
            format!("Failed to read input: {}", clid.args.config.display());
        cond_error!(mcx.is_root(); "{}", errv);
        let err = Error::InputReadError(String::from("hello"));
        return Err(anyhow::Error::from(err));
    }
    // Load Arguments
    let wargs = match serde_saphyr::from_str::<WorkflowArgs>(&rstr?) {
        Ok(mut wargs) => {
            cond_info!(mcx.is_root(); "Parsed successfully: {:?}", wargs);
            cond_info!(mcx.is_root(); "Data H5AD : {}", wargs.h5ad_file);
            let (nobs, nvars) = xds_dimensions(&wargs.h5ad_file)?;
            wargs.update_dims(&[nobs, nvars]);
            wargs
        }
        Err(err) => {
            cond_error!(mcx.is_root(); "Failed to parse YAML: {}", err);
            return Err(anyhow::Error::from(err));
        }
    };

    execute_workflow(mcx, &wargs)
}

fn main() -> Result<()> {
    match cli_init() {
        Ok(clid) => run(clid),
        Err(err) => {
            if let Some(clerr) = err.downcast_ref::<clap::Error>() {
                std::process::exit(clerr.exit_code())
            }
            Err(err)
        }
    }
}
