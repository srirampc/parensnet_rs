//! `gbgrn_cli` — MPI front-end for the gradient-boosted GRN workflow.
//!
//! Thin command-line wrapper that parses a single YAML config file
//! into a [`GBGRNArgs`] and dispatches over [`parensnet_rs::gbn::RunMode`]:
//!
//! * [`RunMode::GBCrossFoldValidation`] →
//!   [`run_cross_fold_gbm`] (CV step only, prints the resulting
//!   [`parensnet_rs::gbn::CVStats`]).
//! * [`RunMode::GBGRNet`] → [`infer_gb_network`] (optional CV step
//!   followed by the distributed gradient-boosting run and HDF5
//!   network output).
//!
//! Designed to be launched via `mpirun`/`srun`; each rank reads the
//! same config file and participates in the collective workflow.

#![allow(dead_code)]
use anyhow::Result;
use clap::Parser;
use sope::reduction::any_of;
use thiserror::Error;

use parensnet_rs::{
    comm::CommIfx,
    cond_error, cond_info,
    gbn::{GBGRNArgs, RunMode, infer_gb_network, run_cross_fold_gbm},
};

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML file with all the arguments
    config: std::path::PathBuf,
}

/// Bundle returned by [`cli_init`]: parsed CLI args plus the
/// initialised MPI communicator handle.
struct CLIInit {
    /// Parsed command-line arguments.
    args: CLIArgs,
    /// Initialised MPI communicator interface ([`CommIfx`]).
    mpi_ifx: CommIfx,
}

impl std::fmt::Display for CLIArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(config: {:?})", self.config)
    }
}

/// CLI-level errors.
#[derive(Error, Debug)]
enum Error {
    /// The config file path could not be read on at least one rank.
    /// Wraps the path as a string for the user-facing message.
    #[error("Failed to read {0}")]
    InputReadError(String),
}

/// Initialise the env logger, the MPI world, and parse the CLI
/// arguments. On parse failure the error is printed on rank 0 only
/// and then propagated so [`main`] can exit with clap's exit code.
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

/// Body of the binary: load the YAML config, parse it as a
/// [`GBGRNArgs`], and dispatch over its
/// [`GBGRNArgs::mode`](parensnet_rs::gbn::GBGRNArgs::mode) to
/// either [`infer_gb_network`] or [`run_cross_fold_gbm`].
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

    let gargs = match serde_saphyr::from_str::<GBGRNArgs>(&rstr?) {
        Ok(wargs) => {
            cond_info!(mcx.is_root(); "Parsed successfully: {:?}", wargs);
            cond_info!(mcx.is_root(); "Data H5AD : {}", wargs.h5ad_file);
            wargs
        }

        Err(err) => {
            cond_error!(mcx.is_root(); "Failed to parse YAML: {}", err);
            return Err(anyhow::Error::from(err));
        }
    };

    match gargs.mode {
        RunMode::GBGRNet => {
            infer_gb_network(&gargs, mcx)?;
        }
        RunMode::GBCrossFoldValidation => {
            run_cross_fold_gbm(&gargs, mcx)?;
        }
    };

    Ok(())
}

/// Process entry point. Initialises the CLI / MPI world via
/// [`cli_init`] and forwards to [`run`]. Clap parse errors short-
/// circuit through [`std::process::exit`] with clap's exit code so
/// `--help` / `--version` exit cleanly on every rank.
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
