#![allow(dead_code)]
use clap::Parser;
use parensnet_rs::comm;
use parensnet_rs::util::GenericError;
use parensnet_rs::workflow::{self};
use parensnet_rs::{cond_error, cond_info};
use std::fmt;

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML file with all the arguments
    config: std::path::PathBuf,
}

struct CLIInit {
    args: CLIArgs,
    mpi_ifx: comm::CommIfx,
}

impl fmt::Display for CLIArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(config: {})", self.config.to_str().unwrap())
    }
}

#[derive(Debug)]
enum Error {
    InputReadError(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputReadError(in_file) => {
                write!(f, "Failed to write {}", in_file)
            }
        }
    }
}

impl std::error::Error for Error {}

fn cli_init() -> Result<CLIInit, GenericError> {
    env_logger::try_init()?;
    let mpi_ifx = comm::CommIfx::init();
    // Parse command line arguments
    match CLIArgs::try_parse() {
        Ok(args) => Ok(CLIInit { args, mpi_ifx }),
        Err(err) => {
            if mpi_ifx.rank == 0 {
                let _ = err.print();
            };
            Err(GenericError::from(err))
        }
    }
}

fn run(clid: CLIInit) -> Result<(), GenericError> {
    let mcx = &clid.mpi_ifx;
    cond_info!(mcx.is_root(); "Command Line Arguments : {}", clid.args);
    // Read input fail
    let mut in_flag: i32 = 1;
    let rstr = std::fs::read_to_string(&clid.args.config);
    if rstr.is_err() {
        in_flag = 0;
        log::error!(
            "RANK {} :: Failed to read input: {:?}",
            mcx.rank,
            rstr.as_ref().err()
        );
    }
    let rsum: i32 = mcx.collect_counts(in_flag).iter().sum();
    if rsum < mcx.size {
        let errv =
            format!("Failed to read input: {}", clid.args.config.display());
        cond_error!(mcx.is_root(); "{}", errv);
        let err = Error::InputReadError(String::from("hello"));
        return Err(GenericError::from(err));
    }
    // Load Arguments
    let wargs = match serde_saphyr::from_str::<workflow::WorkflowArgs>(&rstr?) {
        Ok(mut wargs) => {
            cond_info!(mcx.is_root(); "Parsed successfully: {:?}", wargs);
            cond_info!(mcx.is_root(); "Data H5AD : {}", wargs.h5ad_file);
            wargs.update()?;
            wargs
        }
        Err(err) => {
            cond_error!(mcx.is_root(); "Failed to parse YAML: {}", err);
            return Err(GenericError::from(err));
        }
    };

    workflow::execute_workflow(mcx, &wargs)?;
    Ok(())
}

fn main() -> Result<(), GenericError> {
    match cli_init() {
        Ok(clid) => run(clid),
        Err(err) => {
            if let Some(clerr) = err.downcast_ref::<clap::Error>() {
                std::process::exit(clerr.exit_code())
            }
            Err(GenericError::from(err))
        }
    }
}
