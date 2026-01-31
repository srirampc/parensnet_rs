use anyhow::Result;
use clap::Parser;
use parensnet_rs::{
    anndata::AnnData,
    comm::CommIfx,
    cond_error, cond_info,
    pucn::{WorkflowArgs, execute_workflow},
};
use thiserror::Error;

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML file with all the arguments
    config: std::path::PathBuf,
}

struct CLIInit {
    args: CLIArgs,
    mpi_ifx: CommIfx,
}

impl std::fmt::Display for CLIArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(config: {:?})", self.config.to_str())
    }
}

#[derive(Error, Debug)]
enum Error {
    #[error("Failed to read {0}")]
    InputReadError(String),
}

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

fn run(clid: CLIInit) -> Result<()> {
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
        return Err(anyhow::Error::from(err));
    }
    // Load Arguments
    let (wargs, adata) =
        match serde_saphyr::from_str::<WorkflowArgs>(&rstr?) {
            Ok(mut wargs) => {
                cond_info!(mcx.is_root(); "Parsed successfully: {:?}", wargs);
                cond_info!(mcx.is_root(); "Data H5AD : {}", wargs.h5ad_file);
                let adata = AnnData::new(
                    &wargs.h5ad_file,
                    Some("_index".to_string()),
                )?;
                wargs.update_dims(&[adata.nobs, adata.nvars]);
                (wargs, adata)
            }
            Err(err) => {
                cond_error!(mcx.is_root(); "Failed to parse YAML: {}", err);
                return Err(anyhow::Error::from(err));
            }
        };

    execute_workflow(mcx, &wargs, &adata)?;
    Ok(())
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
