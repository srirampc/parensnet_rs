#![allow(dead_code)]
use anyhow::Result;
use clap::Parser;
use mpi::traits::CommunicatorCollectives;
use thiserror::Error;

use parensnet_rs::{
    anndata::{AnnData, GeneSetAD},
    comm::CommIfx,
    cond_info,
    gbn::{
        CVConfig, GBMParams, mpi_gradient_boosting_grn,
        mpi_optimal_gbm_iterations,
    },
};

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input H5AD file with all the arguments
    h5ad_file: std::path::PathBuf,
    /// Path to input H5AD file with all the arguments
    tf_file: std::path::PathBuf,
    /// Path to output H5 file with all the arguments
    out_file: std::path::PathBuf,
}

struct CLIInit {
    args: CLIArgs,
    mpi_ifx: CommIfx,
}

impl std::fmt::Display for CLIArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(h5ad_file: {:?}, tf_file: {:?}, out_file: {:?})",
            self.h5ad_file.to_str(),
            self.tf_file.to_str(),
            self.out_file.to_str()
        )
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
    let ndecimals: usize = 3;
    let mcx = &clid.mpi_ifx;
    let (ad_file, tf_csv) = (
        clid.args
            .h5ad_file
            .to_str()
            .ok_or(Error::InputReadError("H5AD".to_string()))?
            .to_owned(),
        clid.args
            .tf_file
            .to_str()
            .ok_or(Error::InputReadError("H5AD".to_string()))?
            .to_owned(),
    );

    cond_info!(mcx.is_root(); "Data H5AD : {}", ad_file);
    let adata = AnnData::new(&ad_file, Some("_index".to_string()))?;
    let params: GBMParams = GBMParams {
        num_threads: 4,
        verbose: 0,
        ..Default::default()
    };

    let config = CVConfig {
        n_sample_genes: 100,
        params: params.clone(),
        ..Default::default()
    };

    cond_info!(mcx.is_root(); "TF File  : {}", tf_csv);
    let tf_set = GeneSetAD::new(&adata, tf_csv.as_str(), None, Some(ndecimals))?;
    cond_info!(mcx.is_root(); "TF Set   : {:?}", tf_set.len());
    cond_info!(mcx.is_root(); "CV Config : {:?}", config);
    let opt_args = mpi_optimal_gbm_iterations(&tf_set, &config, mcx)?;
    if clid.mpi_ifx.is_root() {
        opt_args.print();
    }

    cond_info!(mcx.is_root(); "Optimal Median: {} ", opt_args.median);
    mcx.comm().barrier();
    cond_info!(mcx.is_root(); " START GRAD BOOSTING", );
    let params = GBMParams {
        num_iterations: opt_args.median,
        ..params
    };
    let net_edges = mpi_gradient_boosting_grn(&tf_set, mcx, params, true)?;
    let nedges = sope::reduction::allreduce_sum(&net_edges.len(), mcx.comm());
    cond_info!(mcx.is_root(); "NET EDGES: {} ", nedges);
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
