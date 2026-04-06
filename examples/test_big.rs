use anyhow::{Ok, Result, bail};
use clap::Parser;
use parensnet_rs::anndata::AnnData;
use parensnet_rs::hist::bayesian_blocks_bin_edges;
use parensnet_rs::types::PNFloat;

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input data file
    data_file: std::path::PathBuf,
}

// #[test]
// pub fn test_anndata_big() -> Result<()> {
//     use super::bayesian_blocks_bin_edges;
//     use crate::anndata::AnnData;
//     let bg_file =
//         crate::test_data_file_path!("pbmc_scrna/800K/pbmc800K.20K.h5ad");
//     let adata = AnnData::new(bg_file, None)?;
//     for i in 0..5 {
//         let ax = adata.read_column_around::<f32>(i, 2)?;
//         let bins = bayesian_blocks_bin_edges(ax.view());
//         debug!("BINS :: {:?}", bins.len());
//     }
//     Ok(())
// }

fn run<T: hdf5::H5Type + PNFloat>(args: CLIArgs) -> Result<()> {
    let adata = AnnData::new(args.data_file.to_str().unwrap(), None, None)?;
    for i in 0..6 {
        let ax = adata.read_column_around::<T>(i, 2)?;
        let bins = bayesian_blocks_bin_edges(ax.view());
        log::debug!("BINS for {} :: {:?}", i, bins.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    match CLIArgs::try_parse() {
        Result::Ok(args) => match run::<f32>(args) {
            Err(err) => {
                log::error!("{}", err);
                bail!(err)
            }
            _ => {
                log::info!("Mege Completed");
                //Ok(())
            }
        },
        Err(err) => {
            log::error!("{}", err);
            let _ = err.print();
            bail!(err)
        }
    }
    Ok(())
}
