use anyhow::{Ok, Result, bail};
use clap::Parser;
use ndarray::Array1;
use parensnet_rs::h5::io;
use parensnet_rs::types::PNFloat;

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML
    merge_io: std::path::PathBuf,
    /// Path to input files
    merge_files: Vec<std::path::PathBuf>,
}

//
fn get_sizes(args: &CLIArgs) -> Result<(usize, usize)> {
    //    - read
    let mut phist_size = 0usize;
    let mut pmi_size = 0usize;
    // TODO::
    // 1. For each file
    for fname in &args.merge_files {
        //log::info!("Reading {:?}", fname);
        let file = hdf5::File::open(fname)?;
        let ds = file.dataset("data/pair_hist")?;
        phist_size += ds.shape()[0];
        let ds = file.dataset("data/mi")?;
        pmi_size += ds.shape()[0];
        file.close()?;
    }
    log::info!("Total Sizes: MI: {}, HIST: {}", pmi_size, phist_size);
    Ok((pmi_size, phist_size))
}

fn read_data<T: hdf5::H5Type + PNFloat>(
    args: &CLIArgs,
) -> Result<(Array1<T>, Array1<T>)> {
    let (pmi_size, phist_size) = get_sizes(args)?;
    let mut mi = Array1::<T>::zeros(pmi_size);
    let mut pair_hist = Array1::<T>::zeros(phist_size);
    let mut mi_offset = 0usize;
    let mut phist_offset = 0usize;
    for fname in &args.merge_files {
        log::info!("Reading File {:?}", fname);
        let file = hdf5::File::open(fname)?;
        let ds = file.dataset("data/mi")?.read_1d()?;
        let nsize = ds.shape()[0];
        mi.slice_mut(ndarray::s![mi_offset..(mi_offset + nsize)])
            .assign(&ds);
        mi_offset += nsize;
        let ds = file.dataset("data/pair_hist")?.read_1d()?;
        let nsize = ds.shape()[0];
        pair_hist
            .slice_mut(ndarray::s![phist_offset..(phist_offset + nsize)])
            .assign(&ds);
        phist_offset += nsize;
    }
    log::info!(
        "Read data: MI: {}, HIST: {}",
        mi.shape()[0],
        pair_hist.shape()[0]
    );
    assert!(mi_offset == pmi_size);
    assert!(phist_offset == phist_size);
    Ok((mi, pair_hist))
}

fn run<T: hdf5::H5Type + PNFloat>(args: CLIArgs) -> Result<()> {
    log::info!("Merge CLI Args :: {:?}", args);
    let (mi, pair_hist) = read_data::<T>(&args)?;
    // 2. Read datasets
    let file = hdf5::File::open_rw(&args.merge_io)?;
    let data_group = file.group("data")?;
    io::write_1d(&data_group, "mi", &mi)?;
    log::info!("Wrote mi to :: {:?}", args.merge_io);
    io::write_1d(&data_group, "pair_hist", &pair_hist)?;
    log::info!("Writing completed :: {:?}", args.merge_io);
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
                Ok(())
            }
        },
        Err(err) => {
            log::error!("{}", err);
            let _ = err.print();
            bail!(err)
        }
    }
}
