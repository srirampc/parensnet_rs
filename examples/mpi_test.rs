use clap::Parser;
use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use parensnet_rs::comm::CommIfx;
use parensnet_rs::h5::mpio::{
    block_read1d, block_read2d, block_write1d, block_write2d, create_file,
    create_write2d,
};
use parensnet_rs::util::GenericError;
use parensnet_rs::util::vec::Vec2d;
use parensnet_rs::workflow::{collect_samples, generate_samples};
use parensnet_rs::{cond_error, cond_info, cond_println};
use serde::{Deserialize, Serialize};

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML file with all the arguments
    config: std::path::PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InArgs {
    // Mandatory Fileds
    //  - Files/Paths
    pub hdf_in: String,
    pub hdf_out: String,
    pub hdf_out2: String,
    pub nsamples: usize,
    pub nvars: usize,
    pub nrounds: usize,
}

fn print_data_dist(comm_ifx: &CommIfx, ldata: usize) {
    let ndata = comm_ifx.collect_counts(ldata);
    let tdata: usize = ndata.iter().sum();
    cond_println!(comm_ifx.is_root(); "ndata :: {:?} ; tdata: {}", ndata, tdata);
}

fn test_puc(comm_ifx: &CommIfx, wargs: &InArgs) {
    let idx_data: ndarray::Array2<i32> =
        block_read2d(comm_ifx, &wargs.hdf_in, "data/index").unwrap();
    print_data_dist(comm_ifx, idx_data.shape()[0]);
    let puc_data: ndarray::Array1<f32> =
        block_read1d(comm_ifx, &wargs.hdf_in, "data/puc").unwrap();
    print_data_dist(comm_ifx, puc_data.shape()[0]);

    let h_file = create_file(comm_ifx, &wargs.hdf_out2).unwrap();
    let h_group = h_file.create_group("data").unwrap();
    block_write2d(comm_ifx, &h_group, "index", &idx_data).unwrap();
    block_write1d(comm_ifx, &h_group, "puc", &puc_data).unwrap();
}

fn test_index(comm_ifx: &CommIfx, wargs: &InArgs) {
    let rdata: ndarray::Array2<i32> =
        block_read2d(comm_ifx, &wargs.hdf_in, "data/index").unwrap();
    print_data_dist(comm_ifx, rdata.shape()[0]);
    create_write2d(comm_ifx, &wargs.hdf_out, "data", "index", &rdata).unwrap();
}

fn parse_args(mcx: &CommIfx, args: &CLIArgs) -> Result<InArgs, GenericError> {
    match serde_saphyr::from_str::<InArgs>(&std::fs::read_to_string(
        &args.config,
    )?) {
        Ok(wargs) => {
            cond_info!(mcx.is_root(); "Parsed successfully: {:?}", wargs);
            Ok(wargs)
        }
        Err(err) => {
            cond_error!(mcx.is_root(); "Failed to parse YAML: {}", err);
            Err(GenericError::from(err))
        }
    }
}

fn test_h5(mcx: &CommIfx, args: &CLIArgs) -> Result<(), GenericError> {
    let wargs = parse_args(mcx, args)?;
    test_index(mcx, &wargs);
    mcx.comm.barrier();
    test_puc(mcx, &wargs);
    Ok(())
}

fn test_samples(
    mcx: &CommIfx,
    args: &CLIArgs,
) -> Result<Vec2d<usize>, GenericError> {
    let wargs = parse_args(mcx, args)?;
    let mut sample_arr = if mcx.is_root() {
        generate_samples(wargs.nvars, wargs.nrounds, wargs.nsamples)
    } else {
        vec![0usize; wargs.nrounds * wargs.nsamples]
    };
    let rootp = mcx.comm.process_at_rank(0);
    rootp.broadcast_into(&mut sample_arr);
    Ok(Vec2d::new(sample_arr, wargs.nsamples, wargs.nrounds))
}

fn run(mcx: &CommIfx, args: CLIArgs) -> Result<(), GenericError> {
    let wargs = parse_args(mcx, &args)?;
    let rtest = test_samples(mcx, &args)?;
    cond_println!(mcx.is_root();"RCTS: {:?}", rtest);
    let rtest = collect_samples(mcx, wargs.nvars, wargs.nrounds, wargs.nsamples)?;
    let rsum: usize = rtest.flatten().iter().sum();
    let rcounts = mcx.collect_counts(rsum);
    cond_println!(mcx.is_root();"RCTS: {:?}", rcounts);
    test_h5(mcx, &args)
}

fn main() {
    let comm_ifx = CommIfx::init();
    match CLIArgs::try_parse() {
        Ok(args) => {
            let _ = run(&comm_ifx, args);
        }
        Err(err) => {
            if comm_ifx.rank == 0 {
                let _ = err.print();
            };
        }
    };
}
