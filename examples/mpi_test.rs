use anyhow::Result;
use clap::Parser;
use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use ndarray::{Array1, Array2};
use parensnet_rs::{
    anndata::AnnData,
    comm::CommIfx,
    cond_error, cond_info, cond_println,
    h5::{
        io,
        mpio::{
            block_read1d, block_read2d, block_write1d, block_write2d,
            create_file, create_write2d, read2d_row_slice, read2d_slice_of_rows,
        },
    },
    pucn::{collect_samples, generate_samples},
    util::{
        PairWorkDistributor, Vec2d, block_owner, exc_prefix_sum,
        triu_pair_to_index,
    },
};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::iter::zip;

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML file with all the arguments
    config: std::path::PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Test {
    #[serde(alias = "h5")]
    H5,
    #[serde(alias = "samples")]
    Samples,
    #[serde(alias = "pair_dist")]
    PairDist,
    #[serde(alias = "slice_reads")]
    SliceReads,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InArgs {
    // Mandatory Fileds
    //  - Files/Paths
    pub hdf_in: String,
    pub hdf_out: String,
    pub hdf_out2: String,
    pub misi_file: String,
    pub nsamples: usize,
    pub nvars: usize,
    pub nrounds: usize,

    #[serde(default)]
    pub tests: Vec<Test>,
}

const ALL_TESTS: [Test; 4] =
    [Test::H5, Test::Samples, Test::PairDist, Test::SliceReads];

/// Number of rows each rank writes in the slice-read test dataset.
const SLICE_NROWS_PER_RANK: usize = 5;
/// Number of columns of the slice-read test dataset.
const SLICE_NCOLS: usize = 7;

fn print_data_dist(comm_ifx: &CommIfx, ldata: usize) {
    let ndata = comm_ifx.collect_counts(ldata);
    let tdata: usize = ndata.iter().sum();
    cond_println!(comm_ifx.is_root(); "ndata :: {:?} ; tdata: {}", ndata, tdata);
}

fn test_read_puc(
    comm_ifx: &CommIfx,
    wargs: &InArgs,
) -> (Array2<i32>, Array1<f32>) {
    let idx_data: Array2<i32> =
        block_read2d(comm_ifx, &wargs.hdf_in, "data/index", None).unwrap();
    print_data_dist(comm_ifx, idx_data.shape()[0]);
    let puc_data: Array1<f32> =
        block_read1d(comm_ifx, &wargs.hdf_in, "data/puc", None).unwrap();
    print_data_dist(comm_ifx, puc_data.shape()[0]);

    (idx_data, puc_data)
}

fn test_puc(comm_ifx: &CommIfx, wargs: &InArgs) {
    let (idx_data, puc_data) = test_read_puc(comm_ifx, wargs);
    let h_file = create_file(comm_ifx, &wargs.hdf_out2).unwrap();
    let h_group = h_file.create_group("data").unwrap();
    block_write2d(comm_ifx, &h_group, "index", &idx_data).unwrap();
    block_write1d(comm_ifx, &h_group, "puc", &puc_data).unwrap();
}

fn test_index(comm_ifx: &CommIfx, wargs: &InArgs) {
    let rdata: Array2<i32> =
        block_read2d(comm_ifx, &wargs.hdf_in, "data/index", None).unwrap();
    print_data_dist(comm_ifx, rdata.shape()[0]);
    create_write2d(comm_ifx, &wargs.hdf_out, "data", "index", &rdata).unwrap();
}

fn parse_args(mcx: &CommIfx, args: &CLIArgs) -> Result<InArgs> {
    match serde_saphyr::from_str::<InArgs>(&std::fs::read_to_string(
        &args.config,
    )?) {
        Ok(wargs) => {
            cond_info!(mcx.is_root(); "Parsed successfully: {:?}", wargs);
            Ok(wargs)
        }
        Err(err) => {
            cond_error!(mcx.is_root(); "Failed to parse YAML: {}", err);
            Err(anyhow::Error::from(err))
        }
    }
}

fn test_h5(mcx: &CommIfx, args: &CLIArgs) -> Result<()> {
    let wargs = parse_args(mcx, args)?;
    test_index(mcx, &wargs);
    mcx.comm().barrier();
    test_puc(mcx, &wargs);
    Ok(())
}

fn gen_collect_samples(mcx: &CommIfx, args: &CLIArgs) -> Result<Vec2d<usize>> {
    let wargs = parse_args(mcx, args)?;
    let mut sample_arr = if mcx.is_root() {
        generate_samples(wargs.nvars, wargs.nrounds, wargs.nsamples)
    } else {
        vec![0usize; wargs.nrounds * wargs.nsamples]
    };
    let rootp = mcx.comm().process_at_rank(0);
    rootp.broadcast_into(&mut sample_arr);
    Ok(Vec2d::new(sample_arr, wargs.nsamples, wargs.nrounds))
}

fn test_samples(mcx: &CommIfx, args: &CLIArgs, wargs: &InArgs) -> Result<()> {
    let rtest = gen_collect_samples(mcx, args)?;
    cond_println!(mcx.is_root();"RCTS: {:?}", rtest);
    let rtest = collect_samples(mcx, wargs.nvars, wargs.nrounds, wargs.nsamples)?;
    let rsum: usize = rtest.flatten().iter().sum();
    let rcounts = mcx.collect_counts(rsum);
    cond_println!(mcx.is_root();"RCTS: {:?}", rcounts);
    Ok(())
}

fn test_blocks_2d(
    mcx: &CommIfx,
    nvars: usize,
    npairs: usize,
    hist_dim: &Array1<usize>,
    wdistr: PairWorkDistributor,
) -> Result<()> {
    let blocks2d = wdistr.pairs_2d();
    let nbatches = blocks2d.num_batches();
    let mut pairs: Vec<(usize, usize, usize)> = (0..nbatches)
        .flat_map(|bidx| {
            let (rows, cols) = blocks2d.batch_range(bidx, mcx.rank);
            rows.clone().flat_map(|rid| {
                cols.clone().filter_map(move |cid| {
                    if rid < cid {
                        let pidx = triu_pair_to_index(nvars, rid, cid);
                        Some((pidx, rid, cid))
                    } else {
                        None
                    }
                })
            })
        })
        .collect();
    pairs.sort();

    let mut p_ranks: Vec<i32> = pairs
        .iter()
        .map(|x| block_owner(x.0, mcx.size, npairs))
        .collect();
    let is_ranks_sorted: bool = p_ranks.is_sorted();
    p_ranks.dedup();

    let np = mcx.size as usize;
    let nsi: usize = hist_dim.iter().sum::<usize>() * nvars;
    let nord_pairs: usize = nvars * nvars;
    let si_starts: Vec<usize> = exc_prefix_sum(hist_dim.iter().cloned(), nvars);
    let (snd_pairs, snd_tabs, snd_si0, snd_si1, snd_ord) = pairs.iter().fold(
        (
            vec![0usize; np],
            vec![0usize; np],
            vec![0usize; np],
            vec![0usize; np],
            vec![0usize; np],
        ),
        |mut vx, idx| {
            let p_own = block_owner(idx.0, mcx.size, npairs);
            vx.0[p_own as usize] += 1;
            vx.1[p_own as usize] += hist_dim[idx.1] * hist_dim[idx.2];
            let rdx = si_starts[idx.1] + (idx.2 * hist_dim[idx.1]);
            let p_own = block_owner(rdx, mcx.size, nsi);
            vx.2[p_own as usize] += hist_dim[idx.1];
            let sdx = si_starts[idx.2] + (idx.1 * hist_dim[idx.2]);
            let p_own = block_owner(sdx, mcx.size, nsi);
            vx.3[p_own as usize] += hist_dim[idx.2];
            let rdx = (idx.1 * nvars) + idx.2;
            assert!(rdx < nord_pairs);
            let p_own = block_owner(rdx, mcx.size, nord_pairs);
            vx.4[p_own as usize] += hist_dim[idx.1];
            let sdx = (idx.2 * nvars) + idx.1;
            let p_own = block_owner(sdx, mcx.size, nord_pairs);
            vx.4[p_own as usize] += hist_dim[idx.2];
            vx
        },
    );

    //assert_eq!(rcounts, frcounts);
    let (to_snd_pairs, to_snd_tabs, to_snd_s0, to_snd_s1) =
        zip(snd_pairs.iter(), snd_tabs.iter())
            .zip(zip(snd_si0.iter(), snd_si1.iter()))
            .enumerate()
            .map(|(i, ((x, y), (s0, s1)))| {
                if i != mcx.rank as usize {
                    (*x, *y, *s0, *s1)
                } else {
                    (0, 0, 0, 0)
                }
            })
            .fold(
                (0usize, 0usize, 0usize, 0usize),
                |(acc_x, acc_y, acc_s0, acc_s1), (x, y, s0, s1)| {
                    (acc_x + x, acc_y + y, acc_s0 + s0, acc_s1 + s1)
                },
            );
    let _to_snd_ord_pairs = snd_ord
        .iter()
        .enumerate()
        .map(|(i, x)| if i != mcx.rank as usize { *x } else { 0usize })
        .sum::<usize>();

    let total_pairs = sope::reduction::allreduce_sum(&(pairs.len()), mcx.comm());
    let tabs_size: usize = snd_tabs.iter().sum();
    let total_tabs = sope::reduction::allreduce_sum(&(tabs_size), mcx.comm());
    let s0_size: usize = snd_si0.iter().sum();
    let total_s0 = sope::reduction::allreduce_sum(&(s0_size), mcx.comm());
    let s1_size: usize = snd_si1.iter().sum();
    let total_s1 = sope::reduction::allreduce_sum(&(s1_size), mcx.comm());
    let ord_snd_size: usize = snd_ord.iter().sum();
    let total_ord = sope::reduction::allreduce_sum(&ord_snd_size, mcx.comm());
    sope::gather_println!(
        mcx.comm();
        "({} {} {} {})::({} {} {:.2}):: ({} {} {:.2})::  ({} {} {:.2}) :: ({} {} {:.2})::({} {} {:.2}) ",
        npairs,
        blocks2d.num_batches(),
        pairs.len(),
        is_ranks_sorted,
        total_pairs,
        to_snd_pairs,
        to_snd_pairs as f64/pairs.len() as f64,
        total_tabs,
        to_snd_tabs,
        to_snd_tabs as f64/total_tabs as f64,
        total_s0,
        to_snd_s0,
        to_snd_s0 as f64/total_s0 as f64,
        total_s1,
        to_snd_s1,
        to_snd_s1 as f64/total_s1 as f64,
        total_ord,
        ord_snd_size,
        ord_snd_size as f64/total_ord as f64,
    );

    sope::gather_println!(
        mcx.comm();
        "C[{:?}] K[{:?}]",
        snd_pairs,
        p_ranks,
    );
    Ok(())
}

fn test_pair_dist(mcx: &CommIfx, args: &InArgs) -> Result<()> {
    let file = hdf5::File::open(&args.misi_file)?;
    let data_g = file.group("data")?;
    // attributes
    let nvars = io::read_scalar_attr::<i64>(&data_g, "nvars")? as usize;
    let npairs = io::read_scalar_attr::<i64>(&data_g, "npairs")? as usize;
    let hist_dim: Array1<usize> = data_g
        .dataset("hist_dim")?
        .read_1d::<i32>()?
        .map(|x| *x as usize);
    cond_info!(mcx.is_root(); "NVARS, NPAIRS [{} {}]", nvars, npairs);
    test_blocks_2d(
        mcx,
        nvars,
        npairs,
        &hist_dim,
        PairWorkDistributor::new(nvars, npairs, mcx.rank, mcx.size),
    )?;
    mcx.comm().barrier();
    cond_info!(mcx.is_root(); "--");
    test_blocks_2d(
        mcx,
        nvars,
        npairs,
        &hist_dim,
        PairWorkDistributor::new_seq(nvars, npairs, mcx.rank, mcx.size),
    )?;
    Ok(())
}

/// Collectively create `fname` and write a small 2-D dataset `data/X` in
/// which element `(i, j) == i * SLICE_NCOLS + j`, with each rank writing a
/// contiguous stripe of rows. Shared setup for the slice-read tests;
/// returns `(total_nrows, ncols)`.
fn write_slice_test_data(mcx: &CommIfx, fname: &str) -> (usize, usize) {
    let irow0 = mcx.rank as usize * SLICE_NROWS_PER_RANK;
    let data =
        Array2::from_shape_fn((SLICE_NROWS_PER_RANK, SLICE_NCOLS), |(i, j)| {
            ((irow0 + i) * SLICE_NCOLS + j) as i32
        });
    create_write2d(mcx, fname, "data", "X", &data).unwrap();
    mcx.comm().barrier();
    (mcx.size as usize * SLICE_NROWS_PER_RANK, SLICE_NCOLS)
}

/// Test [`mpio::read2d_row_slice`] on the dataset written by
/// [`write_slice_test_data`]. Every rank reads its own first global row
/// over `2..5` columns and checks it against the expected values.
fn test_read2d_row_slice(mcx: &CommIfx) -> Result<()> {
    let fname = "tmp/test_slice_reads.h5";
    let (_, ncols) = write_slice_test_data(mcx, fname);

    let row = mcx.rank as usize * SLICE_NROWS_PER_RANK;
    let cbounds = 2..5;
    let r_slice =
        read2d_row_slice::<i32>(fname, "data/X", row, cbounds.clone(), mcx)
            .unwrap();
    let expected: Array1<i32> =
        Array1::from_iter(cbounds.clone().map(|j| (row * ncols + j) as i32));
    assert_eq!(r_slice, expected);
    cond_info!(
        mcx.is_root();
        "[slice_reads] read2d_row_slice OK: row {} cols {:?}",
        row,
        cbounds
    );
    Ok(())
}

/// Test [`mpio::read2d_slice_of_rows`] on the dataset written by
/// [`write_slice_test_data`]. Every rank reads rows `{0, last, own}` over
/// `1..4` columns and checks each row against the expected values.
fn test_read2d_slice_of_rows(mcx: &CommIfx) -> Result<()> {
    let fname = "tmp/test_slice_reads.h5";
    let (nrows_total, ncols) = write_slice_test_data(mcx, fname);

    let indices = [
        0usize,
        nrows_total - 1,
        mcx.rank as usize * SLICE_NROWS_PER_RANK,
    ];
    let cbounds = 1..4;
    let r_block = read2d_slice_of_rows::<i32>(
        fname,
        "data/X",
        &indices,
        cbounds.clone(),
        mcx,
    )
    .unwrap();
    for (k, ri) in indices.iter().enumerate() {
        let expected: Array1<i32> =
            Array1::from_iter(cbounds.clone().map(|j| (ri * ncols + j) as i32));
        assert_eq!(r_block.row(k), expected);
    }
    cond_info!(
        mcx.is_root();
        "[slice_reads] read2d_slice_of_rows OK: rows {:?} cols {:?}",
        indices,
        cbounds
    );
    Ok(())
}

fn test_read2d_slice_of_rows_large(mcx: &CommIfx) -> Result<()> {
    let fname = "./data/pbmc_scrna/0800K/pbmc800K.20K.rmajor.h5";
    let hdf5_path = "./data/pbmc_scrna/0800K/pbmc800K.20K.h5ad";
    let adata = AnnData::new(hdf5_path, None, None)?;

    let n_sample_genes = 3;
    let mut rng = rand::rng();
    let mut var_indices: Vec<usize> = (0..adata.nvars).collect();
    var_indices.shuffle(&mut rng);
    let indices: Vec<usize> =
        var_indices.into_iter().take(n_sample_genes).collect();

    let cbounds = 0..adata.nobs;

    let r_block =
        read2d_slice_of_rows::<f32>(fname, "X", &indices, cbounds.clone(), mcx)
            .unwrap();

    cond_info!(
        mcx.is_root();
        "[slice_reads_large] read2d_slice_of_rows OK: dim [{:?}], rows [{:?}] cols {:?}",
        r_block.dim(), indices, cbounds
    );

    cond_info!(
        mcx.is_root();
        "[slice_reads_large] Top Corner: {:?}", r_block
    );

    if mcx.is_root() {
        let adata2 = AnnData::new(hdf5_path, None, Some(fname.to_string()))?;
        let sr_block = adata2.read_submatrix::<f32>(&indices)?;
        // let h5ptr = hdf5::File::open(fname)?;
        //let sr_block =
        //    io::read2d_slice_of_rows::<f32>(&h5ptr, "X", &indices, cbounds)
        //        .unwrap();
        log::info!(
            "[slice_reads_large] sr_block [{:?}] top: {:?}",
            sr_block.dim(),
            sr_block,
        );
        let sr_mat = adata.read_submatrix::<f32>(&indices)?;
        log::info!(
            "[slice_reads_large] Top sr_mat [{:?}] top: {:?}",
            sr_mat.dim(),
            sr_mat,
        );
    }
    Ok(())
}

fn run(mcx: &CommIfx, args: CLIArgs) -> Result<()> {
    let wargs = parse_args(mcx, &args)?;
    cond_info!(mcx.is_root(); "ARGS {:?}", wargs);
    let tests = if wargs.tests.is_empty() {
        ALL_TESTS.as_slice()
    } else {
        &wargs.tests
    };
    for tx in tests {
        match *tx {
            Test::H5 => {
                test_h5(mcx, &args)?;
            }
            Test::Samples => {
                test_samples(mcx, &args, &wargs)?;
            }
            Test::PairDist => {
                test_pair_dist(mcx, &wargs)?;
            }
            Test::SliceReads => {
                test_read2d_row_slice(mcx)?;
                test_read2d_slice_of_rows(mcx)?;
                test_read2d_slice_of_rows_large(mcx)?;
            }
        }
    }
    Ok(())
}

fn main() {
    let comm_ifx = CommIfx::init();
    env_logger::init();
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
