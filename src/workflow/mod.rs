#![allow(dead_code)]
mod args;
pub mod puc;
pub use self::args::RunMode;
pub use self::args::WorkflowArgs;

use crate::{
    comm::CommIfx,
    util::{
        BatchBlocks2D, GenericError, all_block_ranges, exc_prefix_sum, vec::Vec2d,
    },
};
use mpi::traits::{Communicator, Root};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use num::{FromPrimitive, Integer, Zero};
use std::ops::Range;

#[derive(Debug)]
pub enum Error {
    InvalidSamples(usize, usize, usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidSamples(nvars, nrounds, nsamples) => {
                write!(f, "Error reading file ({nvars}, {nrounds}, {nsamples})")
            }
        }
    }
}

impl std::error::Error for Error {}

pub struct WorkDistributor {
    rank: i32,
    size: i32,
    var_dist: Vec<Range<usize>>,
    pairs1d_dist: Vec<Range<usize>>,
    pairs2d: BatchBlocks2D,
}

impl WorkDistributor {
    pub fn new(nvars: usize, npairs: usize, rank: i32, size: i32) -> Self {
        WorkDistributor {
            rank,
            size,
            var_dist: all_block_ranges(size, nvars),
            pairs1d_dist: all_block_ranges(size, npairs),
            pairs2d: BatchBlocks2D::new(nvars, size as usize),
        }
    }
}

pub struct IdVResults<T, S> {
    index: Array2<T>,
    val: Array1<S>,
}

impl<T: Clone + Zero, S: Clone + Zero> IdVResults<T, S> {
    pub fn new(index: Array2<T>, val: Array1<S>) -> Self {
        Self { index, val }
    }

    pub fn merge(vpreds: &[Self]) -> Self {
        let nsizes: Vec<usize> = vpreds.iter().map(|x| x.val.len()).collect();
        let nstarts: Vec<usize> = exc_prefix_sum(nsizes.into_iter(), 1);
        let ntotal: usize = vpreds.iter().map(|x| x.val.len()).sum();
        let mut pindices: Array2<T> = Array2::zeros((ntotal, 2));
        let mut preds: Array1<S> = Array1::zeros(ntotal);

        for (idx, rstart) in nstarts.iter().enumerate() {
            let rsize = vpreds[idx].val.len();
            let rend = rstart + rsize;
            pindices
                .slice_mut(ndarray::s![*rstart..rend, ..])
                .assign(&vpreds[idx].index);
            preds
                .slice_mut(ndarray::s![*rstart..rend])
                .assign(&vpreds[idx].val);
        }
        Self::new(pindices, preds)
    }
}

pub fn generate_samples(
    nvars: usize,
    nrounds: usize,
    nsamples: usize,
) -> Vec<usize> {
    let mut sid_vec = vec![0usize; nrounds * nsamples];
    let var_ids = Array1::<usize>::from_iter(0..nvars);
    for row in 0..nrounds {
        let row_samples = var_ids.sample_axis(
            Axis(0),
            nsamples,
            SamplingStrategy::WithoutReplacement,
        );
        for (col, val) in row_samples.iter().enumerate() {
            sid_vec[row * nsamples + col] = *val;
        }
    }
    sid_vec
}

pub fn generate_random_samples(
    nvars: usize,
    nrounds: usize,
    nsamples: usize,
) -> Array2<usize> {
    let idarr = Array1::<usize>::from_iter(0..nvars);
    let mut sample_arr = Array2::<usize>::from_elem((nsamples, nrounds), 0);
    for col in 0..nrounds {
        let ridx = idarr.sample_axis(
            Axis(0),
            nsamples,
            SamplingStrategy::WithoutReplacement,
        );
        let mut sample_col = sample_arr.column_mut(col);
        sample_col.assign(&ridx);
    }
    sample_arr
}

pub fn collect_samples<T: Integer + Clone + FromPrimitive>(
    mcx: &CommIfx,
    nvars: usize,
    nrounds: usize,
    nsamples: usize,
) -> Result<Vec2d<T>, Error> {
    if nrounds == 0 || nsamples == 0 {
        return Result::Err(Error::InvalidSamples(nvars, nrounds, nsamples));
    }
    let mut sample_arr = if mcx.is_root() {
        generate_samples(nvars, nrounds, nsamples)
    } else {
        vec![0usize; nrounds * nsamples]
    };
    let rootp = mcx.comm.process_at_rank(0);
    rootp.broadcast_into(&mut sample_arr);
    let tsamples = sample_arr
        .iter()
        .map(|val| T::from_usize(*val).unwrap())
        .collect();
    Ok(Vec2d::new(tsamples, nsamples, nrounds))
}

pub fn execute_workflow(
    mpi_ifx: &CommIfx,
    args: &WorkflowArgs,
) -> Result<(), GenericError> {
    // Compute Distributions
    let wdistr =
        WorkDistributor::new(args.nvars, args.npairs, mpi_ifx.rank, mpi_ifx.size);

    for rmode in &args.mode {
        match rmode {
            RunMode::SamplesRanges => {
                let spuc = puc::SampledPUC {
                    args,
                    wdistr: &wdistr,
                    mpi_ifx,
                };
                spuc.run()?;
            }
            RunMode::PUCLMR => {
                let lpuc = puc::LMRPUC {
                    args,
                    wdistr: &wdistr,
                    mpi_ifx,
                };
                lpuc.run()?;
            }
            _ => todo!("Mode {:?} Not Completed Yet", rmode),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_config_file_path;
    use crate::types::{DiscretizerMethod, LogBase};
    use crate::util::read_file_to_string;
    use log::{debug, error, info};

    macro_rules! parse_wflow_cfg {
        ($name:literal) => {
            serde_saphyr::from_str::<WorkflowArgs>(
                &(read_file_to_string(test_config_file_path!($name))?),
            )
        };
    }

    #[test]
    pub fn test_parse_workflow_args() -> Result<(), GenericError> {
        crate::tests::log_init();
        match parse_wflow_cfg!("pbmc20k_500_lpuc.yml") {
            Ok(wargs) => {
                info!("Parsed successfully: {:?}", wargs);
                assert_eq!(wargs.disc_method, DiscretizerMethod::BayesianBlocks);
                assert_eq!(wargs.tbase, LogBase::Two);
            }
            Err(e) => {
                error!("Failed to parse YAML: {}", e);
            }
        };

        match parse_wflow_cfg!("pbmc20k_500_S6x4p.yml") {
            Ok(p_wargs) => {
                info!("Parsed successfully: {:?}", p_wargs);
                assert_eq!(p_wargs.mode, vec![RunMode::SamplesLMRRanges]);
            }
            Err(e) => {
                error!("Failed to parse YAML: {}", e);
            }
        }
        Ok(())
    }

    pub fn test_parse_workflow_distr() -> Result<(), GenericError> {
        let wargs = parse_wflow_cfg!("pbmc20k_500_S6x4p.yml").unwrap();
        let wdistr = WorkDistributor::new(wargs.nvars, wargs.npairs, 2, 16);
        info!("Work Distribution : {:?}", wdistr.pairs2d);
        Ok(())
    }

    #[test]
    pub fn test_sample() {
        crate::tests::log_init();
        let samples = super::generate_random_samples(500, 4, 6);
        debug!("Samples {:?}", samples);
        let samples2 = super::generate_samples(500, 4, 6);
        debug!("Samples {:?}", samples2);
    }
}
