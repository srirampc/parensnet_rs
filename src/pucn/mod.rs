#![allow(dead_code)]
mod args;
mod misiw;
mod puc;
pub use self::args::{RunMode, WorkflowArgs};

use crate::{
    anndata::AnnData,
    comm::CommIfx,
    util::{
        BatchBlocks2D, EBBlocks2D, RangePair, Vec2d, all_block_ranges,
        exc_prefix_sum,
    },
};
use anyhow::{Result, bail};
use itertools::iproduct;
use mpi::traits::{Communicator, Root};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use num::{FromPrimitive, Integer, Zero};
use std::ops::{AddAssign, Range};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WorkflowError {
    #[error("Error reading file (nvars:{0}, nrounds:{1}, nsamples:{2})")]
    InvalidSamples(usize, usize, usize),
}

pub struct WorkDistributor {
    rank: i32,
    size: i32,
    var_dist: Vec<Range<usize>>,
    pairs1d_dist: Vec<Range<usize>>,
    pairs2d: EBBlocks2D,
}

impl WorkDistributor {
    pub fn new(nvars: usize, npairs: usize, rank: i32, size: i32) -> Self {
        WorkDistributor {
            rank,
            size,
            var_dist: all_block_ranges(size, nvars),
            pairs1d_dist: all_block_ranges(size, npairs),
            pairs2d: EBBlocks2D::new_diag(nvars, size as usize),
        }
    }

    pub fn new_seq(nvars: usize, npairs: usize, rank: i32, size: i32) -> Self {
        WorkDistributor {
            rank,
            size,
            var_dist: all_block_ranges(size, nvars),
            pairs1d_dist: all_block_ranges(size, npairs),
            pairs2d: EBBlocks2D::new_seq(nvars, size as usize),
        }
    }

    pub fn pairs_2d(&self) -> &dyn BatchBlocks2D {
        self.pairs2d.trait_ref()
    }

    pub fn pairs_1d(&self) -> &[Range<usize>] {
        &self.pairs1d_dist
    }

    pub fn vars_dist(&self) -> &[Range<usize>] {
        &self.var_dist
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

    pub fn len(&self) -> usize {
        self.val.len()
    }

    pub fn is_empty(&self) -> bool {
        self.val.is_empty()
    }

    pub fn merge(vpreds: &[Self]) -> Self {
        let nsizes: Vec<usize> = vpreds.iter().map(|x| x.len()).collect();
        let nstarts: Vec<usize> = exc_prefix_sum(nsizes.clone().into_iter(), 1);
        let ntotal: usize = vpreds.iter().map(|x| x.len()).sum();
        let mut pindices: Array2<T> = Array2::zeros((ntotal, 2));
        let mut preds: Array1<S> = Array1::zeros(ntotal);

        for (idx, rstart) in nstarts.iter().enumerate() {
            let rsize = vpreds[idx].val.len();
            let rend = *rstart + rsize;
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

fn pair_indices<T>(st_ranges: RangePair<usize>) -> Array2<T>
where
    T: Integer + AddAssign + FromPrimitive + Clone,
{
    let (s_range, t_range) = st_ranges;
    let (s_vec, t_vec): (Vec<T>, Vec<T>) = iproduct!(s_range, t_range)
        .filter(|(src, tgt)| src < tgt)
        .map(|(src, tgt)| {
            (T::from_usize(src).unwrap(), T::from_usize(tgt).unwrap())
        })
        .unzip();

    let mut st_arr = Array2::<T>::zeros((s_vec.len(), 2));
    st_arr
        .slice_mut(ndarray::s![.., 0])
        .assign(&Array1::from_vec(s_vec));
    st_arr
        .slice_mut(ndarray::s![.., 1])
        .assign(&Array1::from_vec(t_vec));
    st_arr
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
) -> Result<Vec2d<T>> {
    if nrounds == 0 || nsamples == 0 {
        bail!(WorkflowError::InvalidSamples(nvars, nrounds, nsamples,));
    }
    let mut sample_arr = if mcx.is_root() {
        generate_samples(nvars, nrounds, nsamples)
    } else {
        vec![0usize; nrounds * nsamples]
    };
    let rootp = mcx.comm().process_at_rank(0);
    rootp.broadcast_into(&mut sample_arr);
    let tsamples = sample_arr
        .iter()
        .map(|val| T::from_usize(*val).unwrap())
        .collect();
    Ok(Vec2d::new(tsamples, nsamples, nrounds))
}

pub fn execute_workflow(mpi_ifx: &CommIfx, args: &WorkflowArgs) -> Result<()> {
    // Compute Distributions
    let wdistr =
        WorkDistributor::new(args.nvars, args.npairs, mpi_ifx.rank, mpi_ifx.size);

    for rmode in &args.mode {
        match rmode {
            RunMode::SamplesRanges => {
                let spuc = puc::SampledPUCWorkflow {
                    args,
                    wdistr: &wdistr,
                    mpi_ifx,
                };
                spuc.run()?;
            }
            RunMode::PUCLMR => {
                let lpuc = puc::LMRPUCWorkflow {
                    args,
                    wdistr: &wdistr,
                    mpi_ifx,
                };
                lpuc.run()?;
            }
            RunMode::MISI => {
                let adata = AnnData::new(&args.h5ad_file, None)?;
                let rmisi = misiw::MISIWorkFlow {
                    args,
                    adata: &adata,
                    wdistr: &wdistr,
                    mpi_ifx,
                };
                rmisi.run()?;
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
    pub fn test_parse_workflow_args() -> Result<()> {
        crate::tests::log_init();
        match parse_wflow_cfg!("/pucn/pbmc20k_500_lpuc.yml") {
            Ok(wargs) => {
                info!("Parsed successfully: {:?}", wargs);
                assert_eq!(wargs.disc_method, DiscretizerMethod::BayesianBlocks);
                assert_eq!(wargs.tbase, LogBase::Natural);
            }
            Err(e) => {
                error!("Failed to parse YAML: {}", e);
            }
        };

        match parse_wflow_cfg!("/pucn/pbmc20k_500_S6x4p.yml") {
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

    pub fn test_parse_workflow_distr() -> Result<()> {
        let wargs = parse_wflow_cfg!("pbmc20k_500_S6x4p.yml").unwrap();
        let wdistr = WorkDistributor::new(wargs.nvars, wargs.npairs, 2, 16);
        debug!("Work Distribution : {:?}", wdistr.pairs2d);
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
