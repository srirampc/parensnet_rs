use super::{
    IdVResults, WorkDistributor, WorkflowArgs, collect_samples, pair_indices,
};
use crate::{
    comm::CommIfx,
    cond_info,
    h5::mpio::{block_write1d, block_write2d, create_file},
    mvim::{
        misi::MISIRangePair,
        rv::{Error as RVError, MRVTrait},
    },
    util::{RangePair, Vec2d},
};

use anyhow::{Result, anyhow};
use hdf5::Error as H5Error;
use ndarray::{Array1, Array2, ArrayView1};
use sope::reduction::any_of;

pub type PUCResults = IdVResults<i32, f32>;

fn save_puc(
    puc_results: PUCResults,
    mpi_ifx: &CommIfx,
    puc_file: &str,
) -> Result<(), H5Error> {
    let h_file = create_file(mpi_ifx, puc_file)?;
    cond_info!(
        mpi_ifx.is_root();
        "Saving Data :: {:?} {:?} {}",
        puc_results.index.shape(),
        puc_results.val.shape(),
        puc_file,
    );
    sope::gather_debug!(mpi_ifx.comm(); "{:?}", puc_results.len());
    let h_group = h_file.create_group("data").unwrap();
    block_write2d(mpi_ifx, &h_group, "index", &puc_results.index)?;
    block_write1d(mpi_ifx, &h_group, "puc", &puc_results.val)?;
    Ok(())
}

pub struct SampledPUCWorkflow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
}

impl<'a> SampledPUCWorkflow<'a> {
    fn run_with_samples(
        &self,
        st_row: ArrayView1<i32>,
        r_samples: &Vec2d<i32>,
        m_range: &MISIRangePair<i64, i32, f32>,
    ) -> Result<f32, RVError> {
        let mut avg_red: f32 = 0.0;
        let (src, tgt) = (st_row[0], st_row[1]);
        let nrounds = r_samples.nrows();
        for irow in 0..nrounds {
            let redv = m_range.accumulate_redundancies_for(
                src,
                tgt,
                r_samples.row(irow),
            )?;
            avg_red += redv;
        }
        avg_red /= nrounds as f32;
        Ok(avg_red)
    }

    fn run_st_ranges(
        &self,
        st_ranges: &RangePair<usize>,
        r_samples: Option<&Vec2d<i32>>,
    ) -> Result<PUCResults> {
        let m_range = MISIRangePair::<i64, i32, f32>::new(
            &self.args.misi_data_file,
            st_ranges.clone(),
        )?;
        let r_pindex = pair_indices(st_ranges.clone());
        let mut r_pucs = Array1::from_vec(vec![0.0f32; r_pindex.nrows()]);
        for (idx, st_row) in r_pindex.rows().into_iter().enumerate() {
            r_pucs[idx] = match r_samples {
                Some(samples_vec) => {
                    self.run_with_samples(st_row, samples_vec, &m_range)?
                }
                None => {
                    let (src, tgt) = (st_row[0], st_row[1]);
                    m_range.accumulate_redundancies(src, tgt)?
                }
            };
        }
        Ok(PUCResults::new(r_pindex, r_pucs))
    }

    fn run_batch(
        &self,
        bid: usize,
        rsamples: Option<&Vec2d<i32>>,
    ) -> Result<PUCResults> {
        self.run_st_ranges(
            self.wdistr.pairs_2d().batch_range(bid, self.mpi_ifx.rank),
            rsamples,
        )
    }

    pub fn run(&self) -> Result<()> {
        let nbatches = self.wdistr.pairs_2d().num_batches();
        // allow for samples being none
        let rsamples = match collect_samples::<i32>(
            self.mpi_ifx,
            self.args.nvars,
            self.args.nrounds,
            self.args.nsamples,
        ) {
            Ok(samples_vec) => {
                cond_info!(
                    self.mpi_ifx.is_root();
                    "Running with {:?} Samples Generated", samples_vec.shape()
                );
                Some(samples_vec)
            }
            Err(err) => {
                cond_info!(
                    self.mpi_ifx.is_root();
                    "{} No Samples; Running Full PUC", err
                );
                None
            }
        };
        let bat_results: Result<Vec<_>> = (0..nbatches)
            .map(|bidx| self.run_batch(bidx, rsamples.as_ref()))
            .collect();
        match bat_results {
            Ok(vpair_reds) => {
                let merged_results = PUCResults::merge(&vpair_reds);
                save_puc(merged_results, self.mpi_ifx, &self.args.puc_file)?;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }
}

pub struct LMRPUCWorkflow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
}

impl<'a> LMRPUCWorkflow<'a> {
    fn run_with_misi_range(
        &self,
        m_range: &MISIRangePair<i64, i32, f32>,
        r_pindex: &Array2<i32>,
        r_pucs: &mut Array1<f32>,
    ) -> Result<()> {
        for (idx, st_row) in r_pindex.rows().into_iter().enumerate() {
            let (src, tgt) = (st_row[0], st_row[1]);
            let rpuc = m_range.compute_lm_puc(src, tgt)?;
            r_pucs[idx] += rpuc;
        }
        Ok(())
    }

    fn run_st_ranges(
        &self,
        st_ranges: &RangePair<usize>,
        r_samples: Option<&Vec2d<i32>>,
    ) -> Result<PUCResults> {
        let r_pindex = pair_indices(st_ranges.clone());
        let mut r_pucs = Array1::from_vec(vec![0.0f32; r_pindex.nrows()]);
        let mut m_range = MISIRangePair::<i64, i32, f32>::new(
            &self.args.misi_data_file,
            st_ranges.clone(),
        )?;

        let nrounds = match r_samples {
            Some(samples_vec) => {
                let nrounds = samples_vec.nrows();
                for irow in 0..nrounds {
                    m_range.set_lmr_ds(Some(samples_vec.row(irow)))?;
                    self.run_with_misi_range(&m_range, &r_pindex, &mut r_pucs)?;
                }
                nrounds
            }
            None => {
                m_range.set_lmr_ds(None)?;
                self.run_with_misi_range(&m_range, &r_pindex, &mut r_pucs)?;
                1usize
            }
        };
        for rval in r_pucs.iter_mut() {
            *rval /= nrounds as f32;
        }
        Ok(PUCResults::new(r_pindex, r_pucs))
    }

    fn run_batch(
        &self,
        bid: usize,
        rsamples: Option<&Vec2d<i32>>,
    ) -> Result<PUCResults> {
        self.run_st_ranges(
            self.wdistr.pairs_2d().batch_range(bid, self.mpi_ifx.rank),
            rsamples,
        )
    }

    pub fn run(&self) -> Result<()> {
        let nbatches = self.wdistr.pairs_2d().num_batches();
        let rsamples = match collect_samples::<i32>(
            self.mpi_ifx,
            self.args.nvars,
            self.args.nrounds,
            self.args.nsamples,
        ) {
            Ok(samples_vec) => {
                cond_info!(
                    self.mpi_ifx.is_root();
                    "Running with {:?} Samples Generated", samples_vec.shape()
                );
                Some(samples_vec)
            }
            Err(err) => {
                cond_info!(
                    self.mpi_ifx.is_root();
                    "{} No Samples; Running Full PUC", err
                );
                None
            }
        };

        let bat_results: Result<Vec<PUCResults>> = (0..nbatches)
            .map(|bidx| self.run_batch(bidx, rsamples.as_ref()))
            .collect();
        if any_of(bat_results.is_err(), self.mpi_ifx.comm()) {
            if let Err(err) = bat_results {
                return Err(err);
            } else {
                return Err(anyhow!(
                    "Failed to find results in one of the procs."
                ));
            }
        }

        match bat_results {
            Ok(vpair_reds) => {
                let presults = PUCResults::merge(&vpair_reds);
                println!("Z {}", presults.len());
                save_puc(presults, self.mpi_ifx, &self.args.puc_file)?;
                Ok(())
            }
            Err(err) => Err(err),
        }
        //Ok(())
    }
}

#[cfg(test)]
mod tests {}
