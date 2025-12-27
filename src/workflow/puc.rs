use super::{IdVResults, WorkDistributor, WorkflowArgs, collect_samples};
use crate::{
    comm::CommIfx,
    cond_info,
    h5::mpio::{block_write1d, block_write2d, create_file},
    mvim::{misi::MISIRangePair, rv::Error as RVError, rv::MRVTrait},
    util::{GenericError, RangePair, vec::Vec2d},
};

use hdf5_metno::Error as H5Error;
use itertools::iproduct;
use ndarray::{Array1, Array2, ArrayView1};

pub type PUCResults = IdVResults<i32, f32>;

fn pair_indices(st_ranges: RangePair<usize>) -> Array2<i32> {
    let (s_range, t_range) = st_ranges;
    let (s_vec, t_vec): (Vec<i32>, Vec<i32>) = iproduct!(s_range, t_range)
        .filter(|(src, tgt)| src < tgt)
        .map(|(src, tgt)| (src as i32, tgt as i32))
        .unzip();

    let mut st_arr = Array2::<i32>::zeros((s_vec.len(), 2));
    st_arr
        .slice_mut(ndarray::s![.., 0])
        .assign(&Array1::from_vec(s_vec));
    st_arr
        .slice_mut(ndarray::s![.., 1])
        .assign(&Array1::from_vec(t_vec));
    st_arr
}

fn save_puc(
    puc_results: PUCResults,
    mpi_ifx: &CommIfx,
    puc_file: &str,
) -> Result<(), H5Error> {
    let h_file = create_file(mpi_ifx, puc_file)?;
    let h_group = h_file.create_group("data").unwrap();
    block_write2d(mpi_ifx, &h_group, "index", &puc_results.index)?;
    block_write1d(mpi_ifx, &h_group, "puc", &puc_results.val)?;
    Ok(())
}

pub struct SampledPUC<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
}

impl<'a> SampledPUC<'a> {
    fn run_with_samples(
        &self,
        st_row: ArrayView1<i32>,
        r_samples: &Vec2d<i32>,
        m_range: &MISIRangePair<i32, f32>,
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
        r_samples: &Option<Vec2d<i32>>,
    ) -> Result<PUCResults, GenericError> {
        let m_range = MISIRangePair::<i32, f32>::new(
            &self.args.h5ad_file,
            st_ranges.clone(),
        )?;
        let r_pindex = pair_indices(st_ranges.clone());
        let mut r_pucs = Array1::from_vec(vec![0.0f32; r_pindex.len()]);
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
        rsamples: &Option<Vec2d<i32>>,
    ) -> Result<PUCResults, GenericError> {
        self.run_st_ranges(
            self.wdistr
                .pairs2d
                .batch_ranges
                .at(bid, self.mpi_ifx.rank as usize),
            rsamples,
        )
    }

    pub fn run(&self) -> Result<(), GenericError> {
        let nbatches = self.wdistr.pairs2d.n_batches;
        // allow for samples being none
        let rsamples = match collect_samples::<i32>(
            self.mpi_ifx,
            self.args.nvars,
            self.args.nrounds,
            self.args.nsamples,
        ) {
            Ok(samples_vec) => Some(samples_vec),
            Err(err) => {
                cond_info!(self.mpi_ifx.is_root(); "{} No Samples; Running Full PUC", err);
                None
            }
        };
        let bat_results: Result<Vec<_>, GenericError> = (0..nbatches)
            .map(|bidx| self.run_batch(bidx, &rsamples))
            .collect();
        match bat_results {
            Ok(vpair_reds) => Ok(save_puc(
                PUCResults::merge(&vpair_reds),
                self.mpi_ifx,
                &self.args.puc_file,
            )?),
            Err(err) => Err(err),
        }
    }
}

pub struct LMRPUC<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
}

impl<'a> LMRPUC<'a> {
    fn run_with_misi_range(
        &self,
        m_range: &MISIRangePair<i32, f32>,
        r_pindex: &Array2<i32>,
        r_pucs: &mut Array1<f32>,
    ) -> Result<(), GenericError> {
        for (idx, st_row) in r_pindex.rows().into_iter().enumerate() {
            let (src, tgt) = (st_row[0], st_row[1]);
            r_pucs[idx] += m_range.compute_lm_puc(src, tgt)?;
        }
        Ok(())
    }

    fn run_st_ranges(
        &self,
        st_ranges: &RangePair<usize>,
        r_samples: &Option<Vec2d<i32>>,
    ) -> Result<PUCResults, GenericError> {
        let r_pindex = pair_indices(st_ranges.clone());
        let mut r_pucs = Array1::from_vec(vec![0.0f32; r_pindex.len()]);
        let mut m_range = MISIRangePair::<i32, f32>::new(
            &self.args.h5ad_file,
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
        rsamples: &Option<Vec2d<i32>>,
    ) -> Result<PUCResults, GenericError> {
        self.run_st_ranges(
            self.wdistr
                .pairs2d
                .batch_ranges
                .at(bid, self.mpi_ifx.rank as usize),
            rsamples,
        )
    }

    pub fn run(&self) -> Result<(), GenericError> {
        let nbatches = self.wdistr.pairs2d.n_batches;
        let rsamples = match collect_samples::<i32>(
            self.mpi_ifx,
            self.args.nvars,
            self.args.nrounds,
            self.args.nsamples,
        ) {
            Ok(samples_vec) => Some(samples_vec),
            Err(err) => {
                cond_info!(self.mpi_ifx.is_root(); "{} No Samples; Running Full PUC", err);
                None
            }
        };

        let bat_results: Result<Vec<PUCResults>, GenericError> = (0..nbatches)
            .map(|bidx| self.run_batch(bidx, &rsamples))
            .collect();

        match bat_results {
            Ok(vpair_reds) => Ok(save_puc(
                PUCResults::merge(&vpair_reds),
                self.mpi_ifx,
                &self.args.puc_file,
            )?),
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {}
