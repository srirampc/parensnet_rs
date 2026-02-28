use super::{
    IdVResults, WorkDistributor, WorkflowArgs, collect_samples, pair_indices,
};
use crate::{
    comm::CommIfx,
    cond_info,
    h5::mpio,
    mvim::{
        misi::MISIRangePair,
        rv::{Error as RVError, MRVTrait},
    },
    types::{PNFloat, PNInteger},
    util::{RangePair, Vec2d},
};

use anyhow::{Result, anyhow};
use hdf5::{Error as H5Error, H5Type};
use mpi::traits::Equivalence;
use ndarray::{Array1, Array2, ArrayView1};
use sope::{
    reduction::{allreduce_sum, any_of},
    timer::SectionTimer,
};
use std::marker::PhantomData;

pub type PUCResults<IntT, FloatT> = IdVResults<IntT, FloatT>;

pub trait PUCRTrait<IntT, FloatT> {
    fn save(&self, mpi_ifx: &CommIfx, puc_file: &str) -> Result<(), H5Error>;
}

impl<IntT, FloatT> PUCRTrait<IntT, FloatT> for PUCResults<IntT, FloatT>
where
    IntT: H5Type + Default + Equivalence,
    FloatT: H5Type + Default + Equivalence,
{
    fn save(&self, mpi_ifx: &CommIfx, puc_file: &str) -> Result<(), H5Error> {
        let h_file = mpio::create_file(mpi_ifx, puc_file)?;
        cond_info!(
            mpi_ifx.is_root();
            "Saving Data :: {:?} {:?} {}",
            self.index.shape(),
            self.val.shape(),
            puc_file,
        );
        sope::gather_debug!(mpi_ifx.comm(); "{:?}", self.val.len());
        let h_group = h_file.create_group("data").unwrap();
        mpio::block_write2d(mpi_ifx, &h_group, "index", &self.index)?;
        mpio::block_write1d(mpi_ifx, &h_group, "puc", &self.val)?;
        Ok(())
    }
}

pub struct SampledPUCWorkflow<'a> {
    pub mpi_ifx: &'a CommIfx,
    pub wdistr: &'a WorkDistributor,
    pub args: &'a WorkflowArgs,
}

struct SampledWorkFlowHelper<SizeT, IntT, FloatT> {
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> SampledWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    fn rows_puc(
        st_row: ArrayView1<IntT>,
        r_samples: &Vec2d<IntT>,
        m_range: &MISIRangePair<SizeT, IntT, FloatT>,
    ) -> Result<FloatT, RVError> {
        let mut avg_red: FloatT = FloatT::zero();
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
        avg_red /= FloatT::from_usize(nrounds).unwrap();
        Ok(avg_red)
    }

    fn ranges_puc(
        swf: &SampledPUCWorkflow,
        st_ranges: &RangePair<usize>,
        r_samples: Option<&Vec2d<IntT>>,
    ) -> Result<PUCResults<IntT, FloatT>> {
        let m_range = MISIRangePair::<SizeT, IntT, FloatT>::new(
            &swf.args.misi_data_file,
            st_ranges.clone(),
        )?;
        let r_pindex = pair_indices(st_ranges.clone());
        let mut r_pucs = Array1::from_vec(vec![FloatT::zero(); r_pindex.nrows()]);
        for (idx, st_row) in r_pindex.rows().into_iter().enumerate() {
            r_pucs[idx] = match r_samples {
                Some(samples_vec) => {
                    Self::rows_puc(st_row, samples_vec, &m_range)?
                }
                None => {
                    let (src, tgt) = (st_row[0], st_row[1]);
                    m_range.accumulate_redundancies(src, tgt)?
                }
            };
        }
        Ok(PUCResults::new(r_pindex, r_pucs))
    }

    fn batch_puc(
        swf: &SampledPUCWorkflow,
        bid: usize,
        rsamples: Option<&Vec2d<IntT>>,
    ) -> Result<PUCResults<IntT, FloatT>> {
        Self::ranges_puc(
            swf,
            swf.wdistr.pairs_2d().batch_range(bid, swf.mpi_ifx.rank),
            rsamples,
        )
    }
}

impl<'a> SampledPUCWorkflow<'a> {
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
                    "{} ; Running Full PUC", err
                );
                None
            }
        };
        let bat_results: Result<Vec<_>> = (0..nbatches)
            .map(|bidx| {
                SampledWorkFlowHelper::<i64, i32, f32>::batch_puc(
                    self,
                    bidx,
                    rsamples.as_ref(),
                )
            })
            .collect();
        match bat_results {
            Ok(vpair_reds) => {
                let merged_results = PUCResults::merge(&vpair_reds);
                merged_results.save(self.mpi_ifx, &self.args.puc_file)?;
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

struct LMRWorkFlowHelper<SizeT, IntT, FloatT> {
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> LMRWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    fn misi_range_puc(
        m_range: &MISIRangePair<SizeT, IntT, FloatT>,
        r_pindex: &Array2<IntT>,
        r_pucs: &mut Array1<FloatT>,
    ) -> Result<()> {
        for (idx, st_row) in r_pindex.rows().into_iter().enumerate() {
            let (src, tgt) = (st_row[0], st_row[1]);
            let rpuc = m_range.compute_lm_puc(src, tgt)?;
            r_pucs[idx] += rpuc;
        }
        Ok(())
    }

    fn ranges_puc(
        lwf: &LMRPUCWorkflow,
        st_ranges: &RangePair<usize>,
        r_samples: Option<&Vec2d<IntT>>,
    ) -> Result<PUCResults<IntT, FloatT>> {
        let r_pindex = pair_indices(st_ranges.clone());
        let mut r_pucs =
            Array1::from_vec(vec![FloatT::default(); r_pindex.nrows()]);
        let mut m_range = MISIRangePair::<SizeT, IntT, FloatT>::new(
            &lwf.args.misi_data_file,
            st_ranges.clone(),
        )?;

        let nrounds = match r_samples {
            Some(samples_vec) => {
                let nrounds = samples_vec.nrows();
                for irow in 0..nrounds {
                    m_range.set_lmr_ds(Some(samples_vec.row(irow)))?;
                    Self::misi_range_puc(&m_range, &r_pindex, &mut r_pucs)?;
                }
                nrounds
            }
            None => {
                m_range.set_lmr_ds(None)?;
                Self::misi_range_puc(&m_range, &r_pindex, &mut r_pucs)?;
                1usize
            }
        };
        for rval in r_pucs.iter_mut() {
            *rval /= FloatT::from_usize(nrounds).unwrap();
        }
        Ok(PUCResults::new(r_pindex, r_pucs))
    }

    fn batch_puc(
        lwf: &LMRPUCWorkflow,
        bid: usize,
        rsamples: Option<&Vec2d<IntT>>,
    ) -> Result<PUCResults<IntT, FloatT>> {
        LMRWorkFlowHelper::<SizeT, IntT, FloatT>::ranges_puc(
            lwf,
            lwf.wdistr.pairs_2d().batch_range(bid, lwf.mpi_ifx.rank),
            rsamples,
        )
    }
}

impl<'a> LMRPUCWorkflow<'a> {
    pub fn run(&self) -> Result<()> {
        type HelperT = LMRWorkFlowHelper<i64, i32, f32>;
        type PT = PUCResults<i32, i64>;
        let mut s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");

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
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Collect Samples");
            cond_info!(
                self.mpi_ifx.is_root();
                "Samples: {:?}", rsamples.as_ref().map(|x| x.shape())
            );
            s_timer.reset();
        }

        let bat_results = (0..nbatches)
            .map(|bidx| HelperT::batch_puc(self, bidx, rsamples.as_ref()))
            .collect::<Result<Vec<_>>>();
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Compute PUC");
            let n_batches =
                bat_results.as_ref().map(|x| x.len()).unwrap_or_default();
            let n_vpuc = bat_results
                .as_ref()
                .map(|x| x.iter().map(|y| y.len()).sum::<usize>())
                .unwrap_or_default();
            let n_batches = allreduce_sum(&n_batches, self.mpi_ifx.comm());
            let n_vpuc = allreduce_sum(&n_vpuc, self.mpi_ifx.comm());
            // let nv =
            //    gather_one(&n_vpuc, 0, wf.mpi_ifx.comm())?.unwrap_or_default();
            cond_info!(
                self.mpi_ifx.is_root();
                "Batches Completed. NBATCHES: {} NPUC: {}", n_batches, n_vpuc
            );
            s_timer.reset();
        }

        if any_of(bat_results.is_err(), self.mpi_ifx.comm()) {
            if let Err(err) = bat_results {
                return Err(err);
            } else {
                return Err(anyhow!(
                    "Failed to find results in one of the procs."
                ));
            }
        }
        let bat_results = bat_results.unwrap_or_default();
        let m_results = PUCResults::merge(&bat_results);
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Merge Results");
            let n_vpuc = allreduce_sum(&(m_results.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root();
                "Merged Completed : {}", n_vpuc
            );
            s_timer.reset();
        }
        m_results.save(self.mpi_ifx, &self.args.puc_file)?;
        //
        s_timer.info_section("Save PUC Pairs");
        Ok(())
    }
}

#[cfg(test)]
mod tests {}
