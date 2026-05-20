use anyhow::{Result, anyhow};
use hdf5::H5Type;
use mpi::traits::Equivalence;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use sope::{
    reduction::{allreduce_sum, any_of},
    timer::{CumulativeTimer, SectionTimer},
};
use std::{iter::zip, marker::PhantomData};

use crate::{
    comm::CommIfx,
    cond_info,
    h5::io,
    pucn::puc::{PUCRTrait, PUCResults},
    types::{PNFloat, PNInteger},
    util::{PairWorkDistributor, RangePair, pair_indices},
};

use super::WorkflowArgs;

/// Per-rank PIDC output container: pair indices in `index` and the
/// matching PIDC scores in `val`. Aliased to [`IdVResults`] so the
/// generic `merge` / `save` machinery can be reused.
pub type PIDCResults<IntT, FloatT> = PUCResults<IntT, FloatT>;

/// Driver for the Context workflow -- PUC2PIDC.
///
/// Has references to the MPI context, the pair-work distributor and the
/// parsed configuration.
/// PIDC is computed from PUC by TODO::.
/// Run when mode is [`crate::pucn::RunMode::PUC2PIDC`].
pub struct ContextWorkflow<'a> {
    /// MPI communicator wrapper used by every collective call.
    pub mpi_ifx: &'a CommIfx,
    /// Pair-work distributor describing the 2-D pair grid batches.
    pub wdistr: &'a PairWorkDistributor,
    /// Parsed workflow configuration.
    pub args: &'a WorkflowArgs,
    /// Cumulative IO timer; the helpers add elapsed read/write time
    /// to it on every parallel-IO call.
    pub io_timer: CumulativeTimer<'a>,
}

/// Stateless namespace for the per-batch helpers used by
/// [`ContextWorkflow`].
///
/// Carries a [`PhantomData`] marker so the three numeric type
/// parameters can be supplied once at the call site.
struct ContextWorkFlowHelper<SizeT, IntT, FloatT> {
    /// Phantom marker for the three numeric type parameters.
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> ContextWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    // def get_clr_weight(puc_scores: NDFloatArray, i: int, j: int):
    //     score = puc_scores[(i, j)]
    //     scores_i = np.concat([puc_scores[:i, i], puc_scores[(i+1):, i]])
    //     scores_j = np.concat([puc_scores[:j, j], puc_scores[(j+1):, j]])
    //     diff_i = score - np.mean(scores_i)
    //     diff_j = score - np.mean(scores_j)
    //     var_i = np.var(scores_i)
    //     var_j = np.var(scores_j)
    //     return np.sqrt((
    //         0.0 if (var_i == 0 or diff_i < 0) else (np.square(diff_i) / var_i)
    //     ) + (
    //         0.0 if (var_j == 0 or diff_j < 0) else (np.square(diff_j) / var_j)
    //     ))

    fn array_sub_i(in_array: ArrayView1<FloatT>, i: usize) -> Array1<FloatT> {
        let rlen = in_array.len();
        assert!(rlen > 1);
        assert!(i < rlen);
        let mut rarray = Array1::<FloatT>::zeros(rlen - 1);
        if i > 0 {
            rarray.slice_mut(s![..i]).assign(&in_array.slice(s![..i]));
        }
        if i < rlen - 1 {
            rarray
                .slice_mut(s![i..])
                .assign(&in_array.slice(s![i + 1..]));
        }
        rarray
    }

    fn close_to_zero(x: FloatT) -> bool {
        x.sub(FloatT::zero()).abs().to_f64().unwrap_or_default() < 1e-14
    }

    fn clr_term(var: FloatT, diff: FloatT) -> FloatT {
        if Self::close_to_zero(var) || diff.lt(&FloatT::zero()) {
            FloatT::zero()
        } else {
            diff.powi(2).div(var)
        }
    }

    /// Compute the CLR z-score
    pub fn get_clr_score(
        puc_scores: ArrayView2<FloatT>,
        i: usize,
        j: usize,
    ) -> FloatT {
        let score = puc_scores[(i, j)];
        let scores_i = Self::array_sub_i(puc_scores.row(i), i);
        let scores_j = Self::array_sub_i(puc_scores.row(j), j);
        let diff_i = score - scores_i.mean().unwrap_or_default();
        let diff_j = score - scores_j.mean().unwrap_or_default();
        let var_i = scores_i.var(FloatT::zero());
        let var_j = scores_j.var(FloatT::zero());
        (Self::clr_term(var_i, diff_i) + Self::clr_term(var_j, diff_j)).sqrt()
    }

    /// Initialize PUC scores from the HDF5 file generated either by
    /// [`mod@puc`]  or  [`mod@pidc`]
    ///
    /// Loads the "data/index" and "data/puc" values from the input HDF5 file,
    /// and builds a 2-D symmetric matrix such that both (i, j) and (j, i)
    /// entries are assigned the PUC scores.
    pub fn load_puc_scores(lwf: &ContextWorkflow) -> Result<Array2<FloatT>> {
        let index: Array2<IntT> = io::read_2d(&lwf.args.puc_file, "data/index")?;
        let val: Array1<FloatT> = io::read_1d(&lwf.args.puc_file, "data/puc")?;
        let nvars = lwf.args.nvars;
        let mut puc_scores: Array2<FloatT> = Array2::zeros((nvars, nvars));
        for (idx, v) in zip(index.axis_iter(Axis(0)), val.iter()) {
            let i = idx[0].to_usize().unwrap();
            let j = idx[1].to_usize().unwrap();
            puc_scores[(i, j)] = *v;
            puc_scores[(j, i)] = *v;
        }
        Ok(puc_scores)
    }

    /// Compute the PIDC scores for every `src < tgt` pair inside the
    /// `(s_range, t_range)` rectangle.
    ///
    /// Dnumerates the upper triangular pairs in the rectangle, and
    /// computes for each pair, PIDC values from [`Self::load_puc_scores`].
    fn ranges_pidc(
        st_ranges: &RangePair<usize>,
        puc_scores: ArrayView2<FloatT>,
    ) -> Result<PIDCResults<IntT, FloatT>> {
        let r_pindex: Array2<IntT> = pair_indices(st_ranges.clone());
        let mut r_pidcs =
            Array1::from_vec(vec![FloatT::zero(); r_pindex.nrows()]);
        for (idx, st_row) in r_pindex.rows().into_iter().enumerate() {
            let (src, tgt) =
                (st_row[0].to_usize().unwrap(), st_row[1].to_usize().unwrap());
            r_pidcs[idx] = Self::get_clr_score(puc_scores, src, tgt);
        }
        Ok(PIDCResults::new(r_pindex, r_pidcs))
    }

    /// Gather the range of  `(rows, cols)` assigned to the batch `bid`,
    /// and compute [`Self::ranges_pidc`].
    fn batch_pidc(
        lwf: &ContextWorkflow,
        bid: usize,
        puc_scores: ArrayView2<FloatT>,
    ) -> Result<PIDCResults<IntT, FloatT>> {
        Self::ranges_pidc(
            lwf.wdistr.pairs_2d().batch_range(bid, lwf.mpi_ifx.rank),
            puc_scores,
        )
    }
}

impl<'a> ContextWorkflow<'a> {
    /// Run the LMR-based PUC workflow.
    ///
    /// If  `self.args.nsamples` is given, generate `nrounds x nsamples`
    /// random samples, otherwise use all the `variables`.  
    /// After computing PUC values using LMR-based algorithm for each batch,
    /// merges all the resutsl, and writes the final result to
    /// [`WorkflowArgs::puc_file`].
    /// Uses LMR-based algorithm to compute PUC.
    pub fn run(&self) -> Result<()> {
        type HelperT = ContextWorkFlowHelper<i64, i32, f32>;
        type PT = PIDCResults<i32, i64>;
        let s_timer = SectionTimer::from_comm(self.mpi_ifx.comm(), ",");

        let puc_matrix = HelperT::load_puc_scores(self)?;
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Collect Samples");
            cond_info!(
                self.mpi_ifx.is_root();
                "PUC MATRIX: {:?}", puc_matrix.shape()
            );
            s_timer.reset();
        }

        let nbatches = self.wdistr.pairs_2d().num_batches();
        let bat_results: Result<Vec<_>> = (0..nbatches)
            .map(|bidx| HelperT::batch_pidc(self, bidx, puc_matrix.view()))
            .collect();

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
        let m_results = PIDCResults::merge(&bat_results);
        if log::log_enabled!(log::Level::Info) {
            s_timer.info_section("Merge Results");
            let n_vpuc = allreduce_sum(&(m_results.len()), self.mpi_ifx.comm());
            cond_info!(
                self.mpi_ifx.is_root();
                "Merged Completed : {}", n_vpuc
            );
            s_timer.reset();
        }
        m_results.save(self.mpi_ifx, &self.args.pidc_file, "pidc")?;
        //
        s_timer.info_section("Save PIDC Pairs");

        Ok(())
    }
}
