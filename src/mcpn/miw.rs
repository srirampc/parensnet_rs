//! Batched, MPI-distributed driver for the MCPNet MI kernels.
//!
//! Hosts the [`MIWorkFlow`] struct and the generic
//! [`MIWorkFlowHelper`] that performs the per-rank work:
//!
//! * [`MIWorkFlowHelper::construct_bspline_weights`] reads each
//!   rank's variables out of the AnnData expression matrix,
//!   evaluates [`bspline_weights`] per column, and returns a
//!   `(nvars_local, nweights)` matrix.
//! * [`MIWorkFlowHelper::construct_bspline_mi_pairs`] iterates the
//!   pair batches, loads the relevant weight rows, and 
//!   evaluates [`bspline_mi`] on every upper-triangular `(row, col)` pair.
//! * [`MIWorkFlowHelper::write_weights_h5`] and
//!   [`MIWorkFlowHelper::save_mi`] save the results with the
//!   parallel-HDF5 writers.
//!
//! [`MIResults`] is the `(index matrix, value vector)` container, serves as
//! a data store for the computed results.

use anyhow::{Ok, Result};
use hdf5::H5Type;
use mpi::traits::Equivalence;
use ndarray::{Array1, Array2};
use sope::{reduction::allreduce_sum, timer::CumulativeTimer};
use std::marker::PhantomData;

use crate::{
    anndata::AnnData,
    comm::CommIfx,
    cond_debug, cond_info,
    corr::mi::{bspline_mi, bspline_weights},
    h5::mpio,
    types::{PNFloat, PNInteger},
    util::{IdVResults, PairWorkDistributor},
};

use super::WorkflowArgs;

/// Per-stage workflow context for the MCPNet B-spline MI pipeline.
///
/// Borrowed references to:
/// * an MPI communicator interface ([`CommIfx`]),
/// * the input expression matrix ([`AnnData`]),
/// * the parsed configuration ([`WorkflowArgs`]),
/// * the 2-D pair work distribution ([`PairWorkDistributor`]),
/// * a cumulative timer used to accumulate IO time ([`CumulativeTimer`]).
pub struct MIWorkFlow<'a> {
    /// MPI communicator interface (rank/size + `comm()`).
    pub comm_ifx: &'a CommIfx,
    /// AnnData handle providing access to the expression matrix.
    pub adata: &'a AnnData,
    /// Parsed workflow configuration.
    pub args: &'a WorkflowArgs,
    /// Pair work distribution across batches.
    pub wf_dist: &'a PairWorkDistributor,
    /// Cumulative timer used to accumulate IO time across stages.
    pub io_timer: CumulativeTimer<'a>,
}

/// Container for the MI computation results: index matrix + value vector. 
/// Type alias of [`IdVResults`] specialised to the integer/float types.
pub type MIResults<IntT, FloatT> = IdVResults<IntT, FloatT>;
/// Type-parametric collection of helper functions used by
/// [`MIWorkFlow`].
///
/// The type parameters select the integer/float widths used for
/// the various pieces of data:
///
/// * `SizeT` — HDF5 dimension type used by the parallel writer.
/// * `IntT`  — integer width used for the `(row, col)` pair
///   indices stored alongside each MI value.
/// * `FloatT` — floating-point width used for the expression
///   matrix, B-spline weights, and MI values.
pub(super) struct MIWorkFlowHelper<SizeT, IntT, FloatT> {
    /// Phantom marker for the generic parameters.
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> MIWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
    /// Load the row and column halves of one batch from the
    /// HDF5 weights dataset.
    ///
    /// For all the `(rows, cols)` ranges for the batch `bidx`,
    /// calls  [`mpio::read_range_data`] to retrieve weights from
    /// `args.mi_file:/<weights_ds>` datasets. 
    pub(super) fn load_batch_data(
        wf: &MIWorkFlow,
        rank: i32,
        bidx: usize,
    ) -> Result<(Array2<FloatT>, Array2<FloatT>)> {
        let (rows, cols) = wf.wf_dist.pairs_2d().batch_range(bidx, rank);
        wf.io_timer.reset();
        let wdim = wf.args.weights_dim();
        let block_data = (
            mpio::read_range_data(
                &wf.args.weights_file,
                &wf.args.weights_ds,
                0..wdim,
                rows.clone(),
                wf.comm_ifx,
            )?,
            mpio::read_range_data(
                &wf.args.weights_file,
                &wf.args.weights_ds,
                0..wdim,
                cols.clone(),
                wf.comm_ifx,
            )?,
        );
        wf.io_timer.add_elapsed();
        Ok(block_data)
    }

    /// Build this rank's slice of the B-spline weight matrix.
    ///
    /// Reads the column-striped submatrix corresponding to the variables
    /// assigned to this `rank` from the AnnData object,
    /// then evaluates [`bspline_weights`] on each column to fill a
    /// `(nvars_local, weights_dim())` output matrix.
    pub fn construct_bspline_weights(
        wf: &MIWorkFlow,
        rank: i32,
    ) -> Result<Array2<FloatT>> {
        let columns = wf.wf_dist.vars_dist()[rank as usize].clone();
        wf.io_timer.reset();
        let rdata = wf.adata.par_read_range_data_around::<FloatT>(
            columns.clone(),
            wf.args.nroundup,
            wf.comm_ifx,
        )?;
        let nvars = columns.end - columns.start;
        let nweights = wf.args.weights_dim();
        // TODO::
        let mut spline_weights: Array2<FloatT> = Array2::zeros((nvars, nweights));
        columns.enumerate().for_each(|(i, _cx)| {
            let wt = bspline_weights(
                rdata.column(i).as_slice().unwrap_or_default(),
                wf.args.nbins,
                wf.args.spline_order,
                wf.args.nobs,
            );
            spline_weights.row_mut(i).assign(&Array1::from_vec(wt));
        });
        wf.io_timer.add_elapsed();
        Ok(spline_weights)
    }

    /// Collectively write the rank's slice of the weight matrix to
    /// `weights_data_file:/data/weights` via parallel HDF5.
    pub fn write_weights_h5(
        wf: &MIWorkFlow,
        weights_data_file: &str,
        weights: &Array2<FloatT>,
    ) -> Result<()> {
        let hfptr = mpio::create_file(wf.comm_ifx, weights_data_file)?;
        let data_group = hfptr.create_group("data")?;
        mpio::block_write2d(wf.comm_ifx, &data_group, "weights", &weights)?;
        Ok(())
    }

    /// Compute the B-spline MI for every upper-triangular pair in
    /// one batch.
    ///
    /// Loads the batch's row/column weight blocks, then, for each pair 
    /// (rx, cx), rx < cx, calls [`bspline_mi`] on the corresponding 
    /// pre-computed weight vectors and packs `(rx, cx, mi)` into the returned
    /// [`MIResults`].
    pub fn batch_mi_pairs(
        wf: &MIWorkFlow,
        rank: i32,
        bidx: usize,
    ) -> Result<MIResults<IntT, FloatT>> {
        let (row_data, col_data) = Self::load_batch_data(wf, rank, bidx)?;
        if log::log_enabled!(log::Level::Debug) {
            let n_hist = allreduce_sum(&(row_data.len()), wf.comm_ifx.comm());
            cond_debug!(
                wf.comm_ifx.is_root(); "Loaded Batch {} with : {} ", bidx, n_hist
            );
        }
        let (rows, cols) = wf.wf_dist.pairs_2d().batch_range(bidx, rank);
        let capacity = (rows.end - rows.start) * (cols.end - cols.start);
        let mut pair_indices: Vec<IntT> = Vec::with_capacity(capacity * 2);
        let mut mi_vals: Vec<FloatT> = Vec::with_capacity(capacity);
        for (i, rx) in rows.clone().enumerate() {
            let r_data = row_data.column(i);
            for (j, cx) in cols.clone().enumerate() {
                if rx < cx {
                    let c_data = col_data.column(j);
                    pair_indices.push(IntT::from_usize(rx).unwrap());
                    pair_indices.push(IntT::from_usize(cx).unwrap());
                    //let index = triu_pair_to_index(wf.args.nvars, rx, cx);
                    let rx = r_data.as_slice().unwrap_or_default();
                    let cx = c_data.as_slice().unwrap_or_default();
                    // TODO::
                    let mi = bspline_mi(rx, cx, wf.args.nbins, wf.args.nobs);
                    mi_vals.push(mi);
                }
            }
        }

        let mvr = MIResults::new(
            Array2::from_shape_vec((mi_vals.len(), 2), pair_indices)?,
            Array1::from_vec(mi_vals),
        );

        Ok(mvr)
    }

    /// Run [`Self::batch_mi_pairs`] over every batch in
    /// `wf.wf_dist.pairs_2d()` and concatenate the results.
    ///
    /// Returns a single [`MIResults`] holding all `(row, col, mi)`
    /// triples produced on this rank.
    pub fn construct_bspline_mi_pairs(
        wf: &MIWorkFlow,
        rank: i32,
    ) -> Result<MIResults<IntT, FloatT>> {
        let nbatches = wf.wf_dist.pairs_2d().num_batches();

        let v_mi = (0..nbatches)
            .map(|bidx| Self::batch_mi_pairs(wf, rank, bidx))
            .collect::<Result<Vec<_>>>()?;
        Ok(MIResults::<IntT, FloatT>::merge(&v_mi))
    }

    /// Collectively write a [`MIResults`] to `mi_file:/data` via
    /// parallel HDF5.
    ///
    /// Writes the `(n, 2)` index matrix as `data/index` and the
    /// length-`n` MI vector as `data/mi`.
    pub fn save_mi(
        mir: &MIResults<IntT, FloatT>,
        mpi_ifx: &CommIfx,
        mi_file: &str,
    ) -> Result<()> {
        let h_file = mpio::create_file(mpi_ifx, mi_file)?;
        cond_info!(
            mpi_ifx.is_root();
            "Saving Data :: {:?} {:?} {}",
            mir.index.shape(),
            mir.val.shape(),
            mi_file,
        );
        sope::gather_debug!(mpi_ifx.comm(); "{:?}", mir.val.len());
        let h_group = h_file.create_group("data").unwrap();
        mpio::block_write2d(mpi_ifx, &h_group, "index", &mir.index)?;
        mpio::block_write1d(mpi_ifx, &h_group, "mi", &mir.val)?;
        Ok(())
    }
}

impl<'a> MIWorkFlow<'a> {
    /// Execute the [`RunMode::MIBSplineWeights`](crate::mcpn::RunMode::MIBSplineWeights)
    /// stage: compute this rank's slice of the B-spline weight
    /// matrix and write it to [`WorkflowArgs::weights_file`].
    ///
    /// Uses `i64` for HDF5 dim indices, `i32` for pair indices,
    /// and `f32` weights.
    pub fn run_bspline_weights(&self) -> Result<()> {
        type HelperT = MIWorkFlowHelper<i64, i32, f32>;
        let weights =
            HelperT::construct_bspline_weights(&self, self.comm_ifx.rank)?;
        HelperT::write_weights_h5(&self, &self.args.weights_file, &weights)
    }

    /// Execute the [`RunMode::MIBSpline`](crate::mcpn::RunMode::MIBSpline)
    /// stage: compute the pairwise B-spline MI from the previously
    /// persisted weight matrix and write the `(index, mi)` pairs
    /// to [`WorkflowArgs::mi_file`].
    ///
    /// Uses `i64` HDF5 dim indices, `i32` pair indices, and `f32` MI values.
    pub fn run_bspline_mi(&self) -> Result<()> {
        type HelperT = MIWorkFlowHelper<i64, i32, f32>;
        let mir = HelperT::construct_bspline_mi_pairs(&self, self.comm_ifx.rank)?;
        HelperT::save_mi(&mir, self.comm_ifx, &self.args.mi_file)?;
        Ok(())
    }
}
