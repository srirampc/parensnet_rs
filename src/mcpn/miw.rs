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

pub struct MIWorkFlow<'a> {
    pub comm_ifx: &'a CommIfx,
    pub adata: &'a AnnData,
    pub args: &'a WorkflowArgs,
    pub wf_dist: &'a PairWorkDistributor,
    pub io_timer: CumulativeTimer<'a>,
}

pub type MIResults<IntT, FloatT> = IdVResults<IntT, FloatT>;
pub(super) struct MIWorkFlowHelper<SizeT, IntT, FloatT> {
    _a: PhantomData<(SizeT, IntT, FloatT)>,
}

impl<SizeT, IntT, FloatT> MIWorkFlowHelper<SizeT, IntT, FloatT>
where
    SizeT: 'static + PNInteger + H5Type + Default + Equivalence,
    IntT: PNInteger + H5Type + Default + Equivalence,
    FloatT: 'static + PNFloat + H5Type + Default + Equivalence,
{
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
                &wf.args.mi_file,
                &wf.args.weights_ds,
                0..wdim,
                rows.clone(),
                wf.comm_ifx,
            )?,
            mpio::read_range_data(
                &wf.args.mi_file,
                &wf.args.weights_ds,
                0..wdim,
                cols.clone(),
                wf.comm_ifx,
            )?,
        );
        wf.io_timer.add_elapsed();
        Ok(block_data)
    }

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
    pub fn run_bspline_weights(&self) -> Result<()> {
        type HelperT = MIWorkFlowHelper<i64, i32, f32>;
        let weights =
            HelperT::construct_bspline_weights(&self, self.comm_ifx.rank)?;
        HelperT::write_weights_h5(&self, &self.args.weights_file, &weights)
    }

    pub fn run_bspline_mi(&self) -> Result<()> {
        type HelperT = MIWorkFlowHelper<i64, i32, f32>;
        let mir = HelperT::construct_bspline_mi_pairs(&self, self.comm_ifx.rank)?;
        HelperT::save_mi(&mir, self.comm_ifx, &self.args.mi_file)?;
        Ok(())
    }
}
