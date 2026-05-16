//! GRNBoost-style network constructors and HDF5 writers.
//!
//! This module hosts the key functions of the GBN pipeline:
//!
//! 1. [`feature_importances`] / [`feature_importances_ad`] that trains a 
//!    set of targets, running one [`lightgbm3::Booster`] per target gene
//!    via [`train`] and extracts gain-based feature importances with
//!    [`get_importances`].
//! 2. MPI driver [`mpi_gradient_boosting_grn`] that computes the
//!    block range of targets owned by each rank, optionally caches
//!    the target expression matrix in a [`GeneSetAD`], and turns the
//!    importance matrix into a flat [`Array1`] of [`TFOutEdge`].
//! 3. HDF5 writers [`write_h5`] (single-rank) and [`mpi_write_h5`]
//!    (gathers all edges to the root before writing) that emit the
//!    `/gbnet/{data,target,tf}` datasets.

use std::ops::Range;

use anyhow::{Result, ensure};
use hdf5::{H5Type, types::VarLenUnicode};
use itertools::Itertools;
use lightgbm3::{Booster, ImportanceType};
use mpi::traits::Equivalence;
use ndarray::{Array1, Array2, ArrayView1};
use sope::collective::gatherv_full_vec;
use std::str::FromStr;

use super::{GBMParams, train};
use crate::{
    anndata::{AnnData, GeneSetAD},
    comm::CommIfx,
    util::block_range,
};

/// Single edge of the inferred TF→target network.
///
/// Marked `repr(C)` and `H5Type` so it can be written directly as a
/// compound HDF5 dataset (`/gbnet/data`) and exchanged via MPI as a
/// single [`Equivalence`] type.
#[derive(H5Type, Clone, PartialEq, Debug, Default, Equivalence)]
#[repr(C)]
pub struct TFOutEdge {
    /// Index of the source transcription factor in the TF list
    /// (offset already applied by [`Self::from_matrix`]).
    tf: u32,
    /// Index of the target gene in the AnnData matrix (offset
    /// already applied by [`Self::from_matrix`]).
    target: u32,
    /// Gain-based feature importance produced by LightGBM
    /// ([`ImportanceType::Gain`]).
    importance: f32,
}

impl TFOutEdge {
    /// Direct field-wise constructor.
    pub fn new(tf_id: u32, gene_id: u32, wt: f32) -> Self {
        Self {
            tf: tf_id,
            target: gene_id,
            importance: wt,
        }
    }

    /// Convert a `(|target| x |TF|)` importance matrix into a flat
    /// [`Array1`] of [`TFOutEdge`].
    ///
    /// Iterates the matrix in row-major order and emits one edge
    /// for every strictly positive entry. Row indices 
    ///  and column indices are shifted by `tf_offset/tgt_offset` so the 
    ///  resulting `tf` / `target` fields are global gene indices.
    pub fn from_matrix(
        weights_matrix: Array2<f64>,
        tgt_offset: usize,
        tf_offset: usize,
    ) -> Array1<TFOutEdge> {
        let tf_tgt_net: Vec<TFOutEdge> = weights_matrix
            .indexed_iter()
            .filter_map(|((i, j), x)| {
                if *x > 0.0 {
                    Some(TFOutEdge::new(
                        (j + tf_offset) as u32,
                        (i + tgt_offset) as u32,
                        *x as f32,
                    ))
                } else {
                    None
                }
            })
            .collect();
        Array1::from_vec(tf_tgt_net)
    }
}

/// Single-rank writer for the inferred network.
///
/// Creates `out_file` and writes three datasets under the `/gbnet`
/// group:
///
/// * `data`  — the [`TFOutEdge`] records (compound HDF5 type).
/// * `target` — the AnnData gene names (one per target row of the
///   expression matrix), as variable-length unicode strings.
/// * `tf`     — the names of the genes referenced by `tf_indices`.
pub fn write_h5(
    adata: &AnnData,
    tf_indices: &[usize],
    edges: &Array1<TFOutEdge>,
    out_file: &str,
) -> Result<()> {
    let tgt_genes: Array1<VarLenUnicode> = adata
        .genes_ref()
        .iter()
        .map(|x| VarLenUnicode::from_str(x).unwrap())
        .collect();
    let tf_genes: Array1<VarLenUnicode> = tf_indices
        .iter()
        .map(|i| VarLenUnicode::from_str(adata.gene_at(*i)).unwrap())
        .collect();
    let h5_file = hdf5::File::with_options().create(out_file)?;
    let h5_group = h5_file.create_group("/gbnet")?;
    h5_group
        .new_dataset_builder()
        .empty::<TFOutEdge>()
        .shape(hdf5::Extents::from(edges.len()))
        .create("data")?
        .as_writer()
        .write(edges)?;

    h5_group
        .new_dataset_builder()
        .empty::<VarLenUnicode>()
        .shape(hdf5::Extents::from(tgt_genes.len()))
        .create("target")?
        .write(&tgt_genes)?;

    h5_group
        .new_dataset_builder()
        .empty::<VarLenUnicode>()
        .shape(hdf5::Extents::from(tf_genes.len()))
        .create("tf")?
        .write(&tf_genes)?;
    Ok(())
}

/// MPI variant of [`write_h5`].
///
/// Gathers every rank's edge slice onto rank 0 with
/// [`gatherv_full_vec`] and then dispatches to [`write_h5`] there.
/// Non-root ranks write nothing.
/// TODO:: The truly-distributed HDF5 writing.
pub fn mpi_write_h5(
    adata: &AnnData,
    tf_indices: &[usize],
    edges: &Array1<TFOutEdge>,
    out_file: &str,
    mcx: &CommIfx,
) -> Result<()> {
    let all_edges = gatherv_full_vec(edges.as_slice().unwrap(), 0, mcx.comm())?;
    if mcx.is_root()
        && let Some(all_edges) = all_edges
    {
        write_h5(adata, tf_indices, &Array1::from_vec(all_edges), out_file)?;
    }

    // TODO:: the distributed write fails; debug this later
    //let tgt_range = block_range(mcx.rank, mcx.size, adata.genes_ref().len());
    //let tgt_genes: Array1<VarLenUnicode> = Array1::from_vec(
    //    adata.genes_ref()[tgt_range]
    //        .iter()
    //        .map(|x| VarLenUnicode::from_str(x).unwrap())
    //        .collect(),
    //);

    //let tf_range = block_range(mcx.rank, mcx.size, tf_indices.len());
    //let tf_genes: Array1<VarLenUnicode> = Array1::from_vec(
    //    tf_indices[tf_range]
    //        .iter()
    //        .map(|i| VarLenUnicode::from_str(adata.gene_at(*i)).unwrap())
    //        .collect(),
    //);

    //let h_file = mpio::create_file(mcx, out_file)?;
    //let h_group = h_file.create_group("/gbnet")?;
    //mpio::block_write1d(mcx, &h_group, "data", edges)?;
    //mpio::block_write1d(mcx, &h_group, "target", &tgt_genes)?;
    //mpio::block_write1d(mcx, &h_group, "tf", &tf_genes)?;

    Ok(())
}

/// Read gain-based feature importances out of a trained
/// [`lightgbm3::Booster`] and re-align them to the full TF list.
///
/// When the target gene is itself in the TF set, the predictor
/// matrix passed to LightGBM had its corresponding column dropped;
/// this function inserts a zero importance at the missing TF index
/// so the returned vector has length [`GeneSetAD::len`] regardless.
fn get_importances(
    tf_set: &GeneSetAD<f32>,
    tgt_id: usize,
    t_booster: Booster,
) -> Result<Vec<f64>> {
    let importances = t_booster.feature_importance(ImportanceType::Gain)?;
    let updated_importances = if tf_set.contains(tgt_id) {
        ensure!(importances.len() == tf_set.len() - 1);
        let tlen = importances.len();
        let mut r_importances = vec![0f64; tf_set.len()];
        let i = tf_set.gene_index(tgt_id)?;
        if 0 < i && i < tlen {
            r_importances[..i]
                .as_mut()
                .clone_from_slice(&importances[..i]);
            r_importances[(i + 1)..]
                .as_mut()
                .clone_from_slice(&importances[i..]);
        } else if i == 0 {
            r_importances[1..].as_mut().clone_from_slice(&importances);
        } else {
            r_importances[..tlen]
                .as_mut()
                .clone_from_slice(&importances);
        }
        r_importances
    } else {
        ensure!(importances.len() == tf_set.len());
        importances
    };
    Ok(updated_importances)
}

/// Train one LightGBM regressor predicting the target's expression
/// vector from the TF expression matrix.
///
/// When the target is one of the TFs, use a submatrix that drops that column
/// from the predictors so the model; Otherwise the full matrix is used.
fn train_for_target(
    tf_set: &GeneSetAD<f32>,
    tgt_label: ArrayView1<f32>,
    tgt_id: usize,
    params: &GBMParams,
) -> Result<Booster> {
    let params = params.as_json();
    let t_booster = if tf_set.contains(tgt_id) {
        // TODO:: Use the cache for gene_id
        let expr_mat = tf_set.expr_matrix_sub_gene_index(tgt_id)?;
        train(expr_mat.view(), tgt_label.view(), None, &params)?
    } else {
        train(
            tf_set.expr_matrix_ref().view(),
            tgt_label.view(),
            None,
            &params,
        )?
    };
    Ok(t_booster)
}

/// Compute the `(|tgt_range| x |TF|)` importance matrix for a
/// contiguous range of targets, reading labels from a [`GeneSetAD`] cache.
///
/// For each `tgt_idx` in `tgt_range` the corresponding column of
/// `tgt_set` is the regression label and the importances returned
/// by [`get_importances`] populate row `tgt_idx - tgt_range.start`
/// of the output matrix.
pub fn feature_importances(
    tf_set: &GeneSetAD<f32>,
    tgt_set: &GeneSetAD<f32>,
    tgt_range: Range<usize>,
    params: &GBMParams,
) -> Result<Array2<f64>> {
    let tgt_start = tgt_range.start;
    let tgt_size = tgt_range.end - tgt_range.start;
    let mut tgt_importances = Array2::<f64>::zeros((tgt_size, tf_set.len()));
    for tgt_idx in tgt_range {
        let tgt_label = tgt_set.column(tgt_idx)?;
        let t_booster =
            train_for_target(tf_set, tgt_label.view(), tgt_idx, params)?;
        let t_weights = get_importances(tf_set, tgt_idx, t_booster)?;
        tgt_importances
            .row_mut(tgt_idx - tgt_start)
            .assign(&Array1::from_vec(t_weights));
    }
    Ok(tgt_importances)
}

/// Compute the `(|tgt_range| x |TF|)` importance matrix for a
/// contiguous range of targets, reading labels from `adata`.
///
/// For each `tgt_idx` in `tgt_range` the corresponding column of
/// `tgt_set` is the regression label and the importances returned
/// by [`get_importances`] populate row `tgt_idx - tgt_range.start`
/// of the output matrix.
pub fn feature_importances_ad(
    tf_set: &GeneSetAD<f32>,
    adata: &AnnData,
    tgt_range: Range<usize>,
    params: &GBMParams,
) -> Result<Array2<f64>> {
    let tgt_start = tgt_range.start;
    let tgt_size = tgt_range.end - tgt_range.start;
    let mut tgt_importances = Array2::<f64>::zeros((tgt_size, tf_set.len()));
    for tgt_idx in tgt_range {
        let tgt_label = adata.read_column(tgt_idx)?;
        let t_booster =
            train_for_target(tf_set, tgt_label.view(), tgt_idx, params)?;
        let t_weights = get_importances(tf_set, tgt_idx, t_booster)?;
        tgt_importances
            .row_mut(tgt_idx - tgt_start)
            .assign(&Array1::from_vec(t_weights));
    }
    Ok(tgt_importances)
}

/// Distributed driver of the GBN inference loop.
///
/// Computes the rank's contiguous block of target genes,
/// runs either [`feature_importances`] (when `cache_tgt` is `true`, 
/// with the target columns pre-loaded into a temporary [`GeneSetAD`]) or
/// [`feature_importances_ad`] (otherwise), and returns the
/// resulting importances as a flat [`Array1`] of [`TFOutEdge`]s.
/// NOTE:: The returned array contains only this rank's edges.
pub fn mpi_gradient_boosting_grn(
    tf_set: &GeneSetAD<f32>,
    mpi_ifx: &CommIfx,
    params: GBMParams,
    cache_tgt: bool,
) -> Result<Array1<TFOutEdge>> {
    let adata = tf_set.ann_data();
    let tgt_range = block_range(mpi_ifx.rank, mpi_ifx.size, adata.nvars);
    let tgt_start = tgt_range.start;
    let tgt_importances = if cache_tgt {
        let tgt_set = GeneSetAD::from_indices(
            adata,
            &tgt_range.clone().collect_vec(),
            tf_set.decimals(),
        )?;
        feature_importances(tf_set, &tgt_set, tgt_range, &params)?
    } else {
        feature_importances_ad(tf_set, adata, tgt_range, &params)?
    };

    let tf_tgt_net = TFOutEdge::from_matrix(tgt_importances, 0, tgt_start);
    Ok(tf_tgt_net)
}
