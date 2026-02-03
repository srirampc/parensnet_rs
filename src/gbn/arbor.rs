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
    //h5::mpio::create_write1d,
    util::block_range,
};

#[derive(H5Type, Clone, PartialEq, Debug, Default, Equivalence)] // register with HDF5
#[repr(C)]
pub struct TFOutEdge {
    tf: u32,
    target: u32,
    importance: f32,
}

impl TFOutEdge {
    pub fn new(tf_id: u32, gene_id: u32, wt: f32) -> Self {
        Self {
            tf: tf_id,
            target: gene_id,
            importance: wt,
        }
    }

    // Weights matrix is expected of size |Target| x |TF|
    // Output is a array of {row_index+offset, column_index+offset, wt}
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

    // TODO:: the distributed write hangs debug this later
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
