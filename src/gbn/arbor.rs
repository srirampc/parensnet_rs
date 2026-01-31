use std::ops::Range;

use anyhow::{Result, ensure};
use itertools::Itertools;
use lightgbm3::{Booster, ImportanceType};
use ndarray::{Array1, Array2, ArrayView1};

use super::{GBMParams, TFOutEdge, train};
use crate::{
    anndata::{AnnData, GeneSetAD},
    comm::CommIfx,
    //h5::mpio::create_write1d,
    util::block_range,
};

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
    let t_booster = if tf_set.contains(tgt_id) {
        // TODO:: Use the cache for gene_id
        let expr_mat = tf_set.expr_matrix_sub_gene_index(tgt_id)?;
        train(expr_mat.view(), tgt_label.view(), None, params.as_json())?
    } else {
        train(
            tf_set.expr_matrix_ref().view(),
            tgt_label.view(),
            None,
            params.as_json(),
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

    // TODO:: Write output?
    //create_write1d(mpi_ifx, "test.h5", "/", "data", &tf_tgt_net)?;
    Ok(tf_tgt_net)
}
