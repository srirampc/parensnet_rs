use anyhow::{Ok, Result};
use hdf5::H5Type;
use lightgbm3::{Booster, Dataset};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::{anndata::AnnData, util::around};

#[derive(Debug, Clone)]
pub struct GBMParams {
    pub verbose: i32,
    pub num_threads: i32,
    pub num_iterations: usize,
    pub early_stopping_rounds: usize,
    pub bagging_fraction: f32,
    pub bagging_freq: i32,
    pub metric: String,
}

impl Default for GBMParams {
    fn default() -> Self {
        GBMParams {
            verbose: 1,
            num_threads: 0,
            num_iterations: 100,
            early_stopping_rounds: 10,
            bagging_fraction: 0.9,
            bagging_freq: 1,
            metric: "rmse".to_string(),
        }
    }
}

impl GBMParams {
    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json! {{
            "verbose": self.verbose,
            "num_threads": self.num_threads,
            "num_iterations": self.num_iterations,
            "early_stopping_rounds": self.early_stopping_rounds,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "objective": "regression",
            "metric": self.metric,
        }}
    }
}

// Cross-Validation Configuration
#[derive(Debug, Clone)]
pub struct CVConfig {
    pub n_folds: usize,        // Number of CV folds (typically 5)
    pub n_sample_genes: usize, // Number of genes to sample for CV (100-1000)
    pub max_rounds: usize,     // Maximum boosting rounds to try (e.g., 500)
    pub early_stopping_rounds: usize, // Patience for early stopping (e.g., 10)
    // min_rounds: usize,            // Minimum rounds to train (e.g., 20)
    pub params: GBMParams,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            n_sample_genes: 200, // Sample 200 genes for CV
            max_rounds: 500,
            early_stopping_rounds: 10,
            params: GBMParams::default(),
        }
    }
}

#[derive(H5Type, Clone, PartialEq, Debug)] // register with HDF5
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

pub fn train(
    data_matrix: ArrayView2<f32>,
    label: ArrayView1<f32>,
    train_idx: Option<&[usize]>,
    params: serde_json::Value,
) -> Result<Booster> {
    let x_matrix;
    let x_label;
    let (x_train, label_train) = if let Some(train_idx) = train_idx {
        x_matrix = data_matrix.select(Axis(0), train_idx);
        x_label = label.select(Axis(0), train_idx);
        (x_matrix.view(), x_label.view())
    } else {
        (data_matrix, label)
    };

    let nfeatures = x_train.shape()[1] as i32;
    let train_data = Dataset::from_slice(
        x_train.as_slice().unwrap(),
        label_train.as_slice().unwrap(),
        nfeatures,
        true,
    )?;

    Ok(Booster::train(train_data, &params)?)
}

pub fn train_ad(
    adata: &AnnData,
    label_index: usize,
    train_idx: &[usize],
    params: serde_json::Value,
    n_decimals: usize,
) -> Result<Booster> {
    let label = around(adata.read_column(label_index)?.view(), n_decimals);
    let x_train =
        around(adata.read_submatrix::<f32>(train_idx)?.view(), n_decimals);
    let label_train = around(label.select(Axis(0), train_idx).view(), n_decimals);
    train(x_train.view(), label_train.view(), None, params)
}

// Early Stopping for Single Fold
pub struct EarlyStoppingResult {
    best_iteration: i32,
    //TODO:: removed this since lightgbm3 doesn't expose the score
    //best_score: f64,
    //training_history: Vec<f64>,
}

pub fn train_with_early_stopping_ad(
    adata: &AnnData,
    label_index: usize,
    indices: (&[usize], &[usize]),
    gb_params: GBMParams,
    n_decimals: usize,
) -> Result<EarlyStoppingResult> {
    let label = adata.read_column(label_index)?;
    let (train_idx, val_idx) = indices;
    let x_train =
        around(adata.read_submatrix::<f32>(train_idx)?.view(), n_decimals);
    let x_val = around(adata.read_submatrix::<f32>(val_idx)?.view(), n_decimals);
    let label_train = around(label.select(Axis(0), train_idx).view(), n_decimals);
    let label_val = around(label.select(Axis(0), val_idx).view(), n_decimals);

    let subdar = x_train.slice(ndarray::s![..5, ..5]);
    println!("SUBARRAY {:?}", subdar);
    let train_data = Dataset::from_slice(
        x_train.as_slice().unwrap(),
        label_train.as_slice().unwrap(),
        train_idx.len() as i32,
        true,
    )?;
    let val_data = Dataset::from_slice_with_reference(
        x_val.flatten().as_slice().unwrap(),
        label_val.as_slice().unwrap(),
        val_idx.len() as i32,
        true,
        Some(&train_data),
    )?;

    let params = gb_params.as_json();
    let booster = Booster::train_with_valid(train_data, Some(val_data), &params)?;
    Ok(EarlyStoppingResult {
        best_iteration: booster.num_iterations(),
    })
}

pub fn train_with_early_stopping(
    data_matrix: ArrayView2<f32>,
    label: ArrayView1<f32>,
    indices: (&[usize], &[usize]),
    gb_params: GBMParams,
) -> Result<EarlyStoppingResult> {
    let nfeatures = data_matrix.shape()[1] as i32;
    let (train_idx, val_idx) = indices;
    let x_train = data_matrix.select(Axis(0), train_idx);
    let x_val = data_matrix.select(Axis(0), val_idx);
    let label_train = label.select(Axis(0), train_idx);
    let label_val = label.select(Axis(0), val_idx);

    //let subdar = x_train.slice(ndarray::s![..5, ..5]);
    //println!("TRAIN {:?};; TEST {:?}", x_train.shape(), x_val.shape());
    let train_data = Dataset::from_slice(
        x_train.as_slice().unwrap(),
        label_train.as_slice().unwrap(),
        nfeatures,
        true,
    )?;
    let val_data = Dataset::from_slice_with_reference(
        x_val.flatten().as_slice().unwrap(),
        label_val.as_slice().unwrap(),
        nfeatures,
        true,
        Some(&train_data),
    )?;

    let params = gb_params.as_json();
    let booster = Booster::train_with_valid(train_data, Some(val_data), &params)?;
    //booster.feature_importance(importance_type)
    Ok(EarlyStoppingResult {
        best_iteration: booster.num_iterations(),
    })
}

mod arbor;
mod cv;

pub use cv::{
    OptimalGBM, cross_validate_target, mpi_optimal_gbm_iterations,
    optimal_gbm_iterations,
};

pub use arbor::{
    feature_importances, feature_importances_ad, mpi_gradient_boosting_grn,
};
