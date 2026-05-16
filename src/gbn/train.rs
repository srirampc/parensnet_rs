//! Thin training wrappers around [`lightgbm3::Booster`].
//!
//! Two flavours, each available for both raw `ndarray` views and
//! [`AnnData`] columns:
//!
//! * Plain training over the full feature matrix or an explicit row
//!   subset ([`train`], [`train_ad`]).
//! * Training with a held-out validation set and LightGBM's
//!   early-stopping callback ([`train_with_early_stopping`],
//!   [`train_with_early_stopping_ad`]).
//!
//! All routines round the input data to `n_decimals` digits via
//! [`crate::util::around`] before constructing the LightGBM
//! [`Dataset`], matching the rounding used elsewhere in the
//! pipeline.

use anyhow::Result;
use lightgbm3::{Booster, Dataset};
use ndarray::{ArrayView1, ArrayView2, Axis};

use crate::{anndata::AnnData, util::around};

// use lightgbm3_sys;
// pub fn train_min_rounds(
//     data_matrix: ArrayView2<f32>,
//     label: ArrayView1<f32>,
//     train_idx: Option<&[usize]>,
//     params: serde_json::Value,
// ) -> Result<()> {
//     let x_matrix;
//     let x_label;
//     let (x_train, label_train) = if let Some(train_idx) = train_idx {
//         x_matrix = data_matrix.select(Axis(0), train_idx);
//         x_label = label.select(Axis(0), train_idx);
//         (x_matrix.view(), x_label.view())
//     } else {
//         (data_matrix, label)
//     };
//
//     let nfeatures = x_train.shape()[1] as i32;
//     let train_data = Dataset::from_slice(
//         x_train.as_slice().unwrap(),
//         label_train.as_slice().unwrap(),
//         nfeatures,
//         true,
//     )?;
//
//     unsafe {
//         lightgbm3_sys::LGBM_BoosterCreate(
//             train_data.handle,
//             params_cstring.as_ptr(),
//             &mut handle
//         );
//     };
// //lightgbm3_sys::LGBM_BoosterUpdateOneIter(handle, is_finished)
//
//     Ok(());
//
//
// }

/// Train a LightGBM regressor on `(data_matrix, label)` and return
/// the resulting [`lightgbm3::Booster`].
///
/// If `train_idx` is available, the rows it lists are first selected
/// from `data_matrix` and `label` to form the training set, otherwise 
/// the full inputs are used as it is.
/// `params` is forwarded verbatim to [`Booster::train`].
pub fn train(
    data_matrix: ArrayView2<f32>,
    label: ArrayView1<f32>,
    train_idx: Option<&[usize]>,
    params: &serde_json::Value,
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

    Ok(Booster::train(train_data, params)?)
}

/// [`AnnData`]-aware variant of [`train`].
///
/// Reads the `label_index` column as the training label and the
/// rows in `train_idx` as the predictor sub-matrix from `adata`,
/// rounds both arrays to `n_decimals`, then calls [`train`].
pub fn train_ad(
    adata: &AnnData,
    label_index: usize,
    train_idx: &[usize],
    params: &serde_json::Value,
    n_decimals: usize,
) -> Result<Booster> {
    let label = around(adata.read_column(label_index)?.view(), n_decimals);
    let x_train =
        around(adata.read_submatrix::<f32>(train_idx)?.view(), n_decimals);
    let label_train = around(label.select(Axis(0), train_idx).view(), n_decimals);
    train(x_train.view(), label_train.view(), None, params)
}

/// [`AnnData`]-aware training with a held-out validation set.
///
/// Reads the `label_index` column from `adata`, splits the matrix
/// rows into `(train_idx, val_idx)` (the two halves of `indices`),
/// rounds the resulting predictor and label arrays to `n_decimals`
/// digits, and runs [`Booster::train_with_valid`].
///
/// The validation [`Dataset`] is constructed with
/// [`Dataset::from_slice_with_reference`] so its bin boundaries
/// match the training set.
pub fn train_with_early_stopping_ad(
    adata: &AnnData,
    label_index: usize,
    indices: (&[usize], &[usize]),
    gb_params: &serde_json::Value,
    n_decimals: usize,
) -> Result<Booster> {
    let label = adata.read_column(label_index)?;
    let (train_idx, val_idx) = indices;
    let x_train =
        around(adata.read_submatrix::<f32>(train_idx)?.view(), n_decimals);
    let x_val = around(adata.read_submatrix::<f32>(val_idx)?.view(), n_decimals);
    let label_train = around(label.select(Axis(0), train_idx).view(), n_decimals);
    let label_val = around(label.select(Axis(0), val_idx).view(), n_decimals);

    let subdar = x_train.slice(ndarray::s![..5, ..5]);
    log::debug!("SUBARRAY {:?}", subdar);
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

    //let params = gb_params.as_json();
    let booster = Booster::train_with_valid(train_data, Some(val_data), gb_params)?;
    Ok(booster)
}

/// Train a LightGBM regressor with a validation set and the
/// early-stopping callback configured by `es_params`.
///
/// The two halves of `indices` (`train_idx`, `val_idx`) select the
/// rows of `data_matrix` / `label` that go into the training and
/// validation [`Dataset`]s respectively. `gb_params` is the standard
/// LightGBM parameter object and `es_params` includes early-stopping
/// parameters. Returns the [`Booster`], which  the caller
/// can use to obtain the early-stopped iteration count via
/// [`lightgbm3::Booster::num_iterations`].
pub fn train_with_early_stopping(
    data_matrix: ArrayView2<f32>,
    label: ArrayView1<f32>,
    indices: (&[usize], &[usize]),
    gb_params: &serde_json::Value,
    es_params: &serde_json::Value,
) -> Result<Booster> {
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

    //let params = gb_params.as_json();
    //let booster = Booster::train_with_valid(train_data, Some(val_data), &params)?;
    let booster = Booster::train_with_early_stopping(
        train_data, val_data, gb_params, es_params,
    )?;
    //booster.feature_importance(importance_type)
    Ok(booster)
}
