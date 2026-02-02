use anyhow::Result;
use lightgbm3::{Booster, Dataset};
use ndarray::{ArrayView1, ArrayView2, Axis};

use super::args::GBMParams;
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

pub fn train_with_early_stopping_ad(
    adata: &AnnData,
    label_index: usize,
    indices: (&[usize], &[usize]),
    gb_params: GBMParams,
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
    Ok(booster)
}

pub fn train_with_early_stopping(
    data_matrix: ArrayView2<f32>,
    label: ArrayView1<f32>,
    indices: (&[usize], &[usize]),
    gb_params: GBMParams,
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

    let params = gb_params.as_json();
    let booster = Booster::train_with_valid(train_data, Some(val_data), &params)?;
    //booster.feature_importance(importance_type)
    Ok(booster)
}
