use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis};
use num::{Float, FromPrimitive};

use crate::hist::bb_joint_histogram;
use crate::types::{AssignOps, DbgDisplay, LogBase, OrderedFloat};

pub fn log_function<T: Float>(tbase: LogBase) -> impl Fn(T) -> T {
    match tbase {
        LogBase::Two => T::log2,
        LogBase::Ten => T::log10,
        LogBase::Natural => T::ln,
    }
}

pub fn log<T, D>(pdata: ArrayView<T, D>, tbase: LogBase) -> Array<T, D>
where
    T: 'static + Float,
    D: ndarray::Dimension,
{
    match tbase {
        LogBase::Two => pdata.log2(),
        LogBase::Ten => pdata.log10(),
        LogBase::Natural => pdata.ln(),
    }
}

///
/// Give table for (X,Y), X, and Y
/// compute the table whose (i, j) entry is log (xy(i,j) * weight / x_i * y_j))
///
pub fn log_jvi_ratio<T>(
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
    tbase: LogBase,
    tweight: T,
) -> Array2<T>
where
    T: 'static + Float + AssignOps + Clone,
{
    let base_log_fn = log_function(tbase);
    let mut jvi_ratio =
        Array2::from_shape_fn((x_tab.len(), y_tab.len()), |(i, j)| {
            base_log_fn((xy_tab[[i, j]] * tweight) / (x_tab[i] * y_tab[j]))
        });
    jvi_ratio.map_inplace(|lgvr| {
        if lgvr.is_nan() || lgvr.is_infinite() {
            *lgvr = T::zero()
        }
    });
    jvi_ratio
}

///
/// Given table for (X,Y), X, and Y; compute mi
///
pub fn mi_from_tab<T>(
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
    tbase: LogBase,
    opt_weight: Option<T>,
) -> T
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    let tweight = opt_weight.unwrap_or(x_tab.sum());
    let mut mi_prod = log_jvi_ratio(xy_tab, x_tab, y_tab, tbase, tweight);
    mi_prod *= &xy_tab;
    mi_prod.sum() / tweight
}

///
/// Given table for log(P(X,Y)/P(X)P(Y)), compute mi
///
pub fn mi_from_ljvi<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> T
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    let tweight = opt_weight.unwrap_or(xy_tab.sum());

    let mi_sum = xy_tab
        .iter()
        .zip(ljvi_ratio.iter())
        .fold(T::zero(), |acc, (a, b)| acc + (*a * *b));
    mi_sum / tweight
}

///
/// Given data for X and Y; compute mi
///
pub fn mi_from_data_with_bb<T>(
    x_data: ArrayView1<T>,
    y_data: ArrayView1<T>,
    tbase: LogBase,
) -> T
where
    T: 'static + OrderedFloat + FromPrimitive + AssignOps + Clone + DbgDisplay,
{
    let (xy_tab, x_tab, y_tab) = bb_joint_histogram(x_data, y_data);
    mi_from_tab(
        xy_tab.view(),
        x_tab.view(),
        y_tab.view(),
        tbase,
        T::from_usize(x_data.len()),
    )
}

///
/// Given table for log(P(X,Y)/P(X)P(Y)), compute (si_x, si_y)
///
pub fn si_from_ljvi<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    let xy_shape = (x_tab.len(), y_tab.len());
    let x_ratio = Array2::from_shape_fn(xy_shape, |(i, j)| {
        xy_tab[[i, j]] * ljvi_ratio[[i, j]] / x_tab[i]
    });

    let y_ratio = Array2::from_shape_fn(xy_shape, |(i, j)| {
        xy_tab[[i, j]] * ljvi_ratio[[i, j]] / y_tab[j]
    });
    (
        x_ratio.fold_axis(Axis(1), T::zero(), |acc, x| {
            *acc + if x.is_infinite() || x.is_nan() {
                T::zero()
            } else {
                *x
            }
        }),
        y_ratio.fold_axis(Axis(0), T::zero(), |acc, y| {
            *acc + if y.is_infinite() || y.is_nan() {
                T::zero()
            } else {
                *y
            }
        }),
    )
}
///
/// Given table for (X,Y), X, and Y, compute (si_x, si_y)
///
pub fn si_from_tab<T>(
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
    tbase: LogBase,
    opt_weight: Option<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    let tweight = opt_weight.unwrap_or(x_tab.sum());
    let ljvi_ratio = log_jvi_ratio(xy_tab, x_tab, y_tab, tbase, tweight);
    si_from_ljvi(ljvi_ratio.view(), xy_tab, x_tab, y_tab)
}

pub fn redundancy<T>(
    z_tab: ArrayView1<T>,
    x_si: ArrayView1<T>,
    y_si: ArrayView1<T>,
    opt_weight: Option<T>,
) -> T
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    let tfactor: T = match opt_weight {
        Some(wt) => T::one() / wt,
        None => T::one(),
    };

    (0..x_si.len())
        .map(|ix| z_tab[ix] * x_si[ix].min(y_si[ix]) * tfactor)
        .fold(T::zero(), |acc, valx| acc + valx)
}

pub fn lmr_about_x_from_lvji<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> Array1<T>
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    assert_eq!(ljvi_ratio.dim(), xy_tab.dim());
    //   tfactor = 1.0/(np.sum(xytab) if tweight is None else tweight)
    let tfactor: T = T::one() / opt_weight.unwrap_or(xy_tab.sum());
    // elp_tab = xytab * ljvi_ratio * tfactor
    let elp_tab = Array2::<T>::from_shape_fn(xy_tab.dim(), |(i, j)| {
        xy_tab[[i, j]] * ljvi_ratio[[i, j]] * tfactor
    });
    elp_tab.sum_axis(Axis(1))
}

//
pub fn lmr_about_y_from_lvji<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> Array1<T>
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    assert_eq!(ljvi_ratio.dim(), xy_tab.dim());
    //   tfactor = 1.0/(np.sum(xytab) if tweight is None else tweight)
    let tfactor: T = T::one() / opt_weight.unwrap_or(xy_tab.sum());
    // elp_tab = xytab * ljvi_ratio * tfactor
    let elp_tab = Array2::<T>::from_shape_fn(xy_tab.dim(), |(i, j)| {
        xy_tab[[i, j]] * ljvi_ratio[[i, j]] * tfactor
    });
    elp_tab.sum_axis(Axis(0))
}

pub fn lmr_from_lvji<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    assert_eq!(ljvi_ratio.dim(), xy_tab.dim());
    //   tfactor = 1.0/(np.sum(xytab) if tweight is None else tweight)
    let tfactor: T = T::one() / opt_weight.unwrap_or(xy_tab.sum());
    // elp_tab = xytab * ljvi_ratio * tfactor
    let elp_tab = Array2::<T>::from_shape_fn(xy_tab.dim(), |(i, j)| {
        xy_tab[[i, j]] * ljvi_ratio[[i, j]] * tfactor
    });
    (elp_tab.sum_axis(Axis(0)), elp_tab.sum_axis(Axis(1)))
}

pub fn lmr_from_histogram<T>(
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
    tbase: LogBase,
    opt_weight: Option<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + Float + AssignOps + Clone + DbgDisplay,
{
    let (xsize, ysize) = xy_tab.dim();
    assert_eq!((xsize, ysize), (x_tab.len(), y_tab.len()));
    //
    let tweight = opt_weight.unwrap_or(xy_tab.sum());
    //let tweight = if let Some(twt) = opt_weight { twt } else { x_tab.sum() };
    let ljvi_ratio = log_jvi_ratio(xy_tab, x_tab, y_tab, tbase, tweight);
    lmr_from_lvji(ljvi_ratio.view(), xy_tab, opt_weight)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use itertools::Itertools;
    use lazy_static::lazy_static;
    use log::debug;
    use ndarray::{Array1, Array2, Axis, array};

    use super::{mi_from_data_with_bb, redundancy, si_from_tab};
    use crate::{
        hist::bb_joint_histogram,
        mvim::imeasures::lmr_from_histogram,
        tests::{
            PUCTestData, puc_test4_data, puc_test4_data_w_lmr,
            test_exp_sub_matrix,
        },
        types::LogBase,
        util::around,
    };

    lazy_static! {
        static ref PUC_DATA: PUCTestData = puc_test4_data().unwrap();
        static ref PUC_DATAW_LMR: PUCTestData = puc_test4_data_w_lmr().unwrap();
    }

    #[test]
    pub fn test_imeasures_log() {
        crate::tests::log_init();
        use std::f32::consts::LOG2_10;
        let test_data2: Array1<f32> = Array1::from_vec(vec![
            7.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0, 5.0, 9.0, 2.0, 4.0, 3.0, 3.0, 4.0,
            8.0, 6.0, 5.0, 8.0, 4.0, 2.0, 2.0, 5.0, 9.0, 5.0, 1.0, 10.0, 12.0,
        ]);
        let expected_out2: Array1<f32> = Array1::from_vec(vec![
            2.8074, 2.0000, 3.0000, 1.5850, 0.0000, 0.0000, 1.0000, 2.3219,
            3.1699, 1.0000, 2.0000, 1.5850, 1.5850, 2.0000, 3.0000, 2.5850,
            2.3219, 3.0000, 2.0000, 1.0000, 1.0000, 2.3219, 3.1699, 2.3219,
            0.0000, LOG2_10, 3.5850,
        ]);
        let log_test2 = super::log(test_data2.view(), LogBase::Two);
        debug!("Test Data :: {:8.4}", test_data2);
        debug!("Log Test  :: {:8.4}", log_test2);
        debug!(
            "Diff      :: {:?}",
            log_test2.abs_diff_eq(&expected_out2, 1e-4)
        );
        assert!(log_test2.abs_diff_eq(&expected_out2, 1e-4));
    }

    #[test]
    pub fn test_jvir_ratio() {
        crate::tests::log_init();
        let test_xytab: Array2<f32> = array![
            [7.0, 4.0, 8.0, 3.0, 1.0,],
            [1.0, 2.0, 5.0, 9.0, 2.0,],
            [4.0, 3.0, 3.0, 4.0, 8.0,],
            [6.0, 5.0, 8.0, 4.0, 2.0,],
            [2.0, 5.0, 9.0, 5.0, 1.0,],
            [10.0, 12.0, 3.0, 3.0, 4.0,],
        ];
        let test_x_tab = test_xytab.sum_axis(Axis(1));
        let test_y_tab = test_xytab.sum_axis(Axis(0));
        debug!("XYTAB :: {:8.4}", test_xytab);
        debug!("XTAB  :: {:8.4}", test_x_tab);
        debug!("YTAB  :: {:8.4}", test_y_tab);
        let jvi_out = super::log_jvi_ratio(
            test_xytab.view(),
            test_x_tab.view(),
            test_y_tab.view(),
            LogBase::Two,
            1.0,
        );
        debug!("JVI :: {:8.4}", jvi_out);
    }

    #[test]
    pub fn test_mi() -> Result<()> {
        crate::tests::log_init();
        for (nobs, nvars) in [(1000, 3), (10000, 3)] {
            let px_data = &PUC_DATA.data[&nobs.to_string()];
            let ematrix = test_exp_sub_matrix(0..nobs, 0..nvars)?;
            let ematrix = around(ematrix.view(), 4);
            debug!("EM:{:?}; TDN: {}", ematrix.shape(), px_data.nodes.len());
            (0..nvars).combinations(2).for_each(|vx| {
                let (x, y) = (vx[0], vx[1]);
                let mi = Array1::from_elem(1, px_data.get_mi(x, y));
                let tmi = mi_from_data_with_bb(
                    ematrix.column(x).view(),
                    ematrix.column(y).view(),
                    LogBase::Two,
                );
                debug!("X, Y:({}, {}); MI: {};  TMI: {}", x, y, mi, tmi);
                assert!(mi.abs_diff_eq(&Array1::from_elem(1, tmi), 1e-4));
            });
        }

        Ok(())
    }

    #[test]
    pub fn test_si() -> Result<()> {
        crate::tests::log_init();
        for (nobs, nvars) in [(1000, 3), (10000, 3)] {
            let px_data = &PUC_DATA.data[&nobs.to_string()];
            let ematrix = test_exp_sub_matrix(0..nobs, 0..nvars)?;
            let ematrix = around(ematrix.view(), 4);
            debug!("EM:{:?}; TDN: {}", ematrix.shape(), px_data.nodes.len());
            (0..nvars).combinations(2).for_each(|vx| {
                let (x, y) = (vx[0], vx[1]);
                let (x_si, y_si) = (px_data.get_si(x, y), px_data.get_si(y, x));
                let (xy_tab, x_tab, y_tab) = bb_joint_histogram(
                    ematrix.column(x).view(),
                    ematrix.column(y).view(),
                );
                let (x_tsi, y_tsi) = si_from_tab(
                    xy_tab.view(),
                    x_tab.view(),
                    y_tab.view(),
                    LogBase::Two,
                    None,
                );
                debug!("X,Y:({},{}):({}, {})", x, y, x_tab.len(), y_tab.len());
                debug!(
                    " -> SI:({}, {}); TSI:({}, {}); DIFF:({:?}, {:?})",
                    x_si.len(),
                    y_si.len(),
                    x_tsi.len(),
                    y_tsi.len(),
                    x_si.abs_diff_eq(&x_tsi, 1e-4),
                    y_si.abs_diff_eq(&y_tsi, 1e-4),
                );
                assert!(x_si.abs_diff_eq(&x_tsi, 1e-4));
                assert!(y_si.abs_diff_eq(&y_tsi, 1e-4));
            });
        }

        Ok(())
    }

    #[test]
    pub fn test_redundancy() -> Result<()> {
        crate::tests::log_init();
        for (nobs, nvars) in [(1000, 3), (10000, 3)] {
            let px_data = &PUC_DATA.data[&nobs.to_string()];
            let ematrix = test_exp_sub_matrix(0..nobs, 0..nvars)?;
            let ematrix = around(ematrix.view(), 4);
            debug!("RED : {:?}", px_data.redundancy_values);
            (0..nvars).combinations(3).for_each(|vx| {
                let (x, y, z) = (vx[0], vx[1], vx[2]);
                debug!("X,Y,Z:({}, {}, {})", x, y, z);
                let red = Array1::from_elem(1, px_data.get_redundancy(x, y, z));
                let (x_col, y_col, z_col) =
                    (ematrix.column(x), ematrix.column(y), ematrix.column(z));
                let (xz_tab, x_tab, _) =
                    bb_joint_histogram(x_col.view(), z_col.view());
                let (yz_tab, y_tab, z_tab) =
                    bb_joint_histogram(y_col.view(), z_col.view());
                let (_x_tsi, z_wx_tsi) = si_from_tab(
                    xz_tab.view(),
                    x_tab.view(),
                    z_tab.view(),
                    LogBase::Two,
                    None,
                );
                let (_y_tsi, z_wy_tsi) = si_from_tab(
                    yz_tab.view(),
                    y_tab.view(),
                    z_tab.view(),
                    LogBase::Two,
                    None,
                );
                let t_red = Array1::from_elem(
                    1,
                    redundancy(
                        z_tab.view(),
                        z_wx_tsi.view(),
                        z_wy_tsi.view(),
                        Some(nobs as f32),
                    ),
                );
                let rdiff = red.abs_diff_eq(&t_red, 1e-4);
                debug!(" -> red:{} RED: {} diff {}", red, t_red, rdiff);
                assert!(rdiff);
            });
        }

        Ok(())
    }

    #[test]
    pub fn test_lmr() -> Result<()> {
        crate::tests::log_init();
        //tests for lmr
        for (nobs, nvars) in [(1000, 3), (10000, 3)] {
            let px_data = &PUC_DATAW_LMR.data[&nobs.to_string()];
            let ematrix = test_exp_sub_matrix(0..nobs, 0..nvars)?;
            let ematrix = around(ematrix.view(), 4);
            debug!("EM:{:?}; ", ematrix.shape());
            (0..nvars).combinations(2).for_each(|vx| {
                let (x, y) = (vx[0], vx[1]);
                let (rx_lmr, ry_lmr) =
                    (px_data.get_lmr(x, y), px_data.get_lmr(y, x));
                let (xy_tab, x_tab, y_tab) = bb_joint_histogram(
                    ematrix.column(x).view(),
                    ematrix.column(y).view(),
                );
                let (x_lmr, y_lmr) = lmr_from_histogram(
                    xy_tab.view(),
                    x_tab.view(),
                    y_tab.view(),
                    LogBase::Two,
                    None,
                );
                debug!("X,Y:({},{}):({}, {})", x, y, x_tab.len(), y_tab.len());
                let rx_len = rx_lmr.as_ref().map(|rw| rw.len());
                let ry_len = ry_lmr.as_ref().map(|rw| rw.len());
                let rx_diff =
                    ry_lmr.as_ref().map(|rw| rw.abs_diff_eq(&x_lmr, 1e-4));
                let ry_diff =
                    ry_lmr.as_ref().map(|rw| rw.abs_diff_eq(&x_lmr, 1e-4));
                debug!(
                    " -> LMR:({}, {}); RLMR({:?}, {:?}); DIFF({:?}, {:?})",
                    x_lmr.len(),
                    y_lmr.len(),
                    rx_len,
                    ry_len,
                    rx_diff,
                    ry_diff,
                );
                assert!(rx_diff.unwrap_or(false));
                assert!(ry_diff.unwrap_or(false));
            });
        }
        Ok(())
    }
}
