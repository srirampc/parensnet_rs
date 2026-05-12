//! Information measures over discrete (binned) joint distributions.
//!
//! This module turns joint and marginal histograms — typically produced by
//! [`crate::hist::bb_joint_histogram`] from continuous gene-expression data —
//! into the mutual-information–based quantities consumed by the network
//! construction kernels in [`crate::mcpn`], [`crate::pucn`] and
//! [`crate::mvim::misi`].
//!
//! The functions cover three families of measures:
//!
//! * **Mutual information (MI)** — [`mi_from_tab`], [`mi_from_ljvi`] and the
//!   convenience [`mi_from_data_with_bb`] which discretizes raw samples with
//!   Bayesian-blocks histograms before computing `I(X;Y)`.
//! * **Specific information (SI)** — [`si_from_tab`] / [`si_from_ljvi`] return
//!   per-bin contributions of one variable to information about the other and
//!   feed the [`redundancy`] term used by partial-information decomposition.
//! * **Log marginal ratios (LMR)** — [`lmr_from_histogram`],
//!   [`lmr_from_ljvi`], [`lmr_about_x_from_ljvi`] and
//!   [`lmr_about_y_from_ljvi`] return the per-bin mutual-information density
//!   summed along one axis of the joint table. "LMR" is local terminology
//!   (not a standard literature acronym) for these axis-marginalized sums of
//!   the log-ratio table.
//!
//! Helpers [`log_function`], [`log`] and [`log_jvi_ratio`] centralize the
//! choice of logarithm base via [`LogBase`] and the log-ratio
//! `log(P(X,Y) / (P(X) P(Y)))` that all of the measures above share.
//!
//! All log/ratio helpers replace `NaN` and `±∞` results (produced by zero
//! marginals or empty bins) with zero so they can be summed safely.

use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis};
use num::traits::float::TotalOrder;
use num::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::hist::bb_joint_histogram;
use crate::types::{AssignOps, LogBase};

/// Marker trait for floats that can carry information-measure computations.
///
/// It bundles the numeric capabilities the routines in this module need:
/// [`Float`] for log/arithmetic, [`AssignOps`] for in-place ndarray ops,
/// [`FromPrimitive`] to convert sample counts, and [`Debug`]/[`Clone`] for
/// diagnostics. Blanket-implemented for any type that satisfies the bounds,
/// so `f32` and `f64` are valid out of the box.
pub trait IMFloat: Float + AssignOps + FromPrimitive + Debug + Clone {}
impl<T: Float + AssignOps + FromPrimitive + Debug + Clone> IMFloat for T {}

/// Return the scalar logarithm function selected by `tbase`.
///
/// Used by routines that need to apply the chosen log base inside a closure
/// (e.g. inside [`log_jvi_ratio`]) without branching on every element.
pub fn log_function<T: Float>(tbase: LogBase) -> impl Fn(T) -> T {
    match tbase {
        LogBase::Two => T::log2,
        LogBase::Ten => T::log10,
        LogBase::Natural => T::ln,
    }
}

/// Element-wise logarithm of an n-dimensional array using the chosen base.
///
/// Wraps `ndarray`'s `log2`/`log10`/`ln` so callers do not need to branch
/// on [`LogBase`] themselves. The output has the same shape as `pdata`.
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

/// Pointwise log-ratio `log( xy(i,j) * tweight / (x_tab[i] * y_tab[j]) )`.
///
/// `xy_tab` is the joint histogram of `(X, Y)`, `x_tab` and `y_tab` are the
/// matching marginals (typically `xy_tab.sum_axis(Axis(1))` and
/// `xy_tab.sum_axis(Axis(0))`), and `tweight` is the total mass — usually
/// the sample count `N` so that `xy / N`, `x / N`, `y / N` are probabilities.
///
/// This is the kernel of the mutual-information sum
/// `I(X;Y) = Σ p(x,y) · log(p(x,y) / (p(x) p(y)))`. Cells with zero marginals
/// produce `NaN`/`±∞`; those entries are zeroed in place so the table can be
/// reduced safely by [`mi_from_ljvi`], [`si_from_ljvi`] and the `lmr_*`
/// helpers.
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

/// Mutual information `I(X;Y)` from a joint histogram and its marginals.
///
/// `tweight` defaults to `x_tab.sum()` (i.e. the sample count) when
/// `opt_weight` is `None`. Computed as
/// `Σ_{i,j} xy_tab[i,j] · log(xy_tab[i,j] · tweight / (x_tab[i] · y_tab[j])) / tweight`,
/// which equals the textbook MI when the inputs are raw counts. The log base
/// is selected by `tbase`, so the result is in bits ([`LogBase::Two`]),
/// nats ([`LogBase::Natural`]) or bans ([`LogBase::Ten`]).
pub fn mi_from_tab<T>(
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
    tbase: LogBase,
    opt_weight: Option<T>,
) -> T
where
    T: 'static + IMFloat,
{
    let tweight = opt_weight.unwrap_or(x_tab.sum());
    let mut mi_prod = log_jvi_ratio(xy_tab, x_tab, y_tab, tbase, tweight);
    mi_prod *= &xy_tab;
    mi_prod.sum() / tweight
}

/// Mutual information from a precomputed log-ratio table.
///
/// `ljvi_ratio` is the output of [`log_jvi_ratio`] and must have the same
/// shape as `xy_tab`. `tweight` defaults to `xy_tab.sum()`. This variant is
/// preferred when the same `(X,Y)` pair feeds into several measures (MI, SI,
/// LMR) so the log-ratio table can be computed once and shared.
pub fn mi_from_ljvi<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> T
where
    T: 'static + IMFloat,
{
    let tweight = opt_weight.unwrap_or(xy_tab.sum());

    let mi_sum = xy_tab
        .iter()
        .zip(ljvi_ratio.iter())
        .fold(T::zero(), |acc, (a, b)| acc + (*a * *b));
    mi_sum / tweight
}

/// End-to-end mutual information from raw paired samples.
///
/// Discretizes `x_data` and `y_data` with [`bb_joint_histogram`]
/// (Bayesian-blocks edges per axis) and feeds the resulting joint and
/// marginal tables into [`mi_from_tab`] with `tweight = x_data.len()`.
/// Convenient for one-off pairs; the network kernels generally cache the
/// per-variable bin edges and call [`mi_from_tab`] / [`mi_from_ljvi`]
/// directly to avoid re-binning each variable many times.
pub fn mi_from_data_with_bb<T>(
    x_data: ArrayView1<T>,
    y_data: ArrayView1<T>,
    tbase: LogBase,
) -> T
where
    T: 'static + IMFloat + TotalOrder,
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

/// Specific information `(SI_X, SI_Y)` from a precomputed log-ratio table.
///
/// `SI_X[i] = Σ_j (xy[i,j] / x_tab[i]) · ljvi[i,j]` is the contribution of
/// observing `X = i` to information about `Y` (and symmetrically for `SI_Y`).
/// The two arrays returned have lengths `x_tab.len()` and `y_tab.len()`
/// respectively. `NaN`/`±∞` entries (zero marginals) are dropped from the
/// reductions so the per-bin sums remain finite, matching the convention
/// used by [`redundancy`].
pub fn si_from_ljvi<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + IMFloat,
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
/// Specific information `(SI_X, SI_Y)` directly from histograms.
///
/// Computes the log-ratio table internally with [`log_jvi_ratio`] and then
/// delegates to [`si_from_ljvi`]. `tweight` defaults to `x_tab.sum()` and
/// only affects the global additive constant of the log-ratio (it cancels in
/// the SI sums). Returns `(SI_X, SI_Y)` of lengths `(x_tab.len(),
/// y_tab.len())`.
pub fn si_from_tab<T>(
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
    tbase: LogBase,
    opt_weight: Option<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + IMFloat,
{
    let tweight = opt_weight.unwrap_or(x_tab.sum());
    let ljvi_ratio = log_jvi_ratio(xy_tab, x_tab, y_tab, tbase, tweight);
    si_from_ljvi(ljvi_ratio.view(), xy_tab, x_tab, y_tab)
}

/// Williams–Beer redundancy `R(Z; {X, Y}) = Σ_z p(z) · min(SI_X(z), SI_Y(z))`.
///
/// `z_tab` is the marginal histogram over `Z` and `x_si`, `y_si` are the
/// specific informations of `Z` given `X` and `Y` respectively (i.e. the
/// `SI_Y`-style outputs of [`si_from_tab`] applied to the `(X,Z)` and
/// `(Y,Z)` joint tables — see the `redundancy` test for an example).
/// `opt_weight` divides the running sum to convert raw counts into a
/// probability average; pass `Some(N)` for a sample count of `N`, or `None`
/// to leave the result unnormalized.
pub fn redundancy<T>(
    z_tab: ArrayView1<T>,
    x_si: ArrayView1<T>,
    y_si: ArrayView1<T>,
    opt_weight: Option<T>,
) -> T
where
    T: 'static + IMFloat,
{
    let tfactor: T = match opt_weight {
        Some(wt) => T::one() / wt,
        None => T::one(),
    };

    (0..x_si.len())
        .map(|ix| z_tab[ix] * x_si[ix].min(y_si[ix]) * tfactor)
        .fold(T::zero(), |acc, valx| acc + valx)
}

/// Log marginal ratio summed over `Y` for each value of `X`.
///
/// Returns a length-`xsize` array whose `i`-th entry is
/// `Σ_j xy_tab[i,j] · ljvi_ratio[i,j] / tweight` — the per-`X`-bin
/// contribution to `I(X;Y)`. `tweight` defaults to `xy_tab.sum()` and
/// rescales counts to probabilities. Panics if `ljvi_ratio` and `xy_tab` do
/// not share the same shape.
pub fn lmr_about_x_from_ljvi<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> Array1<T>
where
    T: 'static + IMFloat,
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

/// Log marginal ratio summed over `X` for each value of `Y`.
///
/// Mirror of [`lmr_about_x_from_ljvi`]: returns a length-`ysize` array whose
/// `j`-th entry is `Σ_i xy_tab[i,j] · ljvi_ratio[i,j] / tweight`. `tweight`
/// defaults to `xy_tab.sum()`. Panics if `ljvi_ratio` and `xy_tab` do not
/// share the same shape.
pub fn lmr_about_y_from_ljvi<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> Array1<T>
where
    T: 'static + IMFloat,
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

/// Compute both LMR projections from a precomputed log-ratio table.
///
/// Returns the pair `(lmr_about_y, lmr_about_x)` — i.e. the first array has
/// length `ysize` ([`lmr_about_y_from_ljvi`]) and the second has length
/// `xsize` ([`lmr_about_x_from_ljvi`]). Both arrays sum to `I(X;Y)` modulo
/// the `tweight` normalization (defaulting to `xy_tab.sum()`). Panics if
/// `ljvi_ratio` and `xy_tab` differ in shape.
pub fn lmr_from_ljvi<T>(
    ljvi_ratio: ArrayView2<T>,
    xy_tab: ArrayView2<T>,
    opt_weight: Option<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + IMFloat,
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

/// Compute both LMR projections directly from histograms.
///
/// Builds the log-ratio table with [`log_jvi_ratio`] (using `tweight =
/// xy_tab.sum()` when `opt_weight` is `None`) and forwards to
/// [`lmr_from_ljvi`]. Returns `(lmr_about_y, lmr_about_x)` of lengths
/// `(ysize, xsize)`. Panics if the joint and marginal shapes are
/// inconsistent.
pub fn lmr_from_histogram<T>(
    xy_tab: ArrayView2<T>,
    x_tab: ArrayView1<T>,
    y_tab: ArrayView1<T>,
    tbase: LogBase,
    opt_weight: Option<T>,
) -> (Array1<T>, Array1<T>)
where
    T: 'static + IMFloat,
{
    let (xsize, ysize) = xy_tab.dim();
    assert_eq!((xsize, ysize), (x_tab.len(), y_tab.len()));
    //
    let tweight = opt_weight.unwrap_or(xy_tab.sum());
    //let tweight = if let Some(twt) = opt_weight { twt } else { x_tab.sum() };
    let ljvi_ratio = log_jvi_ratio(xy_tab, x_tab, y_tab, tbase, tweight);
    lmr_from_ljvi(ljvi_ratio.view(), xy_tab, opt_weight)
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
