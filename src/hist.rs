//! Functions to compute edges for optimal binning with the Bayesian
//! blocks algorithm, and routines to construct histograms from a set
//! of bin edges.
//!
//! The module exposes:
//!
//! * Plain histogramming primitives — [`histogram_1d`],
//!   [`histogram_2d`] and [`joint_histogram`] — that bucket data into
//!   user-supplied bin edges.
//! * The [`bayesian_blocks_bin_edges`] routine, which selects an
//!   "optimal" set of bin edges for a 1-D sample using Scargle et al.
//!   (2012) Bayesian blocks (events / point-measurement form).
//! * Convenience wrappers [`bb_histogram`] and [`bb_joint_histogram`]
//!   that combine the two: compute Bayesian-block edges, then bin.
//! * The [`HSFloat`] marker trait collecting the float bounds used
//!   throughout the file.
//!
//! All of the algorithms operate on `ndarray` views and produce owned
//! `Array1` / `Array2` outputs.

use ndarray::{Array1, Array2, ArrayView1, s};
use num::{Float, FromPrimitive, traits::float::TotalOrder};
use std::clone::Clone;
use std::fmt::Debug;

use crate::types::{AddFromZero, AssignOps};
use crate::util::unique;

/// Marker trait for the float types accepted by histogramming and
/// Bayesian-block routines in this module.
///
/// It bundles together the bounds shared by every public function:
/// [`Float`] for general arithmetic, [`TotalOrder`] for NaN-safe
/// comparisons , [`FromPrimitive`] for using `f64` constants,
/// in-crate [`AssignOps`] (in-place arithmetic) for assign operations .
/// A blanket implementation provides it for any type satisfying the bounds, so
/// `f32` and `f64` qualify automatically.
pub trait HSFloat:
    Float + TotalOrder + FromPrimitive + AssignOps + Debug + Clone
{
}
impl<T: Float + TotalOrder + FromPrimitive + AssignOps + Debug + Clone> HSFloat
    for T
{
}

/// Map a sample value to the index of the bin that contains it.
///
/// `bin_edges` must be sorted in ascending order (with respect to
/// `TotalOrder`).
/// With a binary search on `bin_edges`, identify
/// the index of the half-open bin `[edges[i], edges[i+1])` that the
/// query `qry_dx` falls into.
/// Values at or above the last edge map to `bin_edges.len() - 1`.
fn map_data2bin<T: Float + TotalOrder>(bin_edges: &[T], qry_dx: &T) -> usize {
    let fidx = match bin_edges.binary_search_by(|ex| ex.total_cmp(qry_dx)) {
        Ok(kidx) => kidx,
        Err(eidx) => eidx,
    };
    if fidx > 0 {
        if fidx <= bin_edges.len() {
            fidx - 1
        } else {
            bin_edges.len() - 1
        }
    } else {
        fidx
    }
}

/// Compute a 1-D histogram of `data` using the given bin `edges`.
///
/// The histogram has `edges.len() - 1` bins; values that fall above
/// the last edge are clamped into the final bin (matching NumPy's
/// `numpy.histogram` behavior on the right edge). `edges` is required
/// to be sorted in ascending order.
///
/// The count type `S` is generic, so callers can request integer or
/// floating counts as long as `S` implements [`AddFromZero`].
pub fn histogram_1d<T, S>(data: ArrayView1<T>, edges: &[T]) -> Array1<S>
where
    T: Float + TotalOrder + Debug,
    S: AddFromZero + Clone,
{
    let mut hist = Array1::<S>::from_elem(edges.len() - 1, S::zero());
    //let ed_slice = edges.as_slice().unwrap();
    for dx in data {
        let mut idx = map_data2bin(edges, dx);
        if idx >= hist.len() {
            idx = hist.len() - 1;
        }
        hist[idx] += S::one();
    }
    hist
}

/// Compute a 2-D histogram (joint count grid) of paired samples
/// `(x_data[i], y_data[i])`.
///
/// `x_data` and `y_data` are iterated in lock-step (so should have the
/// same length); each pair is bucketed using the corresponding edge
/// vectors, and the resulting `(x_bins, y_bins)` count matrix has
/// shape `(x_bin_edges.len() - 1, y_bin_edges.len() - 1)`. Indices are
/// clamped to the last bin on either axis. Both edge vectors must be
/// sorted ascending.
pub fn histogram_2d<T, S>(
    x_data: ArrayView1<T>,
    x_bin_edges: &[T],
    y_data: ArrayView1<T>,
    y_bin_edges: &[T],
) -> Array2<S>
where
    T: Float + TotalOrder,
    S: AddFromZero + Clone,
{
    let x_dim = x_bin_edges.len() - 1;
    let y_dim = y_bin_edges.len() - 1;
    let mut hist = Array2::<S>::from_elem((x_dim, y_dim), S::zero());
    for (dx, dy) in std::iter::zip(x_data.iter(), y_data.iter()) {
        let x_idx = map_data2bin(x_bin_edges, dx).min(x_dim - 1);
        let y_idx = map_data2bin(y_bin_edges, dy).min(y_dim - 1);
        hist[[x_idx, y_idx]] += S::one();
    }
    hist
}

/// Compute the joint and the two marginal histograms of a paired
/// sample.
///
/// Returns `(joint, x_marginal, y_marginal)` where `joint` is the 2-D
/// count grid produced by [`histogram_2d`] and the two marginals are
/// the per-axis counts produced by [`histogram_1d`]. The count type
/// is the same as the data type `T`, which makes this convenient when
/// the downstream routines (e.g. mutual information) want floating
/// counts.
pub fn joint_histogram<T>(
    x_data: ArrayView1<T>,
    x_bin_edges: &[T],
    y_data: ArrayView1<T>,
    y_bin_edges: &[T],
) -> (Array2<T>, Array1<T>, Array1<T>)
where
    T: Float + TotalOrder + AddFromZero + Clone + Debug,
{
    (
        histogram_2d(x_data, x_bin_edges, y_data, y_bin_edges),
        histogram_1d(x_data, x_bin_edges),
        histogram_1d(y_data, y_bin_edges),
    )
}

/// Output of the dynamic-programming step of the Bayesian-blocks
/// algorithm.
///
/// * `_best[k]` — best fitness obtainable for the first `k+1` cells.
///   Stored for completeness/debugging; the public driver only
///   consults `last`.
/// * `last[k]` — index of the optimum change point that ends a block
///   at cell `k`. The trailing change-point trace is reconstructed by
///   following these indices backward from `n` to `0`.
struct OptimumBlocks<T> {
    /// Best fitness value attainable for prefixes of length `k+1`.
    _best: Array1<T>,
    /// Optimum predecessor change-point for each prefix.
    last: Array1<usize>,
}

/// Run the dynamic program algorithm from Scargle et al. (2012) that selects the
/// optimum partition of point measurements into Bayesian blocks.
///
/// For each prefix of length `k` the routine evaluates the log-
/// likelihood fitness function (Scargle eq. 19) and applies the
/// `ncp_prior` penalty (eq. 21) before recording the best previous
/// change point. The returned [`OptimumBlocks`] then drives the
/// backtrace in [`bayesian_blocks_bin_edges`].
fn optimal_bayesian_blocks<T>(
    counts: &[T],
    block_sizes: &Array1<T>,
) -> OptimumBlocks<T>
where
    T: 'static + HSFloat,
{
    let n = counts.len();
    let mut block_counts = Array1::<T>::zeros(n);
    let mut best = Array1::<T>::zeros(n);
    let mut last = Array1::<usize>::zeros(n);

    for k in 1..n + 1 {
        let cindex = k - 1;
        // widths = block_length[:K] - block_length[K]
        let widths =
            Array1::<T>::from_shape_fn(k, |i| block_sizes[i] - block_sizes[k]);
        // count_vec[:K] += nn_vec[cindex]
        block_counts
            .slice_mut(s![..k])
            .mapv_inplace(|vx| vx + counts[cindex]);
        // # Fitness fn. (eq. 19 from Scargle 2012)
        // fit_vec = count_vec[:K] * np.log(count_vec[:K] / widths)
        let mut fitness = Array1::<T>::zeros(k);
        fitness += &block_counts.slice(s![..k]);
        fitness *=
            &Array1::from_shape_fn(k, |i| block_counts[i] / widths[i]).ln();
        // # Prior (eq. 21 from Scargle 2012)
        //fit_vec -= 4 - np.log(73.53 * 0.05 * (K**(-0.478)))
        fitness -= &Array1::from_elem(
            1,
            T::from_f64(
                4.0 - f64::ln(73.53 * 0.05 * f64::powf(k as f64, -0.478)),
            )
            .unwrap(),
        );
        // fit_vec[1:] += best[:cindex]
        fitness
            .slice_mut(s![1..])
            .zip_mut_with(&best.slice(s![..cindex]), |fx, bx| *fx += *bx);
        // i_max = np.argmax(fit_vec)
        // last[cindex] = i_max; best[cindex] = fit_vec[i_max];
        let fit_max = fitness
            .iter()
            .enumerate()
            .max_by(|(_ix, fx), (_iy, fy)| fx.total_cmp(fy))
            .unwrap();
        last[cindex] = fit_max.0;
        best[cindex] = *fit_max.1;
    }
    OptimumBlocks { _best: best, last }
}

/// Compute the optimal histogram bin edges for `data` using the
/// Bayesian-blocks algorithm of Scargle et al. (2012).
///
/// The procedure is:
/// 1. Sort `data` and reduce it to its unique values together with
///    their multiplicities (using [`crate::util::unique`]).
/// 2. Build a candidate edge vector at the midpoints of consecutive
///    unique values, with the data extrema as outer edges.
/// 3. Run the dynamic programming algorithm [`optimal_bayesian_blocks`] to
///    pick the change points that maximise the Scargle fitness function
///    with the recommended `p0 = 0.05` prior.
/// 4. Backtrace the change-point chain and project it onto the
///    candidate edges to produce the final, ascending-sorted edge
///    array.
///
/// The returned `Array1<T>` is suitable for direct use with
/// [`histogram_1d`] / [`histogram_2d`].
pub fn bayesian_blocks_bin_edges<T>(data: ArrayView1<T>) -> Array1<T>
where
    T: 'static + HSFloat,
{
    //
    let dt_half = T::one() / (T::one() + T::one());
    let mut srt_data = data.to_vec();
    srt_data.sort_by(TotalOrder::total_cmp);
    let uniq_data = unique(&srt_data);

    let n = uniq_data.values.len();
    let mut edges = Array1::<T>::zeros(n + 1);
    edges[0] = uniq_data.values[0];
    for (a, b) in std::iter::zip(1..n, 0..(n - 1)) {
        edges[a] = dt_half * (uniq_data.values[a] + uniq_data.values[b]);
    }
    edges[n] = uniq_data.values[n - 1];

    let block_length = edges.map(|x| uniq_data.values[n - 1] - *x);
    let bb_optimal = optimal_bayesian_blocks(&uniq_data.counts, &block_length);
    //log::debug!("Uninq {} : {:?}", uniq_data.values.len(), &uniq_data.values);
    //log::debug!("Optim {}: {}", bb_optimal.last.len(), &bb_optimal.last);

    //change_points = np.zeros(n, dtype=np.int64)
    let mut change_points = Array1::<usize>::zeros(n);
    let mut i_cp = n;
    let mut ind = n;
    loop {
        i_cp -= 1;
        change_points[i_cp] = ind;
        if ind == 0 || i_cp == 0 {
            break;
        }
        ind = bb_optimal.last[ind - 1];
    }
    //dbg!(&change_points.slice(s![i_cp..]));
    let bslice = change_points.slice(s![i_cp..]);
    Array1::from_shape_fn(bslice.len(), |ix| edges[bslice[ix]])
}

/// 1-D Bayesian-blocks histogram.
///
/// Calls [`bayesian_blocks_bin_edges`] on `x_data` and feeds the
/// result to [`histogram_1d`].
/// Counts are returned in the same float type `T` as the input data.
pub fn bb_histogram<T>(x_data: ArrayView1<T>) -> Array1<T>
where
    T: 'static + HSFloat,
{
    histogram_1d(
        x_data,
        bayesian_blocks_bin_edges(x_data).as_slice().unwrap(),
    )
}

/// 2-D Bayesian-blocks joint histogram.
///
/// Computes bayesian-block edges independently for `x_data` and
/// `y_data`, then computes the joint and the two marginal
/// histograms via [`joint_histogram`]. Returns
/// `(joint, x_marginal, y_marginal)`.
pub fn bb_joint_histogram<T>(
    x_data: ArrayView1<T>,
    y_data: ArrayView1<T>,
) -> (Array2<T>, Array1<T>, Array1<T>)
where
    T: 'static + HSFloat,
{
    let (x_bin_edges, y_bin_edges) = (
        bayesian_blocks_bin_edges(x_data),
        bayesian_blocks_bin_edges(y_data),
    );
    joint_histogram(
        x_data,
        x_bin_edges.as_slice().unwrap(),
        y_data,
        y_bin_edges.as_slice().unwrap(),
    )
}

#[cfg(test)]
mod tests {

    use super::bb_joint_histogram;
    use crate::{
        tests::{
            HistTestData, PUCTestData, hist_test_data, puc_test4_data,
            test_exp_sub_matrix,
        },
        util::around,
    };

    use anyhow::Result;
    use itertools::Itertools;
    use lazy_static::lazy_static;
    use log::debug;
    use ndarray::Array1;

    lazy_static! {
        static ref PUC_DATA: PUCTestData = puc_test4_data().unwrap();
        static ref HIST_DATA: HistTestData = hist_test_data().unwrap();
    }

    #[test]
    pub fn test_bayesian_blocks() {
        use super::{bayesian_blocks_bin_edges, histogram_1d};

        crate::tests::log_init();

        for colx in 0..4 {
            // let datand = Array2::from(tdata.data);
            //println!("JDATA:: {:?}", tdata.data.column(0));
            debug!("COLUMN : {}", colx);
            debug!(" ->JBINS : {:?}", HIST_DATA.nodes[colx].bins);
            debug!(" ->JHIST : {:?}", HIST_DATA.nodes[colx].hist);
            let rbins = bayesian_blocks_bin_edges(HIST_DATA.data.column(colx));
            debug!(" ->RBINS : {:?}", rbins);
            let thist: Array1<u32> = histogram_1d(
                HIST_DATA.data.column(colx),
                rbins.as_slice().unwrap(),
            );
            debug!(" ->THIST : {:?}", thist);
            assert_eq!(HIST_DATA.nodes[colx].hist, thist.to_vec());
        }
    }

    #[test]
    pub fn test_histogram_2d() {
        use super::{bayesian_blocks_bin_edges, histogram_2d};
        crate::tests::log_init();
        for npair in &HIST_DATA.node_pairs {
            let phist = &npair.hist;
            let x = npair.pair[0];
            let y = npair.pair[1];
            let x_data = HIST_DATA.data.column(x as usize);
            let y_data = HIST_DATA.data.column(y as usize);
            let x_bin_edges = bayesian_blocks_bin_edges(x_data.view());
            let y_bin_edges = bayesian_blocks_bin_edges(y_data.view());
            let hist2d = histogram_2d::<f32, u32>(
                x_data,
                x_bin_edges.as_slice().unwrap(),
                y_data,
                y_bin_edges.as_slice().unwrap(),
            );
            debug!("True HIST :: {:?}", phist);
            debug!("Calculated HIST :: {:?}", hist2d);
            assert_eq!(phist, hist2d)
        }
    }

    #[test]
    pub fn test_joint_histogram() -> Result<()> {
        crate::tests::log_init();
        for (nobs, nvars) in [(1000, 3), (10000, 3)] {
            let px_data = &PUC_DATA.data[&nobs.to_string()];
            let ematrix = test_exp_sub_matrix(0..nobs, 0..nvars)?;
            let ematrix = around(ematrix.view(), 4);
            debug!("EM:{:?}; TDN: {}", ematrix.shape(), px_data.nodes.len());
            (0..nvars).combinations(2).for_each(|vx| {
                let (x, y) = (vx[0], vx[1]);
                let (dxy_hist, dx_hist, dy_hist) = px_data.get_hist(nobs, x, y);
                let (xy_jhist, x_hist, y_hist) = bb_joint_histogram(
                    ematrix.column(x).view(),
                    ematrix.column(y).view(),
                );
                debug!(
                    "X,Y: {:?} ; dims  {:?} ; bayesian blocks dim {:?}",
                    (x, y),
                    (dx_hist.len(), dy_hist.len()),
                    (x_hist.len(), y_hist.len())
                );
                debug!(
                    " --> in_hist : {:?}; joint_hist : {:?} Diff : {}",
                    (x_hist.shape(), y_hist.shape()),
                    (xy_jhist.shape(), dxy_hist.t().shape()),
                    xy_jhist.abs_diff_eq(&dxy_hist.t(), 1e-3)
                );
                assert!(xy_jhist.abs_diff_eq(&dxy_hist.t(), 1e-3));
                //assert_eq!(xy_jhist.view(), dxy_hist.t());
            });
        }
        Ok(())
    }
}
