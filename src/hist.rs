use ndarray::{Array1, Array2, ArrayView1, s};
use num::{Float, FromPrimitive, traits::float::TotalOrder};
use std::clone::Clone;
use std::fmt::Debug;

use crate::types::{AddFromZero, AssignOps};
use crate::util::unique;

pub trait HSFloat:
    Float + TotalOrder + FromPrimitive + AssignOps + Debug + Clone
{
}
impl<T: Float + TotalOrder + FromPrimitive + AssignOps + Debug + Clone> HSFloat
    for T
{
}

fn map_data2bin<T: Float + TotalOrder>(bin_edges: &[T], qry_dx: &T) -> usize {
    let fidx = match bin_edges.binary_search_by(|ex| ex.total_cmp(qry_dx)) {
        Ok(kidx) => kidx,
        Err(eidx) => eidx,
    };
    if fidx > 0 { fidx - 1 } else { fidx }
}

pub fn histogram_1d<T, S>(data: ArrayView1<T>, edges: &[T]) -> Array1<S>
where
    T: Float + TotalOrder,
    S: AddFromZero + Clone,
{
    let mut hist = Array1::<S>::from_elem(edges.len() - 1, S::zero());
    //let ed_slice = edges.as_slice().unwrap();
    for dx in data {
        hist[map_data2bin(edges, dx)] += S::one();
    }
    hist
}

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
    let mut hist = Array2::<S>::from_elem(
        (x_bin_edges.len() - 1, y_bin_edges.len() - 1),
        S::zero(),
    );
    for (dx, dy) in std::iter::zip(x_data.iter(), y_data.iter()) {
        hist[[map_data2bin(x_bin_edges, dx), map_data2bin(y_bin_edges, dy)]] +=
            S::one();
    }
    hist
}

pub fn joint_histogram<T>(
    x_data: ArrayView1<T>,
    x_bin_edges: &[T],
    y_data: ArrayView1<T>,
    y_bin_edges: &[T],
) -> (Array2<T>, Array1<T>, Array1<T>)
where
    T: Float + TotalOrder + AddFromZero + Clone,
{
    (
        histogram_2d(x_data, x_bin_edges, y_data, y_bin_edges),
        histogram_1d(x_data, x_bin_edges),
        histogram_1d(y_data, y_bin_edges),
    )
}

struct OptimumBlocks<T> {
    _best: Array1<T>,
    last: Array1<usize>,
}

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
    // dbg!(&block_length);
    let bb_optimal = optimal_bayesian_blocks(&uniq_data.counts, &block_length);

    //change_points = np.zeros(n, dtype=np.int64)
    let mut change_points = Array1::<usize>::zeros(n);
    let mut i_cp = n;
    let mut ind = n;
    loop {
        i_cp -= 1;
        change_points[i_cp] = ind;
        if ind == 0 {
            break;
        }
        ind = bb_optimal.last[ind - 1];
    }
    //dbg!(&change_points.slice(s![i_cp..]));
    let bslice = change_points.slice(s![i_cp..]);
    Array1::from_shape_fn(bslice.len(), |ix| edges[bslice[ix]])
}

pub fn bb_histogram<T>(x_data: ArrayView1<T>) -> Array1<T>
where
    T: 'static + HSFloat,
{
    histogram_1d(
        x_data,
        bayesian_blocks_bin_edges(x_data).as_slice().unwrap(),
    )
}

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
