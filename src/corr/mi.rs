//! B-spline mutual-information estimator.
//!
//! Two complementary code paths are provided:
//!
//! 1. Thin wrappers ([`bspline_weights`], [`bspline_mi`]) around the
//!    `f32`-only C kernels exported by the bundled `mcpnet_rs` crate
//!    (`bspline_weights_f32`, `bspline_mi_kernel_f32`). Inputs are
//!    converted to `f32`, dispatched, and the output cast back to the
//!    caller's float type via [`crate::types::PNFloat`].
//! 2. A pure-Rust reference implementation [`BSplineWeights`] that builds
//!    the knot vector once at construction time and exposes the per-sample
//!    weight matrix [`BSplineWeights::w`], its 1D histogram
//!    [`BSplineWeights::hist1d`] and the corresponding Shannon entropy
//!    ([`BSplineWeights::entropy1d`] / [`BSplineWeights::entropy1d_2`]).
//!
//! The Rust path follows Daub et al. ("Estimating mutual information
//! using B-spline functions – an improved similarity measure for analysing
//! gene expression data", BMC Bioinformatics 2004) with the basis-function
//! corrections described on the MathWorld B-spline pages. Entropies are
//! computed in bits (log base 2).

use std::marker::PhantomData;

use crate::types::PNFloat;
use mcpnet_rs::{bspline_mi_kernel_f32, bspline_weights_f32};

/// Compute the B-spline weight matrix for a 1D sample vector via the
/// bundled C kernel `bspline_weights_f32`.
///
/// Returns a length `num_bins * num_samples + 1` vector laid out as
/// `num_bins` consecutive segments of length `num_samples` (the trailing
/// element matches the C ABI). Inputs and outputs are converted to/from
/// `f32` because the underlying kernel only supports single precision.
pub fn bspline_weights<FT>(
    data: &[FT],
    num_bins: usize,
    spline_order: usize,
    num_samples: usize,
) -> Vec<FT>
where
    FT: PNFloat + Default,
{
    let mut out_vec: Vec<f32> = vec![0.0; num_bins * num_samples + 1];
    let in_vec: Vec<f32> = data
        .iter()
        .map(|x| x.to_f32().unwrap_or_default())
        .collect();

    bspline_weights_f32(
        &in_vec,
        num_bins as i32,
        spline_order as i32,
        num_samples as i32,
        &mut out_vec,
    );
    out_vec
        .iter()
        .map(|x| FT::from_f32(*x).unwrap_or_default())
        .collect()
}

/// Compute mutual information `I(X; Y)` between two paired sample vectors
/// using the bundled C kernel `bspline_mi_kernel_f32`.
///
/// `x_data` and `y_data` must have length `num_samples`. The result is in
/// bits (the C kernel uses `log2`). Inputs are converted to `f32` for the
/// call and the result is cast back to `FT`.
pub fn bspline_mi<FT>(
    x_data: &[FT],
    y_data: &[FT],
    num_bins: usize,
    num_samples: usize,
) -> FT
where
    FT: PNFloat + Default,
{
    let first: Vec<f32> = x_data
        .iter()
        .map(|x| x.to_f32().unwrap_or_default())
        .collect();

    let second: Vec<f32> = y_data
        .iter()
        .map(|x| x.to_f32().unwrap_or_default())
        .collect();

    FT::from_f32(bspline_mi_kernel_f32(
        &first,
        &second,
        num_bins as i32,
        num_samples as i32,
    ))
    .unwrap_or_default()
}

/// Build the open uniform knot vector used by the recursive B-spline basis.
///
/// Layout (length `n_bins + spline_order`): the first `spline_order` knots
/// are clamped to `0`, the next `n_bins - spline_order` knots are evenly
/// spaced on `(0, 1)`, and the trailing `spline_order` knots are clamped to
/// `1`. Reproduces the construction in Daub et al. 2004 §2.1.
fn build_knot_vector<KT>(n_bins: usize, spline_order: usize) -> Vec<KT>
where
    KT: PNFloat + Default,
{
    let mut v = vec![KT::zero(); n_bins + spline_order];
    let n_internal_points = n_bins - spline_order;
    let norm_factor: f64 = 1.0 / (n_internal_points + 1) as f64;
    //
    // 	int i;
    // 	for (i = 0; i < splineOrder; ++i) {
    // 		v[i] = static_cast<KT>(0);
    // 	}
    for vx in v.iter_mut().take(spline_order) {
        *vx = KT::zero();
    }
    // 	for (i = splineOrder; i < splineOrder + nInternalPoints; ++i) {
    // 		v[i] = static_cast<KT>(i - splineOrder + 1) * norm_factor;
    // 	}
    for (i, vx) in v
        .iter_mut()
        .enumerate()
        .take(n_internal_points + spline_order)
        .skip(spline_order)
    {
        *vx = KT::from_f64((i - spline_order + 1) as f64 * norm_factor)
            .unwrap_or_default();
    }
    // 	for (i = splineOrder + nInternalPoints; i < 2*splineOrder + nInternalPoints; ++i) {
    // 		v[i] = static_cast<KT>(1);
    // 	}
    // 	return v;
    for vx in v
        .iter_mut()
        .take(2 * spline_order + n_internal_points)
        .skip(spline_order + n_internal_points)
    {
        *vx = KT::one();
    }

    v
}

/// Pure-Rust B-spline weight builder for one variable.
///
/// Bundles the immutable parameters of the basis (`n_bins`, `spline_order`,
/// `n_samples`), the precomputed knot vector and the per-sample
/// normalization factor `1/n_samples`. `DT` is the input data type and `OT`
/// is the output / weight float type — kept separate so a `f32` data array
/// can drive an `f64` accumulation if higher precision is needed for
/// entropy.
///
/// Typical usage: build once with [`new`](Self::new), reuse across multiple
/// columns by feeding sample slices to [`w`](Self::w), which returns the
/// flattened `n_bins × n_samples` weight matrix and the corresponding
/// 1D Shannon entropy in bits.
pub struct BSplineWeights<DT, OT> {
    /// Number of histogram bins (B-spline basis functions).
    n_bins: usize,
    /// Polynomial order of the B-splines (degree + 1).
    spline_order: usize,
    /// Number of samples per evaluation.
    n_samples: usize,
    /// Open uniform knot vector built by [`build_knot_vector`].
    knot_vector: Vec<DT>,
    /// `1 / n_samples`, cached as `f64` for the entropy reduction.
    norm_factor: f64,
    /// Phantom marker tying the impl block to the output float type.
    _ot: PhantomData<OT>,
}

impl<DT, OT> BSplineWeights<DT, OT>
where
    DT: PNFloat + Default,
    OT: PNFloat + Default,
{
    /// Recursive Cox–de Boor evaluation of the `i`-th basis function of
    /// order `p` at parameter `t`.
    ///
    /// Implements the Daub et al. recursion with the corrections from the
    /// MathWorld B-spline pages: returns `1` when `t` lies in
    /// `[knot[i], knot[i+1])` (or exactly hits the right boundary of the
    /// last bin) at order 1, and combines the two child evaluations
    /// proportionally to the knot spacing for higher orders. Tiny negative
    /// rounding artifacts are clamped to `0`.
    fn basis_function(&self, i: usize, p: usize, t: DT) -> OT {
        if p == 1 {
            if (t >= self.knot_vector[i]
                && t < self.knot_vector[i + 1]
                && self.knot_vector[i] < self.knot_vector[i + 1])
                || ((t - self.knot_vector[i + 1])
                    .abs()
                    .to_f64()
                    .unwrap_or_default()
                    < 1e-10
                    && (i + 1 == self.n_bins))
            {
                return OT::one();
            }
            OT::zero();
        }
        //
        let d1 = self.knot_vector[i + p - 1] - self.knot_vector[i];
        let d2 = self.knot_vector[i + p] - self.knot_vector[i + 1];
        //
        if d1.to_f64().unwrap_or_default() < 1e-10
            && d2.to_f64().unwrap_or_default() < 1e-10
        {
            return OT::zero();
        }

        let n1 = t - self.knot_vector[i];
        let n2 = self.knot_vector[i + p] - t;
        let (e1, e2) = if d1.to_f64().unwrap_or_default() < 1e-10 {
            let re2 = OT::from_f64((n2 / d2).to_f64().unwrap_or_default())
                .unwrap_or_default();
            (OT::zero(), re2 * self.basis_function(i + 1, p - 1, t))
        } else if d2.to_f64().unwrap_or_default() < 1e-10 {
            let re1 = OT::from_f64((n1 / d1).to_f64().unwrap_or_default())
                .unwrap_or_default();
            (re1 * self.basis_function(i, p - 1, t), OT::zero())
        } else {
            let re1 = OT::from_f64((n1 / d1).to_f64().unwrap_or_default())
                .unwrap_or_default();
            let re2 = OT::from_f64((n2 / d2).to_f64().unwrap_or_default())
                .unwrap_or_default();
            (
                re1 * self.basis_function(i, p - 1, t),
                re2 * self.basis_function(i + 1, p - 1, t),
            )
        };
        //
        // sometimes, this value is < 0 (only just; rounding error); truncate
        if e1 + e2 < OT::zero() {
            OT::zero()
        } else {
            e1 + e2
        }
    }

    /// Collapse a flat `n_bins × n_samples` weight matrix into a 1D
    /// histogram by summing each bin's segment and scaling by `1/n_samples`.
    pub fn hist1d(&self, weight_vec: &[OT]) -> Vec<OT> {
        //  for (int curBin = 0; curBin < numBins; curBin++) {
        (0..self.n_bins)
            .map(|ix| {
                //   const auto binBegin = curBin * numSamples;
                let bin_begin: usize = ix * self.n_samples;
                // OT ex = 0.0;
                // for (int curSample = 0; curSample < numSamples; curSample++) {
                //     // ex = ex + ((weight_vec[binBegin + curSample]) / static_cast<OT>(numSamples));
                //     ex += (weight_vec[binBegin + curSample]) * norm_factor;
                // }

                let ex = (0..self.n_samples)
                    .map(|csx| {
                        weight_vec[bin_begin + csx].to_f64().unwrap_or_default()
                            * self.norm_factor
                    })
                    .sum();
                OT::from_f64(ex).unwrap_or_default()
            })
            .collect()
    }

    /// Shannon entropy `H = -Σ p log2 p` of the soft histogram derived
    /// from `weight_vec` via [`hist1d`](Self::hist1d). Bins with zero mass
    /// are skipped to avoid `log(0)`.
    pub fn entropy1d(&self, weight_vec: &[OT]) -> OT {
        // OT H = 0.0;
        let hist = self.hist1d(weight_vec);
        //  for (int curBin = 0; curBin < numBins; curBin++) {
        //      if (hist[curBin] > 0) {
        //          H += hist[curBin] * log2(hist[curBin]);
        //      }
        //  }
        let h = hist
            .iter()
            .filter(|x| x.gt(&&OT::zero()))
            .fold(OT::zero(), |acc, x| acc + x.log2() * *x);
        OT::zero() - h
    }

    /// Fused variant of [`entropy1d`](Self::entropy1d): walks
    /// `weight_vec` once, accumulating each bin's mass and reducing to
    /// `H = -Σ p log2 p` without materializing the histogram. Asserts
    /// `weight_vec.len() >= n_bins * n_samples`.
    pub fn entropy1d_2(&self, weight_vec: &[OT]) -> OT {
        assert!(weight_vec.len() >= self.n_bins * self.n_samples);
        //  OT H = 0.0;
        //  int binBegin = 0;
        let norm_factor: OT = OT::from_f64(self.norm_factor).unwrap_or_default();
        //  for (int curBin = 0; curBin < numBins; curBin++) {
        let mut bin_begin: usize = 0;
        let h = (0..self.n_bins)
            .map(|_cb| {
                //  OT ex = 0.0;
                //  for (int curSample = 0; curSample < numSamples; curSample++) {
                //      // ex = ex + ((weight_vec[binBegin + curSample]) / static_cast<OT>(numSamples));
                //      ex += (weight_vec[binBegin]) * norm_factor;
                //      ++binBegin;
                //  }
                (0..self.n_samples)
                    .map(|_csx| {
                        //weight_vec[bin_begin] * self.norm_factor
                        let rfx = weight_vec[bin_begin] * norm_factor;
                        bin_begin += 1;
                        rfx
                    })
                    .fold(OT::zero(), |acc, x| acc + x)
            })
            //  if (ex > 0.0) H += ex * log2(ex);
            .filter(|x| x.gt(&OT::zero()))
            .fold(OT::zero(), |acc, x| acc + x.log2() * x);
        //  }
        OT::zero() - h
        //  return 0.0 - H;
    }

    /// Construct a [`BSplineWeights`] from the basis parameters, building
    /// the open uniform knot vector and caching `1/n_samples`.
    pub fn new(n_bins: usize, spline_order: usize, n_samples: usize) -> Self {
        Self {
            norm_factor: 1.0 / n_samples as f64,
            knot_vector: build_knot_vector(n_bins, spline_order),
            n_bins,
            spline_order,
            n_samples,
            _ot: PhantomData,
        }
    }

    /// Copy the basis parameters and knot vector from `other` into `self`.
    /// Useful when reusing an allocation across variables that share the
    /// same `(n_bins, spline_order, n_samples)` shape.
    pub fn set(&mut self, other: &Self) {
        self.n_bins = other.n_bins;
        self.n_samples = other.n_samples;
        self.spline_order = other.spline_order;
        self.norm_factor = other.norm_factor;
        self.knot_vector = other.knot_vector.clone();
    }

    /// Number of histogram bins (B-spline basis functions) configured.
    pub fn num_bins(&self) -> usize {
        self.n_bins
    }

    /// Evaluate the full B-spline weight matrix and its 1D Shannon
    /// entropy for the sample slice `in_vec` (of length `n_samples`).
    ///
    /// Returns `(weights, H)` where `weights` is a flat
    /// `n_bins * n_samples` vector laid out as `n_bins` consecutive
    /// per-sample segments and `H` is the entropy in bits computed by
    /// [`entropy1d_2`](Self::entropy1d_2). The two outputs together feed
    /// into the joint-entropy / mutual-information assembly performed by
    /// callers.
    pub fn w(&self, in_vec: &[DT]) -> (Vec<OT>, OT) {
        //  int num_weights = numSamples * numBins;
        let n_weights = self.n_samples * self.n_bins;
        //  for (int i = 0; i < num_weights; i++) {
        //  }
        let out_vec: Vec<OT> = (0..n_weights)
            .map(|i| {
                //  int curBin = i / numSamples;
                //  int curSample = i - (curBin * numSamples);
                let cur_bin = i / self.n_samples;
                let cur_sample = i - (cur_bin * self.n_samples);
                //  OT x = static_cast<OT>(in_vec[curSample]);
                //  out_vec[i] = basisFunction(curBin, splineOrder, x);
                self.basis_function(i, self.spline_order, in_vec[cur_sample])
            })
            .collect();

        // for (int curSample = 0; curSample < numSamples; curSample++) {
        //     for (int curBin = 0; curBin < numBins; curBin++) {
        //     OT x = static_cast<OT>(in_vec[curSample]);
        //     out_vec[curBin * numSamples + curSample] =
        //             basisFunction(curBin, splineOrder, x);
        //    // mexPrintf("%d|%f(%f)\t", curBin,
        //    //   weights[curBin * numSamples + curSample],z[curSample]);
        //     }
        // }

        //out_vec[num_weights] = entropy1d_2(out_vec);
        let etpy: OT = self.entropy1d_2(&out_vec[..]);
        (out_vec, etpy)
    }
}
