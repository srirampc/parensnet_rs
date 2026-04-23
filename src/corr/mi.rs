use std::marker::PhantomData;

use crate::types::PNFloat;

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

pub struct BSplineWeights<DT, OT> {
    n_bins: usize,
    spline_order: usize,
    n_samples: usize,
    knot_vector: Vec<DT>,
    norm_factor: f64,
    _ot: PhantomData<OT>,
}

impl<DT, OT> BSplineWeights<DT, OT>
where
    DT: PNFloat + Default,
    OT: PNFloat + Default,
{
    // Follows Daub et al, which contains mistakes;
    // corrections based on spline descriptions on MathWorld pages
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

    pub fn set(&mut self, other: &Self) {
        self.n_bins = other.n_bins;
        self.n_samples = other.n_samples;
        self.spline_order = other.spline_order;
        self.norm_factor = other.norm_factor;
        self.knot_vector = other.knot_vector.clone();
    }

    pub fn num_bins(&self) -> usize {
        self.n_bins
    }

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
