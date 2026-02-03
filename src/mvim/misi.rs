#![allow(dead_code)]
use hdf5::{self, H5Type};
use ndarray::Array1;
use num::ToPrimitive;
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    rc::Rc,
};

use crate::h5::io::{
    read_scalar_attr, read1d_pair_of_points, read1d_pair_of_slices, read1d_point,
};
use crate::map_with_result_to_tuple;
use crate::mvim::rv::{
    Error, LMRDataStructure, LMRSubsetDataStructure, MRVFloat, MRVInteger,
    MRVTrait,
};
use crate::types::Pair;
use crate::util::{exc_prefix_sum, triu_pair_to_index};

pub enum LMRDSRange<IntT: 'static + MRVInteger, Float: 'static + MRVFloat> {
    Complete(Box<Pair<Vec<LMRDataStructure<IntT, Float>>>>),
    Subset(Box<Pair<Vec<LMRSubsetDataStructure<IntT, Float>>>>),
}

pub struct MISIRangePair<IntT, FloatT>
where
    IntT: H5Type + MRVInteger,
    FloatT: H5Type + MRVFloat,
{
    st_ranges: Pair<Range<usize>>,
    nobs: IntT,
    nvars: IntT,
    npairs: IntT,
    nsi: IntT,
    nsjv_dim: IntT,
    hist_dim: Pair<Array1<IntT>>,
    hist_start: Pair<Array1<IntT>>,
    si_start: Pair<Array1<IntT>>,
    hist: Pair<Array1<FloatT>>,
    mi: Array1<FloatT>,
    si: Pair<Array1<FloatT>>,
    lmr: Pair<Array1<FloatT>>,
    range_si_start: Pair<Array1<usize>>,
    range_hist_start: Pair<Array1<usize>>,
    range_set: Pair<HashSet<usize>>,
    range_lookup: Array1<usize>,
    subset_var: Option<Vec<IntT>>,
    subset_map: Rc<HashMap<IntT, usize>>,
    lmr_ds: Option<LMRDSRange<IntT, FloatT>>,
}

pub enum LMRDSPair<IntT: 'static + MRVInteger, Float: 'static + MRVFloat> {
    Complete(Box<Pair<LMRDataStructure<IntT, Float>>>),
    Subset(Box<Pair<LMRSubsetDataStructure<IntT, Float>>>),
}

pub struct MISIPair<IntT, FloatT>
where
    IntT: H5Type + MRVInteger,
    FloatT: H5Type + MRVFloat,
{
    var_pair: Pair<usize>,
    nobs: IntT,
    nvars: IntT,
    npairs: IntT,
    mi: FloatT,
    hist_start: Pair<IntT>,
    hist_dim: Pair<IntT>,
    si_start: Pair<IntT>,
    hist: Pair<Array1<FloatT>>,
    si: Pair<Array1<FloatT>>,
    lmr: Pair<Array1<FloatT>>,
    subset_var: Option<Vec<IntT>>,
    subset_map: Rc<HashMap<IntT, usize>>,
    lmr_ds: Option<LMRDSPair<IntT, FloatT>>,
}

impl<IntT, FloatT> MISIRangePair<IntT, FloatT>
where
    IntT: H5Type + MRVInteger,
    FloatT: H5Type + MRVFloat,
{
    pub fn new(
        h5_file: &str,
        st_ranges: (Range<usize>, Range<usize>),
    ) -> Result<Self, hdf5::Error> {
        let file = hdf5::File::open(h5_file)?;
        let data_g = file.group("data")?;
        // attributes
        let (nobs, nvars, npairs, nsi, nsjv_dim) = map_with_result_to_tuple![
            |x| read_scalar_attr::<IntT>(&data_g, x) ;
            "nobs", "nvars", "npairs", "nsi", "nsjv_dim"
        ];
        // datasets: hist_dim, hist_start, si_start
        let (hist_dim, hist_start, si_start) = map_with_result_to_tuple![
            |x| read1d_pair_of_slices::<IntT, usize>(&data_g, x, &st_ranges) ;
            "hist_dim", "hist_start", "si_start"
        ];
        let mi = data_g.dataset("mi")?.read_1d::<FloatT>()?;
        // hist
        let hist = read1d_pair_of_slices::<FloatT, IntT>(
            &data_g,
            "hist",
            &(hist_start
                .zip_map(&hist_dim, |start, dim| start[0]..(start[0] + dim.sum()))
                .to_tuple()),
        )?;
        // si and lmr
        let sist_ranges = si_start
            .zip_map(&hist_dim, |start, dim| {
                start[0]..(start[0] + nvars * dim.sum())
            })
            .to_tuple();
        let (si, lmr) = map_with_result_to_tuple![
            |x| read1d_pair_of_slices::<FloatT, IntT>(&data_g, x, &sist_ranges) ;
            "si", "lmr"
        ];
        // range lookup arrays/data structures
        let uz_nvars = nvars.to_usize().unwrap_or(0);
        assert_ne!(uz_nvars, 0);
        let range_hist_start: Pair<Array1<usize>> = hist_dim.map(|x| {
            exc_prefix_sum(x.iter().map(|y| y.to_usize().unwrap()), 1usize)
        });
        let range_si_start = range_hist_start.map(|x| x.clone() * uz_nvars);
        let mut range_lookup = Array1::<usize>::from_elem(uz_nvars, 2);
        range_lookup
            .slice_mut(ndarray::s![st_ranges.0.clone()])
            .fill(0);
        range_lookup
            .slice_mut(ndarray::s![st_ranges.1.clone()])
            .fill(1);
        let range_set: Pair<HashSet<usize>> = Pair::new(
            HashSet::from_iter(st_ranges.0.start..st_ranges.0.end),
            HashSet::from_iter(st_ranges.1.start..st_ranges.1.end),
        );

        //
        Ok(MISIRangePair {
            st_ranges: Pair::from_tuple(st_ranges),
            nobs,
            nvars,
            npairs,
            nsi,
            nsjv_dim,
            hist_dim,
            hist_start,
            si_start,
            hist,
            mi,
            si,
            lmr,
            range_hist_start,
            range_si_start,
            range_set,
            range_lookup,
            subset_var: None,
            subset_map: Rc::new(HashMap::new()),
            lmr_ds: None,
        })
    }

    pub fn set_lmr_ds(
        &mut self,
        subset_var: Option<&[IntT]>,
    ) -> Result<(), Error> {
        let vpair: Pair<Vec<IntT>> = self
            .st_ranges
            .map(|x| x.clone().map(|y| IntT::from_usize(y).unwrap()).collect());
        if let Some(sv_vec) = subset_var {
            let mut subset_map = HashMap::<IntT, usize>::new();
            for (idx, vx) in sv_vec.iter().enumerate() {
                subset_map.insert(*vx, idx);
            }
            self.subset_map = Rc::new(subset_map);
            let fvec: Result<Vec<LMRSubsetDataStructure<IntT, FloatT>>, Error> =
                vpair
                    .first
                    .iter()
                    .map(|x| {
                        LMRSubsetDataStructure::<IntT, FloatT>::new(
                            self,
                            *x,
                            self.subset_map.clone(),
                        )
                    })
                    .collect();
            let svec: Result<Vec<LMRSubsetDataStructure<IntT, FloatT>>, Error> =
                vpair
                    .second
                    .iter()
                    .map(|x| {
                        LMRSubsetDataStructure::<IntT, FloatT>::new(
                            self,
                            *x,
                            self.subset_map.clone(),
                        )
                    })
                    .collect();
            self.lmr_ds =
                Some(LMRDSRange::Subset(Box::new(Pair::new(fvec?, svec?))));
        } else {
            let fvec: Result<Vec<LMRDataStructure<IntT, FloatT>>, Error> = vpair
                .first
                .iter()
                .map(|x| LMRDataStructure::<IntT, FloatT>::new(self, *x))
                .collect();
            let svec: Result<Vec<LMRDataStructure<IntT, FloatT>>, Error> = vpair
                .second
                .iter()
                .map(|x| LMRDataStructure::<IntT, FloatT>::new(self, *x))
                .collect();
            self.lmr_ds =
                Some(LMRDSRange::Complete(Box::new(Pair::new(fvec?, svec?))));
        }
        Ok(())
    }

    // Range and location for
    fn _range_offset<T: ToPrimitive>(
        &self,
        i: T,
    ) -> Result<(usize, usize), Error> {
        let uix = i.to_usize().unwrap();
        let ridx = self.range_lookup[uix];
        if ridx < 2 {
            Ok((ridx, uix - self.st_ranges.at(ridx).start))
        //}
        //if self.range_set.first.contains(&uix) {
        //    Ok((0, uix - self.st_ranges.first.start))
        //} else if self.range_set.second.contains(&uix) {
        //    Ok((1, uix - self.st_ranges.second.start))
        } else {
            Result::Err(Error::InvalidIndex(uix))
        }
    }

    fn _si_offset<T: ToPrimitive>(&self, var: T) -> Result<usize, Error> {
        let (ridx, vloc) = self._range_offset(var)?;
        Ok(self.range_si_start.at(ridx)[vloc])
    }

    // hist dim
    fn _hist_dim_for<T: ToPrimitive>(&self, i: T) -> Result<usize, Error> {
        let (ridx, vloc) = self._range_offset(i)?;
        Ok(self.hist_dim.at(ridx)[vloc].to_usize().unwrap())
    }

    //
    fn _hist_start_for<T: ToPrimitive>(&self, i: T) -> Result<usize, Error> {
        let (ridx, vloc) = self._range_offset(i)?;
        Ok(self.hist_start.at(ridx)[vloc].to_usize().unwrap())
    }

    fn _si_bounds_start<T: Clone + ToPrimitive>(
        &self,
        about: T,
        by: T,
    ) -> Result<usize, Error> {
        let about_hdim = self
            ._hist_dim_for(about.clone())
            .map_err(|_err| Error::InvalidAbout(about.to_usize().unwrap()))?;
        let si_offset = self
            ._si_offset(about.clone())
            .map_err(|_err| Error::InvalidAbout(about.to_usize().unwrap()))?;
        let ux_by = by.to_usize().unwrap();
        Ok(si_offset + (ux_by * about_hdim))
    }

    fn _si_bounds<T: Clone + ToPrimitive>(
        &self,
        about: T,
        by: T,
    ) -> Result<Range<usize>, Error> {
        let about_hdim = self
            ._hist_dim_for(about.clone())
            .map_err(|_err| Error::InvalidAbout(about.to_usize().unwrap()))?;
        let si_offset = self
            ._si_offset(about.clone())
            .map_err(|_err| Error::InvalidAbout(about.to_usize().unwrap()))?;

        let bstart = si_offset + (by.to_usize().unwrap() * about_hdim);
        let bend = bstart + about_hdim;
        Ok(bstart..bend)
    }
}

impl<IntT, FloatT> MRVTrait<IntT, FloatT> for MISIRangePair<IntT, FloatT>
where
    IntT: H5Type + MRVInteger,
    FloatT: H5Type + MRVFloat,
{
    fn get_hist(&self, i: IntT) -> Result<Array1<FloatT>, Error> {
        let (ridx, _vloc) = self._range_offset(i)?;
        let hs_start = self._hist_start_for(i)?;
        let hs_stop = hs_start + self._hist_dim_for(i)?;
        Ok(self
            .hist
            .at(ridx)
            .slice(ndarray::s![hs_start..hs_stop])
            .to_owned())
    }

    fn get_hist_dim(&self, i: IntT) -> Result<IntT, Error> {
        let (ridx, vloc) = self._range_offset(i)?;
        Ok(self.hist_dim.at(ridx)[vloc])
    }

    fn get_si(&self, about: IntT, by: IntT) -> Result<Array1<FloatT>, Error> {
        let (ridx, _vloc) = self._range_offset(about)?;
        let srange = self._si_bounds(about, by)?;
        Ok(self.si.at(ridx).slice(ndarray::s![srange]).to_owned())
    }

    fn si_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error> {
        let (ridx, _vloc) = self._range_offset(about)?;
        let abt_si_start = self._si_bounds_start(about, by)?;
        Ok(self.si.at(ridx)[abt_si_start + rstate.to_usize().unwrap()])
    }

    fn get_lmr(&self, about: IntT, by: IntT) -> Result<Array1<FloatT>, Error> {
        let (ridx, _vloc) = self._range_offset(about)?;
        let srange = self._si_bounds(about, by)?;
        Ok(self.lmr.at(ridx).slice(ndarray::s![srange]).to_owned())
    }

    fn lmr_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error> {
        let (ridx, _vloc) = self._range_offset(about)?;
        let srange = self._si_bounds(about, by)?;
        Ok(self.lmr.at(ridx)[srange.start + rstate.to_usize().unwrap()])
    }

    fn get_mi(&self, i: IntT, j: IntT) -> Result<FloatT, Error> {
        let pindex = triu_pair_to_index(self.nvariables(), i, j);
        Ok(self.mi[pindex])
    }

    fn ndata(&self) -> FloatT {
        FloatT::from_usize(self.nobservations()).unwrap()
    }

    fn nobservations(&self) -> usize {
        self.nobs.to_usize().unwrap()
    }

    fn nvariables(&self) -> usize {
        self.nvars.to_usize().unwrap()
    }

    fn get_lmr_minsum(&self, about: IntT, target: IntT) -> Result<FloatT, Error> {
        let (ridx, vloc) = self._range_offset(about)?;
        match &self.lmr_ds {
            None => Ok(self.minsum_list(about, target)?.sum()),
            Some(lmr_dsv) => match lmr_dsv {
                LMRDSRange::Complete(complete_ds) => {
                    complete_ds.at(ridx)[vloc].minsum(target)
                }
                LMRDSRange::Subset(subset_ds) => {
                    subset_ds.at(ridx)[vloc].minsum(self, target)
                }
            },
        }
    }

    fn get_puc_factor(&self, about: IntT, target: IntT) -> Result<FloatT, Error> {
        let (ridx, vloc) = self._range_offset(about)?;
        let pfactor = match &self.lmr_ds {
            None => self.nvariables() - 2,
            Some(lmr_dsv) => match lmr_dsv {
                LMRDSRange::Complete(complete_ds) => {
                    complete_ds.at(ridx)[vloc].mi_factor(target)
                }
                LMRDSRange::Subset(subset_ds) => {
                    subset_ds.at(ridx)[vloc].mi_factor(target)
                }
            },
        };
        Ok(FloatT::from_usize(pfactor).unwrap())
    }

    fn compute_lm_puc(&self, i: IntT, j: IntT) -> Result<FloatT, Error> {
        let mij = self.get_mi(i, j)?;
        if mij <= FloatT::zero() {
            return Ok(mij);
        }
        let (pij_factor, pji_factor) =
            (self.get_puc_factor(i, j)?, self.get_puc_factor(j, i)?);
        Ok((pij_factor - (self.get_lmr_minsum(i, j)? / mij))
            + (pji_factor - (self.get_lmr_minsum(j, i)? / mij)))
    }
}

impl<IntT, FloatT> MISIPair<IntT, FloatT>
where
    IntT: H5Type + MRVInteger,
    FloatT: H5Type + MRVFloat,
{
    pub fn new(
        h5_file: &str,
        var_pair: (usize, usize),
    ) -> Result<Self, hdf5::Error> {
        let (pi, pj) = var_pair;
        let file = hdf5::File::open(h5_file)?;
        let data_g = file.group("data")?;
        // attributes
        let (nobs, nvars, npairs) = map_with_result_to_tuple![
            |x| read_scalar_attr::<IntT>(&data_g, x) ;
            "nobs", "nvars", "npairs"
        ];
        // read mi
        let mi: FloatT =
            read1d_point(&data_g, "mi", triu_pair_to_index(nvars, pi, pj))?;
        // start points of pair pi, pj
        let (hist_dim, hist_start, si_start) = map_with_result_to_tuple![
            |x| read1d_pair_of_points::<IntT, usize>(&data_g, x, (pi, pj)) ;
            "hist_dim", "hist_start", "si_start"
        ];

        //
        let hist = read1d_pair_of_slices::<FloatT, IntT>(
            &data_g,
            "hist",
            &(hist_start
                .zip_map(&hist_dim, |start, dim| *start..(*start + *dim))
                .to_tuple()),
        )?;
        // si and lmr
        let sist_ranges = si_start
            .zip_map(&hist_dim, |start, dim| *start..(*start + nvars * *dim))
            .to_tuple();
        let (si, lmr) = (
            read1d_pair_of_slices::<FloatT, IntT>(&data_g, "si", &sist_ranges)?,
            read1d_pair_of_slices::<FloatT, IntT>(&data_g, "lmr", &sist_ranges)?,
        );

        Ok(MISIPair {
            var_pair: Pair::from_tuple(var_pair),
            nobs,
            nvars,
            npairs,
            mi,
            hist_dim,
            hist_start,
            si_start,
            hist,
            si,
            lmr,
            lmr_ds: None,
            subset_var: None,
            subset_map: Rc::new(HashMap::<IntT, usize>::new()),
        })
    }

    pub fn set_lmr_ds(
        &mut self,
        subset_var: Option<&[IntT]>,
    ) -> Result<(), Error> {
        let vpair = self.var_pair.map(|x| IntT::from_usize(*x).unwrap());
        if let Some(sv_vec) = subset_var {
            let mut subset_map = HashMap::<IntT, usize>::new();
            for (idx, vx) in sv_vec.iter().enumerate() {
                subset_map.insert(*vx, idx);
            }
            self.subset_map = Rc::new(subset_map);
            self.lmr_ds = Some(LMRDSPair::Subset(Box::new(Pair::new(
                LMRSubsetDataStructure::<IntT, FloatT>::new(
                    self,
                    vpair.first,
                    self.subset_map.clone(),
                )?,
                LMRSubsetDataStructure::<IntT, FloatT>::new(
                    self,
                    vpair.second,
                    self.subset_map.clone(),
                )?,
            ))));
        } else {
            self.lmr_ds = Some(LMRDSPair::Complete(Box::new(Pair::new(
                LMRDataStructure::<IntT, FloatT>::new(self, vpair.first)?,
                LMRDataStructure::<IntT, FloatT>::new(self, vpair.second)?,
            ))));
        }
        Ok(())
    }

    // range indicator
    pub fn range_of<T: ToPrimitive>(&self, i: T) -> Result<usize, Error> {
        let uix = i.to_usize().unwrap();
        if self.var_pair.first == uix {
            Ok(0)
        } else if self.var_pair.second == uix {
            Ok(1)
        } else {
            Result::Err(Error::InvalidIndex(uix))
        }
    }

    fn _hist_dim_for<T: ToPrimitive>(&self, i: T) -> Result<usize, Error> {
        let ridx = self.range_of(i)?;
        Ok(self.hist_dim.at(ridx).to_usize().unwrap())
    }

    fn _si_bounds_start<T: Clone + ToPrimitive>(
        &self,
        about: T,
        by: T,
    ) -> Result<usize, Error> {
        let about_hdim = self._hist_dim_for(about)?;
        Ok(by.to_usize().unwrap() * about_hdim.to_usize().unwrap())
    }

    fn _si_bounds<T: Clone + ToPrimitive>(
        &self,
        about: T,
        by: T,
    ) -> Result<Range<usize>, Error> {
        let about_hdim = self._hist_dim_for(about.clone())?;
        let bstart = self._si_bounds_start(about, by)?;
        let bend = bstart + about_hdim;
        Ok(bstart..bend)
    }
}

impl<IntT, FloatT> MRVTrait<IntT, FloatT> for MISIPair<IntT, FloatT>
where
    IntT: H5Type + MRVInteger,
    FloatT: H5Type + MRVFloat,
{
    fn get_hist(&self, i: IntT) -> Result<Array1<FloatT>, Error> {
        let ridx = self.range_of(i)?;
        Ok(self.hist.at(ridx).to_owned())
    }

    fn get_hist_dim(&self, i: IntT) -> Result<IntT, Error> {
        let ridx: usize = self.range_of(i)?;
        Ok(*self.hist_dim.at(ridx))
    }
    fn get_si(&self, about: IntT, by: IntT) -> Result<Array1<FloatT>, Error> {
        let ridx = self.range_of(about)?;
        let srange = self._si_bounds(about, by)?;
        Ok(self.si.at(ridx).slice(ndarray::s![srange]).to_owned())
    }
    fn si_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error> {
        let ridx = self.range_of(about)?;
        let rstate_idx =
            self._si_bounds_start(about, by)? + rstate.to_usize().unwrap();
        Ok(self.si.at(ridx)[rstate_idx])
    }

    fn get_lmr(&self, about: IntT, by: IntT) -> Result<Array1<FloatT>, Error> {
        let ridx = self.range_of(about)?;
        let srange = self._si_bounds(about, by)?;
        Ok(self.lmr.at(ridx).slice(ndarray::s![srange]).to_owned())
    }

    fn lmr_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error> {
        let ridx = self.range_of(about)?;
        let rstate_idx =
            self._si_bounds_start(about, by)? + rstate.to_usize().unwrap();
        Ok(self.lmr.at(ridx)[rstate_idx])
    }

    fn get_mi(&self, _i: IntT, _j: IntT) -> Result<FloatT, Error> {
        Ok(self.mi)
    }

    fn ndata(&self) -> FloatT {
        FloatT::from_usize(self.nobservations()).unwrap()
    }

    fn nobservations(&self) -> usize {
        self.nobs.to_usize().unwrap()
    }

    fn nvariables(&self) -> usize {
        self.nvars.to_usize().unwrap()
    }

    fn get_lmr_minsum(&self, about: IntT, target: IntT) -> Result<FloatT, Error> {
        let ridx = self.range_of(about)?;
        match &self.lmr_ds {
            None => Ok(self.minsum_list(about, target)?.sum()),
            Some(lmr_dsv) => match lmr_dsv {
                LMRDSPair::Complete(complete_ds) => {
                    complete_ds.at(ridx).minsum(target)
                }
                LMRDSPair::Subset(subset_ds) => {
                    subset_ds.at(ridx).minsum(self, target)
                }
            },
        }
    }

    fn get_puc_factor(&self, about: IntT, target: IntT) -> Result<FloatT, Error> {
        let ridx = self.range_of(about)?;
        let pfactor = match &self.lmr_ds {
            None => self.nvariables() - 2,
            Some(lmr_dsv) => match lmr_dsv {
                LMRDSPair::Complete(complete_ds) => {
                    complete_ds.at(ridx).mi_factor(target)
                }
                LMRDSPair::Subset(subset_ds) => {
                    subset_ds.at(ridx).mi_factor(target)
                }
            },
        };
        Ok(FloatT::from_usize(pfactor).unwrap())
    }

    fn compute_lm_puc(&self, i: IntT, j: IntT) -> Result<FloatT, Error> {
        let mij = self.get_mi(i, j)?;
        if mij <= FloatT::zero() {
            return Ok(mij);
        }
        let (pij_factor, pji_factor) =
            (self.get_puc_factor(i, j)?, self.get_puc_factor(j, i)?);
        Ok((pij_factor - (self.get_lmr_minsum(i, j)? / mij))
            + (pji_factor - (self.get_lmr_minsum(j, i)? / mij)))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use lazy_static::lazy_static;
    use log::debug;
    use std::collections::HashMap;

    use super::{LMRDSPair, MISIPair, MISIRangePair};
    use crate::mvim::rv::{Error, MRVTrait};
    use crate::test_data_file_path;

    lazy_static! {
        static ref MISI_H5: &'static str =
            test_data_file_path!("adata.20k.500.misidata.h5");
        static ref PAIRS_LIST: Vec<(i32, i32)> =
            vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];
        static ref PAIRS_MI_PUC: HashMap<(i32, i32), (f32, f32)> =
            HashMap::from([
                ((0, 1), (0.31067935, 60.68014)),
                ((0, 2), (1.2132332, 205.43738)),
                ((0, 3), (2.4000857, 480.43027)),
                ((0, 4), (0.7714661, 128.89265)),
                ((0, 5), (0.82867515, 141.27827)),
            ]);
        static ref SAMPLES_LIST: Vec<Vec<i32>> = vec![
            vec![
                11, 38, 149, 162, 169, 172, 183, 209, 222, 255, 258, 307, 324,
                354, 361, 370, 374, 412, 455, 491
            ],
            vec![
                0, 5, 45, 104, 106, 140, 153, 154, 178, 227, 229, 248, 278, 314,
                338, 356, 412, 431, 477, 495
            ],
        ];
        static ref SAMPLES_PUC: HashMap<(i32, i32), Vec<f32>> = HashMap::from([
            ((0, 1), vec![1.769476, 2.728235]),
            ((0, 2), vec![10.56831, 9.400658]),
            ((0, 3), vec![22.529241, 19.741033]),
            ((0, 4), vec![6.253071, 5.8040924]),
            ((0, 5), vec![7.0357942, 6.366231]),
        ]);
        static ref PAIRS_LIST2: Vec<(i32, i32)> =
            vec![(0, 1), (0, 5), (1, 5), (0, 45)];
        static ref PAIRS_MI_PUC2: HashMap<(i32, i32), (f32, f32)> =
            HashMap::from([
                ((0, 1), (0.3106793, 60.680137)),
                ((0, 5), (0.8286751, 141.27827)),
                ((1, 5), (0.5558097, 200.69673)),
                ((0, 45), (1.684678, 294.56713)),
            ]);
        static ref SAMPLES_LIST2: Vec<Vec<i32>> = vec![
            vec![
                5, 45, 104, 106, 140, 153, 154, 178, 227, 229, 248, 278, 314,
                338, 356, 412, 431, 477, 495
            ],
            vec![
                0, 5, 45, 104, 106, 140, 153, 154, 178, 227, 229, 248, 278, 314,
                338, 356, 412, 431, 477, 495
            ],
            vec![
                1, 5, 45, 104, 106, 140, 153, 154, 178, 227, 229, 248, 278, 314,
                338, 356, 412, 431, 477, 495
            ],
            vec![
                0, 1, 5, 45, 104, 106, 140, 153, 154, 178, 227, 229, 248, 278,
                314, 338, 356, 412, 431, 477, 495
            ],
        ];
        static ref SAMPLES_PUC2: HashMap<(i32, i32), Vec<f32>> = HashMap::from([
            ((0, 1), vec![2.72823, 2.72823, 2.72823, 2.72823]),
            ((0, 5), vec![6.36623, 6.36623, 7.57903, 7.57903]),
            ((0, 45), vec![12.87396, 12.87396, 14.51176, 14.51176]),
            ((1, 5), vec![7.06674, 7.94825, 7.06674, 7.94825]),
        ]);
    }

    #[test]
    pub fn test_misi_pair_full() -> Result<()> {
        crate::tests::log_init();
        for ((pi, pj), (rmi, rpuc)) in PAIRS_MI_PUC.iter() {
            let (i, j) = (*pi, *pj);
            let bpair =
                MISIPair::<i32, f32>::new(&MISI_H5, (i as usize, j as usize))?;
            let pmi = bpair.get_mi(i, j)?;
            let ppuc = bpair.accumulate_redundancies(i, j)?;
            let lpuc = bpair.compute_lm_puc(i, j)?;
            let mi_diff = (rmi - pmi).abs() < 1e-4;
            let puc_diff = (rpuc - ppuc).abs() < 1e-3;
            let lpuc_diff = (rpuc - lpuc).abs() < 1e-3;
            debug!(
                "i, j: ({}, {}); MIs ({}, {}, {}); PUC: ({} {} {} {} {})",
                i, j, rmi, pmi, mi_diff, rpuc, ppuc, puc_diff, lpuc, lpuc_diff,
            );
            assert!(mi_diff);
            assert!(puc_diff);
            assert!(lpuc_diff);
        }
        Ok(())
    }

    #[test]
    pub fn test_misi_range_pair_full() -> Result<()> {
        crate::tests::log_init();
        let mut bprange = MISIRangePair::<i32, f32>::new(&MISI_H5, (0..6, 0..6))?;
        for ((pi, pj), (rmi, rpuc)) in PAIRS_MI_PUC.iter() {
            let (i, j) = (*pi, *pj);
            let pmi = bprange.get_mi(i, j)?;
            let ppuc = bprange.accumulate_redundancies(i, j)?;
            let lpuc = bprange.compute_lm_puc(i, j)?;
            bprange.set_lmr_ds(None)?;
            assert!(bprange.lmr_ds.is_some());
            let dspuc = bprange.compute_lm_puc(i, j)?;
            let mi_diff = (rmi - pmi).abs() < 1e-4;
            let puc_diff = (rpuc - ppuc).abs() < 1e-3;
            let lpuc_diff = (rpuc - lpuc).abs() < 1e-3;
            let dspuc_diff = (dspuc - lpuc).abs() < 1e-3;
            debug!(
                "i, j: ({}, {}); MIs ({}, {}, {}); PUC: [({} ({} {}) ({} {}) ({} {})]",
                i, j, rmi, pmi, mi_diff, rpuc, ppuc, puc_diff, lpuc, lpuc_diff,
                dspuc, dspuc_diff,
            );
            assert!(mi_diff);
            assert!(puc_diff);
            assert!(lpuc_diff);
            assert!(dspuc_diff);
        }
        Ok(())
    }

    #[test]
    pub fn test_misi_pair_subset_minsum() -> Result<()> {
        crate::tests::log_init();
        let (exp_fnm, exp_snm) = (5.9917197, 5.8857145);
        let (i, j) = PAIRS_LIST[0];
        let splist = SAMPLES_LIST[0].clone();
        let mut bpair =
            MISIPair::<i32, f32>::new(&MISI_H5, (i as usize, j as usize))?;
        bpair.set_lmr_ds(Some(&splist))?;
        if let Some(LMRDSPair::Subset(ref subset_dss)) = bpair.lmr_ds {
            let (fmin, smin) = (
                subset_dss.first.minsum(&bpair, j)?,
                subset_dss.second.minsum(&bpair, i)?,
            );
            debug!("-> {:?} {} {} {}", (i, j), fmin, smin, fmin + smin);
            let flmr_diff = (fmin - exp_fnm).abs() < 1e-3;
            let slmr_diff = (smin - exp_snm).abs() < 1e-3;
            assert!(flmr_diff);
            assert!(slmr_diff);
        } else {
            return Err(anyhow::Error::from(Error::Unexpected(
                "LMRDSPair not initialized",
            )));
        }
        let (fmm, smm) =
            (bpair.get_lmr_minsum(i, j)?, bpair.get_lmr_minsum(j, i)?);
        let flmr_diff = (fmm - exp_fnm).abs() < 1e-3;
        let slmr_diff = (smm - exp_snm).abs() < 1e-3;
        debug!("-> BPAIR MINSUM {} {} {}", fmm, smm, fmm + smm);
        assert!(flmr_diff);
        assert!(slmr_diff);
        //
        for (idx, rvlist) in SAMPLES_LIST2.iter().enumerate() {
            bpair.set_lmr_ds(Some(rvlist))?;
            if let Some(LMRDSPair::Subset(ref subset_dss)) = bpair.lmr_ds {
                let (fmin, smin) = (
                    subset_dss.first.minsum(&bpair, j)?,
                    subset_dss.second.minsum(&bpair, i)?,
                );
                let rsum = fmin + smin;
                debug!(" -> I {} {} {} {}", idx, fmin, smin, rsum);
                // rsum == 10.958
                let rsum_diff = (rsum - 10.958).abs() < 1e-3;
                assert!(rsum_diff);
            }
        }
        Ok(())
    }

    #[test]
    pub fn test_misi_pair_subset_lmr1() -> Result<()> {
        crate::tests::log_init();
        for ((i, j), puc_vec) in SAMPLES_PUC.iter() {
            let mut bpair =
                MISIPair::<i32, f32>::new(&MISI_H5, (*i as usize, *j as usize))?;
            let (exp_mi, exp_puc) = PAIRS_MI_PUC[&(*i, *j)];
            let fpuc = bpair.accumulate_redundancies(*i, *j)?;
            for (idx, rvlist) in SAMPLES_LIST.iter().enumerate() {
                bpair.set_lmr_ds(Some(rvlist))?;
                let pucij = bpair.accumulate_redundancies_for(*i, *j, rvlist)?;
                let mij = bpair.get_mi(*i, *j)?;
                let lmrij = bpair.compute_lm_puc(*i, *j)?;
                let msij = (
                    bpair.get_lmr_minsum(*i, *j)?,
                    bpair.get_lmr_minsum(*j, *i)?,
                );
                let rsum = msij.0 + msij.1;
                debug!(
                    " -> IJ {:?} {:?} {} {:?}",
                    (i, j, idx),
                    (mij, exp_mi, fpuc, exp_puc),
                    rsum,
                    (pucij, lmrij, puc_vec[idx])
                );
                let mi_diff = (mij - exp_mi).abs() < 1e-3;
                let puc_diff = (fpuc - exp_puc).abs() < 1e-3;
                let acc_diff = (pucij - puc_vec[idx]).abs() < 1e-3;
                let lmr_diff = (lmrij - puc_vec[idx]).abs() < 1e-3;
                assert!(mi_diff);
                assert!(puc_diff);
                assert!(acc_diff);
                assert!(lmr_diff);
            }
        }
        Ok(())
    }

    #[test]
    pub fn test_misi_pair_subset_lmr2() -> Result<()> {
        crate::tests::log_init();
        for ((i, j), puc_vec) in SAMPLES_PUC2.iter() {
            let (exp_mi, exp_puc) = PAIRS_MI_PUC2[&(*i, *j)];
            let mut bpair =
                MISIPair::<i32, f32>::new(&MISI_H5, (*i as usize, *j as usize))?;
            let fpuc = bpair.accumulate_redundancies(*i, *j)?;
            for (idx, rvlist) in SAMPLES_LIST2.iter().enumerate() {
                bpair.set_lmr_ds(Some(rvlist))?;
                let pucij = bpair.accumulate_redundancies_for(*i, *j, rvlist)?;
                let mij = bpair.get_mi(*i, *j)?;
                let lmrij = bpair.compute_lm_puc(*i, *j)?;
                let msij = (
                    bpair.get_lmr_minsum(*i, *j)?,
                    bpair.get_lmr_minsum(*j, *i)?,
                );
                let rsum = msij.0 + msij.1;
                debug!(
                    " --> IJ {:?} {:?} {} {:?}",
                    (i, j, idx),
                    (mij, exp_mi, fpuc, exp_puc),
                    rsum,
                    (pucij, lmrij, puc_vec[idx])
                );
                let puc_diff = (fpuc - exp_puc).abs() < 1e-3;
                let mi_diff = (mij - exp_mi).abs() < 1e-3;
                let acc_diff = (pucij - puc_vec[idx]).abs() < 1e-3;
                let lmr_diff = (lmrij - puc_vec[idx]).abs() < 1e-3;
                assert!(mi_diff);
                assert!(puc_diff);
                assert!(acc_diff);
                assert!(lmr_diff);
            }
        }
        Ok(())
    }

    #[test]
    pub fn test_misi_range_pair_subset_lmr1() -> Result<()> {
        crate::tests::log_init();
        let mut bprange = MISIRangePair::<i32, f32>::new(&MISI_H5, (0..6, 0..6))?;
        for ((i, j), puc_vec) in SAMPLES_PUC.iter() {
            let (exp_mi, exp_puc) = PAIRS_MI_PUC[&(*i, *j)];
            let fpuc = bprange.accumulate_redundancies(*i, *j)?;
            for (idx, rvlist) in SAMPLES_LIST.iter().enumerate() {
                bprange.set_lmr_ds(Some(rvlist))?;
                let pucij =
                    bprange.accumulate_redundancies_for(*i, *j, rvlist)?;
                let mij = bprange.get_mi(*i, *j)?;
                let lmrij = bprange.compute_lm_puc(*i, *j)?;
                let msij = (
                    bprange.get_lmr_minsum(*i, *j)?,
                    bprange.get_lmr_minsum(*j, *i)?,
                );
                let rsum = msij.0 + msij.1;
                debug!(
                    " -> IJ {:?} {:?} {} {:?}",
                    (i, j, idx),
                    (mij, exp_mi, fpuc, exp_puc),
                    rsum,
                    (pucij, lmrij, puc_vec[idx])
                );
                let mi_diff = (mij - exp_mi).abs() < 1e-3;
                let puc_diff = (fpuc - exp_puc).abs() < 1e-3;
                let acc_diff = (pucij - puc_vec[idx]).abs() < 1e-3;
                let lmr_diff = (lmrij - puc_vec[idx]).abs() < 1e-3;
                assert!(mi_diff);
                assert!(puc_diff);
                assert!(acc_diff);
                assert!(lmr_diff);
            }
        }
        Ok(())
    }

    #[test]
    pub fn test_misi_range_pair_subset_lmr2() -> Result<()> {
        crate::tests::log_init();
        let mut bprange =
            MISIRangePair::<i32, f32>::new(&MISI_H5, (0..6, 0..46))?;
        for ((i, j), puc_vec) in SAMPLES_PUC2.iter() {
            let (exp_mi, exp_puc) = PAIRS_MI_PUC2[&(*i, *j)];
            let fpuc = bprange.accumulate_redundancies(*i, *j)?;
            for (idx, rvlist) in SAMPLES_LIST2.iter().enumerate() {
                bprange.set_lmr_ds(Some(rvlist))?;
                let pucij =
                    bprange.accumulate_redundancies_for(*i, *j, rvlist)?;
                let mij = bprange.get_mi(*i, *j)?;
                let lmrij = bprange.compute_lm_puc(*i, *j)?;
                let msij = (
                    bprange.get_lmr_minsum(*i, *j)?,
                    bprange.get_lmr_minsum(*j, *i)?,
                );
                let rsum = msij.0 + msij.1;
                debug!(
                    " --> IJ {:?} {:?} {} {:?}",
                    (i, j, idx),
                    (mij, exp_mi, fpuc, exp_puc),
                    rsum,
                    (pucij, lmrij, puc_vec[idx])
                );
                let puc_diff = (fpuc - exp_puc).abs() < 1e-3;
                let mi_diff = (mij - exp_mi).abs() < 1e-3;
                let acc_diff = (pucij - puc_vec[idx]).abs() < 1e-3;
                let lmr_diff = (lmrij - puc_vec[idx]).abs() < 1e-3;
                assert!(mi_diff);
                assert!(puc_diff);
                assert!(acc_diff);
                assert!(lmr_diff);
            }
        }
        Ok(())
    }
}
