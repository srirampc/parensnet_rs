use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use crate::{
    mvim::imeasures::{self, redundancy},
    types::{AssignOps, DbgDisplay, FromToPrimitive, OrderedFloat, OrderedInt},
};

pub trait MRVInteger:
    OrderedInt + FromToPrimitive + AssignOps + Hash + DbgDisplay + Copy + Clone
{
}
impl<
    T: OrderedInt + FromToPrimitive + AssignOps + Hash + DbgDisplay + Copy + Clone,
> MRVInteger for T
{
}

pub trait MRVFloat:
    OrderedFloat + FromToPrimitive + AssignOps + DbgDisplay + Copy + Clone
{
}
impl<T: OrderedFloat + FromToPrimitive + AssignOps + DbgDisplay + Copy + Clone>
    MRVFloat for T
{
}

#[derive(Debug)]
pub enum Error {
    InvalidIndex(usize),
    InvalidAbout(usize),
    InvalidBy(usize),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidIndex(idx) => {
                write!(f, "Invalid Index : {idx}")
            }
            Error::InvalidAbout(abt) => {
                write!(f, "Invalid About : {abt}")
            }
            Error::InvalidBy(rby) => {
                write!(f, "Invalid By    : {rby}")
            }
        }
    }
}

impl std::error::Error for Error {}

pub trait MRVTrait<IntT: 'static + MRVInteger, FloatT: 'static + MRVFloat> {
    fn get_hist(&self, i: IntT) -> Result<Array1<FloatT>, Error>;
    fn get_hist_dim(&self, i: IntT) -> Result<IntT, Error>;
    fn get_mi(&self, i: IntT, j: IntT) -> Result<FloatT, Error>;
    fn get_si(&self, about: IntT, by: IntT) -> Result<Array1<FloatT>, Error>;
    fn si_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error>;
    fn get_lmr(&self, i: IntT, j: IntT) -> Result<Array1<FloatT>, Error>;
    fn lmr_value(
        &self,
        about: IntT,
        by: IntT,
        rstate: IntT,
    ) -> Result<FloatT, Error>;
    fn ndata(&self) -> FloatT;
    fn nobservations(&self) -> usize;
    fn nvariables(&self) -> usize;

    //
    fn get_redundancies(
        &self,
        i: IntT,
        j: IntT,
        k: IntT,
    ) -> Result<(FloatT, FloatT, FloatT), Error> {
        Ok((
            imeasures::redundancy(
                self.get_hist(i)?.view(),
                self.get_si(i, j)?.view(),
                self.get_si(i, k)?.view(),
                Some(self.ndata()),
            ),
            imeasures::redundancy(
                self.get_hist(j)?.view(),
                self.get_si(j, i)?.view(),
                self.get_si(j, k)?.view(),
                Some(self.ndata()),
            ),
            imeasures::redundancy(
                self.get_hist(k)?.view(),
                self.get_si(k, i)?.view(),
                self.get_si(k, j)?.view(),
                Some(self.ndata()),
            ),
        ))
    }

    //
    fn mpuc(
        &self,
        i: IntT,
        j: IntT,
        redundancy: FloatT,
    ) -> Result<FloatT, Error> {
        let mi = self.get_mi(i, j)?;
        let puc_score = (mi - redundancy) / mi;
        if puc_score.is_finite() && puc_score >= FloatT::zero() {
            Ok(puc_score)
        } else {
            Ok(FloatT::zero())
        }
    }

    //
    fn redundancy_updates(
        &self,
        i: IntT,
        j: IntT,
        k: IntT,
    ) -> Result<(FloatT, FloatT, FloatT), Error> {
        assert!(i < j);
        let (ri, rj, rk) = self.get_redundancies(i, j, k)?;
        Ok((
            self.mpuc(i, j, ri)? + self.mpuc(i, j, rj)?,
            self.mpuc(i, k, ri)? + self.mpuc(i, k, rk)?,
            self.mpuc(k, k, rj)? + self.mpuc(j, k, rk)?,
        ))
    }

    //
    fn redundancy_update_for(
        &self,
        i: IntT,
        j: IntT,
        by: IntT,
    ) -> Result<FloatT, Error> {
        assert!(i < j);
        assert!(i != by);
        assert!(j != by);
        let iby_si = self.get_si(i, by)?;
        let jby_si = self.get_si(j, by)?;
        let ri = redundancy(
            self.get_hist(i)?.view(),
            self.get_si(i, j)?.view(),
            iby_si.view(),
            Some(self.ndata()),
        );
        let rj = redundancy(
            self.get_hist(j)?.view(),
            self.get_si(j, i)?.view(),
            jby_si.view(),
            Some(self.ndata()),
        );
        Ok(self.mpuc(i, j, ri)? + self.mpuc(i, j, rj)?)
    }

    //
    fn accumulate_redundancies_for(
        &self,
        i: IntT,
        j: IntT,
        //by_nodes: impl Iterator<Item = &'a i32>,
        by_nodes: &[IntT],
    ) -> Result<FloatT, Error> {
        let mut acc: FloatT = FloatT::zero();
        for bx in by_nodes {
            if *bx != i && *bx != j {
                acc += self.redundancy_update_for(i, j, *bx)?
            }
        }
        Ok(acc)
    }

    fn accumulate_redundancies(&self, i: IntT, j: IntT) -> Result<FloatT, Error> {
        let nvars: usize = self.nvariables();
        let vrange: Vec<IntT> =
            (0..nvars).map(|vx| IntT::from_usize(vx).unwrap()).collect();
        self.accumulate_redundancies_for(i, j, &vrange)
    }

    //
    fn compute_puc_matrix(&self) -> Result<Array2<FloatT>, Error> {
        let nvars = self.nvariables();
        let mut puc_network: Array2<FloatT> = Array2::zeros((nvars, nvars));

        for cvec in (0..nvars).combinations(3) {
            let i = cvec[0];
            let j = cvec[1];
            let k = cvec[2];
            let (rij, rik, rjk) = self.redundancy_updates(
                IntT::from_usize(i).unwrap(),
                IntT::from_usize(j).unwrap(),
                IntT::from_usize(k).unwrap(),
            )?;
            puc_network[[i, j]] += rij;
            puc_network[[i, k]] += rik;
            puc_network[[j, k]] += rjk;
        }
        for cvec in (0..nvars).combinations(2) {
            let i = cvec[0];
            let j = cvec[1];
            puc_network[[j, i]] = puc_network[[i, j]];
        }
        Ok(puc_network)
    }

    //
    fn compute_puc_matrix_for(
        &self,
        by_nodes: &[IntT],
    ) -> Result<Array2<FloatT>, Error> {
        let nvars = self.nvariables();
        let mut puc_network: Array2<FloatT> = Array2::zeros((nvars, nvars));
        for cvec in (0..nvars).combinations(2) {
            let i = cvec[0];
            let j = cvec[1];
            puc_network[[i, j]] = self.accumulate_redundancies_for(
                IntT::from_usize(i).unwrap(),
                IntT::from_usize(j).unwrap(),
                by_nodes,
            )?
        }
        Ok(puc_network)
    }

    //
    fn compute_redundancies(
        &self,
    ) -> Result<HashMap<(i32, i32, i32), FloatT>, Error> {
        let mut hsmap: HashMap<(i32, i32, i32), FloatT> = HashMap::new();
        let nvars = self.nvariables() as i32;
        if nvars == 0 {
            return Ok(hsmap);
        }
        for cvec in (0..nvars).combinations(3) {
            let i = cvec[0];
            let j = cvec[1];
            let k = cvec[2];
            let (rij, rik, rjk) = self.redundancy_updates(
                IntT::from_i32(i).unwrap(),
                IntT::from_i32(j).unwrap(),
                IntT::from_i32(k).unwrap(),
            )?;
            hsmap.insert((i, j, k), rij);
            hsmap.insert((i, k, j), rik);
            hsmap.insert((k, j, i), rjk);
        }
        Ok(hsmap)
    }

    fn minsum_list(
        &self,
        about: IntT,
        target: IntT,
    ) -> Result<Array1<FloatT>, Error> {
        let nvars: usize = self.nvariables();
        let lmr_abtgt = self.get_lmr(about, target)?;
        let by_nodes: Vec<IntT> = (0..nvars)
            .map(|x| IntT::from_usize(x).unwrap())
            .filter(|x| *x != about && *x != target)
            .collect();
        let mut ms_list =
            Array1::<FloatT>::from_elem(by_nodes.len(), FloatT::zero());
        for bx in 0..by_nodes.len() {
            let lmx = self.get_lmr(about, by_nodes[bx])?;
            ms_list[bx] = std::iter::zip(lmx.iter(), lmr_abtgt.iter())
                .fold(FloatT::zero(), |acc, (lma, lmb)| acc + lma.min(*lmb));
        }
        Ok(ms_list)
    }

    fn get_lmr_minsum(&self, about: IntT, target: IntT) -> Result<FloatT, Error> {
        Ok(self.minsum_list(about, target)?.sum())
    }

    fn get_puc_factor(&self, _about: IntT, _target: IntT) -> Result<FloatT, Error> {
        Ok(FloatT::from_usize(self.nvariables() - 2).unwrap())
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

    //
    // def get_lmr_minsum(self, about:int, target:int) -> FloatT:
    //     by_nodes: list[int] = list(
    //         x for x in range(self.nvariables) if x != about and x != target
    //     )
    //     lmr_abtgt = self.get_lmr(about=about, by=target)
    //     min_sum_list = np.array([
    //         np.sum(np.minimum(lmr_abtgt, self.get_lmr(about=about, by=by)))
    //         for by in by_nodes
    //     ], self.float_dtype())
    //     return np.sum(min_sum_list)

    // def get_puc_factor(self, about: int, target:int): # pyright: ignore[reportUnusedParameter]
    //     return np.float32(self.nvariables - 2).astype(self.float_dtype())

    // def compute_lmr_puc(self, i:int, j:int):
    //     mij = self.get_mi(i, j)
    //     if mij == 0:
    //         return np.float32(0.0).astype(self.float_dtype())
    //     mi_factor = self.get_puc_factor(i, j)
    //     return (
    //         ( mi_factor - (self.get_lmr_minsum(i, j) / mij) ) +
    //         ( mi_factor - (self.get_lmr_minsum(j, i) / mij) )
    //     )
}

//
// Struct containing sorted lmr arrays
//
pub struct LMRSA<IntT, FloatT>
where
    IntT: 'static + MRVInteger,
    FloatT: 'static + MRVFloat,
{
    _size: usize,
    nvars: usize,
    dim: usize,
    sorted: Array1<FloatT>,
    pfxsum: Array1<FloatT>,
    rank: Array1<IntT>,
}

impl<IntT, FloatT> LMRSA<IntT, FloatT>
where
    IntT: 'static + MRVInteger,
    FloatT: 'static + MRVFloat,
{
    #![allow(clippy::needless_range_loop)]
    pub fn from_mrv_trait(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
    ) -> Result<Self, Error> {
        let nvars = pidata.nvariables();
        let dim = pidata.get_hist_dim(about)?.to_usize().unwrap();
        let size = dim * nvars;
        let mut siv_lst: Vec<Vec<(FloatT, IntT)>> =
            vec![vec![(FloatT::zero(), IntT::zero()); nvars]; dim];
        //
        for vidx in 0..nvars {
            let by_var = IntT::from_usize(vidx).unwrap();
            let lmr_ax = pidata.get_lmr(about, by_var)?;
            for rstate in 0..dim {
                siv_lst[rstate][vidx] = (lmr_ax[rstate], by_var)
            }
        }
        for si_vec in siv_lst.iter_mut() {
            si_vec.sort_by(|(fa, _ia), (fb, _ib)| fa.total_cmp(fb));
        }
        let mut sorted: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut pfxsum: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut rank: Array1<IntT> = Array1::from_elem(size, IntT::zero());
        for (rstate, si_vec) in siv_lst.iter().enumerate() {
            let rsbegin = rstate * nvars;
            let mut curr_sum = FloatT::zero();
            for (ix, (svx, by_var)) in si_vec.iter().enumerate() {
                let by_idx = by_var.to_usize().unwrap();
                curr_sum += *svx;
                rank[rsbegin + by_idx] = IntT::from_usize(ix).unwrap();
                sorted[rsbegin + ix] = *svx;
                pfxsum[rsbegin + ix] = curr_sum;
            }
        }

        Ok(LMRSA {
            _size: size,
            nvars,
            dim,
            sorted,
            pfxsum,
            rank,
        })
    }

    pub fn from_subset_map(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
        subset_map: &HashMap<IntT, usize>,
    ) -> Result<Self, Error> {
        let nvars = subset_map.len();
        let dim = pidata.get_hist_dim(about)?.to_usize().unwrap();
        let size = dim * subset_map.len();
        let mut siv_lst: Vec<Vec<(FloatT, IntT)>> =
            vec![vec![(FloatT::zero(), IntT::zero()); nvars]; dim];
        //
        for (by_var, by_idx) in subset_map.iter() {
            let lmr_ax = pidata.get_lmr(about, *by_var)?;
            for rstate in 0..dim {
                siv_lst[rstate][*by_idx] = (lmr_ax[rstate], *by_var)
            }
        }
        for si_vec in siv_lst.iter_mut() {
            si_vec.sort_by(|(fa, _ia), (fb, _ib)| fa.total_cmp(fb));
        }
        let mut sorted: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut pfxsum: Array1<FloatT> = Array1::from_elem(size, FloatT::zero());
        let mut rank: Array1<IntT> = Array1::from_elem(size, IntT::zero());
        for (rstate, si_vec) in siv_lst.iter().enumerate() {
            let rsbegin = rstate * nvars;
            let mut curr_sum = FloatT::zero();
            for (ix, (svx, by_var)) in si_vec.iter().enumerate() {
                let by_idx = subset_map[by_var];
                curr_sum += *svx;
                rank[rsbegin + by_idx] = IntT::from_usize(ix).unwrap();
                sorted[rsbegin + ix] = *svx;
                pfxsum[rsbegin + ix] = curr_sum;
            }
        }
        Ok(LMRSA {
            _size: size,
            nvars,
            dim,
            sorted,
            pfxsum,
            rank,
        })
    }

    pub fn minsum_wsrc(&self, src_idx: usize) -> FloatT {
        let mut rdsum = FloatT::zero();
        for rstate in 0..self.dim {
            let rsbegin = rstate * self.nvars;
            let lmrank = self.rank[rsbegin + src_idx].to_usize().unwrap();
            let lmv = self.sorted[rsbegin + lmrank];
            let lmlow = if lmrank > 0 {
                self.pfxsum[rsbegin + lmrank - 1]
            } else {
                FloatT::zero()
            };
            let lmhigh =
                FloatT::from_usize(self.nvars - 1 - lmrank).unwrap() * lmv;
            rdsum += lmlow + lmhigh;
        }
        rdsum
    }

    pub fn minsum_nosrc(&self, lmd: ArrayView1<FloatT>) -> FloatT {
        assert!(lmd.len() == self.dim);
        let mut rdsum = FloatT::zero();
        for (rstate, lmv) in lmd.iter().enumerate() {
            let rsbegin = rstate * self.nvars;
            let rsend = rsbegin + self.nvars;
            let lmrank = match self
                .sorted
                .slice(ndarray::s![rsbegin..rsend])
                .as_slice()
                .unwrap()
                .binary_search_by(|x| x.total_cmp(lmv))
            {
                Ok(pos) => pos,
                Err(pos) => pos,
            };
            let lmlow = if lmrank > 0 {
                self.pfxsum[rsbegin + lmrank - 1]
            } else {
                FloatT::zero()
            };
            let lmhigh = FloatT::from_usize(self.nvars - lmrank).unwrap() * *lmv;
            rdsum += lmlow + lmhigh;
        }
        rdsum
    }
}

pub struct LMRDataStructure<IntT, FloatT>
where
    IntT: 'static + MRVInteger,
    FloatT: 'static + MRVFloat,
{
    about: IntT,
    nvars: usize,
    lmr: LMRSA<IntT, FloatT>,
}

impl<IntT, FloatT> LMRDataStructure<IntT, FloatT>
where
    IntT: 'static + MRVInteger,
    FloatT: 'static + MRVFloat,
{
    pub fn new(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
    ) -> Result<Self, Error> {
        Ok(LMRDataStructure {
            about,
            nvars: pidata.nvariables(),
            lmr: LMRSA::from_mrv_trait(pidata, about)?,
        })
    }

    pub fn get_about(&self) -> IntT {
        self.about
    }

    pub fn mi_factor(&self, _target: IntT) -> usize {
        self.nvars - 2
    }

    pub fn minsum(&self, src_var: IntT) -> Result<FloatT, Error> {
        Ok(self.lmr.minsum_wsrc(src_var.to_usize().unwrap()))
    }
}

pub struct LMRSubsetDataStructure<IntT, FloatT>
where
    IntT: 'static + MRVInteger,
    FloatT: 'static + MRVFloat,
{
    subset_map: Rc<HashMap<IntT, usize>>,
    about: IntT,
    nvars: usize,
    lmr: LMRSA<IntT, FloatT>,
}

impl<IntT, FloatT> LMRSubsetDataStructure<IntT, FloatT>
where
    IntT: 'static + MRVInteger,
    FloatT: 'static + MRVFloat,
{
    pub fn new(
        pidata: &impl MRVTrait<IntT, FloatT>,
        about: IntT,
        subset_map: Rc<HashMap<IntT, usize>>,
    ) -> Result<Self, Error> {
        Ok(LMRSubsetDataStructure {
            about,
            nvars: subset_map.len(),
            lmr: LMRSA::from_subset_map(pidata, about, &subset_map)?,
            subset_map,
        })
    }

    pub fn get_about(&self) -> IntT {
        self.about
    }

    pub fn mi_factor(&self, target: IntT) -> usize {
        self.nvars
            - (if self.subset_map.contains_key(&self.about) {
                1
            } else {
                0
            } + if self.subset_map.contains_key(&target) {
                1
            } else {
                0
            })
    }

    pub fn minsum(
        &self,
        pidata: &impl MRVTrait<IntT, FloatT>,
        src_var: IntT,
    ) -> Result<FloatT, Error> {
        if self.subset_map.contains_key(&src_var) {
            Ok(self.lmr.minsum_wsrc(self.subset_map[&src_var]))
        } else {
            let lmd = pidata.get_lmr(self.about, src_var)?;
            Ok(self.lmr.minsum_nosrc(lmd.view()))
        }
    }
}
