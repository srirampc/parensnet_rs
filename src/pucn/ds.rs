//! Data containers used by the `pucn` workflows.
//!
//! Every type in this module is `pub(super)`. They form the in-memory
//! representation of:
//!
//! * per-variable histograms ([`Node`], [`NodeCollection`]),
//! * pair-wise mutual information and joint histograms
//!   ([`PairMI`], [`PairMICollection`]),
//! * specific-information / LMR vectors for ordered pairs
//!   ([`OrdPairSI`], [`OrdPairSICollection`]), and
//! * small tuple aliases ([`NodePair`], [`BatchPairs`],
//!   [`NodePairCollection`]) used as work-batch carriers between
//!   the helper functions in [`crate::pucn::helpers`] and the
//!   distributed workflows.
//!
//! Each `*Collection` type provides:
//!
//! * a `from_vec` builder that flattens a per-rank `Vec<*>` into the
//!   parallel-friendly flat layout (concatenated values plus
//!   per-element dimension/offset arrays);
//! * a `from_h5` builder that loads a previously persisted collection
//!   from an HDF5 file (using [`crate::h5::io`] / [`crate::h5::mpio`]);
//! * a `distribute` method that re-shuffles the contents across MPI
//!   ranks via the `all2all*` primitives.

use anyhow::{Ok, Result};
use hdf5::H5Type;
use mpi::traits::{Communicator, Equivalence};
use ndarray::{Array1, Array2, ArrayView1};
use num::{FromPrimitive, ToPrimitive, Zero};
use sope::{
    collective::{all2all_vec, all2allv_vec, allgather_one, allgatherv_full_vec},
    partition::{ArbitDist, Dist, InterleavedDist},
    reduction::allreduce_sum,
    timer::SectionTimer,
    util::exc_prefix_sum,
};
use std::{collections::HashMap, fmt::Debug, iter::zip, ops::Range};

use super::WorkflowArgs;
use crate::{
    comm::CommIfx,
    h5::{io, mpio},
    hist::{HSFloat, bayesian_blocks_bin_edges, histogram_1d},
    map_with_result_to_tuple,
    types::{AddFromZero, FromToPrimitive, PNInteger},
    util::{block_owner, block_range, triu_index_to_pair},
};

/// Histogram of a single variable produced by the Bayesian-blocks
/// discretisation kernel.
///
/// Stores the bin edges (`bins`) and the per-bin counts (`hist`)
/// alongside their cached lengths (`nbins`, `nhist`). 
/// In a parallel setting, one instance is generated per variable that a rank 
/// owns; the collection of `Node`s is then flattened into a [`NodeCollection`] 
/// for collective use.
pub(super) struct Node<IntT, FloatT> {
    /// Number of bin edges in [`Self::bins`].
    nbins: IntT,
    /// Number of histogram bins in [`Self::hist`] (typically
    /// `nbins - 1`).
    nhist: IntT,
    /// Bin edges produced by [`bayesian_blocks_bin_edges`].
    bins: Array1<FloatT>,
    /// Histogram counts produced by [`histogram_1d`], aligned with
    /// [`Self::bins`].
    hist: Array1<FloatT>,
}

impl<IntT, FloatT> Node<IntT, FloatT> {
    /// Direct field-wise constructor used by [`Self::from_data`].
    fn new(
        nbins: IntT,
        nhist: IntT,
        bins: Array1<FloatT>,
        hist: Array1<FloatT>,
    ) -> Self {
        Node {
            nbins,
            nhist,
            bins,
            hist,
        }
    }

    /// Build a [`Node`] for a single column of the expression matrix.
    ///
    /// Computes the Bayesian-blocks bin edges via
    /// [`bayesian_blocks_bin_edges`] and fills the histogram with
    /// [`histogram_1d`]. The cached `nbins` / `nhist` counters are
    /// derived from the resulting array lengths.
    pub fn from_data(c_data: ArrayView1<FloatT>) -> Self
    where
        FloatT: 'static + HSFloat,
        IntT: AddFromZero + FromPrimitive + Clone,
    {
        let cbins = bayesian_blocks_bin_edges(c_data);
        let chist =
            histogram_1d::<FloatT, FloatT>(c_data, cbins.as_slice().unwrap());
        Self::new(
            IntT::from_usize(cbins.len()).unwrap(),
            IntT::from_usize(chist.len()).unwrap(),
            cbins,
            chist,
        )
    }
}

/// Flattened collection of per-variable [`Node`]s.
///
/// Holds the bin edges and histograms of all variables in two
/// concatenated arrays (`abins` and `ahist`) plus per-variable
/// dimension and offset arrays that allow quick access of 
/// the slice correspdoning ot variable `i` 
/// (See [`Self::bins`] / [`Self::hist`]). Built either by
/// gathering local [`Node`]s across ranks ([`Self::from_nodes`]) or by
/// loading from disk ([`Self::from_h5`]).
///
/// The struct is generic over three numeric types:
///
/// * `SizeT` for offset / count fields large enough to index the
///   flattened arrays;
/// * `IntT`  for per-variable bin / histogram dimensions;
/// * `FloatT` for the bin-edge and histogram element type.
pub(super) struct NodeCollection<SizeT, IntT, FloatT> {
    /// Per-variable number of bin edges (length `nvars`).
    pub bin_dim: Array1<IntT>,
    /// Per-variable number of histogram bins (length `nvars`).
    pub hist_dim: Array1<IntT>,
    /// Exclusive prefix sum of [`Self::bin_dim`]; `bin_start[i]` is
    /// the offset of variable `i`'s edges inside [`Self::abins`].
    pub bin_start: Array1<SizeT>,
    /// Per-variable offset into the specific-information storage of
    /// pair collections; computed from [`Self::hist_dim`] with
    /// [`bin_dim.len()`](Array1::len) used as the seed value.
    pub si_start: Array1<SizeT>,
    /// Exclusive prefix sum of [`Self::hist_dim`]; `hist_start[i]` is
    /// the offset of variable `i`'s histogram inside [`Self::ahist`].
    pub hist_start: Array1<SizeT>,
    /// Total specific-information storage size:
    /// `(sum of hist_dim) * nvars`.
    pub nsi: SizeT,
    // bins/hist flattened to a histogram
    /// Concatenated bin edges across all variables.
    pub abins: Array1<FloatT>,
    /// Concatenated histogram counts across all variables.
    pub ahist: Array1<FloatT>,
}

impl<SizeT, IntT, FloatT> NodeCollection<SizeT, IntT, FloatT> {
    /// Gather a slice of per-rank [`Node`]s into a single,
    /// flattened collection.
    ///
    /// Uses [`allgatherv_full_vec`] to concatenate the local
    /// `bin_dim` / `hist_dim` arrays and the corresponding bin / hist
    /// payloads across all ranks, then computes the offset arrays
    /// (`bin_start`, `hist_start`, `si_start`) and the total
    /// specific-information storage size [`Self::nsi`].
    pub fn from_nodes(
        v_nodes: &[Node<IntT, FloatT>],
        comm: &dyn Communicator,
    ) -> Result<Self>
    where
        SizeT: 'static + PNInteger + Equivalence,
        IntT: Clone + Debug + Default + ToPrimitive + Equivalence,
        FloatT: Clone + Debug + Default + Equivalence,
    {
        // bin/hist sizes
        let bin_dim: Vec<IntT> =
            v_nodes.iter().map(|x| x.nbins.clone()).collect();
        let bin_dim: Vec<IntT> = allgatherv_full_vec(&bin_dim, comm)?;
        let hist_dim: Vec<IntT> =
            v_nodes.iter().map(|x| x.nhist.clone()).collect();
        let hist_dim: Vec<IntT> = allgatherv_full_vec(&hist_dim, comm)?;

        // histogram and bin boundaries
        let vbins: Vec<FloatT> = v_nodes
            .iter()
            .flat_map(|x| x.bins.iter().cloned())
            .collect();
        let vbins: Vec<FloatT> = allgatherv_full_vec(&vbins, comm)?;

        let vhist: Vec<FloatT> = v_nodes
            .iter()
            .flat_map(|x| x.hist.iter().cloned())
            .collect();
        let vhist: Vec<FloatT> = allgatherv_full_vec(&vhist, comm)?;

        // Starting positions in the flattened arrays
        let hist_starts: Vec<SizeT> = exc_prefix_sum(
            hist_dim
                .iter()
                .map(|x| SizeT::from_i64(x.to_i64().unwrap()).unwrap()),
            SizeT::one(),
        );
        let bin_starts: Vec<SizeT> = exc_prefix_sum(
            bin_dim
                .iter()
                .map(|x| SizeT::from_i64(x.to_i64().unwrap()).unwrap()),
            SizeT::one(),
        );
        let si_start: Vec<i64> = exc_prefix_sum(
            hist_dim.iter().map(|x| x.to_i64().unwrap()),
            bin_dim.len() as i64,
        );
        let si_start: Vec<SizeT> = si_start
            .into_iter()
            .map(|x| SizeT::from_i64(x).unwrap())
            .collect();

        let nsi = hist_dim
            .iter()
            .map(|x| x.to_usize().unwrap())
            .sum::<usize>()
            * hist_dim.len();

        Ok(Self {
            bin_dim: Array1::from_vec(bin_dim),
            hist_dim: Array1::from_vec(hist_dim),
            hist_start: Array1::from_vec(hist_starts),
            bin_start: Array1::from_vec(bin_starts),
            si_start: Array1::from_vec(si_start),
            abins: Array1::from_vec(vbins),
            ahist: Array1::from_vec(vhist),
            nsi: SizeT::from_usize(nsi).unwrap(),
        })
    }

    /// Load a previously saved-on-disk [`NodeCollection`] from an HDF5
    /// file.
    ///
    /// Reads the dimension / offset / payload datasets from the
    /// `data` group (`hist_start`, `bins_start`, `si_start`,
    /// `hist_dim`, `bins_dim`, `hist`, `bins`) and the scalar
    /// attributes (`nobs`, `nvars`, `nsi`).
    ///
    /// When `nvars` is in `1..nvars_in_file`, the dimension and
    /// offset arrays are truncated to that prefix so callers can
    /// operate on a sub-range of the persisted variables. Otherwise
    /// the full arrays are returned.
    pub fn from_h5(h5_file: &str, nvars: usize) -> Result<Self>
    where
        SizeT: H5Type + ToPrimitive,
        IntT: H5Type,
        FloatT: H5Type,
    {
        let file = hdf5::File::open(h5_file)?;
        let data_g = file.group("data")?;
        // attributes
        let (_nobs, _nvars, nsi) = map_with_result_to_tuple![
            |x| io::read_scalar_attr::<SizeT>(&data_g, x) ;
            "nobs", "nvars", "nsi"
        ];

        let (hist_start, bin_start, si_start) = map_with_result_to_tuple![
            |x| data_g.dataset(x)?.read_1d::<SizeT>();
           "hist_start", "bins_start", "si_start"
        ];

        let (hist_dim, bin_dim) = map_with_result_to_tuple![
            |x| data_g.dataset(x)?.read_1d::<IntT>();
           "hist_dim", "bins_dim"
        ];

        let (ahist, abins) = map_with_result_to_tuple![
            |x| data_g.dataset(x)?.read_1d::<FloatT>();
           "hist", "bins"
        ];

        // Restrict nvars
        let (hist_start, bin_start, si_start, hist_dim, bin_dim) =
            if nvars > 1 && nvars < _nvars.to_usize().unwrap() {
                (
                    hist_start.slice_move(ndarray::s![..nvars]),
                    bin_start.slice_move(ndarray::s![..nvars]),
                    si_start.slice_move(ndarray::s![..nvars]),
                    hist_dim.slice_move(ndarray::s![..nvars]),
                    bin_dim.slice_move(ndarray::s![..nvars]),
                )
            } else {
                (hist_start, bin_start, si_start, hist_dim, bin_dim)
            };

        Ok(Self {
            hist_dim,
            hist_start,
            bin_dim,
            bin_start,
            si_start,
            nsi,
            ahist,
            abins,
        })
    }

    /// Borrow the histogram counts for variable `idx`, sliced out of
    /// the flattened [`Self::ahist`] using
    /// [`Self::hist_start`] / [`Self::hist_dim`].
    pub fn hist(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        SizeT: ToPrimitive,
        IntT: ToPrimitive,
    {
        let hstart = self.hist_start[idx].to_usize().unwrap();
        let hsize = self.hist_dim[idx].to_usize().unwrap();
        let hend = hstart + hsize;
        self.ahist.slice(ndarray::s![hstart..hend])
    }

    /// Borrow the bin edges for variable `idx`, sliced out of the
    /// flattened [`Self::abins`] using
    /// [`Self::bin_start`] / [`Self::bin_dim`].
    pub fn bins(&self, idx: usize) -> ArrayView1<'_, FloatT>
    where
        SizeT: ToPrimitive,
        IntT: ToPrimitive,
    {
        let bstart = self.bin_start[idx].to_usize().unwrap();
        let bsize = self.bin_dim[idx].to_usize().unwrap();
        let bend = bstart + bsize;
        self.abins.slice(ndarray::s![bstart..bend])
    }

    /// Borrow the per-variable bin-dimension array.
    pub fn bin_dim_ref(&self) -> &Array1<IntT> {
        &self.bin_dim
    }

    /// Number of variables represented (length of [`Self::hist_dim`]).
    pub fn len(&self) -> usize {
        self.hist_dim.len()
    }

    /// `true` when no variables are present.
    pub fn is_empty(&self) -> bool {
        self.hist_dim.is_empty()
    }
}

/// Mutual-information record for a single pair of variables.
///
/// Carries the linearised pair index, the optional `(x, y)` pair, the
/// optional joint 2-D histogram, and the scalar MI value. Built one
/// per pair by the helper kernels in [`crate::pucn::helpers`] before
/// being flattened into a [`PairMICollection`].
pub(super) struct PairMI<IntT, FloatT> {
    /// Flat triangular pair index (see
    /// [`crate::util::triu_index_to_pair`]).
    pub index: usize,
    /// Optional explicit `(x, y)` pair when carrying the indices is
    /// useful (e.g. for sampled workflows).
    pub pair: Option<(IntT, IntT)>,
    /// Optional joint 2-D histogram for this pair; 
    /// None when only the MI value is needed.
    pub xy_tab: Option<Array2<FloatT>>,
    /// Mutual-information value.
    pub mi: FloatT,
}

/// Communicator-wide flattened collection of [`PairMI`] records.
///
/// Stores the pair indices, optional dimension pairs, the optional
/// concatenated joint-histogram payload, and the MI values as
/// parallel arrays. Built either by gathering local [`PairMI`]
/// vectors ([`Self::from_vec`]) or by reading from disk
/// ([`Self::from_h5`]); [`Self::distribute`] reshuffles the contents
/// across MPI ranks.
pub(super) struct PairMICollection<IntT, FloatT> {
    /// Flat pair indices (length = number of pairs on this rank). 
    /// (pair mapped to index by [`crate::util::triu_index_to_pair`]).
    pub index: Vec<usize>,
    /// Optional pair of `(x_dim, y_dim)` arrays, parallel to
    /// [`Self::index`], describing the shape of each joint histogram.
    /// None when only the MI value is needed.
    pub dims: Option<(Vec<IntT>, Vec<IntT>)>,
    /// Optional concatenated joint-histogram payload; the slice for
    /// pair `i` has length `dims.0[i] * dims.1[i]`.
    /// None when only the MI value is needed.
    pub xy_tab: Option<Array1<FloatT>>,
    /// Mutual-information values corresponding to [`Self::index`].
    pub mi: Array1<FloatT>,
}

impl<IntT, FloatT> PairMICollection<IntT, FloatT>
where
    IntT: Clone + Default + Debug + Equivalence + FromToPrimitive + Zero,
    FloatT: Clone + Default + Debug + H5Type + Equivalence + Zero,
{
    /// Parallel-IO load of a [`PairMICollection`] from an HDF5 file.
    ///
    /// Distributes the `args.npairs` pair indices across ranks with
    /// an [`InterleavedDist`], block-reads the matching slice of
    /// `data/mi`, derives the per-pair joint-histogram dimensions
    /// from `hist_dim` via [`triu_index_to_pair`], and finally
    /// block-reads the concatenated `data/pair_hist` payload using an
    /// [`ArbitDist`] sized by per-rank sum of dim(x) * dim(y), 
    /// where x,y are the MI pairs.
    pub fn from_h5(
        cx: &CommIfx,
        args: &WorkflowArgs,
        h5f: &str,
        hist_dim: &[IntT],
    ) -> Result<Self> {
        assert!(args.npairs > 0);
        let mi_dist = InterleavedDist::new(args.npairs, cx.size, cx.rank);
        let mi = mpio::block_read1d(cx, h5f, "data/mi", Some(&mi_dist))?;
        let index = mi_dist.range().collect();
        let dims: (Vec<IntT>, Vec<IntT>) = mi_dist
            .range()
            .map(|x| {
                let (i, j): (usize, usize) = triu_index_to_pair(args.nvars, x);
                (hist_dim[i].clone(), hist_dim[j].clone())
            })
            .collect();
        let size: usize = zip(dims.0.iter(), dims.1.iter())
            .map(|(x, y)| (*x).to_usize().unwrap() * (*y).to_usize().unwrap())
            .sum();
        let sizes: Vec<usize> = allgather_one(&size, cx.comm())?;
        let h_dist =
            ArbitDist::new(sizes.iter().sum::<usize>(), cx.size, cx.rank, sizes);
        let xy_tab =
            mpio::block_read1d(cx, h5f, "data/pair_hist", Some(&h_dist))?;
        Ok(Self {
            index,
            dims: Some(dims),
            xy_tab: Some(xy_tab),
            mi,
        })
    }

    /// Flatten a slice of [`PairMI`] records into a [`PairMICollection`].
    ///
    /// If every [`PairMI`] record carries an `xy_tab`, the joint histograms 
    /// are concatenated and the corresponding `(x_dim, y_dim)` arrays are
    /// stored. If any record is missing its `xy_tab`, the resulting
    /// collection drops both `xy_tab` and `dims` (set to `None`).
    pub fn from_vec(vdata: &[PairMI<IntT, FloatT>]) -> Self {
        let index = vdata.iter().map(|x| x.index).collect();
        let mi = vdata.iter().map(|x| x.mi.clone()).collect();
        let tab_flag = itertools::all(vdata.iter(), |x| x.xy_tab.is_some());
        if !tab_flag {
            return Self {
                index,
                mi,
                xy_tab: None,
                dims: None,
            };
        }
        let dims = vdata
            .iter()
            .map(|x| {
                if let Some(xyt) = x.xy_tab.as_ref() {
                    let d = xyt.shape();
                    (
                        IntT::from_usize(d[0]).unwrap(),
                        IntT::from_usize(d[1]).unwrap(),
                    )
                } else {
                    (IntT::zero(), IntT::zero())
                }
            })
            .collect::<(Vec<_>, Vec<_>)>();
        let xy_tab = vdata
            .iter()
            .flat_map(|x| {
                if let Some(xyt) = x.xy_tab.as_ref() {
                    xyt.flatten().to_owned()
                } else {
                    Array1::zeros(1)
                }
            })
            .collect::<Array1<_>>();
        Self {
            index,
            mi,
            dims: Some(dims),
            xy_tab: Some(xy_tab),
        }
    }

    /// Re-shuffle a [`PairMICollection`] across MPI ranks so that
    /// each pair index lands on its block-distributed owner.
    ///
    /// Computes the per-destination send counts from
    /// [`block_owner`] (using the global pair count obtained via
    /// [`allreduce_sum`]), then issues paired
    /// [`all2all_vec`] / [`all2allv_vec`] exchanges for the index,
    /// MI, dimension and joint-histogram payloads. Section timings
    /// are reported through [`SectionTimer`].
    pub fn distribute(&self, mcx: &CommIfx, hist_dim: &[IntT]) -> Result<Self> {
        let s_timer = SectionTimer::from_comm(mcx.comm(), ",");
        let npairs: usize = allreduce_sum(&(self.index.len()), mcx.comm());
        let np = mcx.size as usize;
        let (snd_pairs, snd_tabs) = self
            .index
            .iter()
            //.zip(zip(self.dims.0.iter(), self.dims.1.iter()))
            .fold((vec![0usize; np], vec![0usize; np]), |mut sv, idx| {
                let p_own = block_owner(*idx, mcx.size, npairs) as usize;
                sv.0[p_own] += 1;
                let (ix, iy): (usize, usize) =
                    triu_index_to_pair(hist_dim.len(), *idx);
                let dx: IntT = hist_dim[ix].clone();
                let dy: IntT = hist_dim[iy].clone();
                let rdim = dx.to_usize().unwrap() * dy.to_usize().unwrap();
                sv.1[p_own] += rdim;
                sv
            });
        s_timer.info_section("PairMICollection::distribute::Preparation");
        s_timer.reset();
        let rcv_pairs = all2all_vec(&snd_pairs, mcx.comm())?;
        let rcv_tabs = all2all_vec(&snd_tabs, mcx.comm())?;

        let index =
            all2allv_vec(&self.index[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let mi = all2allv_vec(
            self.mi.as_slice().unwrap(),
            &snd_pairs,
            &rcv_pairs,
            mcx.comm(),
        )?;

        let dims = if let Some(dims) = self.dims.as_ref() {
            Some((
                all2allv_vec(&dims.0[..], &snd_pairs, &rcv_pairs, mcx.comm())?,
                all2allv_vec(&dims.1[..], &snd_pairs, &rcv_pairs, mcx.comm())?,
            ))
        } else {
            None
        };

        let xy_tab = if let Some(xy_tab) = self.xy_tab.as_ref() {
            let xy_tab = all2allv_vec(
                xy_tab.as_slice().unwrap(),
                &snd_tabs,
                &rcv_tabs,
                mcx.comm(),
            )?;
            Some(Array1::from_vec(xy_tab))
        } else {
            None
        };
        s_timer.info_section("PairMICollection::distribute::All2All");

        Ok(Self {
            index,
            dims,
            mi: Array1::from_vec(mi),
            xy_tab,
        })
    }
}

/// Specific-information / LMR record for a single `(about, by)` 
/// ordered variable pair.
///
/// `about` is the variable whose specific information is being
/// described and `by` is the conditioning variable. `si` is optional
/// because some workflows only retain the LMR (Log Marginal
/// Ratios) trace.
pub(super) struct OrdPairSI<IntT, FloatT> {
    /// Index of the variable being described.
    pub about: IntT,
    /// Index of the conditioning variable.
    pub by: IntT,
    /// Optional specific-information vector of length
    /// `hist_dim[about]`.
    pub si: Option<Array1<FloatT>>,
    /// LMR vector of length `hist_dim[about]`.
    pub lmr: Array1<FloatT>,
}

/// Communicator-wide flattened collection of [`OrdPairSI`] records.
///
/// Stores the `(about, by)` pairs together with per-pair sizes and
/// the concatenated `si` / `lmr` payloads. The pair index space is
/// the full `nvars x nvars` ordered grid (`nord_pairs = nvars *
/// nvars`); diagonal entries (pairs with about = by) are filled
/// by [`Self::fill_diag`].
pub(super) struct OrdPairSICollection<IntT, FloatT> {
    /// Number of variables along one axis.
    pub nvars: usize,
    /// Total number of ordered pairs, `nvars * nvars`.
    pub nord_pairs: usize,
    /// `about` index of each pair (length = no. of pairs on this rank).
    pub about: Vec<IntT>,
    /// `by` index of each pair, parallel to [`Self::about`].
    pub by: Vec<IntT>,
    /// Per-pair length of the `si` / `lmr` slice (equal to
    /// `hist_dim[about]`).
    pub sizes: Vec<IntT>,
    /// Optional concatenated specific-information payload; the slice
    /// for pair `i` has length `sizes[i]`.
    pub si: Option<Array1<FloatT>>,
    /// Concatenated LMR payload; the slice for pair `i` has length
    /// `sizes[i]`.
    pub lmr: Array1<FloatT>,
}

impl<IntT, FloatT> OrdPairSICollection<IntT, FloatT>
where
    IntT: Clone + Default + Debug + Equivalence + FromToPrimitive,
    FloatT: Clone + Default + Debug + Equivalence + Zero,
{
    /// Flatten a slice of [`OrdPairSI`] records into an
    /// [`OrdPairSICollection`] over an `nvars x nvars` ordered pair
    /// space.
    ///
    /// Concatenates the per-pair `lmr` vectors into [`Self::lmr`].
    /// If every record carries an `si` payload, the `si` slices are
    /// also concatenated; otherwise [`Self::si`] is `None`.
    pub fn from_vec(
        nvars: usize,
        vdata: &[OrdPairSI<IntT, FloatT>],
    ) -> Self {
        let about: Vec<IntT> = vdata.iter().map(|x| x.about.clone()).collect();
        let by: Vec<IntT> = vdata.iter().map(|x| x.by.clone()).collect();
        let sizes: Vec<IntT> = vdata
            .iter()
            .map(|x| IntT::from_usize(x.lmr.len()))
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let si_flag = itertools::all(vdata.iter(), |x| x.si.is_some());
        let si: Option<Array1<FloatT>> = if si_flag {
            let si = vdata
                .iter()
                .flat_map(|x| {
                    if let Some(si) = x.si.as_ref() {
                        si.to_owned()
                    } else {
                        Array1::zeros(about.len())
                    }
                })
                .collect::<Array1<_>>();
            Some(si)
        } else {
            None
        };
        let lmr = vdata
            .iter()
            .flat_map(|x| x.lmr.to_owned())
            .collect::<Array1<_>>();

        Self {
            nvars,
            nord_pairs: nvars * nvars,
            about,
            by,
            sizes,
            si,
            lmr,
        }
    }

    /// Re-shuffle the collection across MPI ranks so that each
    /// `(about, by)` pair lands on its block-distributed owner of the
    /// flattened grid of (`nvars x nvars`) pairs .
    ///
    /// Computes the per-destination send counts from
    /// [`block_owner`] applied to `about * nvars + by`, then issues
    /// paired [`all2all_vec`] / [`all2allv_vec`] exchanges for the
    /// pair indices, sizes, and the `si` / `lmr` payloads. 
    pub fn distribute(&self, mcx: &CommIfx) -> Result<Self> {
        let s_timer = SectionTimer::from_comm(mcx.comm(), ",");
        let np = mcx.size as usize;
        let (snd_pairs, snd_si) = self
            .sizes
            .iter()
            .zip(zip(self.about.iter(), self.by.iter()))
            .fold(
                (vec![0usize; np], vec![0usize; np]),
                |mut sv, (sz, (a, b))| {
                    let (ua, ub) = (a.to_usize().unwrap(), b.to_usize().unwrap());
                    // NOTE: using number of ordered pairs.
                    let idx = ua * self.nvars + ub;
                    let p_own =
                        block_owner(idx, mcx.size, self.nord_pairs) as usize;
                    sv.0[p_own] += 1;
                    sv.1[p_own] += sz.to_usize().unwrap();
                    sv
                },
            );
        s_timer.info_section("OrdPairSICollection::distribute::Preparation");
        s_timer.reset();
        let rcv_pairs = all2all_vec(&snd_pairs, mcx.comm())?;
        let rcv_si = all2all_vec(&snd_si, mcx.comm())?;

        let about =
            all2allv_vec(&self.about[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let by = all2allv_vec(&self.by[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let sizes =
            all2allv_vec(&self.sizes[..], &snd_pairs, &rcv_pairs, mcx.comm())?;
        let si = if let Some(lsi) = self.si.as_ref() {
            Some(Array1::from_vec(all2allv_vec(
                lsi.as_slice().unwrap(),
                &snd_si,
                &rcv_si,
                mcx.comm(),
            )?))
        } else {
            None
        };
        let lmr = all2allv_vec(
            self.lmr.as_slice().unwrap(),
            &snd_si,
            &rcv_si,
            mcx.comm(),
        )?;
        s_timer.info_section("OrdPairSICollection::distribute::All2All");

        Ok(Self {
            nvars: self.nvars,
            nord_pairs: self.nord_pairs,
            about,
            by,
            sizes,
            si,
            lmr: Array1::from_vec(lmr),
        })
    }

    /// Build a `pair_index -> (slot_index, payload_offset)` lookup
    /// over the records currently held on this rank.
    ///
    /// The returned map keys each `about * nvars + by` to its
    /// position in [`Self::about`] / [`Self::by`] / [`Self::sizes`]
    /// and to the matching offset in [`Self::si`] / [`Self::lmr`].
    /// Used by [`Self::fill_diag`].
    fn pairs_lookup(&self) -> HashMap<usize, (usize, usize)> {
        let mut p_lookup =
            HashMap::<usize, (usize, usize)>::with_capacity(self.about.len());
        // Should this be pairs ?
        let mut offset = 0;
        for (i, (a, b)) in zip(self.about.iter(), self.by.iter()).enumerate() {
            let tgt_index =
                a.to_usize().unwrap() * self.nvars + b.to_usize().unwrap();
            p_lookup.insert(tgt_index, (i, offset));
            offset += self.sizes[i].to_usize().unwrap();
        }
        assert!(self.si.as_ref().is_none_or(|x| offset == x.len()));
        assert!(offset == self.lmr.len());
        p_lookup
    }

    /// Re-key the collection so that this rank owns the contiguous
    /// block-distributed range of the `nvars x nvars` ordered pair
    /// grid, leaving diagonal entries (`about == by`) zeroed.
    ///
    /// Replaces [`Self::about`], [`Self::by`], [`Self::sizes`] and
    /// the `si` / `lmr` payloads with a freshly allocated layout
    /// covering [`block_range`]`(rank, size, nord_pairs)`. Off-diagonal
    /// pairs whose data are present in the previous layout (located via
    /// [`Self::pairs_lookup`]) are copied over; everything else (in
    /// particular every `about == by` slot) is left as the default
    /// value. `hist_dim` provides the per-`about` slice length used
    /// to size each pair.
    pub fn fill_diag(&mut self, hist_dim: &[IntT], mcx: &CommIfx) {
        //  Initialize array with the allocated ordered pairs
        let brg = block_range(mcx.rank, mcx.size, self.nord_pairs);
        let about = brg.clone().map(|idx| idx / self.nvars).collect::<Vec<_>>();
        let by = brg.clone().map(|idx| idx % self.nvars).collect::<Vec<_>>();
        let n_si: usize = about
            .iter()
            .map(|x| hist_dim[x.to_usize().unwrap()].to_usize().unwrap())
            .sum();

        // Pairs Lookup
        let lookup = self.pairs_lookup();
        let si_slice = self.si.as_ref().map(|x| x.as_slice().unwrap());
        let lmr_slice = self.lmr.as_slice().unwrap();

        let mut sizes: Vec<IntT> = vec![IntT::default(); about.len()];
        let mut si: Option<Vec<FloatT>> =
            self.si.as_ref().map(|_| vec![FloatT::default(); n_si]);
        let mut lmr: Vec<FloatT> = vec![FloatT::default(); n_si];
        let mut offset: usize = 0;
        for (i, (x, y)) in zip(about.iter(), by.iter()).enumerate() {
            let h_dim = hist_dim[x.to_usize().unwrap()].clone();
            let x_dim = h_dim.to_usize().unwrap();
            let sir = offset..(offset + x_dim);
            if x != y {
                let op_idx = x * self.nvars + y;
                if let Some(vx) = lookup.get(&op_idx) {
                    let fmr = vx.1..(vx.1 + x_dim);
                    lmr[sir.clone()].clone_from_slice(&lmr_slice[fmr.clone()]);
                    if let (Some(si_slice), Some(si)) = (si_slice, si.as_mut()) {
                        si[sir].clone_from_slice(&si_slice[fmr]);
                    }
                }
            }
            sizes[i] = h_dim;
            offset += x_dim;
        }

        self.about = about
            .into_iter()
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        self.by = by
            .into_iter()
            .map(|x| IntT::from_usize(x).unwrap())
            .collect();
        self.si = si.map(|x| Array1::from_vec(x));
        self.lmr = Array1::from_vec(lmr);
        self.sizes = sizes;
    }
}

// aliases to
/// Result of computing one pair of variables: two [`OrdPairSI`] records
/// (one for each direction of the ordered pair) plus the corresponding
/// [`PairMI`] entry.
pub(super) type NodePair<IntT, FloatT> = (
    OrdPairSI<IntT, FloatT>,
    OrdPairSI<IntT, FloatT>,
    PairMI<IntT, FloatT>,
);

/// Per-batch accumulator of [`NodePair`] results: parallel `Vec`s of
/// the two ordered SI directions and the corresponding MI records.
/// Implements [`BPTrait`] for ergonomic batch construction.
pub(super) type BatchPairs<IntT, FloatT> = (
    Vec<OrdPairSI<IntT, FloatT>>,
    Vec<OrdPairSI<IntT, FloatT>>,
    Vec<PairMI<IntT, FloatT>>,
);

/// Trait implemented by [`BatchPairs`] to expose a uniform
/// `new` / `push` interface for accumulating [`NodePair`] results
/// during a 2-D pair batch.
pub(super) trait BPTrait<IntT, FloatT> {
    /// Pre-allocate a batch sized for the upper-triangular pairs in
    /// the `(rows, cols)` rectangle (`row < col`).
    fn new(rows: Range<usize>, cols: Range<usize>) -> Self;
    /// Append one [`NodePair`] result to the batch.
    fn push(&mut self, node_pair: NodePair<IntT, FloatT>);
}

impl<IntT, FloatT> BPTrait<IntT, FloatT> for BatchPairs<IntT, FloatT> {
    fn new(rows: Range<usize>, cols: Range<usize>) -> Self {
        let npairs: usize = rows
            .map(|row| cols.clone().filter(|col| row < *col).sum::<usize>())
            .sum();
        (
            Vec::with_capacity(npairs),
            Vec::with_capacity(npairs),
            Vec::with_capacity(npairs),
        )
    }

    fn push(&mut self, node_pair: NodePair<IntT, FloatT>) {
        let (s0, s1, m) = node_pair;
        self.0.push(s0);
        self.1.push(s1);
        self.2.push(m);
    }
}

/// Collection-level analogue of [`NodePair`]: an
/// [`OrdPairSICollection`] holding the ordered SI / LMR data paired
/// with a [`PairMICollection`] holding the matching MI / joint
/// histograms.
pub(super) type NodePairCollection<IntT, FloatT> = (
    OrdPairSICollection<IntT, FloatT>,
    PairMICollection<IntT, FloatT>,
);
