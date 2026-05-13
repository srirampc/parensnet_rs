//! Multi-variable information measures and the data structures that feed
//! the network-construction kernels.
//!
//! The submodules form a small pipeline:
//!
//! * [`imeasures`] — pure functions over discrete (binned) joint
//!   distributions: mutual information `I(X;Y)`, specific information
//!   `SI`, Williams–Beer redundancy and the per-axis log-marginal-ratio
//!   (LMR) projections. These operate on the histograms produced by
//!   [`crate::hist`] and are the lowest layer of the module.
//! * [`rv`] — the [`rv::MRVTrait`] abstraction over a multi-variable
//!   distribution, plus the sorted/prefix-sum/rank LMR tables
//!   ([`rv::UnitLMRSA`], [`rv::LMRSA`], [`rv::LMRDataStructure`],
//!   [`rv::LMRSubsetDataStructure`]) that turn the redundancy / PUC
//!   scoring of Chan et al., 2017 into O(dim) per-edge queries.
//! * [`misi`] — HDF5-backed materializations of MI / SI / LMR data:
//!   [`misi::MISIRangePair`] for a cartesian product of two index ranges
//!   and [`misi::MISIPair`] for a single edge. Both implement
//!   [`rv::MRVTrait`] and cache the fast LMR lookup tables, so the
//!   parallel network kernels in [`crate::pucn`] can compute edges' PUC
//!   scores directly from precomputed files.

pub mod imeasures;
pub mod misi;
pub mod rv;
