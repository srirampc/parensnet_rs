//! B-spline based mutual information utilities.
//!
//! This module provides an alternative to the histogram-based MI estimator
//! in [`crate::mvim::imeasures`]. Instead of binning samples into hard
//! histogram cells, samples are softly assigned to bins via a B-spline
//! basis (Daub et al., 2004), giving a smoother estimator that is less
//! sensitive to bin boundaries on small samples.
//!
//! The single submodule [`mi`] exposes:
//!
//! * [`mi::bspline_weights`] / [`mi::bspline_mi`] — thin wrappers around
//!   the C kernels in the bundled `mcpnet_rs` crate (`bspline_weights_f32`
//!   and `bspline_mi_kernel_f32`), used as the production fast path.
//! * [`mi::BSplineWeights`] — a pure-Rust reference implementation that
//!   builds the knot vector and B-spline weights, then computes the 1D
//!   histogram and Shannon entropy used to assemble MI.
pub mod mi;
