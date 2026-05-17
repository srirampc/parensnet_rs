//! Configuration types for the MCPNet  MI workflows.
//!
//! Defines the YAML/TOML-deserializable structures consumed by
//! [`crate::mcpn::execute_workflow`]:
//!
//! * [`RunMode`] — which stage of the workflow to execute.
//! * [`WorkflowArgs`] — paths, B-spline parameters, and the
//!   per-rank pair-distribution dimensions.
//!
//! All `Deserialize` impls accept the snake-case spellings used in
//! the existing YAML configs via serde aliases.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Default for [`WorkflowArgs::gene_id_col`] (`"gene_ids"`).
fn default_gene_id_col() -> String {
    "gene_ids".to_string()
}

/// Default for [`WorkflowArgs::nroundup`] (`4`).
fn default_roundup() -> usize {
    4
}

/// Default for [`WorkflowArgs::nrounds`] (`8`).
fn default_nrounds() -> usize {
    8
}

/// Default for [`WorkflowArgs::nbins`] (`8`).
fn default_nbins() -> usize {
    8
}

/// Default for [`WorkflowArgs::spline_order`] (`3`, cubic B-splines).
fn default_spline_order() -> usize {
    3
}

/// Default for [`WorkflowArgs::weights_ds`] (`"weights"`).
fn default_weights_attr() -> String {
    String::from_str("weights").unwrap_or_default()
}


/// Stage of the MCPNet workflow to execute.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum RunMode {
    /// Compute the B-spline weight matrix for every variable and
    /// write it to [`WorkflowArgs::weights_file`]. YAML alias:
    /// `"mi_bspline_weights"`.
    #[serde(alias = "mi_bspline_weights")]
    MIBSplineWeights,
    /// Compute pairwise B-spline mutual information from the
    /// previously persisted weight matrix and write the
    /// `(index, mi)` pairs to [`WorkflowArgs::mi_file`]. YAML alias:
    /// `"mi_bspline"`.
    #[serde(alias = "mi_bspline")]
    MIBSpline,
}

/// Top-level configuration for an MCPNet workflow run.
///
/// Constructed by deserialising a YAML/TOML config file. Mandatory
/// fields are the input AnnData file, the output HDF5 paths, and
/// the ordered list of stages in [`Self::mode`]. The remaining
/// fields are optional with the defaults defined by the per-field
/// `default_*` helpers in this module.
///
/// The dimension fields ([`Self::nobs`], [`Self::nvars`],
/// [`Self::npairs`]) are zero by default and are expected to be filled in 
/// at startup.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowArgs {
    /// Path to the input AnnData (`.h5ad`) expression file.
    pub h5ad_file: String,
    /// HDF5 file for the pairwise mutual-information results.
    pub mi_file: String,
    /// HDF5 file for the B-spline weight matrix.
    pub weights_file: String,

    //  - Run Modes
    /// Ordered list of stages to execute.
    pub mode: Vec<RunMode>,
    /// Optional row-major mirror of the AnnData expression matrix
    /// (argument to [`crate::anndata::AnnData::new`]).
    pub row_major_h5_file: Option<String>,

    /// Name of the `var/<column>` dataset holding gene identifiers
    /// in the AnnData file. Defaults to `"gene_ids"`.
    #[serde(default = "default_gene_id_col")]
    pub gene_id_col: String,

    /// Decimal places used when rounding the expression matrix.
    /// Default `4`.
    #[serde(default = "default_roundup")]
    pub nroundup: usize,
    /// Number of sub-rounds the pair-loop is split into. Default: `8` .
    #[serde(default = "default_nrounds")]
    pub nrounds: usize,
    /// Number of observations (rows of the expression matrix). 
    /// Set to `0` in configs and filled in by [`Self::update_dims`].
    #[serde(default)]
    pub nobs: usize,
    /// Number of variables (columns of the expression matrix). 
    /// Set to `0` in configs and filled in by [`Self::update_dims`].
    #[serde(default)]
    pub nvars: usize,
    /// Number of upper-triangular variable pairs (`nvars*(nvars-1)/2`).
    /// Set to `0` in configs and filled in by [`Self::update_dims`].
    #[serde(default)]
    pub npairs: usize,

    /// Number of bins used by the B-spline histogram. Default `8`
    /// (see [`default_nbins`]).
    #[serde(default = "default_nbins")]
    pub nbins: usize,

    /// Order of the B-spline basis (`3` = cubic). Default `3` .
    #[serde(default = "default_spline_order")]
    pub spline_order: usize,

    /// Name of the weights dataset inside the HDF5 file.
    #[serde(default = "default_weights_attr")]
    pub weights_ds: String,
}

impl WorkflowArgs {
    /// Fill in [`Self::nobs`], [`Self::nvars`], and [`Self::npairs`]
    /// from the AnnData matrix dimensions `[nobs, nvars]`.
    /// [`Self::npairs`] is computed as `nvars*(nvars-1)/2`
    /// (upper-triangular pair count).
    pub fn update_dims(&mut self, dims: &[usize]) {
        if self.nobs == 0 {
            self.nobs = dims[0];
        }
        if self.nvars == 0 {
            self.nvars = dims[1];
        }
        if self.npairs == 0 {
            self.npairs = (self.nvars * (self.nvars - 1)) / 2;
        }
    }

    /// Per-variable B-spline weight vector length: `nbins * nobs + 1`.
    /// Used as the column count of the weight matrix stored in HDF5 and 
    /// as the read range when loading a batch of weight rows.
    pub fn weights_dim(&self) -> usize {
        self.nbins * self.nobs + 1
    }
}
