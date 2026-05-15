//! Configuration types for the PUC network construction workflows.
//!
//! This module defines the YAML/TOML-deserializable structures that define
//! workflow configuration. Accepted by [`crate::pucn::execute_workflow`].
//!
//! * [`WorkflowArgs`] — the top-level run configuration (input/output
//!   files, sampling parameters, run flags, log level, and the list of
//!   workflow stages to execute).
//! * [`RunMode`] — the stages a network construction workflow 
//!   (PUC computation, MI/SI computation, sample generation, etc.).
//! * [`LogLevel`] — a `serde`-friendly mirror of [`log::LevelFilter`]
//!   used by config files.
//!
//! All `Deserialize` impls accept the lowercase / aliased spellings.
//! Default values for Optional fields are populated from the
//! `default_*` functions defined at the top of this file.

use log::LevelFilter;
use serde::{Deserialize, Serialize};

use crate::types::{DiscretizerMethod, LogBase};

/// Default value for [`WorkflowArgs::gene_id_col`] (`"gene_ids"`).
fn default_gene_id_col() -> String {
    "gene_ids".to_string()
}

/// Default value for [`WorkflowArgs::nroundup`] (`4`); number of decimal
/// places retained when rounding the expression matrix.
fn default_roundup() -> usize {
    4
}

/// Default value for [`WorkflowArgs::nrounds`] (`8`); number of sampling
/// rounds used by the sampled-PUC workflow.
fn default_nrounds() -> usize {
    8
}

/// Default value for [`WorkflowArgs::nsamples`] (`200`); samples drawn
/// per round.
fn default_nsamples() -> usize {
    200
}

/// Default value for [`WorkflowArgs::lmr_only`] (`false`); when true,
/// LMR-only variants of the PUC kernels are used.
fn default_lmr_only() -> bool {
    false
}

/// Logging verbosity loaded from configuration files.
///
/// Each variant is the `usize` representation of the corresponding
/// [`log::LevelFilter`], so the enum can be cast back to a filter for
/// `env_logger`. The python-style spellings (`"NOTSET"`, `"WARNING"`,
/// `"DEBUG"`, ...) are accepted as serde aliases for compatibility with
/// existing YAML configs. 
/// The default is [`LogLevel::Off`] (alias `"NOTSET"`).
// 'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
#[repr(usize)]
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
pub enum LogLevel {
    /// Logging disabled. Mirrors [`LevelFilter::Off`]. Default.
    #[serde(alias = "NOTSET")]
    #[default]
    Off = LevelFilter::Off as usize,
    /// `error!` and above. Mirrors [`LevelFilter::Error`].
    #[serde(alias = "ERROR")]
    Error = LevelFilter::Error as usize,
    /// `warn!` and above. Mirrors [`LevelFilter::Warn`].
    #[serde(alias = "WARNING")]
    Warn = LevelFilter::Warn as usize,
    /// `info!` and above. Mirrors [`LevelFilter::Info`].
    #[serde(alias = "INFO")]
    Info = LevelFilter::Info as usize,
    /// `debug!` and above. Mirrors [`LevelFilter::Debug`].
    #[serde(alias = "DEBUG")]
    Debug = LevelFilter::Debug as usize,
    /// All log records. Mirrors [`LevelFilter::Trace`].
    Trace = LevelFilter::Trace as usize,
}

/// Stages of the PUC-network workflow that can be run. 
/// 
/// Each variant carries a `serde` alias matching the snake-case
/// spelling used in YAML configs.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum RunMode {
    /// Standard MI/SI (Mutual Information / Specific Information) computation. 
    /// YAML alias: `"misi"`.
    #[serde(alias = "misi")]
    MISI,
    /// Compute per-node histograms only. YAML alias: `"hist_nodes"`.
    #[serde(alias = "hist_nodes")]
    HistNodes,
    /// Compute and persist the histogram distribution. 
    /// YAML alias: `"hist_dist"`.
    #[serde(alias = "hist_dist")]
    HistDist,
    /// Use pre-computed node histograms to run the MI/SI workflow
    /// i.e., don't compute the histograms. 
    /// YAML alias: `"hist2misi_dist"`.
    #[serde(alias = "hist2misi_dist")]
    HistNodes2MISI,
    /// Distribute the MI/SI computation across MPI ranks. 
    /// YAML alias: `"misi_dist"`.
    #[serde(alias = "misi_dist")]
    MISIDist,
    /// Compute sampled PUC values for all the pairs. 
    /// YAML alias: `"sampled_puc_pairs"`.
    #[serde(alias = "sampled_puc_pairs")]
    SampledPUCPairs,
    /// Run the parallel sampled-PUC workflow over per-rank range pairs i.e.,
    ///  a 2D block distribution. 
    /// YAML alias: `"samples_ranges"`.
    #[serde(alias = "samples_ranges")]
    SamplesRanges,
    /// Run the sampled-PUC workflow on user-supplied input sample of data.
    /// YAML alias: `"samples_input"`.
    #[serde(alias = "samples_input")]
    SamplesInput,
    /// LMR variant of [`Self::SamplesRanges`]. 
    /// YAML alias: `"samples_lmr_ranges"`.
    #[serde(alias = "samples_lmr_ranges")]
    SamplesLMRRanges,
    /// LMR variant of [`Self::SamplesInput`]. 
    /// YAML alias: `"samples_lmr_input"`.
    #[serde(alias = "samples_lmr_input")]
    SamplesLMRInput,
    /// PUC over the per-rank pair range pairs. YAML alias: `"puc_ranges"`.
    #[serde(alias = "puc_ranges")]
    PUCRanges,
    /// LMR variant of the standard PUC workflow. YAML alias: `"puc_lmr"`.
    #[serde(alias = "puc_lmr")]
    PUCLMR,
    /// Distributed LMR-PUC workflow. YAML alias: `"puc_lmr_dist"`.
    #[serde(alias = "puc_lmr_dist")]
    PUCLMRDist,
    /// Convert PUC scores to PIDC scores. YAML alias: `"puc2pidc"`.
    #[serde(alias = "puc2pidc")]
    PUC2PIDC,
    /// Union of multiple PUC sub-networks. YAML alias: `"puc_union"`.
    #[serde(alias = "puc_union")]
    PUCUnion,
    /// Union of sub-networks generated for each cluster. 
    /// YAML alias: `"cluster_union"`.
    #[serde(alias = "cluster_union")]
    ClusterUnion,
    /// LMR variant of [`Self::ClusterUnion`]. 
    /// YAML alias: `"cluster_lmr_union"`.
    #[serde(alias = "cluster_lmr_union")]
    ClusterLMRUnion,
}

/// Top-level configuration for a `pucn` workflow run.
///
/// Constructed by deserialising a YAML/TOML config file (see the
/// `config/pucn/*.yml` examples). The struct mixes mandatory file
/// paths with a long tail of optional, defaulted parameters that
/// control sampling, logging, and which intermediate artefacts get
/// persisted. 
/// The list of stages to execute is given by [`Self::mode`]
/// and is consumed by [`crate::pucn::execute_workflow`].
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowArgs {
    // Mandatory Fileds
    //  - Files/Paths
    /// Path to the input AnnData (`.h5ad`) expression file.
    pub h5ad_file: String,
    /// Path to the cached MISI data file (HDF5).
    pub misi_data_file: String,
    /// Path to the PUC scores file written by the PUC computations (HDF5).
    pub puc_file: String,
    /// Path to the PIDC scores file (input to / output of the PUC↔PIDC
    /// conversion stages).
    pub pidc_file: String,
    /// Path to the histogram data file. Defaults to the empty string
    /// when not supplied; 
    /// `HistNodes2MISI` and `MISIDist` use this .
    #[serde(default)]
    pub hist_data_file: String,

    //  - Run Modes
    /// Ordered list of workflow stages to execute. See [`RunMode`].
    /// The dispatcher in [`crate::pucn::execute_workflow`] runs each one 
    /// in the provided order. 
    pub mode: Vec<RunMode>, //

    // Optional Fields with defaults
    /// Optional path to a pre-computed nodes pickle.
    /// (Not used. only for backward compatibility). 
    pub nodes_pickle: Option<String>,
    /// Optional path to a pre-computed node-pairs pickle.
    /// (Not used. only for backward compatibility). 
    pub nodes_pairs_pickle: Option<String>,
    /// Optional path to a samples file used by the
    /// `samples_input` / `samples_lmr_input` modes.
    pub samples_file: Option<String>,
    /// Optional list of sub-network files used by the union stages.
    pub sub_net_files: Option<Vec<String>>,
    /// Working directory under which intermediate artefacts are
    /// written. When `None`, paths are interpreted as-is.
    pub wflow_dir: Option<String>,
    /// Optional sibling row-major HDF5 file (see
    /// [`crate::anndata::AnnData`]) used to accelerate column reads.
    pub row_major_h5_file: Option<String>,

    /// Name of the `var/<column>` dataset holding gene identifiers.
    /// Defaults to `"gene_ids"` (see [`default_gene_id_col`]).
    #[serde(default = "default_gene_id_col")]
    pub gene_id_col: String, // = String::from_str("gene_ids"),

    // Flags
    /// Persist computed nodes to disk. Defaults to `false`.
    #[serde(default)]
    pub save_nodes: bool, // = False
    /// Persist computed node pairs to disk. Defaults to `false`.
    #[serde(default)]
    pub save_node_pairs: bool, // = False
    /// Enable per-stage timing instrumentation. Defaults to `false`
    /// (not used, for now)
    #[serde(default)]
    pub enable_timers: bool, // = True

    /// Decimal places used for rounding the expression matrix at load time. 
    /// Default `4` (see [`default_roundup`]).
    #[serde(default = "default_roundup")]
    pub nroundup: usize, // = 4
    /// Number of sampling rounds for the sampled-PUC workflow.
    /// Default `8` (see [`default_nrounds`]).
    #[serde(default = "default_nrounds")]
    pub nrounds: usize, // = 8
    /// Samples drawn per round. Default `200` (see [`default_nsamples`]).
    #[serde(default = "default_nsamples")]
    pub nsamples: usize, // = 200
    /// Number of observations (rows of `X`); `0` means "infer from the
    /// AnnData file" via [`Self::update_dims`].
    #[serde(default)]
    pub nobs: usize, // = 0
    /// Number of variables / genes (columns of `X`); `0` means "infer".
    #[serde(default)]
    pub nvars: usize, // = 0
    /// Number of unordered variable pairs `nvars * (nvars - 1) / 2`;
    /// `0` means "compute from `nvars`" via [`Self::update_dims`].
    #[serde(default)]
    pub npairs: usize, // = 0

    /// Optional quantile cut-off used by some downstream filtering passes.
    /// `None` disables the filter. (NOT used currently)
    #[serde(default)]
    pub quantile_filter: Option<f32>,

    /// Discretization strategy applied prior to histogramming. See
    /// [`DiscretizerMethod`] (default [`DiscretizerMethod::BayesianBlocks`]).
    #[serde(default)]
    pub disc_method: DiscretizerMethod, // = 'bayesian_blocks',
    /// Logarithm base used by the entropy / MI kernels. 
    /// See [`LogBase`] (default [`LogBase::Two`]).
    #[serde(default)]
    pub tbase: LogBase, // = '2'

    /// Logging verbosity. Default [`LogLevel::Off`] (config alias `"NOTSET"`).
    #[serde(default)]
    pub log_level: LogLevel, // = 'DEBUG'
 
    /// When `true`, keep only the LMR, and not SI, when computing PUC.
    /// Default `false` (see [`default_lmr_only`]).
    #[serde(default = "default_lmr_only")]
    pub lmr_only: bool,
}

impl WorkflowArgs {
    /// Fill in any "auto" dimension fields from a `[nobs, nvars]`
    /// shape pair.
    ///
    /// A field is left untouched once it has been set to a non-zero
    /// value. After this call:
    ///
    /// * [`Self::nobs`] is `dims[0]` if it was previously `0`.
    /// * [`Self::nvars`] is `dims[1]` if it was previously `0`.
    /// * [`Self::npairs`] is `nvars * (nvars - 1) / 2` if it was
    ///   previously `0`.
    ///
    /// Typically called once after opening the AnnData file so the
    /// rest of the pipeline can rely on these counts.
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
}
