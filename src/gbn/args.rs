//! Configuration types for the gradient-boosted GRN workflow.
//!
//! This module defines the YAML/TOML-deserializable structures
//! configuring network construction and CV runs:
//!
//! * [`GBGRNArgs`] — top-level run configuration (input/output
//!   paths, mode, CV/GBM parameters, log level).
//! * [`GBMParams`] — LightGBM hyper-parameters; serializable to the
//!   JSON shape expected by [`lightgbm3::Booster`] via
//!   [`GBMParams::as_json`] / [`GBMParams::as_json_with_seed`].
//! * [`CVConfig`] — k-fold cross-validation parameters.
//! * [`RunMode`] — workflow stage selector (CV-only or full GRN
//!   inference).
//! * [`LogLevel`] — `serde`-friendly mirror of
//!   [`log::LevelFilter`] used by config files.
//!
//! All `Deserialize` impls accept the lowercase / aliased spellings
//! used in the existing YAML configs.

use log::LevelFilter;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Logging verbosity loaded from configuration files.
///
/// Each variant is the `usize` representation of the corresponding
/// [`log::LevelFilter`], so the enum can be cast back to a filter
/// for `env_logger`. python-style spellings (`"NOTSET"`,
/// `"WARNING"`, `"DEBUG"`, ...) are accepted as serde aliases for
/// compatibility with existing YAML configs. The default is
/// [`LogLevel::Off`] (alias `"NOTSET"`).
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

/// Stage of the GBN workflow to execute.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum RunMode {
    /// Run only the k-fold cross-validation step
    /// ([`crate::gbn::run_cross_fold_gbm`]) to estimate a good
    /// `num_iterations`. YAML alias: `"cv_gb"`.
    #[serde(alias = "cv_gb")]
    GBCrossFoldValidation,
    /// Run the full distributed GRN inference pipeline
    /// ([`crate::gbn::infer_gb_network`]). YAML alias: `"gb_grn"`.
    #[serde(alias = "gb_grn")]
    GBGRNet,
}

/// LightGBM hyper-parameters used by the GBN workflows.
///
/// This struct is the `serde`-friendly mirror of the JSON parameter
/// object that [`lightgbm3::Booster`] expects. It is materialized
/// either via [`GBMParams::as_json`] (deterministic seed left to the
/// caller) or [`GBMParams::as_json_with_seed`] (fixed seed `72`).
///
/// Defaults match the values used in the reference Arboreto /
/// GRNBoost configuration; see the per-field `default_*` functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBMParams {
    /// LightGBM `verbose` level (`-1` silent, `0` warn, `1` info,
    /// `2` debug). Default `1` (see [`GBMParams::default_verbose`]).
    #[serde(default = "GBMParams::default_verbose")]
    pub verbose: i32,
    /// Number of OpenMP threads used by LightGBM. `0` lets the
    /// library choose. Default `0` (see [`GBMParams::default_threads`]).
    #[serde(default = "GBMParams::default_threads")]
    pub num_threads: i32,
    /// Maximum number of boosting rounds. Default `300` (see
    /// [`GBMParams::default_iterations`]).
    #[serde(default = "GBMParams::default_iterations")]
    pub num_iterations: usize,
    /// Patience (in rounds) for the early-stopping callback when
    /// training with a validation set. Default `10` (see
    /// [`GBMParams::default_early_stopping_rounds`]).
    #[serde(default = "GBMParams::default_early_stopping_rounds")]
    pub early_stopping_rounds: usize,
    /// Fraction of the training rows sampled for each boosting
    /// iteration (LightGBM `bagging_fraction`). Default `0.9` (see
    /// [`GBMParams::default_bagging_fraction`]).
    #[serde(default = "GBMParams::default_bagging_fraction")]
    pub bagging_fraction: f32,
    /// Bagging frequency (number of iterations between resamples).
    /// Default `1` (see [`GBMParams::default_bagging_freq`]).
    #[serde(default = "GBMParams::default_bagging_freq")]
    pub bagging_freq: i32,
    /// Validation metric reported by LightGBM (used by the
    /// early-stopping callback). Default `"rmse"` (see
    /// [`GBMParams::default_metric`]).
    #[serde(default = "GBMParams::default_metric")]
    pub metric: String,
    /// Fraction of features sampled for each tree (LightGBM
    /// `feature_fraction`). Default `0.1` (see
    /// [`GBMParams::default_feature_fraction`]).
    #[serde(default = "GBMParams::default_feature_fraction")]
    pub feature_fraction: f32,
}

impl GBMParams {
    /// Default for [`Self::verbose`] (`1`).
    fn default_verbose() -> i32 {
        1
    }

    /// Default for [`Self::num_threads`] (`0`, let LightGBM choose).
    fn default_threads() -> i32 {
        0
    }

    /// Default for [`Self::num_iterations`] (`300`).
    fn default_iterations() -> usize {
        300
    }

    /// Default for [`Self::early_stopping_rounds`] (`10`).
    fn default_early_stopping_rounds() -> usize {
        10
    }

    /// Default for [`Self::bagging_fraction`] (`0.9`).
    fn default_bagging_fraction() -> f32 {
        0.9
    }

    /// Default for [`Self::bagging_freq`] (`1`).
    fn default_bagging_freq() -> i32 {
        1
    }

    /// Default for [`Self::feature_fraction`] (`0.1`).
    fn default_feature_fraction() -> f32 {
        0.1
    }

    /// Default for [`Self::metric`] (`"rmse"`).
    fn default_metric() -> String {
        "rmse".to_string()
    }
}

impl Default for GBMParams {
    fn default() -> Self {
        GBMParams {
            verbose: Self::default_verbose(),
            num_threads: Self::default_threads(),
            num_iterations: Self::default_iterations(),
            early_stopping_rounds: Self::default_early_stopping_rounds(),
            bagging_fraction: Self::default_bagging_fraction(),
            bagging_freq: Self::default_bagging_freq(),
            metric: Self::default_metric(),
            feature_fraction: Self::default_feature_fraction(),
        }
    }
}

impl GBMParams {
    /// Serialize the parameters to the JSON object expected by
    /// [`lightgbm3::Booster`].
    ///
    /// Sets `objective` to `"regression"` and copies every field
    /// of [`GBMParams`] one-to-one.
    pub fn as_json(&self) -> Value {
        serde_json::json! {{
            "objective": "regression",
            "verbose": self.verbose,
            "num_threads": self.num_threads,
            "num_iterations": self.num_iterations,
            "early_stopping_rounds": self.early_stopping_rounds,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "feature_fraction": self.feature_fraction,
            "metric": self.metric,
        }}
    }

    /// Variant of [`Self::as_json`] that also emits the LightGBM
    /// random-state fields (`seed`, `bagging_seed`,
    /// `feature_fraction_seed`) all set to `72` so that the CV
    /// stage produces identical splits across MPI ranks.
    pub fn as_json_with_seed(&self) -> Value {
        let mut g_parms = self.as_json();
        let seed_params = json! {{
            "seed": 72,
            "bagging_seed": 72,
            "feature_fraction_seed": 72,
        }};
        if let (Value::Object(gpx), Value::Object(spy)) =
            (&mut g_parms, seed_params)
        {
            for (k, v) in spy {
                gpx.entry(k).or_insert(v);
            }
        }
        g_parms
    }
}

/// Configuration for the k-fold cross-validation step that
/// estimates a good `num_iterations` for the production GBN run.
///
/// The CV loop trains one LightGBM model per `(sampled gene, fold)`
/// pair using [`crate::gbn::train_with_early_stopping`], records
/// the early-stopped iteration count, and finally aggregates the
/// counts in a [`crate::gbn::CVStats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVConfig {
    /// Number of CV folds (typically 5).
    pub n_folds: usize,
    /// Number of target genes randomly sampled for the CV pass
    /// (typical range 100-1000).
    pub n_sample_genes: usize,
    /// Upper bound on boosting rounds tried per fold (e.g. `500`).
    pub max_rounds: usize,
    /// Patience (rounds without improvement) before LightGBM's
    /// early-stopping callback fires (e.g. `10`).
    pub early_stopping_rounds: usize,
    // min_rounds: usize,          // TODO:Minimum rounds to train (e.g., 20)
    /// Minimum boosting rounds before the early-stopping callback
    /// is allowed to trigger.
    pub min_iterations: usize,
    /// Maximum number of consecutive rounds that may produce no
    /// improvement of the validation metric before stopping.
    pub max_empty_rounds: usize,
    /// Absolute tolerance applied to the metric improvement
    /// (`abs(new - best) < score_tolerance` counts as no
    /// improvement).
    pub score_tolerance: f64,
    /// Base [`GBMParams`] applied to every CV booster (the CV loop
    /// overrides `early_stopping_rounds` and `num_iterations` from
    /// the corresponding fields above).
    pub params: GBMParams,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            n_sample_genes: 200, // Sample 200 genes for CV
            max_rounds: 500,
            early_stopping_rounds: 10,
            min_iterations: 32,
            max_empty_rounds: 5,
            score_tolerance: 1e-4,
            params: GBMParams::default(),
        }
    }
}

impl CVConfig {
    /// Build the JSON object passed to LightGBM's early-stopping
    /// callback. Combines [`Self::min_iterations`],
    /// [`Self::max_empty_rounds`], and [`Self::score_tolerance`].
    pub fn es_params(&self) -> serde_json::Value {
        serde_json::json! {{
            "min_iterations": self.min_iterations,
            "max_empty_rounds": self.max_empty_rounds,
            "score_tolerance": self.score_tolerance,
        }}
    }
}

/// Default for [`GBGRNArgs::gene_id_col`] (`"_index"`).
fn default_gene_id_col() -> String {
    "_index".to_string()
}

/// Default for [`GBGRNArgs::nroundup`] (`4`); decimal places retained
/// when rounding the expression matrix.
fn default_roundup() -> usize {
    4
}

/// Default for [`GBGRNArgs::importance_filter`] (`Some(0.0)`); keep
/// every edge with strictly positive importance.
fn default_filter() -> Option<f64> {
    Some(0.0)
}

/// Default for [`GBGRNArgs::n_sample_genes`] (`200`).
fn default_n_samples() -> usize {
    200
}

/// Default for [`GBGRNArgs::skip_cv`] (`false`); run the CV step.
fn default_skip_cv() -> bool {
    false
}

/// Default for [`GBGRNArgs::num_iterations`] (`32`); used when
/// [`GBGRNArgs::skip_cv`] is `true`.
fn default_num_iterations() -> usize {
    32
}


/// Top-level configuration for a GBN workflow run.
///
/// Constructed by deserialising a YAML/TOML config file. Mandatory
/// fields are the input AnnData file, the transcription-factor CSV
/// list, the output HDF5 path, and the [`RunMode`]. The remaining
/// fields are optional with the defaults defined by the per-field
/// `default_*` helpers in this module.
///
/// Consumed by [`crate::gbn::run_cross_fold_gbm`] and
/// [`crate::gbn::infer_gb_network`].
#[derive(Debug, Serialize, Deserialize)]
pub struct GBGRNArgs {
    // Mandatory Fileds
    //  - Files/Paths
    /// Path to the input AnnData (`.h5ad`) expression file.
    pub h5ad_file: String,
    /// Path to the transcription-factor list (CSV); each row names
    /// one TF gene to keep on the predictor side of every model.
    pub tf_csv_file: String,
    /// Path to the HDF5 file written by
    /// [`crate::gbn::mpi_write_h5`] containing the inferred edges.
    pub output_file: String,

    /// Workflow stage to execute (see [`RunMode`]).
    pub mode: RunMode,

    /// Name of the `var/<column>` dataset holding gene identifiers
    /// in the AnnData file. Defaults to `"_index"` (see
    /// [`default_gene_id_col`]).
    #[serde(default = "default_gene_id_col")]
    pub gene_id_col: String,

    /// Decimal places used for rounding the expression matrix at
    /// load time. Default `4` (see [`default_roundup`]).
    #[serde(default = "default_roundup")]
    pub nroundup: usize,

    /// Logging verbosity. Default [`LogLevel::Off`].
    #[serde(default)]
    pub log_level: LogLevel,

    /// Optional importance threshold applied when materialising the
    /// network: edges with `importance <= threshold` are dropped.
    /// Default `Some(0.0)` (see [`default_filter`]).
    #[serde(default = "default_filter")]
    pub importance_filter: Option<f64>,

    /// LightGBM hyper-parameters used by the production GRN run.
    /// Defaults to [`GBMParams::default`].
    #[serde(default)]
    pub gbm_params: GBMParams,

    /// Number of target genes sampled for the CV stage. Default
    /// `200` (see [`default_n_samples`]).
    #[serde(default = "default_n_samples")]
    pub n_sample_genes: usize,

    /// When `true`, skip the CV stage and use
    /// [`Self::num_iterations`] directly for the production run.
    /// Default `false` (see [`default_skip_cv`]).
    #[serde(default = "default_skip_cv")]
    pub skip_cv: bool,

    /// Boosting-round count used when [`Self::skip_cv`] is `true`.
    /// Ignored otherwise (the CV median takes precedence). Default
    /// `32` (see [`default_num_iterations`]).
    #[serde(default = "default_num_iterations")]
    pub num_iterations: usize,
}
