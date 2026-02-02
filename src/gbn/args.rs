use log::LevelFilter;
use serde::{Deserialize, Serialize};

// 'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
#[repr(usize)]
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
pub enum LogLevel {
    #[serde(alias = "NOTSET")]
    #[default]
    Off = LevelFilter::Off as usize,
    #[serde(alias = "ERROR")]
    Error = LevelFilter::Error as usize,
    #[serde(alias = "WARNING")]
    Warn = LevelFilter::Warn as usize,
    #[serde(alias = "INFO")]
    Info = LevelFilter::Info as usize,
    #[serde(alias = "DEBUG")]
    Debug = LevelFilter::Debug as usize,
    Trace = LevelFilter::Trace as usize,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum RunMode {
    #[serde(alias = "optimal_iterations")]
    OptimalIterations,
    #[serde(alias = "gb_grn")]
    GBGRNet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBMParams {
    #[serde(default = "GBMParams::default_verbose")]
    pub verbose: i32,
    #[serde(default = "GBMParams::default_threads")]
    pub num_threads: i32,
    #[serde(default = "GBMParams::default_iterations")]
    pub num_iterations: usize,
    #[serde(default = "GBMParams::default_rounds")]
    pub early_stopping_rounds: usize,
    #[serde(default = "GBMParams::default_bagging_fraction")]
    pub bagging_fraction: f32,
    #[serde(default = "GBMParams::default_bagging_freq")]
    pub bagging_freq: i32,
    #[serde(default = "GBMParams::default_metric")]
    pub metric: String,
    #[serde(default = "GBMParams::default_feature_fraction")]
    pub feature_fraction: f32,
}

impl GBMParams {
    fn default_verbose() -> i32 {
        1
    }

    fn default_threads() -> i32 {
        0
    }

    fn default_iterations() -> usize {
        300
    }

    fn default_rounds() -> usize {
        10
    }

    fn default_bagging_fraction() -> f32 {
        0.9
    }

    fn default_bagging_freq() -> i32 {
        1
    }

    fn default_feature_fraction() -> f32 {
        0.1
    }

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
            early_stopping_rounds: Self::default_rounds(),
            bagging_fraction: Self::default_bagging_fraction(),
            bagging_freq: Self::default_bagging_freq(),
            metric: Self::default_metric(),
            feature_fraction: Self::default_feature_fraction(),
        }
    }
}

impl GBMParams {
    pub fn as_json(&self) -> serde_json::Value {
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
}

// Cross-Validation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVConfig {
    pub n_folds: usize,        // Number of CV folds (typically 5)
    pub n_sample_genes: usize, // Number of genes to sample for CV (100-1000)
    pub max_rounds: usize,     // Maximum boosting rounds to try (e.g., 500)
    pub early_stopping_rounds: usize, // Patience for early stopping (e.g., 10)
    // min_rounds: usize,            // TODO:Minimum rounds to train (e.g., 20)
    pub params: GBMParams,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            n_sample_genes: 200, // Sample 200 genes for CV
            max_rounds: 500,
            early_stopping_rounds: 10,
            params: GBMParams::default(),
        }
    }
}

fn default_gene_id_col() -> String {
    "_index".to_string()
}

fn default_roundup() -> usize {
    4
}

fn default_filter() -> Option<f64> {
    Some(0.0)
}

fn default_n_samples() -> usize {
    200
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GBGRNArgs {
    // Mandatory Fileds
    //  - Files/Paths
    pub h5ad_file: String,
    pub tf_csv_file: String,
    pub output_file: String,

    pub mode: RunMode,

    #[serde(default = "default_gene_id_col")]
    pub gene_id_col: String,

    #[serde(default = "default_roundup")]
    pub nroundup: usize, // =

    #[serde(default)]
    pub log_level: LogLevel, // = 'DEBUG'

    #[serde(default = "default_filter")]
    pub importance_filter: Option<f64>,

    #[serde(default)]
    pub gbm_params: GBMParams,

    #[serde(default = "default_n_samples")]
    pub n_sample_genes: usize,
}
