use crate::{
    types::{DiscretizerMethod, LogBase},
    util::GenericError,
};
use hdf5;
use log::LevelFilter;
use serde::{Deserialize, Serialize};

fn default_gene_id_col() -> String {
    "gene_ids".to_string()
}

fn default_roundup() -> usize {
    4
}

fn default_nrounds() -> usize {
    8
}

fn default_nsamples() -> usize {
    200
}

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
    #[serde(alias = "misi")]
    MISI,
    #[serde(alias = "sampled_puc_pairs")]
    SampledPUCPairs,
    #[serde(alias = "samples_ranges")]
    SamplesRanges,
    #[serde(alias = "samples_input")]
    SamplesInput,
    #[serde(alias = "samples_lmr_ranges")]
    SamplesLMRRanges,
    #[serde(alias = "samples_lmr_input")]
    SamplesLMRInput,
    #[serde(alias = "puc_ranges")]
    PUCRanges,
    #[serde(alias = "puc_lmr")]
    PUCLMR,
    #[serde(alias = "puc2pidc")]
    PUC2PIDC,
    #[serde(alias = "puc_union")]
    PUCUnion,
    #[serde(alias = "cluster_union")]
    ClusterUnion,
    #[serde(alias = "cluster_lmr_union")]
    ClusterLMRUnion,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowArgs {
    // Mandatory Fileds
    //  - Files/Paths
    pub h5ad_file: String,
    pub misi_data_file: String,
    pub puc_file: String,
    pub pidc_file: String,

    //  - Run Modes
    pub mode: Vec<RunMode>, //

    // Optional Fields with defaults
    pub nodes_pickle: Option<String>,
    pub nodes_pairs_pickle: Option<String>,
    pub samples_file: Option<String>,
    pub sub_net_files: Option<Vec<String>>,
    pub wflow_dir: Option<String>,

    #[serde(default = "default_gene_id_col")]
    pub gene_id_col: String, // = String::from_str("gene_ids"),

    // Flags
    #[serde(default)]
    pub save_nodes: bool, // = False
    #[serde(default)]
    pub save_node_pairs: bool, // = False
    #[serde(default)]
    pub enable_timers: bool, // = True

    //
    #[serde(default = "default_roundup")]
    pub nroundup: usize, // = 4
    #[serde(default = "default_nrounds")]
    pub nrounds: usize, // = 8
    #[serde(default = "default_nsamples")]
    pub nsamples: usize, // = 200
    #[serde(default)]
    pub nobs: usize, // = 0
    #[serde(default)]
    pub nvars: usize, // = 0
    #[serde(default)]
    pub npairs: usize, // = 0

    #[serde(default)]
    pub quantile_filter: Option<f32>,

    #[serde(default)]
    pub disc_method: DiscretizerMethod, // = 'bayesian_blocks',
    #[serde(default)]
    pub tbase: LogBase, // = '2'

    #[serde(default)]
    pub log_level: LogLevel, // = 'DEBUG'

    #[serde(skip)]
    h5_fptr: Option<hdf5::File>,
}

impl WorkflowArgs {
    pub fn update(&mut self) -> Result<(), GenericError> {
        let file = hdf5::File::open(&self.h5ad_file)?;
        let ds = file.dataset("X")?;
        let dims = ds.shape();
        assert_eq!(dims.len(), 2);
        if self.nobs == 0 {
            self.nobs = dims[0];
        }
        if self.nvars == 0 {
            self.nvars = dims[1];
        }
        if self.npairs == 0 {
            self.npairs = (self.nvars * (self.nvars - 1)) / 2;
        }
        self.h5_fptr = Some(file);
        Ok(())
    }
}

impl Drop for WorkflowArgs {
    fn drop(&mut self) {
        if let Some(h5_fptr) = self.h5_fptr.take() {
            let _ = h5_fptr.close();
        }
    }
}
