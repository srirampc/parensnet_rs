//! `parensnet_rs` — distributed gene-network construction toolkit.
//!
//! The crate provides MPI-parallel implementations of several
//! gene-regulatory-network (GRN) construction methods over
//! single-cell expression matrices stored in AnnData (`.h5ad`)
//! format. The modules are organised into three broad groups:
//!
//! # Network construction workflows
//!
//! Each of these modules exposes a YAML/TOML-configurable
//! `execute_workflow` (or equivalent) entry point that dispatches
//! over a list of stages and writes the resulting network to HDF5.
//!
//! * [`pucn`] — PUC Network pipeline: builds per-variable histograms
//!   (Bayesian-blocks or fixed-width), computes pairwise MI / SI /
//!   LMR data, and runs the PUC redundancy scoring of Chan et al.,
//!   2017. Supports sampled, LMR-based, and fully distributed PUC
//!   variants ([`pucn::RunMode`]).
//! * [`gbn`] — GRNBoost-style gradient-boosted GRN inference
//!   ([`gbn::infer_gb_network`]) based on Arboreto (Moerman et al. ,
//!   2019). Adds a k-fold cross-validation pass ([`gbn::mpi_cv_gbm`])
//!   to estimate the boosting-round count before block-distributing
//!   the per-target [`lightgbm3::Booster`] training across ranks.
//! * [`mcpn`] — MCPNet B-spline MI workflow: distributes the
//!   B-spline weight construction and the pairwise MI evaluation
//!   ([`mcpn::RunMode::MIBSplineWeights`] /
//!   [`mcpn::RunMode::MIBSpline`]) over MPI and persists the
//!   results to HDF5.
//!
//! # Information-measure kernels
//!
//! Low-level math used by the workflows above.
//!
//! * [`hist`] — variable-width histogram builders (Bayesian-blocks
//!   and Knuth's rule) plus the joint-histogram routines used to
//!   feed the MI / SI estimators.
//! * [`mvim`] — multi-variable information measures (`I(X;Y)`,
//!   specific information, Williams–Beer redundancy, LMR), the
//!   [`mvim::rv::MRVTrait`] abstraction for multi-variable
//!   distributions, and the HDF5-backed [`mvim::misi`] data
//!   structures that cache the LMR sorted/prefix-sum tables.
//! * [`corr`] — B-spline mutual information ([`corr::mi`]) used by
//!   [`mcpn`]; wraps the C kernels in the bundled `mcpnet_rs`
//!   crate and provides a pure-Rust reference implementation.
//!
//! # I/O, MPI plumbing, and common utilities
//!
//! * [`anndata`] — read-side wrapper around AnnData `.h5ad` files
//!   ([`anndata::AnnData`], [`anndata::GeneSetAD`]) with helpers
//!   for column selection, sub-matrix loading, and decimal rounding.
//! * [`h5`] — serial ([`h5::io`]) and MPI-collective
//!   ([`h5::mpio`]) HDF5 readers and writers used by every workflow
//!   to load expression matrices and persist intermediate data.
//! * [`comm`] — [`comm::CommIfx`], the light wrapper around the
//!   `sope` MPI communicator that carries `rank`, `size`, and the
//!   underlying [`mpi::topology::SimpleCommunicator`] used by every
//!   collective in the crate.
//! * [`util`] — pair / block work distributors
//!   ([`util::PairWorkDistributor`]), the [`util::Vec2d`] /
//!   [`util::IdVResults`] containers shared across workflows, and
//!   miscellaneous slice / file helpers.
//! * [`types`] — numeric and serde-friendly marker traits
//!   ([`types::PNInteger`], [`types::PNFloat`], `DiscretizerMethod`,
//!   `LogBase`) used throughout the crate to keep the workflow
//!   helpers generic over integer / float widths.

pub mod comm;
pub mod corr;
pub mod h5;
pub mod hist;
pub mod mcpn;
pub mod mvim;
pub mod types;
pub mod util;
pub mod pucn;
pub mod anndata;
pub mod gbn;

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use flate2::bufread::GzDecoder;
    use log::info;
    use ndarray::{Array1, Array2};
    use serde::{
        de::DeserializeOwned,
        {Deserialize, Serialize},
    };
    use std::{collections::HashMap, fmt::Debug, io::Read, ops::Range};

    #[macro_export]
    macro_rules! test_config_file_path {
        ($name:literal) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/config/", $name)
        };
    }

    #[macro_export]
    macro_rules! test_data_file_path {
        ($name:literal) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/data/", $name)
        };
    }

    #[macro_export]
    macro_rules! test_ut_data_file_path {
        ($name:literal) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/data/unit_tests/", $name)
        };
    }

    pub fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct HistNode {
        pub bins: Vec<f32>,
        pub hist: Vec<u32>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct HistPair {
        pub pair: Vec<u32>,
        pub hist: Array2<u32>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct HistTestData {
        pub data: Array2<f32>,
        pub nodes: Vec<HistNode>,
        pub node_pairs: Vec<HistPair>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct PUCNode {
        pub label: String,
        pub number_of_bins: usize,
        pub probabilities: Vec<f32>,
        pub binned_values: Vec<u32>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct PUCNodePair {
        pub mi: f32,
        pub si: Vec<f32>,
        pub lmr: Option<Vec<f32>>,
        pub pxy: Vec<Vec<f32>>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct PUCData {
        pub nodes: Vec<PUCNode>,
        pub node_pairs: Vec<Vec<PUCNodePair>>,
        pub redundancy_values: HashMap<String, f32>,
        pub puc_scores: Vec<Vec<f32>>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct PUCTestData {
        pub data: HashMap<String, PUCData>,
    }

    impl PUCData {
        pub fn get_mi(&self, x: usize, y: usize) -> f32 {
            self.node_pairs[x][y].mi
        }
        pub fn get_si(&self, x: usize, y: usize) -> Array1<f32> {
            Array1::from_vec(self.node_pairs[x][y].si.clone())
        }

        pub fn get_lmr(&self, x: usize, y: usize) -> Option<Array1<f32>> {
            self.node_pairs[x][y]
                .lmr
                .as_ref()
                .map(|vlmr| Array1::from_vec(vlmr.clone()))
        }
        pub fn get_redundancy(&self, x: usize, y: usize, z: usize) -> f32 {
            let lkey = format!("({}, {}, {})", x + 1, y + 1, z + 1);
            self.redundancy_values[&lkey]
        }
        pub fn get_hist(
            &self,
            nobs: usize,
            x: usize,
            y: usize,
        ) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
            let (x_node, y_node) = (&self.nodes[x], &self.nodes[y]);
            let xy_pair = &self.node_pairs[x][y];
            let rshape = (xy_pair.pxy.len(), xy_pair.pxy[0].len());

            (
                Array2::from_shape_fn(rshape, |(i, j)| {
                    xy_pair.pxy[i][j] * nobs as f32
                }),
                Array1::from_shape_fn(x_node.probabilities.len(), |i| {
                    x_node.probabilities[i] * nobs as f32
                }),
                Array1::from_shape_fn(y_node.probabilities.len(), |i| {
                    y_node.probabilities[i] * nobs as f32
                }),
            )
        }
    }

    pub fn parse_gz_test_data<T>(data_file: &str) -> Result<T>
    where
        T: Debug + Serialize + DeserializeOwned,
    {
        info!("Parsing : [{}]", data_file);
        let bin_data = std::fs::read(data_file)?;
        let mut contents = String::new();
        GzDecoder::new(bin_data.as_slice()).read_to_string(&mut contents)?;
        match serde_json::from_str::<T>(&contents) {
            Ok(puc_data) => Ok(puc_data),
            Err(err) => Err(anyhow::Error::from(err)),
        }
    }

    #[allow(dead_code)]
    pub fn puc_test_data() -> Result<PUCTestData> {
        parse_gz_test_data(test_ut_data_file_path!("puc_data_small.json.gz"))
    }

    #[allow(dead_code)]
    pub fn puc_test4_data() -> Result<PUCTestData> {
        parse_gz_test_data(test_ut_data_file_path!("puc_data4_small.json.gz"))
    }

    #[allow(dead_code)]
    pub fn puc_test4_data_w_lmr() -> Result<PUCTestData> {
        parse_gz_test_data(test_ut_data_file_path!("puc_data4_w_lmr_small.json.gz"))
    }
    #[allow(dead_code)]
    pub fn hist_test_data() -> Result<HistTestData> {
        parse_gz_test_data(test_ut_data_file_path!("bbh_test.json.gz"))
    }

    #[allow(dead_code)]
    pub fn hist_large_test_data() -> Result<HistTestData> {
        parse_gz_test_data(test_ut_data_file_path!("bbh_large_test.json.gz"))
    }

    #[allow(dead_code)]
    pub fn test_exp_matrix() -> Result<Array2<f32>, hdf5::Error> {
        crate::h5::io::read_2d::<f32>(
            test_ut_data_file_path!("d800k_3genes.h5"),
            "X",
        )
    }

    #[allow(dead_code)]
    pub fn test_tf_file() -> &'static str {
        test_data_file_path!("/pbmc/trrust_tf.txt")
    }

    #[allow(dead_code)]
    pub fn test_misi_file() -> &'static str {
        test_data_file_path!("/pbmc/adata.20k.500.misidata.h5")
    }

    #[allow(dead_code)]
    pub fn test_exp_sub_matrix(
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<Array2<f32>, hdf5::Error> {
        crate::h5::io::read2d_slice::<f32, usize>(
            test_ut_data_file_path!("d800k_3genes.h5"),
            "X",
            &rows,
            &cols,
        )
    }
}
