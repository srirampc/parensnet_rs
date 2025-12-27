pub mod comm;
pub mod h5;
pub mod hist;
pub mod mvim;
pub mod types;
pub mod util;
pub mod workflow;

#[cfg(test)]
mod tests {
    use flate2::bufread::GzDecoder;
    use hdf5_metno::{self as hdf5};
    use log::info;
    use ndarray::{Array1, Array2};
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fmt::Debug;
    use std::io::Read;
    use std::ops::Range;

    use crate::util::GenericError;

    #[macro_export]
    macro_rules! test_config_file_path {
        ($name:literal) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/config/test/", $name)
        };
    }

    #[macro_export]
    macro_rules! test_data_file_path {
        ($name:literal) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/data/", $name)
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

    pub fn parse_gz_test_data<T>(data_file: &str) -> Result<T, GenericError>
    where
        T: Debug + Serialize + DeserializeOwned,
    {
        info!("Parsing : [{}]", data_file);
        let bin_data = std::fs::read(data_file)?;
        let mut contents = String::new();
        GzDecoder::new(bin_data.as_slice()).read_to_string(&mut contents)?;

        match serde_json::from_str::<T>(&contents) {
            Ok(puc_data) => Ok(puc_data),
            Err(err) => Err(GenericError::from(err)),
        }
    }

    #[allow(dead_code)]
    pub fn puc_test_data() -> Result<PUCTestData, GenericError> {
        parse_gz_test_data(test_data_file_path!("puc_data_small.json.gz"))
    }

    #[allow(dead_code)]
    pub fn puc_test4_data() -> Result<PUCTestData, GenericError> {
        parse_gz_test_data(test_data_file_path!("puc_data4_small.json.gz"))
    }

    #[allow(dead_code)]
    pub fn puc_test4_data_w_lmr() -> Result<PUCTestData, GenericError> {
        parse_gz_test_data(test_data_file_path!("puc_data4_w_lmr_small.json.gz"))
    }
    #[allow(dead_code)]
    pub fn hist_test_data() -> Result<HistTestData, GenericError> {
        parse_gz_test_data(test_data_file_path!("bbh_test.json.gz"))
    }

    #[allow(dead_code)]
    pub fn hist_large_test_data() -> Result<HistTestData, GenericError> {
        parse_gz_test_data(test_data_file_path!("bbh_large_test.json.gz"))
    }

    #[allow(dead_code)]
    pub fn test_exp_matrix() -> Result<Array2<f32>, hdf5::Error> {
        crate::h5::io::read_2d::<f32>(
            test_data_file_path!("d800k_3genes.h5"),
            "X",
        )
    }

    #[allow(dead_code)]
    pub fn test_exp_sub_matrix(
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<Array2<f32>, hdf5::Error> {
        crate::h5::io::read2d_slice::<f32, usize>(
            test_data_file_path!("d800k_3genes.h5"),
            "X",
            &rows,
            &cols,
        )
    }
}
