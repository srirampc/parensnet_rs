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

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum RunMode {
    #[serde(alias = "mi")]
    MI,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowArgs {
    // Mandatory Fileds
    //  - Files/Paths
    pub h5ad_file: String,
    pub mi_file: String,

    //  - Run Modes
    pub mode: Vec<RunMode>, //
    pub row_major_h5_file: Option<String>,

    #[serde(default = "default_gene_id_col")]
    pub gene_id_col: String, // = String::from_str("gene_ids"),
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


}

impl WorkflowArgs {
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
