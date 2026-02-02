use anyhow::{Ok, Result};
use hdf5::{
    self, Error as H5Error, File, H5Type,
    types::{VarLenAscii, VarLenUnicode},
};
use ndarray::{Array1, Array2};
use num::{Float, FromPrimitive, Zero};
use polars::prelude::{CsvReader, SerReader};
use sope::ensure_eq;
use std::{collections::HashMap, iter::zip, ops::Range};
use thiserror::Error;

use crate::util::around;

#[derive(Error, Debug)]
pub enum AnnDataError {
    #[error(transparent)]
    H5(#[from] H5Error),
    #[error("File {0} is missing")]
    MissingFileError(String),
    #[error("Gene {0} not found in the dataset")]
    MissingGeneError(String),
    #[error("Gene Column {0} not found in the csv file")]
    MissingGeneColumnError(String),
}

pub struct AnnData {
    pub nobs: usize,
    pub nvars: usize,
    pub npairs: usize,
    gene_ids: Vec<String>,
    gene_id_map: HashMap<String, usize>,
    _path: String,
    h5_fptr: hdf5::File,
}

pub fn var_gene_names(h5_fptr: &File, index_column: &str) -> Result<Vec<String>> {
    let ds_path = format!("var/{}", index_column);
    let ds = h5_fptr.dataset(ds_path.as_str())?;
    let raw_uni = ds.read_raw::<VarLenUnicode>();
    match raw_uni {
        Result::Ok(rvec) => {
            Ok(rvec.iter().map(|x| x.as_str().to_owned()).collect())
        }
        Result::Err(_err) => {
            let rvec = ds.read_raw::<VarLenAscii>()?;
            Ok(rvec.iter().map(|x| x.as_str().to_owned()).collect())
        }
    }
}

pub fn build_gene_index(
    h5_fptr: &File,
    index_column: &str,
) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let gene_names = var_gene_names(h5_fptr, index_column)?;
    let gene_index = HashMap::<String, usize>::from_iter(
        gene_names.iter().enumerate().map(|(i, x)| (x.clone(), i)),
    );
    Ok((gene_names, gene_index))
}

impl AnnData {
    pub fn new(path: &str, gene_index_column: Option<String>) -> Result<Self> {
        let h5_fptr = hdf5::File::open(path)?;
        let ds = h5_fptr.dataset("X")?;
        let dims = ds.shape();
        ensure_eq!(dims.len(), 2);
        let (nobs, nvars) = (dims[0], dims[1]);
        let npairs = (nvars * (nvars - 1)) / 2;
        let (gene_ids, genes_id_map) =
            if let Some(index_column) = gene_index_column {
                build_gene_index(&h5_fptr, index_column.as_str())?
            } else {
                (Vec::new(), HashMap::new())
            };
        Ok(Self {
            nobs,
            nvars,
            npairs,
            _path: path.to_owned(),
            gene_ids,
            gene_id_map: genes_id_map,
            h5_fptr,
        })
    }

    pub fn gene_index_map(&self) -> &HashMap<String, usize> {
        &self.gene_id_map
    }

    pub fn genes_ref(&self) -> &[String] {
        &self.gene_ids
    }

    pub fn gene_at(&self, i: usize) -> &String {
        &self.gene_ids[i]
    }

    pub fn read_gene_ids(&self, col_name: &str) -> Result<Vec<String>> {
        var_gene_names(&self.h5_fptr, col_name)
    }

    pub fn get_gene_index(&self, gene_id: &str) -> Option<usize> {
        self.gene_id_map.get(gene_id).cloned()
    }

    pub fn get_gene_indices(&self, gene_ids: &[String]) -> Vec<usize> {
        gene_ids
            .iter()
            .flat_map(|x| self.get_gene_index(x.as_str()))
            .collect()
    }

    pub fn read_range_data<T: H5Type>(
        &self,
        cbounds: Range<usize>,
    ) -> Result<Array2<T>> {
        let ds = self.h5_fptr.dataset("X")?;
        let rbounds = ..self.nobs;
        let rdata: Array2<T> = ds.read_slice_2d(ndarray::s![rbounds, cbounds])?;
        Ok(rdata)
    }

    pub fn read_submatrix<T: H5Type + Clone + Zero>(
        &self,
        indices: &[usize],
    ) -> Result<Array2<T>> {
        // TODO:: How to do this faster?
        let mut smat = Array2::<T>::zeros([self.nobs, indices.len()]);
        let ds = self.h5_fptr.dataset("X")?;
        for (i, col_index) in indices.iter().enumerate() {
            let scolumn = ndarray::s![..self.nobs, *col_index];
            let rd_col: Array1<T> = ds.read_slice(scolumn)?;
            smat.column_mut(i).assign(&rd_col);
        }
        // Python Version::
        //  indexes = np.array(col_indexes)
        //  indexes_asort = np.argsort(col_indexes)
        //  indexes_srtd = indexes[indexes_asort]
        //  submat_srtd: NDFloatArray = hfx["X"][:, indexes_srtd]
        //  submat = np.zeros(shape=submat_srtd.shape,
        //                    dtype=submat_srtd.dtype)
        //  submat[:, indexes_asort] = submat_srtd
        Ok(smat)
    }

    pub fn read_genes_submatrix<T: H5Type + Clone + Zero>(
        &self,
        gene_ids: &[String],
    ) -> Result<Array2<T>> {
        let indices = self.get_gene_indices(gene_ids);
        self.read_submatrix(&indices)
    }

    pub fn read_column<T: H5Type>(
        &self,
        column_index: usize,
    ) -> Result<Array1<T>> {
        // mat_col: NDFloatArray = hfx["X"][:, cindex]
        let ds = self.h5_fptr.dataset("X")?;
        let rbounds = ..self.nobs;
        Ok(ds.read_slice(ndarray::s![rbounds, column_index])?)
    }

    pub fn read_gene_column<T: H5Type>(
        &self,
        gene_name: &str,
    ) -> Result<Array1<T>> {
        let idx = self
            .get_gene_index(gene_name)
            .ok_or(AnnDataError::MissingGeneColumnError(gene_name.to_string()))?;
        self.read_column(idx)
    }
}

pub struct GeneSetAD<'a, T>
where
    T: H5Type + Clone + Zero,
{
    adata: &'a AnnData,
    genes: Vec<String>, // order of gene names in GeneSetAD
    indices: Vec<usize>,
    //pub ad_lookup: HashMap<String, usize>, // gene->location in the AnnData object
    gene_lookup: HashMap<String, usize>, // gene name ->location in the expr_matrix
    ad2geneset_map: HashMap<usize, usize>, // location in AnnData -> gene in matrix
    expr_matrix: Array2<T>,
    n_decimals: Option<usize>,
}

impl<'a, T> GeneSetAD<'a, T>
where
    T: H5Type + Clone + Zero + Float + FromPrimitive,
{
    pub fn new(
        adata: &'a AnnData,
        gene_csv: &str,
        gene_column: Option<&str>,
        n_decimals: Option<usize>,
    ) -> Result<Self> {
        let gene_column = gene_column.unwrap_or("gene");
        let csv_file = std::fs::File::open(gene_csv)?;
        let df = CsvReader::new(csv_file).finish()?;
        let gene_series = df.column(gene_column)?.as_series().ok_or(
            AnnDataError::MissingGeneColumnError(gene_column.to_string()),
        )?;
        let (genes, indices): (Vec<String>, Vec<usize>) = gene_series
            .str()?
            .iter()
            .flat_map(|x| {
                if let Some(y) = x {
                    adata.get_gene_index(y).map(|idx| (y.to_string(), idx))
                } else {
                    None
                }
            })
            .collect();
        let ngenes = genes.len();

        let expr_matrix = adata.read_submatrix::<T>(&indices)?;
        let expr_matrix = if let Some(n_decimals) = n_decimals {
            around(expr_matrix.view(), n_decimals)
        } else {
            expr_matrix
        };

        Ok(GeneSetAD {
            //ad_lookup: HashMap::from_iter(zip(
            //    genes.iter().cloned(),
            //    indices.iter().cloned(),
            //)),
            ad2geneset_map: HashMap::from_iter(
                indices.iter().cloned().enumerate().map(|(i, x)| (x, i)),
            ),
            gene_lookup: HashMap::from_iter(zip(
                genes.iter().cloned(),
                0..ngenes,
            )),
            genes,
            indices,
            adata,
            expr_matrix,
            n_decimals,
        })
    }

    pub fn from_indices(
        adata: &'a AnnData,
        tgt_indices: &[usize],
        n_decimals: Option<usize>,
    ) -> Result<Self> {
        let genes: Vec<String> = tgt_indices
            .iter()
            .map(|i| adata.gene_at(*i).clone())
            .collect();
        let indices = tgt_indices.to_vec();
        let ngenes = genes.len();
        let expr_matrix = adata.read_submatrix::<T>(&indices)?;
        let expr_matrix = if let Some(n_decimals) = n_decimals {
            around(expr_matrix.view(), n_decimals)
        } else {
            expr_matrix
        };
        Ok(GeneSetAD {
            ad2geneset_map: HashMap::from_iter(
                indices.iter().cloned().enumerate().map(|(i, x)| (x, i)),
            ),
            gene_lookup: HashMap::from_iter(zip(
                genes.iter().cloned(),
                0..ngenes,
            )),
            adata,
            genes,
            indices,
            n_decimals,
            expr_matrix,
        })
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    pub fn decimals(&self) -> Option<usize> {
        self.n_decimals
    }

    pub fn expr_matrix_ref(&self) -> &Array2<T> {
        &self.expr_matrix
    }

    pub fn indices_ref(&self) -> &[usize] {
        &self.indices
    }

    pub fn ann_data(&self) -> &AnnData {
        self.adata
    }

    pub fn contains_gene(&self, l_gene: &str) -> bool {
        self.gene_lookup.contains_key(l_gene)
    }

    pub fn contains(&self, gene_idx: usize) -> bool {
        self.ad2geneset_map.contains_key(&gene_idx)
    }

    pub fn gene_index(&self, ad_gene_idx: usize) -> Result<usize> {
        let i = self
            .ad2geneset_map
            .get(&ad_gene_idx)
            .ok_or(AnnDataError::MissingGeneError(format!("{}", ad_gene_idx)))?;
        Ok(*i)
    }

    pub fn column_for(&self, target_gene: &str) -> Result<Array1<T>> {
        let gene_idx = self
            .gene_lookup
            .get(target_gene)
            .ok_or(AnnDataError::MissingGeneError(target_gene.to_owned()))?;
        Ok(self.expr_matrix.column(*gene_idx).to_owned())
    }

    pub fn column(&self, ad_gene_index: usize) -> Result<Array1<T>> {
        let gene_idx = self.ad2geneset_map.get(&ad_gene_index).ok_or(
            AnnDataError::MissingGeneError(format!("{}", ad_gene_index)),
        )?;
        Ok(self.expr_matrix.column(*gene_idx).to_owned())
    }

    fn expr_matrix_sub_i(&self, i: usize) -> Array2<T> {
        let sub_len = self.len() - 1;
        if 0 < i && i < sub_len {
            // TODO:: How to do this faster?
            let mut smatrix =
                Array2::<T>::zeros([self.expr_matrix.nrows(), sub_len]);
            for j in 0..i {
                let rd_col = self.expr_matrix.column(j);
                smatrix.column_mut(j).assign(&rd_col);
            }
            for j in i + 1..self.len() {
                let rd_col = self.expr_matrix.column(j);
                smatrix.column_mut(j - 1).assign(&rd_col);
            }
            smatrix
        } else if i == 0 {
            self.expr_matrix.slice(ndarray::s![.., 1..]).to_owned()
        } else {
            self.expr_matrix
                .slice(ndarray::s![.., ..sub_len])
                .to_owned()
        }
    }

    pub fn expr_matrix_sub_gene(&self, target_gene: &str) -> Result<Array2<T>> {
        let i = self
            .gene_lookup
            .get(target_gene)
            .ok_or(AnnDataError::MissingGeneError(target_gene.to_owned()))?;
        Ok(self.expr_matrix_sub_i(*i))
    }

    pub fn expr_matrix_sub_gene_index(
        &self,
        ad_gene_idx: usize,
    ) -> Result<Array2<T>> {
        let i = self.gene_index(ad_gene_idx)?;
        Ok(self.expr_matrix_sub_i(i))
    }
}
