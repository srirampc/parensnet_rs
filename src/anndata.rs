//! Lightweight interface to [anndata](https://anndata.readthedocs.io)
//! files.
//!
//! Implements just enough of the
//! [HDF5 file format](https://anndata.readthedocs.io/en/latest/fileformat-prose.html)
//! used by `anndata` to load the dense observation × variable
//! expression matrix `X`, the gene name vector under `var/<column>`,
//! and to support both sequential and parallel-IO (MPI) reads of
//! column ranges or whole sub-matrices.
//!
//! The module exposes two main types:
//!
//! * [`AnnData`] — opens an `.h5ad` file, caches the matrix shape and
//!   the gene-name → column-index lookup, and exposes a family of
//!   `read_*` / `par_read_*` helpers for both column ranges and
//!   user-selected sub-matrices.
//! * [`GeneSetAD`] — borrows an [`AnnData`] and an externally supplied
//!   subset of genes (from a CSV file or an explicit index list),
//!   extracts the corresponding expression sub-matrix form HDF5 file and
//!   provides bidirectional lookups between gene names, original
//!   AnnData indices, and the columns of that sub-matrix.
//!
//! Errors raised by the module are gathered in [`AnnDataError`].

use anyhow::{Ok, Result};
use hdf5::{
    self, Error as H5Error, File, H5Type,
    types::{VarLenAscii, VarLenUnicode},
};
use ndarray::{Array1, Array2};
use num::{Float, FromPrimitive, Zero};
use sope::ensure_eq;
use std::{collections::HashMap, iter::zip, ops::Range};
use thiserror::Error;

use crate::{
    comm::CommIfx,
    cond_debug,
    h5::mpio,
    util::{around, read_csv_column},
};

/// Errors that can be produced while reading an anndata file.
#[derive(Error, Debug)]
pub enum AnnDataError {
    /// Pass-through wrapper around a low-level `hdf5::Error`.
    #[error(transparent)]
    H5(#[from] H5Error),
    /// The requested input file does not exist on disk.
    #[error("File {0} is missing")]
    MissingFileError(String),
    /// A gene name (or gene index) that was looked up in the AnnData
    /// gene-id map could not be found.
    #[error("Gene {0} not found in the dataset")]
    MissingGeneError(String),
    /// The gene-name column requested from a CSV file is not present
    /// in its header.
    #[error("Gene Column {0} not found in the csv file")]
    MissingGeneColumnError(String),
}

/// Light wrapper for anndata.
///
/// # Description
/// Holds the size: number of obervations, variables, gene_ids.
/// Includes a hashmap for fast identification of gene identifiers from names
/// Optionally, row major file is present, which enables a fast reading
/// of the matrix. Row major file enbales faster gene-striped reading of X.
pub struct AnnData {
    /// Number of observations (rows of `X`).
    pub nobs: usize,
    /// Number of variables / genes (columns of `X`).
    pub nvars: usize,
    /// Number of unordered variable pairs `nvars * (nvars - 1) / 2`,
    /// pre-computed for downstream pair-wise kernels.
    pub npairs: usize,
    /// Gene identifiers in the same order as the columns of `X`.
    /// Empty when no `gene_index_column` was supplied to [`AnnData::new`].
    gene_ids: Vec<String>,
    /// Reverse lookup `gene name -> column index`.
    gene_id_map: HashMap<String, usize>,
    /// Path to the column-major (`anndata`) HDF5 file.
    path: String,
    /// Optional sibling file storing `X` in row-major order. When
    /// present, the `*_rmajor_*` readers prefer it because column
    /// ranges then live in contiguous storage and read much faster.
    row_major_h5: Option<String>,
}

/// Returns the dimension of the 'X' dataset in the input hdf5 file.
/// Expects that 'X' be a 2D dataset
pub fn xds_dimensions(ad_fname: &str) -> Result<(usize, usize)> {
    let h5_fptr = hdf5::File::open(ad_fname)?;
    let ds = h5_fptr.dataset("X")?;
    let dims = ds.shape();
    ensure_eq!(dims.len(), 2);
    let (nobs, nvars) = (dims[0], dims[1]);
    Ok((nobs, nvars))
}

/// Returns the gene names stored in the `var/<index_column>` dataset
/// of an anndata file.
///
/// Use variable-length unicode first (`VarLenUnicode`), and
/// if that fails, falls back to ASCII (`VarLenAscii`).
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

/// Returns the gene names and the corresponding gene indices
///
/// # Description
/// Given a HDF5 file object pointing an AnnData file and an 'vars' column
/// name corresponding to the gene names, returns the gene names
/// and gene ids.
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
    /// Construct an AnnData object for a given path,  optional gene 
    /// name column to index with, and optional row major HDF5 file.
    pub fn new(
        path: &str,
        gene_index_column: Option<String>,
        row_major_h5: Option<String>,
    ) -> Result<Self> {
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
            path: path.to_owned(),
            row_major_h5,
            gene_ids,
            gene_id_map: genes_id_map,
        })
    }

    /// Open the underlying HDF5 file for sequential reading.
    fn open_r(&self) -> Result<hdf5::File> {
        let h5fptr = hdf5::File::open(&self.path)?;
        Ok(h5fptr)
    }

    /// Collectively open the underlying HDF5 file with the MPIO
    /// driver from [`crate::h5::mpio::open_file`].
    fn open_mpio(&self, cx: &CommIfx) -> Result<hdf5::File> {
        let h5fptr = mpio::open_file(cx, &self.path)?;
        Ok(h5fptr)
    }

    /// Borrow the cached `gene name -> column index` map.
    pub fn gene_index_map(&self) -> &HashMap<String, usize> {
        &self.gene_id_map
    }

    /// Borrow the cached gene-identifier vector. Indexed in the same
    /// order as the columns of `X`.
    pub fn genes_ref(&self) -> &[String] {
        &self.gene_ids
    }

    /// Return the gene identifier in column `i` of `X`.
    pub fn gene_at(&self, i: usize) -> &String {
        &self.gene_ids[i]
    }

    /// Re-read the gene-id column `col_name` from disk (as opposed to
    /// returning the cached copy stored on the [`AnnData`] instance).
    pub fn read_gene_ids(&self, col_name: &str) -> Result<Vec<String>> {
        var_gene_names(&self.open_r()?, col_name)
    }

    /// Look up a single gene name in the cached map.
    pub fn get_gene_index(&self, gene_id: &str) -> Option<usize> {
        self.gene_id_map.get(gene_id).cloned()
    }

    /// Look up many gene names at once; filters any name
    /// that is not present in the dataset. Returned indices are
    /// in the order of the *kept* names from `gene_ids`.
    pub fn get_gene_indices(&self, gene_ids: &[String]) -> Vec<usize> {
        gene_ids
            .iter()
            .flat_map(|x| self.get_gene_index(x.as_str()))
            .collect()
    }

    /// Use MPI parallel IO to block-read a column range from the
    /// row-major sibling file across all processes.
    ///
    /// Uses [`AnnData::row_major_h5`] to read the ranges of columns with 
    /// [`mpio::read_range_data_t`]. Otherwise, it falls back to
    /// [`AnnData::par_read_range_data`] on the column-major file.
    pub fn par_rmajor_range_data<T: H5Type + Clone>(
        &self,
        cbounds: Range<usize>,
        cx: &CommIfx,
    ) -> Result<Array2<T>> {
        if let Some(h5path) = self.row_major_h5.as_ref() {
            let rdata =
                mpio::read_range_data_t(h5path, "X", cbounds, 0..self.nobs, cx)?;
            Ok(rdata)
        } else {
            cond_debug!(cx.is_root(); "No row file; Switching to default read");
            self.par_read_range_data(cbounds, cx)
        }
    }

    /// Column-wise parallel block read followed by an [`around`] pass
    /// that rounds every element to `n_decimals` decimal places.
    ///
    /// When `n_decimals == 0` the rounding step is skipped and the
    /// raw matrix is returned untouched.
    pub fn par_rmajor_read_range_data_around<
        T: H5Type + Float + FromPrimitive,
    >(
        &self,
        cbounds: Range<usize>,
        n_decimals: usize,
        cx: &CommIfx,
    ) -> Result<Array2<T>> {
        let rdata = self.par_rmajor_range_data(cbounds, cx)?;
        Ok(if n_decimals > 0 {
            around(rdata.view(), n_decimals)
        } else {
            rdata
        })
    }

    /// Parallel-IO read of the `..self.nobs x cbounds` slice of `X`
    /// using independent (non-collective) HDF5 selections.
    ///
    /// Each rank requests its own column range and the file is opened
    /// collectively via [`open_mpio`](Self::open_mpio).
    pub fn par_read_range_data<T: H5Type>(
        &self,
        cbounds: Range<usize>,
        cx: &CommIfx,
    ) -> Result<Array2<T>> {
        // TODO::
        let ds = self.open_mpio(cx)?.dataset("X")?;
        let selection = ndarray::s![..self.nobs, cbounds];
        let rdata: Array2<T> = ds.as_reader().indi_read_slice_2d(selection)?;
        Ok(rdata)
    }

    /// Same as [`Self::par_read_range_data`] but rounds the result to
    /// `n_decimals` decimal places (skipped when `n_decimals == 0`).
    pub fn par_read_range_data_around<T: H5Type + Float + FromPrimitive>(
        &self,
        cbounds: Range<usize>,
        n_decimals: usize,
        cx: &CommIfx,
    ) -> Result<Array2<T>> {
        let rdata = self.par_read_range_data(cbounds, cx)?;
        Ok(if n_decimals > 0 {
            around(rdata.view(), n_decimals)
        } else {
            rdata
        })
    }

    /// Sequentially read the `..self.nobs x cbounds` sub-block of
    /// `X` from the column-major file.
    pub fn read_range_data<T: H5Type>(
        &self,
        cbounds: Range<usize>,
    ) -> Result<Array2<T>> {
        let ds = self.open_r()?.dataset("X")?;
        let rbounds = ..self.nobs;
        let rdata: Array2<T> = ds.read_slice_2d(ndarray::s![rbounds, cbounds])?;
        Ok(rdata)
    }

    /// Sequentially read the `..self.nobs x cbounds` sub-block of `X`
    /// preferring the row-major sibling file when available.
    ///
    /// Falls back to [`Self::read_range_data`] on the column-major
    /// file when no row-major sibling has been registered.
    pub fn read_rmajor_range_data<T: H5Type>(
        &self,
        cbounds: Range<usize>,
    ) -> Result<Array2<T>> {
        if let Some(h5path) = self.row_major_h5.as_ref() {
            let h5fptr = hdf5::File::open(h5path)?;
            let ds = h5fptr.dataset("X")?;
            let rbounds = ..self.nobs;
            let rdata: Array2<T> =
                ds.read_slice_2d(ndarray::s![rbounds, cbounds])?;
            Ok(rdata)
        } else {
            self.read_range_data(cbounds)
        }
    }

    /// [`Self::read_range_data`] composed with an [`around`] pass that
    /// rounds every element to `n_decimals` decimal places (skipped
    /// when `n_decimals == 0`).
    pub fn read_range_data_around<T: H5Type + Float + FromPrimitive>(
        &self,
        cbounds: Range<usize>,
        n_decimals: usize,
    ) -> Result<Array2<T>> {
        let rdata = self.read_range_data(cbounds)?;
        Ok(if n_decimals > 0 {
            around(rdata.view(), n_decimals)
        } else {
            rdata
        })
    }

    /// Read an arbitrary subset of columns of `X` and return them as
    /// a `nobs x indices.len()` matrix.
    ///
    /// Columns are read one at a time and assembled into the output
    /// in the order given by `indices`. Each `indices[i]` must be a
    /// valid column of the original `X` dataset.
    pub fn read_submatrix<T: H5Type + Clone + Zero>(
        &self,
        indices: &[usize],
    ) -> Result<Array2<T>> {
        // TODO:: How to do this faster?
        let mut smat = Array2::<T>::zeros([self.nobs, indices.len()]);
        let ds = self.open_r()?.dataset("X")?;
        for (i, col_index) in indices.iter().enumerate() {
            let scolumn = ndarray::s![..self.nobs, *col_index];
            let rd_col: Array1<T> = ds.read_slice(scolumn)?;
            smat.column_mut(i).assign(&rd_col);
        }
        // TODO:: use the rmajor file, if available
        Ok(smat)
    }

    /// Read the columns named by `gene_ids` and return the resulting
    /// sub-matrix. Names that are not present in the gene-id map are
    /// silently dropped (see [`Self::get_gene_indices`]).
    pub fn read_genes_submatrix<T: H5Type + Clone + Zero>(
        &self,
        gene_ids: &[String],
    ) -> Result<Array2<T>> {
        let indices = self.get_gene_indices(gene_ids);
        self.read_submatrix(&indices)
    }

    /// Read a single column `column_index` of `X` as an `Array1`.
    pub fn read_column<T: H5Type>(
        &self,
        column_index: usize,
    ) -> Result<Array1<T>> {
        // mat_col: NDFloatArray = hfx["X"][:, cindex]
        let ds = self.open_r()?.dataset("X")?;
        let rbounds = ..self.nobs;
        Ok(ds.read_slice(ndarray::s![rbounds, column_index])?)
    }

    /// [`Self::read_column`] followed by an [`around`] rounding pass
    /// to `n_decimals` decimal places (skipped when `n_decimals == 0`).
    pub fn read_column_around<T: H5Type + Float + FromPrimitive>(
        &self,
        column_index: usize,
        n_decimals: usize,
    ) -> Result<Array1<T>> {
        // mat_col: NDFloatArray = hfx["X"][:, cindex]
        let ds = self.open_r()?.dataset("X")?;
        let rbounds = ..self.nobs;
        let cdata = ds.read_slice(ndarray::s![rbounds, column_index])?;
        Ok(if n_decimals > 0 {
            around(cdata.view(), n_decimals)
        } else {
            cdata
        })
    }

    /// Look up `gene_name` in the cached map and return the
    /// corresponding column of `X`. Errors with
    /// [`AnnDataError::MissingGeneColumnError`] when the name is
    /// unknown.
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

/// A pre-loaded subset of an [`AnnData`]'s expression matrix together
/// with the lookup tables needed to navigate it.
///
/// Built either from a CSV file listing the genes of interest
/// ([`GeneSetAD::new`]) or from an explicit list of column indices
/// ([`GeneSetAD::from_indices`]). The type stores:
///
/// * an owned `nobs x ngenes` expression matrix in
///   [`expr_matrix`](Self::expr_matrix_ref) order;
/// * the gene names selected (in the same column order); and
/// * forward and reverse lookups between gene names, AnnData column
///   indices, and the local sub-matrix column indices.
///
/// An optional `n_decimals` rounds the cached matrix once at load.
pub struct GeneSetAD<'a, T>
where
    T: H5Type + Clone + Zero,
{
    /// Borrow of the parent [`AnnData`] this gene set was extracted
    /// from.
    adata: &'a AnnData,
    /// Gene names in the order they appear in `expr_matrix` columns.
    genes: Vec<String>,
    /// AnnData column index for each `genes[i]`; same length as
    /// `genes`.
    indices: Vec<usize>,
    //pub ad_lookup: HashMap<String, usize>, // gene->location in the AnnData object
    /// `gene name -> column index in expr_matrix`.
    gene_lookup: HashMap<String, usize>,
    /// `AnnData column index -> column index in expr_matrix`.
    ad2geneset_map: HashMap<usize, usize>,
    /// Cached expression sub-matrix (`nobs x genes.len()`).
    expr_matrix: Array2<T>,
    /// Optional decimal-rounding applied to `expr_matrix` at build
    /// time; `None` means the values were left untouched.
    n_decimals: Option<usize>,
}

impl<'a, T> GeneSetAD<'a, T>
where
    T: H5Type + Clone + Zero + Float + FromPrimitive,
{
    /// Build a [`GeneSetAD`] from a CSV file of gene names.
    ///
    /// `gene_csv` should be a CSV with a header row; `gene_column`
    /// names the column to read (defaulting to `"gene"`). Names that
    /// are not present in the parent [`AnnData`] are silently
    /// dropped. The corresponding columns of `X` are read into a
    /// dense matrix and rounded to `n_decimals` decimal places when
    /// supplied.
    pub fn new(
        adata: &'a AnnData,
        gene_csv: &str,
        gene_column: Option<&str>,
        n_decimals: Option<usize>,
    ) -> Result<Self> {
        let gene_column = gene_column.unwrap_or("gene");
        let in_genes = read_csv_column(gene_csv, gene_column)?;
        let (genes, indices): (Vec<String>, Vec<usize>) = in_genes
            .into_iter()
            .flat_map(|y| adata.get_gene_index(&y).map(|idx| (y, idx)))
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

    /// Build a [`GeneSetAD`] from an explicit list of AnnData column
    /// indices.
    ///
    /// Equivalent to [`Self::new`] but skips the CSV/name resolution
    /// step. The resulting column order matches `tgt_indices`.
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

    /// Number of genes (and therefore columns of `expr_matrix`) in
    /// this set.
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    /// `true` when no genes were retained by the constructor.
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    /// The decimal-rounding setting captured at build time.
    pub fn decimals(&self) -> Option<usize> {
        self.n_decimals
    }

    /// Borrow the cached expression sub-matrix.
    pub fn expr_matrix_ref(&self) -> &Array2<T> {
        &self.expr_matrix
    }

    /// Borrow the AnnData column indices that back the columns of
    /// `expr_matrix`.
    pub fn indices_ref(&self) -> &[usize] {
        &self.indices
    }

    /// Borrow the parent [`AnnData`] this set was built from.
    pub fn ann_data(&self) -> &AnnData {
        self.adata
    }

    /// Test membership by gene name.
    pub fn contains_gene(&self, l_gene: &str) -> bool {
        self.gene_lookup.contains_key(l_gene)
    }

    /// Test membership by AnnData column index.
    pub fn contains(&self, gene_idx: usize) -> bool {
        self.ad2geneset_map.contains_key(&gene_idx)
    }

    /// Translate an AnnData column index to its position in the local
    /// `expr_matrix`. Errors with [`AnnDataError::MissingGeneError`]
    /// when the index is not part of this set.
    pub fn gene_index(&self, ad_gene_idx: usize) -> Result<usize> {
        let i = self
            .ad2geneset_map
            .get(&ad_gene_idx)
            .ok_or(AnnDataError::MissingGeneError(format!("{}", ad_gene_idx)))?;
        Ok(*i)
    }

    /// Return the column of `expr_matrix` for `target_gene` (looked
    /// up by name).
    pub fn column_for(&self, target_gene: &str) -> Result<Array1<T>> {
        let gene_idx = self
            .gene_lookup
            .get(target_gene)
            .ok_or(AnnDataError::MissingGeneError(target_gene.to_owned()))?;
        Ok(self.expr_matrix.column(*gene_idx).to_owned())
    }

    /// Return the column of `expr_matrix` corresponding to AnnData
    /// column index `ad_gene_index`.
    pub fn column(&self, ad_gene_index: usize) -> Result<Array1<T>> {
        let gene_idx = self.ad2geneset_map.get(&ad_gene_index).ok_or(
            AnnDataError::MissingGeneError(format!("{}", ad_gene_index)),
        )?;
        Ok(self.expr_matrix.column(*gene_idx).to_owned())
    }

    /// Build the leave-one-out sub-matrix that omits the `i`-th
    /// column of `expr_matrix`.
    ///
    /// Used by the network-building kernels to produce the predictor
    /// matrix for a target gene by dropping that gene's own column.
    /// Special-cased for `i == 0` and `i == len() - 1` so a single
    /// contiguous slice can be reused without copying.
    fn expr_matrix_sub_i(&self, i: usize) -> Array2<T> {
        let sub_len = self.len() - 1;
        if 0 < i && i < sub_len {
            // TODO:: This step is quite slow. Make this faster.
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

    /// Return `expr_matrix` with `target_gene`'s column removed.
    /// Convenience wrapper around [`Self::expr_matrix_sub_i`].
    pub fn expr_matrix_sub_gene(&self, target_gene: &str) -> Result<Array2<T>> {
        let i = self
            .gene_lookup
            .get(target_gene)
            .ok_or(AnnDataError::MissingGeneError(target_gene.to_owned()))?;
        Ok(self.expr_matrix_sub_i(*i))
    }

    /// Return `expr_matrix` with the column for AnnData index
    /// `ad_gene_idx` removed. Companion to [`Self::expr_matrix_sub_gene`]
    /// for callers that already hold the integer index.
    pub fn expr_matrix_sub_gene_index(
        &self,
        ad_gene_idx: usize,
    ) -> Result<Array2<T>> {
        let i = self.gene_index(ad_gene_idx)?;
        Ok(self.expr_matrix_sub_i(i))
    }
}
