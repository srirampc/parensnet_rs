//! Cross-validation utilities for picking a boosting-round count.
//!
//! The CV stage trains one LightGBM model per `(target gene, fold)`
//! pair via [`train_with_early_stopping`], records the early-stopped
//! iteration count, and aggregates the counts into a [`CVStats`].
//! The median of the recorded counts is then used as the
//! `num_iterations` for the production GBN run.
//!
//! Two entry points are exposed:
//! * [`cv_gbm`] runs the loop sequentially on a single rank.
//! * [`mpi_cv_gbm`] block-distributes the `(gene, fold)` runs across
//!   MPI ranks (see the [`DistCVConfig`] helper) and gathers the
//!   results with [`allgatherv_full_vec`].
//!
//! Both share [`cross_validate_target`], which performs the per-gene
//! K-fold loop, and [`KFold`], which shuffles a row index vector and
//! emits the train/validation split for a given fold.

use anyhow::Result;
use mpi::traits::CommunicatorCollectives;
use ndarray::{ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use sope::{
    bcast::bcast, collective::allgatherv_full_vec, shift::right_shift_vec,
};
use std::{cell::RefCell, fmt::Display, ops::Range};

use super::{CVConfig, GBMParams, train_with_early_stopping};
use crate::{
    anndata::{AnnData, GeneSetAD},
    comm::CommIfx,
    util::{Vec2d, block_high, block_low, block_range},
};

/// Sklearn-style K-fold splitter used by the CV loops.
///
/// Owns a (possibly shuffled) row index vector and turns it into
/// `(train_indices, val_indices)` pairs on demand via
/// [`Self::split_for`] / [`Self::split`].
struct KFold {
    /// Number of folds (matches [`CVConfig::n_folds`]).
    n_splits: usize,
    /// Permuted row indices (`0..ndata`) shared by every fold.
    indices: Vec<usize>,
}

impl KFold {
    /// Build a [`KFold`] over `0..ndata` rows split into
    /// `n_splits` folds. When `shuffle == true` the row indices are
    /// permuted so each call returns a different CV partition.
    pub fn new(ndata: usize, n_splits: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..ndata).collect();

        if shuffle {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        Self { n_splits, indices }
    }


    /// Return the `(train_indices, val_indices)` for the given `fold`.
    ///
    /// The validation slice is the fold-th block with in the `indices` and
    /// the training slice is the rest.
    pub fn split_for(&self, fold: usize) -> (Vec<usize>, Vec<usize>) {
        let val_range =
            block_range(fold as i32, self.n_splits as i32, self.indices.len());
        let (val_start, val_end) = (val_range.start, val_range.end);

        // Validation indices for this fold
        let val_indices: Vec<usize> = self.indices[val_range].to_vec();

        // Training indices (everything except validation)
        let train_indices: Vec<usize> = self.indices[..val_start]
            .iter()
            .chain(self.indices[val_end..].iter())
            .copied()
            .collect();
        (train_indices, val_indices)
    }

    /// Return the `(train, val)` splits for all `n_splits` fold.
    pub fn split(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        (0..self.n_splits).map(|x| self.split_for(x)).collect()
    }
}

/// Run K-fold CV with early stopping for one target gene and return
/// the early-stopped iteration count of every fold.
///
/// Builds a fresh shuffled [`KFold`] over `data_matrix.nrows()`,
/// then trains [`config.n_folds`](CVConfig::n_folds) boosters. 
/// Each booster uses the LightGBM JSON parameters derived from 
/// `config.params`. The early-stopping callback uses [`CVConfig::es_params`].
pub fn cross_validate_target(
    data_matrix: ArrayView2<f32>,
    label: ArrayView1<f32>,
    config: &CVConfig,
) -> Result<Vec<usize>> {
    let ndata = data_matrix.shape()[0];
    let kfold = KFold::new(ndata, config.n_folds, true);
    let splits = kfold.split();
    let mut best_iterations = Vec::new();
    let gb_params = GBMParams {
        early_stopping_rounds: config.early_stopping_rounds,
        num_iterations: config.max_rounds,
        ..config.params.clone()
    };
    let params = gb_params.as_json();
    let es_params = config.es_params();

    for (fold_idx, (train_idx, val_idx)) in splits.iter().enumerate() {
        log::info!("  Fold {}/{}", fold_idx + 1, config.n_folds);
        let result = train_with_early_stopping(
            data_matrix,
            label,
            (train_idx, val_idx),
            &params,
            &es_params,
        )?;
        best_iterations.push(result.num_iterations() as usize);
    }

    Ok(best_iterations)
}

/// Aggregated statistics over the early-stopped iteration counts
/// returned by [`cross_validate_target`] across every sampled gene
/// and fold.
///
/// Holds the raw counts, a sorted copy, and the basic order
/// statistics used to summarise the CV pass.
pub struct CVStats {
    /// Raw per-(gene, fold) iteration counts laid out as a
    /// `(n_sample_genes, n_folds)` [`Vec2d`].
    pub all_rounds: Vec2d<usize>,
    /// Same counts as a sorted flat vector (used for percentile
    /// queries and the `Range` line of [`CVStats::print`]).
    pub sorted_rounds: Vec<usize>,
    /// Mean of [`Self::all_rounds`].
    pub mean: f64,
    /// Population standard deviation of [`Self::all_rounds`].
    pub stdev: f64,
    /// Median iteration count; consumed by
    /// [`crate::gbn::infer_gb_network`] as `num_iterations`.
    pub median: usize,
    /// 25th percentile of [`Self::sorted_rounds`].
    pub p25: usize,
    /// 75th percentile of [`Self::sorted_rounds`].
    pub p75: usize,
}

impl Display for CVStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Median: {}; Mean: {:.2}; Std. dev.: {:.2}; CV: {:.2},\
                P25: {}; P75 {} Range: {} - {}]",
            self.median,
            self.mean,
            self.stdev,
            self.stdev / self.mean,
            self.p25,
            self.p75,
            self.sorted_rounds[0],
            self.sorted_rounds[self.sorted_rounds.len() - 1]
        )
    }
}

impl CVStats {
    /// Compute the order statistics of `all_rounds` and initialize
    /// [`CVStats`].
    ///
    /// NOTE:: `all_rounds` is expected to contain
    /// `n_sample_genes * n_folds` entries laid out gene-major .
    pub fn new(
        all_rounds: Vec<usize>,
        n_sample_genes: usize,
        n_folds: usize,
    ) -> Self {
        // Calculate statistics
        let mean_rounds: f64 =
            all_rounds.iter().cloned().map(|r| r as f64).sum::<f64>()
                / all_rounds.len() as f64;

        let variance = all_rounds
            .iter()
            .map(|&r| (r as f64 - mean_rounds).powi(2))
            .sum::<f64>()
            / all_rounds.len() as f64;

        let mut sorted_rounds = all_rounds.clone();
        sorted_rounds.sort_unstable();
        let median_rounds = sorted_rounds[sorted_rounds.len() / 2];
        let p25 = sorted_rounds[sorted_rounds.len() / 4];
        let p75 = sorted_rounds[3 * sorted_rounds.len() / 4];

        Self {
            all_rounds: Vec2d::new(all_rounds, n_sample_genes, n_folds),
            sorted_rounds,
            mean: mean_rounds,
            stdev: variance.sqrt(),
            median: median_rounds,
            p25,
            p75,
        }
    }

    /// Print the CV stats in a multi-line, human-readable format.
    /// The single-line `Display` impl is used for log lines.
    pub fn print(&self) {
        println!("\n=== Cross-Validation Results ===");
        println!("CV Mean rounds: {}", self.mean);
        println!("CV rounds Std dev: {:.2}", self.stdev);
        println!("CV Median CV rounds: {}", self.median);
        println!(
            "Range: {} - {}",
            self.sorted_rounds[0],
            self.sorted_rounds[self.sorted_rounds.len() - 1]
        );
    }
}

/// Single-rank cross-validation.
///
/// Randomly samples [`CVConfig::n_sample_genes`] target genes from
/// `adata`, then for each one calls [`cross_validate_target`] using
/// the TF expression matrix of `tf_set` as predictors.
/// Aggregates every per-fold iteration count, returns them  as [`CVStats`].
pub fn cv_gbm(
    adata: &AnnData,
    tf_set: &GeneSetAD<f32>,
    config: &CVConfig,
) -> Result<CVStats> {
    let mut rng = rand::rng();
    let mut var_indices: Vec<usize> = (0..adata.nvars).collect();
    var_indices.shuffle(&mut rng);
    let sampled_genes: Vec<usize> = var_indices
        .into_iter()
        .take(config.n_sample_genes)
        .collect();

    let mut per_gene_rounds: Vec<Vec<usize>> = Vec::new();
    for gid in sampled_genes.into_iter() {
        let target_gene = adata.gene_at(gid);
        let label = adata.read_gene_column(target_gene)?;

        let cv_iters = if tf_set.contains_gene(target_gene) {
            let expr_mat = tf_set.expr_matrix_sub_gene(target_gene)?;
            cross_validate_target(expr_mat.view(), label.view(), config)?
        } else {
            cross_validate_target(
                tf_set.expr_matrix_ref().view(),
                label.view(),
                config,
            )?
        };
        per_gene_rounds.push(cv_iters);
    }
    // Aggregate results across all genes and folds
    let all_rounds: Vec<usize> = per_gene_rounds
        .iter()
        .flat_map(|gene_rounds| gene_rounds.iter().copied())
        .collect();
    Ok(CVStats::new(
        all_rounds,
        config.n_sample_genes,
        config.n_folds,
    ))
}

/// Per-rank state for the distributed CV loop in [`mpi_cv_gbm`].
///
/// Includes: 
/// * Reference to a [`CVConfig`] object;
/// * the rank's slice of the global `(sampled_gene, fold)` run
///   list, expressed as a contiguous `Range<usize>` over
///   `0..n_sample_genes * n_folds` (one run per element);
/// * a [`KFold`] cache used to share splits with the previous rank 
///   so that runs straddling a rank boundary use identical row permutations.
struct DistCVConfig<'a> {
    /// Number of observations (rows of the expression matrix).
    ndata: usize,
    /// Total number of `(gene, fold)` runs in the global loop
    /// (`n_sample_genes * n_folds`); kept for diagnostics.
    _nruns: usize,
    /// Half-open run-index range owned by this rank.
    p_range: Range<usize>,
    /// Borrowed CV configuration.
    config: &'a CVConfig,
    /// `(sample_id, kfold)` pair for the sample currently being
    /// processed. Wrapped in [`RefCell`] so [`Self::fold_split_for`]
    /// can re-key it when crossing a sample boundary.
    current_sample_kfold: RefCell<(usize, KFold)>,
    /// `(sample_id, kfold)` pair for the rank's last sample. Built
    /// once at construction time and shifted to the next rank so
    /// the boundary sample is processed with identical splits on
    /// both sides.
    last_sample_kfold: (usize, KFold),
}

impl<'a> DistCVConfig<'a> {
    /// Build the per-rank state.
    ///
    /// Computes the rank's range of fold-runs  among
    /// `n_sample_genes * n_folds` possible fold-runs,
    /// constructs the rank's `last_sample` [`KFold`] , and
    /// uses [`right_shift_vec`] to pull the previous rank's last
    /// `KFold` indices into this rank's `current_sample_kfold` when
    /// the two sample IDs match.
    fn new(ndata: usize, config: &'a CVConfig, cifx: &CommIfx) -> Self {
        let nruns = config.n_sample_genes * config.n_folds;
        let prev_last_sample = if cifx.rank > 0 {
            Some(
                block_high(cifx.rank - 1, cifx.size, nruns)
                    / config.n_sample_genes,
            )
        } else {
            None
        };
        let first_sample =
            block_low(cifx.rank, cifx.size, ndata) / config.n_sample_genes;
        let last_kfold = KFold::new(ndata, config.n_folds, true);
        let prev_indices = right_shift_vec(
            if prev_last_sample.is_some_and(|x| x == first_sample) {
                &last_kfold.indices
            } else {
                &[]
            },
            cifx.comm(),
        );
        let current_kfold = if let Some(first_indices) = prev_indices
            && !first_indices.is_empty()
        {
            KFold {
                n_splits: config.n_folds,
                indices: first_indices,
            }
        } else {
            KFold::new(ndata, config.n_folds, true)
        };
        Self {
            ndata,
            _nruns: nruns,
            config,
            p_range: block_range(cifx.rank, cifx.size, nruns),
            current_sample_kfold: RefCell::new((first_sample, current_kfold)),
            last_sample_kfold: (
                block_high(cifx.rank, cifx.size, ndata) / config.n_sample_genes,
                last_kfold,
            ),
        }
    }

    /// Total number of `(gene, fold)` runs in the global loop.
    fn _n_runs(&self) -> usize {
        self._nruns
    }

    /// Map a global `run_id` to the index of its sampled gene.
    fn sample_id(&self, run_id: usize) -> usize {
        run_id / self.config.n_sample_genes
    }

    /// Map a global `run_id` to its fold index.
    fn fold_id(&self, run_id: usize) -> usize {
        run_id % self.config.n_folds
    }

    /// Half-open run-index range owned by this rank.
    fn run_range(&self) -> Range<usize> {
        self.p_range.clone()
    }

    /// Get the sample IDs corresponding to each run in `r_runs` range
    fn run_samples(&self, r_runs: Range<usize>) -> Vec<usize> {
        r_runs
            .clone()
            .map(|run_id| self.sample_id(run_id))
            .collect()
    }

    /// Unique Sample IDs touched by this rank's [`Self::run_range`], with
    /// neighbouring duplicates collapsed
    /// (the run list is gene-major so duplicates are always contiguous).
    fn run_samples_dedup(&self) -> Vec<usize> {
        let mut rgenes = self.run_samples(self.run_range());
        rgenes.dedup();
        rgenes
    }

    /// Randomly pick [`CVConfig::n_sample_genes`] indices from
    /// `0..n_genes` (without replacement). Local to the rank; use
    /// [`Self::dist_sample_genes`] for the broadcast variant.
    fn sample_genes(&self, n_genes: usize) -> Vec<usize> {
        let mut rng = rand::rng();
        let mut var_indices: Vec<usize> = (0..n_genes).collect();
        var_indices.shuffle(&mut rng);
        var_indices
            .into_iter()
            .take(self.config.n_sample_genes)
            .collect()
    }

    /// Generate the sampled-gene list on the root rank and broadcast
    /// it to every rank. Returns an [`Vec`] of length
    /// [`CVConfig::n_sample_genes`].
    fn dist_sample_genes(
        &self,
        n_genes: usize,
        mpi_ifx: &CommIfx,
    ) -> Result<Vec<usize>> {
        let mut s_genes: Vec<usize> = if mpi_ifx.rank == 0 {
            self.sample_genes(n_genes)
        } else {
            vec![0; self.config.n_sample_genes]
        };
        bcast(&mut s_genes, 0, mpi_ifx.comm())?;
        Ok(s_genes)
    }

    /// Construct the [`GBMParams`] used by every CV booster.
    fn gbm_params(&self) -> GBMParams {
        GBMParams {
            early_stopping_rounds: self.config.early_stopping_rounds,
            num_iterations: self.config.max_rounds,
            ..self.config.params.clone()
        }
    }

    /// Return the `(train, val)` split for the given global `run_id`.
    ///
    /// Reuses the cached [`KFold`] when `run_id` belongs to either
    /// the rank's `last_sample` or the currently active sample.
    /// Otherwise builds a fresh [`KFold`] and updates the
    /// `current_sample_kfold` cache.
    fn fold_split_for(&self, run_id: usize) -> (Vec<usize>, Vec<usize>) {
        let sample_id: usize = self.sample_id(run_id);
        let fold_id: usize = self.fold_id(run_id);
        let current = self.current_sample_kfold.borrow();
        let last = &self.last_sample_kfold;
        if sample_id == last.0 {
            last.1.split_for(fold_id)
        } else if sample_id == current.0 {
            current.1.split_for(fold_id)
        } else {
            self.current_sample_kfold.replace((
                sample_id,
                KFold::new(self.ndata, self.config.n_folds, true),
            ));
            self.current_sample_kfold.borrow().1.split_for(fold_id)
        }
    }

    /// Forward to [`CVConfig::es_params`].
    fn es_params(&self) -> serde_json::Value {
        self.config.es_params()
    }
}

/// Per-rank CV loop driven by a [`DistCVConfig`].
///
/// Iterates over [`DistCVConfig::run_range`], reads the matching
/// target column from `run_tgt_set` (which contains only the genes
/// touched by this rank), and trains one [`Booster`] per run with
/// [`train_with_early_stopping`] using the predictors from
/// `tf_set`. 
/// Returns the per-run early-stopped iteration counts in the same
/// order as the runs.
fn dist_cross_validate(
    tf_set: &GeneSetAD<f32>,
    run_tgt_set: &GeneSetAD<f32>,
    tgt_genes: &[usize],
    config: &DistCVConfig,
) -> Result<Vec<usize>> {
    let gb_params = config.gbm_params();
    let params = gb_params.as_json_with_seed();
    let es_params = config.es_params();

    let mut best_iterations: Vec<usize> = Vec::new();
    for ((train_idx, val_idx), tgt_id) in config.run_range().map(|run_id| {
        (
            config.fold_split_for(run_id),
            tgt_genes[config.sample_id(run_id)],
        )
    }) {
        let tgt_label = run_tgt_set.column(tgt_id)?;
        let cv_iters = if tf_set.contains(tgt_id) {
            // TODO:: Use the cache for gene_id
            let expr_mat = tf_set.expr_matrix_sub_gene_index(tgt_id)?;
            train_with_early_stopping(
                expr_mat.view(),
                tgt_label.view(),
                (&train_idx, &val_idx),
                &params,
                &es_params,
            )?
        } else {
            train_with_early_stopping(
                tf_set.expr_matrix_ref().view(),
                tgt_label.view(),
                (&train_idx, &val_idx),
                &params,
                &es_params,
            )?
        };
        best_iterations.push(cv_iters.num_iterations() as usize);
    }
    Ok(best_iterations)
}

/// Distributed cross-validation.
///
/// Block-distributes the global `(sampled_gene, fold)` runs across
/// the MPI ranks (see [`DistCVConfig`]), pre-loads the target
/// columns each rank will read into a small [`GeneSetAD`] cache,
/// runs [`dist_cross_validate`] locally, and finally
/// [`allgatherv_full_vec`]s the per-rank iteration counts before
/// wrapping them into a [`CVStats`].
pub fn mpi_cv_gbm(
    tf_set: &GeneSetAD<f32>,
    config: &CVConfig,
    mpi_ifx: &CommIfx,
) -> Result<CVStats> {
    sope::cond_info!(mpi_ifx.is_root(); "START INIT CONFIG LOAD");
    let ndata = tf_set.ann_data().nobs;
    let d_config = DistCVConfig::new(ndata, config, mpi_ifx);
    let s_genes = d_config.dist_sample_genes(tf_set.ann_data().nvars, mpi_ifx)?;
    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE INIT CONFIG LOAD");
        sope::cond_info!(mpi_ifx.is_root(); "START LOAD TARGET DATA");
    }

    // Assuming all are unique
    let run_tgt_indices: Vec<usize> = d_config
        .run_samples_dedup()
        .iter()
        .map(|x| s_genes[*x])
        .collect();
    let run_tgt_set = GeneSetAD::<f32>::from_indices(
        tf_set.ann_data(),
        &run_tgt_indices,
        tf_set.decimals(),
    )?;

    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE LOAD TARGET DATA");
        sope::cond_info!(mpi_ifx.is_root(); "START DIST CROSS VALIDATE");
    }

    let local_rounds =
        dist_cross_validate(tf_set, &run_tgt_set, &s_genes, &d_config)?;

    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE DIST CROSS VALIDATE");
        sope::cond_info!(mpi_ifx.is_root(); "START GATHER");
    }

    let all_rounds = allgatherv_full_vec(&local_rounds, mpi_ifx.comm())?;

    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE GATHER");
        sope::cond_info!(mpi_ifx.is_root(); "START STATISTICS");
    }

    sope::cond_debug!(
        mpi_ifx.is_root(); "ALL ROUNDS : {} {:?}", all_rounds.len(), all_rounds
    );
    let opt_gbm =
        CVStats::new(all_rounds, config.n_sample_genes, config.n_folds);

    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE STATISTICS");
    }

    Ok(opt_gbm)
}
