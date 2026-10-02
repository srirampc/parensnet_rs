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
//!
//! Fold permutations are seeded deterministically per sampled gene
//! (see [`CVConfig::cv_seed`]), so splits are reproducible and
//! identical on every rank without cross-rank communication.

use anyhow::Result;
use mpi::traits::CommunicatorCollectives;
use ndarray::{ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use sope::{bcast::bcast, collective::allgatherv_full_vec};
use std::{fmt::Display, ops::Range};

use super::{CVConfig, GBMParams, train_with_early_stopping};
use crate::{
    anndata::{AnnData, GeneSetAD},
    comm::CommIfx,
    util::{Vec2d, block_range},
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

    /// Build a [`KFold`] whose row permutation is seeded
    /// deterministically from `seed`.
    ///
    /// Unlike [`Self::new`], this is reproducible: the same
    /// `(ndata, n_splits, seed)` always yields the same permutation,
    /// on any rank or thread.
    pub fn from_seed(ndata: usize, n_splits: usize, seed: u64) -> Self {
        use rand::{SeedableRng, rngs::StdRng};
        let mut indices: Vec<usize> = (0..ndata).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
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

impl Display for KFold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[splits: {}; Indices Size: {}]",
            self.n_splits,
            self.indices.len(),
        )
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

/// Mix a base seed with an index into a well-distributed `u64` seed
/// (splitmix64 finalizer). Used to derive an independent — yet fully
/// deterministic — shuffle seed per sampled gene.
fn mix_seed(base: u64, id: u64) -> u64 {
    let mut z = base ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A contiguous stretch of runs belonging to a single sampled gene.
///
/// Because the global run list is gene-major
/// (`run_id = sample_id * n_folds + fold_id`), the runs a rank owns
/// decompose into at most one partial segment for the first gene, one
/// per fully-owned gene, and at most one partial segment for the last
/// gene. Every run in a segment shares the same sampled gene, so a
/// single [`KFold`] can be built and reused across all its folds.
struct RunSegment {
    /// Index of the sampled gene (`0..n_sample_genes`).
    sample_id: usize,
    /// First fold (inclusive) of this segment within the gene.
    fstart: usize,
    /// Last fold (exclusive) of this segment within the gene.
    fend: usize,
    /// Offset of this segment's first run within the rank's
    /// [`DistCVConfig::run_range`].
    offset: usize,
}

/// Per-rank state for the distributed CV loop in [`mpi_cv_gbm`].
///
/// Holds a reference to the [`CVConfig`] plus the rank's slice of the
/// global `(sampled_gene, fold)` run list, expressed as a contiguous
/// `Range<usize>` over `0..n_sample_genes * n_folds` (one run per
/// element, gene-major).
///
/// Fold splits are derived deterministically from [`CVConfig::cv_seed`]
/// and the sampled-gene index (see [`Self::kfold_for`]), so every rank
/// computes the *same* row permutation for a given gene without any
/// cross-rank communication.
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
}

impl<'a> DistCVConfig<'a> {
    /// Build the per-rank state.
    ///
    /// Computes the rank's contiguous range of fold-runs among the
    /// `n_sample_genes * n_folds` global runs. No cross-rank
    /// communication is needed: the per-gene fold permutations are
    /// re-derived deterministically from [`CVConfig::cv_seed`].
    fn new(ndata: usize, config: &'a CVConfig, cifx: &CommIfx) -> Self {
        let nruns = config.n_sample_genes * config.n_folds;
        Self {
            ndata,
            _nruns: nruns,
            config,
            p_range: block_range(cifx.rank, cifx.size, nruns),
        }
    }

    /// Total number of `(gene, fold)` runs in the global loop.
    fn n_runs(&self) -> usize {
        self._nruns
    }

    /// Map a global `run_id` to the index of its sampled gene.
    fn sample_id(&self, run_id: usize) -> usize {
        run_id / self.config.n_folds
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

    /// Number of worker threads to use for the rank-local CV loop.
    ///
    /// A configured value of `0` means "one thread per available
    /// core"; the result is always at least `1`.
    fn n_threads(&self) -> usize {
        match self.config.n_threads {
            0 => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            n => n,
        }
    }

    /// Construct the [`GBMParams`] used by every CV booster.
    ///
    /// When the rank-local loop is multi-threaded ([`Self::n_threads`]
    /// is greater than `1`), LightGBM's own thread count is forced to
    /// `1` so the two levels of parallelism do not oversubscribe the
    /// cores.
    fn gbm_params(&self) -> GBMParams {
        let mut params = GBMParams {
            early_stopping_rounds: self.config.early_stopping_rounds,
            num_iterations: self.config.max_rounds,
            ..self.config.params.clone()
        };
        if self.n_threads() > 1 {
            params.num_threads = 1;
        }
        params
    }

    /// Deterministic shuffle seed for a sampled gene, derived from
    /// [`CVConfig::cv_seed`] and the sampled-gene index.
    fn sample_seed(&self, sample_id: usize) -> u64 {
        mix_seed(self.config.cv_seed, sample_id as u64)
    }

    /// Build the deterministic [`KFold`] for a sampled gene.
    ///
    /// The same `(cv_seed, sample_id)` yields the same permutation on
    /// every rank, so a gene's folds split consistently even when its
    /// runs straddle a rank boundary.
    fn kfold_for(&self, sample_id: usize) -> KFold {
        KFold::from_seed(
            self.ndata,
            self.config.n_folds,
            self.sample_seed(sample_id),
        )
    }

    /// Decompose this rank's [`Self::run_range`] into gene-contiguous
    /// [`RunSegment`]s, in run-ascending order.
    ///
    /// Each segment covers the folds of one sampled gene that fall
    /// inside the rank's range; `offset` places the segment's first run
    /// within the rank-local result vector. The segments therefore
    /// tile `0..run_range().len()` without gaps or overlaps, which lets
    /// [`dist_cross_validate`] scatter results by offset instead of
    /// appending in order.
    fn segments(&self) -> Vec<RunSegment> {
        let nf = self.config.n_folds;
        let p = self.run_range();
        if p.start >= p.end {
            return Vec::new();
        }

        let first_sample = p.start / nf;
        let last_sample = (p.end - 1) / nf;
        (first_sample..=last_sample)
            .map(|sample_id| {
                let gene_start = sample_id * nf;
                let fstart = p.start.max(gene_start) - gene_start;
                let fend = p.end.min(gene_start + nf) - gene_start;
                RunSegment {
                    sample_id,
                    fstart,
                    fend,
                    offset: gene_start + fstart - p.start,
                }
            })
            .collect()
    }

    /// Forward to [`CVConfig::es_params`].
    fn es_params(&self) -> serde_json::Value {
        self.config.es_params()
    }
}

impl<'a> Display for DistCVConfig<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[ndata: {}; nruns: {}; prange: ({}, {}); cfg : {:?}]",
            self.ndata,
            self.n_runs(),
            self.p_range.start,
            self.p_range.end,
            self.config,
        )
    }
}

/// Train the CV boosters for a contiguous slice of `segments` and
/// return `(offset, iterations)` pairs, one per run in the slice.
///
/// Each segment builds one deterministic [`KFold`] for its sampled gene
/// (see [`DistCVConfig::kfold_for`]) and reuses it across all of the
/// gene's folds, reads the matching target column from `run_tgt_set`
/// (which contains only the genes touched by this rank), and trains one
/// [`Booster`] per fold with [`train_with_early_stopping`].
///
/// `params` / `es_params` are precomputed by the caller so they can be
/// shared read-only across worker threads. No MPI or HDF5 calls are made
/// here, so this is safe to run on a worker thread; results are returned
/// as `(offset, iterations)` pairs so the caller can scatter them
/// without depending on thread scheduling order.
fn run_segment_chunk(
    tf_set: &GeneSetAD<f32>,
    run_tgt_set: &GeneSetAD<f32>,
    tgt_genes: &[usize],
    config: &DistCVConfig,
    params: &serde_json::Value,
    es_params: &serde_json::Value,
    segments: &[RunSegment],
) -> Result<Vec<(usize, usize)>> {
    let n_runs: usize = segments.iter().map(|s| s.fend - s.fstart).sum();
    let mut results = Vec::with_capacity(n_runs);

    for seg in segments {
        let tgt_id = tgt_genes[seg.sample_id];
        let tgt_label = run_tgt_set.column(tgt_id)?;
        // Build the fold permutation once per gene, not once per fold.
        let kfold = config.kfold_for(seg.sample_id);
        // The predictor matrix (all genes, or all TFs when the target is
        // itself a TF) is likewise fixed for the whole segment.
        let expr_mat = if tf_set.contains(tgt_id) {
            // TODO:: Use the cache for gene_id
            Some(tf_set.expr_matrix_sub_gene_index(tgt_id)?)
        } else {
            None
        };

        for fold in seg.fstart..seg.fend {
            let (train_idx, val_idx) = kfold.split_for(fold);
            let expr_view = match &expr_mat {
                Some(mat) => mat.view(),
                None => tf_set.expr_matrix_ref().view(),
            };
            let cv_iters = train_with_early_stopping(
                expr_view,
                tgt_label.view(),
                (&train_idx, &val_idx),
                params,
                es_params,
            )?;
            results.push((
                seg.offset + (fold - seg.fstart),
                cv_iters.num_iterations() as usize,
            ));
        }
    }
    Ok(results)
}

/// Per-rank CV loop driven by a [`DistCVConfig`].
///
/// Splits the rank's [`DistCVConfig::segments`] (gene-contiguous run
/// stretches) into contiguous chunks and processes them one chunk per
/// worker thread when [`CVConfig::n_threads`] is greater than one, or on
/// the calling thread otherwise. Because chunks are contiguous, each
/// sampled gene's [`KFold`] is built once per thread.
///
/// Results are scattered by run offset, so the returned counts are in
/// run-ascending order regardless of how the work was scheduled (the
/// order [`allgatherv_full_vec`] concatenates by rank). No MPI calls are
/// made in the workers, so all collectives stay on the caller's thread.
fn dist_cross_validate(
    tf_set: &GeneSetAD<f32>,
    run_tgt_set: &GeneSetAD<f32>,
    tgt_genes: &[usize],
    config: &DistCVConfig,
) -> Result<Vec<usize>> {
    let gb_params = config.gbm_params();
    let params = gb_params.as_json_with_seed();
    let es_params = config.es_params();

    let segments = config.segments();
    let n_runs = config.run_range().len();
    let mut best_iterations: Vec<usize> = vec![0; n_runs];
    if n_runs == 0 {
        return Ok(best_iterations);
    }

    // Never spawn more threads than there are segments to process.
    let n_threads = config.n_threads().min(segments.len()).max(1);

    let chunk_results: Result<Vec<Vec<(usize, usize)>>> = if n_threads == 1 {
        // Sequential fast path: no scoped threads, no extra bookkeeping.
        run_segment_chunk(
            tf_set,
            run_tgt_set,
            tgt_genes,
            config,
            &params,
            &es_params,
            &segments,
        )
        .map(|r| vec![r])
    } else {
        let chunk_size = segments.len().div_ceil(n_threads);
        let params_ref = &params;
        let es_params_ref = &es_params;
        std::thread::scope(|scope| {
            let handles: Vec<_> = segments
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        run_segment_chunk(
                            tf_set,
                            run_tgt_set,
                            tgt_genes,
                            config,
                            params_ref,
                            es_params_ref,
                            chunk,
                        )
                    })
                })
                .collect();

            let mut out = Vec::with_capacity(handles.len());
            for handle in handles {
                match handle.join() {
                    Ok(res) => out.push(res?),
                    Err(_) => anyhow::bail!("CV worker thread panicked"),
                }
            }
            Ok(out)
        })
    };

    for chunk in chunk_results? {
        for (offset, iters) in chunk {
            best_iterations[offset] = iters;
        }
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
    sope::cond_info!(mpi_ifx.is_root(); "START INIT CONFIG");
    let ndata = tf_set.ann_data().nobs;
    let d_config = DistCVConfig::new(ndata, config, mpi_ifx);
    let s_genes = d_config.dist_sample_genes(tf_set.ann_data().nvars, mpi_ifx)?;
    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE INIT CONFIG; DistCVConfig: {}", d_config);
        sope::cond_info!(mpi_ifx.is_root(); "START LOAD TARGET DATA");
    }
    sope::cond_debug!(
        mpi_ifx.is_root();
        "SGN {:?}; RRANGE {:?}", s_genes, d_config.run_range(),
    );

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
    let opt_gbm = CVStats::new(all_rounds, config.n_sample_genes, config.n_folds);

    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE STATISTICS");
    }

    Ok(opt_gbm)
}
