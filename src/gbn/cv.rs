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

// K-Fold Cross-Validation
struct KFold {
    n_splits: usize,
    indices: Vec<usize>,
}

impl KFold {
    pub fn new(ndata: usize, n_splits: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..ndata).collect();

        if shuffle {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        Self { n_splits, indices }
    }

    //NOTE::  distributed KFold functionality is in the DistCVConfig
    //pub fn new_dist(
    //    ndata: usize,
    //    n_splits: usize,
    //    shuffle: bool,
    //    mpi_ifx: &CommIfx,
    //) -> Result<Self> {
    //    let mut indices: Vec<usize> = if mpi_ifx.rank == 0 {
    //        let kf = Self::new(ndata, n_splits, shuffle);
    //        kf.indices
    //    } else {
    //        vec![0; ndata]
    //    };
    //    bcast(&mut indices, 0, mpi_ifx.comm())?;
    //    Ok(Self { n_splits, indices })
    //}

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

    pub fn split(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        (0..self.n_splits).map(|x| self.split_for(x)).collect()
    }
}

// Cross-Validation for One Target Gene
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

    for (fold_idx, (train_idx, val_idx)) in splits.iter().enumerate() {
        log::info!("  Fold {}/{}", fold_idx + 1, config.n_folds);
        let result = train_with_early_stopping(
            data_matrix,
            label,
            (train_idx, val_idx),
            gb_params.clone(),
        )?;
        best_iterations.push(result.num_iterations() as usize);
    }

    Ok(best_iterations)
}

pub struct OptimalGBM {
    pub all_rounds: Vec2d<usize>,
    pub sorted_rounds: Vec<usize>,
    pub mean: f64,
    pub variance: f64,
    pub median: usize,
}

impl Display for OptimalGBM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Mean: {} ; Median: {}; Variance: {:.2}; Std. dev: {:.2}; Range: {} - {}]",
            self.mean,
            self.median,
            self.variance,
            self.variance.sqrt(),
            self.sorted_rounds[0],
            self.sorted_rounds[self.sorted_rounds.len() - 1]
        )
    }
}

impl OptimalGBM {
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

        Self {
            all_rounds: Vec2d::new(all_rounds, n_sample_genes, n_folds),
            sorted_rounds,
            mean: mean_rounds,
            variance,
            median: median_rounds,
        }
    }
    pub fn print(&self) {
        println!("\n=== Cross-Validation Results ===");
        println!("Mean optimal rounds: {}", self.mean);
        println!("Std dev: {:.2}", self.variance.sqrt());
        println!("Median optimal rounds: {}", self.median);
        println!(
            "Range: {} - {}",
            self.sorted_rounds[0],
            self.sorted_rounds[self.sorted_rounds.len() - 1]
        );
    }
}

pub fn optimal_gbm_iterations(
    adata: &AnnData,
    tf_set: &GeneSetAD<f32>,
    config: &CVConfig,
) -> Result<OptimalGBM> {
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
    Ok(OptimalGBM::new(
        all_rounds,
        config.n_sample_genes,
        config.n_folds,
    ))
}

struct DistCVConfig<'a> {
    ndata: usize,
    _nruns: usize,
    p_range: Range<usize>,
    config: &'a CVConfig,
    current_sample_kfold: RefCell<(usize, KFold)>,
    last_sample_kfold: (usize, KFold),
}

impl<'a> DistCVConfig<'a> {
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

    fn _n_runs(&self) -> usize {
        self._nruns
    }

    fn sample_id(&self, run_id: usize) -> usize {
        run_id / self.config.n_sample_genes
    }

    fn fold_id(&self, run_id: usize) -> usize {
        run_id % self.config.n_folds
    }

    fn run_range(&self) -> Range<usize> {
        self.p_range.clone()
    }

    fn run_samples(&self, r_runs: Range<usize>) -> Vec<usize> {
        r_runs
            .clone()
            .map(|run_id| self.sample_id(run_id))
            .collect()
    }

    fn run_samples_dedup(&self) -> Vec<usize> {
        let mut rgenes = self.run_samples(self.run_range());
        rgenes.dedup();
        rgenes
    }

    fn sample_genes(&self, n_genes: usize) -> Vec<usize> {
        let mut rng = rand::rng();
        let mut var_indices: Vec<usize> = (0..n_genes).collect();
        var_indices.shuffle(&mut rng);
        var_indices
            .into_iter()
            .take(self.config.n_sample_genes)
            .collect()
    }

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

    fn gbm_params(&self) -> GBMParams {
        GBMParams {
            early_stopping_rounds: self.config.early_stopping_rounds,
            num_iterations: self.config.max_rounds,
            ..self.config.params.clone()
        }
    }

    fn fold_split_for(&self, run_id: usize) -> (Vec<usize>, Vec<usize>) {
        let sample_id: usize = self.sample_id(run_id);
        let fold_id: usize = self.fold_id(run_id);
        if sample_id == self.last_sample_kfold.0 {
            self.last_sample_kfold.1.split_for(fold_id)
        } else if sample_id == self.current_sample_kfold.borrow().0 {
            self.current_sample_kfold.borrow().1.split_for(fold_id)
        } else {
            self.current_sample_kfold.replace((
                sample_id,
                KFold::new(self.ndata, self.config.n_folds, true),
            ));
            self.current_sample_kfold.borrow().1.split_for(fold_id)
        }
    }
}

// Cross-Validation for One Target Gene
fn dist_cross_validate(
    tf_set: &GeneSetAD<f32>,
    run_tgt_set: &GeneSetAD<f32>,
    tgt_genes: &[usize],
    config: &DistCVConfig,
) -> Result<Vec<usize>> {
    let gb_params = config.gbm_params();

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
                gb_params.clone(),
            )?
        } else {
            train_with_early_stopping(
                tf_set.expr_matrix_ref().view(),
                tgt_label.view(),
                (&train_idx, &val_idx),
                gb_params.clone(),
            )?
        };
        best_iterations.push(cv_iters.num_iterations() as usize);
    }
    Ok(best_iterations)
}

pub fn mpi_optimal_gbm_iterations(
    tf_set: &GeneSetAD<f32>,
    config: &CVConfig,
    mpi_ifx: &CommIfx,
) -> Result<OptimalGBM> {
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
        OptimalGBM::new(all_rounds, config.n_sample_genes, config.n_folds);

    if log::log_enabled!(log::Level::Info) {
        mpi_ifx.comm().barrier();
        sope::cond_info!(mpi_ifx.is_root(); "COMPLETE STATISTICS");
    }

    Ok(opt_gbm)
}
