//! Gradient-boosted gene-regulatory-network (GRN) construction.
//!
//! This module re-implements the GRNBoost approach of Moerman et al.
//! (Arboreto, 2019) on top of [LightGBM](https://lightgbm.readthedocs.io/)
//! via the [`lightgbm3`] crate, with two extensions targeting large
//! single-cell datasets:
//!
//! 1. A **cross-validation step** ([`cv`]) that estimates a single
//!    "good enough" boosting-round count by running k-fold CV with
//!    early stopping on a randomly sampled subset of target genes.
//!    The median of the per-fold best-iteration values is used as
//!    the `num_iterations` for the production run.
//! 2. An **MPI-distributed training loop** ([`arbor`]) that block-
//!    distributes the target genes across ranks, trains one LightGBM
//!    regressor per target on the transcription-factor (TF) matrix,
//!    extracts gain-based feature importances, and gathers the
//!    resulting `(tf, target, importance)` edges into an HDF5 file.
//!
//! # Submodules
//! * [`args`] — YAML/TOML-deserializable configuration types
//!   ([`GBGRNArgs`], [`GBMParams`], [`CVConfig`], [`RunMode`]).
//! * [`train`] — thin wrappers around [`lightgbm3::Booster`] for
//!   plain training and training with early stopping, both for raw
//!   `ndarray` views and for [`AnnData`] columns.
//! * [`cv`] — k-fold cross-validation utilities ([`cv_gbm`],
//!   [`mpi_cv_gbm`], [`CVStats`]) used to pick `num_iterations`.
//! * [`arbor`] — feature-importance extraction, the per-rank
//!   target-gene loop, and HDF5 output ([`TFOutEdge`],
//!   [`mpi_gradient_boosting_grn`], [`mpi_write_h5`]).
//!
//! # Public entry points
//! * [`run_cross_fold_gbm`] — load the AnnData / TF set and run only
//!   the CV stage, returning the resulting [`CVStats`] (corresponds
//!   to [`RunMode::GBCrossFoldValidation`]).
//! * [`infer_gb_network`] — full GRN inference pipeline: optional CV
//!   step (skipped via [`GBGRNArgs::skip_cv`]) followed by the
//!   distributed gradient-boosting run and HDF5 write
//!   (corresponds to [`RunMode::GBGRNet`]).

use anyhow::{Ok, Result};
use mpi::traits::CommunicatorCollectives;

use crate::{
    anndata::{AnnData, GeneSetAD},
    comm::CommIfx,
    cond_info, cond_println,
};

mod args;
pub use args::{CVConfig, GBGRNArgs, GBMParams, RunMode};

mod train;
pub use train::{
    train, train_ad, train_with_early_stopping, train_with_early_stopping_ad,
};

mod cv;
pub use cv::{CVStats, cross_validate_target, cv_gbm, mpi_cv_gbm};

mod arbor;
pub use arbor::{
    TFOutEdge, feature_importances, feature_importances_ad,
    mpi_gradient_boosting_grn, mpi_write_h5, write_h5,
};

/// Internal helper: run [`mpi_cv_gbm`] for the supplied TF gene set
/// using a [`CVConfig`] derived from `args`.
///
/// Forces `verbose = 0` on [`GBMParams`] so the CV pass doesn't flood the log.
fn mpi_cv_gbn_for(
    tf_set: &GeneSetAD<f32>,
    args: &GBGRNArgs,
    mcx: &CommIfx,
) -> Result<CVStats> {
    let params: GBMParams = GBMParams {
        verbose: 0,
        ..args.gbm_params.clone()
    };

    let config = CVConfig {
        n_sample_genes: args.n_sample_genes,
        params,
        ..Default::default()
    };

    cond_info!(mcx.is_root(); "CV Config : {:?}", config);
    let cv_stats = mpi_cv_gbm(tf_set, &config, mcx)?;
    if log::log_enabled!(log::Level::Debug) && mcx.is_root() {
        cv_stats.print();
    }

    if log::log_enabled!(log::Level::Info) {
        mcx.comm().barrier();
    }

    cond_println!(
        mcx.is_root();
        "[{}] CV Stats: {} ",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"), cv_stats
    );
    Ok(cv_stats)
}

/// Run only the cross-validation stage of the GBN workflow and
/// return the per-target [`CVStats`].
pub fn run_cross_fold_gbm(args: &GBGRNArgs, mcx: &CommIfx) -> Result<CVStats> {
    cond_info!(mcx.is_root(); "Data H5AD : {}", args.h5ad_file);
    let adata = AnnData::new(&args.h5ad_file, Some(args.gene_id_col.clone()), None)?;
    cond_info!(mcx.is_root(); "TF File  : {}", args.tf_csv_file);
    let tf_set =
        GeneSetAD::new(&adata, &args.tf_csv_file, None, Some(args.nroundup))?;
    cond_info!(mcx.is_root(); "TF Set   : {:?}", tf_set.len());
    mpi_cv_gbn_for(&tf_set, args, mcx)
}

/// Run the full distributed GBN inference pipeline.
///
/// Pipeline:
/// 1. Load the AnnData expression file and the TF gene set.
/// 2. Decide on a boosting-round count: when
///    [`GBGRNArgs::skip_cv`] is `true`, [`GBGRNArgs::num_iterations`]
///    is used directly; otherwise [`mpi_cv_gbn_for`] is invoked and
///    its [`CVStats::median`] is used.
/// 3. Call [`mpi_gradient_boosting_grn`] (with target-side caching
///    enabled) to train one regressor per target gene and collect
///    the [`TFOutEdge`] edges on each rank.
/// 4. Reduce the per-rank edge counts for logging, then call
///    [`mpi_write_h5`] to gather and save the network into
///    [`GBGRNArgs::output_file`].
///
/// Corresponds to the [`RunMode::GBGRNet`] dispatch path.
pub fn infer_gb_network(args: &GBGRNArgs, mcx: &CommIfx) -> Result<()> {
    cond_info!(mcx.is_root(); "Data H5AD : {}", args.h5ad_file);
    let adata = AnnData::new(&args.h5ad_file, Some(args.gene_id_col.clone()), None)?;
    cond_info!(mcx.is_root(); "TF File  : {}", args.tf_csv_file);
    let tf_set =
        GeneSetAD::new(&adata, &args.tf_csv_file, None, Some(args.nroundup))?;
    cond_info!(mcx.is_root(); "TF Set   : {:?}", tf_set.len());
    let num_iterations = if args.skip_cv {
        cond_info!(
            mcx.is_root();
            "Using Default num_iterations : {}", args.num_iterations
        );
        args.num_iterations
    } else {
        let cv_stats = mpi_cv_gbn_for(&tf_set, args, mcx)?;
        cond_info!(
            mcx.is_root();
            "CV estimated num_iterations: {}", cv_stats.median
        );
        cv_stats.median
    };
    cond_info!(mcx.is_root(); "START GRAD BOOSTING", );
    let params = GBMParams {
        num_iterations,
        ..GBMParams::default()
    };
    let net_edges = mpi_gradient_boosting_grn(&tf_set, mcx, params, true)?;
    let nedges = sope::reduction::allreduce_sum(&net_edges.len(), mcx.comm());
    cond_println!(
        mcx.is_root(); "[{}] NET EDGES: {} ",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        nedges
    );

    if log::log_enabled!(log::Level::Info) {
        cond_info!(mcx.is_root(); "FINISH GRAD BOOSTING", );
        mcx.comm().barrier();
        cond_info!(mcx.is_root(); "START WRITING OUTPUT FILE", );
    }
    mpi_write_h5(
        &adata,
        tf_set.indices_ref(),
        &net_edges,
        &args.output_file,
        mcx,
    )?;
    cond_info!(mcx.is_root(); "FINISH WRITING OUTPUT FILE", );
    Ok(())
}
