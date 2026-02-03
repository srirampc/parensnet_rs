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

pub fn run_cross_fold_gbm(args: &GBGRNArgs, mcx: &CommIfx) -> Result<CVStats> {
    cond_info!(mcx.is_root(); "Data H5AD : {}", args.h5ad_file);
    let adata = AnnData::new(&args.h5ad_file, Some(args.gene_id_col.clone()))?;
    cond_info!(mcx.is_root(); "TF File  : {}", args.tf_csv_file);
    let tf_set =
        GeneSetAD::new(&adata, &args.tf_csv_file, None, Some(args.nroundup))?;
    cond_info!(mcx.is_root(); "TF Set   : {:?}", tf_set.len());
    mpi_cv_gbn_for(&tf_set, args, mcx)
}

pub fn infer_gb_network(args: &GBGRNArgs, mcx: &CommIfx) -> Result<()> {
    cond_info!(mcx.is_root(); "Data H5AD : {}", args.h5ad_file);
    let adata = AnnData::new(&args.h5ad_file, Some(args.gene_id_col.clone()))?;
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
