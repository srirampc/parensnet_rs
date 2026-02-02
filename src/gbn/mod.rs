use anyhow::{Ok, Result};
use mpi::traits::{CommunicatorCollectives};

use crate::{
    anndata::{AnnData, GeneSetAD},
    comm::CommIfx,
    cond_info,
};

mod args;
pub use args::{CVConfig, GBGRNArgs, GBMParams};

mod train;
pub use train::{
    train, train_ad, train_with_early_stopping, train_with_early_stopping_ad,
};

mod cv;
pub use cv::{
    OptimalGBM, cross_validate_target, mpi_optimal_gbm_iterations,
    optimal_gbm_iterations,
};

mod arbor;
pub use arbor::{
    TFOutEdge, feature_importances, feature_importances_ad,
    mpi_gradient_boosting_grn, mpi_write_h5, write_h5,
};

pub fn infer_gb_network(args: &GBGRNArgs, mcx: &CommIfx) -> Result<()> {
    cond_info!(mcx.is_root(); "Data H5AD : {}", args.h5ad_file);
    let adata = AnnData::new(&args.h5ad_file, Some(args.gene_id_col.clone()))?;
    let params: GBMParams = GBMParams {
        verbose: 0,
        ..args.gbm_params.clone()
    };

    let config = CVConfig {
        n_sample_genes: args.n_sample_genes,
        params: params.clone(),
        ..Default::default()
    };

    cond_info!(mcx.is_root(); "TF File  : {}", args.tf_csv_file);
    let tf_set =
        GeneSetAD::new(&adata, &args.tf_csv_file, None, Some(args.nroundup))?;
    cond_info!(mcx.is_root(); "TF Set   : {:?}", tf_set.len());
    cond_info!(mcx.is_root(); "CV Config : {:?}", config);
    let opt_args = mpi_optimal_gbm_iterations(&tf_set, &config, mcx)?;
    if log::log_enabled!(log::Level::Debug) && mcx.is_root() {
        opt_args.print();
    }

    if log::log_enabled!(log::Level::Info) {
        cond_info!(mcx.is_root(); "Optimal GBM: {} ", opt_args);
        mcx.comm().barrier();
        cond_info!(mcx.is_root(); " START GRAD BOOSTING", );
    }
    let params = GBMParams {
        num_iterations: opt_args.median,
        ..params
    };
    let net_edges = mpi_gradient_boosting_grn(&tf_set, mcx, params, true)?;
    let nedges = sope::reduction::allreduce_sum(&net_edges.len(), mcx.comm());
    cond_info!(mcx.is_root(); "NET EDGES: {} ", nedges);

    if log::log_enabled!(log::Level::Info) {
        mcx.comm().barrier();
        cond_info!(mcx.is_root(); " START WRITING OUTPUT FILE", );
    }
    mpi_write_h5(
        &adata,
        tf_set.indices_ref(),
        &net_edges,
        &args.output_file,
        mcx,
    )?;
    cond_info!(mcx.is_root(); " COMPLETE WRITING OUTPUT FILE", );
    Ok(())
}
