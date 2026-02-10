use anyhow::{Ok, Result};
use itertools::Itertools;

use parensnet_rs::{
    anndata::{AnnData, GeneSetAD},
    gbn::{
        CVConfig, TFOutEdge, feature_importances, cv_gbm,
        write_h5,
    },
};

//use lightgbm3::{Booster, Dataset};
//use serde_json::json;
//const G4: [&str; 4] = ["TNFRSF25", "LDLRAP1", "LINC00853", "TAL1"];
//const G10: [&str; 10] = [
//    "TNFRSF25",
//    "PLA2G5",
//    "AL031005.1",
//    "LDLRAP1",
//    "AL020997.4",
//    "SPOCD1",
//    "ZSWIM5",
//    "LINC00853",
//    "PDZK1IP1",
//    "TAL1",
//];

//fn small_booster_test(adata: &AnnData) -> Result<()> {
//    let genes8: Vec<String> = G10[..8].iter().map(|x| x.to_string()).collect();
//    let darr8 = adata.read_genes_submatrix::<f32>(&genes8)?;
//    let val_genes2: Vec<String> =
//        G10[8..].iter().map(|x| x.to_string()).collect();
//    let val_darr8 = adata.read_genes_submatrix::<f32>(&val_genes2)?;
//    let label = adata.read_column(10)?;
//
//    let gdset = Dataset::from_slice(
//        darr8.flatten().as_slice().unwrap(),
//        label.as_slice().unwrap(),
//        8,
//        true,
//    )?;
//    let params = json! {{
//        "verbose": 0,
//        "num_iterations": 100,
//        "objective": "regression",
//    }};
//
//    let booster = Booster::train(gdset, &params)?;
//    println!("BOOSTER WITHOUT VALID {}", booster.num_iterations());
//
//    let train_dset = Dataset::from_slice(
//        darr8.flatten().as_slice().unwrap(),
//        label.as_slice().unwrap(),
//        8,
//        true,
//    )?;
//    let val_dset = Dataset::from_slice_with_reference(
//        val_darr8.flatten().as_slice().unwrap(),
//        label.as_slice().unwrap(),
//        2,
//        true,
//        Some(&train_dset),
//    )?;
//    let eparams = json! {{
//        "verbose": 0,
//        "early_stopping_rounds": 10,
//        "num_iterations": 100,
//        "objective": "regression",
//        "metric": "rmse",
//    }};
//    let vbooster =
//        Booster::train_with_valid(train_dset, Some(val_dset), &eparams)?;
//    println!("BOOSTER W. VALID {}", vbooster.num_iterations());
//
//    Ok(())
//}

//fn subarray_test(adata: &AnnData) -> Result<()> {
//    println!("TEST INDEX {:?}", adata.get_gene_index("TNFRSF25"));
//    let genes: Vec<String> = G4.iter().map(|x| x.to_string()).collect();
//    let rindices = adata.get_gene_indices(&genes);
//    println!("{:?}", rindices);
//    let darray = adata.read_submatrix::<f32>(&rindices)?;
//    println!(
//        "{:?} {} {}",
//        darray.shape(),
//        darray.row(0).is_standard_layout(),
//        darray.column(0).is_standard_layout(),
//    );
//    let subdar = darray.slice(ndarray::s![..5, ..]);
//    println!("SUBARRAY {:?}", subdar);
//    let rx = darray.row(0);
//    let rsl = rx.as_slice().unwrap();
//    let cx = darray.column(0).to_owned();
//    let csl = cx.as_slice().unwrap();
//    println!("ROW/COL OWNED {:?}", (rsl.len(), csl.len()));
//
//    Ok(())
//}

fn run_gb_grn(ad_fname: &str, tf_csv: &str, out_file: &str) -> Result<()> {
    let ndecimals: usize = 3;
    let adata = AnnData::new(ad_fname, Some("_index".to_string()))?;
    let tf_set = GeneSetAD::new(&adata, tf_csv, None, Some(ndecimals))?;
    let config = CVConfig {
        n_sample_genes: 10,
        ..Default::default()
    };

    let opt_gm = cv_gbm(&adata, &tf_set, &config)?;
    opt_gm.print();

    let tgt_set = GeneSetAD::<f32>::from_indices(
        &adata,
        &((0..20usize).collect_vec()),
        Some(ndecimals),
    )?;
    let weights = feature_importances(&tf_set, &tgt_set, 0..20, &config.params)?;
    let edges = TFOutEdge::from_matrix(weights, 0, 0);

    write_h5(&adata, tf_set.indices_ref(), &edges, out_file)?;

    //let subdar = weight_matrix.slice(ndarray::s![..5, ..5]);
    //println!("SUBARRAY {:?}", subdar);
    Ok(())
}

pub fn main() -> Result<()> {
    env_logger::try_init()?;
    let fname = "/localscratch/schockalingam6/tmp/pbmc20k.500/adata.20k.500.h5ad";
    let tf_csv = "./data/pbmc/trrust_tf.txt";
    let out_file = "tmp/out_data.h5";
    //let rtest = arr2(&[[0, 1, 2], [4, 5, 6]]);
    //let rtvec: Vec<(usize, usize, i32)> =
    //    rtest.indexed_iter().map(|((i, j), k)| (i, j, *k)).collect();
    //println!("{:?}", rtvec);
    run_gb_grn(fname, tf_csv, out_file)?;
    //let fname = "/localscratch/schockalingam6/tmp/pbmc20k.5k/adata.20k.5k.h5ad";
    //let tf_csv = "./data/pbmc/trrust_tf.txt";
    //hdf5_test(fname, tf_csv)?;
    Ok(())
}
