use hdf5::H5Type;
use mpi::collective::SystemOperation;
use mpi::traits::AsRaw;
use ndarray::{Array1, Array2};
use sope::{
    partition::{self, Dist, InterleavedDist},
    reduction::{allreduce_sum, exclusive_scan},
};

use crate::{comm::CommIfx, cond_debug};

pub fn create_file(
    mcx: &CommIfx,
    fname: &str,
) -> Result<hdf5::File, hdf5::Error> {
    hdf5::File::with_options()
        .with_fapl(|fapl| {
            fapl.mpio(mcx.comm().as_raw(), None)
                .all_coll_metadata_ops(true)
                .coll_metadata_write(true)
        })
        .create(fname)
}

pub fn open_file(mcx: &CommIfx, fname: &str) -> Result<hdf5::File, hdf5::Error> {
    hdf5::File::with_options()
        .with_fapl(|fapl| {
            fapl.mpio(mcx.comm().as_raw(), None)
                .all_coll_metadata_ops(true)
                .coll_metadata_write(true)
        })
        .open(fname)
}

pub fn open_file_rw(
    fx_comm: &CommIfx,
    fname: &str,
) -> Result<hdf5::File, hdf5::Error> {
    hdf5::File::with_options()
        .with_fapl(|fapl| {
            fapl.mpio(fx_comm.comm().as_raw(), None)
                .all_coll_metadata_ops(true)
                .coll_metadata_write(true)
        })
        .open_rw(fname)
}

pub fn block_read1d_ds<T: H5Type>(
    mcx: &CommIfx,
    h_ds: &hdf5::Dataset,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array1<T>, hdf5::Error> {
    cond_debug!(mcx.rank == 0 ; "Shape :: {:?} ", h_ds.shape());

    let srange = if let Some(dist) = dist {
        dist.range()
    } else {
        InterleavedDist::new(h_ds.shape()[0], mcx.size, mcx.rank).range()
    };
    let rdata: ndarray::Array1<T> =
        h_ds.as_reader().coll_read_slice_1d(ndarray::s![srange])?;
    cond_debug!(mcx.rank == 0 ; "RShape :: {:?} ", rdata.shape());
    Ok(rdata)
}

pub fn block_read1d_grp<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dataset: &str,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array1<T>, hdf5::Error> {
    let ds = h_group.dataset(dataset)?;
    block_read1d_ds(mcx, &ds, dist)
}

pub fn block_read1d<T: H5Type>(
    mcx: &CommIfx,
    fname: &str,
    dataset: &str,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array1<T>, hdf5::Error> {
    let file = open_file(mcx, fname)?;
    let ds = file.dataset(dataset)?;
    block_read1d_ds(mcx, &ds, dist)
}

pub fn block_read2d_ds<T: H5Type>(
    mcx: &CommIfx,
    h_ds: &hdf5::Dataset,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array2<T>, hdf5::Error> {
    cond_debug!(mcx.rank == 0 ; "Shape :: {:?} ", h_ds.shape());

    let srange = if let Some(dist) = dist {
        dist.range_at(mcx.rank)
    } else {
        InterleavedDist::new(h_ds.shape()[0], mcx.size, mcx.rank).range()
    };

    let rdata: ndarray::Array2<T> =
        h_ds.as_reader().coll_read_slice_2d(ndarray::s![srange, ..])?;
    cond_debug!(mcx.rank == 0 ; "RShape :: {:?} ", rdata.shape());
    Ok(rdata)
}

pub fn block_read2d_grp<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dataset: &str,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array2<T>, hdf5::Error> {
    let ds = h_group.dataset(dataset)?;
    block_read2d_ds(mcx, &ds, dist)
}

pub fn block_read2d<T: H5Type>(
    mcx: &CommIfx,
    fname: &str,
    dataset: &str,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array2<T>, hdf5::Error> {
    //
    let file = open_file(mcx, fname)?;
    let ds = file.dataset(dataset)?;
    block_read2d_ds(mcx, &ds, dist)
}

pub fn block_write1d<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dsname: &str,
    data: &ndarray::Array1<T>,
) -> Result<(), hdf5::Error> {
    let n_data = allreduce_sum(&(data.len()), mcx.comm());
    let b_start =
        exclusive_scan(&(data.len()), mcx.comm(), SystemOperation::sum());
    let b_end = b_start + data.len();
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from(n_data))
        .create(dsname)?
        .as_writer()
        .coll_write_slice(data, ndarray::s![b_start..b_end])?;
    Ok(())
}

pub fn block_write2d<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dsname: &str,
    data: &ndarray::Array2<T>,
) -> Result<(), hdf5::Error> {
    let n_rows = allreduce_sum(&(data.nrows()), mcx.comm());
    let row_start =
        exclusive_scan(&(data.nrows()), mcx.comm(), SystemOperation::sum());
    let row_end = row_start + data.nrows();
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from([n_rows, data.ncols()]))
        .create(dsname)?
        .as_writer()
        .coll_write_slice(data, ndarray::s![row_start..row_end, ..])?;
    Ok(())
}

pub fn create_write2d<T: H5Type>(
    fx_comm: &CommIfx,
    fname: &str,
    group: &str,
    dsname: &str,
    data: &ndarray::Array2<T>,
) -> Result<(), hdf5::Error> {
    let h_file = create_file(fx_comm, fname)?;
    let h_group = h_file.create_group(group)?;
    block_write2d(fx_comm, &h_group, dsname, data)?;
    Ok(())
}

pub fn create_write1d<T: H5Type>(
    fx_comm: &CommIfx,
    fname: &str,
    group: &str,
    dsname: &str,
    data: &ndarray::Array1<T>,
) -> Result<(), hdf5::Error> {
    let h_file = create_file(fx_comm, fname)?;
    let h_group = h_file.create_group(group)?;
    block_write1d(fx_comm, &h_group, dsname, data)?;
    Ok(())
}
