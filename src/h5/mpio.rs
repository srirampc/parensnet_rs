use hdf5::{H5Type};
use mpi::collective::SystemOperation;
use mpi::traits::{AsRaw, CommunicatorCollectives};
use ndarray::{Array1, Array2};

use crate::comm::CommIfx;
use crate::cond_debug;
use crate::util::block_range;

pub fn block_read1d<T: H5Type>(
    mcx: &CommIfx,
    fname: &str,
    dataset: &str,
) -> Result<Array1<T>, hdf5::Error> {
    let file = hdf5::File::with_options()
        .with_fapl(|fapl| {
            fapl.mpio(mcx.comm().as_raw(), None)
                .all_coll_metadata_ops(true)
                .coll_metadata_write(true)
        })
        .open(fname)?;
    let ds = file.dataset(dataset)?;
    cond_debug!(mcx.rank == 0 ; "Shape :: {:?} ", ds.shape());

    let srange = block_range(mcx.rank, mcx.size, ds.shape()[0]);
    let rdata: ndarray::Array1<T> =
        ds.as_reader().coll_read_slice_1d(ndarray::s![srange])?;
    cond_debug!(mcx.rank == 0 ; "RShape :: {:?} ", rdata.shape());
    Ok(rdata)
}

pub fn block_read2d<T: H5Type>(
    mcx: &CommIfx,
    fname: &str,
    dataset: &str,
) -> Result<Array2<T>, hdf5::Error> {
    //
    let file = hdf5::File::with_options()
        .with_fapl(|fapl| {
            fapl.mpio(mcx.comm().as_raw(), None)
                .all_coll_metadata_ops(true)
                .coll_metadata_write(true)
        })
        .open(fname)?;
    let ds = file.dataset(dataset)?;
    cond_debug!(mcx.rank == 0 ; "Shape :: {:?} ", ds.shape());

    let srange = block_range(mcx.rank, mcx.size, ds.shape()[0]);
    let rdata: ndarray::Array2<T> =
        ds.as_reader().coll_read_slice_2d(ndarray::s![srange, ..])?;
    cond_debug!(mcx.rank == 0 ; "RShape :: {:?} ", rdata.shape());
    Ok(rdata)
}

pub fn block_write1d<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dsname: &str,
    data: &ndarray::Array1<T>,
) -> Result<(), hdf5::Error> {
    let mut n_data: usize = 0;
    mcx.comm()
        .all_reduce_into(&(data.len()), &mut n_data, SystemOperation::sum());
    let s_range = block_range(mcx.rank, mcx.size, n_data);
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from(n_data))
        .create(dsname)?
        .as_writer()
        .coll_write_slice(data, ndarray::s![s_range])?;
    Ok(())
}

pub fn block_write2d<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dsname: &str,
    data: &ndarray::Array2<T>,
) -> Result<(), hdf5::Error> {
    let mut n_rows: usize = 0;
    mcx.comm().all_reduce_into(
        &(data.nrows()),
        &mut n_rows,
        SystemOperation::sum(),
    );

    let srange = block_range(mcx.rank, mcx.size, n_rows);
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from([n_rows, data.ncols()]))
        .create(dsname)?
        .as_writer()
        .coll_write_slice(data, ndarray::s![srange, ..])?;
    Ok(())
}

pub fn create_file(
    fx_comm: &CommIfx,
    fname: &str,
) -> Result<hdf5::File, hdf5::Error> {
    hdf5::File::with_options()
        .with_fapl(|fapl| {
            fapl.mpio(fx_comm.comm().as_raw(), None)
                .all_coll_metadata_ops(true)
                .coll_metadata_write(true)
        })
        .create(fname)
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
