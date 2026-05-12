//! Parallel HDF5 ("MPIO") helpers.
//!
//! Module includes wrappers do collective reads and writes through MPI-IO. 
//! Every function takes a [`crate::comm::CommIfx`] (MPI communicator wrapper).
//! Read functions use either a caller-supplied data partitioner from
//! `sope::partition::Dist` or fall back to an
//! [`InterleavedDist`](sope::partition::InterleavedDist) over rank 0.
//! Write functions compute global sizes and per-rank offsets via
//! `allreduce_sum`/`exclusive_scan` before collective slice writes.

use hdf5::H5Type;
use mpi::collective::SystemOperation;
use mpi::traits::AsRaw;
use ndarray::{Array1, Array2};
use sope::{
    partition::{self, Dist, InterleavedDist},
    reduction::{all_same, allreduce_sum, exclusive_scan},
};

use crate::{comm::CommIfx, cond_debug};

/// Collectively create (truncate) an HDF5 file at `fname` with the
/// MPIO file access property list configured from `mcx`.
///
/// Configures the underlying file access property list with `mpio` plus 
/// collective metadata operations so that all ranks participate in the I/O.
/// All ranks of `mcx` must call this in the same order; the resulting
/// `hdf5::File` is suitable for collective dataset creation and
/// writes.
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

/// Collectively open an existing HDF5 file `fname` for reading via
/// MPIO. Companion to [`create_file`] for read-only consumers.
pub fn open_file(mcx: &CommIfx, fname: &str) -> Result<hdf5::File, hdf5::Error> {
    hdf5::File::with_options()
        .with_fapl(|fapl| {
            fapl.mpio(mcx.comm().as_raw(), None)
                .all_coll_metadata_ops(true)
                .coll_metadata_write(true)
        })
        .open(fname)
}

/// Collectively open an existing HDF5 file `fname` for read/write
/// (in place) via MPIO.
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

/// Collectively read this rank's slice of a 1-D dataset.
///
/// `dist` selects the partition strategy that decides which slice
/// each rank reads. When `None` an
/// [`InterleavedDist`](sope::partition::InterleavedDist) over the
/// dataset's first dimension is used, which round-robins entries
/// across the `mcx.size` ranks.
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

/// Collectively read this rank's slice of a 1-D dataset given a
/// parent group and the dataset name. Convenience wrapper around
/// [`block_read1d_ds`].
pub fn block_read1d_grp<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dataset: &str,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array1<T>, hdf5::Error> {
    let ds = h_group.dataset(dataset)?;
    block_read1d_ds(mcx, &ds, dist)
}

/// Collectively open `fname` and read this rank's slice of the named
/// 1-D dataset. Convenience wrapper around [`block_read1d_ds`] that
/// also opens the file.
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

/// Collectively read the row stripe of a 2-D dataset assigned to
/// this rank.
///
/// Row range is chosen by `dist.range_at(mcx.rank)` when supplied
/// or by an interleaved distribution over the first dimension
/// otherwise. All columns are read.
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

    let rdata: ndarray::Array2<T> = h_ds
        .as_reader()
        .coll_read_slice_2d(ndarray::s![srange, ..])?;
    cond_debug!(mcx.rank == 0 ; "RShape :: {:?} ", rdata.shape());
    Ok(rdata)
}

/// Companion to [`block_read2d_ds`] with group as input: looks up `dataset`
/// inside `h_group` and forwards to it.
pub fn block_read2d_grp<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dataset: &str,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array2<T>, hdf5::Error> {
    let ds = h_group.dataset(dataset)?;
    block_read2d_ds(mcx, &ds, dist)
}

/// Companion to [`block_read2d_ds`] with file as input: opens `fname`
/// collectively and reads the requested 2-D dataset.
pub fn block_read2d<T: H5Type>(
    mcx: &CommIfx,
    fname: &str,
    dataset: &str,
    dist: Option<&dyn partition::Dist>,
) -> Result<Array2<T>, hdf5::Error> {
    let file = open_file(mcx, fname)?;
    let ds = file.dataset(dataset)?;
    block_read2d_ds(mcx, &ds, dist)
}

/// Collectively write a per-rank 1-D array `data` as one global
/// dataset `dsname` inside `h_group`.
///
/// Total length is computed via `allreduce_sum` and each rank's
/// write offset by `exclusive_scan`, so calling this function on the
/// same dataset name from every rank produces a single dataset whose
/// blocks are laid end-to-end in rank order.
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

/// Collectively write a per-rank 2-D array `data` as one global
/// dataset, stacking the row stripes from each rank.
///
/// In all ranks, `data` must have the same number of columns; the row
/// dimension is the union of every rank's `data.nrows()`.
pub fn block_write2d<T: H5Type>(
    mcx: &CommIfx,
    h_group: &hdf5::Group,
    dsname: &str,
    data: &ndarray::Array2<T>,
) -> Result<(), hdf5::Error> {
    assert!(all_same(&(data.ncols()), mcx.comm()));
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

/// Collectively create `fname`, build group `group`, and write the
/// per-rank 2-D `data` into dataset `dsname` via [`block_write2d`].
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

/// Collectively create `fname`, build group `group`, and write the
/// per-rank 1-D `data` into dataset `dsname` via [`block_write1d`].
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

/// Independent (non-collective) read of a `cbounds x rbounds`
/// rectangular sub-region of the 2-D dataset `ds_name` in the file
/// at `h5path`.
///
/// The communicator `cx` is used only to open the file in MPIO mode;
/// the slice itself is fetched with `indi_read_slice_2d` so each rank
/// can request a different region without coordinating with peers.
pub fn read_range_data<T: H5Type + Clone>(
    h5path: &str,
    ds_name: &str,
    cbounds: std::ops::Range<usize>,
    rbounds: std::ops::Range<usize>,
    cx: &CommIfx,
) -> Result<ndarray::Array2<T>, hdf5::Error> {
    let h5fptr = open_file(cx, h5path)?;
    let ds = h5fptr.dataset(ds_name)?;
    let selection = ndarray::s![cbounds, rbounds];
    let rdata: Array2<T> = ds.as_reader().indi_read_slice_2d(selection)?;
    Ok(rdata)
}

/// Same as [`read_range_data`] but returns the transposed slice.
///
/// Convenience helper for call sites that store data in a layout
/// opposite to what is on disk.
pub fn read_range_data_t<T: H5Type + Clone>(
    h5path: &str,
    ds_name: &str,
    cbounds: std::ops::Range<usize>,
    rbounds: std::ops::Range<usize>,
    cx: &CommIfx,
) -> Result<ndarray::Array2<T>, hdf5::Error> {
    let rdata = read_range_data(h5path, ds_name, cbounds, rbounds, cx)?;
    Ok(rdata.t().to_owned())
}
