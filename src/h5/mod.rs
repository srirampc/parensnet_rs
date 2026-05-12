//! HDF5 I/O wrappers used throughout `parensnet_rs`.
//!
//! The crate reads/saves matrices, arrays, and
//! intermediate data structures as HDF5 datasets. This module includes
//! thin convenience wrappers around the bundled `ext/hdf5-rust` crate
//! for opening files, building dataset shapes, and selecting slices.
//!
//! Two flavors of I/O are provided:
//!
//! * [`io`] — sequential (single-process) reads and writes used by
//!   non-MPI binaries. Each helper takes a path or an already-opened 
//!   `hdf5::Group` and returns owned `ndarray` arrays or scalar attributes.
//!
//! * [`mpio`] — collective parallel-HDF5 ("MPIO") reads and writes
//!   that go through an MPI communicator wrapper, [`crate::comm::CommIfx`].
//!   The helpers configure the file access property list with
//!   `mpio` + collective metadata, partition the data with the helpers
//!   from `sope::partition`, and use the parallel `coll_*` reader and
//!   writer methods so that every rank participates in the I/O.
//!
//! Both variants speak in terms of `ndarray::Array1` / `Array2` and
//! lean on the `hdf5::H5Type` trait.

pub mod mpio;
pub mod io;
