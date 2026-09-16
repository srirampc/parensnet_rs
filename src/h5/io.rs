//! Sequential (single-process) HDF5 helpers.
//!
//! Module contains wrapper functions for non-MPI I/O.
//! Functions take filesystem paths or already-opened
//! `hdf5::Group` handles and return `ndarray` arrays or scalar
//! attribute values. For collective parallel I/O over MPI, see the
//! sibling module [`super::mpio`].

use crate::types::Pair;
use hdf5::H5Type;
use ndarray::{Array1, Array2};
use num::{ToPrimitive, Zero};
use std::ops::Range;

/// Create (or truncate) an HDF5 file at `fname` and return the
/// `hdf5::File` handle.
///
/// Thin wrapper around `hdf5::File::create` so call sites read the
/// same as the parallel variant in [`super::mpio::create_file`].
pub fn create_file(fname: &str) -> Result<hdf5::File, hdf5::Error> {
    hdf5::File::create(fname)
}

/// Get dataset from a group, log an error if not found
pub fn get_group_ds(
    group: &hdf5::Group,
    dset_name: &str,
) -> Result<hdf5::Dataset, hdf5::Error> {
    match group.dataset(dset_name) {
        Ok(ds) => Ok(ds),
        Err(ds_err) => {
            log::error!("Missing Dataset {}?", dset_name);
            return Err(ds_err);
        }
    }
}

/// Get dataset from a file, log an error if not found
pub fn get_dataset(
    fname: &str,
    dset_name: &str,
) -> Result<hdf5::Dataset, hdf5::Error> {
    let fptr = hdf5::File::open(fname)?;
    get_group_ds(&fptr, dset_name)
}

/// Read a single scalar attribute named `name` from `group`.
///
/// `T` may be any type that implements `H5Type` (numeric primitives,
/// strings, custom POD types, etc.).
pub fn read_scalar_attr<T: H5Type>(
    group: &hdf5::Group,
    name: &str,
) -> Result<T, hdf5::Error> {
    let h5attr = match group.attr(name) {
        Ok(ds) => ds,
        Err(ds_err) => {
            log::error!("Missing Attribute {}", name);
            return Err(ds_err);
        }
    };
    h5attr.read_scalar::<T>()
}

/// Open `fname` and read the dataset `dset_name` as an owned 1-D
/// array.
pub fn read_1d<T: H5Type>(
    fname: &str,
    dset_name: &str,
) -> Result<Array1<T>, hdf5::Error> {
    let h5ds = get_dataset(fname, dset_name)?;
    h5ds.read_1d()
}

/// Open `fname` and read the dataset `dset_name` as an owned 2-D
/// array.
pub fn read_2d<T: H5Type>(
    fname: &str,
    dset_name: &str,
) -> Result<Array2<T>, hdf5::Error> {
    let h5ds = get_dataset(fname, dset_name)?;
    h5ds.read_2d()
}

/// Read a rectangular `r_range x c_range` sub-region of a 2-D dataset.
///
/// The two ranges may use any integer-like type implementing
/// [`ToPrimitive`]; values that cannot be converted fall back to `0`
/// (start) or `1` (end), matching the conservative defaults used
/// elsewhere in the crate.
pub fn read2d_slice<T: H5Type, S: ToPrimitive>(
    fname: &str,
    dset_name: &str,
    r_range: &Range<S>,
    c_range: &Range<S>,
) -> Result<Array2<T>, hdf5::Error> {
    let rbounds = r_range.start.to_usize().unwrap_or(0)
        ..r_range.end.to_usize().unwrap_or(1);
    let cbounds = c_range.start.to_usize().unwrap_or(0)
        ..c_range.end.to_usize().unwrap_or(1);

    let h5ds = get_dataset(fname, dset_name)?;
    h5ds.read_slice_2d(ndarray::s![rbounds, cbounds])
}

/// Read a contiguous slice of a 1-D dataset that already lives inside
/// `group`.
///
/// Used when the caller has already opened the parent file/group and
/// wants to avoid re-opening it for every slice.
pub fn read1d_slice<T: H5Type, S: ToPrimitive>(
    group: &hdf5::Group,
    name: &str,
    s_range: &Range<S>,
) -> Result<Array1<T>, hdf5::Error> {
    let urange = s_range.start.to_usize().unwrap_or(0)
        ..s_range.end.to_usize().unwrap_or(1);
    let h5ds = match group.dataset(name) {
        Ok(ds) => ds,
        Err(ds_err) => {
            log::error!("Missing Dataset {}", name);
            return Err(ds_err);
        }
    };
    h5ds.read_slice_1d(ndarray::s![urange])
}

/// Read of a slice of rows: `rows x col_bounds`
/// sub-slice of the 2-D dataset `ds_name` in the given group.
pub fn read2d_slice_of_rows<T: H5Type + Clone + Zero>(
    group: &hdf5::Group,
    ds_name: &str,
    row_indices: &[usize],
    col_bounds: std::ops::Range<usize>,
) -> Result<ndarray::Array2<T>, hdf5::Error> {
    let h5ds = get_group_ds(group, ds_name)?;
    let mut r_mat = Array2::<T>::zeros([row_indices.len(), col_bounds.len()]);
    for (i, row_index) in row_indices.iter().enumerate() {
        let selection = ndarray::s![*row_index, col_bounds.clone()];
        let r_slice = h5ds.read_slice_1d(selection)?;
        r_mat.row_mut(i).assign(&r_slice);
    }
    Ok(r_mat)
}

/// Read of a slice of columns: `row_bounds x columns`
/// sub-slice of the 2-D dataset `ds_name` in the given group.
pub fn read2d_slice_of_cols<T: H5Type + Clone + Zero>(
    group: &hdf5::Group,
    ds_name: &str,
    col_indices: &[usize],
    row_bounds: std::ops::Range<usize>,
) -> Result<ndarray::Array2<T>, hdf5::Error> {
    let h5ds = get_group_ds(group, ds_name)?;
    let mut r_mat = Array2::<T>::zeros([row_bounds.len(), col_indices.len()]);
    for (i, col_index) in col_indices.iter().enumerate() {
        let selection = ndarray::s![row_bounds.clone(), *col_index];
        let r_slice = h5ds.read_slice_1d(selection)?;
        r_mat.column_mut(i).assign(&r_slice);
    }
    Ok(r_mat)
}

/// Read a single element at index `s_idx` of a 1-D dataset and return
/// it by value.
///
/// Internally this asks for the half-open slice `[s_idx, s_idx + 1)`
/// and clones the lone element out.
pub fn read1d_point<T: Clone + H5Type, S: ToPrimitive>(
    group: &hdf5::Group,
    name: &str,
    s_idx: S,
) -> Result<T, hdf5::Error> {
    let suidx = s_idx.to_usize().unwrap_or(0);
    let h5ds = match group.dataset(name) {
        Ok(ds) => ds,
        Err(ds_err) => {
            log::error!("Missing Dataset {}", name);
            return Err(ds_err);
        }
    };
    let tval: Array1<T> = h5ds.read_slice_1d(ndarray::s![suidx..suidx + 1])?;
    Ok(tval[0].clone())
}

/// Read two slices from the same 1-D dataset and return them packaged
/// in a [`Pair`].
///
/// Convenience helper used when the same dataset must supply the
/// "source" and "target" sub-vectors for a pair-wise computation.
pub fn read1d_pair_of_slices<T: H5Type, S: ToPrimitive>(
    group: &hdf5::Group,
    name: &str,
    st_ranges: &(Range<S>, Range<S>),
) -> Result<Pair<Array1<T>>, hdf5::Error> {
    Ok(Pair {
        first: read1d_slice(group, name, &st_ranges.0)?,
        second: read1d_slice(group, name, &st_ranges.1)?,
    })
}

/// Read two individual points from the same 1-D dataset and return
/// them as a [`Pair<T>`]. Companion to [`read1d_pair_of_slices`] but
/// for single-element lookups.
pub fn read1d_pair_of_points<T: Clone + H5Type, S: ToPrimitive>(
    group: &hdf5::Group,
    name: &str,
    st_indices: (S, S),
) -> Result<Pair<T>, hdf5::Error> {
    Ok(Pair::new(
        read1d_point(group, name, st_indices.0)?,
        read1d_point(group, name, st_indices.1)?,
    ))
}

/// Write a scalar attribute named `name` (with value `val`) on
/// `group`. The attribute must already exist.
pub fn write_scalar_attr<T: H5Type>(
    group: &hdf5::Group,
    name: &str,
    val: &T,
) -> Result<(), hdf5::Error> {
    group.attr(name)?.write_scalar::<T>(val)
}

/// Write a 2-D `ndarray` as a dataset named `dset_name` inside
/// `h_group`.
///
/// If a dataset with the same name already exists it is unlinked
/// first, which lets the caller overwrite stale data without manual
/// cleanup.
pub fn write_2d<T: H5Type>(
    h_group: &hdf5::Group,
    dset_name: &str,
    data: &Array2<T>,
) -> Result<(), hdf5::Error> {
    if let Result::Ok(_ds) = h_group.dataset(dset_name) {
        h_group.unlink(dset_name)?;
    }
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from([data.nrows(), data.ncols()]))
        .create(dset_name)?
        .as_writer()
        .write(data)
}

/// Write a 1-D `ndarray` as a dataset named `dsname` inside
/// `h_group`. Existing datasets with the same name are unlinked
/// before the new one is created.
pub fn write_1d<T: H5Type>(
    h_group: &hdf5::Group,
    dsname: &str,
    data: &Array1<T>,
) -> Result<(), hdf5::Error> {
    if let Result::Ok(_ds) = h_group.dataset(dsname) {
        h_group.unlink(dsname)?;
    }
    let n_data = data.len();
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from(n_data))
        .create(dsname)?
        .as_writer()
        .write(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Create a temporary HDF5 file holding a `6 x 8` i32 dataset `X`
    /// with `X[i][j] == i * 8 + j`, and return the file handle together
    /// with the in-memory reference matrix.
    fn write_test_ds(fname: &str) -> (hdf5::File, Array2<i32>) {
        let hfile = create_file(fname).unwrap();
        let mat =
            Array2::from_shape_fn((6usize, 8usize), |(i, j)| (i * 8 + j) as i32);
        write_2d(&hfile, "X", &mat).unwrap();
        (hfile, mat)
    }

    #[test]
    fn test_read2d_slice_of_rows() {
        let fname = std::env::temp_dir().join("parensnet_io_rows.h5");
        let fpath = fname.to_str().unwrap();
        let (hfile, mat) = write_test_ds(fpath);

        let row_indices = [0usize, 3, 5];
        let col_bounds = 2..6;
        let r_mat: Array2<i32> =
            read2d_slice_of_rows(&hfile, "X", &row_indices, col_bounds.clone())
                .unwrap();

        assert_eq!(r_mat.shape(), &[3, 4]);
        let expected = Array2::from_shape_fn((3usize, 4usize), |(i, j)| {
            mat[(row_indices[i], col_bounds.start + j)]
        });
        assert_eq!(r_mat, expected);

        drop(hfile);
        std::fs::remove_file(fpath).ok();
    }

    #[test]
    fn test_read2d_slice_of_cols() {
        let fname = std::env::temp_dir().join("parensnet_io_cols.h5");
        let fpath = fname.to_str().unwrap();
        let (hfile, mat) = write_test_ds(fpath);

        let col_indices = [1usize, 4, 7];
        let row_bounds = 1..5;
        let c_mat: Array2<i32> =
            read2d_slice_of_cols(&hfile, "X", &col_indices, row_bounds.clone())
                .unwrap();

        assert_eq!(c_mat.shape(), &[4, 3]);
        let expected = Array2::from_shape_fn((4usize, 3usize), |(r, c)| {
            mat[(row_bounds.start + r, col_indices[c])]
        });
        assert_eq!(c_mat, expected);

        drop(hfile);
        std::fs::remove_file(fpath).ok();
    }
}
