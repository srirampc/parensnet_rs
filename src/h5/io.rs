use crate::types::Pair;
use hdf5::H5Type;
use ndarray::{Array1, Array2};
use num::ToPrimitive;
use std::ops::Range;

pub fn create_file(fname: &str) -> Result<hdf5::File, hdf5::Error> {
    hdf5::File::create(fname)
}

pub fn read_scalar_attr<T: H5Type>(
    group: &hdf5::Group,
    name: &str,
) -> Result<T, hdf5::Error> {
    group.attr(name)?.read_scalar::<T>()
}

pub fn read_2d<T: H5Type>(
    fname: &str,
    dset_name: &str,
) -> Result<Array2<T>, hdf5::Error> {
    hdf5::File::open(fname)?.dataset(dset_name)?.read_2d()
}

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

    hdf5::File::open(fname)?
        .dataset(dset_name)?
        .read_slice_2d(ndarray::s![rbounds, cbounds])
}

pub fn read1d_slice<T: H5Type, S: ToPrimitive>(
    group: &hdf5::Group,
    name: &str,
    s_range: &Range<S>,
) -> Result<Array1<T>, hdf5::Error> {
    let urange = s_range.start.to_usize().unwrap_or(0)
        ..s_range.end.to_usize().unwrap_or(1);
    group.dataset(name)?.read_slice_1d(ndarray::s![urange])
}

pub fn read1d_point<T: Clone + H5Type, S: ToPrimitive>(
    group: &hdf5::Group,
    name: &str,
    s_idx: S,
) -> Result<T, hdf5::Error> {
    let suidx = s_idx.to_usize().unwrap_or(0);
    let h5ds = group.dataset(name)?;
    let tval: Array1<T> = h5ds.read_slice_1d(ndarray::s![suidx..suidx + 1])?;
    Ok(tval[0].clone())
}

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

pub fn write_scalar_attr<T: H5Type>(
    group: &hdf5::Group,
    name: &str,
    val: &T,
) -> Result<(), hdf5::Error> {
    group.attr(name)?.write_scalar::<T>(val)
}

pub fn write_2d<T: H5Type>(
    h_group: &hdf5::Group,
    dset_name: &str,
    data: &Array2<T>,
) -> Result<(), hdf5::Error> {
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from([data.nrows(), data.ncols()]))
        .create(dset_name)?
        .as_writer()
        .write(data)
}

pub fn write_1d<T: H5Type>(
    h_group: &hdf5::Group,
    dsname: &str,
    data: &Array1<T>,
) -> Result<(), hdf5::Error> {
    let n_data = data.len();
    h_group
        .new_dataset_builder()
        .empty::<T>()
        .shape(hdf5::Extents::from(n_data))
        .create(dsname)?
        .as_writer()
        .write(data)
}
