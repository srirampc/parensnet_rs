mod vec;

pub use self::vec::Vec2d;
use ndarray::{Array, ArrayView, Dimension};
use num::{Float, FromPrimitive, Integer, One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    ops::{AddAssign, Mul, Range},
};

#[macro_export]
macro_rules! map_to_tuple {
    // match load_data![load_fn; "a"]
    ($apply_fn: expr ; $name:literal) => {
        $apply_fn($name)
    };

    // match load_data![load_fn; "a", "b", "c"]
    ($apply_fn: expr ; $($name: literal), *) => {
        (
            $($apply_fn($name),)*
        )
    };

    // match load_data![load_fn; "a", ]
    ($apply_fn: expr ;$($name:literal,)*) => {
        $crate::map_to_tuple![$apply_fn; $($name),*]
    };
}

#[macro_export]
macro_rules! map_with_result_to_tuple {
    // match load_data![load_fn; "a"]
    ($apply_fn: expr ; $name:literal) => {
        $apply_fn($name)?
    };

    // match load_data![load_fn; "a", "b", "c"]
    ($apply_fn: expr ; $($name: literal), *) => {
        (
            $($apply_fn($name)?,)*
        )
    };

    // match load_data![load_fn; "a", ]
    ($apply_fn: expr ;$($name:literal,)*) => {
        $crate::map_to_tuple_with_result![$apply_fn; $($name),*]
    };
}


#[macro_export]
macro_rules! cond_info {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!(::log::Level::Info) {
          if $cond_expr {
            ::log::info!($($args)*)
          }
        }
    };
}

#[macro_export]
macro_rules! cond_error {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!(::log::Level::Error) {
          if $cond_expr {
            ::log::error!($($args)*)
          }
        }
    };
}

#[macro_export]
macro_rules! cond_debug {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!(::log::Level::Debug) {
          if $cond_expr {
            ::log::debug!($($args)*)
          }
        }
    };
}

#[macro_export]
macro_rules! cond_warn {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!(::log::Level::Warn) {
          if $cond_expr {
            ::log::warn!($($args)*)
          }
        }
    };
}

#[macro_export]
macro_rules! cond_println {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            println!($($args)*)
        }
    };
}

#[macro_export]
macro_rules! cond_eprintln {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            eprintln!($($args)*)
        }
    };
}

pub type RangePair<T> = (Range<T>, Range<T>);
pub type VecPair<T> = (Vec<T>, Vec<T>);

#[derive(Debug)]
pub enum Error {
    ReadFileError(String, std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ReadFileError(fname, err) => {
                write!(f, "Error reading file {fname}:: Soure Error {err}")
            }
        }
    }
}

impl std::error::Error for Error {}

pub fn read_file_to_string(input_file: &str) -> Result<String, Error> {
    use log::info;
    let data_file = std::path::PathBuf::from(input_file);
    info!("Loading from file : [{}]", data_file.display());

    std::fs::read_to_string(data_file).map_err(|err| {
        Error::ReadFileError(
            format!("I/O error while reading file {}", input_file),
            err,
        )
    })
}

pub fn halve_range<T: Integer + Copy + FromPrimitive>(
    inr: &Range<T>,
) -> RangePair<T> {
    let range_mid = (inr.end - inr.start) / (T::one() + T::one());
    (inr.start..range_mid, range_mid..inr.end)
}

pub fn half_split_ranges<T: Integer + Copy + FromPrimitive>(
    in_ranges: &[Range<T>],
) -> VecPair<Range<T>> {
    in_ranges
        .iter()
        .map(|inr| {
            let (hra, hrb) = halve_range(inr);
            (hra, hrb)
        })
        .unzip()
}

///
/// Exclusive prefix sum (i-th sum excludes i-th entry, only until i-1)
pub fn exc_prefix_sum<ItrT, T, SeqT>(in_itr: ItrT, scale: T) -> SeqT
where
    ItrT: Iterator<Item = T>,
    T: Zero + Mul<Output = T> + AddAssign + Clone,
    SeqT: FromIterator<T>,
{
    in_itr
        .scan(T::zero(), |state, x| {
            let cstate = (*state).clone() * scale.clone();
            *state += x.clone();
            Some(cstate)
        })
        .collect::<SeqT>()
}

///
/// Inclusive prefix sum (i-th sum includes i-th entry)
pub fn inc_prefix_sum<ItrT, T, SeqT>(in_itr: ItrT, scale: T) -> SeqT
where
    ItrT: Iterator<Item = T>,
    T: Zero + Mul<Output = T> + AddAssign + Clone,
    SeqT: FromIterator<T>,
{
    in_itr
        .scan(T::zero(), |state, x| {
            *state += x;
            let cstate = (*state).clone() * scale.clone();
            Some(cstate)
        })
        .collect::<SeqT>()
}

///
/// Mapping upper triangular indices of a square matrices into a flat array.
///  Example showing results for n=5:
///   i/j   0 1 2 3 4
///   0    [. 0 1 2 3]
///   1    [- . 4 5 6]
///   2    [- - . 7 8]
///   3    [- - - . 9]
///   4    [- - - - .]
///
pub fn triu_pair_to_index<T, S>(n: T, i: S, j: S) -> T
where
    T: Zero + PartialOrd + FromPrimitive + ToPrimitive,
    S: Zero + PartialOrd + ToPrimitive,
{
    debug_assert!(n > T::zero());
    debug_assert!(i >= S::zero() && i < j);

    let (nux, iux, jux) = (
        n.to_usize().unwrap(),
        i.to_usize().unwrap(),
        j.to_usize().unwrap(),
    );

    let idx = iux * nux + jux - (((iux + 2) * (iux + 1)) / 2);
    T::from_usize(idx).unwrap()
}

///
/// Reverse lookup for a flattened upper triangular indices of a square matrices
///  Example showing results for n=5:
///   i/j   0 1 2 3 4
///   0    [. 0 1 2 3]
///   1    [- . 4 5 6]
///   2    [- - . 7 8]
///   3    [- - - . 9]
///   4    [- - - - .]
///
pub fn triu_index_to_pair<T, S>(n: T, k: T) -> (S, S)
where
    T: Zero + PartialOrd + ToPrimitive,
    S: FromPrimitive,
{
    debug_assert!(n > T::zero());
    debug_assert!(k < n);

    let (un, uk) = (n.to_usize().unwrap(), k.to_usize().unwrap());
    let ctfk = uk as f64;
    let i = un
        - 2
        - ((-8.0 * ctfk + (4 * un * (un - 1) - 7) as f64).sqrt() / 2.0 - 0.5)
            as usize;
    let j = uk + i + 1 - (un * (un - 1) / 2) + ((un - i) * ((un - i) - 1) / 2);
    (S::from_usize(i).unwrap(), S::from_usize(j).unwrap())
}

//
// block distribution of data of size n among p processors
//
pub fn block_low(rank: i32, p: i32, n: usize) -> usize {
    ((rank as usize) * n) / (p as usize)
}

pub fn block_high(rank: i32, p: i32, n: usize) -> usize {
    ((((rank as usize) + 1) * n) / (p as usize)) - 1
}

pub fn block_size(rank: i32, p: i32, n: usize) -> usize {
    block_low(rank + 1, p, n) - block_low(rank, p, n)
}

pub fn block_owner(j: usize, p: i32, n: usize) -> i32 {
    (((p as usize) * ((j) + 1) - 1) / (n)) as i32
}

pub fn block_range(rank: i32, p: i32, n: usize) -> Range<usize> {
    block_low(rank, p, n)..(block_high(rank, p, n) + 1)
}

#[allow(clippy::single_range_in_vec_init)]
pub fn all_block_ranges(p: i32, n: usize) -> Vec<Range<usize>> {
    if p > 1 && n > p as usize {
        (0..p).map(|rx| block_range(rx, p, n)).collect()
    } else {
        vec![(0..n)]
    }
}

///
/// Returns the (i, j)-th block of a pair-wise distribution of n data points.
/// Includes all pairs (i, j) for i in 0..n and j in 0..n
/// Paris are distributed in p x p blocks with each block size of n/p x n/p
///
pub fn block_range_2d(i: i32, j: i32, p: i32, n: usize) -> RangePair<usize> {
    (block_range(i, p, n), block_range(j, p, n))
}

///
/// Returns the all the blocks of a pair-wise distribution of n data points.
/// Includes all pairs (i, j) for i in 0..n and j in 0..n
/// Paris are distributed in p x p blocks with each block size of n/p x n/p
///
pub fn all_block_ranges_2d(p: i32, n: usize) -> Vec2d<RangePair<usize>> {
    Vec2d::new(
        itertools::iproduct!(0..p, 0..p)
            .map(|(i, j)| block_range_2d(i, j, p, n))
            .collect(),
        p as usize,
        p as usize,
    )
}

///
/// Returns the all the upper triangular blocks of a pair-wise distribution of n data points.
/// Includes all pairs (i, j) for i in 0..n and j in 0..n
/// Pairs are distributed in p x p blocks with each block size of n/p x n/p
///
pub fn triu_block_ranges_2d(p: i32, n: usize) -> Vec<Vec<RangePair<usize>>> {
    (0..p)
        .map(|i| (i..p).map(|j| block_range_2d(i, j, p, n)).collect())
        .collect()
}

///
/// Diagonal of the indices of a square matrix n x n with offset
///
pub fn matrix_diagonal(n: usize, offset: usize) -> Vec<(usize, usize)> {
    (0..(n - offset)).map(|x| (x, x + offset)).collect()
}

///
/// Distribute pair-wise work in batches across p processes by distributing
/// across the diagonals. The distribution makes sure that each batch has
/// exactly p blocks.
/// Example below showing distribution for 5 processors running in 3 batches.
/// Batches are shown as a, b and c, starting with the diagonal and then
/// the offset diagonals in the upper triangular part of the matrix
///     [a b c c b]
///     [- a b c c]
///     [- - a b c]
///     [- - - a b]
///     [- - - - a]
///
pub fn diag_batch_distribution(p: usize) -> Vec2d<(usize, usize)> {
    let n_batches = 1 + (p / 2);
    Vec2d::new(
        (0..n_batches)
            .flat_map(|offset| {
                let mut batch = matrix_diagonal(p, offset);
                batch.extend(matrix_diagonal(p, p - offset));
                batch
            })
            .collect(),
        n_batches,
        p,
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchBlocks2D {
    pub n: usize,
    pub blocks: Vec2d<RangePair<usize>>, // p x p blocks with each one of size n/p x n/p
    pub batches: Vec2d<(usize, usize)>, // bp x p, row i blocks assigned for the batch
    pub batch_ranges: Vec2d<RangePair<usize>>,
    pub n_batches: usize,
}

impl BatchBlocks2D {
    pub fn new(n: usize, p: usize) -> Self {
        let blocks = all_block_ranges_2d(p as i32, n);
        let batches = diag_batch_distribution(p);
        let (n_batches, _) = batches.shape();
        let vbr = itertools::iproduct!(0..n_batches, 0..p)
            .map(|(bid, pid)| {
                let (prow, pcol) = batches.at(bid, pid);
                let (r_range, c_range) = blocks.at(*prow, *pcol).clone();
                if p.is_multiple_of(2) && bid == n_batches - 1 {
                    let r_mid =
                        r_range.start + ((r_range.end - r_range.start) / 2);
                    if pid * 2 < p {
                        (r_range.start..r_mid, c_range)
                    } else {
                        (r_mid..r_range.end, c_range)
                    }
                } else {
                    (r_range, c_range)
                }
            })
            .collect();
        let batch_ranges = Vec2d::new(vbr, n_batches, p);

        BatchBlocks2D {
            n,
            blocks,
            batches,
            batch_ranges,
            n_batches,
        }
    }
}

///
/// Unique Elements
///
#[derive(Debug)]
pub struct UniqCounts<T, S> {
    pub values: Vec<T>,
    pub counts: Vec<S>,
}

pub fn unique<T, S>(srt_data: &[T]) -> UniqCounts<T, S>
where
    T: PartialEq + Clone + Debug + Display,
    S: Zero + One + AddAssign + Clone + Debug + Display,
{
    let mut values = srt_data.to_vec();
    values.dedup();
    let nv = values.len();

    // all unique
    if nv == srt_data.len() {
        UniqCounts {
            values,
            counts: vec![S::one(); nv],
        }
    } else {
        // count unique data
        let mut counts = vec![S::zero(); nv];
        let mut ix = 0;
        counts[ix] = S::one();
        for stx in 1..srt_data.len() {
            if srt_data[stx] != srt_data[stx - 1] {
                ix += 1;
            }
            counts[ix] += S::one();
        }
        UniqCounts { values, counts }
    }
}

pub fn around<A, D>(in_data: ArrayView<A, D>, decimals: usize) -> Array<A, D>
where
    A: Float + FromPrimitive,
    D: Dimension,
{
    match A::from_f64(10.0) {
        Some(aften) => {
            let round_factor: A = num::Float::powi(aften, decimals as i32);
            in_data.map(|vx| (*vx * round_factor).round() / round_factor)
        }
        None => in_data.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use log::{info, debug};
    #[test]
    pub fn test_around() {
        use super::around;
        crate::tests::log_init();
        let va = Array1::<f32>::from_vec(vec![
            7.7659, 4.4812, 8.3781, 3.1042, 1.6313, 1.5413, 2.8511, 5.3320,
            9.6224, 2.8369, 4.6207, 3.8657, 3.2937, 4.8751, 0.6236, 6.7702,
            5.4186, 8.9017, 4.7538, 2.1902, 2.3579, 5.4603, 9.2659, 5.7142,
            1.3616,
        ]);
        let vb = Array1::<f64>::from_vec(vec![
            10.4682, 5.6456, 16.0329, 5.3554, 11.1620, 6.6694, 5.1717, 6.5787,
            9.9559, 11.0062, 4.7483, 10.6010, 5.8069, 12.0569, 3.4808, 11.9356,
            12.1310, 10.1458, 12.7421, 9.8602, 5.7607, 9.8992, 17.2601, 11.3510,
            9.5717,
        ]);

        let expected_nrda = Array1::<f32>::from_vec(vec![
            7.77, 4.48, 8.38, 3.10, 1.63, 1.54, 2.85, 5.33, 9.62, 2.84, 4.62,
            3.87, 3.29, 4.88, 0.62, 6.77, 5.42, 8.90, 4.75, 2.19, 2.36, 5.46,
            9.27, 5.71, 1.36,
        ]);
        let expected_nrdb = Array1::<f64>::from_vec(vec![
            10.47, 5.65, 16.03, 5.36, 11.16, 6.67, 5.17, 6.58, 9.96, 11.01, 4.75,
            10.60, 5.81, 12.06, 3.48, 11.94, 12.13, 10.15, 12.74, 9.86, 5.76,
            9.90, 17.26, 11.35, 9.57,
        ]);

        let nrdva = around(va.view(), 2);
        let nrdvb = around(vb.view(), 2);
        info!("NRDA {:8.4}", nrdva);
        info!("NRDB {:8.4}", nrdvb);

        assert_eq!(nrdva, expected_nrda);
        assert_eq!(nrdvb, expected_nrdb);
    }

    #[test]
    pub fn test_unique() {
        use super::{UniqCounts, unique};
        use num::traits::float::TotalOrder;

        crate::tests::log_init();
        let test_data = vec![
            7.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0, 5.0, 9.0, 2.0, 4.0, 3.0, 3.0, 4.0,
            0.0, 6.0, 5.0, 8.0, 4.0, 2.0, 2.0, 5.0, 9.0, 5.0, 1.0,
        ];
        let test_data2 = vec![
            7.0, 4.0, 8.0, 3.0, 1.0, 1.0, 2.0, 5.0, 9.0, 2.0, 4.0, 3.0, 3.0, 4.0,
            0.0, 6.0, 5.0, 8.0, 4.0, 2.0, 2.0, 5.0, 9.0, 5.0, 1.0, 10.0, 12.0,
        ];

        let mut srt_data = test_data.to_vec();
        srt_data.sort_by(TotalOrder::total_cmp);

        let mut srt_data2 = test_data2.to_vec();
        srt_data2.sort_by(TotalOrder::total_cmp);

        let result: UniqCounts<f64, i32> = unique(&srt_data);
        assert_eq!(
            result.values,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(result.counts, vec![1, 3, 4, 3, 4, 4, 1, 1, 2, 2]);
        debug!("Unique Results :: ");
        debug!("  -> sorted {:?}", Array1::from_vec(srt_data));
        debug!("  -> values {:?}", Array1::from_vec(result.values));
        debug!("  -> counts {:?}", Array1::from_vec(result.counts));

        let result2: UniqCounts<f64, f32> = unique(&srt_data2);
        assert_eq!(
            result2.values,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
        );
        assert_eq!(
            result2.counts,
            vec![1.0, 3.0, 4.0, 3.0, 4.0, 4.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]
        );

        debug!("Unique Results 2 :: ");
        debug!("  -> sorted {:?}", Array1::from_vec(srt_data2));
        debug!("  -> values {:?}", Array1::from_vec(result2.values));
        debug!("  -> counts {:?}", Array1::from_vec(result2.counts));
    }
}
