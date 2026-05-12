//! General-purpose utility module for `parensnet_rs`.
//!
//! This module collects assorted helpers used throughout the crate:
//!
//! * Small declarative macros for building tuples from a function and a
//!   list of literals (`map_to_tuple!`, `map_with_result_to_tuple!`) and
//!   for conditional logging (`cond_info!`, `cond_error!`,
//!   `cond_debug!`, `cond_warn!`, `cond_println!`, `cond_eprintln!`).
//! * Range arithmetic helpers ([`top_half_range`], [`bottom_half_range`],
//!   [`halve_range`], [`half_split_ranges`]).
//! * Prefix sum iterators ([`exc_prefix_sum`], [`inc_prefix_sum`]).
//! * Conversions between flat indices and `(i, j)` pairs of an upper
//!   triangular matrix ([`triu_pair_to_index`], [`triu_index_to_pair`]).
//! * Block distribution helpers in 1D and 2D ([`block_low`],
//!   [`block_high`], [`block_size`], [`block_owner`], [`block_range`],
//!   [`all_block_ranges`], [`block_range_2d`], [`all_block_ranges_2d`],
//!   [`triu_block_ranges_2d`], [`triu_block_ranges_2d_no_diag`],
//!   [`diag_block_ranges`], [`matrix_diagonal`],
//!   [`diag_batch_distribution`]).
//! * Pair-work distribution structures ([`BatchBlocks2D`],
//!   [`DiagBatchBlocks2D`], [`SeqBatchBlocks2D`], [`EBBlocks2D`],
//!   [`PairWorkDistributor`]).
//! * Numerical helpers ([`unique`], [`UniqCounts`], [`around`]).
//! * Index/value result containers ([`IdVResults`]).
//! * I/O helpers ([`read_file_to_string`], [`read_csv_column`]).
//! * The 2D matrix wrapper [`Vec2d`].
//!
use anyhow::{Result, bail};
use itertools::iproduct;
use ndarray::{Array, Array1, Array2, ArrayView, Dimension};
use num::{Float, FromPrimitive, Integer, One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    ops::{AddAssign, Mul, Range},
};
use thiserror::Error;

mod vec;
pub use self::vec::Vec2d;

/// Apply a callable to a comma separated list of string literals and
/// return the results as a tuple.
///
/// Useful for invoking a loader once per named input. With a single
/// literal the macro returns the bare value (not a 1-tuple).
///
/// ```ignore
/// // (load("a"), load("b"), load("c"))
/// let triple = map_to_tuple!(load; "a", "b", "c");
/// ```
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

/// Like [`map_to_tuple!`] but `?`-propagates a `Result` from each call.
///
/// Each invocation of `$apply_fn` is followed by `?`, so the macro must
/// be expanded inside a function returning a compatible `Result`/`Option`.
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

/// Emit a `log::info!` message, but only when `$cond_expr` is true and
/// the `Info` log level is enabled. Saves the cost of formatting when
/// either the level is disabled or the caller-supplied condition is
/// false.
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

/// Emit a `log::error!` message gated by `$cond_expr` and the `Error`
/// log level. See [`cond_info!`] for rationale.
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

/// Emit a `log::debug!` message gated by `$cond_expr` and the `Debug`
/// log level. See [`cond_info!`] for rationale.
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

/// Emit a `log::warn!` message gated by `$cond_expr` and the `Warn`
/// log level. See [`cond_info!`] for rationale.
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

/// Run `println!` only when `$cond_expr` is true. Useful for terse
/// rank-gated output in MPI runs (e.g. only print on rank 0).
#[macro_export]
macro_rules! cond_println {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            println!($($args)*)
        }
    };
}

/// Run `eprintln!` only when `$cond_expr` is true. Companion to
/// [`cond_println!`] for stderr output.
#[macro_export]
macro_rules! cond_eprintln {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            eprintln!($($args)*)
        }
    };
}

/// A pair of `Range<T>` describing a 2-D rectangular region; the first
/// element is the row range and the second is the column range.
pub type RangePair<T> = (Range<T>, Range<T>);
/// A pair of `Vec<T>` typically used to return two parallel sequences
/// from an unzipped iterator.
pub type VecPair<T> = (Vec<T>, Vec<T>);

/// Errors that can be produced by helpers in this module.
#[derive(Error, Debug)]
pub enum UtilError {
    /// Wraps a `std::io::Error` raised while reading a plain file.
    #[error("Error reading file {0}:: Soure Error {1}")]
    ReadFileError(String, std::io::Error),
    /// Wraps a CSV-specific failure as a string (the underlying CSV
    /// error is converted to a message before being stored here).
    #[error("Error reading csv file {0}:: Soure Error {1}")]
    CSVReadFileError(String, String),
    /// Returned by [`read_csv_column`] when the requested column name
    /// is not present in the CSV header.
    #[error("CSV Input doesn't have column {0}")]
    MissingColumnError(String),
}

/// Read the entire contents of `input_file` as a UTF-8 string.
///
/// Logs the file path at `info` level. On failure, the underlying I/O
/// error is wrapped in [`UtilError::ReadFileError`] and propagated via
/// `anyhow`.
pub fn read_file_to_string(input_file: &str) -> Result<String> {
    use log::info;
    let data_file = std::path::PathBuf::from(input_file);
    info!("Loading from file : [{}]", data_file.display());

    match std::fs::read_to_string(data_file) {
        Ok(contents) => Ok(contents),
        Err(err) => bail!(UtilError::ReadFileError(
            format!("I/O error while reading file {}", input_file),
            err,
        )),
    }
}

/// Return the lower (first) half of the input range, `inr`.
///
/// For `inr = a..b` the result is `a..a + (b - a) / 2`. Integer
/// division means the lower half is the smaller half when `b - a` is
/// odd.
pub fn top_half_range<T: Integer + Copy + FromPrimitive>(
    inr: &Range<T>,
) -> Range<T> {
    let range_mid = (inr.end - inr.start) / (T::one() + T::one());
    inr.start..inr.start + range_mid
}

/// Return the upper (second) half of the input range, `inr`.
///
/// Complement of [`top_half_range`]. For `inr = a..b` the result is
/// `a + (b - a) / 2 .. b`.
pub fn bottom_half_range<T: Integer + Copy + FromPrimitive>(
    inr: &Range<T>,
) -> Range<T> {
    let range_mid = (inr.end - inr.start) / (T::one() + T::one());
    inr.start + range_mid..inr.end
}

/// Split `inr` into a `(lower, upper)` [`RangePair`] at its midpoint.
///
/// Equivalent to calling [`top_half_range`] and [`bottom_half_range`]
/// together. The midpoint uses integer division.
pub fn halve_range<T: Integer + Copy + FromPrimitive>(
    inr: &Range<T>,
) -> RangePair<T> {
    let range_mid = (inr.end - inr.start) / (T::one() + T::one());
    let inr_mid = inr.start + range_mid;
    (inr.start..inr_mid, inr_mid..inr.end)
}

/// Apply [`halve_range`] to every element of `in_ranges` and unzip the
/// results into `(lower_halves, upper_halves)`.
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

/// Exclusive prefix sum (i-th sum excludes i-th entry, only until i-1)
/// with the running sum  uniformly scaled with `scale`.
///
/// Iterates `in_itr`, multiplying the running total by `scale` before
/// emitting each element. 
/// The output sequence satisfies `out[i] = scale * sum(in[0..i])`. 
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

/// Inclusive prefix sum (i-th sum includes i-th entry) with the running sum
/// uniformly scaled with `scale`.
///
/// The output sequence satisfies `out[i] = scale * sum(in[0..=i])`. 
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

/// Map an entry of upper triangular position `(i, j)` of an `n x n` square
/// matrix to its index, when laid out as a flat 1-D array, row-by-row.
///
/// Only entries with `i < j` (i.e. above the diagonal) are valid, and
/// the diagonal itself is not included in the enumeration.
///  Example showing results for of square matrix of side, n=5:
///   i/j   0 1 2 3 4
///   0    [. 0 1 2 3]
///   1    [- . 4 5 6]
///   2    [- - . 7 8]
///   3    [- - - . 9]
///   4    [- - - - .]
pub fn triu_pair_to_index<T, S>(n: T, i: S, j: S) -> T
where
    T: Zero + PartialOrd + FromPrimitive + ToPrimitive,
    S: Zero + PartialOrd + ToPrimitive,
{
    debug_assert!(n > T::zero());
    debug_assert!(i >= S::zero());
    debug_assert!(j >= S::zero());
    debug_assert!(i < j);

    let (nux, iux, jux) = (
        n.to_usize().unwrap(),
        i.to_usize().unwrap(),
        j.to_usize().unwrap(),
    );

    debug_assert!(iux < nux);
    debug_assert!(jux < nux);

    let idx = iux * nux + jux - (((iux + 2) * (iux + 1)) / 2);
    T::from_usize(idx).unwrap()
}

/// Reverse lookup for a 1-D flattened upper triangular matrix to its
/// 2-D index in the `n x n` matrix.
///
/// Inverse of [`triu_pair_to_index`]; given a flat index `k` returns
/// the `(i, j)` (with `i < j`) it corresponds to.
///  Example showing results for n=5:
///   i/j   0 1 2 3 4
///   0    [. 0 1 2 3]
///   1    [- . 4 5 6]
///   2    [- - . 7 8]
///   3    [- - - . 9]
///   4    [- - - - .]
pub fn triu_index_to_pair<T, S>(n: T, k: T) -> (S, S)
where
    T: Zero + PartialOrd + ToPrimitive,
    S: FromPrimitive,
{
    let (un, uk) = (n.to_i64().unwrap(), k.to_i64().unwrap());
    debug_assert!(un > 0);
    debug_assert!(uk < (un * (un - 1)) / 2);

    let ctfk = uk as f64;
    let i = un
        - 2
        - ((-8.0 * ctfk + (4 * un * (un - 1) - 7) as f64).sqrt() / 2.0 - 0.5)
            as i64;
    let j = uk + i + 1 - (un * (un - 1) / 2) + ((un - i) * ((un - i) - 1) / 2);
    (S::from_i64(i).unwrap(), S::from_i64(j).unwrap())
}

// NOTE::
// Block distribution functions of data of size `n` among `p` processors.
// These follow the standard "Quinn block decomposition": when `n` is
// not divisible by `p`, the first `n mod p` ranks own one extra
// element so the total is exactly `n`.

/// First (inclusive) global index owned by `rank` in a block
/// distribution of `n` elements over `p` processors.
pub fn block_low(rank: i32, p: i32, n: usize) -> usize {
    ((rank as usize) * n) / (p as usize)
}

/// Last (inclusive) global index owned by `rank` in a block
/// distribution of `n` elements over `p` processors.
pub fn block_high(rank: i32, p: i32, n: usize) -> usize {
    ((((rank as usize) + 1) * n) / (p as usize)) - 1
}

/// Number of elements owned by `rank` (i.e. `block_high - block_low + 1`).
pub fn block_size(rank: i32, p: i32, n: usize) -> usize {
    block_low(rank + 1, p, n) - block_low(rank, p, n)
}

/// Inverse lookup: rank that owns the global index `j`.
pub fn block_owner(j: usize, p: i32, n: usize) -> i32 {
    (((p as usize) * ((j) + 1) - 1) / (n)) as i32
}

/// Half-open `Range<usize>` of indices owned by `rank`.
pub fn block_range(rank: i32, p: i32, n: usize) -> Range<usize> {
    block_low(rank, p, n)..(block_high(rank, p, n) + 1)
}

/// Vector of [`block_range`] for all `p` ranks. If `p <= 1` or
/// `n <= p`, returns the single range `0..n`.
#[allow(clippy::single_range_in_vec_init)]
pub fn all_block_ranges(p: i32, n: usize) -> Vec<Range<usize>> {
    if p > 1 && n > p as usize {
        (0..p).map(|rx| block_range(rx, p, n)).collect()
    } else {
        vec![(0..n)]
    }
}

/// Returns the (i, j)-th block of a pair-wise distribution of n data points.
///
/// Includes all pairs (i, j) for i in 0..n and j in 0..n
/// Paris are distributed in p x p blocks with each block size of n/p x n/p
pub fn block_range_2d(i: i32, j: i32, p: i32, n: usize) -> RangePair<usize> {
    (block_range(i, p, n), block_range(j, p, n))
}

/// Returns the all the blocks of a pair-wise distribution of n data points.
///
/// Includes all pairs (i, j) for i in 0..n and j in 0..n
/// Paris are distributed in p x p blocks with each block size of n/p x n/p
pub fn all_block_ranges_2d(p: i32, n: usize) -> Vec2d<RangePair<usize>> {
    Vec2d::new(
        itertools::iproduct!(0..p, 0..p)
            .map(|(i, j)| block_range_2d(i, j, p, n))
            .collect(),
        p as usize,
        p as usize,
    )
}

/// Returns the all the upper triangular blocks of a pair-wise distribution of 
/// n data points.
///
/// Includes all pairs (i, j) for i in 0..n and j in 0..n
/// Pairs are distributed in p x p blocks with each block size of n/p x n/p
/// Returns a tuple for sparse representation:
/// (flat vector of all block ranges, counts for each row)
pub fn triu_block_ranges_2d(
    p: i32,
    n: usize,
) -> (Vec<RangePair<usize>>, Vec<usize>) {
    (
        (0..p)
            .flat_map(|i| (i..p).map(move |j| block_range_2d(i, j, p, n)))
            .collect(),
        (1..(p as usize + 1)).rev().collect(),
    )
}

/// Returns the all the upper triangular blocks of a pair-wise distribution
/// of n data points.
///
/// Includes all pairs (i, j) for i in 0..n and j in 0..n
/// Pairs are distributed in p x p blocks with each block size of n/p x n/p
/// Returns a tuple for sparse representation:
/// (flat vector of all block ranges, counts for each row)
pub fn triu_block_ranges_2d_no_diag(
    p: i32,
    n: usize,
) -> (Vec<RangePair<usize>>, Vec<usize>) {
    (
        (0..p)
            .flat_map(|i| (i + 1..p).map(move |j| block_range_2d(i, j, p, n)))
            .collect(),
        (1..p as usize).rev().collect(),
    )
}

/// Block ranges along the main diagonal of the `p x p` block grid:
/// `[(R_0, C_0), (R_1, C_1), ..., (R_{p-1}, C_{p-1})]` with
/// `R_i == C_i == block_range(i, p, n)`.
pub fn diag_block_ranges(p: i32, n: usize) -> Vec<RangePair<usize>> {
    (0..p).map(|j| block_range_2d(j, j, p, n)).collect()
}

/// Diagonal of the indices of a square matrix n x n with offset.
///
/// `offset = 0` returns the main diagonal `[(0,0), (1,1), ...]`;
/// `offset = k > 0` returns `[(0, k), (1, k+1), ...]` of length
/// `n - k`. 
/// Used by [`diag_batch_distribution`] to enumerate the
/// off-diagonals of the per-batch assignment.
pub fn matrix_diagonal(n: usize, offset: usize) -> Vec<(usize, usize)> {
    (0..(n - offset)).map(|x| (x, x + offset)).collect()
}

/// Distribute pair-wise work in batches across p processes by distributing
/// across the diagonals. 
///
/// The distribution makes sure that each batch has exactly p blocks.
/// Example below showing distribution for 5 processors running in 3 batches.
/// Batches are shown as a, b and c, starting with the diagonal and then
/// the offset diagonals in the upper triangular part of the matrix
///     [a b c c b]
///     [- a b c c]
///     [- - a b c]
///     [- - - a b]
///     [- - - - a]
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

/// Common interface for 2-D batched block distributions.
///
/// Implementors decide how the `n x n` pair-wise work is partitioned
/// into a sequence of "batches" so that, in every batch, each of the
/// `p` participating ranks receives one rectangular sub-block.
pub trait BatchBlocks2D {
    /// Side length `n` of the square problem.
    fn dim(&self) -> usize;
    /// Number of sequential batches required to cover all the work.
    fn num_batches(&self) -> usize;
    /// Sub-block (row range, column range) assigned to `rank` in
    /// batch `bidx`.
    fn batch_range(&self, bidx: usize, rank: i32) -> &RangePair<usize>;
}

/// Pair-wise work distribution that schedules diagonals of the `p x p`
/// block grid in successive batches.
///
/// Built by [`diag_batch_distribution`]: batch 0 takes the main
/// diagonal, batch 1 the first off-diagonal pair, and so on. When `p`
/// is even the final batch only contains half-blocks (handled by the
/// constructor below) so total work stays balanced.
/// Example below showing distribution for 5 processors running in 3 batches.
/// Batches are shown as a, b and c, starting with the diagonal and then
/// the offset diagonals in the upper triangular part of the matrix
///     [a b c c b]
///     [- a b c c]
///     [- - a b c]
///     [- - - a b]
///     [- - - - a]
#[derive(Debug, Serialize, Deserialize)]
pub struct DiagBatchBlocks2D {
    /// Side length of the square problem.
    n: usize,
    /// `p x p` block grid produced by [`all_block_ranges_2d`]; each
    /// entry is roughly an `n/p x n/p` sub-block.
    blocks: Vec2d<RangePair<usize>>,
    /// `bp x p` mapping `[batch][rank] -> (row_block, col_block)`
    /// where `bp = num_batches`.
    batches: Vec2d<(usize, usize)>,
    /// `bp x p` resolved sub-block ranges, possibly halved on the
    /// final batch when `p` is even.
    batch_ranges: Vec2d<RangePair<usize>>,
    /// Number of batches.
    n_batches: usize,
}

impl BatchBlocks2D for DiagBatchBlocks2D {
    fn dim(&self) -> usize {
        self.n
    }

    fn num_batches(&self) -> usize {
        self.n_batches
    }

    fn batch_range(&self, bidx: usize, rank: i32) -> &RangePair<usize> {
        self.batch_ranges.at(bidx, rank as usize)
    }
}

impl DiagBatchBlocks2D {
    /// Construct the diagonal batch distribution for an `n x n`
    /// problem partitioned across `p` ranks. When `p` is even the
    /// last batch is split in halves to avoid double-counting work.
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

        DiagBatchBlocks2D {
            n,
            blocks,
            batches,
            batch_ranges,
            n_batches,
        }
    }
}

/// Pair-wise work distribution where each rank processes its assigned
/// blocks sequentially, walking row-by-row through the strictly upper
/// triangular part of the `p x p` block grid (plus the diagonal block
/// it owns).
///
/// In contrast to [`DiagBatchBlocks2D`] (which schedules by diagonal),
/// `SeqBatchBlocks2D` indexes the batch ranges as `[rank][batch]`.
/// Example below showing distribution for 5 processors running in 3 batches.
/// Batches are shown as a, b and c, starting with the first row and then
/// the second and so on.
///     [a a a a a]
///     [- b b b b]
///     [- - b c c]
///     [- - - c c]
///     [- - - - c]

#[derive(Debug, Serialize, Deserialize)]
pub struct SeqBatchBlocks2D {
    /// Side length of the square problem.
    n: usize,
    /// Sparse representation of the strictly upper triangular block
    /// grid (without the diagonal): `(flat ranges, per-row counts)`.
    blocks: (Vec<RangePair<usize>>, Vec<usize>),
    /// Diagonal blocks of the `p x p` grid, one per rank.
    diag_blocks: Vec<RangePair<usize>>,
    /// `p x bp` resolved sub-block ranges; row index = rank,
    /// column index = batch.
    batch_ranges: Vec2d<RangePair<usize>>,
    /// Number of sequential batches per rank.
    n_batches: usize,
}

impl SeqBatchBlocks2D {
    /// Construct the sequential batch distribution for an `n x n`
    /// problem across `p` ranks.
    ///
    /// When `p` is even, the boundary block on the final off-diagonal
    /// is split in half between the two ranks that share it so each
    /// batch produces an equal amount of work.
    pub fn new(n: usize, p: usize) -> Self {
        let blocks = triu_block_ranges_2d_no_diag(p as i32, n);
        let diag_blocks = diag_block_ranges(p as i32, n);
        let n_batches = 1 + (p / 2);
        let n_odbz = n_batches - 1; // off diagonal batches
        let bvr: Vec<RangePair<usize>> = if p.is_multiple_of(2) {
            assert!(blocks.0.len().is_multiple_of(p / 2));
            assert!(n_odbz == (blocks.0.len() / (p - 1)));
            (0..p)
                .flat_map(|i| {
                    let bstart = i * n_odbz - i.div_ceil(2);
                    let bend = bstart + n_odbz;
                    let mut rvec = blocks.0[bstart..bend].to_vec();
                    if i.is_even() {
                        rvec[n_odbz - 1] = (
                            top_half_range(&rvec[n_odbz - 1].0),
                            rvec[n_odbz - 1].1.clone(),
                        );
                    }
                    if i.is_odd() {
                        rvec[0] =
                            (bottom_half_range(&rvec[0].0), rvec[0].1.clone())
                    }
                    rvec.insert(0, diag_blocks[i].clone());
                    rvec.into_iter()
                })
                .collect()
        } else {
            assert!(blocks.0.len().is_multiple_of(p));
            (0..p)
                .flat_map(|i| {
                    let (bstart, bend) = (i * n_odbz, (i + 1) * n_odbz);
                    let mut rvec = blocks.0[bstart..bend].to_vec();
                    rvec.insert(0, diag_blocks[i].clone());
                    rvec.into_iter()
                })
                .collect()
        };
        let batch_ranges = Vec2d::new(bvr, p, n_batches);

        Self {
            n,
            blocks,
            diag_blocks,
            batch_ranges,
            n_batches,
        }
    }
}

impl BatchBlocks2D for SeqBatchBlocks2D {
    fn dim(&self) -> usize {
        self.n
    }

    fn num_batches(&self) -> usize {
        self.n_batches
    }

    fn batch_range(&self, bidx: usize, rank: i32) -> &RangePair<usize> {
        // batch_ranges is stored as p x n_batches 2d matrix
        self.batch_ranges.at(rank as usize, bidx)
    }
}

/// Sum type that lets callers store either a [`DiagBatchBlocks2D`] or a
/// [`SeqBatchBlocks2D`] behind one type while still being able to hand
/// out a `&dyn BatchBlocks2D` at use sites.
#[derive(Debug)]
pub enum EBBlocks2D {
    /// Diagonal-style scheduling.
    Diag(DiagBatchBlocks2D),
    /// Sequential-style scheduling.
    Seq(SeqBatchBlocks2D),
}

impl EBBlocks2D {
    /// Build the diagonal variant (see [`DiagBatchBlocks2D::new`]).
    pub fn new_diag(n: usize, p: usize) -> Self {
        Self::Diag(DiagBatchBlocks2D::new(n, p))
    }

    /// Build the sequential variant (see [`SeqBatchBlocks2D::new`]).
    pub fn new_seq(n: usize, p: usize) -> Self {
        Self::Seq(SeqBatchBlocks2D::new(n, p))
    }

    /// Return a `&dyn BatchBlocks2D` view of whichever variant is
    /// stored, allowing trait-object based dispatch.
    pub fn trait_ref(&self) -> &dyn BatchBlocks2D {
        match self {
            Self::Diag(diag_b) => diag_b,
            Self::Seq(seq_b) => seq_b,
        }
    }
}

///
/// Composite work distributor for pair-wise computations.
///
/// Bundles the three views typically needed by parallel pair-wise computation
/// kernels:
/// * a 1-D block decomposition of `nvars` (per-variable work);
/// * a 1-D block decomposition of `npairs` (linearised pair work); and
/// * a 2-D batched block decomposition of the `nvars x nvars` pair
///   grid (the [`EBBlocks2D`] field).
pub struct PairWorkDistributor {
    /// Owning rank (informational; not currently consulted directly).
    _rank: i32,
    /// Total number of ranks (informational).
    _size: i32,
    /// Per-rank ranges over the `nvars` variable axis.
    var_dist: Vec<Range<usize>>,
    /// Per-rank ranges over the linearised pair index space.
    pairs1d_dist: Vec<Range<usize>>,
    /// 2-D batched pair distribution.
    pairs2d: EBBlocks2D,
}

impl PairWorkDistributor {
    /// Construct a distributor that uses the diagonal 2-D scheduler
    /// (see [`DiagBatchBlocks2D`]).
    pub fn new(nvars: usize, npairs: usize, rank: i32, size: i32) -> Self {
        PairWorkDistributor {
            _rank: rank,
            _size: size,
            var_dist: all_block_ranges(size, nvars),
            pairs1d_dist: all_block_ranges(size, npairs),
            pairs2d: EBBlocks2D::new_diag(nvars, size as usize),
        }
    }

    /// Construct a distributor that uses the sequential 2-D scheduler
    /// (see [`SeqBatchBlocks2D`]).
    pub fn new_seq(nvars: usize, npairs: usize, rank: i32, size: i32) -> Self {
        PairWorkDistributor {
            _rank: rank,
            _size: size,
            var_dist: all_block_ranges(size, nvars),
            pairs1d_dist: all_block_ranges(size, npairs),
            pairs2d: EBBlocks2D::new_seq(nvars, size as usize),
        }
    }

    /// Underlying 2-D batched-block scheduler (the [`EBBlocks2D`] sum).
    pub fn pair_blocks(&self) -> &EBBlocks2D {
        &self.pairs2d
    }

    /// Trait-object view of the 2-D scheduler.
    pub fn pairs_2d(&self) -> &dyn BatchBlocks2D {
        self.pairs2d.trait_ref()
    }

    /// Per-rank slice of the linearised pair index distribution.
    pub fn pairs_1d(&self) -> &[Range<usize>] {
        &self.pairs1d_dist
    }

    /// Per-rank slice of the variable-axis distribution.
    pub fn vars_dist(&self) -> &[Range<usize>] {
        &self.var_dist
    }
}

///
/// Struct to represent the unique-with-counts reduction (see [`unique`]).
///
/// `values[i]` is the `i`-th distinct value (in their pre-existing
/// sorted order) and `counts[i]` records how many times it occurred in
/// the source slice.
///
#[derive(Debug)]
pub struct UniqCounts<T, S> {
    /// Distinct values, deduplicated while preserving input order.
    pub values: Vec<T>,
    /// Multiplicities of each entry in `values`.
    pub counts: Vec<S>,
}

/// Compute the unique values and per-value counts for a sorted slice.
///
/// Ues a generic count type `S`  so callers can ask for either integer or 
/// floating multiplicities.
/// NOTE:: `srt_data` __MUST__ already be sorted with respect to `PartialEq` so
/// that equal elements appear consecutively (the implementation relies
/// on `Vec::dedup`). 
pub fn unique<T, S>(srt_data: &[T]) -> UniqCounts<T, S>
where
    T: PartialEq + Clone + Debug,
    S: Zero + One + AddAssign + Clone + Debug,
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

/// Round every element of `in_data` to `decimals` decimal places.
///
/// Mirrors NumPy's `numpy.around`. Returns an owned `Array` of the
/// same shape.
/// NOTE:: If `10.0` cannot be represented in `A`, input is
/// returned unchanged.
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

/// Read a single string-valued column out of a CSV file.
///
/// `f_csv` is the path to a CSV with a header row, and `column` is the
/// name of the column to extract. Returns the values of that column in
/// row order. 
/// Fails with [`UtilError::MissingColumnError`] if the column name is
/// not present in the header. However, parsing errors are silently skipped.
pub fn read_csv_column(f_csv: &str, column: &str) -> Result<Vec<String>> {
    let csv_file = std::fs::File::open(f_csv)?;
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(csv_file);
    let header = rdr.headers()?;
    let (col_index, _) = header
        .iter()
        .enumerate()
        .find(|(_i, x)| (*x).eq(column))
        .ok_or(UtilError::MissingColumnError(column.to_string()))?;

    let rvec: Vec<String> = rdr
        .records()
        .filter_map(|x| match x {
            std::result::Result::Ok(record) => {
                record.get(col_index).map(|x| x.to_string())
            }
            Err(_err) => None,
        })
        .collect();
    Ok(rvec)
}

/// Give a pair (s_range, t_range),  build an `N x 2` matrix of 
/// all `(src, tgt)` pairs in `s_range x t_range` with `src < tgt`.
///
/// Returns the pairs as 2D array with two columns, whose first column is 
/// the source index and second column is the target index.
/// Indices are used to enumerate the unordered pair work inside a 2-D block. 
pub fn pair_indices<T>(st_ranges: RangePair<usize>) -> Array2<T>
where
    T: Integer + AddAssign + FromPrimitive + Clone,
{
    let (s_range, t_range) = st_ranges;
    let (s_vec, t_vec): (Vec<T>, Vec<T>) = iproduct!(s_range, t_range)
        .filter(|(src, tgt)| src < tgt)
        .map(|(src, tgt)| {
            (T::from_usize(src).unwrap(), T::from_usize(tgt).unwrap())
        })
        .unzip();

    let mut st_arr = Array2::<T>::zeros((s_vec.len(), 2));
    st_arr
        .slice_mut(ndarray::s![.., 0])
        .assign(&Array1::from_vec(s_vec));
    st_arr
        .slice_mut(ndarray::s![.., 1])
        .assign(&Array1::from_vec(t_vec));
    st_arr
}



/// Container for storing a sparse matrix result as index/value pairs.
///
/// Holds an `N x 2` index matrix `index` and a length-`N` value vector
/// `val` so that the `i`-th `(row, col)` pair in `index` is associated
/// with `val[i]`. Used throughout the network construction code to
/// emit sparse predictions/scores from each rank.
/// NOTE:: Expected that index and val have the same lengths.
pub struct IdVResults<T, S> {
    /// `N x 2` matrix of `(i, j)` indices, one row per result.
    pub index: Array2<T>,
    /// Length-`N` vector of associated values.
    pub val: Array1<S>,
}

impl<T: Clone + Zero, S: Clone + Zero> IdVResults<T, S> {
    /// Build a result container from an index matrix and a value
    /// vector. They should share the same number of rows; this is not
    /// checked.
    pub fn new(index: Array2<T>, val: Array1<S>) -> Self {
        // TODO: check both have same lengths
        Self { index, val }
    }

    /// Number of `(index, value)` entries.
    pub fn len(&self) -> usize {
        self.val.len()
    }

    /// `true` when there are no entries.
    pub fn is_empty(&self) -> bool {
        self.val.is_empty()
    }

    /// Concatenate several result sets into one.
    ///
    /// Allocates a single output matrix and vector sized for the total
    /// number of entries and then copies each input slice into its
    /// position computed from an exclusive prefix sum of the per-input
    /// sizes (see [`exc_prefix_sum`]).
    pub fn merge(vpreds: &[Self]) -> Self {
        let nsizes: Vec<usize> = vpreds.iter().map(|x| x.len()).collect();
        let nstarts: Vec<usize> = exc_prefix_sum(nsizes.clone().into_iter(), 1);
        let ntotal: usize = vpreds.iter().map(|x| x.len()).sum();
        let mut pindices: Array2<T> = Array2::zeros((ntotal, 2));
        let mut preds: Array1<S> = Array1::zeros(ntotal);

        for (idx, rstart) in nstarts.iter().enumerate() {
            let rsize = vpreds[idx].val.len();
            let rend = *rstart + rsize;
            pindices
                .slice_mut(ndarray::s![*rstart..rend, ..])
                .assign(&vpreds[idx].index);
            preds
                .slice_mut(ndarray::s![*rstart..rend])
                .assign(&vpreds[idx].val);
        }
        Self::new(pindices, preds)
    }
}



#[cfg(test)]
mod tests {
    use anyhow::Result;
    use log::debug;
    use ndarray::Array1;

    use crate::util::{BatchBlocks2D, RangePair, SeqBatchBlocks2D};
    #[test]
    pub fn test_triu_index() {
        use super::{triu_index_to_pair, triu_pair_to_index};
        crate::tests::log_init();
        let n = 500;
        for i in 0..6 {
            for j in (i + 1)..6 {
                let rk = triu_pair_to_index(n, i, j);
                let ix: (usize, usize) = triu_index_to_pair(n, rk);
                debug!("{} {} {} {:?}", i, j, rk, ix)
            }
        }
    }

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
        debug!("NRDA {:8.4}", nrdva);
        debug!("NRDB {:8.4}", nrdvb);

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

    #[test]
    fn test_seq_blocks2d() {
        crate::tests::log_init();
        let np: usize = 8;
        let sb = SeqBatchBlocks2D::new(500, np);

        debug!("SB {} SBL {:?}", sb.blocks.0.len(), sb.blocks.0);

        for rank in 0..np {
            let rvec = (0..sb.n_batches)
                .map(|bx| sb.batch_range(bx, rank as i32))
                .cloned()
                .collect::<Vec<RangePair<usize>>>();
            let rvec = rvec[..rvec.len() - 1].to_vec();
            debug!(" {} {:?}", rank, rvec,);
        }
    }

    #[test]
    fn test_read_csv() -> Result<()> {
        crate::tests::log_init();
        use super::read_csv_column;

        let tf_csv = crate::tests::test_tf_file();
        let csv_strs = read_csv_column(tf_csv, "gene")?;

        debug!("V: {}", csv_strs.len());
        assert_eq!(csv_strs.len(), 2862);
        assert_eq!(
            csv_strs[..5],
            vec!["A2M", "AATF", "ABCA1", "ABCA3", "ABCB1"]
        );
        assert_eq!(
            csv_strs[2857..],
            vec!["ZNF444", "ZNF652", "ZNF750", "ZNF76", "ZNRD1"]
        );

        Ok(())
    }
}
