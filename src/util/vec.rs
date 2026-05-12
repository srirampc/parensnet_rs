use std::fmt;

use serde::{Deserialize, Serialize};

///
/// A simple 2d Vector.
///
/// Stores elements in a single, contiguous, row-major `Vec<T>` so that
/// indexing into row `r` and column `c` resolves to `data[r * ncol + c]`.
/// It is `Serialize`/`Deserialize` so that it can be persisted alongside
/// the other distribution structures used in the crate (e.g.
/// [`super::DiagBatchBlocks2D`], [`super::SeqBatchBlocks2D`]).
///
#[derive(Debug, Serialize, Deserialize)]
pub struct Vec2d<T> {
    /// Backing storage in row-major order; length is `nrow * ncol`.
    data: Vec<T>,
    /// Number of rows.
    nrow: usize,
    /// Number of columns.
    ncol: usize,
}

impl<T> Vec2d<T> {
    /// Build a `Vec2d` from a flat row-major vector.
    ///
    /// Panics (in debug and release) if `vec.len() != row * col`.
    pub fn new(vec: Vec<T>, row: usize, col: usize) -> Self {
        assert!(vec.len() == row * col);
        Self { data: vec, nrow: row, ncol: col }
    }

    /// Replace the underlying storage and dimensions in place.
    ///
    /// Useful for re-using an allocation when the matrix shape changes.
    /// Panics if `vec.len() != row * col`.
    pub fn reset(&mut self, vec: Vec<T>, row: usize, col: usize) {
        assert!(vec.len() == row * col);
        self.data = vec;
        self.nrow = row;
        self.ncol = col;
    }

    /// Return a reference to the flat row-major backing vector.
    pub fn flatten(&self) -> &Vec<T> {
        &self.data
    }

    /// Return an immutable slice over the elements of `row`.
    pub fn row(&self, row: usize) -> &[T] {
        let i = self.ncol * row;
        &self.data[i..(i + self.ncol)]
    }

    /// Return a reference to the element at `(row, col)`.
    pub fn at(&self, row: usize, col: usize) -> &T {
        let i = self.ncol * row;
        &self.data[i + col]
    }

    /// Return a mutable reference to the element at `(row, col)`.
    pub fn mut_at(&mut self, row: usize, col: usize) -> &mut T {
        let i = self.ncol * row;
        &mut self.data[i + col]
    }

    /// Return `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrow
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.ncol
    }
}


impl<T: Clone> Vec2d<T> {
    /// Construct a `row x col` matrix with every cell initialised to a
    /// clone of `val`.
    pub fn new_with(row: usize, col: usize, val: T) -> Self {
        Self { data: vec![val.clone() ; row * col], nrow: row, ncol: col }
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for Vec2d<T> {
    /// Pretty-print the matrix as `[row0, row1, ...]` where each row is
    /// formatted with `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut str = String::new();
        for i in 0..self.nrow {
            if i != 0 {
                str.push_str(", ");
            }
            str.push_str(&format!("{:?}", &self.row(i)));
        }
        write!(f, "[{}]", str)
    }
}
