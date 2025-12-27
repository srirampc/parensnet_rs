use std::fmt;

use serde::{Deserialize, Serialize};

///
/// A simple 2d Vector
///
#[derive(Debug, Serialize, Deserialize)]
pub struct Vec2d<T> {
    data: Vec<T>,
    nrow: usize,
    ncol: usize,
}

impl<T> Vec2d<T> {
    pub fn new(vec: Vec<T>, row: usize, col: usize) -> Self {
        assert!(vec.len() == row * col);
        Self { data: vec, nrow: row, ncol: col }
    }

    pub fn reset(&mut self, vec: Vec<T>, row: usize, col: usize) {
        assert!(vec.len() == row * col);
        self.data = vec;
        self.nrow = row;
        self.ncol = col;
    }

    pub fn flatten(&self) -> &Vec<T> {
        &self.data
    }

    pub fn row(&self, row: usize) -> &[T] {
        let i = self.ncol * row;
        &self.data[i..(i + self.ncol)]
    }

    pub fn at(&self, row: usize, col: usize) -> &T {
        let i = self.ncol * row;
        &self.data[i + col]
    }

    pub fn mut_at(&mut self, row: usize, col: usize) -> &mut T {
        let i = self.ncol * row;
        &mut self.data[i + col]
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    pub fn nrows(&self) -> usize {
        self.nrow
    }

    pub fn ncols(&self) -> usize {
        self.ncol
    }
}


impl<T: Clone> Vec2d<T> {
    pub fn new_with(row: usize, col: usize, val: T) -> Self {
        Self { data: vec![val.clone() ; row * col], nrow: row, ncol: col }
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for Vec2d<T> {
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

