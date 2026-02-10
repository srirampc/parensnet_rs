#![allow(dead_code)]

use num::{
    traits::float::TotalOrder,
    {Float, FromPrimitive, Integer, One, ToPrimitive, Zero},
};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    hash::Hash,
    marker::Sized,
    ops::{AddAssign, MulAssign, SubAssign},
};

pub trait AddFromZero: Zero + One + AddAssign {}
impl<T: Zero + One + AddAssign> AddFromZero for T {}

pub trait AssignOps: Sized + AddAssign + MulAssign + SubAssign {}
impl<T: Sized + AddAssign + MulAssign + SubAssign> AssignOps for T {}

pub trait FromToPrimitive: FromPrimitive + ToPrimitive {}
impl<T: FromPrimitive + ToPrimitive> FromToPrimitive for T {}

pub trait PNInteger:
    Integer + Copy + Clone + Debug + Hash + PartialOrd + FromToPrimitive + AssignOps
{
}
impl<
    T: Integer
        + Copy
        + Clone
        + Debug
        + Hash
        + PartialOrd
        + FromToPrimitive
        + AssignOps,
> PNInteger for T
{
}

pub trait PNFloat:
    Float + Debug + Clone + TotalOrder + FromToPrimitive + AssignOps
{
}
impl<T: Float + Debug + Clone + TotalOrder + FromToPrimitive + AssignOps> PNFloat
    for T
{
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
pub enum LogBase {
    #[serde(alias = "e")]
    Natural,
    #[default]
    #[serde(alias = "2")]
    Two,
    #[serde(alias = "10")]
    Ten,
    //OneP,
    //
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
pub enum DiscretizerMethod {
    #[default]
    #[serde(alias = "bayesian_blocks")]
    BayesianBlocks,
    #[serde(alias = "uniform")]
    Uniform,
}

#[derive(Debug)]
pub struct Pair<T> {
    pub first: T,
    pub second: T,
}

impl<T> Pair<T> {
    pub fn new(first: T, second: T) -> Self {
        Pair { first, second }
    }

    pub fn from_tuple((first, second): (T, T)) -> Self {
        Pair { first, second }
    }

    pub fn at(&self, i: usize) -> &T {
        if i == 0 { &self.first } else { &self.second }
    }

    pub fn map<B, F>(&self, map_fn: F) -> Pair<B>
    where
        F: Fn(&T) -> B,
    {
        Pair::new(map_fn(&self.first), map_fn(&self.second))
    }

    pub fn zip_map<S, B, F>(&self, other: &Pair<S>, map_fn: F) -> Pair<B>
    where
        F: Fn(&T, &S) -> B,
    {
        Pair::new(
            map_fn(&self.first, &other.first),
            map_fn(&self.second, &other.second),
        )
    }
}

impl<T: Clone> Pair<T> {
    pub fn to_tuple(&self) -> (T, T) {
        (self.first.clone(), self.second.clone())
    }
}

impl<T: Clone> Clone for Pair<T> {
    fn clone(&self) -> Self {
        Pair::new(self.first.clone(), self.second.clone())
    }
}
