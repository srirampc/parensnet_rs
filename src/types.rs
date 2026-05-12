//! Common type aliases, marker traits, and small helper types used
//! throughout `parensnet_rs`.
//!
//! The module has three groups of types:
//!
//! * **Numeric marker traits** ([`AddFromZero`], [`AssignOps`],
//!   [`FromToPrimitive`], [`PNInteger`], [`PNFloat`]) that bundle the
//!   bounds repeatedly required by the histogramming, statistics,
//!   and network-construction routines so that downstream signatures
//!   stay short.
//! * **Configuration enums** ([`LogBase`], [`DiscretizerMethod`]) that
//!   are loaded from YAML/TOML config and therefore implement
//!   `Serialize` / `Deserialize` with friendly aliases.
//! * The simple [`Pair`] container used as a typed alternative to
//!   tuples in places where a "first/second" distinction reads more
//!   clearly (e.g. source/target indices in pair-wise computations).

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
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

/// Marker trait for numeric types that can be accumulated starting
/// from `T::zero()`.
pub trait AddFromZero: Zero + One + AddAssign {}
impl<T: Zero + One + AddAssign> AddFromZero for T {}

/// Marker trait for types supporting the four compound-assignment
/// arithmetic operators (`+=`, `-=`, `*=`, `/=`).
pub trait AssignOps:
    Sized + AddAssign + SubAssign + MulAssign + DivAssign
{
}
impl<T: Sized + AddAssign + SubAssign + MulAssign + DivAssign> AssignOps for T {}

/// Convenience marker for types that can be converted both *from* and
/// *to* the primitive numeric types via the `num` crate.
pub trait FromToPrimitive: FromPrimitive + ToPrimitive {}
impl<T: FromPrimitive + ToPrimitive> FromToPrimitive for T {}

/// Trait bounds for the integer types accepted by the crate's generic numeric routines.
/// Implemented by the standard signed and unsigned integer primitives.
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

/// Trait bounds for the floating-point types accepted by the
/// crate's generic numeric routines. `f32` and `f64` qualify automatically.
pub trait PNFloat:
    Float + Debug + Clone + TotalOrder + FromToPrimitive + AssignOps
{
}
impl<T: Float + Debug + Clone + TotalOrder + FromToPrimitive + AssignOps> PNFloat
    for T
{
}

/// Logarithm base used by the entropy / mutual-information kernels.
///
/// The serde aliases (`"e"`, `"2"`, `"10"`) let configs name the base
/// directly. [`Two`](Self::Two) is the default.
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
pub enum LogBase {
    /// Natural logarithm (base `e`); results are in nats.
    #[serde(alias = "e")]
    Natural,
    /// Base-2 logarithm; results are in bits.
    #[default]
    #[serde(alias = "2")]
    Two,
    /// Base-10 logarithm; results are in dits / hartleys.
    #[serde(alias = "10")]
    Ten,
    //OneP,
    //
}

/// Strategy used to discretize a continuous variable into bins prior
/// to histogramming.
///
/// Values are deserialised from config files via the `bayesian_blocks`
/// and `uniform` aliases. [`BayesianBlocks`](Self::BayesianBlocks) is
/// the default.
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
pub enum DiscretizerMethod {
    /// Use the Bayesian-blocks edges from
    /// [`crate::hist::bayesian_blocks_bin_edges`].
    #[default]
    #[serde(alias = "bayesian_blocks")]
    BayesianBlocks,
    /// Use uniformly spaced bins.
    #[serde(alias = "uniform")]
    Uniform,
}

/// A simple two-element container with named `first` / `second`
/// fields.
///
/// Used as a more readable alternative to `(T, T)` in places where the
/// two slots are semantically distinct (e.g. source/target,
/// upper/lower halves). Also provides convenience helpers like
/// [`Pair::map`] and [`Pair::zip_map`] that aren't available on plain
/// tuples.
#[derive(Debug)]
pub struct Pair<T> {
    /// First element of the pair.
    pub first: T,
    /// Second element of the pair.
    pub second: T,
}

impl<T> Pair<T> {
    /// Build a pair from two owned values.
    pub fn new(first: T, second: T) -> Self {
        Pair { first, second }
    }

    /// Build a pair from a `(first, second)` tuple. Useful when an
    /// upstream API still hands out tuples.
    pub fn from_tuple((first, second): (T, T)) -> Self {
        Pair { first, second }
    }

    /// Index access: `i == 0` returns `&first`, anything else returns
    /// `&second`. Mirrors the ergonomics of a length-2 array.
    pub fn at(&self, i: usize) -> &T {
        if i == 0 { &self.first } else { &self.second }
    }

    /// Apply `map_fn` to both elements and return the resulting
    /// `Pair<B>`. Analogous to `Option::map` but for the two slots
    /// of the pair.
    pub fn map<B, F>(&self, map_fn: F) -> Pair<B>
    where
        F: Fn(&T) -> B,
    {
        Pair::new(map_fn(&self.first), map_fn(&self.second))
    }

    /// Element-wise combine two pairs by applying `map_fn` to the
    /// corresponding `first` and `second` slots.
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
    /// Convert into a plain `(first, second)` tuple by cloning the fields.
    pub fn to_tuple(&self) -> (T, T) {
        (self.first.clone(), self.second.clone())
    }
}

impl<T: Clone> Clone for Pair<T> {
    /// Element-wise `Clone`. 
    fn clone(&self) -> Self {
        Pair::new(self.first.clone(), self.second.clone())
    }
}
