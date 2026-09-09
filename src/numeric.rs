//! The value a program is evaluated over.
//!
//! The reference interpreter is written against this trait rather than against
//! `f64` directly, so the same code runs at both widths a kernel can be
//! compiled at and is compared against the kernel at each.

use crate::ast::BuiltinOp;
use std::fmt;

/// The comparisons the language and the IR both need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
    /// What a non-zero test means: `!=`, true for NaN.
    Ne,
}

pub trait Numeric: Clone + fmt::Debug {
    type Bool: Clone + fmt::Debug;

    fn constant(value: f64) -> Self;
    /// The value when it is known, which is what lets `^` recognise a whole
    /// exponent whether it is running on numbers or on symbols.
    fn as_constant(&self) -> Option<f64>;

    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    /// Exponentiation with an exponent that is not a whole number.
    fn power(&self, other: &Self) -> Self;
    /// A named function over one value.
    fn unary(&self, op: BuiltinOp) -> Self;

    fn compare(&self, other: &Self, how: Compare) -> Self::Bool;
    fn select(condition: &Self::Bool, when_true: &Self, when_false: &Self) -> Self;

    /// The flag's value. The exit of a `⇒` loop is the one decision in the
    /// language that asks.
    fn truth(flag: &Self::Bool) -> bool;
}

impl Numeric for f64 {
    type Bool = bool;

    fn constant(value: f64) -> Self {
        value
    }
    fn as_constant(&self) -> Option<f64> {
        Some(*self)
    }
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    fn div(&self, other: &Self) -> Self {
        self / other
    }
    fn power(&self, other: &Self) -> Self {
        f64::powf(*self, *other)
    }
    fn unary(&self, op: BuiltinOp) -> Self {
        match op {
            BuiltinOp::Exp => self.exp(),
            BuiltinOp::Log => self.ln(),
            BuiltinOp::Sqrt => self.sqrt(),
            BuiltinOp::Sin => self.sin(),
            BuiltinOp::Cos => self.cos(),
            BuiltinOp::Abs => self.abs(),
            BuiltinOp::Indicator => {
                if *self != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    fn compare(&self, other: &Self, how: Compare) -> bool {
        match how {
            Compare::Gt => self > other,
            Compare::Lt => self < other,
            Compare::Gte => self >= other,
            Compare::Lte => self <= other,
            Compare::Eq => self == other,
            Compare::Ne => self != other,
        }
    }
    fn select(condition: &bool, when_true: &Self, when_false: &Self) -> Self {
        if *condition {
            *when_true
        } else {
            *when_false
        }
    }
    fn truth(flag: &bool) -> bool {
        *flag
    }
}

/// How wide the numbers a kernel computes with are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    F64,
    F32,
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::F64 => "f64",
            Precision::F32 => "f32",
        }
    }

    /// The LLVM type a value of this precision has.
    pub fn llvm_type(self) -> &'static str {
        match self {
            Precision::F64 => "double",
            Precision::F32 => "float",
        }
    }

    /// The suffix LLVM's intrinsics carry.
    pub fn intrinsic_suffix(self) -> &'static str {
        match self {
            Precision::F64 => "f64",
            Precision::F32 => "f32",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Precision::F64 => 8,
            Precision::F32 => 4,
        }
    }

    /// Unit roundoff: every arithmetic result is the exact one times (1 + δ)
    /// with |δ| no larger than this. Narrowing the numbers widens the doubt,
    /// which is exactly why a proof is worth more at f32 than at f64.
    pub fn unit_roundoff(self) -> f64 {
        match self {
            Precision::F64 => 1.0 / 9_007_199_254_740_992.0, // 2^-53
            Precision::F32 => 1.0 / 16_777_216.0,            // 2^-24
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// The same arithmetic at single precision. The interpreter runs on this
/// directly, so a narrow kernel is checked against narrow arithmetic rather
/// than against a rounded-down wide one.
impl Numeric for f32 {
    type Bool = bool;

    fn constant(value: f64) -> Self {
        value as f32
    }
    fn as_constant(&self) -> Option<f64> {
        Some(*self as f64)
    }
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    fn div(&self, other: &Self) -> Self {
        self / other
    }
    fn power(&self, other: &Self) -> Self {
        f32::powf(*self, *other)
    }
    fn unary(&self, op: BuiltinOp) -> Self {
        match op {
            BuiltinOp::Exp => self.exp(),
            BuiltinOp::Log => self.ln(),
            BuiltinOp::Sqrt => self.sqrt(),
            BuiltinOp::Sin => self.sin(),
            BuiltinOp::Cos => self.cos(),
            BuiltinOp::Abs => self.abs(),
            BuiltinOp::Indicator => {
                if *self != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    fn compare(&self, other: &Self, how: Compare) -> bool {
        match how {
            Compare::Gt => self > other,
            Compare::Lt => self < other,
            Compare::Gte => self >= other,
            Compare::Lte => self <= other,
            Compare::Eq => self == other,
            Compare::Ne => self != other,
        }
    }
    fn select(condition: &bool, when_true: &Self, when_false: &Self) -> Self {
        if *condition {
            *when_true
        } else {
            *when_false
        }
    }
    fn truth(flag: &bool) -> bool {
        *flag
    }
}

/// `x^n` as repeated multiplication when `n` is a small whole number.
///
/// The definition lives here so the interpreter, the IR reader and the solver
/// all inherit exactly the same one. Leaving it to a maths library made two
/// implementations round a square differently.
pub fn integer_power<S: Numeric>(base: &S, exponent: &S) -> S {
    let Some(n) = exponent.as_constant() else {
        return base.power(exponent);
    };
    if n != n.trunc() || n.abs() > 64.0 {
        return base.power(exponent);
    }
    let steps = n.abs() as u32;
    let mut acc = S::constant(1.0);
    for _ in 0..steps {
        acc = acc.mul(base);
    }
    if n < 0.0 {
        S::constant(1.0).div(&acc)
    } else {
        acc
    }
}
