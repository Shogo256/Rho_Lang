// SPDX-License-Identifier: Apache-2.0
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
    /// The largest whole number not above the value. What a residue is built
    /// from; not a named function of the language.
    fn floor(&self) -> Self;

    fn compare(&self, other: &Self, how: Compare) -> Self::Bool;
    fn select(condition: &Self::Bool, when_true: &Self, when_false: &Self) -> Self;

    /// The flag's value. The exit of a `⇒` loop is the one decision in the
    /// language that asks.
    fn truth(flag: &Self::Bool) -> bool;

    /// This value, marked with whatever `other` carries besides its value:
    /// a number is just itself, a tainted number takes the other's taint.
    /// A read at a position the data names has the position's taint even
    /// when it yields zero.
    fn carrying(&self, _other: &Self) -> Self {
        self.clone()
    }
}

/// The roll: a number in [0, 1) from the bits of a value. splitmix64's
/// finaliser over the value's bits, the top 53 of the result scaled down.
/// Every NaN hashes as one value, since the kernel's NaN and the
/// interpreter's need not share a payload. Not for cryptography.
pub fn roll(value: f64) -> f64 {
    let bits = if value.is_nan() { 0 } else { value.to_bits() };
    let mut z = bits.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
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
    fn floor(&self) -> Self {
        f64::floor(*self)
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
            BuiltinOp::Roll => roll(*self),
            BuiltinOp::Floor => self.floor(),
            BuiltinOp::Ceil => self.ceil(),
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
    fn floor(&self) -> Self {
        f32::floor(*self)
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
            // Hashed at double width and rounded once, as the kernel does.
            BuiltinOp::Roll => roll(f64::from(*self)) as f32,
            BuiltinOp::Floor => self.floor(),
            BuiltinOp::Ceil => self.ceil(),
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

/// A fixed-point number: `raw` / 2^F in two's complement of W bits. The
/// arithmetic an integer datapath does, and the reference for a circuit's
/// bits: a sum wraps, a product is shifted down with the floor, a quotient
/// is truncated toward zero and a division by zero gives zero, a literal is
/// rounded to the nearest step (half away from zero). There is no NaN, no
/// infinity and no negative zero. The named functions that a circuit cannot
/// do in integers — exp, log, sqrt, sin, cos, a power with a fractional
/// exponent — go through a double here and are refused by the emitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fixed<const W: u32, const F: u32> {
    pub raw: i64,
}

impl<const W: u32, const F: u32> Fixed<W, F> {
    pub const ONE: i64 = 1i64 << F;
    const MASK: i64 = (1i64 << F) - 1;

    /// Keep the low W bits, sign-extended: what a W-bit register holds.
    pub fn wrap(v: i128) -> Self {
        let shift = 128 - W;
        Fixed {
            raw: ((v << shift) >> shift) as i64,
        }
    }

    /// The most positive and most negative values the width holds.
    pub const MAX: i64 = (1i64 << (W - 1)) - 1;
    pub const MIN: i64 = -(1i64 << (W - 1));

    /// Round to the nearest step, half away from zero; an infinity is the
    /// end of the range it points to (so a fold's identity is the extreme),
    /// a NaN is zero; a value past the range wraps as the register would.
    pub fn from_f64(v: f64) -> Self {
        if v.is_nan() {
            return Fixed { raw: 0 };
        }
        if v == f64::INFINITY {
            return Fixed { raw: Self::MAX };
        }
        if v == f64::NEG_INFINITY {
            return Fixed { raw: Self::MIN };
        }
        Self::wrap((v * (1u64 << F) as f64).round() as i128)
    }

    pub fn to_f64(self) -> f64 {
        self.raw as f64 / (1u64 << F) as f64
    }
}

impl<const W: u32, const F: u32> Numeric for Fixed<W, F> {
    type Bool = bool;

    fn constant(value: f64) -> Self {
        Self::from_f64(value)
    }
    fn as_constant(&self) -> Option<f64> {
        Some(self.to_f64())
    }
    fn add(&self, o: &Self) -> Self {
        Self::wrap(self.raw as i128 + o.raw as i128)
    }
    fn sub(&self, o: &Self) -> Self {
        Self::wrap(self.raw as i128 - o.raw as i128)
    }
    fn mul(&self, o: &Self) -> Self {
        Self::wrap((self.raw as i128 * o.raw as i128) >> F)
    }
    fn div(&self, o: &Self) -> Self {
        if o.raw == 0 {
            return Fixed { raw: 0 };
        }
        Self::wrap(((self.raw as i128) << F) / o.raw as i128)
    }
    fn power(&self, o: &Self) -> Self {
        Self::from_f64(self.to_f64().powf(o.to_f64()))
    }
    fn unary(&self, op: BuiltinOp) -> Self {
        match op {
            BuiltinOp::Abs => Self::wrap((self.raw as i128).abs()),
            BuiltinOp::Indicator => Fixed {
                raw: if self.raw != 0 { Self::ONE } else { 0 },
            },
            BuiltinOp::Floor => Fixed {
                raw: self.raw & !Self::MASK,
            },
            BuiltinOp::Ceil => Self::wrap(((self.raw as i128) + Self::MASK as i128) & !(Self::MASK as i128)),
            // The roll over the value's W bits, sign-extended to 64: the top
            // F bits of the hash are the fraction of a number in [0, 1).
            BuiltinOp::Roll => {
                let bits = self.raw as u64;
                let mut z = bits.wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                Fixed {
                    raw: (z >> (64 - F)) as i64,
                }
            }
            BuiltinOp::Exp => Self::from_f64(self.to_f64().exp()),
            BuiltinOp::Log => Self::from_f64(self.to_f64().ln()),
            BuiltinOp::Sqrt => Self::from_f64(self.to_f64().sqrt()),
            BuiltinOp::Sin => Self::from_f64(self.to_f64().sin()),
            BuiltinOp::Cos => Self::from_f64(self.to_f64().cos()),
        }
    }
    fn floor(&self) -> Self {
        self.unary(BuiltinOp::Floor)
    }
    fn compare(&self, o: &Self, how: Compare) -> bool {
        match how {
            Compare::Gt => self.raw > o.raw,
            Compare::Lt => self.raw < o.raw,
            Compare::Gte => self.raw >= o.raw,
            Compare::Lte => self.raw <= o.raw,
            Compare::Eq => self.raw == o.raw,
            Compare::Ne => self.raw != o.raw,
        }
    }
    fn select(condition: &bool, a: &Self, b: &Self) -> Self {
        if *condition {
            *a
        } else {
            *b
        }
    }
    fn truth(flag: &bool) -> bool {
        *flag
    }
}

#[cfg(test)]
mod fixed_tests {
    use super::*;

    type Q = Fixed<32, 16>;

    #[test]
    fn a_literal_rounds_to_the_nearest_step_and_wraps_to_the_width() {
        assert_eq!(Q::constant(1.0).raw, 1 << 16);
        assert_eq!(Q::constant(0.5).raw, 1 << 15);
        // Half a step away from zero rounds away from zero, as `round` does.
        assert_eq!(Q::constant(1.0 / 131072.0).raw, 1);
        assert_eq!(Q::constant(-1.0 / 131072.0).raw, -1);
        assert_eq!(Q::constant(f64::NAN).raw, 0);
        assert_eq!(Q::constant(f64::INFINITY).raw, i32::MAX as i64);
        assert_eq!(Q::constant(f64::NEG_INFINITY).raw, i32::MIN as i64);
        // 2^15 does not fit in Q16.16: it wraps to the negative end.
        assert_eq!(Q::constant(32768.0).raw, i32::MIN as i64);
        assert_eq!(Q::constant(1.5).to_f64(), 1.5);
    }

    #[test]
    fn products_floor_quotients_truncate_and_zero_divides_to_zero() {
        let (a, b) = (Q::constant(1.5), Q::constant(-2.25));
        assert_eq!(a.mul(&b).to_f64(), -3.375);
        assert_eq!(a.add(&b).to_f64(), -0.75);
        assert_eq!(a.sub(&b).to_f64(), 3.75);
        // -1 / 3 in Q16.16: the exact quotient is -21845.33 steps; toward zero.
        assert_eq!(Q::constant(-1.0).div(&Q::constant(3.0)).raw, -21845);
        // A tiny product shifts down with the floor: -1 step x 0.5 is -0.5 step, floored to -1.
        assert_eq!(Fixed::<32, 16> { raw: -1 }.mul(&Q::constant(0.5)).raw, -1);
        assert_eq!(Q::constant(1.0).div(&Q::constant(0.0)).raw, 0);
        // The sum of two large values wraps rather than saturating.
        assert_eq!(Q::constant(20000.0).add(&Q::constant(20000.0)).to_f64(), -25536.0);
    }

    #[test]
    fn the_named_functions_a_datapath_has() {
        assert_eq!(Q::constant(-2.5).unary(BuiltinOp::Abs).to_f64(), 2.5);
        assert_eq!(Q::constant(-2.5).unary(BuiltinOp::Floor).to_f64(), -3.0);
        assert_eq!(Q::constant(-2.5).unary(BuiltinOp::Ceil).to_f64(), -2.0);
        assert_eq!(Q::constant(2.0).unary(BuiltinOp::Ceil).to_f64(), 2.0);
        assert_eq!(Q::constant(0.001).unary(BuiltinOp::Indicator).to_f64(), 1.0);
        assert_eq!(Q::constant(0.0).unary(BuiltinOp::Indicator).to_f64(), 0.0);
        let rolls: Vec<f64> = (0..64).map(|i| Q::constant(i as f64).unary(BuiltinOp::Roll).to_f64()).collect();
        assert!(rolls.iter().all(|r| (0.0..1.0).contains(r)));
        assert!(rolls.windows(2).any(|w| w[0] != w[1]));
        assert!(Q::constant(1.0).compare(&Q::constant(0.999), Compare::Gt));
        assert_eq!(integer_power(&Q::constant(1.5), &Q::constant(3.0)).to_f64(), 3.375);
    }
}
