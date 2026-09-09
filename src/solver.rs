//! Static checking of `!` constraints.
//!
//! The expanded program is evaluated in interval arithmetic, which needs no
//! dependencies and soundly over-approximates every value. A program is
//! rejected only when a violation is certain; anything the intervals cannot
//! settle is reported as unproven rather than failing the build. Its claims
//! are checked against real runs by `difftest`, which compiles random programs
//! and compares what the kernel produced with what was claimed.

use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use crate::numeric::Precision;
use crate::symbolic::{expand, Cmp, Domain, Sym};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Holds for every input.
    Proved,
    /// Neither proved nor refuted; the reason names what stopped the backend.
    Unproven(String),
    /// Fails for some reachable input.
    Violated(String),
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub subject: String,
    pub verdict: Verdict,
    /// Source line the obligation came from.
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub backend: &'static str,
    /// The width the analysis reasoned at, which every finding assumes.
    pub precision: Precision,
    pub constraints: Vec<Finding>,
    pub divisions: Vec<Finding>,
    /// Arguments that have to stay inside a named function's domain.
    pub domains: Vec<Finding>,
    /// Bounds on every cell the kernel writes. Infinite ends mean "not bounded".
    pub output_range: Interval,
}

impl Report {
    pub fn violations(&self) -> impl Iterator<Item = &Finding> {
        self.constraints
            .iter()
            .chain(self.divisions.iter())
            .chain(self.domains.iter())
            .filter(|f| matches!(f.verdict, Verdict::Violated(_)))
    }
}

pub struct ConstraintSolver;

impl ConstraintSolver {
    /// Analyse a program at the default width.
    pub fn analyze(block: &ToposBlock, tau: f64) -> Report {
        Self::analyze_at(block, tau, Precision::F64)
    }

    /// Analyse every constraint and every division in the program.
    pub fn analyze_at(block: &ToposBlock, tau: f64, precision: Precision) -> Report {
        ROUNDOFF.with(|r| r.set(precision.unit_roundoff()));
        let expansion = expand(block, tau);

        let constraints = expansion
            .obligations
            .iter()
            .map(|o| Finding {
                subject: o.source.clone(),
                verdict: check_interval(o.cmp, &o.lhs, &o.rhs),
                line: o.line,
            })
            .collect();

        let divisions = expansion
            .divisions
            .iter()
            .map(|(text, denom, line)| Finding {
                subject: text.clone(),
                verdict: check_denominator(denom),
                line: *line,
            })
            .collect();

        let domains = expansion
            .domains
            .iter()
            .map(|(text, argument, domain, line)| Finding {
                subject: text.clone(),
                verdict: check_domain(argument, *domain),
                line: *line,
            })
            .collect();

        // The last flow is what the caller receives; its range is stated so
        // a run can be checked against it.
        let output_range = expansion
            .output
            .as_ref()
            .map(eval_interval)
            .unwrap_or(Interval::UNBOUNDED);

        Report {
            backend: "interval",
            precision,
            constraints,
            divisions,
            domains,
            output_range,
        }
    }

    /// Reject the program if any obligation is certainly violated.
    pub fn verify_constraints(block: &ToposBlock) -> Result<()> {
        Self::verify(block, 0.0).map(|_| ())
    }

    pub fn verify(block: &ToposBlock, tau: f64) -> Result<Report> {
        Self::verify_at(block, tau, Precision::F64)
    }

    pub fn verify_at(block: &ToposBlock, tau: f64, precision: Precision) -> Result<Report> {
        let report = Self::analyze_at(block, tau, precision);
        if let Some(bad) = report.violations().next() {
            let detail = match &bad.verdict {
                Verdict::Violated(why) => why.clone(),
                _ => String::new(),
            };
            return Err(HarmonyDisruption::LogicErr {
                expr: format!("{} — {detail}", bad.subject),
                line: bad.line,
            });
        }
        Ok(report)
    }
}

// ---------------------------------------------------------------- intervals

/// A closed range that always contains the true value. Infinite bounds mean
/// "unknown in that direction"; every operation widens rather than guesses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub const UNBOUNDED: Interval = Interval {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };

    pub fn point(v: f64) -> Interval {
        if v.is_nan() {
            Interval::UNBOUNDED
        } else {
            Interval { lo: v, hi: v }
        }
    }

    fn hull(a: Interval, b: Interval) -> Interval {
        Interval {
            lo: a.lo.min(b.lo),
            hi: a.hi.max(b.hi),
        }
    }

    fn contains_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.lo == 0.0 && self.hi == 0.0
    }
}

/// Bounds recovered from signs alone, for the cases where the endpoint
/// products are indeterminate — `inf / inf` and `0 * inf` both come out NaN,
/// but the sign of the result is still known and worth keeping.
fn sign_bounds(a: Interval, b: Interval, dividing: bool) -> Interval {
    let a_nonneg = a.lo >= 0.0;
    let a_nonpos = a.hi <= 0.0;
    let b_pos = if dividing { b.lo > 0.0 } else { b.lo >= 0.0 };
    let b_neg = if dividing { b.hi < 0.0 } else { b.hi <= 0.0 };

    let nonneg = (a_nonneg && b_pos) || (a_nonpos && b_neg);
    let nonpos = (a_nonpos && b_pos) || (a_nonneg && b_neg);

    if nonneg && nonpos {
        Interval::point(0.0)
    } else if nonneg {
        Interval {
            lo: 0.0,
            hi: f64::INFINITY,
        }
    } else if nonpos {
        Interval {
            lo: f64::NEG_INFINITY,
            hi: 0.0,
        }
    } else {
        Interval::UNBOUNDED
    }
}

fn mul(a: Interval, b: Interval) -> Interval {
    // Multiplying by an exact zero gives zero, whatever the other side ranges
    // over. Reaching this through the endpoint products below would compute
    // 0 * inf = NaN and fall back to unbounded, losing the fact that a
    // denominator like `0.0 × INPUT` can never be anything but zero.
    if a.is_zero() || b.is_zero() {
        return Interval::point(0.0);
    }
    let candidates = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candidates {
        if c.is_nan() {
            return sign_bounds(a, b, false);
        }
        lo = lo.min(c);
        hi = hi.max(c);
    }
    Interval { lo, hi }
}

fn div(a: Interval, b: Interval) -> Interval {
    if b.contains_zero() {
        return Interval::UNBOUNDED;
    }
    let candidates = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for c in candidates {
        // inf / inf is NaN, but the quotient's sign is still determined.
        if c.is_nan() {
            return sign_bounds(a, b, true);
        }
        lo = lo.min(c);
        hi = hi.max(c);
    }
    Interval { lo, hi }
}

use std::cell::Cell;

thread_local! {
    /// The unit roundoff every result is widened by: 2^-53 at f64, 2^-24 at
    /// f32. Narrow numbers round harder, and a bound found at f64 does not
    /// carry over to f32.
    static ROUNDOFF: Cell<f64> = const { Cell::new(1.0 / 9_007_199_254_740_992.0) };
}

fn unit_roundoff() -> f64 {
    ROUNDOFF.with(|r| r.get())
}

/// Widen a range to cover the rounding the hardware will apply to it.
///
/// Without this the analysis reasons about ℝ while the kernel computes in
/// binary64, and identities that hold exactly over the reals — `(x*x)/x = x`,
/// say — can fail on the machine.
fn rounded(iv: Interval) -> Interval {
    if iv.lo.is_nan() || iv.hi.is_nan() {
        return Interval::UNBOUNDED;
    }
    let out = |v: f64, up: bool| {
        let u = unit_roundoff();
        let factor = if (v >= 0.0) == up { 1.0 + u } else { 1.0 - u };
        v * factor
    };
    Interval {
        lo: out(iv.lo, false),
        hi: out(iv.hi, true),
    }
}

/// `x * x` is never negative, but plain interval multiplication cannot see it:
/// it treats the two operands as independent and returns [-inf, inf] for an
/// unbounded x. Recognising a squared expression closes that gap.
fn square(base: Interval) -> Interval {
    let a = base.lo.abs();
    let b = base.hi.abs();
    let hi = a.max(b);
    let lo = if base.contains_zero() { 0.0 } else { a.min(b) };
    Interval {
        lo: lo * lo,
        hi: hi * hi,
    }
}

fn pow(base: Interval, exp: Interval) -> Interval {
    // Only a constant exponent gives a shape we can reason about.
    if exp.lo != exp.hi {
        return Interval::UNBOUNDED;
    }
    let n = exp.lo;

    if n == n.trunc() && n.abs() < 1024.0 {
        let k = n as i64;
        if k == 0 {
            return Interval::point(1.0);
        }
        if k > 0 && k % 2 == 0 {
            // Even powers are non-negative and fold around zero.
            let a = base.lo.abs();
            let b = base.hi.abs();
            let far = a.max(b).powi(k as i32);
            let near = if base.contains_zero() {
                0.0
            } else {
                a.min(b).powi(k as i32)
            };
            return Interval { lo: near, hi: far };
        }
        if k > 0 {
            // Odd powers are monotone.
            return Interval {
                lo: base.lo.powi(k as i32),
                hi: base.hi.powi(k as i32),
            };
        }
    }

    // Fractional or negative exponents need a positive base to stay real.
    if base.lo > 0.0 {
        let a = base.lo.powf(n);
        let b = base.hi.powf(n);
        if a.is_finite() && b.is_finite() {
            return Interval {
                lo: a.min(b),
                hi: a.max(b),
            };
        }
    }
    Interval::UNBOUNDED
}

/// Evaluate an expanded value in interval arithmetic.
pub fn eval_interval(sym: &Sym) -> Interval {
    match sym {
        Sym::Free(_) => Interval::UNBOUNDED,
        Sym::Const(v) => Interval::point(*v),
        Sym::Add(a, b) => {
            let (a, b) = (eval_interval(a), eval_interval(b));
            rounded(Interval {
                lo: a.lo + b.lo,
                hi: a.hi + b.hi,
            })
        }
        Sym::Sub(a, b) => {
            let (a, b) = (eval_interval(a), eval_interval(b));
            rounded(Interval {
                lo: a.lo - b.hi,
                hi: a.hi - b.lo,
            })
        }
        // A squared value stays non-negative through rounding, so widening it
        // outward keeps the useful lower bound of zero.
        Sym::Mul(a, b) if a == b => rounded(square(eval_interval(a))),
        Sym::Mul(a, b) => rounded(mul(eval_interval(a), eval_interval(b))),
        Sym::Div(a, b) => rounded(div(eval_interval(a), eval_interval(b))),
        Sym::Pow(a, b) => rounded(pow(eval_interval(a), eval_interval(b))),
        // A boundary cell contributes 0; an interior one contributes its value.
        Sym::Boundary { interior, .. } => {
            Interval::hull(Interval::point(0.0), eval_interval(interior))
        }
        // A named function's range is often much tighter than its argument's,
        // which is where interval arithmetic earns its keep: a sine is bounded
        // whatever it was given, and an exponential is always positive.
        Sym::Named { op, operand } => {
            let inner = eval_interval(operand);
            match op {
                BuiltinOp::Exp => Interval {
                    lo: if inner.lo.is_finite() { inner.lo.exp() } else { 0.0 },
                    hi: if inner.hi.is_finite() {
                        inner.hi.exp()
                    } else {
                        f64::INFINITY
                    },
                },
                BuiltinOp::Log => Interval {
                    lo: if inner.lo > 0.0 {
                        inner.lo.ln()
                    } else {
                        f64::NEG_INFINITY
                    },
                    hi: if inner.hi > 0.0 && inner.hi.is_finite() {
                        inner.hi.ln()
                    } else {
                        f64::INFINITY
                    },
                },
                BuiltinOp::Sqrt => Interval {
                    lo: if inner.lo >= 0.0 { inner.lo.sqrt() } else { 0.0 },
                    hi: if inner.hi >= 0.0 && inner.hi.is_finite() {
                        inner.hi.sqrt()
                    } else {
                        f64::INFINITY
                    },
                },
                // A sine or cosine is bounded however wild its argument.
                BuiltinOp::Sin | BuiltinOp::Cos => Interval { lo: -1.0, hi: 1.0 },
                BuiltinOp::Abs => Interval {
                    lo: if inner.contains_zero() {
                        0.0
                    } else {
                        inner.lo.abs().min(inner.hi.abs())
                    },
                    hi: inner.lo.abs().max(inner.hi.abs()),
                },
                BuiltinOp::Indicator => Interval { lo: 0.0, hi: 1.0 },
            }
        }
        // A fold is not expanded term by term: the number of terms is a
        // property of the grid, not of the cell the constraint speaks about.
        // What is known is what the operator can produce.
        Sym::Fold { op, .. } => match op {
            // A sum or product of unknown reals is unknown.
            FoldOp::Sum | FoldOp::Product => Interval::UNBOUNDED,
            // A max or min is one of the operands, so it inherits their range.
            FoldOp::Max | FoldOp::Min => Interval::UNBOUNDED,
        },
        // A masked value is either the left operand where the test holds, or 0.
        Sym::Mask { cmp, lhs, rhs } => {
            let l = eval_interval(lhs);
            let r = eval_interval(rhs);
            let passing = match cmp {
                Cmp::Gt | Cmp::Gte => Interval {
                    lo: l.lo.max(r.lo),
                    hi: l.hi,
                },
                Cmp::Lt | Cmp::Lte => Interval {
                    lo: l.lo,
                    hi: l.hi.min(r.hi),
                },
                Cmp::Eq => Interval {
                    lo: l.lo.max(r.lo),
                    hi: l.hi.min(r.hi),
                },
            };
            // An empty passing range means the mask always yields 0.
            if passing.lo > passing.hi {
                Interval::point(0.0)
            } else {
                Interval::hull(Interval::point(0.0), passing)
            }
        }
    }
}

/// Decide one obligation by interval arithmetic. Public so a caller can
/// compare the two backends on the same program.
pub fn check_interval(cmp: Cmp, lhs: &Sym, rhs: &Sym) -> Verdict {
    let l = eval_interval(lhs);
    let r = eval_interval(rhs);
    let diff = Interval {
        lo: l.lo - r.hi,
        hi: l.hi - r.lo,
    };
    let range = format!("value range [{}, {}]", fmt_bound(l.lo), fmt_bound(l.hi));

    match cmp {
        Cmp::Gte if diff.lo >= 0.0 => Verdict::Proved,
        Cmp::Gte if diff.hi < 0.0 => Verdict::Violated(format!("always below the bound; {range}")),
        Cmp::Gt if diff.lo > 0.0 => Verdict::Proved,
        Cmp::Gt if diff.hi <= 0.0 => Verdict::Violated(format!("never above the bound; {range}")),
        Cmp::Lte if diff.hi <= 0.0 => Verdict::Proved,
        Cmp::Lte if diff.lo > 0.0 => Verdict::Violated(format!("always above the bound; {range}")),
        Cmp::Lt if diff.hi < 0.0 => Verdict::Proved,
        Cmp::Lt if diff.lo >= 0.0 => Verdict::Violated(format!("never below the bound; {range}")),
        Cmp::Eq if diff.is_zero() => Verdict::Proved,
        Cmp::Eq if !diff.contains_zero() => {
            Verdict::Violated(format!("the two sides cannot be equal; {range}"))
        }
        _ => Verdict::Unproven(format!(
            "intervals leave it open; {range}"
        )),
    }
}

/// Decide whether an argument stays inside a named function's domain.
pub fn check_domain(argument: &Sym, domain: Domain) -> Verdict {
    let range = eval_interval(argument);
    let (inside, outside) = match domain {
        Domain::Positive => (range.lo > 0.0, range.hi <= 0.0),
        Domain::NonNegative => (range.lo >= 0.0, range.hi < 0.0),
    };
    if inside {
        Verdict::Proved
    } else if outside {
        Verdict::Violated(format!(
            "the argument {domain} but ranges over [{}, {}]",
            fmt_bound(range.lo),
            fmt_bound(range.hi)
        ))
    } else {
        Verdict::Unproven(format!(
            "the argument {domain}; it ranges over [{}, {}]",
            fmt_bound(range.lo),
            fmt_bound(range.hi)
        ))
    }
}

/// Decide whether a denominator can vanish, by interval arithmetic.
pub fn check_denominator(denom: &Sym) -> Verdict {
    let d = eval_interval(denom);
    if d.is_zero() {
        Verdict::Violated("the denominator is always zero".to_string())
    } else if d.contains_zero() {
        Verdict::Unproven(format!(
            "the denominator ranges over [{}, {}], which includes zero",
            fmt_bound(d.lo),
            fmt_bound(d.hi)
        ))
    } else {
        Verdict::Proved
    }
}

pub fn fmt_bound(v: f64) -> String {
    if v == f64::NEG_INFINITY {
        "-inf".to_string()
    } else if v == f64::INFINITY {
        "+inf".to_string()
    } else {
        format!("{v}")
    }
}
