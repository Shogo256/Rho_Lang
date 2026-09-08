//! SMT backend for `!` constraints, built on Z3.
//!
//! Each obligation is discharged by asserting its negation: `unsat` means the
//! constraint holds for every input, `sat` hands back a counterexample. Terms
//! Z3 cannot reason about exactly (a fractional exponent, say) are replaced by
//! fresh unconstrained reals. That only ever widens the search space, so a
//! proof stays sound; a counterexample found under an approximation is reported
//! as unproven rather than as a violation.

use super::{Finding, Report, Verdict};
use crate::symbolic::{Cmp, Expansion, Sym};
use std::collections::BTreeMap;
use z3::ast::{Ast, Bool, Int, Real};
use z3::{Config, Context, SatResult, Solver};

/// Give up on a single obligation rather than stalling a build.
const SOLVER_TIMEOUT_MS: u32 = 5_000;

/// 2^-53, the unit roundoff of binary64 under round-to-nearest.
const ROUNDOFF_DENOMINATOR: i64 = 9_007_199_254_740_992;

pub fn analyze(expansion: &Expansion) -> Report {
    let mut cfg = Config::new();
    cfg.set_timeout_msec(SOLVER_TIMEOUT_MS as u64);
    let ctx = Context::new(&cfg);

    let constraints = expansion
        .obligations
        .iter()
        .map(|o| Finding {
            subject: o.source.clone(),
            verdict: check_obligation(&ctx, o.cmp, &o.lhs, &o.rhs),
        })
        .collect();

    let divisions = expansion
        .divisions
        .iter()
        .map(|(text, denom)| Finding {
            subject: text.clone(),
            verdict: check_denominator(&ctx, denom),
        })
        .collect();

    Report {
        backend: "z3",
        constraints,
        divisions,
    }
}

struct Translator<'ctx> {
    ctx: &'ctx Context,
    vars: BTreeMap<String, Real<'ctx>>,
    flags: BTreeMap<String, Bool<'ctx>>,
    /// Set when a term had to be replaced by an unconstrained value.
    approximated: bool,
    fresh: usize,
    /// `|δ| ≤ 2^-53` for every rounding term introduced.
    rounding_bounds: Vec<Bool<'ctx>>,
}

impl<'ctx> Translator<'ctx> {
    fn new(ctx: &'ctx Context) -> Self {
        Self {
            ctx,
            vars: BTreeMap::new(),
            flags: BTreeMap::new(),
            approximated: false,
            fresh: 0,
            rounding_bounds: Vec::new(),
        }
    }

    fn zero(&self) -> Real<'ctx> {
        Real::from_int(&Int::from_i64(self.ctx, 0))
    }

    fn var(&mut self, name: &str) -> Real<'ctx> {
        self.vars
            .entry(name.to_string())
            .or_insert_with(|| Real::new_const(self.ctx, name))
            .clone()
    }

    fn flag(&mut self, name: &str) -> Bool<'ctx> {
        self.flags
            .entry(name.to_string())
            .or_insert_with(|| Bool::new_const(self.ctx, name))
            .clone()
    }

    fn unknown(&mut self) -> Real<'ctx> {
        self.approximated = true;
        self.fresh += 1;
        Real::new_const(self.ctx, format!("approx{}", self.fresh))
    }

    fn constant(&mut self, v: f64) -> Real<'ctx> {
        match rational(v) {
            Some((num, den)) => {
                let n = Real::from_int(&Int::from_i64(self.ctx, num));
                if den == 1 {
                    n
                } else {
                    let d = Real::from_int(&Int::from_i64(self.ctx, den));
                    n.div(&d)
                }
            }
            None => self.unknown(),
        }
    }

    /// Wrap an exact result in the rounding the hardware applies to it:
    /// `fl(a op b) = (a op b)(1 + δ)` with `|δ| ≤ 2^-53`, the standard model for
    /// round-to-nearest. Without it the solver proves identities that hold over
    /// ℝ but not in binary64 — `(x*x)/x = x` being the one that caught us.
    fn round(&mut self, exact: Real<'ctx>) -> Real<'ctx> {
        self.fresh += 1;
        let delta = Real::new_const(self.ctx, format!("delta{}", self.fresh));
        let bound = Real::from_int(&Int::from_i64(self.ctx, 1))
            .div(&Real::from_int(&Int::from_i64(self.ctx, ROUNDOFF_DENOMINATOR)));
        let zero = Real::from_int(&Int::from_i64(self.ctx, 0));
        let neg_bound = Real::sub(self.ctx, &[&zero, &bound]);
        self.rounding_bounds.push(delta.le(&bound));
        self.rounding_bounds.push(delta.ge(&neg_bound));

        let one = Real::from_int(&Int::from_i64(self.ctx, 1));
        let scale = Real::add(self.ctx, &[&one, &delta]);
        Real::mul(self.ctx, &[&exact, &scale])
    }

    fn build(&mut self, sym: &Sym) -> Real<'ctx> {
        match sym {
            Sym::Free(name) => self.var(name),
            Sym::Const(v) => self.constant(*v),
            Sym::Add(a, b) => {
                let (a, b) = (self.build(a), self.build(b));
                let exact = Real::add(self.ctx, &[&a, &b]);
                self.round(exact)
            }
            Sym::Sub(a, b) => {
                let (a, b) = (self.build(a), self.build(b));
                let exact = Real::sub(self.ctx, &[&a, &b]);
                self.round(exact)
            }
            Sym::Mul(a, b) => {
                let (a, b) = (self.build(a), self.build(b));
                let exact = Real::mul(self.ctx, &[&a, &b]);
                self.round(exact)
            }
            Sym::Div(a, b) => {
                let (a, b) = (self.build(a), self.build(b));
                let exact = a.div(&b);
                self.round(exact)
            }
            Sym::Pow(base, exp) => self.build_pow(base, exp),
            Sym::Boundary { flag, interior } => {
                let cond = self.flag(flag);
                let zero = self.zero();
                let value = self.build(interior);
                cond.ite(&zero, &value)
            }
            Sym::Mask { cmp, lhs, rhs } => {
                let l = self.build(lhs);
                let r = self.build(rhs);
                let zero = self.zero();
                let holds = compare(cmp, &l, &r);
                holds.ite(&l, &zero)
            }
        }
    }

    /// A small whole-number exponent becomes repeated multiplication, which Z3
    /// reasons about far better than a general power term.
    fn build_pow(&mut self, base: &Sym, exp: &Sym) -> Real<'ctx> {
        let Sym::Const(e) = exp else {
            return self.unknown();
        };
        if *e != e.trunc() || !(0.0..=16.0).contains(e) {
            return self.unknown();
        }
        let k = *e as u32;
        let b = self.build(base);
        if k == 0 {
            return Real::from_int(&Int::from_i64(self.ctx, 1));
        }
        let mut acc = b.clone();
        for _ in 1..k {
            let exact = Real::mul(self.ctx, &[&acc, &b]);
            acc = self.round(exact);
        }
        acc
    }
}

fn compare<'ctx>(cmp: &Cmp, l: &Real<'ctx>, r: &Real<'ctx>) -> Bool<'ctx> {
    match cmp {
        Cmp::Gt => l.gt(r),
        Cmp::Lt => l.lt(r),
        Cmp::Gte => l.ge(r),
        Cmp::Lte => l.le(r),
        Cmp::Eq => l._eq(r),
    }
}

fn check_obligation(ctx: &Context, cmp: Cmp, lhs: &Sym, rhs: &Sym) -> Verdict {
    let mut tr = Translator::new(ctx);
    let l = tr.build(lhs);
    let r = tr.build(rhs);
    let holds = compare(&cmp, &l, &r);

    let solver = Solver::new(ctx);
    for bound in &tr.rounding_bounds {
        solver.assert(bound);
    }
    solver.assert(&holds.not());

    match solver.check() {
        SatResult::Unsat => Verdict::Proved,
        SatResult::Sat if tr.approximated => Verdict::Unproven(
            "a counterexample exists only under an approximated term (fractional power \
             or unsupported operator), so it may not be reachable"
                .to_string(),
        ),
        SatResult::Sat => {
            let witness = solver
                .get_model()
                .map(|m| format_model(&m, &tr))
                .unwrap_or_default();
            Verdict::Violated(if witness.is_empty() {
                "a counterexample exists".to_string()
            } else {
                format!("counterexample: {witness}")
            })
        }
        SatResult::Unknown => Verdict::Unproven(
            solver
                .get_reason_unknown()
                .unwrap_or_else(|| "the solver returned unknown".to_string()),
        ),
    }
}

fn check_denominator(ctx: &Context, denom: &Sym) -> Verdict {
    let mut tr = Translator::new(ctx);
    let d = tr.build(denom);
    let zero = tr.zero();

    // Can the denominator be zero at all?
    let solver = Solver::new(ctx);
    for bound in &tr.rounding_bounds {
        solver.assert(bound);
    }
    solver.assert(&d._eq(&zero));
    match solver.check() {
        SatResult::Unsat => Verdict::Proved,
        SatResult::Unknown => {
            Verdict::Unproven("the solver could not decide whether the denominator is zero".into())
        }
        SatResult::Sat => {
            // Is it zero for *every* input? Only then is the program certainly wrong.
            let always = Solver::new(ctx);
            for bound in &tr.rounding_bounds {
                always.assert(bound);
            }
            always.assert(&d._eq(&zero).not());
            match always.check() {
                SatResult::Unsat => {
                    Verdict::Violated("the denominator is zero for every input".to_string())
                }
                _ if tr.approximated => Verdict::Unproven(
                    "the denominator may be zero, under an approximated term".to_string(),
                ),
                _ => {
                    let witness = solver
                        .get_model()
                        .map(|m| format_model(&m, &tr))
                        .unwrap_or_default();
                    Verdict::Unproven(if witness.is_empty() {
                        "the denominator can be zero".to_string()
                    } else {
                        format!("the denominator is zero when {witness}")
                    })
                }
            }
        }
    }
}

fn format_model(model: &z3::Model, tr: &Translator) -> String {
    let mut parts = Vec::new();
    for (name, var) in tr.vars.iter().take(6) {
        if let Some(value) = model.eval(var, true) {
            parts.push(format!("{name} = {value}"));
        }
    }
    for (name, flag) in tr.flags.iter().take(3) {
        if let Some(value) = model.eval(flag, true) {
            parts.push(format!("{name} = {value}"));
        }
    }
    parts.join(", ")
}

/// Exact rational for a literal, taken from its shortest decimal form so the
/// solver sees the number the source actually wrote.
fn rational(v: f64) -> Option<(i64, i64)> {
    if !v.is_finite() {
        return None;
    }
    if v == v.trunc() && v.abs() < 9.0e15 {
        return Some((v as i64, 1));
    }
    let text = format!("{v}");
    if text.contains('e') || text.contains('E') {
        return None;
    }
    let (int_part, frac_part) = text.split_once('.')?;
    if frac_part.len() > 17 {
        return None;
    }
    let digits = format!("{int_part}{frac_part}");
    let num: i64 = digits.parse().ok()?;
    let den = 10i64.checked_pow(frac_part.len() as u32)?;
    Some((num, den))
}
