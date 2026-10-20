// SPDX-License-Identifier: Apache-2.0
//! A program's functions as Lean 4 definitions over the integers, for
//! proofs about them: `rhoc --emit-lean FILE.lean`.
//!
//! The subset is the arithmetic of whole numbers that a kernel computes
//! exactly: `+`, `-` and `×`; `⌊` of a quotient by a positive whole
//! constant, which over the integers is Lean's `/` (Euclidean, so the floor
//! for a positive divisor); `⌊` of anything else, which of a whole number is
//! itself; `ind` of a comparison or of a value; `abs`; the greater and the
//! lesser; calls of the functions defined before. Every literal must be a
//! whole number. Where every value of a function stays an integer under
//! 2^53, binary64 computes exactly these integers, which is what makes the
//! definitions the kernel's: the fixed-point interpreter at 64.0 computes
//! them too, and checks the claim on any input it is given.
//!
//! A function outside the subset (a shift, a gather, a fold, a fraction, a
//! power, a transcendental) is not emitted; a comment names it and says
//! why, and so does every function that calls it.

use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// A name as Lean reads it whatever it is: `«...»` quotes any identifier.
fn ident(name: &str) -> String {
    format!("«{name}»")
}

fn whole(v: f64) -> Option<i64> {
    (v.is_finite() && v.fract() == 0.0 && v.abs() < 9.0e15).then_some(v as i64)
}

fn literal(v: f64) -> std::result::Result<String, String> {
    match whole(v) {
        Some(n) if n < 0 => Ok(format!("({n} : Int)")),
        Some(n) => Ok(format!("({n} : Int)")),
        None => Err(format!("the literal {v} is not a whole number")),
    }
}

struct Emitter<'a> {
    /// Functions emitted so far, which a body may call.
    known: &'a BTreeSet<String>,
}

impl Emitter<'_> {
    fn expr(&self, e: &Expr) -> std::result::Result<String, String> {
        Ok(match e {
            Expr::Var(name) => ident(name),
            Expr::Number(v) => literal(*v)?,
            Expr::AuditTrace(inner) => self.expr(inner)?,
            Expr::BinaryOp { op, lhs, rhs } => {
                let sym = match op {
                    BinaryOpKind::Add => "+",
                    BinaryOpKind::Sub => "-",
                    BinaryOpKind::Mul => "*",
                    BinaryOpKind::Max => return Ok(format!("(max {} {})", self.expr(lhs)?, self.expr(rhs)?)),
                    BinaryOpKind::Min => return Ok(format!("(min {} {})", self.expr(lhs)?, self.expr(rhs)?)),
                    BinaryOpKind::Div => return Err("`/` other than under `⌊`, by a positive whole constant".to_string()),
                    BinaryOpKind::Gt | BinaryOpKind::Lt | BinaryOpKind::Gte | BinaryOpKind::Lte | BinaryOpKind::Eq => {
                        return Err("a comparison other than under `ind`".to_string())
                    }
                    BinaryOpKind::Pow => return Err("`^`".to_string()),
                    BinaryOpKind::Residue => return Err("`|`".to_string()),
                };
                format!("({} {sym} {})", self.expr(lhs)?, self.expr(rhs)?)
            }
            Expr::Builtin { op, operand } => match op {
                BuiltinOp::Floor => match operand.as_ref() {
                    Expr::BinaryOp { op: BinaryOpKind::Div, lhs, rhs } => match rhs.as_ref() {
                        Expr::Number(c) if whole(*c).is_some_and(|n| n > 0) => {
                            format!("({} / {})", self.expr(lhs)?, literal(*c)?)
                        }
                        _ => return Err("`⌊` of a quotient by other than a positive whole constant".to_string()),
                    },
                    other => self.expr(other)?,
                },
                BuiltinOp::Indicator => match operand.as_ref() {
                    Expr::BinaryOp { op, lhs, rhs }
                        if matches!(
                            op,
                            BinaryOpKind::Gt | BinaryOpKind::Lt | BinaryOpKind::Gte | BinaryOpKind::Lte | BinaryOpKind::Eq
                        ) =>
                    {
                        let sym = match op {
                            BinaryOpKind::Gt => ">",
                            BinaryOpKind::Lt => "<",
                            BinaryOpKind::Gte => "≥",
                            BinaryOpKind::Lte => "≤",
                            _ => "=",
                        };
                        format!("(if {} {sym} {} then (1 : Int) else 0)", self.expr(lhs)?, self.expr(rhs)?)
                    }
                    other => format!("(if {} ≠ 0 then (1 : Int) else 0)", self.expr(other)?),
                },
                BuiltinOp::Abs => format!("(({}).natAbs : Int)", self.expr(operand)?),
                other => return Err(format!("`{other:?}`")),
            },
            Expr::Call { name, args } => {
                if !self.known.contains(name) {
                    return Err(format!("a call of `{name}`, which is not emitted"));
                }
                let mut s = format!("({}", ident(name));
                for a in args {
                    s.push(' ');
                    s.push_str(&self.expr(a)?);
                }
                s.push(')');
                s
            }
            Expr::Shift { .. } => return Err("a shift".to_string()),
            Expr::Rotate { .. } | Expr::Reverse { .. } => return Err("`⌽`".to_string()),
            Expr::Gather { .. } => return Err("`⌷`".to_string()),
            Expr::Reshape { .. } => return Err("`⍴`".to_string()),
            Expr::Transpose { .. } => return Err("`⍉`".to_string()),
            Expr::Take { .. } | Expr::Drop { .. } => return Err("`↑` or `↓`".to_string()),
            Expr::Index { .. } => return Err("`⍳`".to_string()),
            Expr::Lift { .. } => return Err("`□`".to_string()),
            Expr::Reduce { .. } | Expr::Scan { .. } => return Err("a fold".to_string()),
        })
    }

    fn function(&self, f: &Function) -> std::result::Result<String, String> {
        let params: Vec<String> = f.params.iter().map(|p| format!("({} : Int)", ident(p))).collect();
        let mut out = format!("def {} {} : Int :=\n", ident(&f.name), params.join(" "));
        match &f.body {
            FunctionBody::Expression(e, _) => {
                let _ = writeln!(out, "  {}", self.expr(e)?);
            }
            FunctionBody::Flows(flows) => {
                for (stmt, _) in flows {
                    match stmt {
                        Statement::Flow { src, target: FlowTarget::Var(name) } => {
                            let _ = writeln!(out, "  let {} := {}", ident(name), self.expr(src)?);
                        }
                        Statement::Flow { src, target: FlowTarget::Equilibrium } => {
                            let _ = writeln!(out, "  {}", self.expr(src)?);
                        }
                        _ => return Err("a statement other than a flow".to_string()),
                    }
                }
            }
        }
        Ok(out)
    }
}

/// The functions in Lean, in a namespace of that name; `source` names the
/// file they came from, in the header.
pub fn emit(functions: &[Function], namespace: &str, source: &str) -> Result<String> {
    if namespace.is_empty() || !namespace.split('.').all(|p| !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '_')) {
        return Err(HarmonyDisruption::LoweringErr {
            detail: format!("`{namespace}` is not a Lean namespace"),
            line: 0,
        });
    }
    let mut out = String::new();
    let _ = writeln!(out, "/-");
    let _ = writeln!(out, "Written by `rhoc --emit-lean` from {source}: its functions over the");
    let _ = writeln!(out, "integers. Not to be edited; written again from the source.");
    let _ = writeln!(out, "-/\n");
    let _ = writeln!(out, "namespace {namespace}\n");
    let mut known = BTreeSet::new();
    for f in functions {
        let emitted = Emitter { known: &known }.function(f);
        match emitted {
            Ok(text) => {
                out.push_str(&text);
                out.push('\n');
                known.insert(f.name.clone());
            }
            Err(why) => {
                let _ = writeln!(out, "-- `{}` (line {}) is not emitted: {why}.\n", f.name, f.line);
            }
        }
    }
    let _ = writeln!(out, "end {namespace}");
    Ok(out)
}
