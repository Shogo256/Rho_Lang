// SPDX-License-Identifier: Apache-2.0
//! `rhoc --constant-time`: the rules a program keeps when its cells are
//! secrets, checked on the program after its functions are expanded.
//!
//! ρ has no branch: every cell evaluates the same expression, so the
//! instructions a sweep runs never depend on the values. Three things still
//! can, and this refuses them:
//!
//! - a division or residue by anything but a constant, and `sqrt`, `exp`,
//!   `log`, `sin`, `cos` or a power that is not a whole literal — on many
//!   processors their latency depends on the operands;
//! - `⇒`, whose number of rounds depends on the data;
//! - a `⌷` whose positions are computed from anything but `⍳`, constants and
//!   the spaces named public (and spaces made from those alone): the address
//!   it reads would follow a secret.
//!
//! Shifts, turns, folds and masks read and write fixed places and are fine.

use crate::ast::{BinaryOpKind, BuiltinOp, Expr, FlowTarget, Statement, ToposBlock};
use std::collections::BTreeSet;

/// One place the program breaks a rule: its line and what it does.
#[derive(Debug, Clone, PartialEq)]
pub struct Breach {
    pub line: usize,
    pub detail: String,
}

/// Whether the expression is made of literals only.
fn constant(expr: &Expr) -> bool {
    match expr {
        Expr::Number(_) => true,
        Expr::BinaryOp { lhs, rhs, .. } => constant(lhs) && constant(rhs),
        Expr::AuditTrace(inner) => constant(inner),
        _ => false,
    }
}

/// Whether every value the expression reads is public: a literal, a
/// coordinate (`⍳` measures a shape, never a value), or a public space.
fn public(expr: &Expr, known: &BTreeSet<String>) -> bool {
    match expr {
        Expr::Number(_) | Expr::Index { .. } => true,
        Expr::Var(name) => known.contains(name) || name == "𝜏" || name == "τ",
        Expr::AuditTrace(inner)
        | Expr::Builtin { operand: inner, .. }
        | Expr::Shift { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. } => public(inner, known),
        Expr::BinaryOp { lhs, rhs, .. } | Expr::Gather { index: lhs, operand: rhs } => {
            public(lhs, known) && public(rhs, known)
        }
        Expr::Call { args, .. } => args.iter().all(|a| public(a, known)),
    }
}

/// Every breach in one expression, evaluated at `line`.
fn breaches(expr: &Expr, known: &BTreeSet<String>, line: usize, out: &mut Vec<Breach>) {
    let mut say = |detail: &str| out.push(Breach { line, detail: detail.to_string() });
    match expr {
        Expr::BinaryOp { op, lhs, rhs } => {
            match op {
                BinaryOpKind::Div if !constant(rhs) => {
                    say("`/` divides by a value that is not a constant; its latency can follow it")
                }
                // `A | B` is B modulo A: A is the divisor.
                BinaryOpKind::Residue if !constant(lhs) => {
                    say("`|` takes a residue modulo a value that is not a constant")
                }
                BinaryOpKind::Pow if crate::ast::whole_exponent(rhs).is_none() => {
                    say("`^` with an exponent that is not a whole literal is a transcendental")
                }
                _ => {}
            }
            breaches(lhs, known, line, out);
            breaches(rhs, known, line, out);
        }
        Expr::Builtin { op, operand } => {
            if matches!(op, BuiltinOp::Sqrt | BuiltinOp::Exp | BuiltinOp::Log | BuiltinOp::Sin | BuiltinOp::Cos) {
                say(&format!("`{}` has a latency that can follow its operand", op.name()));
            }
            breaches(operand, known, line, out);
        }
        Expr::Gather { index, operand } => {
            if !public(index, known) {
                say("`⌷` reads at positions computed from a value that is not public; name the tables with --public");
            }
            breaches(index, known, line, out);
            breaches(operand, known, line, out);
        }
        Expr::AuditTrace(inner)
        | Expr::Shift { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. } => breaches(inner, known, line, out),
        Expr::Call { args, .. } => {
            for a in args {
                breaches(a, known, line, out);
            }
        }
        Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } => {}
    }
}

/// Check the program; `public` names the spaces that hold no secret (the
/// tables). A space every one of whose flows reads only public values is
/// public too, so positions may pass through intermediates.
pub fn check(block: &ToposBlock, public_spaces: &BTreeSet<String>) -> Vec<Breach> {
    let mut known = public_spaces.clone();
    let mut tainted: BTreeSet<String> = BTreeSet::new();
    let mut out = Vec::new();
    for (index, stmt) in block.statements.iter().enumerate() {
        let line = block.line_of(index);
        match stmt {
            Statement::Flow { src, target } => {
                breaches(src, &known, line, &mut out);
                let name = match target {
                    FlowTarget::Var(n) => n.clone(),
                    FlowTarget::Equilibrium => "OUTPUT".to_string(),
                };
                if public(src, &known) && !tainted.contains(&name) {
                    known.insert(name);
                } else {
                    known.remove(&name);
                    tainted.insert(name);
                }
            }
            Statement::Iterate { .. } => out.push(Breach {
                line,
                detail: "`⇒` runs a number of rounds that depends on the data".to_string(),
            }),
            Statement::Constraint(_) | Statement::AuditTrace(_) => {}
            Statement::SpaceDef(_) | Statement::ExtBind(_) => {}
        }
    }
    out
}
