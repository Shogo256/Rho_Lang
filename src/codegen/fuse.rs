// SPDX-License-Identifier: Apache-2.0
//! Flow fusion for the compiled kernel.
//!
//! Each `→` is a sweep that writes its whole space, and the next flow reads
//! it back. When exactly one later flow reads an intermediate, and reads it
//! cell for cell — a plain name, under a lift, inside a fold's operand, as
//! the index side of `⌷` — the intermediate need not exist between the two:
//! its expression goes into the reader's, and the reader's sweep computes
//! both. Nothing in the language observes the difference (a cell is one
//! expression of other cells), and the bits are the same operations in the
//! same order, only not stored and loaded in between.
//!
//! What is not fused: a space read by a shift, a turn, `⍳` or the gathered
//! side of `⌷` (those take a place in a declared space); a space read by
//! more than one statement (the reader would recompute it); an expression
//! with a turn or a gather in it (they would force the reader's sweep onto
//! the scalar path); an intermediate a loop reads (recomputed every round);
//! and a reader that writes something the expression reads, or has such a
//! write between it and the flow (the values would no longer be the ones
//! the flow saw). The space stays declared, and its flow still runs when a
//! caller of `rho_kernel_exec_spaces` passes a buffer for it.
use crate::ast::*;
use std::collections::{BTreeMap, BTreeSet};

fn is_tau(name: &str) -> bool {
    name == "𝜏" || name == "τ"
}

/// Every space an expression reads.
fn reads(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Var(name) => {
            if !is_tau(name) {
                out.insert(name.clone());
            }
        }
        Expr::Number(_) => {}
        Expr::AuditTrace(inner)
        | Expr::Builtin { operand: inner, .. }
        | Expr::Shift { operand: inner, .. }
        | Expr::Index { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. } => reads(inner, out),
        Expr::BinaryOp { lhs, rhs, .. } | Expr::Gather { index: lhs, operand: rhs } => {
            reads(lhs, out);
            reads(rhs, out);
        }
        Expr::Call { args, .. } => {
            for a in args {
                reads(a, out);
            }
        }
    }
}

/// How many times the expression names the space.
fn occurrences(expr: &Expr, name: &str) -> usize {
    match expr {
        Expr::Var(n) => usize::from(n == name),
        Expr::Number(_) => 0,
        Expr::AuditTrace(inner)
        | Expr::Builtin { operand: inner, .. }
        | Expr::Shift { operand: inner, .. }
        | Expr::Index { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. } => occurrences(inner, name),
        Expr::BinaryOp { lhs, rhs, .. } | Expr::Gather { index: lhs, operand: rhs } => {
            occurrences(lhs, name) + occurrences(rhs, name)
        }
        Expr::Call { args, .. } => args.iter().map(|a| occurrences(a, name)).sum(),
    }
}

/// Whether every read of the space is by the cell itself: never under a
/// shift, a turn, `⍳`, or as what `⌷` gathers from, which take a place in
/// a declared space.
fn pointwise(expr: &Expr, name: &str, placed: bool) -> bool {
    match expr {
        Expr::Var(n) => n != name || !placed,
        Expr::Number(_) => true,
        Expr::AuditTrace(inner)
        | Expr::Builtin { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. } => pointwise(inner, name, placed),
        Expr::Shift { operand: inner, .. }
        | Expr::Index { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. } => pointwise(inner, name, true),
        Expr::BinaryOp { lhs, rhs, .. } => pointwise(lhs, name, placed) && pointwise(rhs, name, placed),
        Expr::Gather { index, operand } => pointwise(index, name, placed) && pointwise(operand, name, true),
        Expr::Call { args, .. } => args.iter().all(|a| pointwise(a, name, placed)),
    }
}

/// Whether the expression may stand inside another flow's sweep: nothing in
/// it forces the scalar path or reads by place (a turn, `⍳`, a gather), and
/// no call is left in it.
fn portable(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) | Expr::Number(_) => true,
        Expr::AuditTrace(inner)
        | Expr::Builtin { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Shift { operand: inner, .. } => portable(inner),
        Expr::BinaryOp { lhs, rhs, .. } => portable(lhs) && portable(rhs),
        Expr::Index { .. }
        | Expr::Rotate { .. }
        | Expr::Reverse { .. }
        | Expr::Reshape { .. }
        | Expr::Transpose { .. }
        | Expr::Take { .. }
        | Expr::Drop { .. }
        | Expr::Gather { .. }
        | Expr::Call { .. } => false,
    }
}

/// The expression with every read of the space replaced by `value`.
fn substitute(expr: &Expr, name: &str, value: &Expr) -> Expr {
    let sub = |e: &Expr| Box::new(substitute(e, name, value));
    match expr {
        Expr::Var(n) if n == name => value.clone(),
        Expr::Var(_) | Expr::Number(_) => expr.clone(),
        Expr::AuditTrace(inner) => Expr::AuditTrace(sub(inner)),
        Expr::Builtin { op, operand } => Expr::Builtin { op: *op, operand: sub(operand) },
        Expr::Shift { dir, axis, operand } => Expr::Shift { dir: *dir, axis: *axis, operand: sub(operand) },
        Expr::Index { axis, operand } => Expr::Index { axis: *axis, operand: sub(operand) },
        Expr::Lift { axis, operand } => Expr::Lift { axis: *axis, operand: sub(operand) },
        Expr::Reduce { op, axis, operand } => Expr::Reduce { op: *op, axis: *axis, operand: sub(operand) },
        Expr::Scan { op, axis, operand } => Expr::Scan { op: *op, axis: *axis, operand: sub(operand) },
        Expr::BinaryOp { op, lhs, rhs } => Expr::BinaryOp { op: op.clone(), lhs: sub(lhs), rhs: sub(rhs) },
        Expr::Rotate { by, axis, operand } => Expr::Rotate { by: *by, axis: *axis, operand: sub(operand) },
        Expr::Reverse { axis, operand } => Expr::Reverse { axis: *axis, operand: sub(operand) },
        Expr::Reshape { shape, operand } => Expr::Reshape { shape: shape.clone(), operand: sub(operand) },
        Expr::Transpose { axes, operand } => Expr::Transpose { axes: axes.clone(), operand: sub(operand) },
        Expr::Take { count, axis, operand } => Expr::Take { count: *count, axis: *axis, operand: sub(operand) },
        Expr::Drop { count, axis, operand } => Expr::Drop { count: *count, axis: *axis, operand: sub(operand) },
        Expr::Gather { index, operand } => Expr::Gather { index: sub(index), operand: sub(operand) },
        Expr::Call { name: f, args } => Expr::Call { name: f.clone(), args: args.iter().map(|a| substitute(a, name, value)).collect() },
    }
}

/// The spaces a statement writes when it runs.
fn writes_of(stmt: &Statement) -> Vec<String> {
    match stmt {
        Statement::Flow { target: FlowTarget::Var(t), .. } => vec![t.clone()],
        Statement::Flow { target: FlowTarget::Equilibrium, .. } => vec!["OUTPUT".to_string()],
        Statement::Iterate { prelude, target, .. } => {
            let mut w = vec![target.clone()];
            w.extend(prelude.iter().flat_map(writes_of));
            w
        }
        _ => Vec::new(),
    }
}

/// The expressions a statement evaluates when it runs. A constraint and a
/// trace are settled at compile time and read nothing then.
fn evaluated(stmt: &Statement) -> Vec<&Expr> {
    match stmt {
        Statement::Flow { src, .. } => vec![src],
        Statement::Iterate { prelude, src, .. } => {
            let mut e: Vec<&Expr> = prelude.iter().flat_map(evaluated).collect();
            e.push(src);
            e
        }
        _ => Vec::new(),
    }
}

/// More reads of one space in one expression than this and the fusion is
/// not attempted: the reader's IR would carry that many copies for LLVM to
/// fold back together.
const MOST_READS: usize = 8;

/// The block with every fusable intermediate folded into its reader, and
/// the names of those intermediates, whose flows run only when a caller
/// asked for the space.
pub fn fuse(block: &ToposBlock) -> (ToposBlock, BTreeSet<String>) {
    let mut statements = block.statements.clone();
    let mut fused: BTreeSet<String> = BTreeSet::new();
    let mut writes: BTreeMap<String, usize> = BTreeMap::new();
    for stmt in &statements {
        for w in writes_of(stmt) {
            *writes.entry(w).or_insert(0) += 1;
        }
    }
    let written_once = |name: &str| writes.get(name) == Some(&1);

    // The program's flows, in order: a fused reader may itself be fused on.
    for i in 0..statements.len() {
        let (src, name) = match &statements[i] {
            Statement::Flow {
                src,
                target: FlowTarget::Var(t),
            } if t != "OUTPUT" => (src.clone(), t.clone()),
            _ => continue,
        };
        if !written_once(&name) || !portable(&src) {
            continue;
        }
        let readers: Vec<usize> = (i + 1..statements.len())
            .filter(|&j| evaluated(&statements[j]).iter().any(|e| occurrences(e, &name) > 0))
            .collect();
        let [j] = readers[..] else { continue };
        let Statement::Flow {
            src: reader,
            target: reader_target,
        } = &statements[j]
        else {
            continue; // a loop would recompute it every round
        };
        if !pointwise(reader, &name, false) || occurrences(reader, &name) > MOST_READS {
            continue;
        }
        let mut sources = BTreeSet::new();
        reads(&src, &mut sources);
        let reader_writes = match reader_target {
            FlowTarget::Var(t) => t.clone(),
            FlowTarget::Equilibrium => "OUTPUT".to_string(),
        };
        if sources.contains(&reader_writes) {
            continue;
        }
        if (i + 1..j).any(|k| writes_of(&statements[k]).iter().any(|w| sources.contains(w))) {
            continue;
        }
        let fused_src = substitute(reader, &name, &src);
        if let Statement::Flow { src, .. } = &mut statements[j] {
            *src = fused_src;
        }
        fused.insert(name);
    }

    // A loop's prelude: a flow read once by a later flow of the prelude or
    // by the update, and by nothing after the loop.
    let read_after: Vec<BTreeSet<String>> = (0..statements.len())
        .map(|i| {
            let mut names = BTreeSet::new();
            for stmt in &statements[i + 1..] {
                for e in evaluated(stmt) {
                    reads(e, &mut names);
                }
            }
            names
        })
        .collect();
    for (i, stmt) in statements.iter_mut().enumerate() {
        let Statement::Iterate { prelude, src: update, .. } = stmt else { continue };
        for p in 0..prelude.len() {
            let (e, name) = match &prelude[p] {
                Statement::Flow {
                    src,
                    target: FlowTarget::Var(t),
                } => (src.clone(), t.clone()),
                _ => continue,
            };
            if !written_once(&name) || !portable(&e) || read_after[i].contains(&name) {
                continue;
            }
            let mut readers: Vec<Option<usize>> = (p + 1..prelude.len())
                .filter(|&q| evaluated(&prelude[q]).iter().any(|x| occurrences(x, &name) > 0))
                .map(Some)
                .collect();
            if occurrences(update, &name) > 0 {
                readers.push(None);
            }
            let [reader] = readers[..] else { continue };
            let mut sources = BTreeSet::new();
            reads(&e, &mut sources);
            let (reader_expr, reader_writes, between) = match reader {
                Some(q) => {
                    let Statement::Flow {
                        src: r,
                        target: FlowTarget::Var(t),
                    } = &prelude[q]
                    else {
                        continue;
                    };
                    (r, Some(t.clone()), p + 1..q)
                }
                // The update writes the loop's target into the next grid and
                // reads the current one, so reading the target here is fine.
                None => (&*update, None, p + 1..prelude.len()),
            };
            if !pointwise(reader_expr, &name, false) || occurrences(reader_expr, &name) > MOST_READS {
                continue;
            }
            if reader_writes.as_ref().is_some_and(|t| sources.contains(t)) {
                continue;
            }
            if between.clone().any(|k| writes_of(&prelude[k]).iter().any(|w| sources.contains(w))) {
                continue;
            }
            let fused_expr = substitute(reader_expr, &name, &e);
            match reader {
                Some(q) => {
                    if let Statement::Flow { src, .. } = &mut prelude[q] {
                        *src = fused_expr;
                    }
                }
                None => *update = fused_expr,
            }
            fused.insert(name);
        }
    }

    (
        ToposBlock {
            statements,
            lines: block.lines.clone(),
            origins: block.origins.clone(),
        },
        fused,
    )
}
