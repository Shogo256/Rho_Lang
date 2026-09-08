//! Symbolic expansion of a ρ program, shared by every constraint backend.
//!
//! A constraint talks about one cell, but that cell's value is defined by the
//! flows that ran before it, and a shift reaches into neighbouring cells. This
//! module inlines those definitions into a single expression tree whose only
//! free terms are cells of spaces nobody writes — that is, the caller's input —
//! and boundary flags. Both are genuinely free, so a counterexample found over
//! this tree corresponds to a real input.

use crate::ast::*;
use std::collections::BTreeMap;
use std::fmt;

/// Comparison used by a masking operator or a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
}

impl Cmp {
    pub fn from_op(op: &BinaryOpKind) -> Option<Cmp> {
        match op {
            BinaryOpKind::Gt => Some(Cmp::Gt),
            BinaryOpKind::Lt => Some(Cmp::Lt),
            BinaryOpKind::Gte => Some(Cmp::Gte),
            BinaryOpKind::Lte => Some(Cmp::Lte),
            BinaryOpKind::Eq => Some(Cmp::Eq),
            _ => None,
        }
    }
}

impl fmt::Display for Cmp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Cmp::Gt => ">",
            Cmp::Lt => "<",
            Cmp::Gte => ">=",
            Cmp::Lte => "<=",
            Cmp::Eq => "==",
        };
        write!(f, "{s}")
    }
}

/// A fully expanded scalar value at one cell.
///
/// `PartialEq` is structural, which is what lets the interval backend notice
/// that both operands of a product are the same expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Sym {
    /// A cell of an unwritten space, named `SPACE@offset`. Ranges over all reals.
    Free(String),
    Const(f64),
    Add(Box<Sym>, Box<Sym>),
    Sub(Box<Sym>, Box<Sym>),
    Mul(Box<Sym>, Box<Sym>),
    Div(Box<Sym>, Box<Sym>),
    Pow(Box<Sym>, Box<Sym>),
    /// `lhs` where the comparison holds, `0` elsewhere.
    Mask {
        cmp: Cmp,
        lhs: Box<Sym>,
        rhs: Box<Sym>,
    },
    /// A shift that may or may not sit on a boundary; `flag` names the choice.
    Boundary {
        flag: String,
        interior: Box<Sym>,
    },
}

/// A constraint lifted to `lhs cmp rhs` over expanded values.
pub struct Obligation {
    pub source: String,
    pub cmp: Cmp,
    pub lhs: Sym,
    pub rhs: Sym,
}

pub struct Expansion {
    pub obligations: Vec<Obligation>,
    /// Every division that appears in a lowered flow, with its denominator.
    pub divisions: Vec<(String, Sym)>,
}

struct Builder<'a> {
    /// Flow definitions in source order: (target, source expression).
    defs: &'a [(String, Expr)],
    shapes: &'a BTreeMap<String, Vec<usize>>,
    fallback_shape: Vec<usize>,
    elements: usize,
    tau: f64,
    flags: usize,
}

impl Builder<'_> {
    fn shape_of(&self, name: &str) -> Vec<usize> {
        self.shapes
            .get(name)
            .filter(|s| s.iter().product::<usize>() == self.elements)
            .cloned()
            .unwrap_or_else(|| self.fallback_shape.clone())
    }

    /// Expand `expr` as it would be evaluated by flow `before` at cell offset
    /// `offset`. Substituting a space always moves to an earlier flow, so the
    /// recursion is bounded by the number of flows.
    fn build(&mut self, expr: &Expr, before: usize, offset: i64) -> Sym {
        match expr {
            Expr::Number(v) => Sym::Const(*v),
            Expr::Var(name) if is_tau(name) => Sym::Const(self.tau),
            Expr::Var(name) => match self.definition(name, before) {
                Some((idx, def)) => self.build(&def, idx, offset),
                None => Sym::Free(format!("{name}@{offset}")),
            },
            Expr::AuditTrace(inner) => self.build(inner, before, offset),
            Expr::Shift { dir, axis, operand } => {
                let Some(name) = place_name(operand) else {
                    return Sym::Free(format!("shift@{offset}"));
                };
                let shape = self.shape_of(&name);
                let Some((stride, extent)) = axis_geometry(&shape, *axis) else {
                    return Sym::Const(0.0);
                };
                if extent <= 1 {
                    return Sym::Const(0.0);
                }
                let next = match dir {
                    ShiftDir::Positive => offset - stride as i64,
                    ShiftDir::Negative => offset + stride as i64,
                };
                self.flags += 1;
                let flag = format!("edge{}", self.flags);
                let interior = self.build(&Expr::Var(name), before, next);
                Sym::Boundary {
                    flag,
                    interior: Box::new(interior),
                }
            }
            Expr::BinaryOp { op, lhs, rhs } => {
                let l = self.build(lhs, before, offset);
                let r = self.build(rhs, before, offset);
                match op {
                    BinaryOpKind::Add => Sym::Add(Box::new(l), Box::new(r)),
                    BinaryOpKind::Sub => Sym::Sub(Box::new(l), Box::new(r)),
                    BinaryOpKind::Mul => Sym::Mul(Box::new(l), Box::new(r)),
                    BinaryOpKind::Div => Sym::Div(Box::new(l), Box::new(r)),
                    BinaryOpKind::Pow => Sym::Pow(Box::new(l), Box::new(r)),
                    other => Sym::Mask {
                        cmp: Cmp::from_op(other).unwrap_or(Cmp::Eq),
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                }
            }
        }
    }

    /// The definition of `name` in force just before flow `before`.
    fn definition(&self, name: &str, before: usize) -> Option<(usize, Expr)> {
        self.defs[..before.min(self.defs.len())]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, (target, _))| target == name)
            .map(|(idx, (_, expr))| (idx, expr.clone()))
    }
}

/// Expand every `!` constraint and collect the divisions the program performs.
pub fn expand(block: &ToposBlock, tau: f64) -> Expansion {
    let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for stmt in &block.statements {
        match stmt {
            Statement::SpaceDef(d) => {
                shapes.insert(d.name.clone(), d.dimensions.clone());
            }
            Statement::ExtBind(b) => {
                shapes.insert(b.space.name.clone(), b.space.dimensions.clone());
            }
            _ => {}
        }
    }
    let fallback_shape = shapes
        .get("INPUT")
        .cloned()
        .or_else(|| shapes.values().next().cloned())
        .unwrap_or_else(|| vec![4]);
    let elements = fallback_shape.iter().product::<usize>().max(1);

    // Flow definitions in order, and where each constraint sits among them.
    let mut defs: Vec<(String, Expr)> = Vec::new();
    let mut constraints: Vec<(usize, Expr)> = Vec::new();
    for stmt in &block.statements {
        match stmt {
            Statement::Flow { src, target } => {
                let name = match target {
                    FlowTarget::Var(n) => n.clone(),
                    FlowTarget::Equilibrium => "OUTPUT".to_string(),
                };
                shapes
                    .entry(name.clone())
                    .or_insert_with(|| fallback_shape.clone());
                defs.push((name, src.clone()));
            }
            Statement::Constraint(expr) => constraints.push((defs.len(), expr.clone())),
            _ => {}
        }
    }

    let defs_snapshot = defs.clone();
    let mut builder = Builder {
        defs: &defs_snapshot,
        shapes: &shapes,
        fallback_shape,
        elements,
        tau,
        flags: 0,
    };

    // One entry per division written in the source, with its denominator
    // expanded in the context of the flow that performs it.
    let mut divisions = Vec::new();
    for (idx, (_, expr)) in defs_snapshot.iter().enumerate() {
        collect_divisions(expr, idx, &mut builder, &mut divisions);
    }

    let mut obligations = Vec::new();
    for (at, expr) in &constraints {
        let source = format!("{}", ExprGlyphs(expr));
        let (cmp, lhs, rhs) = match expr {
            Expr::BinaryOp { op, lhs, rhs } if Cmp::from_op(op).is_some() => (
                Cmp::from_op(op).unwrap(),
                builder.build(lhs, *at, 0),
                builder.build(rhs, *at, 0),
            ),
            other => (Cmp::Eq, builder.build(other, *at, 0), Sym::Const(0.0)),
        };
        obligations.push(Obligation {
            source,
            cmp,
            lhs,
            rhs,
        });
    }

    Expansion {
        obligations,
        divisions,
    }
}

/// Record every division written in `expr`, expanding each denominator as flow
/// `at` would evaluate it.
fn collect_divisions(
    expr: &Expr,
    at: usize,
    builder: &mut Builder,
    out: &mut Vec<(String, Sym)>,
) {
    match expr {
        Expr::BinaryOp { op, lhs, rhs } => {
            if matches!(op, BinaryOpKind::Div) {
                let denom = builder.build(rhs, at, 0);
                out.push((format!("{}", ExprGlyphs(expr)), denom));
            }
            collect_divisions(lhs, at, builder, out);
            collect_divisions(rhs, at, builder, out);
        }
        Expr::Shift { operand: inner, .. } | Expr::AuditTrace(inner) => {
            collect_divisions(inner, at, builder, out)
        }
        Expr::Var(_) | Expr::Number(_) => {}
    }
}

fn is_tau(name: &str) -> bool {
    name == "𝜏" || name == "τ"
}

fn place_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Var(name) => Some(name.clone()),
        Expr::AuditTrace(inner) => place_name(inner),
        _ => None,
    }
}

/// Render an expression back in RHO glyphs for diagnostics.
pub struct ExprGlyphs<'a>(pub &'a Expr);

impl fmt::Display for ExprGlyphs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Expr::Number(v) => write!(f, "{v}"),
            Expr::Var(name) => write!(f, "{name}"),
            Expr::AuditTrace(inner) => write!(f, "$ {}", ExprGlyphs(inner)),
            Expr::Shift { dir, axis, operand } => match axis {
                Some(a) => write!(f, "{dir}{a}{}", ExprGlyphs(operand)),
                None => write!(f, "{dir}{}", ExprGlyphs(operand)),
            },
            Expr::BinaryOp { op, lhs, rhs } => {
                write!(f, "({} {} {})", ExprGlyphs(lhs), op, ExprGlyphs(rhs))
            }
        }
    }
}
