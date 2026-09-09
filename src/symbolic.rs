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
    /// A coordinate along an axis: which cell this is, the analysis does not
    /// know, but it lies between 0 and the extent less one.
    Coordinate { name: String, extent: usize },
    Const(f64),
    Add(Box<Sym>, Box<Sym>),
    Sub(Box<Sym>, Box<Sym>),
    Mul(Box<Sym>, Box<Sym>),
    Div(Box<Sym>, Box<Sym>),
    Pow(Box<Sym>, Box<Sym>),
    /// The greater and the lesser of two values.
    Max(Box<Sym>, Box<Sym>),
    Min(Box<Sym>, Box<Sym>),
    /// APL's residue of the right by the left: the sign of the left, or the
    /// right itself when the left is zero.
    Residue(Box<Sym>, Box<Sym>),
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
    /// A named function applied to a value.
    Named {
        op: BuiltinOp,
        operand: Box<Sym>,
    },
    /// The result of folding an axis away. A constraint speaks about one cell,
    /// and a fold's cell is a function of many, so it is modelled by what the
    /// fold's operator can produce rather than expanded term by term.
    Fold {
        op: FoldOp,
        id: usize,
    },
}

/// What a named function needs of its argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// `log` is undefined at zero and below.
    Positive,
    /// `sqrt` of a negative is not a real number.
    NonNegative,
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Domain::Positive => write!(f, "must be positive"),
            Domain::NonNegative => write!(f, "must not be negative"),
        }
    }
}

/// A constraint lifted to `lhs cmp rhs` over expanded values.
pub struct Obligation {
    pub source: String,
    pub cmp: Cmp,
    pub lhs: Sym,
    pub rhs: Sym,
    pub line: usize,
}

pub struct Expansion {
    pub obligations: Vec<Obligation>,
    /// Every division that appears in a lowered flow: text, denominator, line.
    pub divisions: Vec<(String, Sym, usize)>,
    /// Every argument that has to stay inside a function's domain: the call as
    /// written, the argument, the bound it must respect, and the line.
    pub domains: Vec<(String, Sym, Domain, usize)>,
    /// The value written at the equilibrium point, expanded through every flow
    /// that produced it. This is what a caller of the kernel actually receives.
    pub output: Option<Sym>,
}

struct Builder<'a> {
    /// Flow definitions in source order: (target, source expression).
    defs: &'a [(String, Expr)],
    /// Which definitions are a `⇒` rather than a `→`.
    looped: &'a [bool],
    shapes: &'a BTreeMap<String, Vec<usize>>,
    fallback_shape: Vec<usize>,
    tau: f64,
    folds: usize,
}

impl Builder<'_> {
    /// Whether `name`, read while expanding definition `at`, is that
    /// definition's own iterate rather than an earlier flow's value.
    fn is_iterate_of(&self, at: usize, name: &str) -> bool {
        self.looped.get(at).copied().unwrap_or(false) && self.defs[at].0 == name
    }

    /// What a `⇒` leaves in its target. Nothing is assumed about it: the
    /// iterate is a free value, which is sound and is all the intervals need.
    fn iterate_value(&self, idx: usize, name: &str, offset: i64) -> Sym {
        Sym::Free(format!("{name}⇒{idx}@{}", offset_tag(offset)))
    }

    fn shape_of(&self, name: &str) -> Vec<usize> {
        self.shapes
            .get(name)
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
            Expr::Var(name) => {
                // Inside the body of a `⇒`, its own target is the iterate,
                // not the starting value the previous flow wrote.
                if self.is_iterate_of(before, name) {
                    return self.iterate_value(before, name, offset);
                }
                match self.definition(name, before) {
                    Some((idx, _)) if self.looped[idx] => self.iterate_value(idx, name, offset),
                    Some((idx, def)) => self.build(&def, idx, offset),
                    None => Sym::Free(format!("{name}@{offset}")),
                }
            }
            Expr::AuditTrace(inner) | Expr::Lift { operand: inner, .. } => {
                self.build(inner, before, offset)
            }
            // A rotation reads the same space `by` cells along; which cell
            // that is after the wrap is not known for "any cell", but every
            // cell's expansion has the same range, so the relative offset
            // serves. A reversal's partner is likewise some cell of the same
            // space, and is given this cell's own expansion for its range.
            Expr::Rotate { by, axis, operand } => match place_name(operand) {
                Some(name) => {
                    let stride = axis_geometry(&self.shape_of(&name), *axis)
                        .map(|(stride, _)| stride as i64)
                        .unwrap_or(1);
                    self.build(&Expr::Var(name), before, offset + by * stride)
                }
                None => Sym::Free(format!("rotate@{}", offset_tag(offset))),
            },
            Expr::Reverse { operand, .. } => match place_name(operand) {
                Some(name) => self.build(&Expr::Var(name), before, offset),
                None => Sym::Free(format!("reverse@{}", offset_tag(offset))),
            },
            // A constraint speaks about any cell, so its coordinate is a free
            // value within the axis; the operand is only measured.
            Expr::Index { axis, operand } => {
                let extent = place_name(operand)
                    .and_then(|name| axis_geometry(&self.shape_of(&name), *axis))
                    .map(|(_, extent)| extent);
                match extent {
                    Some(extent) => Sym::Coordinate {
                        name: format!("⍳{}@{}", axis.map(|a| a.to_string()).unwrap_or_default(), offset_tag(offset)),
                        extent,
                    },
                    None => Sym::Free(format!("⍳@{}", offset_tag(offset))),
                }
            }

            // A named function keeps the cell it was applied to, so it can be
            // reasoned about pointwise. `ind` is bounded on both ends.
            Expr::Builtin { op, operand } => {
                let inner = self.build(operand, before, offset);
                Sym::Named {
                    op: *op,
                    operand: Box::new(inner),
                }
            }
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
                // The flag names a boundary *condition*, not an occurrence. Two
                // reads of the same neighbour in one expression — `GX × GX`, say
                // — sit on the boundary together or not at all, so they must
                // share a flag. Giving each occurrence its own let the solver
                // pick different answers for the same cell and reject valid
                // programs.
                let flag = format!(
                    "edge_o{}_s{stride}_e{extent}_{}",
                    offset_tag(offset),
                    match dir {
                        ShiftDir::Positive => "p",
                        ShiftDir::Negative => "n",
                    }
                );
                let interior = self.build(&Expr::Var(name), before, next);
                Sym::Boundary {
                    flag,
                    interior: Box::new(interior),
                }
            }
            // A fold collapses many cells into one, so a single-cell view of
            // the program cannot expand it. It becomes an opaque value whose
            // range the interval backend still bounds.
            Expr::Scan { op, operand, .. } | Expr::Reduce { op, operand, .. } => {
                let _ = self.build(operand, before, offset);
                self.folds += 1;
                Sym::Fold {
                    op: *op,
                    id: self.folds,
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
                    BinaryOpKind::Max => Sym::Max(Box::new(l), Box::new(r)),
                    BinaryOpKind::Min => Sym::Min(Box::new(l), Box::new(r)),
                    BinaryOpKind::Residue => Sym::Residue(Box::new(l), Box::new(r)),
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

    // Flow definitions in order, and where each constraint sits among them.
    let mut defs: Vec<(String, Expr)> = Vec::new();
    let mut looped: Vec<bool> = Vec::new();
    let mut def_lines: Vec<usize> = Vec::new();
    let mut constraints: Vec<(usize, Expr, usize)> = Vec::new();
    for (index, stmt) in block.statements.iter().enumerate() {
        match stmt {
            Statement::Flow { src, target } => {
                let name = match target {
                    FlowTarget::Var(n) => n.clone(),
                    FlowTarget::Equilibrium => "OUTPUT".to_string(),
                };
                // The shape a flow writes is the shape of what it flows; the
                // grid's is only a last resort. A shift's boundary flag is
                // free either way, but an index's extent is not.
                let inferred = expr_shape(src, &shapes).unwrap_or_else(|| fallback_shape.clone());
                shapes.entry(name.clone()).or_insert(inferred);
                defs.push((name, src.clone()));
                looped.push(false);
                def_lines.push(block.line_of(index));
            }
            Statement::Iterate { src, target } => {
                let inferred = expr_shape(src, &shapes).unwrap_or_else(|| fallback_shape.clone());
                shapes.entry(target.clone()).or_insert(inferred);
                defs.push((target.clone(), src.clone()));
                looped.push(true);
                def_lines.push(block.line_of(index));
            }
            Statement::Constraint(expr) => {
                constraints.push((defs.len(), expr.clone(), block.line_of(index)))
            }
            _ => {}
        }
    }

    let defs_snapshot = defs.clone();
    let mut builder = Builder {
        defs: &defs_snapshot,
        looped: &looped,
        shapes: &shapes,
        fallback_shape,
        tau,
        folds: 0,
    };

    // One entry per division written in the source, with its denominator
    // expanded in the context of the flow that performs it.
    let mut divisions = Vec::new();
    let mut domains = Vec::new();
    for (idx, (_, expr)) in defs_snapshot.iter().enumerate() {
        let line = def_lines.get(idx).copied().unwrap_or(0);
        collect_divisions(expr, idx, line, &mut builder, &mut divisions);
        collect_domains(expr, idx, line, &mut builder, &mut domains);
    }

    let mut obligations = Vec::new();
    for (at, expr, line) in &constraints {
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
            line: *line,
        });
    }

    // The last flow is what the caller sees; expand it so its range can be
    // stated as part of the kernel's contract.
    let output = defs_snapshot
        .len()
        .checked_sub(1)
        .map(|last| builder.build(&defs_snapshot[last].1, last, 0));

    Expansion {
        obligations,
        divisions,
        domains,
        output,
    }
}

/// Record every division written in `expr`, expanding each denominator as flow
/// `at` would evaluate it.
fn collect_divisions(
    expr: &Expr,
    at: usize,
    line: usize,
    builder: &mut Builder,
    out: &mut Vec<(String, Sym, usize)>,
) {
    match expr {
        Expr::BinaryOp { op, lhs, rhs } => {
            if matches!(op, BinaryOpKind::Div) {
                let denom = builder.build(rhs, at, 0);
                out.push((format!("{}", ExprGlyphs(expr)), denom, line));
            }
            collect_divisions(lhs, at, line, builder, out);
            collect_divisions(rhs, at, line, builder, out);
        }
        Expr::Shift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Builtin { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::AuditTrace(inner) => {
            collect_divisions(inner, at, line, builder, out)
        }
        // An index measures its operand and never evaluates it.
        Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } => {}
    }
}

/// Render a signed offset as an identifier fragment.
fn offset_tag(offset: i64) -> String {
    if offset < 0 {
        format!("m{}", offset.unsigned_abs())
    } else {
        offset.to_string()
    }
}

/// Record every argument a named function constrains.
fn collect_domains(
    expr: &Expr,
    at: usize,
    line: usize,
    builder: &mut Builder,
    out: &mut Vec<(String, Sym, Domain, usize)>,
) {
    match expr {
        Expr::Builtin { op, operand } => {
            let needs = match op {
                BuiltinOp::Log => Some(Domain::Positive),
                BuiltinOp::Sqrt => Some(Domain::NonNegative),
                _ => None,
            };
            if let Some(domain) = needs {
                let argument = builder.build(operand, at, 0);
                out.push((format!("{}", ExprGlyphs(expr)), argument, domain, line));
            }
            collect_domains(operand, at, line, builder, out);
        }
        Expr::BinaryOp { lhs, rhs, .. } => {
            collect_domains(lhs, at, line, builder, out);
            collect_domains(rhs, at, line, builder, out);
        }
        Expr::Shift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::AuditTrace(inner) => collect_domains(inner, at, line, builder, out),
        Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } => {}
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
            Expr::Lift { axis, operand } => write!(f, "□{axis}{}", ExprGlyphs(operand)),
            Expr::Index { axis, operand } => match axis {
                Some(a) => write!(f, "⍳{a}{}", ExprGlyphs(operand)),
                None => write!(f, "⍳{}", ExprGlyphs(operand)),
            },
            Expr::Rotate { by, axis, operand } => match axis {
                Some(a) => write!(f, "({by} ⌽{a} {})", ExprGlyphs(operand)),
                None => write!(f, "({by} ⌽ {})", ExprGlyphs(operand)),
            },
            Expr::Reverse { axis, operand } => match axis {
                Some(a) => write!(f, "⌽{a}{}", ExprGlyphs(operand)),
                None => write!(f, "⌽{}", ExprGlyphs(operand)),
            },
            Expr::Shift { dir, axis, operand } => match axis {
                Some(a) => write!(f, "{dir}{a}{}", ExprGlyphs(operand)),
                None => write!(f, "{dir}{}", ExprGlyphs(operand)),
            },
            Expr::Builtin { op, operand } => write!(f, "{op} {}", ExprGlyphs(operand)),
            Expr::Reduce { op, axis, operand } => match axis {
                Some(a) => write!(f, "{op}{a}{}", ExprGlyphs(operand)),
                None => write!(f, "{op}{}", ExprGlyphs(operand)),
            },
            Expr::Scan { op, axis, operand } => {
                let running = format!("{op}").replace('◇', "◈");
                match axis {
                    Some(a) => write!(f, "{running}{a}{}", ExprGlyphs(operand)),
                    None => write!(f, "{running}{}", ExprGlyphs(operand)),
                }
            }
            Expr::BinaryOp { op, lhs, rhs } => {
                write!(f, "({} {} {})", ExprGlyphs(lhs), op, ExprGlyphs(rhs))
            }
        }
    }
}
