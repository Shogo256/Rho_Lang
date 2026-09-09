//! A reference interpreter for ρ, written from the semantics rather than from
//! the code generator.
//!
//! Its job is to be an independent second opinion. The compiler is checked
//! against it by differential testing: if the two disagree on any program, at
//! least one of them is wrong, and the disagreement says where to look. That is
//! only worth anything while this file stays independent — it deliberately
//! shares nothing with `codegen` beyond the shape algebra in `ast`, which *is*
//! the specification.
//!
//! It is written for clarity, not speed. Nothing here should be clever.

use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use std::collections::BTreeMap;

/// One space's contents, with the shape that gives its cells meaning.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub shape: Vec<usize>,
    pub cells: Vec<f64>,
}

impl Grid {
    pub fn zeros(shape: Vec<usize>) -> Grid {
        let len = shape.iter().product::<usize>().max(1);
        Grid {
            shape,
            cells: vec![0.0; len],
        }
    }

    pub fn from(shape: Vec<usize>, cells: Vec<f64>) -> Grid {
        let len = shape.iter().product::<usize>().max(1);
        let mut cells = cells;
        cells.resize(len, 0.0);
        Grid { shape, cells }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

}

/// Every space the program has produced so far.
pub type Env = BTreeMap<String, Grid>;

/// Run a program over the given inputs, returning every space it produced.
///
/// `tau` binds the threshold symbol, matching `--tau`.
pub fn interpret(block: &ToposBlock, inputs: &Env, tau: f64) -> Result<Env> {
    let mut env: Env = inputs.clone();

    // Declared spaces that the caller did not supply start at zero.
    for stmt in &block.statements {
        let decl = match stmt {
            Statement::SpaceDef(d) => Some((d.name.clone(), d.dimensions.clone())),
            Statement::ExtBind(b) => Some((b.space.name.clone(), b.space.dimensions.clone())),
            _ => None,
        };
        if let Some((name, shape)) = decl {
            env.entry(name).or_insert_with(|| Grid::zeros(shape));
        }
    }

    for (index, stmt) in block.statements.iter().enumerate() {
        let Statement::Flow { src, target } = stmt else {
            continue;
        };
        let line = block.line_of(index);
        let value = eval(src, &env, tau, line)?;
        let name = match target {
            FlowTarget::Var(n) => n.clone(),
            FlowTarget::Equilibrium => "OUTPUT".to_string(),
        };
        env.insert(name, value);

        // `=` ends the pipeline; later statements are not evaluated.
        if matches!(target, FlowTarget::Equilibrium) {
            break;
        }
    }

    Ok(env)
}

fn err(line: usize, detail: impl Into<String>) -> HarmonyDisruption {
    HarmonyDisruption::LoweringErr {
        detail: detail.into(),
        line,
    }
}

/// The shape an expression produces, which is also the shape it is evaluated at.
fn shape_of(expr: &Expr, env: &Env, line: usize) -> Result<Option<Vec<usize>>> {
    Ok(match expr {
        Expr::Number(_) => None,
        Expr::Var(name) if is_tau(name) => None,
        Expr::Var(name) => Some(
            env.get(name)
                .ok_or_else(|| HarmonyDisruption::SpaceErr {
                    space_name: name.clone(),
                    line,
                })?
                .shape
                .clone(),
        ),
        Expr::AuditTrace(inner) | Expr::Shift { operand: inner, .. } => {
            shape_of(inner, env, line)?
        }
        Expr::Scan { operand, .. } => shape_of(operand, env, line)?,
        Expr::Lift { axis, operand } => match shape_of(operand, env, line)? {
            Some(inner) => Some(shape_with_unit_axis(&inner, *axis).ok_or_else(|| {
                err(line, format!("axis {axis} is past the end of {inner:?}"))
            })?),
            None => None,
        },
        Expr::Reduce { axis, operand, .. } => match shape_of(operand, env, line)? {
            Some(inner) => {
                let a = axis.unwrap_or_else(|| default_axis(&inner));
                if a >= inner.len() {
                    return Err(err(line, format!("axis {a} is past the end of {inner:?}")));
                }
                Some(shape_without_axis(&inner, a))
            }
            None => None,
        },
        Expr::BinaryOp { lhs, rhs, .. } => {
            match (shape_of(lhs, env, line)?, shape_of(rhs, env, line)?) {
                (Some(l), Some(r)) => Some(broadcast_shapes(&l, &r).ok_or_else(|| {
                    HarmonyDisruption::DimensionErr {
                        space_a: "left".to_string(),
                        shape_a: l,
                        space_b: "right".to_string(),
                        shape_b: r,
                        line,
                    }
                })?),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
    })
}

/// Evaluate an expression to a whole grid.
fn eval(expr: &Expr, env: &Env, tau: f64, line: usize) -> Result<Grid> {
    let shape = shape_of(expr, env, line)?
        .ok_or_else(|| err(line, "an expression with no space in it has no shape"))?;
    let cells = (0..shape.iter().product::<usize>().max(1))
        .map(|i| eval_cell(expr, env, tau, line, &shape, i, &[]))
        .collect::<Result<Vec<f64>>>()?;
    Ok(Grid { shape, cells })
}

/// Evaluate one cell of an expression.
///
/// `at_shape` is the shape being walked and `index` a cell of it; `lifts`
/// records the unit axes the enclosing `□`s inserted, which is what lets a
/// shorter operand stretch.
fn eval_cell(
    expr: &Expr,
    env: &Env,
    tau: f64,
    line: usize,
    at_shape: &[usize],
    index: usize,
    lifts: &[usize],
) -> Result<f64> {
    match expr {
        Expr::Number(v) => Ok(*v),
        Expr::Var(name) if is_tau(name) => Ok(tau),

        Expr::Var(name) => {
            let grid = env.get(name).ok_or_else(|| HarmonyDisruption::SpaceErr {
                space_name: name.clone(),
                line,
            })?;
            let view = lifted(&grid.shape, lifts);
            let mapped = map_index(&view, at_shape, index);
            Ok(grid.cells.get(mapped).copied().unwrap_or(0.0))
        }

        Expr::AuditTrace(inner) => eval_cell(inner, env, tau, line, at_shape, index, lifts),

        Expr::Lift { axis, operand } => {
            let mut nested = lifts.to_vec();
            nested.push(*axis);
            eval_cell(operand, env, tau, line, at_shape, index, &nested)
        }

        // A shift reads the neighbour along one axis, or 0 at its boundary.
        Expr::Shift { dir, axis, operand } => {
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "a shift needs an operand with a shape"))?;
            let view = lifted(&inner_shape, lifts);
            let mapped = map_index(&view, at_shape, index);

            let a = axis.unwrap_or_else(|| default_axis(&inner_shape));
            let (stride, extent) = axis_geometry(&inner_shape, Some(a))
                .ok_or_else(|| err(line, format!("axis {a} is past the end of {inner_shape:?}")))?;
            if extent <= 1 {
                return Ok(0.0);
            }

            let position = (mapped / stride) % extent;
            let neighbour = match dir {
                ShiftDir::Positive => {
                    if position == 0 {
                        return Ok(0.0);
                    }
                    mapped - stride
                }
                ShiftDir::Negative => {
                    if position + 1 == extent {
                        return Ok(0.0);
                    }
                    mapped + stride
                }
            };
            eval_at(operand, env, tau, line, &inner_shape, neighbour)
        }

        // A fold answers "what is the total" for the line through this cell.
        Expr::Reduce { op, axis, operand } => {
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "a fold needs an operand with a shape"))?;
            let a = axis.unwrap_or_else(|| default_axis(&inner_shape));
            let result_shape = shape_without_axis(&inner_shape, a);
            let view = lifted(&result_shape, lifts);
            let surviving = map_index(&view, at_shape, index);
            // `surviving` indexes the shape the fold leaves behind. The line it
            // summarises starts elsewhere in the full shape.
            let start = line_start_of(&inner_shape, a, surviving);
            fold_line(op, operand, env, tau, line, &inner_shape, a, start, None)
        }

        // A scan answers "what is the total so far", so it stops at this cell.
        Expr::Scan { op, axis, operand } => {
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "a scan needs an operand with a shape"))?;
            let a = axis.unwrap_or_else(|| default_axis(&inner_shape));
            let view = lifted(&inner_shape, lifts);
            let mapped = map_index(&view, at_shape, index);

            let (stride, extent) = axis_geometry(&inner_shape, Some(a))
                .ok_or_else(|| err(line, format!("axis {a} is past the end of {inner_shape:?}")))?;
            let position = (mapped / stride) % extent;
            fold_line(
                op,
                operand,
                env,
                tau,
                line,
                &inner_shape,
                a,
                flat_of_line(&inner_shape, a, mapped),
                Some(position),
            )
        }

        Expr::BinaryOp { op, lhs, rhs } => {
            let l = eval_cell(lhs, env, tau, line, at_shape, index, lifts)?;
            let r = eval_cell(rhs, env, tau, line, at_shape, index, lifts)?;
            Ok(match op {
                BinaryOpKind::Add => l + r,
                BinaryOpKind::Sub => l - r,
                BinaryOpKind::Mul => l * r,
                BinaryOpKind::Div => l / r,
                // A whole-number exponent is repeated multiplication, matching
                // what the compiler emits. powf goes through a maths library
                // whose rounding of a square differs by an ulp.
                BinaryOpKind::Pow => integer_power(l, r),
                // A comparison masks: the left value passes where it holds.
                BinaryOpKind::Gt => pass(l > r, l),
                BinaryOpKind::Lt => pass(l < r, l),
                BinaryOpKind::Gte => pass(l >= r, l),
                BinaryOpKind::Lte => pass(l <= r, l),
                BinaryOpKind::Eq => pass(l == r, l),
            })
        }
    }
}

/// `x^n` as repeated multiplication when `n` is a small whole number.
fn integer_power(base: f64, exponent: f64) -> f64 {
    if exponent != exponent.trunc() || exponent.abs() > 64.0 {
        return base.powf(exponent);
    }
    let n = exponent as i32;
    let mut acc = 1.0f64;
    for _ in 0..n.abs() {
        acc *= base;
    }
    if n < 0 {
        1.0 / acc
    } else {
        acc
    }
}

fn pass(holds: bool, value: f64) -> f64 {
    if holds {
        value
    } else {
        0.0
    }
}

/// Evaluate an expression at one cell of its own shape.
fn eval_at(
    expr: &Expr,
    env: &Env,
    tau: f64,
    line: usize,
    shape: &[usize],
    index: usize,
) -> Result<f64> {
    eval_cell(expr, env, tau, line, shape, index, &[])
}

/// Flat index, in the full shape, of the line a surviving cell summarises.
fn line_start_of(shape: &[usize], axis: usize, surviving: usize) -> usize {
    let Some((stride, extent)) = axis_geometry(shape, Some(axis)) else {
        return surviving;
    };
    let outer = surviving / stride.max(1);
    let within = surviving % stride.max(1);
    outer * extent * stride + within
}

/// Index of the start of the line through `index` along `axis`.
fn flat_of_line(shape: &[usize], axis: usize, index: usize) -> usize {
    let Some((stride, extent)) = axis_geometry(shape, Some(axis)) else {
        return index;
    };
    let position = (index / stride) % extent;
    index - position * stride
}

/// Fold the line starting at `line_start`, stopping after `upto` steps when a
/// scan asked for a running answer.
#[allow(clippy::too_many_arguments)]
fn fold_line(
    op: &FoldOp,
    operand: &Expr,
    env: &Env,
    tau: f64,
    line: usize,
    shape: &[usize],
    axis: usize,
    line_start: usize,
    upto: Option<usize>,
) -> Result<f64> {
    let (stride, extent) = axis_geometry(shape, Some(axis))
        .ok_or_else(|| err(line, format!("axis {axis} is past the end of {shape:?}")))?;
    let last = upto.map(|p| p + 1).unwrap_or(extent);

    let mut acc = op.identity();
    for step in 0..last.min(extent) {
        let value = eval_at(operand, env, tau, line, shape, line_start + step * stride)?;
        acc = match op {
            FoldOp::Sum => acc + value,
            FoldOp::Product => acc * value,
            FoldOp::Max => {
                if value > acc {
                    value
                } else {
                    acc
                }
            }
            FoldOp::Min => {
                if value < acc {
                    value
                } else {
                    acc
                }
            }
        };
    }
    Ok(acc)
}

/// A shape with the enclosing lifts' unit axes inserted, outermost last.
fn lifted(shape: &[usize], lifts: &[usize]) -> Vec<usize> {
    let mut view = shape.to_vec();
    for &axis in lifts.iter().rev() {
        match shape_with_unit_axis(&view, axis) {
            Some(next) => view = next,
            None => return view,
        }
    }
    view
}

/// Translate a cell of `at_shape` into a cell of `source`, dropping the
/// coordinates of axes the source has stretched.
fn map_index(source: &[usize], at_shape: &[usize], index: usize) -> usize {
    if source == at_shape || source.len() != at_shape.len() {
        return index;
    }
    let src_strides = strides_of(source);
    let at_strides = strides_of(at_shape);
    (0..at_shape.len())
        .filter(|&axis| source[axis] > 1)
        .map(|axis| {
            let coord = (index / at_strides[axis]) % at_shape[axis].max(1);
            coord * src_strides[axis]
        })
        .sum()
}

fn is_tau(name: &str) -> bool {
    name == "𝜏" || name == "τ"
}
