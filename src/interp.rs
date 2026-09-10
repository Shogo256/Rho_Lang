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
use crate::numeric::{integer_power, Compare, Numeric};
use std::collections::BTreeMap;

/// One space's contents, with the shape that gives its cells meaning.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid<S = f64> {
    pub shape: Vec<usize>,
    pub cells: Vec<S>,
}

impl<S: Numeric> Grid<S> {
    pub fn zeros(shape: Vec<usize>) -> Grid<S> {
        let len = shape.iter().product::<usize>().max(1);
        Grid {
            shape,
            cells: vec![S::constant(0.0); len],
        }
    }

    pub fn from(shape: Vec<usize>, cells: Vec<S>) -> Grid<S> {
        let len = shape.iter().product::<usize>().max(1);
        let mut cells = cells;
        cells.resize(len, S::constant(0.0));
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
pub type Env<S = f64> = BTreeMap<String, Grid<S>>;

/// The cap on a `⇒` loop when the caller names none: enough for a small
/// relaxation to settle, small enough that a mistake cannot hang a test.
pub const DEFAULT_MAX_SWEEPS: usize = 1000;

/// What a run is parameterised by: the threshold symbol, matching `--tau`,
/// and the most sweeps a `⇒` may take, matching `--max-iter`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub tau: f64,
    pub max_sweeps: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            tau: 0.0,
            max_sweeps: DEFAULT_MAX_SWEEPS,
        }
    }
}

/// Run a program over the given inputs, returning every space it produced.
///
/// `tau` binds the threshold symbol, matching `--tau`; a `⇒` is capped at
/// [`DEFAULT_MAX_SWEEPS`]. Use [`interpret_with`] to set the cap.
pub fn interpret<S: Numeric>(block: &ToposBlock, inputs: &Env<S>, tau: f64) -> Result<Env<S>> {
    let options = Options {
        tau,
        ..Options::default()
    };
    interpret_with(block, inputs, &options)
}

/// As [`interpret`], with every parameter of a run spelled out.
pub fn interpret_with<S: Numeric>(
    block: &ToposBlock,
    inputs: &Env<S>,
    options: &Options,
) -> Result<Env<S>> {
    let tau = options.tau;
    let mut env: Env<S> = inputs.clone();

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
        let line = block.line_of(index);
        match stmt {
            Statement::Flow { src, target } => {
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
            Statement::Iterate { prelude, src, target } => {
                iterate(prelude, src, target, &mut env, options, line)?;
            }
            _ => {}
        }
    }

    Ok(env)
}

/// `expr ⇒ target`: sweep until no cell moves by more than 𝜏, or the cap is
/// reached. Every sweep reads the whole previous grid, so this is a Jacobi
/// iteration; the cap alone guarantees it ends.
fn iterate<S: Numeric>(
    prelude: &[Statement],
    src: &Expr,
    target: &str,
    env: &mut Env<S>,
    options: &Options,
    line: usize,
) -> Result<()> {
    let tau = S::constant(options.tau);
    let mut sweeps = 0usize;
    loop {
        // The prelude first, every round: the flows a call expanded into.
        for flow in prelude {
            if let Statement::Flow {
                src,
                target: FlowTarget::Var(t),
            } = flow
            {
                let value = eval(src, env, options.tau, line)?;
                env.insert(t.clone(), value);
            }
        }
        let previous = env
            .get(target)
            .ok_or_else(|| HarmonyDisruption::SpaceErr {
                space_name: target.to_string(),
                line,
            })?
            .clone();
        let next = eval(src, env, options.tau, line)?;
        if next.shape != previous.shape {
            return Err(HarmonyDisruption::DimensionErr {
                space_a: "⇒ body".to_string(),
                shape_a: next.shape,
                space_b: target.to_string(),
                shape_b: previous.shape,
                line,
            });
        }

        // The largest move of any cell. A NaN move never counts as larger, so
        // a NaN grid never settles and runs to the cap.
        let mut largest = S::constant(0.0);
        for (after, before) in next.cells.iter().zip(&previous.cells) {
            let moved = after.sub(before).unary(BuiltinOp::Abs);
            largest = S::select(&moved.compare(&largest, Compare::Gt), &moved, &largest);
        }
        env.insert(target.to_string(), next);
        sweeps += 1;

        // The cap decides first, then the tolerance.
        if sweeps >= options.max_sweeps {
            return Ok(());
        }
        if S::truth(&largest.compare(&tau, Compare::Lte)) {
            return Ok(());
        }
    }
}

fn err(line: usize, detail: impl Into<String>) -> HarmonyDisruption {
    HarmonyDisruption::LoweringErr {
        detail: detail.into(),
        line,
    }
}

/// The shape an expression produces, which is also the shape it is evaluated at.
fn shape_of<S: Numeric>(expr: &Expr, env: &Env<S>, line: usize) -> Result<Option<Vec<usize>>> {
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
        Expr::Scan { operand, .. }
        | Expr::Builtin { operand, .. }
        | Expr::Index { operand, .. }
        | Expr::Rotate { operand, .. }
        | Expr::Reverse { operand, .. } => shape_of(operand, env, line)?,
        Expr::Reshape { shape, operand } => shape_of(operand, env, line)?.map(|_| shape.clone()),
        Expr::Take { count, axis, operand } | Expr::Drop { count, axis, operand } => {
            match shape_of(operand, env, line)? {
                Some(inner) => Some(
                    taken_shape(&inner, *axis, *count, matches!(expr, Expr::Drop { .. }))
                        .ok_or_else(|| err(line, format!("`{count}` of {inner:?} leaves nothing")))?,
                ),
                None => None,
            }
        }
        Expr::Call { name, .. } => {
            return Err(err(line, format!("the call to `{name}` was not expanded")))
        }
        Expr::Transpose { axes, operand } => match shape_of(operand, env, line)? {
            Some(inner) => Some(transposed_shape(&inner, axes.as_deref()).ok_or_else(|| {
                err(line, format!("⍉ needs a permutation of the axes of {inner:?}"))
            })?),
            None => None,
        },
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
fn eval<S: Numeric>(expr: &Expr, env: &Env<S>, tau: f64, line: usize) -> Result<Grid<S>> {
    let shape = shape_of(expr, env, line)?
        .ok_or_else(|| err(line, "an expression with no space in it has no shape"))?;
    let cells = (0..shape.iter().product::<usize>().max(1))
        .map(|i| eval_cell(expr, env, tau, line, &shape, i, &[]))
        .collect::<Result<Vec<S>>>()?;
    Ok(Grid { shape, cells })
}

/// Evaluate one cell of an expression.
///
/// `at_shape` is the shape being walked and `index` a cell of it; `lifts`
/// records the unit axes the enclosing `□`s inserted, which is what lets a
/// shorter operand stretch.
fn eval_cell<S: Numeric>(
    expr: &Expr,
    env: &Env<S>,
    tau: f64,
    line: usize,
    at_shape: &[usize],
    index: usize,
    lifts: &[usize],
) -> Result<S> {
    match expr {
        Expr::Number(v) => Ok(S::constant(*v)),
        Expr::Var(name) if is_tau(name) => Ok(S::constant(tau)),

        Expr::Var(name) => {
            let grid = env.get(name).ok_or_else(|| HarmonyDisruption::SpaceErr {
                space_name: name.clone(),
                line,
            })?;
            let view = lifted(&grid.shape, lifts);
            let mapped = map_index(&view, at_shape, index);
            Ok(grid
                .cells
                .get(mapped)
                .cloned()
                .unwrap_or_else(|| S::constant(0.0)))
        }

        Expr::AuditTrace(inner) => eval_cell(inner, env, tau, line, at_shape, index, lifts),

        // `ind` of a comparison is that comparison's truth. Evaluating the
        // operand first would hand back the mask, which cannot tell a blocked
        // cell from one that passed a zero through.
        Expr::Builtin {
            op: BuiltinOp::Indicator,
            operand,
        } => {
            let one = S::constant(1.0);
            let zero = S::constant(0.0);
            if let Expr::BinaryOp { op, lhs, rhs } = &**operand {
                if let Some(how) = compare_of(op) {
                    let l = eval_cell(lhs, env, tau, line, at_shape, index, lifts)?;
                    let r = eval_cell(rhs, env, tau, line, at_shape, index, lifts)?;
                    return Ok(S::select(&l.compare(&r, how), &one, &zero));
                }
            }
            let value = eval_cell(operand, env, tau, line, at_shape, index, lifts)?;
            Ok(S::select(&value.compare(&zero, Compare::Eq), &zero, &one))
        }

        Expr::Builtin { op, operand } => {
            Ok(eval_cell(operand, env, tau, line, at_shape, index, lifts)?.unary(*op))
        }

        Expr::Lift { axis, operand } => {
            let mut nested = lifts.to_vec();
            nested.push(*axis);
            eval_cell(operand, env, tau, line, at_shape, index, &nested)
        }

        // A rotation reads the cell `by` further along the axis, wrapping; a
        // reversal reads the cell at the other end. Both are the operand at
        // another cell of its own shape.
        Expr::Rotate { axis, operand, .. } | Expr::Reverse { axis, operand } => {
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "⌽ needs an operand with a shape"))?;
            let view = lifted(&inner_shape, lifts);
            let mapped = map_index(&view, at_shape, index);
            let a = axis.unwrap_or_else(|| default_axis(&inner_shape));
            let (stride, extent) = axis_geometry(&inner_shape, Some(a))
                .ok_or_else(|| err(line, format!("axis {a} is past the end of {inner_shape:?}")))?;
            let position = (mapped / stride) % extent;
            let target = match expr {
                Expr::Rotate { by, .. } => {
                    (position as i64 + by).rem_euclid(extent as i64) as usize
                }
                _ => extent - 1 - position,
            };
            let cell = mapped - position * stride + target * stride;
            eval_at(operand, env, tau, line, &inner_shape, cell)
        }

        Expr::Call { name, .. } => Err(err(line, format!("the call to `{name}` was not expanded"))),

        // The cell `count` in from the start or the end of the axis, or zero
        // where a take runs past the source.
        Expr::Take { count, axis, operand } | Expr::Drop { count, axis, operand } => {
            let dropping = matches!(expr, Expr::Drop { .. });
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "↑ and ↓ need an operand with a shape"))?;
            let a = axis.unwrap_or_else(|| default_axis(&inner_shape));
            let result_shape = taken_shape(&inner_shape, Some(a), *count, dropping)
                .ok_or_else(|| err(line, format!("`{count}` of {inner_shape:?} leaves nothing")))?;
            let view = lifted(&result_shape, lifts);
            let mapped = map_index(&view, at_shape, index);
            match taken_source(&inner_shape, a, *count, dropping, mapped) {
                Some(source) => eval_at(operand, env, tau, line, &inner_shape, source),
                None => Ok(S::constant(0.0)),
            }
        }

        // The cell whose coordinates are this cell's, permuted.
        Expr::Transpose { axes, operand } => {
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "⍉ needs an operand with a shape"))?;
            let perm = transpose_axes(inner_shape.len(), axes.as_deref())
                .ok_or_else(|| err(line, format!("⍉ needs a permutation of the axes of {inner_shape:?}")))?;
            let result_shape = transposed_shape(&inner_shape, Some(&perm))
                .ok_or_else(|| err(line, "⍉ could not shape its result"))?;
            let view = lifted(&result_shape, lifts);
            let mapped = map_index(&view, at_shape, index);
            let source = transposed_source(&inner_shape, &perm, mapped);
            eval_at(operand, env, tau, line, &inner_shape, source)
        }

        // The operand's cells in row-major order, read into the new shape;
        // round again when the new shape has more cells.
        Expr::Reshape { shape, operand } => {
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "⍴ needs an operand with a shape"))?;
            let count = inner_shape.iter().product::<usize>().max(1);
            let view = lifted(shape, lifts);
            let mapped = map_index(&view, at_shape, index);
            eval_at(operand, env, tau, line, &inner_shape, mapped % count)
        }

        // The coordinate of this cell along one axis of the operand's shape,
        // from zero. The operand is never evaluated, only measured.
        Expr::Index { axis, operand } => {
            let inner_shape = shape_of(operand, env, line)?
                .ok_or_else(|| err(line, "⍳ needs an operand with a shape"))?;
            let view = lifted(&inner_shape, lifts);
            let mapped = map_index(&view, at_shape, index);
            let a = axis.unwrap_or_else(|| default_axis(&inner_shape));
            let (stride, extent) = axis_geometry(&inner_shape, Some(a))
                .ok_or_else(|| err(line, format!("axis {a} is past the end of {inner_shape:?}")))?;
            Ok(S::constant(((mapped / stride) % extent) as f64))
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
                return Ok(S::constant(0.0));
            }

            let position = (mapped / stride) % extent;
            let neighbour = match dir {
                ShiftDir::Positive => {
                    if position == 0 {
                        return Ok(S::constant(0.0));
                    }
                    mapped - stride
                }
                ShiftDir::Negative => {
                    if position + 1 == extent {
                        return Ok(S::constant(0.0));
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
            let zero = S::constant(0.0);
            // A comparison masks: the left value passes where it holds.
            let mask = |how: Compare| S::select(&l.compare(&r, how), &l, &zero);
            Ok(match op {
                BinaryOpKind::Add => l.add(&r),
                BinaryOpKind::Sub => l.sub(&r),
                BinaryOpKind::Mul => l.mul(&r),
                BinaryOpKind::Div => l.div(&r),
                // Repeated multiplication when the exponent is spelled as a
                // whole number, a library power otherwise — the compiler's
                // rule, taken from the same place.
                BinaryOpKind::Pow => match whole_exponent(rhs) {
                    Some(n) => integer_power(&l, &S::constant(n as f64)),
                    None => l.power(&r),
                },
                BinaryOpKind::Gt => mask(Compare::Gt),
                BinaryOpKind::Lt => mask(Compare::Lt),
                BinaryOpKind::Gte => mask(Compare::Gte),
                BinaryOpKind::Lte => mask(Compare::Lte),
                BinaryOpKind::Eq => mask(Compare::Eq),
                // The greater and the lesser, as IEEE 754-2019 defines them.
                BinaryOpKind::Max => extreme(&l, &r, true),
                BinaryOpKind::Min => extreme(&l, &r, false),
                // APL's residue: B - A × ⌊B ÷ A⌋, so the result takes the
                // sign of A, and 0 | B is B.
                BinaryOpKind::Residue => {
                    let quotient = r.div(&l).floor();
                    let remainder = r.sub(&l.mul(&quotient));
                    S::select(&l.compare(&zero, Compare::Eq), &r, &remainder)
                }
            })
        }
    }
}



/// IEEE 754-2019 `maximum` (`greater`) or `minimum`: a NaN on either side is
/// the answer, -0 orders below +0, and otherwise the greater or the lesser.
/// These are the rules the kernel's intrinsics follow, spelled out with the
/// operations the trait has: a value that is not equal to itself is a NaN,
/// and the sign of a zero shows in its reciprocal.
fn extreme<S: Numeric>(l: &S, r: &S, greater: bool) -> S {
    let one = S::constant(1.0);
    let by_value = if greater {
        S::select(&l.compare(r, Compare::Gt), l, r)
    } else {
        S::select(&l.compare(r, Compare::Lt), l, r)
    };
    // Equal values that differ in the sign of zero: order by the reciprocal,
    // which is -inf for -0 and +inf for +0.
    let (inv_l, inv_r) = (one.div(l), one.div(r));
    let by_sign = if greater {
        S::select(&inv_l.compare(&inv_r, Compare::Gt), l, r)
    } else {
        S::select(&inv_l.compare(&inv_r, Compare::Lt), l, r)
    };
    let ordered = S::select(&l.compare(r, Compare::Eq), &by_sign, &by_value);
    let r_or_ordered = S::select(&r.compare(r, Compare::Ne), r, &ordered);
    S::select(&l.compare(l, Compare::Ne), l, &r_or_ordered)
}

/// Evaluate an expression at one cell of its own shape.
fn eval_at<S: Numeric>(
    expr: &Expr,
    env: &Env<S>,
    tau: f64,
    line: usize,
    shape: &[usize],
    index: usize,
) -> Result<S> {
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
fn fold_line<S: Numeric>(
    op: &FoldOp,
    operand: &Expr,
    env: &Env<S>,
    tau: f64,
    line: usize,
    shape: &[usize],
    axis: usize,
    line_start: usize,
    upto: Option<usize>,
) -> Result<S> {
    let (stride, extent) = axis_geometry(shape, Some(axis))
        .ok_or_else(|| err(line, format!("axis {axis} is past the end of {shape:?}")))?;
    let last = upto.map(|p| p + 1).unwrap_or(extent);

    let mut acc = S::constant(op.identity());
    for step in 0..last.min(extent) {
        let value = eval_at(operand, env, tau, line, shape, line_start + step * stride)?;
        acc = match op {
            FoldOp::Sum => acc.add(&value),
            FoldOp::Product => acc.mul(&value),
            FoldOp::Max => S::select(&value.compare(&acc, Compare::Gt), &value, &acc),
            FoldOp::Min => S::select(&value.compare(&acc, Compare::Lt), &value, &acc),
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

/// The comparison a binary operator stands for, if it is one.
fn compare_of(op: &BinaryOpKind) -> Option<Compare> {
    Some(match op {
        BinaryOpKind::Gt => Compare::Gt,
        BinaryOpKind::Lt => Compare::Lt,
        BinaryOpKind::Gte => Compare::Gte,
        BinaryOpKind::Lte => Compare::Lte,
        BinaryOpKind::Eq => Compare::Eq,
        _ => return None,
    })
}

fn is_tau(name: &str) -> bool {
    name == "𝜏" || name == "τ"
}
