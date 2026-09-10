use std::fmt;

/// Binary Operator Kinds
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOpKind {
    Add, // +
    Sub, // -
    Mul, // × or *
    Div, // /
    Pow, // ^
    Gt,  // >
    Lt,  // <
    Gte, // >=
    Lte, // <=
    Eq,  // ==
    /// `⌈` — the greater of the two, element-wise (APL's dyadic ⌈)
    Max,
    /// `⌊` — the lesser of the two, element-wise (APL's dyadic ⌊)
    Min,
    /// `|` — APL's residue: `A | B` is B modulo A, with the sign of A,
    /// and `0 | B` is B
    Residue,
}

impl fmt::Display for BinaryOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOpKind::Add => write!(f, "+"),
            BinaryOpKind::Sub => write!(f, "-"),
            BinaryOpKind::Mul => write!(f, "×"),
            BinaryOpKind::Div => write!(f, "/"),
            BinaryOpKind::Pow => write!(f, "^"),
            BinaryOpKind::Gt => write!(f, ">"),
            BinaryOpKind::Lt => write!(f, "<"),
            BinaryOpKind::Gte => write!(f, ">="),
            BinaryOpKind::Lte => write!(f, "<="),
            BinaryOpKind::Eq => write!(f, "=="),
            BinaryOpKind::Max => write!(f, "⌈"),
            BinaryOpKind::Min => write!(f, "⌊"),
            BinaryOpKind::Residue => write!(f, "|"),
        }
    }
}

/// Direction of a neighbourhood shift along one axis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftDir {
    /// ▷ reads the preceding cell along the axis
    Positive,
    /// ▽ reads the following cell along the axis
    Negative,
}

impl fmt::Display for ShiftDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShiftDir::Positive => write!(f, "▷"),
            ShiftDir::Negative => write!(f, "▽"),
        }
    }
}

/// How a reduction folds the cells along an axis.
///
/// Only associative operators are accepted: a fold has no defined meaning for
/// subtraction or division, whose result would depend on the traversal order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldOp {
    /// `◇+` — sum (Wasan: 垜積, the summing of piles)
    Sum,
    /// `◇×` — product
    Product,
    /// `◇>` — the greater of the two, i.e. maximum
    Max,
    /// `◇<` — the lesser of the two, i.e. minimum
    Min,
}

impl FoldOp {
    pub fn from_op(op: &BinaryOpKind) -> Option<FoldOp> {
        match op {
            BinaryOpKind::Add => Some(FoldOp::Sum),
            BinaryOpKind::Mul => Some(FoldOp::Product),
            BinaryOpKind::Gt => Some(FoldOp::Max),
            BinaryOpKind::Lt => Some(FoldOp::Min),
            _ => None,
        }
    }

    /// The value a fold starts from, which is also its result on an empty axis.
    pub fn identity(self) -> f64 {
        match self {
            FoldOp::Sum => 0.0,
            FoldOp::Product => 1.0,
            FoldOp::Max => f64::NEG_INFINITY,
            FoldOp::Min => f64::INFINITY,
        }
    }
}

impl fmt::Display for FoldOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let glyph = match self {
            FoldOp::Sum => "+",
            FoldOp::Product => "×",
            FoldOp::Max => ">",
            FoldOp::Min => "<",
        };
        write!(f, "◇{glyph}")
    }
}

/// A named function applied to every cell.
///
/// The board operations are glyphs because they describe a shape; these are
/// named because they describe a *quantity*. Wasan drew the same line: enri
/// (円理), Seki and Takebe's theory of series, was how the analytic functions
/// were reached, and it was a named technique rather than a bead movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinOp {
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Abs,
    /// 1 where the operand holds, 0 elsewhere. Applied to a comparison it is
    /// that comparison's truth; applied to anything else it asks whether the
    /// value is not zero. This is how a program counts, which masking alone
    /// cannot do: a mask that passes a value which happens to be zero is
    /// indistinguishable from one that blocked it.
    Indicator,
}

impl BuiltinOp {
    pub fn from_name(name: &str) -> Option<BuiltinOp> {
        Some(match name {
            "exp" => BuiltinOp::Exp,
            "log" => BuiltinOp::Log,
            "sqrt" => BuiltinOp::Sqrt,
            "sin" => BuiltinOp::Sin,
            "cos" => BuiltinOp::Cos,
            "abs" => BuiltinOp::Abs,
            "ind" => BuiltinOp::Indicator,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            BuiltinOp::Exp => "exp",
            BuiltinOp::Log => "log",
            BuiltinOp::Sqrt => "sqrt",
            BuiltinOp::Sin => "sin",
            BuiltinOp::Cos => "cos",
            BuiltinOp::Abs => "abs",
            BuiltinOp::Indicator => "ind",
        }
    }

    /// Every name that cannot also be a space.
    pub const ALL: [&'static str; 7] = ["exp", "log", "sqrt", "sin", "cos", "abs", "ind"];
}

impl fmt::Display for BuiltinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Topological Expression
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Var(String),
    Number(f64),
    /// ▷X / ▽X, optionally pinned to an axis: ▷0X shifts along axis 0.
    /// `None` means the last (contiguous) axis.
    Shift {
        dir: ShiftDir,
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    BinaryOp {
        op: BinaryOpKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `exp X`, `log X`, `ind (A > B)` — a named function over every cell.
    Builtin {
        op: BuiltinOp,
        operand: Box<Expr>,
    },
    /// `k ⌽ X` — APL's rotate: the cell `k` further along the axis, wrapping
    /// at the end, so `result[i] = X[(i + k) mod n]`. `k` is a whole number
    /// written as a literal, negative to rotate the other way; a digit after
    /// the glyph names the axis. Where `▷` pads with zero at the edge, `⌽`
    /// wraps, which is what a periodic boundary is.
    Rotate {
        by: i64,
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    /// `⌽X` — APL's reverse: the cell at the other end of the axis,
    /// `result[i] = X[n - 1 - i]`.
    Reverse {
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    /// `2 3 ⍴ X` — APL's reshape: X's cells in row-major order, read into a
    /// new shape written as a list of literals. With the same number of
    /// cells nothing moves: it is a reinterpretation. With fewer, X is read
    /// round again, `result[i] = X[i mod n]`, as APL does; with more, the
    /// tail is dropped.
    Reshape {
        shape: Vec<usize>,
        operand: Box<Expr>,
    },
    /// `⍉X` and `1 0 ⍉ X` — APL's transpose. Alone, the axes in reverse
    /// order; with a permutation written on the left, source axis `k`
    /// becomes result axis `axes[k]`, so `0 2 1 ⍉ X` swaps the last two axes
    /// of a rank-3 X. `result[j] = X[i]` where `i[k] = j[axes[k]]`.
    Transpose {
        axes: Option<Vec<usize>>,
        operand: Box<Expr>,
    },
    /// `k ↑ X` and `k ↓ X` — APL's take and drop along one axis. Take keeps
    /// the first `k` cells along the axis, or the last `|k|` for a negative
    /// `k`, padding with zero past the end as a shift does; drop removes the
    /// first `k`, or the last `|k|`. `k` is a whole number written as a
    /// literal, so the result's shape is known at compile time, and a digit
    /// after the glyph names the axis.
    Take {
        count: i64,
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    Drop {
        count: i64,
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    /// `⍳X` — the coordinate of each cell of X's shape along one axis,
    /// counted from zero; `⍳0X` names the axis, a bare `⍳X` takes the
    /// innermost axis with more than one cell, as `▷` and `◇` do. X is read
    /// for its shape alone and never evaluated. APL's `⍳` with `⎕IO←0`, and
    /// what position-dependent computation is written with: a window, a
    /// distance from the centre, a Vandermonde matrix, a decay.
    Index {
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    /// `□2X` — view X with a length-1 axis inserted at that position.
    ///
    /// A length-1 axis costs nothing and stores nothing; it exists so an
    /// element-wise operation can stretch it against a longer axis. `□1A × □0B`
    /// on vectors is an outer product, and lifting both operands of a matrix
    /// product lines their shared axis up so a fold can contract it.
    Lift {
        axis: usize,
        operand: Box<Expr>,
    },
    /// `◈+0X` — a running fold along one axis, keeping the shape.
    ///
    /// Where `◇+` answers "what is the total", `◈+` answers "what is the total
    /// so far" for every cell: `◈+ [1,2,3,4]` is `[1,3,6,10]`.
    Scan {
        op: FoldOp,
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    /// `◇+0X` — fold the cells along one axis, collapsing it.
    /// The result has the operand's shape with that axis removed.
    Reduce {
        op: FoldOp,
        axis: Option<usize>,
        operand: Box<Expr>,
    },
    AuditTrace(Box<Expr>), // $
    /// `smooth X`, `blend A (B + C)` — a call to a function the program
    /// defined above. Exists only between parsing and expansion: every call
    /// is replaced by the function's body, with the arguments bound, before
    /// anything else looks at the program.
    Call {
        name: String,
        args: Vec<Expr>,
    },
}

/// A function of the program's own: a named block with parameters, expanded
/// at each call. `smooth:{ X ((▷X + X + ▽X) / 3.0) }` has an expression body;
/// a body of flows ends in `→ =`, which names its result. A body sees its
/// parameters, `𝜏` and constants, nothing of the caller's. Definitions come
/// before their use, which is what rules recursion out: a function is not
/// defined while its own body is being read.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    pub body: FunctionBody,
    /// Source line of the definition's header.
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionBody {
    /// A single expression, with its source line: the result.
    Expression(Expr, usize),
    /// Flows ending in `→ =`, each with its source line.
    Flows(Vec<(Statement, usize)>),
}

/// Where an expanded statement came from, so a diagnostic can point at the
/// function's body as well as at the call that expanded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    pub function: String,
    pub body_line: usize,
}

/// Space Matrix Initialization & Boundary Declaration ([Name]:◯ □ [Dim1] [Dim2]...)
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceDecl {
    pub name: String,
    pub dimensions: Vec<usize>,
}

/// Zero-Copy External Memory Binding (&[Address]:[SpaceDecl])
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalBinding {
    pub address: u64,
    pub space: SpaceDecl,
}

/// Flow Destination Target
#[derive(Debug, Clone, PartialEq)]
pub enum FlowTarget {
    Var(String),
    Equilibrium, // =
}

/// Statement
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    SpaceDef(SpaceDecl),
    ExtBind(ExternalBinding),
    Flow {
        src: Expr,
        target: FlowTarget,
    },
    /// `expr ⇒ NAME`: sweep `expr` into NAME again and again until no cell
    /// moves by more than 𝜏, or the cap on sweeps is reached. Every sweep reads
    /// the whole of the previous one — a Jacobi step, never a Gauss–Seidel one
    /// — so `→`'s rule that a shift sees a finished grid still holds inside
    /// the loop. NAME must have been written by an earlier flow: the starting
    /// point of an iteration is part of its meaning and is spelled out.
    Iterate {
        /// Flows that run on every round before the update, in order: what a
        /// call to a function with a body of flows expands into inside the
        /// loop. Their targets are spaces of their own, refilled each round.
        prelude: Vec<Statement>,
        src: Expr,
        target: String,
    },
    Constraint(Expr), // ! (Expr)
    AuditTrace(Expr), // $ Expr
}

/// Topos (Computational Universe Block) { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct ToposBlock {
    pub statements: Vec<Statement>,
    /// Source line of each statement, parallel to `statements`, so a diagnostic
    /// can point at the line the reader actually wrote.
    pub lines: Vec<usize>,
    /// For a statement that a function call expanded into, which function
    /// and which line of its body; parallel to `statements`.
    pub origins: Vec<Option<Origin>>,
}

impl ToposBlock {
    /// Point a diagnostic at the function it arose in as well as at the call:
    /// an error on a line where a call was expanded names the body line too.
    pub fn attribute(&self, error: crate::error::HarmonyDisruption) -> crate::error::HarmonyDisruption {
        use crate::error::HarmonyDisruption;
        if matches!(error, HarmonyDisruption::InFunction { .. }) {
            return error;
        }
        let Some(line) = error.line() else {
            return error;
        };
        let origin = self
            .lines
            .iter()
            .zip(&self.origins)
            .find(|(l, o)| **l == line && o.is_some())
            .and_then(|(_, o)| o.clone());
        match origin {
            Some(origin) => HarmonyDisruption::InFunction {
                function: origin.function,
                defined: origin.body_line,
                line,
                inner: Box::new(error),
            },
            None => error,
        }
    }

    /// Source line of statement `index`, or 0 when it is not known.
    pub fn line_of(&self, index: usize) -> usize {
        self.lines.get(index).copied().unwrap_or(0)
    }

    /// Source line of the first statement matching `pred`.
    pub fn line_where(&self, pred: impl Fn(&Statement) -> bool) -> usize {
        self.statements
            .iter()
            .position(pred)
            .map(|i| self.line_of(i))
            .unwrap_or(0)
    }
}

/// The shape an expression produces, given the shapes of the spaces it reads,
/// or None for an expression of literals alone. The compiler, the reference
/// interpreter's static check and the `!` analysis all read shapes from here,
/// so an intermediate has the same shape to each of them.
pub fn expr_shape(
    expr: &Expr,
    shapes: &std::collections::BTreeMap<String, Vec<usize>>,
) -> Option<Vec<usize>> {
    match expr {
        Expr::Var(name) => shapes.get(name).cloned(),
        Expr::Shift { operand: inner, .. } | Expr::AuditTrace(inner) => expr_shape(inner, shapes),
        Expr::Lift { axis, operand } => {
            let inner = expr_shape(operand, shapes)?;
            shape_with_unit_axis(&inner, *axis)
        }
        Expr::Scan { operand, .. }
        | Expr::Builtin { operand, .. }
        | Expr::Index { operand, .. }
        | Expr::Rotate { operand, .. }
        | Expr::Reverse { operand, .. } => expr_shape(operand, shapes),
        Expr::Reshape { shape, operand } => expr_shape(operand, shapes).map(|_| shape.clone()),
        Expr::Transpose { axes, operand } => {
            let inner = expr_shape(operand, shapes)?;
            transposed_shape(&inner, axes.as_deref())
        }
        Expr::Take { count, axis, operand } => {
            let inner = expr_shape(operand, shapes)?;
            taken_shape(&inner, *axis, *count, false)
        }
        Expr::Drop { count, axis, operand } => {
            let inner = expr_shape(operand, shapes)?;
            taken_shape(&inner, *axis, *count, true)
        }
        Expr::Reduce { axis, operand, .. } => {
            let inner = expr_shape(operand, shapes)?;
            let a = axis.unwrap_or_else(|| default_axis(&inner));
            (a < inner.len()).then(|| shape_without_axis(&inner, a))
        }
        Expr::BinaryOp { lhs, rhs, .. } => {
            match (expr_shape(lhs, shapes), expr_shape(rhs, shapes)) {
                (Some(l), Some(r)) => broadcast_shapes(&l, &r),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
        // A call is expanded before shapes are asked for.
        Expr::Number(_) | Expr::Call { .. } => None,
    }
}

/// The permutation a transpose applies: the one written, or the reversal of
/// the axes when none was. None when what was written is not a permutation
/// of the operand's axes.
pub fn transpose_axes(rank: usize, axes: Option<&[usize]>) -> Option<Vec<usize>> {
    let perm: Vec<usize> = match axes {
        Some(written) => written.to_vec(),
        None => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        return None;
    }
    let mut seen = vec![false; rank];
    for &axis in &perm {
        if axis >= rank || seen[axis] {
            return None;
        }
        seen[axis] = true;
    }
    Some(perm)
}

/// The shape a transpose produces: source axis `k` becomes result axis
/// `perm[k]`.
pub fn transposed_shape(shape: &[usize], axes: Option<&[usize]>) -> Option<Vec<usize>> {
    let perm = transpose_axes(shape.len(), axes)?;
    let mut out = vec![0usize; shape.len()];
    for (k, &extent) in shape.iter().enumerate() {
        out[perm[k]] = extent;
    }
    Some(out)
}

/// The flat source index a transposed cell reads: `result[j] = X[i]` with
/// `i[k] = j[perm[k]]`.
pub fn transposed_source(shape: &[usize], perm: &[usize], flat_result: usize) -> usize {
    let result_shape = {
        let mut out = vec![0usize; shape.len()];
        for (k, &extent) in shape.iter().enumerate() {
            out[perm[k]] = extent;
        }
        out
    };
    let result_strides = strides_of(&result_shape);
    let source_strides = strides_of(shape);
    (0..shape.len())
        .map(|k| {
            let a = perm[k];
            let coord = (flat_result / result_strides[a]) % result_shape[a].max(1);
            coord * source_strides[k]
        })
        .sum()
}

/// The shape a take (or, `dropping`, a drop) of `count` along `axis` leaves.
/// None when the axis is out of range or nothing would be left.
pub fn taken_shape(shape: &[usize], axis: Option<usize>, count: i64, dropping: bool) -> Option<Vec<usize>> {
    let a = axis.unwrap_or_else(|| default_axis(shape));
    if a >= shape.len() {
        return None;
    }
    let extent = shape[a] as i64;
    let kept = if dropping {
        extent - count.abs()
    } else {
        count.abs()
    };
    if kept <= 0 {
        return None;
    }
    let mut out = shape.to_vec();
    out[a] = kept as usize;
    Some(out)
}

/// Where a take or drop reads along its axis: the offset added to a result
/// position to reach the source position. A drop of the first k starts k in;
/// a take of the last |k| starts |k| before the end, which is negative — and
/// so padded — when the take is longer than the axis.
pub fn taken_offset(extent: usize, count: i64, dropping: bool) -> i64 {
    match (dropping, count >= 0) {
        (true, true) => count,
        (true, false) => 0,
        (false, true) => 0,
        (false, false) => extent as i64 - count.abs(),
    }
}

/// The flat source cell a taken or dropped result cell reads, or None where
/// the take runs past the source and reads zero.
pub fn taken_source(
    shape: &[usize],
    axis: usize,
    count: i64,
    dropping: bool,
    flat_result: usize,
) -> Option<usize> {
    let result_shape = taken_shape(shape, Some(axis), count, dropping)?;
    let result_strides = strides_of(&result_shape);
    let source_strides = strides_of(shape);
    let offset = taken_offset(shape[axis], count, dropping);
    let mut flat = 0usize;
    for b in 0..shape.len() {
        let coord = (flat_result / result_strides[b]) % result_shape[b].max(1);
        if b == axis {
            let source = coord as i64 + offset;
            if source < 0 || source >= shape[axis] as i64 {
                return None;
            }
            flat += source as usize * source_strides[b];
        } else {
            flat += coord * source_strides[b];
        }
    }
    Some(flat)
}

/// The exponent of `x ^ n` when `n` is written as a small whole number, in
/// which case the power is repeated multiplication rather than a call to the
/// maths library. Decided by spelling, not by value: `x ^ Y` with a space Y
/// whose cells happen to be whole is still a library power. The compiler and
/// the interpreter both take the answer from here, so they cannot disagree.
pub fn whole_exponent(exponent: &Expr) -> Option<i32> {
    let Expr::Number(v) = exponent else {
        return None;
    };
    if *v != v.trunc() || v.abs() > 64.0 {
        return None;
    }
    Some(*v as i32)
}

/// The shape with a length-1 axis inserted at `axis`.
pub fn shape_with_unit_axis(shape: &[usize], axis: usize) -> Option<Vec<usize>> {
    if axis > shape.len() {
        return None;
    }
    let mut out = shape.to_vec();
    out.insert(axis, 1);
    Some(out)
}

/// The shape an element-wise operation produces from two operands.
///
/// Ranks must match: a length-1 axis stretches against a longer one, and
/// anything else is a mismatch. Requiring equal rank keeps the rule something a
/// reader can check by eye, and `□` is how a shape gains the axes it needs.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    if a.len() != b.len() {
        return None;
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            if x == y {
                Some(x)
            } else if x == 1 {
                Some(y)
            } else if y == 1 {
                Some(x)
            } else {
                None
            }
        })
        .collect()
}

/// Row-major strides for a shape.
pub fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1].max(1);
    }
    strides
}

/// The shape left after folding away `axis`.
///
/// A rank-1 space collapses to a single cell rather than to rank 0, so every
/// value in the language still lives in a grid that can be addressed.
pub fn shape_without_axis(shape: &[usize], axis: usize) -> Vec<usize> {
    if shape.len() <= 1 {
        return vec![1];
    }
    let mut out = shape.to_vec();
    out.remove(axis);
    out
}

/// The axis a bare `▷` / `◇` operates on: the innermost with more than one cell.
pub fn default_axis(shape: &[usize]) -> usize {
    shape.iter().rposition(|&d| d > 1).unwrap_or(shape.len().saturating_sub(1))
}

/// Geometry of one axis of a row-major shape: `(stride, extent)`.
///
/// `axis: None` selects the innermost axis that has more than one cell, so a
/// bare `▷` on `◯ □ 8 1` walks the 8 cells instead of the degenerate trailing
/// axis. Returns `None` when the shape is empty or the axis is out of range.
pub fn axis_geometry(shape: &[usize], axis: Option<usize>) -> Option<(usize, usize)> {
    if shape.is_empty() {
        return None;
    }
    let a = match axis {
        Some(a) => a,
        None => shape.iter().rposition(|&d| d > 1).unwrap_or(shape.len() - 1),
    };
    if a >= shape.len() {
        return None;
    }
    Some((shape[a + 1..].iter().product::<usize>().max(1), shape[a]))
}
