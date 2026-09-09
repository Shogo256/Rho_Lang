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
}

impl ToposBlock {
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
