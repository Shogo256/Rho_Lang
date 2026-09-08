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
