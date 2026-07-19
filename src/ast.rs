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

/// Topological Expression
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Var(String),
    Number(f64),
    ShiftRight(Box<Expr>), // ▷
    ShiftLeft(Box<Expr>),  // ▽
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
