use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};

/// Static Constraint Solver for ! (Expr)
pub struct ConstraintSolver;

impl ConstraintSolver {
    pub fn verify_constraints(block: &ToposBlock) -> Result<()> {
        for stmt in &block.statements {
            if let Statement::Constraint(expr) = stmt {
                Self::solve_expr(expr)?;
            }
        }
        Ok(())
    }

    fn solve_expr(expr: &Expr) -> Result<()> {
        match expr {
            Expr::BinaryOp { op, lhs, rhs } => {
                // Static check for division by zero
                if *op == BinaryOpKind::Div {
                    if let Expr::Number(val) = **rhs {
                        if val == 0.0 {
                            return Err(HarmonyDisruption::LogicErr {
                                expr: format!("{:?} / 0", lhs),
                            });
                        }
                    }
                }
                Self::solve_expr(lhs)?;
                Self::solve_expr(rhs)?;
            }
            Expr::ShiftRight(inner) | Expr::ShiftLeft(inner) | Expr::AuditTrace(inner) => {
                Self::solve_expr(inner)?;
            }
            _ => {}
        }
        Ok(())
    }
}
