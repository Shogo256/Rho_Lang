use crate::ast::*;
use std::collections::HashSet;

/// Generate TLA+ Formal Specification (rho_harmony.tla) from ToposBlock AST
pub fn generate_tla_spec(module_name: &str, block: &ToposBlock) -> String {
    let mut tla = String::new();
    tla.push_str(&format!("---------------- MODULE {} ----------------\n", module_name));
    tla.push_str("EXTENDS Naturals, Sequences, FiniteSets, TLC\n\n");

    // 1. Extract variables from space declarations
    let mut variables = HashSet::new();
    for stmt in &block.statements {
        match stmt {
            Statement::SpaceDef(decl) => {
                variables.insert(decl.name.clone());
            }
            Statement::ExtBind(bind) => {
                variables.insert(bind.space.name.clone());
            }
            Statement::Flow { target, .. } => {
                if let FlowTarget::Var(name) = target {
                    variables.insert(name.clone());
                }
            }
            _ => {}
        }
    }

    let var_list: Vec<String> = variables.into_iter().collect();
    if !var_list.is_empty() {
        tla.push_str(&format!("VARIABLES {}\n\n", var_list.join(", ")));
    }

    // 2. Initial state Init
    tla.push_str("Init ==\n");
    for (i, var) in var_list.iter().enumerate() {
        let sep = if i == var_list.len() - 1 { "" } else { " /\\" };
        tla.push_str(&format!("  /\\ {} = [i \\in 1..10 |-> 0]{}\n", var, sep));
    }
    tla.push_str("\n");

    // 3. Shift operators in TLA+
    tla.push_str("\\* ShiftRight (▷): Shift elements right by 1 cell, fill left boundary with 0\n");
    tla.push_str("ShiftRight(seq) == [i \\in 1..Len(seq) |-> IF i = 1 THEN 0 ELSE seq[i-1]]\n\n");

    tla.push_str("\\* ShiftLeft (▽): Shift elements left by 1 cell, fill right boundary with 0\n");
    tla.push_str("ShiftLeft(seq) == [i \\in 1..Len(seq) |-> IF i = Len(seq) THEN 0 ELSE seq[i+1]]\n\n");

    // 4. State transition Next
    tla.push_str("Next ==\n");
    tla.push_str("  /\\ UNCHANGED <<");
    tla.push_str(&var_list.join(", "));
    tla.push_str(">>\n\n");

    // 5. Spec & Invariants
    tla.push_str("Spec == Init /\\ [][Next]_<<");
    tla.push_str(&var_list.join(", "));
    tla.push_str(">>\n\n");

    tla.push_str("====================================================\n");
    tla
}
