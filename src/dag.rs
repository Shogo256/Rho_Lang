use crate::ast::*;
use crate::error::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DagNode {
    pub label: String,
    pub expr: Option<Expr>,
}

pub struct RhoDag {
    pub graph: DiGraph<DagNode, String>,
    pub node_map: HashMap<String, NodeIndex>,
}

impl RhoDag {
    pub fn build(block: &ToposBlock) -> Result<Self> {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        // 1. Register declaration nodes
        for stmt in &block.statements {
            match stmt {
                Statement::SpaceDef(decl) => {
                    let idx = graph.add_node(DagNode {
                        label: decl.name.clone(),
                        expr: None,
                    });
                    node_map.insert(decl.name.clone(), idx);
                }
                Statement::ExtBind(bind) => {
                    let idx = graph.add_node(DagNode {
                        label: format!("&[{:#X}]:{}", bind.address, bind.space.name),
                        expr: None,
                    });
                    node_map.insert(bind.space.name.clone(), idx);
                }
                _ => {}
            }
        }

        // 2. Build directed edges from flow statements (→)
        for stmt in &block.statements {
            if let Statement::Flow { src, target } = stmt {
                let target_label = match target {
                    FlowTarget::Var(name) => name.clone(),
                    FlowTarget::Equilibrium => "EQUILIBRIUM(=)".to_string(),
                };

                let target_idx = *node_map.entry(target_label.clone()).or_insert_with(|| {
                    graph.add_node(DagNode {
                        label: target_label.clone(),
                        expr: Some(src.clone()),
                    })
                });

                let dependencies = extract_dependencies(src);
                for dep in dependencies {
                    if let Some(&dep_idx) = node_map.get(&dep) {
                        graph.add_edge(dep_idx, target_idx, "flow".to_string());
                    }
                }
            }
        }

        Ok(Self { graph, node_map })
    }

    /// `$ Audit Tracer`: Topological reverse traversal & DAG log tree output
    pub fn print_audit_trace(&self) -> String {
        let mut out = String::new();
        out.push_str("====================================================\n");
        out.push_str("   ρ (RHO) Language - Active Audit DAG Trace ($)\n");
        out.push_str("====================================================\n");

        if let Some(&eq_idx) = self.node_map.get("EQUILIBRIUM(=)") {
            self.trace_recursive(eq_idx, 0, &mut out);
        } else {
            out.push_str("[WARNING] EQUILIBRIUM(=) node not found.\n");
        }

        out.push_str("====================================================\n");
        out
    }

    fn trace_recursive(&self, node_idx: NodeIndex, indent: usize, out: &mut String) {
        let node = &self.graph[node_idx];
        let pad = "  ".repeat(indent);
        out.push_str(&format!("{}[Node] {}\n", pad, node.label));

        if let Some(ref expr) = node.expr {
            out.push_str(&format!("{}  └─ Expr: {:?}\n", pad, expr));
        }

        let mut neighbors = self
            .graph
            .neighbors_directed(node_idx, petgraph::Direction::Incoming);

        while let Some(parent) = neighbors.next() {
            self.trace_recursive(parent, indent + 1, out);
        }
    }

    /// Extract independent nodes that can be executed in parallel
    pub fn extract_parallel_nodes(&self) -> Vec<Vec<String>> {
        let mut parallel_groups = Vec::new();

        let root_nodes: Vec<String> = self
            .graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, petgraph::Direction::Incoming)
                    .count()
                    == 0
            })
            .map(|idx| self.graph[idx].label.clone())
            .collect();

        if !root_nodes.is_empty() {
            parallel_groups.push(root_nodes);
        }

        parallel_groups
    }
}

fn extract_dependencies(expr: &Expr) -> Vec<String> {
    let mut deps = Vec::new();
    match expr {
        Expr::Var(name) => {
            if name != "𝜏" && name != "τ" {
                deps.push(name.clone());
            }
        }
        Expr::ShiftRight(inner) | Expr::ShiftLeft(inner) | Expr::AuditTrace(inner) => {
            deps.extend(extract_dependencies(inner));
        }
        Expr::BinaryOp { lhs, rhs, .. } => {
            deps.extend(extract_dependencies(lhs));
            deps.extend(extract_dependencies(rhs));
        }
        Expr::Number(_) => {}
    }
    deps
}
