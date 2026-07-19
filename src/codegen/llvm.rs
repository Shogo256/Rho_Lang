use crate::ast::*;
use crate::error::Result;
use std::collections::HashMap;
use std::fs;
use std::process::Command;

pub struct LlvmCodeGen {
    pub module_name: String,
    pub space_shapes: HashMap<String, Vec<usize>>,
    pub ext_bindings: HashMap<String, u64>,
}

impl LlvmCodeGen {
    pub fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
            space_shapes: HashMap::new(),
            ext_bindings: HashMap::new(),
        }
    }

    /// Generate LLVM 22 IR (.ll) from AST ToposBlock
    pub fn generate_llvm_ir(&mut self, block: &ToposBlock) -> Result<String> {
        let mut ir = String::new();

        ir.push_str(&format!("; ModuleID = '{}'\n", self.module_name));
        ir.push_str("source_filename = \"rho_kernel.rho\"\n");
        ir.push_str("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n");
        ir.push_str("target triple = \"x86_64-unknown-linux-gnu\"\n\n");

        // Collect space declarations
        for stmt in &block.statements {
            match stmt {
                Statement::SpaceDef(decl) => {
                    self.space_shapes.insert(decl.name.clone(), decl.dimensions.clone());
                }
                Statement::ExtBind(bind) => {
                    self.space_shapes.insert(bind.space.name.clone(), bind.space.dimensions.clone());
                    self.ext_bindings.insert(bind.space.name.clone(), bind.address);
                }
                _ => {}
            }
        }

        // 1. Function prototype: void @rho_kernel_exec()
        ir.push_str("define void @rho_kernel_exec() #0 {\n");
        ir.push_str("entry:\n");

        // Memory allocation and zero-copy pointer bindings
        for (name, shape) in &self.space_shapes {
            let total_size: usize = shape.iter().product();
            let total_size = if total_size == 0 { 1024 } else { total_size };

            if let Some(addr) = self.ext_bindings.get(name) {
                ir.push_str(&format!(
                    "  ; Zero-Copy Binding for [{name}] at address {:#X}\n",
                    addr
                ));
                ir.push_str(&format!(
                    "  %{name}_ptr = inttoptr i64 {addr} to ptr\n"
                ));
            } else {
                ir.push_str(&format!(
                    "  ; Space Allocation [{name}] Shape: {:?}\n",
                    shape
                ));
                ir.push_str(&format!(
                    "  %{name}_alloc = alloca [{total_size} x double], align 64\n"
                ));
            }
        }

        ir.push_str("\n  ; Dataflow Pipeline (Clocks-Eliminated Execution)\n");

        for stmt in &block.statements {
            if let Statement::Flow { src, target } = stmt {
                let target_name = match target {
                    FlowTarget::Var(n) => n.as_str(),
                    FlowTarget::Equilibrium => "EQUILIBRIUM_OUT",
                };
                ir.push_str(&format!("  ; Flow: -> {}\n", target_name));
                self.emit_expr_ir(src, &mut ir);
            }
        }

        ir.push_str("\n  ; Equilibrium Point Convergence (=)\n");
        ir.push_str("  ret void\n");
        ir.push_str("}\n\n");

        // 2. Extended C-ABI Function prototype: void @rho_kernel_exec_with_args(ptr %in_ptr, ptr %out_ptr)
        ir.push_str("define void @rho_kernel_exec_with_args(ptr %in_ptr, ptr %out_ptr) #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  ; Direct Argument Buffer Binding\n");
        ir.push_str("  call void @rho_kernel_exec()\n");
        ir.push_str("  ret void\n");
        ir.push_str("}\n\n");

        ir.push_str("attributes #0 = { nounwind uwtable \"target-cpu\"=\"x86-64-v3\" }\n");

        Ok(ir)
    }

    fn emit_expr_ir(&self, expr: &Expr, ir: &mut String) {
        match expr {
            Expr::ShiftRight(inner) => {
                ir.push_str("  ; Vector Shift Right (▷)\n");
                self.emit_expr_ir(inner, ir);
            }
            Expr::ShiftLeft(inner) => {
                ir.push_str("  ; Vector Shift Left (▽)\n");
                self.emit_expr_ir(inner, ir);
            }
            Expr::BinaryOp { op, lhs, rhs } => {
                self.emit_expr_ir(lhs, ir);
                self.emit_expr_ir(rhs, ir);
                ir.push_str(&format!("  ; Topological Op: {}\n", op));
            }
            _ => {}
        }
    }

    /// Compile LLVM IR to native shared library (.so) using clang-22
    pub fn compile_to_so(&self, ir_content: &str, output_path: &str) -> std::io::Result<()> {
        let temp_ll = format!("{}.ll", output_path);
        fs::write(&temp_ll, ir_content)?;

        let status = Command::new("clang-22")
            .args(&["-shared", "-fPIC", "-O3", &temp_ll, "-o", output_path])
            .status()
            .or_else(|_| {
                Command::new("clang")
                    .args(&["-shared", "-fPIC", "-O3", &temp_ll, "-o", output_path])
                    .status()
            })?;

        if status.success() {
            let _ = fs::remove_file(temp_ll);
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "clang-22 Compilation Failed",
            ))
        }
    }
}
