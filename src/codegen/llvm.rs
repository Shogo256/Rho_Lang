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

    /// Generate complete LLVM 22 IR (.ll) from ToposBlock AST
    pub fn generate_llvm_ir(&mut self, block: &ToposBlock) -> Result<String> {
        let mut ir = String::new();

        ir.push_str(&format!("; ModuleID = '{}'\n", self.module_name));
        ir.push_str("source_filename = \"rho_kernel.rho\"\n");
        ir.push_str("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n");
        ir.push_str("target triple = \"x86_64-unknown-linux-gnu\"\n\n");

        // Declarations for intrinsics
        ir.push_str("declare double @llvm.pow.f64(double, double)\n");
        ir.push_str("declare i32 @puts(ptr)\n\n");

        // Collect space declarations and shapes
        for stmt in &block.statements {
            match stmt {
                Statement::SpaceDef(decl) => {
                    self.space_shapes.insert(decl.name.clone(), decl.dimensions.clone());
                }
                Statement::ExtBind(bind) => {
                    self.space_shapes.insert(bind.space.name.clone(), bind.space.dimensions.clone());
                    self.ext_bindings.insert(bind.space.name.clone(), bind.address);
                }
                Statement::Flow { target, .. } => {
                    if let FlowTarget::Var(name) = target {
                        if !self.space_shapes.contains_key(name) {
                            self.space_shapes.insert(name.clone(), vec![4, 4]);
                        }
                    }
                }
                _ => {}
            }
        }

        // Generate JSON Metadata constant string
        let metadata_json = self.generate_metadata_json();
        let metadata_len = metadata_json.len() + 1;
        ir.push_str(&format!(
            "@.rho_meta_str = private unnamed_addr constant [{metadata_len} x i8] c\"{}\\00\", align 1\n\n",
            metadata_json.replace('\n', "").replace('"', "\\22")
        ));

        // 1. Static Execution Entrypoint: void @rho_kernel_exec()
        ir.push_str("define void @rho_kernel_exec() #0 {\n");
        ir.push_str("entry:\n");

        let mut in_ptr_name = String::from("%INPUT_ptr");
        let mut out_ptr_name = String::from("%OUTPUT_alloc");
        let mut used_external_input = false;

        for (name, shape) in &self.space_shapes {
            let total_size: usize = shape.iter().product();
            let total_size = if total_size == 0 { 1024 } else { total_size };

            if let Some(addr) = self.ext_bindings.get(name) {
                if name == "INPUT" {
                    used_external_input = true;
                }
                in_ptr_name = format!("%{name}_ptr");
                ir.push_str(&format!("  ; Zero-Copy Binding for [{name}] at address {:#X}\n", addr));
                ir.push_str(&format!("  %{name}_ptr = inttoptr i64 {addr} to ptr\n"));
            } else {
                if name == "OUTPUT" || out_ptr_name == "%OUTPUT_alloc" {
                    out_ptr_name = format!("%{name}_alloc");
                }
                ir.push_str(&format!("  ; Space Allocation [{name}] Shape: {:?}\n", shape));
                ir.push_str(&format!("  %{name}_alloc = alloca [{total_size} x double], align 64\n"));
            }
        }

        if !self.space_shapes.contains_key("OUTPUT") {
            ir.push_str("  ; Default output allocation for equilibrium flows\n");
            ir.push_str("  %OUTPUT_alloc = alloca [4 x double], align 64\n");
        }

        if used_external_input {
            ir.push_str("  ; External bindings are ignored for runtime execution; runtime input pointer is used instead\n");
        }

        // Lower statements to IR loops
        self.emit_statements_lowering(&block.statements, &mut ir, &in_ptr_name, &out_ptr_name, "entry");

        ir.push_str("  ret void\n");
        ir.push_str("}\n\n");

        // 2. Dynamic C-ABI Entrypoint: void @rho_kernel_exec_with_args(ptr %in_ptr, ptr %out_ptr)
        ir.push_str("define void @rho_kernel_exec_with_args(ptr %in_ptr, ptr %out_ptr) #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  ; Null Pointer Safety Check\n");
        ir.push_str("  %in_null = icmp eq ptr %in_ptr, null\n");
        ir.push_str("  br i1 %in_null, label %safety_fail, label %exec_start\n\n");

        ir.push_str("safety_fail:\n");
        ir.push_str("  ret void\n\n");

        ir.push_str("exec_start:\n");
        ir.push_str("  %out_null = icmp eq ptr %out_ptr, null\n");
        ir.push_str("  %out_effective = select i1 %out_null, ptr %in_ptr, ptr %out_ptr\n");
        self.emit_statements_lowering(&block.statements, &mut ir, "%in_ptr", "%out_effective", "exec_start");

        ir.push_str("  ret void\n");
        ir.push_str("}\n\n");

        // 3. C-ABI Metadata Export API: ptr @rho_kernel_metadata()
        ir.push_str("define ptr @rho_kernel_metadata() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  ret ptr @.rho_meta_str\n");
        ir.push_str("}\n\n");

        ir.push_str("attributes #0 = { nounwind uwtable \"target-cpu\"=\"x86-64-v3\" }\n");

        Ok(ir)
    }

    fn generate_metadata_json(&self) -> String {
        let mut json = String::from("{\"spaces\":[");
        let mut entries = Vec::new();
        for (name, shape) in &self.space_shapes {
            entries.push(format!("{{\"name\":\"{}\",\"shape\":{:?}}}", name, shape));
        }
        json.push_str(&entries.join(","));
        json.push_str("]}");
        json
    }

    fn emit_statements_lowering(
        &self,
        statements: &[Statement],
        ir: &mut String,
        in_ptr_sym: &str,
        out_ptr_sym: &str,
        entry_block: &str,
    ) {
        let sample_size = 4; // Process each element of the input array

        ir.push_str("  ; Element-wise Topological Dataflow Pipeline\n");
        ir.push_str("  br label %loop.header\n\n");

        ir.push_str("loop.header:\n");
        ir.push_str(&format!("  %idx = phi i64 [ 0, %{entry_block} ], [ %next_idx, %loop.body ]\n"));
        ir.push_str(&format!("  %cmp = icmp ult i64 %idx, {sample_size}\n"));
        ir.push_str("  br i1 %cmp, label %loop.body, label %loop.end\n\n");

        ir.push_str("loop.body:\n");
        ir.push_str(&format!("  %in_gep = getelementptr inbounds double, ptr {in_ptr_sym}, i64 %idx\n"));
        ir.push_str("  %val_in = load double, ptr %in_gep, align 8\n");

        let mut current_val_sym = String::from("%val_in");
        let mut val_counter = 0;

        for stmt in statements {
            if let Statement::Flow { src, target } = stmt {
                val_counter += 1;
                let next_sym = format!("%calc_val_{val_counter}");
                self.emit_expr_lowering(src, &current_val_sym, &next_sym, ir);
                current_val_sym = next_sym;

                if *target == FlowTarget::Equilibrium {
                    ir.push_str(&format!("  %out_gep = getelementptr inbounds double, ptr {out_ptr_sym}, i64 %idx\n"));
                    ir.push_str(&format!("  store double {current_val_sym}, ptr %out_gep, align 8\n"));
                    break;
                }
            }
        }

        ir.push_str("  %next_idx = add i64 %idx, 1\n");
        ir.push_str("  br label %loop.header\n\n");

        ir.push_str("loop.end:\n");
    }

    fn emit_expr_lowering(&self, expr: &Expr, in_sym: &str, out_sym: &str, ir: &mut String) {
        match expr {
            Expr::BinaryOp { op, lhs, rhs } => {
                let lhs_sym = format!("{out_sym}_lhs");
                let rhs_sym = format!("{out_sym}_rhs");
                self.emit_expr_lowering(lhs, in_sym, &lhs_sym, ir);
                self.emit_expr_lowering(rhs, in_sym, &rhs_sym, ir);

                match op {
                    BinaryOpKind::Add => {
                        ir.push_str(&format!("  {out_sym} = fadd double {lhs_sym}, {rhs_sym}\n"));
                    }
                    BinaryOpKind::Sub => {
                        ir.push_str(&format!("  {out_sym} = fsub double {lhs_sym}, {rhs_sym}\n"));
                    }
                    BinaryOpKind::Mul => {
                        ir.push_str(&format!("  {out_sym} = fmul double {lhs_sym}, {rhs_sym}\n"));
                    }
                    BinaryOpKind::Div => {
                        ir.push_str(&format!("  {out_sym} = fdiv double {lhs_sym}, {rhs_sym}\n"));
                    }
                    BinaryOpKind::Pow => {
                        ir.push_str(&format!("  {out_sym} = call double @llvm.pow.f64(double {lhs_sym}, double {rhs_sym})\n"));
                    }
                    BinaryOpKind::Gt => {
                        let bool_sym = format!("{out_sym}_bool");
                        ir.push_str(&format!("  {bool_sym} = fcmp ogt double {lhs_sym}, {rhs_sym}\n"));
                        ir.push_str(&format!("  {out_sym} = select i1 {bool_sym}, double {lhs_sym}, double 0.0\n"));
                    }
                    BinaryOpKind::Lt => {
                        let bool_sym = format!("{out_sym}_bool");
                        ir.push_str(&format!("  {bool_sym} = fcmp olt double {lhs_sym}, {rhs_sym}\n"));
                        ir.push_str(&format!("  {out_sym} = select i1 {bool_sym}, double {lhs_sym}, double 0.0\n"));
                    }
                    _ => {
                        ir.push_str(&format!("  {out_sym} = fadd double {lhs_sym}, {rhs_sym}\n"));
                    }
                }
            }
            Expr::ShiftRight(inner) => {
                let inner_sym = format!("{out_sym}_shift_in");
                self.emit_expr_lowering(inner, in_sym, &inner_sym, ir);
                ir.push_str(&format!("  {out_sym} = fmul double {inner_sym}, 1.0\n"));
            }
            Expr::ShiftLeft(inner) => {
                let inner_sym = format!("{out_sym}_shift_in");
                self.emit_expr_lowering(inner, in_sym, &inner_sym, ir);
                ir.push_str(&format!("  {out_sym} = fmul double {inner_sym}, 1.0\n"));
            }
            Expr::Number(val) => {
                ir.push_str(&format!("  {out_sym} = fadd double 0.0, {val:.6}\n"));
            }
            Expr::Var(name) => {
                if name == "INPUT" {
                    ir.push_str(&format!("  {out_sym} = fadd double 0.0, {in_sym}\n"));
                } else if name == "OUTPUT" {
                    ir.push_str(&format!("  {out_sym} = fadd double 0.0, {in_sym}\n"));
                } else {
                    ir.push_str(&format!("  {out_sym} = fadd double 0.0, {in_sym}\n"));
                }
            }
            Expr::AuditTrace(inner) => {
                self.emit_expr_lowering(inner, in_sym, out_sym, ir);
            }
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
