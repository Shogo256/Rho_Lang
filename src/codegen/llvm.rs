use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use std::collections::BTreeMap;
use std::fs;
use std::process::Command;

/// Spaces larger than this are heap-allocated instead of living on the stack,
/// so a 1024x1024 grid cannot blow the 8 MB default stack.
const STACK_LIMIT_BYTES: usize = 64 * 1024;

/// Lanes per vector step. Four doubles is one AVX register; clang widens
/// further when the target allows it.
const VECTOR_WIDTH: usize = 4;

/// Whether a chunk of a flow is lowered one cell at a time or a vector at a time.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Scalar,
    Vector(usize),
}

impl Mode {
    fn lanes(self) -> usize {
        match self {
            Mode::Scalar => 1,
            Mode::Vector(w) => w,
        }
    }

    fn ty(self) -> String {
        match self {
            Mode::Scalar => "double".to_string(),
            Mode::Vector(w) => format!("<{w} x double>"),
        }
    }

    fn int_ty(self) -> String {
        match self {
            Mode::Scalar => "i64".to_string(),
            Mode::Vector(w) => format!("<{w} x i64>"),
        }
    }

    fn bool_ty(self) -> String {
        match self {
            Mode::Scalar => "i1".to_string(),
            Mode::Vector(w) => format!("<{w} x i1>"),
        }
    }

    fn zero(self) -> String {
        match self {
            Mode::Scalar => "0.0".to_string(),
            Mode::Vector(_) => "zeroinitializer".to_string(),
        }
    }

    /// A floating constant broadcast across every lane.
    fn splat(self, value: f64) -> String {
        let lit = LlvmCodeGen::f64_literal(value);
        match self {
            Mode::Scalar => lit,
            Mode::Vector(w) => {
                let lanes: Vec<String> = (0..w).map(|_| format!("double {lit}")).collect();
                format!("<{}>", lanes.join(", "))
            }
        }
    }

    fn int_splat(self, value: u64) -> String {
        match self {
            Mode::Scalar => value.to_string(),
            Mode::Vector(w) => {
                let lanes: Vec<String> = (0..w).map(|_| format!("i64 {value}")).collect();
                format!("<{}>", lanes.join(", "))
            }
        }
    }

    fn pow_intrinsic(self) -> String {
        match self {
            Mode::Scalar => "@llvm.pow.f64".to_string(),
            Mode::Vector(w) => format!("@llvm.pow.v{w}f64"),
        }
    }
}

/// How one flow's sweep is split between scalar and vector loops.
enum Sweep {
    AllScalar,
    Split {
        vector_start: u64,
        vector_end: u64,
        width: usize,
        total: u64,
    },
}

impl Sweep {
    fn describe(&self) -> String {
        match self {
            Sweep::AllScalar => " (scalar)".to_string(),
            Sweep::Split {
                vector_start,
                vector_end,
                width,
                ..
            } => format!(
                " (scalar 0..{vector_start}, {width}-wide vector {vector_start}..{vector_end}, scalar tail)"
            ),
        }
    }
}

/// Buffers visible to a kernel body: space name -> LLVM pointer symbol.
struct Buffers {
    map: BTreeMap<String, String>,
    /// Buffer holding each precomputed fold, keyed by the AST node that asked
    /// for it. Folds are lowered before the sweep that reads them, which keeps
    /// the sweep body straight-line and lets it stay vectorised.
    folds: BTreeMap<usize, String>,
    heap: Vec<String>,
    /// Cells this body sweeps: a literal, or an i64 symbol for a bounded call.
    bound: String,
}

pub struct LlvmCodeGen {
    pub module_name: String,
    /// Declared and inferred shapes. A BTreeMap keeps IR generation
    /// deterministic: with a HashMap the same source emitted different IR on
    /// every run, because iteration order leaked into allocation order.
    pub space_shapes: BTreeMap<String, Vec<usize>>,
    pub ext_bindings: BTreeMap<String, u64>,
    /// Value bound to the threshold symbol 𝜏.
    pub tau: f64,
    /// Addresses supplied at compile time, overriding the `&[0x...]` literals.
    /// This is what makes the zero-copy path reachable from a host language:
    /// the caller passes the address of its own buffer.
    pub binding_overrides: BTreeMap<String, u64>,
    /// Emit vector loops for the constant-length entrypoints.
    pub simd: bool,
    /// Source line of each statement, copied from the block being lowered.
    statement_lines: Vec<usize>,
    /// What the solver proved, embedded in the artifact so a caller can check
    /// it at load time instead of trusting a line from the build log.
    contract_json: Option<String>,
    /// Number of cells swept by every flow loop.
    elements: usize,
    /// Source line of the flow being lowered, so a lowering failure can point
    /// at the statement the reader wrote.
    current_line: std::cell::Cell<usize>,
}

impl LlvmCodeGen {
    pub fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
            space_shapes: BTreeMap::new(),
            ext_bindings: BTreeMap::new(),
            tau: 0.0,
            binding_overrides: BTreeMap::new(),
            simd: true,
            statement_lines: Vec::new(),
            contract_json: None,
            elements: 0,
            current_line: std::cell::Cell::new(0),
        }
    }

    /// Bind the threshold symbol 𝜏 to a concrete value (default 0.0).
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Record what the solver proved, so it ships with the kernel.
    pub fn with_contract(mut self, contract_json: String) -> Self {
        self.contract_json = Some(contract_json);
        self
    }

    /// Turn vector lowering off and emit only scalar loops.
    pub fn without_simd(mut self) -> Self {
        self.simd = false;
        self
    }

    /// Bind a space to a concrete address, overriding any `&[0x...]` literal.
    pub fn bind(mut self, name: &str, address: u64) -> Self {
        self.binding_overrides.insert(name.to_string(), address);
        self
    }

    /// Cells swept by each flow loop; also what a caller must allocate.
    pub fn element_count(&self) -> usize {
        self.elements
    }

    /// Generate complete LLVM IR (.ll) from a ToposBlock AST
    pub fn generate_llvm_ir(&mut self, block: &ToposBlock) -> Result<String> {
        self.statement_lines = block.lines.clone();
        self.collect_shapes(block);
        for (name, addr) in &self.binding_overrides {
            self.ext_bindings.insert(name.clone(), *addr);
            self.space_shapes
                .entry(name.clone())
                .or_insert_with(|| vec![4]);
        }
        self.elements = self.resolve_element_count();

        let mut ir = String::new();

        ir.push_str(&format!("; ModuleID = '{}'\n", self.module_name));
        ir.push_str("source_filename = \"rho_kernel.rho\"\n");
        // No target triple or datalayout: clang fills in the host's. Hardcoding
        // x86_64-unknown-linux-gnu made every build warn about an overridden
        // triple and made the compiler unusable off x86-64 Linux.
        ir.push('\n');

        ir.push_str("declare double @llvm.pow.f64(double, double)\n");
        if self.simd {
            ir.push_str(&format!(
                "declare <{w} x double> @llvm.pow.v{w}f64(<{w} x double>, <{w} x double>)\n",
                w = VECTOR_WIDTH
            ));
        }
        ir.push_str("declare ptr @malloc(i64)\n");
        ir.push_str("declare void @free(ptr)\n\n");

        let (meta_escaped, meta_len) = Self::c_string(&self.generate_metadata_json());
        ir.push_str(&format!(
            "@.rho_meta_str = private unnamed_addr constant [{meta_len} x i8] c\"{meta_escaped}\", align 1\n\n"
        ));


        // 1. Static entrypoint: void @rho_kernel_exec()
        let elements = self.elements.to_string();
        ir.push_str("define void @rho_kernel_exec() #0 {\n");
        ir.push_str("entry:\n");

        // Every space a flow never writes is read from memory the caller owns,
        // so the entrypoint is only meaningful once all of them are bound. A
        // kernel with two inputs and no INPUT at all is perfectly ordinary.
        let sources = self.source_spaces(block);
        let unbound: Vec<&String> = sources
            .iter()
            .filter(|name| !self.ext_bindings.contains_key(*name))
            .collect();

        if unbound.is_empty() && !self.ext_bindings.is_empty() {
            let mut heap = Vec::new();
            let in_sym = match self.ext_bindings.get("INPUT").copied() {
                Some(addr) => {
                    ir.push_str(&format!("  ; Zero-copy binding for [INPUT] at {addr:#X}\n"));
                    ir.push_str("  %INPUT_ext = inttoptr i64 ".to_string().as_str());
                    ir.push_str(&format!("{addr} to ptr\n"));
                    "%INPUT_ext".to_string()
                }
                None => self.emit_scratch(&mut ir, "INPUT_local", self.elements, &mut heap),
            };
            // An unbound OUTPUT means the grid is transformed in place.
            let out_sym = match self.ext_bindings.get("OUTPUT").copied() {
                Some(addr) => {
                    ir.push_str(&format!("  ; Zero-copy binding for [OUTPUT] at {addr:#X}\n"));
                    ir.push_str(&format!("  %OUTPUT_ext = inttoptr i64 {addr} to ptr\n"));
                    "%OUTPUT_ext".to_string()
                }
                None if self.ext_bindings.contains_key("INPUT") => in_sym.clone(),
                None => self.emit_scratch(&mut ir, "OUTPUT_local", self.elements, &mut heap),
            };
            self.emit_body(block, &mut ir, &in_sym, &out_sym, "entry", heap, &elements)?;
        } else {
            ir.push_str("  ; Not every space this kernel reads has an & binding:\n");
            for name in &unbound {
                ir.push_str(&format!("  ;   [{name}] is unbound\n"));
            }
            ir.push_str("  ; Use rho_kernel_exec_with_args, or compile with --bind NAME=0x...\n");
            ir.push_str("  ret void\n");
        }
        ir.push_str("}\n\n");

        // 2. C-ABI entrypoint: void @rho_kernel_exec_with_args(ptr, ptr)
        ir.push_str("define void @rho_kernel_exec_with_args(ptr %in_ptr, ptr %out_ptr) #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  ; Null pointer safety check\n");
        ir.push_str("  %in_null = icmp eq ptr %in_ptr, null\n");
        ir.push_str("  br i1 %in_null, label %safety_fail, label %exec_start\n\n");
        ir.push_str("safety_fail:\n");
        ir.push_str("  ret void\n\n");
        ir.push_str("exec_start:\n");
        ir.push_str("  %out_null = icmp eq ptr %out_ptr, null\n");
        ir.push_str("  %out_effective = select i1 %out_null, ptr %in_ptr, ptr %out_ptr\n");
        self.emit_body(
            block,
            &mut ir,
            "%in_ptr",
            "%out_effective",
            "exec_start",
            Vec::new(),
            &elements,
        )?;
        ir.push_str("}\n\n");

        // 3. Length-checked C-ABI entrypoint. The two-argument form has to trust
        // the caller to have allocated element_count() doubles; this one clamps
        // the sweep to whatever the caller actually owns.
        ir.push_str(
            "define void @rho_kernel_exec_bounded(ptr %in_ptr, ptr %out_ptr, i64 %n) #0 {\n",
        );
        ir.push_str("entry:\n");
        ir.push_str("  %b_in_null = icmp eq ptr %in_ptr, null\n");
        ir.push_str("  br i1 %b_in_null, label %bounded_fail, label %bounded_start\n\n");
        ir.push_str("bounded_fail:\n");
        ir.push_str("  ret void\n\n");
        ir.push_str("bounded_start:\n");
        ir.push_str("  %b_out_null = icmp eq ptr %out_ptr, null\n");
        ir.push_str("  %b_out_effective = select i1 %b_out_null, ptr %in_ptr, ptr %out_ptr\n");
        ir.push_str(&format!(
            "  %b_short = icmp ult i64 %n, {}\n",
            self.elements
        ));
        ir.push_str(&format!(
            "  %sweep = select i1 %b_short, i64 %n, i64 {}\n",
            self.elements
        ));
        self.emit_body(
            block,
            &mut ir,
            "%in_ptr",
            "%b_out_effective",
            "bounded_start",
            Vec::new(),
            "%sweep",
        )?;
        ir.push_str("}\n\n");

        // 4. Metadata export: ptr @rho_kernel_metadata()
        ir.push_str("define ptr @rho_kernel_metadata() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  ret ptr @.rho_meta_str\n");
        ir.push_str("}\n\n");

        // 5. Required buffer length: i64 @rho_kernel_element_count()
        ir.push_str("define i64 @rho_kernel_element_count() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str(&format!("  ret i64 {}\n", self.elements));
        ir.push_str("}\n\n");

        // No target-cpu pin: -O3 vectorises for whatever clang is targeting.
        ir.push_str("attributes #0 = { nounwind uwtable }\n");

        Ok(ir)
    }

    // ---------------------------------------------------------------- shapes

    fn collect_shapes(&mut self, block: &ToposBlock) {
        let mut shapes = std::mem::take(&mut self.space_shapes);

        for stmt in &block.statements {
            match stmt {
                Statement::SpaceDef(decl) => {
                    shapes.insert(decl.name.clone(), decl.dimensions.clone());
                }
                Statement::ExtBind(bind) => {
                    shapes.insert(bind.space.name.clone(), bind.space.dimensions.clone());
                    self.ext_bindings.insert(bind.space.name.clone(), bind.address);
                }
                _ => {}
            }
        }

        // A flow target inherits the shape of its source rather than a fixed
        // guess, so temporaries are never smaller than the loop that fills them.
        for stmt in &block.statements {
            if let Statement::Flow {
                src,
                target: FlowTarget::Var(name),
            } = stmt
            {
                if !shapes.contains_key(name) {
                    let inferred = Self::expr_shape(src, &shapes)
                        .or_else(|| Self::primary_shape(&shapes))
                        .unwrap_or_else(|| vec![4]);
                    shapes.insert(name.clone(), inferred);
                }
            }
        }

        self.space_shapes = shapes;
    }

    fn expr_shape(expr: &Expr, shapes: &BTreeMap<String, Vec<usize>>) -> Option<Vec<usize>> {
        match expr {
            Expr::Var(name) => shapes.get(name).cloned(),
            Expr::Shift { operand: inner, .. } | Expr::AuditTrace(inner) => {
                Self::expr_shape(inner, shapes)
            }
            Expr::Lift { axis, operand } => {
                let inner = Self::expr_shape(operand, shapes)?;
                crate::ast::shape_with_unit_axis(&inner, *axis)
            }
            Expr::Reduce { axis, operand, .. } => {
                let inner = Self::expr_shape(operand, shapes)?;
                let a = axis.unwrap_or_else(|| crate::ast::default_axis(&inner));
                (a < inner.len()).then(|| crate::ast::shape_without_axis(&inner, a))
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                match (Self::expr_shape(lhs, shapes), Self::expr_shape(rhs, shapes)) {
                    (Some(l), Some(r)) => crate::ast::broadcast_shapes(&l, &r),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            Expr::Number(_) => None,
        }
    }

    fn primary_shape(shapes: &BTreeMap<String, Vec<usize>>) -> Option<Vec<usize>> {
        shapes
            .get("INPUT")
            .cloned()
            .or_else(|| shapes.values().next().cloned())
    }

    fn resolve_element_count(&self) -> usize {
        Self::primary_shape(&self.space_shapes)
            .map(|s| s.iter().product::<usize>())
            .filter(|n| *n > 0)
            .unwrap_or(4)
    }

    // --------------------------------------------------------------- buffers

    fn emit_scratch(
        &self,
        ir: &mut String,
        label: &str,
        count: usize,
        heap: &mut Vec<String>,
    ) -> String {
        let count = count.max(1);
        let sym = format!("%{label}");
        if count * 8 > STACK_LIMIT_BYTES {
            ir.push_str(&format!("  ; [{label}] {count} doubles -> heap\n"));
            ir.push_str(&format!("  {sym} = call ptr @malloc(i64 {})\n", count * 8));
            heap.push(sym.clone());
        } else {
            ir.push_str(&format!("  ; [{label}] {count} doubles -> stack\n"));
            ir.push_str(&format!("  {sym} = alloca [{count} x double], align 64\n"));
        }
        sym
    }

    fn emit_buffers(
        &self,
        ir: &mut String,
        in_sym: &str,
        out_sym: &str,
        heap: Vec<String>,
        bound: &str,
    ) -> Buffers {
        let mut map = BTreeMap::new();
        let mut heap = heap;

        map.insert("INPUT".to_string(), in_sym.to_string());
        map.insert("OUTPUT".to_string(), out_sym.to_string());

        for (name, shape) in &self.space_shapes {
            if name == "INPUT" || name == "OUTPUT" {
                continue;
            }
            let sanitized = self.sanitize_ident(name);

            // A space bound to an address is the caller's memory, not ours.
            // Only INPUT and OUTPUT arrive as parameters; every other space
            // reaches the kernel through its binding, which is what lets a
            // kernel take more than one input.
            if let Some(addr) = self.ext_bindings.get(name) {
                let sym = format!("%{sanitized}_ext");
                ir.push_str(&format!("  ; Zero-copy binding for [{name}] at {addr:#X}\n"));
                ir.push_str(&format!("  {sym} = inttoptr i64 {addr} to ptr\n"));
                map.insert(name.clone(), sym);
                continue;
            }

            let count = shape.iter().product::<usize>().max(1);
            let label = format!("{sanitized}_buf");
            let sym = self.emit_scratch(ir, &label, count, &mut heap);
            map.insert(name.clone(), sym);
        }

        Buffers {
            map,
            folds: BTreeMap::new(),
            heap,
            bound: bound.to_string(),
        }
    }

    // ---------------------------------------------------------------- bodies

    #[allow(clippy::too_many_arguments)]
    fn emit_body(
        &self,
        block: &ToposBlock,
        ir: &mut String,
        in_sym: &str,
        out_sym: &str,
        entry_label: &str,
        heap: Vec<String>,
        bound: &str,
    ) -> Result<()> {
        let mut bufs = self.emit_buffers(ir, in_sym, out_sym, heap, bound);
        let mut counter = 0usize;
        self.emit_flows(&block.statements, ir, &mut bufs, entry_label, &mut counter)?;
        for ptr in &bufs.heap {
            ir.push_str(&format!("  call void @free(ptr {ptr})\n"));
        }
        ir.push_str("  ret void\n");
        Ok(())
    }

    /// Each `→` is its own full sweep of the grid. Fusing them into one loop
    /// made every space read the previous statement's scalar instead of its
    /// own buffer, and made neighbourhood shifts read cells that the current
    /// sweep had not written yet.
    ///
    /// A sweep of known length is split into a scalar head, a vector body and a
    /// scalar tail. The head and tail cover the cells whose neighbours would fall
    /// outside the buffer, so every vector load in the body is in bounds.
    fn emit_flows(
        &self,
        statements: &[Statement],
        ir: &mut String,
        bufs: &mut Buffers,
        entry_label: &str,
        counter: &mut usize,
    ) -> Result<()> {
        let mut pred = entry_label.to_string();
        let mut loop_id = 0usize;

        for (index, stmt) in statements.iter().enumerate() {
            let Statement::Flow { src, target } = stmt else {
                continue;
            };
            self.current_line.set(self.statement_lines.get(index).copied().unwrap_or(0));

            let target_ptr = match target {
                FlowTarget::Var(name) => self.lookup(bufs, name)?,
                FlowTarget::Equilibrium => self.lookup(bufs, "OUTPUT")?,
            };

            loop_id += 1;

            // Fold every reduction in this flow into its own buffer first. The
            // sweep below then reads a plain array, so it stays straight-line.
            pred = self.emit_fold_prepass(src, ir, bufs, &pred, counter)?;

            let sweep = self.sweep_length(src, target, bufs);
            let result_shape = self.sweep_shape(src, target);
            // A stretched read is not contiguous, so it cannot be vector loaded.
            let plan = if self.needs_broadcast(src, &result_shape, &[]) {
                Sweep::AllScalar
            } else {
                self.plan_sweep(src, &sweep)?
            };
            ir.push_str(&format!(
                "  ; Flow {loop_id}: sweep {sweep} cells{}\n",
                plan.describe()
            ));

            match plan {
                Sweep::AllScalar => {
                    pred = self.emit_range_loop(
                        ir,
                        &format!("f{loop_id}"),
                        &pred,
                        "0",
                        &sweep,
                        Mode::Scalar,
                        src,
                        &target_ptr,
                        bufs,
                        counter,
                        &result_shape,
                    )?;
                }
                Sweep::Split {
                    vector_start,
                    vector_end,
                    width,
                    total,
                } => {
                    pred = self.emit_range_loop(
                        ir,
                        &format!("f{loop_id}.head"),
                        &pred,
                        "0",
                        &vector_start.to_string(),
                        Mode::Scalar,
                        src,
                        &target_ptr,
                        bufs,
                        counter,
                        &result_shape,
                    )?;
                    pred = self.emit_range_loop(
                        ir,
                        &format!("f{loop_id}.vec"),
                        &pred,
                        &vector_start.to_string(),
                        &vector_end.to_string(),
                        Mode::Vector(width),
                        src,
                        &target_ptr,
                        bufs,
                        counter,
                        &result_shape,
                    )?;
                    pred = self.emit_range_loop(
                        ir,
                        &format!("f{loop_id}.tail"),
                        &pred,
                        &vector_end.to_string(),
                        &total.to_string(),
                        Mode::Scalar,
                        src,
                        &target_ptr,
                        bufs,
                        counter,
                        &result_shape,
                    )?;
                }
            }

            if matches!(target, FlowTarget::Equilibrium) {
                break;
            }
        }

        Ok(())
    }

    /// Lower every fold in `expr` into its own buffer, innermost first, so a
    /// nested fold's result exists before the fold that consumes it.
    /// Returns the block control flow lands on.
    fn emit_fold_prepass(
        &self,
        expr: &Expr,
        ir: &mut String,
        bufs: &mut Buffers,
        pred: &str,
        counter: &mut usize,
    ) -> Result<String> {
        let mut block = pred.to_string();
        match expr {
            Expr::Var(_) | Expr::Number(_) => {}
            Expr::AuditTrace(inner)
            | Expr::Shift { operand: inner, .. }
            | Expr::Lift { operand: inner, .. } => {
                block = self.emit_fold_prepass(inner, ir, bufs, &block, counter)?;
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                block = self.emit_fold_prepass(lhs, ir, bufs, &block, counter)?;
                block = self.emit_fold_prepass(rhs, ir, bufs, &block, counter)?;
            }
            Expr::Reduce { op, axis, operand } => {
                block = self.emit_fold_prepass(operand, ir, bufs, &block, counter)?;
                block = self.emit_fold(expr, *op, *axis, operand, ir, bufs, &block, counter)?;
            }
        }
        Ok(block)
    }

    /// One fold: an outer loop over the cells that survive, and an inner loop
    /// walking the axis being collapsed.
    #[allow(clippy::too_many_arguments)]
    fn emit_fold(
        &self,
        node: &Expr,
        op: FoldOp,
        axis: Option<usize>,
        operand: &Expr,
        ir: &mut String,
        bufs: &mut Buffers,
        pred: &str,
        counter: &mut usize,
    ) -> Result<String> {
        let shape = Self::expr_shape(operand, &self.space_shapes).ok_or_else(|| {
            HarmonyDisruption::LoweringErr {
                line: self.current_line.get(),
                detail: format!("{op} needs an operand with a known shape"),
            }
        })?;
        let a = axis.unwrap_or_else(|| crate::ast::default_axis(&shape));
        if a >= shape.len() {
            return Err(HarmonyDisruption::LoweringErr {
                line: self.current_line.get(),
                detail: format!(
                    "axis {a} is out of range for a rank-{} space (shape {shape:?})",
                    shape.len()
                ),
            });
        }

        let extent = shape[a].max(1);
        let inner: usize = shape[a + 1..].iter().product::<usize>().max(1);
        let outer: usize = shape[..a].iter().product::<usize>().max(1);
        let out_cells = (outer * inner).max(1);

        *counter += 1;
        let id = *counter;
        let label = format!("fold{id}");
        let buffer = self.emit_scratch(ir, &format!("{label}_buf"), out_cells, &mut bufs.heap);

        ir.push_str(&format!(
            "  ; {op} over axis {a} of {shape:?} -> {out_cells} cells\n"
        ));
        ir.push_str(&format!("  br label %{label}.header\n\n"));

        // Outer loop: one iteration per surviving cell.
        ir.push_str(&format!("{label}.header:\n"));
        ir.push_str(&format!(
            "  %{label}.j = phi i64 [ 0, %{pred} ], [ %{label}.j.next, %{label}.tail ]\n"
        ));
        ir.push_str(&format!(
            "  %{label}.go = icmp ult i64 %{label}.j, {out_cells}\n"
        ));
        ir.push_str(&format!(
            "  br i1 %{label}.go, label %{label}.body, label %{label}.end\n\n"
        ));

        // Map the surviving index back to where its axis starts.
        ir.push_str(&format!("{label}.body:\n"));
        ir.push_str(&format!("  %{label}.o = udiv i64 %{label}.j, {inner}\n"));
        ir.push_str(&format!("  %{label}.t = urem i64 %{label}.j, {inner}\n"));
        ir.push_str(&format!(
            "  %{label}.block = mul i64 %{label}.o, {}\n",
            extent * inner
        ));
        ir.push_str(&format!(
            "  %{label}.base = add i64 %{label}.block, %{label}.t\n"
        ));
        ir.push_str(&format!("  br label %{label}.inner\n\n"));

        // Inner loop: walk the axis, folding as it goes.
        ir.push_str(&format!("{label}.inner:\n"));
        ir.push_str(&format!(
            "  %{label}.m = phi i64 [ 0, %{label}.body ], [ %{label}.m.next, %{label}.step ]\n"
        ));
        ir.push_str(&format!(
            "  %{label}.acc = phi double [ {}, %{label}.body ], [ %{label}.acc.next, %{label}.step ]\n",
            Self::f64_literal(op.identity())
        ));
        ir.push_str(&format!(
            "  %{label}.more = icmp ult i64 %{label}.m, {extent}\n"
        ));
        ir.push_str(&format!(
            "  br i1 %{label}.more, label %{label}.step, label %{label}.tail\n\n"
        ));

        ir.push_str(&format!("{label}.step:\n"));
        ir.push_str(&format!(
            "  %{label}.off = mul i64 %{label}.m, {inner}\n"
        ));
        ir.push_str(&format!(
            "  %{label}.at = add i64 %{label}.base, %{label}.off\n"
        ));
        let element = self.emit_expr(
            operand,
            ir,
            bufs,
            &format!("%{label}.at"),
            counter,
            Mode::Scalar,
            &shape,
            &[],
        )?;
        match op {
            FoldOp::Sum => ir.push_str(&format!(
                "  %{label}.acc.next = fadd double %{label}.acc, {element}\n"
            )),
            FoldOp::Product => ir.push_str(&format!(
                "  %{label}.acc.next = fmul double %{label}.acc, {element}\n"
            )),
            FoldOp::Max | FoldOp::Min => {
                let pred_op = if matches!(op, FoldOp::Max) { "ogt" } else { "olt" };
                ir.push_str(&format!(
                    "  %{label}.win = fcmp {pred_op} double {element}, %{label}.acc\n"
                ));
                ir.push_str(&format!(
                    "  %{label}.acc.next = select i1 %{label}.win, double {element}, double %{label}.acc\n"
                ));
            }
        }
        ir.push_str(&format!("  %{label}.m.next = add i64 %{label}.m, 1\n"));
        ir.push_str(&format!("  br label %{label}.inner\n\n"));

        ir.push_str(&format!("{label}.tail:\n"));
        ir.push_str(&format!(
            "  %{label}.slot = getelementptr inbounds double, ptr {buffer}, i64 %{label}.j\n"
        ));
        ir.push_str(&format!(
            "  store double %{label}.acc, ptr %{label}.slot, align 8\n"
        ));
        ir.push_str(&format!("  %{label}.j.next = add i64 %{label}.j, 1\n"));
        ir.push_str(&format!("  br label %{label}.header\n\n"));

        ir.push_str(&format!("{label}.end:\n"));

        bufs.folds.insert(node as *const Expr as usize, buffer);
        Ok(format!("{label}.end"))
    }

    /// Emit one loop over `[lo, hi)`, stepping by the mode's lane count.
    /// Returns the label control flow lands on, so ranges can be chained.
    #[allow(clippy::too_many_arguments)]
    fn emit_range_loop(
        &self,
        ir: &mut String,
        label: &str,
        pred: &str,
        lo: &str,
        hi: &str,
        mode: Mode,
        src: &Expr,
        target_ptr: &str,
        bufs: &Buffers,
        counter: &mut usize,
        result_shape: &[usize],
    ) -> Result<String> {
        // A statically empty range needs no loop at all.
        if let (Ok(a), Ok(b)) = (lo.parse::<u64>(), hi.parse::<u64>()) {
            if a >= b {
                return Ok(pred.to_string());
            }
        }

        let header = format!("{label}.header");
        let body = format!("{label}.body");
        let end = format!("{label}.end");
        let idx = format!("%{label}.idx");
        let next = format!("%{label}.next");
        let cond = format!("%{label}.cond");
        let step = mode.lanes();

        ir.push_str(&format!("  br label %{header}\n\n"));
        ir.push_str(&format!("{header}:\n"));
        ir.push_str(&format!(
            "  {idx} = phi i64 [ {lo}, %{pred} ], [ {next}, %{body} ]\n"
        ));
        ir.push_str(&format!("  {cond} = icmp ult i64 {idx}, {hi}\n"));
        ir.push_str(&format!("  br i1 {cond}, label %{body}, label %{end}\n\n"));

        ir.push_str(&format!("{body}:\n"));
        let value = self.emit_expr(src, ir, bufs, &idx, counter, mode, result_shape, &[])?;
        let gep = Self::fresh(counter);
        ir.push_str(&format!(
            "  {gep} = getelementptr inbounds double, ptr {target_ptr}, i64 {idx}\n"
        ));
        ir.push_str(&format!(
            "  store {} {value}, ptr {gep}, align 8\n",
            mode.ty()
        ));
        ir.push_str(&format!("  {next} = add i64 {idx}, {step}\n"));
        ir.push_str(&format!("  br label %{header}\n\n"));

        ir.push_str(&format!("{end}:\n"));
        Ok(end)
    }

    /// Declared spaces that no flow ever writes: the kernel's inputs.
    fn source_spaces(&self, block: &ToposBlock) -> Vec<String> {
        let written: Vec<String> = block
            .statements
            .iter()
            .filter_map(|stmt| match stmt {
                Statement::Flow {
                    target: FlowTarget::Var(name),
                    ..
                } => Some(name.clone()),
                Statement::Flow {
                    target: FlowTarget::Equilibrium,
                    ..
                } => Some("OUTPUT".to_string()),
                _ => None,
            })
            .collect();
        block
            .statements
            .iter()
            .filter_map(|stmt| match stmt {
                Statement::SpaceDef(d) => Some(d.name.clone()),
                Statement::ExtBind(b) => Some(b.space.name.clone()),
                _ => None,
            })
            .filter(|name| !written.contains(name))
            .collect()
    }

    /// The shape this flow writes into.
    fn sweep_shape(&self, src: &Expr, target: &FlowTarget) -> Vec<usize> {
        let name = match target {
            FlowTarget::Var(n) => n.clone(),
            FlowTarget::Equilibrium => "OUTPUT".to_string(),
        };
        Self::expr_shape(src, &self.space_shapes)
            .or_else(|| self.space_shapes.get(&name).cloned())
            .unwrap_or_else(|| vec![self.elements])
    }

    /// How many cells this flow writes: the length of what it flows into,
    /// clamped by a bounded call.
    fn sweep_length(&self, src: &Expr, target: &FlowTarget, bufs: &Buffers) -> String {
        // A bounded call carries a runtime limit and stays scalar.
        if bufs.bound.starts_with('%') {
            return bufs.bound.clone();
        }
        let name = match target {
            FlowTarget::Var(n) => n.clone(),
            FlowTarget::Equilibrium => "OUTPUT".to_string(),
        };
        let shape = Self::expr_shape(src, &self.space_shapes)
            .or_else(|| self.space_shapes.get(&name).cloned());
        match shape {
            Some(s) => s.iter().product::<usize>().max(1).to_string(),
            None => self.elements.to_string(),
        }
    }

    /// Decide how to split one flow's sweep.
    fn plan_sweep(&self, src: &Expr, bound: &str) -> Result<Sweep> {
        // Vector loads read a fixed window, so the length has to be known here.
        let Ok(total) = bound.parse::<u64>() else {
            return Ok(Sweep::AllScalar);
        };
        if !self.simd {
            return Ok(Sweep::AllScalar);
        }

        let width = VECTOR_WIDTH as u64;
        let reach = self.max_shift_stride(src)?;

        // The body may touch [idx - reach, idx + reach + width - 1].
        let start = reach;
        let Some(limit) = total.checked_sub(reach + width - 1) else {
            return Ok(Sweep::AllScalar);
        };
        if limit <= start || limit - start < width {
            return Ok(Sweep::AllScalar);
        }
        let chunks = (limit - start) / width;
        Ok(Sweep::Split {
            vector_start: start,
            vector_end: start + chunks * width,
            width: width as usize,
            total,
        })
    }

    /// The furthest a shift in this expression reaches from the current cell.
    fn max_shift_stride(&self, expr: &Expr) -> Result<u64> {
        Ok(match expr {
            Expr::Var(_) | Expr::Number(_) => 0,
            Expr::AuditTrace(inner) => self.max_shift_stride(inner)?,
            Expr::BinaryOp { lhs, rhs, .. } => self
                .max_shift_stride(lhs)?
                .max(self.max_shift_stride(rhs)?),
            // A fold is precomputed into its own buffer before the sweep, so
            // it contributes no reach to the sweep itself.
            Expr::Reduce { .. } => 0,
            Expr::Lift { operand, .. } => self.max_shift_stride(operand)?,
            Expr::Shift { axis, operand, .. } => {
                let inner = self.max_shift_stride(operand)?;
                let Some(name) = Self::place_name(operand) else {
                    return Ok(inner);
                };
                let shape = self.shape_for(&name);
                let (stride, extent) = self.axis_geometry(&shape, *axis)?;
                if extent <= 1 {
                    inner
                } else {
                    inner + stride as u64
                }
            }
        })
    }

    fn axis_geometry(&self, shape: &[usize], axis: Option<usize>) -> Result<(usize, usize)> {
        let line = self.current_line.get();
        crate::ast::axis_geometry(shape, axis).ok_or_else(|| HarmonyDisruption::LoweringErr {
            line,
            detail: format!(
                "axis {} is out of range for a rank-{} space (shape {shape:?})",
                axis.map(|a| a.to_string())
                    .unwrap_or_else(|| "default".to_string()),
                shape.len()
            ),
        })
    }

    /// A shape with the enclosing lifts' unit axes inserted. The lifts arrive
    /// outermost-first, so they are applied in reverse to land where written.
    fn lifted_shape(shape: &[usize], lifts: &[usize]) -> Vec<usize> {
        let mut view = shape.to_vec();
        for &axis in lifts.iter().rev() {
            match crate::ast::shape_with_unit_axis(&view, axis) {
                Some(next) => view = next,
                None => return view,
            }
        }
        view
    }

    /// Translate an index into `result_shape` into an index into `source`.
    ///
    /// They agree except where `source` has a length-1 axis stretched against a
    /// longer one; those axes contribute nothing, which is exactly what makes a
    /// value repeat along them. Identical shapes need no arithmetic at all.
    fn emit_index_map(
        &self,
        ir: &mut String,
        idx: &str,
        source: &[usize],
        result_shape: &[usize],
        counter: &mut usize,
    ) -> Result<String> {
        if source == result_shape || result_shape.is_empty() || source.len() != result_shape.len() {
            return Ok(idx.to_string());
        }

        let src_strides = crate::ast::strides_of(source);
        let res_strides = crate::ast::strides_of(result_shape);

        let mut terms: Vec<String> = Vec::new();
        for axis in 0..result_shape.len() {
            // A stretched axis holds the same value for every position along it.
            if source[axis] <= 1 {
                continue;
            }
            let coord = Self::fresh(counter);
            let scaled = Self::fresh(counter);
            if res_strides[axis] == 1 {
                ir.push_str(&format!(
                    "  {coord} = urem i64 {idx}, {}\n",
                    result_shape[axis]
                ));
            } else {
                let div = Self::fresh(counter);
                ir.push_str(&format!("  {div} = udiv i64 {idx}, {}\n", res_strides[axis]));
                ir.push_str(&format!(
                    "  {coord} = urem i64 {div}, {}\n",
                    result_shape[axis]
                ));
            }
            ir.push_str(&format!(
                "  {scaled} = mul i64 {coord}, {}\n",
                src_strides[axis]
            ));
            terms.push(scaled);
        }

        if terms.is_empty() {
            return Ok("0".to_string());
        }
        let mut acc = terms[0].clone();
        for term in &terms[1..] {
            let next = Self::fresh(counter);
            ir.push_str(&format!("  {next} = add i64 {acc}, {term}\n"));
            acc = next;
        }
        Ok(acc)
    }

    /// Whether any read in this expression is stretched, which rules out the
    /// contiguous vector loads the sweep would otherwise use.
    fn needs_broadcast(&self, expr: &Expr, result_shape: &[usize], lifts: &[usize]) -> bool {
        match expr {
            Expr::Number(_) => false,
            Expr::Var(name) => Self::lifted_shape(&self.shape_for(name), lifts) != result_shape,
            Expr::AuditTrace(inner) | Expr::Shift { operand: inner, .. } => {
                self.needs_broadcast(inner, result_shape, lifts)
            }
            Expr::Lift { axis, operand } => {
                let mut nested = lifts.to_vec();
                nested.push(*axis);
                self.needs_broadcast(operand, result_shape, &nested)
            }
            // A fold is read from its own buffer, which already has the result shape.
            Expr::Reduce { .. } => false,
            Expr::BinaryOp { lhs, rhs, .. } => {
                self.needs_broadcast(lhs, result_shape, lifts)
                    || self.needs_broadcast(rhs, result_shape, lifts)
            }
        }
    }

    /// The shape a space is traversed with; falls back to the grid when a
    /// declared shape does not cover the sweep.
    fn shape_for(&self, name: &str) -> Vec<usize> {
        // Shapes stopped being uniform when folds arrived, so a space's own
        // shape is authoritative even when its length differs from the grid's.
        self.space_shapes
            .get(name)
            .cloned()
            .or_else(|| Self::primary_shape(&self.space_shapes))
            .unwrap_or_else(|| vec![self.elements])
    }

    // ----------------------------------------------------------- expressions

    #[allow(clippy::too_many_arguments)]
    fn emit_expr(
        &self,
        expr: &Expr,
        ir: &mut String,
        bufs: &Buffers,
        idx: &str,
        counter: &mut usize,
        mode: Mode,
        result_shape: &[usize],
        lifts: &[usize],
    ) -> Result<String> {
        let ty = mode.ty();
        match expr {
            Expr::Number(value) => Ok(mode.splat(*value)),

            Expr::Var(name) if Self::is_tau(name) => Ok(mode.splat(self.tau)),

            Expr::Var(name) => {
                let ptr = self.lookup(bufs, name)?;
                let view = Self::lifted_shape(&self.shape_for(name), lifts);
                let read_at = self.emit_index_map(ir, idx, &view, result_shape, counter)?;
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds double, ptr {ptr}, i64 {read_at}\n"
                ));
                ir.push_str(&format!("  {val} = load {ty}, ptr {gep}, align 8\n"));
                Ok(val)
            }

            Expr::AuditTrace(inner) => {
                self.emit_expr(inner, ir, bufs, idx, counter, mode, result_shape, lifts)
            }

            // A lift stores nothing. It records that the operand is viewed with
            // an extra length-1 axis, which the read below folds into its index
            // mapping.
            Expr::Lift { axis, operand } => {
                let mut nested = lifts.to_vec();
                nested.push(*axis);
                self.emit_expr(operand, ir, bufs, idx, counter, mode, result_shape, &nested)
            }

            Expr::Shift { dir, axis, operand } => {
                self.emit_shift(operand, *dir, *axis, ir, bufs, idx, counter, mode)
            }

            // The fold ran before this sweep started; read its result.
            Expr::Reduce { .. } => {
                let key = expr as *const Expr as usize;
                let ptr = bufs
                    .folds
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "a fold was not precomputed before the sweep that reads it"
                            .to_string(),
                    })?;
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds double, ptr {ptr}, i64 {idx}\n"
                ));
                ir.push_str(&format!("  {val} = load {ty}, ptr {gep}, align 8\n"));
                Ok(val)
            }

            Expr::BinaryOp { op, lhs, rhs } => {
                let l = self.emit_expr(lhs, ir, bufs, idx, counter, mode, result_shape, lifts)?;
                let r = self.emit_expr(rhs, ir, bufs, idx, counter, mode, result_shape, lifts)?;
                let out = Self::fresh(counter);

                match op {
                    BinaryOpKind::Add => {
                        ir.push_str(&format!("  {out} = fadd {ty} {l}, {r}\n"))
                    }
                    BinaryOpKind::Sub => {
                        ir.push_str(&format!("  {out} = fsub {ty} {l}, {r}\n"))
                    }
                    BinaryOpKind::Mul => {
                        ir.push_str(&format!("  {out} = fmul {ty} {l}, {r}\n"))
                    }
                    BinaryOpKind::Div => {
                        ir.push_str(&format!("  {out} = fdiv {ty} {l}, {r}\n"))
                    }
                    BinaryOpKind::Pow => ir.push_str(&format!(
                        "  {out} = call {ty} {}({ty} {l}, {ty} {r})\n",
                        mode.pow_intrinsic()
                    )),
                    // Threshold comparison masks the grid: the left value passes
                    // through where the predicate holds, elsewhere the cell is 0.
                    BinaryOpKind::Gt
                    | BinaryOpKind::Lt
                    | BinaryOpKind::Gte
                    | BinaryOpKind::Lte
                    | BinaryOpKind::Eq => {
                        let predicate = match op {
                            BinaryOpKind::Gt => "ogt",
                            BinaryOpKind::Lt => "olt",
                            BinaryOpKind::Gte => "oge",
                            BinaryOpKind::Lte => "ole",
                            _ => "oeq",
                        };
                        let flag = Self::fresh(counter);
                        ir.push_str(&format!("  {flag} = fcmp {predicate} {ty} {l}, {r}\n"));
                        ir.push_str(&format!(
                            "  {out} = select {} {flag}, {ty} {l}, {ty} {}\n",
                            mode.bool_ty(),
                            mode.zero()
                        ));
                    }
                }

                Ok(out)
            }
        }
    }

    /// ▷X reads the preceding cell along an axis and ▽X the following one, zero
    /// at that axis's boundary. On a 1024x1024 grid a bare ▷ stops at the end of
    /// each row instead of wrapping into the previous one.
    ///
    /// In vector mode the neighbours are one contiguous load at a shifted base;
    /// only the boundary test is per lane. The sweep planner guarantees that
    /// load stays inside the buffer.
    #[allow(clippy::too_many_arguments)]
    fn emit_shift(
        &self,
        operand: &Expr,
        dir: ShiftDir,
        axis: Option<usize>,
        ir: &mut String,
        bufs: &Buffers,
        idx: &str,
        counter: &mut usize,
        mode: Mode,
    ) -> Result<String> {
        let name = Self::place_name(operand).ok_or_else(|| HarmonyDisruption::LoweringErr {
            line: self.current_line.get(),
            detail: format!(
                "{dir} is a neighbourhood shift over a declared space, so it cannot be applied to a computed value. Flow the sub-expression into its own space first."
            ),
        })?;
        let ptr = self.lookup(bufs, &name)?;
        let shape = self.shape_for(&name);
        let (stride, extent) = self.axis_geometry(&shape, axis)?;

        let axis_label = match axis {
            Some(a) => format!("axis {a}"),
            None => "last axis".to_string(),
        };

        // An axis of length 1 puts every cell on both boundaries at once.
        if extent <= 1 {
            ir.push_str(&format!(
                "  ; {dir}{name} along {axis_label}: extent {extent}, every cell is a boundary\n"
            ));
            return Ok(mode.zero());
        }

        ir.push_str(&format!(
            "  ; {dir}{name} along {axis_label} (stride {stride}, extent {extent})\n"
        ));

        let ty = mode.ty();
        let int_ty = mode.int_ty();

        // Lane indices: idx for scalar, idx + <0,1,..,W-1> for vector.
        let lane_idx = match mode {
            Mode::Scalar => idx.to_string(),
            Mode::Vector(w) => {
                let seed = Self::fresh(counter);
                let splat = Self::fresh(counter);
                let lanes = Self::fresh(counter);
                let offsets: Vec<String> = (0..w).map(|l| format!("i64 {l}")).collect();
                ir.push_str(&format!(
                    "  {seed} = insertelement {int_ty} poison, i64 {idx}, i64 0\n"
                ));
                ir.push_str(&format!(
                    "  {splat} = shufflevector {int_ty} {seed}, {int_ty} poison, <{w} x i32> zeroinitializer\n"
                ));
                ir.push_str(&format!(
                    "  {lanes} = add {int_ty} {splat}, <{}>\n",
                    offsets.join(", ")
                ));
                lanes
            }
        };

        // Position along the axis: (i / stride) % extent, per lane.
        let pos = if stride == 1 {
            let p = Self::fresh(counter);
            ir.push_str(&format!(
                "  {p} = urem {int_ty} {lane_idx}, {}\n",
                mode.int_splat(extent as u64)
            ));
            p
        } else {
            let q = Self::fresh(counter);
            let p = Self::fresh(counter);
            ir.push_str(&format!(
                "  {q} = udiv {int_ty} {lane_idx}, {}\n",
                mode.int_splat(stride as u64)
            ));
            ir.push_str(&format!(
                "  {p} = urem {int_ty} {q}, {}\n",
                mode.int_splat(extent as u64)
            ));
            p
        };

        let at_edge = Self::fresh(counter);
        let edge_value = match dir {
            ShiftDir::Positive => 0u64,
            ShiftDir::Negative => (extent - 1) as u64,
        };
        ir.push_str(&format!(
            "  {at_edge} = icmp eq {int_ty} {pos}, {}\n",
            mode.int_splat(edge_value)
        ));

        // Base of the neighbour window, as a scalar offset from idx.
        let base = Self::fresh(counter);
        match dir {
            ShiftDir::Positive => {
                ir.push_str(&format!("  {base} = sub i64 {idx}, {stride}\n"));
            }
            ShiftDir::Negative => {
                ir.push_str(&format!("  {base} = add i64 {idx}, {stride}\n"));
            }
        }

        let safe = match mode {
            // The planner keeps every vector window inside the buffer.
            Mode::Vector(_) => base.clone(),
            // A scalar step can sit on the very edge, and a bounded call may stop
            // short of the declared grid, so clamp before touching memory.
            Mode::Scalar => {
                let past_end = Self::fresh(counter);
                let skip = Self::fresh(counter);
                let clamped = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {past_end} = icmp uge i64 {base}, {}\n",
                    bufs.bound
                ));
                ir.push_str(&format!("  {skip} = or i1 {at_edge}, {past_end}\n"));
                ir.push_str(&format!(
                    "  {clamped} = select i1 {skip}, i64 0, i64 {base}\n"
                ));
                // Reuse the widened predicate for the value select below.
                ir.push_str("  ; boundary or past the sweep -> 0\n");
                return {
                    let gep = Self::fresh(counter);
                    let val = Self::fresh(counter);
                    let out = Self::fresh(counter);
                    ir.push_str(&format!(
                        "  {gep} = getelementptr inbounds double, ptr {ptr}, i64 {clamped}\n"
                    ));
                    ir.push_str(&format!("  {val} = load {ty}, ptr {gep}, align 8\n"));
                    ir.push_str(&format!(
                        "  {out} = select i1 {skip}, {ty} {}, {ty} {val}\n",
                        mode.zero()
                    ));
                    Ok(out)
                };
            }
        };

        let gep = Self::fresh(counter);
        let val = Self::fresh(counter);
        let out = Self::fresh(counter);
        ir.push_str(&format!(
            "  {gep} = getelementptr inbounds double, ptr {ptr}, i64 {safe}\n"
        ));
        ir.push_str(&format!("  {val} = load {ty}, ptr {gep}, align 8\n"));
        ir.push_str(&format!(
            "  {out} = select {} {at_edge}, {ty} {}, {ty} {val}\n",
            mode.bool_ty(),
            mode.zero()
        ));

        Ok(out)
    }

    fn place_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Var(name) => Some(name.clone()),
            Expr::AuditTrace(inner) => Self::place_name(inner),
            _ => None,
        }
    }

    fn lookup(&self, bufs: &Buffers, name: &str) -> Result<String> {
        bufs.map
            .get(name)
            .cloned()
            .ok_or_else(|| HarmonyDisruption::SpaceErr {
                space_name: name.to_string(),
                line: self.current_line.get(),
            })
    }

    fn is_tau(name: &str) -> bool {
        name == "𝜏" || name == "τ"
    }

    fn fresh(counter: &mut usize) -> String {
        *counter += 1;
        format!("%v{counter}")
    }

    /// Exact bit pattern, so literals survive round-tripping and NaN/inf stay legal.
    pub(crate) fn f64_literal(value: f64) -> String {
        format!("0x{:016X}", value.to_bits())
    }

    // -------------------------------------------------------------- metadata

    fn generate_metadata_json(&self) -> String {
        let spaces: Vec<String> = self
            .space_shapes
            .iter()
            .map(|(name, shape)| format!("{{\"name\":\"{name}\",\"shape\":{shape:?}}}"))
            .collect();
        let bindings: Vec<String> = self
            .ext_bindings
            .iter()
            .map(|(name, addr)| format!("{{\"name\":\"{name}\",\"address\":{addr}}}"))
            .collect();
        let contract = match &self.contract_json {
            Some(json) => format!(",\"contract\":{json}"),
            None => String::new(),
        };
        format!(
            "{{\"elements\":{},\"spaces\":[{}],\"bindings\":[{}]{}}}",
            self.elements,
            spaces.join(","),
            bindings.join(","),
            contract
        )
    }

    /// Escape for an LLVM c"..." literal and report the byte length it denotes.
    fn c_string(text: &str) -> (String, usize) {
        let mut escaped = String::new();
        let mut len = 0usize;
        for byte in text.as_bytes() {
            match byte {
                b'"' => escaped.push_str("\\22"),
                b'\\' => escaped.push_str("\\5C"),
                0x20..=0x7E => escaped.push(*byte as char),
                other => escaped.push_str(&format!("\\{other:02X}")),
            }
            len += 1;
        }
        escaped.push_str("\\00");
        (escaped, len + 1)
    }

    // --------------------------------------------------------------- backend

    /// Compile LLVM IR to a native shared library (.so) using clang
    pub fn compile_to_so(&self, ir_content: &str, output_path: &str) -> std::io::Result<()> {
        let temp_ll = format!("{}.ll", output_path);
        fs::write(&temp_ll, ir_content)?;

        // The module carries no target triple on purpose, so clang supplies the
        // host's. That substitution is what -Woverride-module reports; it is the
        // intended behaviour here, not a problem to surface on every build.
        let args = [
            "-shared",
            "-fPIC",
            "-O3",
            "-Wno-override-module",
            &temp_ll,
            "-o",
            output_path,
        ];
        let status = Command::new("clang-22")
            .args(args)
            .status()
            .or_else(|_| Command::new("clang").args(args).status())?;

        if status.success() {
            let _ = fs::remove_file(temp_ll);
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "clang failed on {temp_ll}; the IR was kept for inspection"
            )))
        }
    }

    fn sanitize_ident(&self, ident: &str) -> String {
        let mut sanitized = String::new();
        for ch in ident.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                sanitized.push(ch);
            } else {
                sanitized.push_str(&format!("_u{:04X}", ch as u32));
            }
        }
        if let Some(first) = sanitized.chars().next() {
            if !first.is_ascii_alphabetic() && first != '_' {
                sanitized.insert(0, 's');
            }
        }
        sanitized
    }
}
