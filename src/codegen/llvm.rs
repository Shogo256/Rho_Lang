// SPDX-License-Identifier: Apache-2.0
use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use crate::numeric::Precision;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::process::Command;

/// The runtime every kernel is linked with: the thread pool a sweep is split
/// across. Compiled next to the IR by the same clang call.
const RUNTIME_C: &str = include_str!("rho_rt.c");
/// The most parts one sweep is split into; the runtime caps its threads here.
const MAX_PARTS: usize = 256;
/// Stands in an entrypoint for the length of the call's table of pointers,
/// which is known once the whole module has been emitted.
const CTX_SLOTS: &str = "@@rho_ctx_slots@@";

/// Spaces larger than this are heap-allocated instead of living on the stack,
/// so a 1024x1024 grid cannot blow the 8 MB default stack.
const STACK_LIMIT_BYTES: usize = 64 * 1024;

/// Bytes per vector step: one AVX register, four doubles or eight singles;
/// clang widens further when the target allows it.
const VECTOR_BYTES: usize = 32;

/// Whether a chunk of a flow is lowered one cell at a time or a vector at a time.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Scalar(Precision),
    Vector(usize, Precision),
}

impl Mode {
    fn lanes(self) -> usize {
        match self {
            Mode::Scalar(_) => 1,
            Mode::Vector(w, _) => w,
        }
    }

    fn precision(self) -> Precision {
        match self {
            Mode::Scalar(p) | Mode::Vector(_, p) => p,
        }
    }

    fn ty(self) -> String {
        let element = self.precision().llvm_type();
        match self {
            Mode::Scalar(_) => element.to_string(),
            Mode::Vector(w, _) => format!("<{w} x {element}>"),
        }
    }

    fn int_ty(self) -> String {
        match self {
            Mode::Scalar(_) => "i64".to_string(),
            Mode::Vector(w, _) => format!("<{w} x i64>"),
        }
    }

    fn bool_ty(self) -> String {
        match self {
            Mode::Scalar(_) => "i1".to_string(),
            Mode::Vector(w, _) => format!("<{w} x i1>"),
        }
    }

    fn zero(self) -> String {
        match self {
            Mode::Scalar(_) => "0.0".to_string(),
            Mode::Vector(..) => "zeroinitializer".to_string(),
        }
    }

    /// A floating constant broadcast across every lane.
    fn splat(self, value: f64) -> String {
        let lit = LlvmCodeGen::float_literal(value, self.precision());
        let element = self.precision().llvm_type();
        match self {
            Mode::Scalar(_) => lit,
            Mode::Vector(w, _) => {
                let lanes: Vec<String> = (0..w).map(|_| format!("{element} {lit}")).collect();
                format!("<{}>", lanes.join(", "))
            }
        }
    }

    fn int_splat(self, value: u64) -> String {
        match self {
            Mode::Scalar(_) => value.to_string(),
            Mode::Vector(w, _) => {
                let lanes: Vec<String> = (0..w).map(|_| format!("i64 {value}")).collect();
                format!("<{}>", lanes.join(", "))
            }
        }
    }

    fn unary_intrinsic(self, name: &str) -> String {
        let suffix = self.precision().intrinsic_suffix();
        match self {
            Mode::Scalar(_) => format!("@llvm.{name}.{suffix}"),
            Mode::Vector(w, _) => format!("@llvm.{name}.v{w}{suffix}"),
        }
    }

    fn pow_intrinsic(self) -> String {
        self.unary_intrinsic("pow")
    }
}

/// How one flow's sweep is split between scalar and vector loops.
/// What a `⇒` round's sweep measures as it writes: the grid being replaced,
/// and where the largest move so far is kept — one scalar, one vector of
/// lanes — for the part to reduce when its range is done.
struct Measure {
    cur: String,
    acc: String,
    acc_vec: String,
}

enum Sweep {
    AllScalar,
    Split {
        vector_start: u64,
        vector_end: u64,
        width: usize,
        total: u64,
        /// The vector body starts each chunk at a multiple of the width,
        /// which a stretched read needs (see `stretched_contiguously`).
        aligned: bool,
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

/// What a fold's loops need to know about the axis they walk.
/// The coordinates of one cell, known from the loops that reach it.
struct CoordHint {
    idx: String,
    shape: Vec<usize>,
    axes: Vec<String>,
}

/// One `⍳` in a position that is a whole-number combination of coordinates.
struct AffineTerm {
    coefficient: i64,
    axis: Option<usize>,
    operand: Expr,
    lifts: Vec<usize>,
}

/// A `⌷` position of the form `c0 + Σ ci · ⍳`, which is an integer the
/// cell's coordinates fix.
struct AffinePosition {
    constant: i64,
    terms: Vec<AffineTerm>,
}

struct FoldGeometry<'a> {
    op: FoldOp,
    running: bool,
    extent: usize,
    inner: usize,
    /// The axis being walked, in `shape`.
    axis: usize,
    shape: &'a [usize],
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
    /// Whether sweeps and folds are outlined into parts and run on the pool.
    /// A bounded call has a length known only at run time, so it stays
    /// inline and on one thread.
    parallel: bool,
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
    /// How wide the numbers are. Narrower numbers halve the memory traffic
    /// these kernels are bound by, and widen the doubt a proof has to carry.
    pub precision: Precision,
    /// Source line of each statement, copied from the block being lowered.
    statement_lines: Vec<usize>,
    /// The most sweeps a `⇒` may take, from `--max-iter`. A program that
    /// iterates cannot be lowered without one: the cap is what makes every
    /// kernel terminate, and it is part of what the kernel computes.
    max_sweeps: Option<usize>,
    /// Whether the block being lowered has a `⇒` at all.
    iterates: bool,
    /// Spaces some flow writes. Every other space is read from memory the
    /// caller owns, which is what decides whether an entrypoint may run.
    written_spaces: BTreeSet<String>,
    /// Number of cells swept by every flow loop.
    elements: usize,
    /// Source line of the flow being lowered, so a lowering failure can point
    /// at the statement the reader wrote.
    current_line: std::cell::Cell<usize>,
    /// While a fold walks a line, the coordinates of the cell being read:
    /// the index symbol they belong to, the shape they index, and one
    /// symbol per axis. A read at that index takes its coordinates from
    /// here rather than dividing the index up again, which, inside the
    /// innermost loop of a fold, was most of the work.
    coords: RefCell<Option<CoordHint>>,
    /// Threads a sweep is split across; 0 lets the runtime take one per CPU.
    /// RHO_THREADS in the environment overrides it at run time.
    pub threads: usize,
    /// Compile for the machine at hand (-march=native). Off, the kernel runs
    /// on any x86-64 but with the baseline vector width.
    pub native: bool,
    /// Cells below which a sweep is not worth splitting and runs on the
    /// calling thread. RHO_GRAIN in the environment overrides it.
    pub grain: usize,
    /// The functions a sweep, a fold or a comparison was outlined into: each
    /// takes a range [lo, hi) and a part number, and the runtime hands every
    /// thread its own. Appended after the entrypoints.
    outlined: RefCell<String>,
    /// Slot in the call's table of every pointer an outlined function may
    /// need, by the symbol the function that made it knows it as. The
    /// entrypoints store the spaces there, the body stores what it
    /// allocates, and a part loads what it reads. The table is on the
    /// entrypoint's stack and handed down as `%rho.ctx`, so two calls at
    /// once each have their own: a kernel that keeps no scratch is
    /// reentrant.
    slots: RefCell<BTreeMap<String, usize>>,
    slot_count: Cell<usize>,
    /// Names for the instructions that address the table.
    ctx_tmp: Cell<usize>,
    outlined_count: Cell<usize>,
    /// The globals that hold scratch kept between calls, and their count.
    keeps: RefCell<String>,
    keep_count: Cell<usize>,
    /// Intermediates fused into their reader (see `fuse`), each with the
    /// space its reader writes: their flows run only when a caller passed a
    /// buffer for the space.
    fused: BTreeMap<String, String>,
    /// The spaces an entrypoint keeps on its stack, by label, with their
    /// bytes: what `rho_kernel_zeroize` has to overwrite.
    stack_spaces: RefCell<BTreeMap<String, usize>>,
    /// The scratch kept between calls, by label, with its bytes.
    kept_spaces: RefCell<BTreeMap<String, usize>>,
    /// What became of each flow of the shared body, for `explain`; recorded
    /// while that body is emitted (the bounded entrypoint repeats the flows).
    notes: RefCell<Vec<FlowNote>>,
    recording: Cell<bool>,
    /// The `⇒` whose round is being emitted.
    in_loop: RefCell<Option<String>>,
}

/// What the compiler made of one flow, for `explain`.
struct FlowNote {
    line: usize,
    target: String,
    cells: String,
    /// How the sweep runs: vector lanes, or one cell at a time and why.
    plan: String,
    /// Folds and scans, each computed as a pass of its own before the sweep.
    folds: usize,
    /// For a flow fused into its reader, the space the reader writes; such
    /// a flow runs only when the caller passes its space.
    fused_into: Option<String>,
    /// The `⇒` whose every round runs this sweep.
    round_of: Option<String>,
}

/// Stack wiped past the spaces themselves: the spills and the small frames
/// of the body and the parts, which can hold a cell or a lane of one.
const ZEROIZE_MARGIN: usize = 16 * 1024;

impl LlvmCodeGen {
    pub fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
            space_shapes: BTreeMap::new(),
            ext_bindings: BTreeMap::new(),
            tau: 0.0,
            binding_overrides: BTreeMap::new(),
            simd: true,
            precision: Precision::F64,
            statement_lines: Vec::new(),
            max_sweeps: None,
            iterates: false,
            written_spaces: BTreeSet::new(),
            elements: 0,
            current_line: std::cell::Cell::new(0),
            coords: RefCell::new(None),
            threads: 0,
            native: true,
            grain: 16384,
            outlined: RefCell::new(String::new()),
            slots: RefCell::new(BTreeMap::new()),
            slot_count: Cell::new(0),
            ctx_tmp: Cell::new(0),
            outlined_count: Cell::new(0),
            keeps: RefCell::new(String::new()),
            keep_count: Cell::new(0),
            fused: BTreeMap::new(),
            stack_spaces: RefCell::new(BTreeMap::new()),
            kept_spaces: RefCell::new(BTreeMap::new()),
            notes: RefCell::new(Vec::new()),
            recording: Cell::new(false),
            in_loop: RefCell::new(None),
        }
    }

    /// Whether the program has a `⇒` (known once IR has been generated).
    pub fn iterates(&self) -> bool {
        self.iterates
    }

    /// Split every sweep across this many threads; 0 means one per CPU.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.min(MAX_PARTS);
        self
    }

    /// Compile for any x86-64 rather than for this machine.
    pub fn portable(mut self) -> Self {
        self.native = false;
        self
    }

    /// Split even a sweep this short across the threads. The default, 16384
    /// cells, is where splitting started to pay when measured (at 4096 it
    /// cost more than it saved); difftest lowers it to 1 so that every
    /// sweep is split.
    pub fn with_grain(mut self, cells: usize) -> Self {
        self.grain = cells.max(1);
        self
    }

    /// Bind the threshold symbol 𝜏 to a concrete value (default 0.0).
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Record what the solver proved, so it ships with the kernel.
    /// Compute at the given width.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// Turn vector lowering off and emit only scalar loops.
    /// Cap every `⇒` at this many sweeps.
    pub fn with_max_sweeps(mut self, sweeps: usize) -> Self {
        self.max_sweeps = Some(sweeps);
        self
    }

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

    /// What `generate_llvm_ir` made of the program, as a table: each sweep a
    /// call runs, vector or one cell at a time and why, the folds computed as
    /// passes of their own, the intermediates fused into it, what a `⇒` runs
    /// every round, and the memory `rho_kernel_zeroize` overwrites. Empty
    /// before the IR is generated.
    pub fn explain(&self) -> String {
        let notes = self.notes.borrow();
        let mut out = String::new();
        if notes.is_empty() {
            return out;
        }
        // A space fused into a space fused into another is carried by the last.
        let carrier = |name: &str| -> String {
            let mut at = name.to_string();
            for _ in 0..=self.fused.len() {
                match self.fused.get(&at) {
                    Some(next) => at = next.clone(),
                    None => break,
                }
            }
            at
        };
        let plural = |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
        // Each fused flow belongs to the first sweep after it that writes the
        // space carrying it (a space may be written more than once).
        let shown: Vec<usize> = (0..notes.len()).filter(|&i| notes[i].fused_into.is_none()).collect();
        let mut carried: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
        for (p, m) in notes.iter().enumerate().filter(|(_, m)| m.fused_into.is_some()) {
            let home = carrier(&m.target);
            if let Some(&r) = shown
                .iter()
                .find(|&&r| r > p && notes[r].target == home && notes[r].round_of == m.round_of)
            {
                carried.entry(r).or_default().push(m.target.as_str());
            }
        }
        let width = shown.iter().map(|&i| notes[i].target.chars().count()).max().unwrap_or(0).max(6);
        out.push_str(&format!("  sweep  line  {:<width$}  {:>7}  plan\n", "target", "cells"));
        let (mut once, mut once_folds) = (0usize, 0usize);
        let mut rounds: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for (i, &at) in shown.iter().enumerate() {
            let note = &notes[at];
            let mut plan = note.plan.clone();
            if note.folds > 0 {
                plan.push_str(&format!("; {} first", plural(note.folds, "fold pass", "fold passes")));
            }
            if let Some(l) = &note.round_of {
                plan.push_str(&format!("; every round of ⇒ {l}"));
            }
            out.push_str(&format!(
                "  {:>5}  {:>4}  {:<width$}  {:>7}  {plan}\n",
                i + 1,
                note.line,
                note.target,
                note.cells
            ));
            if let Some(carried) = carried.get(&at) {
                let list = if carried.len() > 6 {
                    format!("{} … ({} in all)", carried[..5].join(", "), carried.len())
                } else {
                    carried.join(", ")
                };
                out.push_str(&format!("  {:>5}  {:>4}  {:<width$}  {:>7}  ↳ fused into it: {list}\n", "", "", "", ""));
            }
            match &note.round_of {
                Some(l) => {
                    let r = rounds.entry(l.clone()).or_insert((0, 0));
                    r.0 += 1;
                    r.1 += note.folds;
                }
                None => {
                    once += 1;
                    once_folds += note.folds;
                }
            }
        }
        out.push_str(&format!(
            "  a call: {}, {}\n",
            plural(once, "sweep", "sweeps"),
            plural(once_folds, "fold pass", "fold passes")
        ));
        for (l, (sweeps, folds)) in &rounds {
            out.push_str(&format!(
                "  ⇒ {l}: {} and {} a round, at most {} rounds\n",
                plural(*sweeps, "sweep", "sweeps"),
                plural(*folds, "fold pass", "fold passes"),
                self.max_sweeps.unwrap_or(0)
            ));
        }
        let fused = notes.iter().filter(|n| n.fused_into.is_some()).count();
        if fused > 0 {
            out.push_str(&format!(
                "  fused: {}, each computed inside the sweep that reads it; a fused space's own sweep runs only when the caller passes the space\n",
                plural(fused, "intermediate", "intermediates")
            ));
        }
        let stack: usize = self.stack_spaces.borrow().values().sum::<usize>() + ZEROIZE_MARGIN;
        let kept: usize = self.kept_spaces.borrow().values().sum();
        out.push_str(&format!(
            "  rho_kernel_zeroize overwrites {} KB of stack and {} KB kept between calls\n",
            stack.div_ceil(1024),
            kept.div_ceil(1024)
        ));
        out
    }

    /// Generate complete LLVM IR (.ll) from a ToposBlock AST
    pub fn generate_llvm_ir(&mut self, block: &ToposBlock) -> Result<String> {
        // An intermediate one flow reads cell for cell is computed inside
        // that flow's sweep; its own sweep runs only for a caller who asked
        // for the space. The interpreter and the analysis see the program
        // as written.
        let (fused_block, fused) = super::fuse::fuse(block);
        self.fused = fused;
        let block = &fused_block;
        self.statement_lines = block.lines.clone();
        self.written_spaces = Self::written_spaces(block);
        self.iterates = block
            .statements
            .iter()
            .any(|s| matches!(s, Statement::Iterate { .. }));
        self.collect_shapes(block);
        for (name, addr) in &self.binding_overrides {
            self.ext_bindings.insert(name.clone(), *addr);
            self.space_shapes
                .entry(name.clone())
                .or_insert_with(|| vec![4]);
        }
        self.elements = self.resolve_element_count();
        // Every space has a slot the entrypoints fill, then the call's
        // partial results and its record of which fused spaces were passed;
        // the body's own buffers take the slots after them.
        self.outlined.borrow_mut().clear();
        self.slots.borrow_mut().clear();
        self.slot_count.set(self.space_shapes.len() + 2);
        self.ctx_tmp.set(0);
        self.outlined_count.set(0);

        let mut ir = String::new();

        ir.push_str(&format!("; ModuleID = '{}'\n", self.module_name));
        ir.push_str("source_filename = \"rho_kernel.rho\"\n");
        // No target triple or datalayout: clang fills in the host's. Hardcoding
        // x86_64-unknown-linux-gnu made every build warn about an overridden
        // triple and made the compiler unusable off x86-64 Linux.
        ir.push('\n');

        let elem = self.precision.llvm_type();
        let suffix = self.precision.intrinsic_suffix();
        ir.push_str(&format!(
            "declare {elem} @llvm.pow.{suffix}({elem}, {elem})\n"
        ));
        if self.simd {
            ir.push_str(&format!(
                "declare <{w} x {elem}> @llvm.pow.v{w}{suffix}(<{w} x {elem}>, <{w} x {elem}>)\n",
                w = self.lanes()
            ));
        }
        // The greater and the lesser are IEEE 754-2019 maximum and minimum:
        // a NaN propagates and -0 orders below +0. Both are fully specified,
        // which a compare-and-select is not once the optimiser treats it as
        // a NaN-free minimum: at -O2 that folding turned `NaN < 0 ? NaN : 0`
        // into NaN. An intrinsic cannot be re-read that way.
        for name in ["maximum", "minimum"] {
            ir.push_str(&format!(
                "declare {elem} @llvm.{name}.{suffix}({elem}, {elem})\n"
            ));
            if self.simd {
                ir.push_str(&format!(
                    "declare <{w} x {elem}> @llvm.{name}.v{w}{suffix}(<{w} x {elem}>, <{w} x {elem}>)\n",
                    w = self.lanes()
                ));
            }
        }
        for name in ["exp", "log", "sqrt", "sin", "cos", "fabs", "floor", "ceil"] {
            ir.push_str(&format!(
                "declare {elem} @llvm.{name}.{suffix}({elem})\n"
            ));
            if self.simd {
                ir.push_str(&format!(
                    "declare <{w} x {elem}> @llvm.{name}.v{w}{suffix}(<{w} x {elem}>)\n",
                    w = self.lanes()
                ));
            }
        }
        ir.push_str("declare ptr @malloc(i64)\n");
        ir.push_str("declare void @free(ptr)\n");
        ir.push_str("declare i64 @llvm.umax.i64(i64, i64)\n");
        ir.push_str("declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\n");
        ir.push_str("declare i64 @llvm.umin.i64(i64, i64)\n");
        // The pool: run a part function over [0, n) split across threads,
        // and say how many parts that was.
        ir.push_str("declare void @rho_rt_run(ptr, ptr, i64, i64, i64)\n");
        ir.push_str("declare ptr @rho_rt_scratch(ptr, i64)\n");
        ir.push_str("declare void @rho_rt_zeroize(i64)\n");
        ir.push_str("declare i64 @rho_rt_parts()\n\n");

        // What the last call's `⇒` loops did: how many sweeps in all, and
        // whether every one of them stopped on the tolerance rather than on
        // the cap. Internal, so two kernels in one process do not share them;
        // atomic, so two calls at once do not race (the record is then one
        // call's or the other's).
        ir.push_str("@rho_sweeps = internal global i64 0\n");
        ir.push_str("@rho_converged = internal global i64 1\n\n");

        // 0. The body, once: every entrypoint resolves the spaces to pointers,
        // stores them in the call's table and calls this with it. Its sweeps
        // are outlined into parts the pool runs.
        self.emit_shared_body(block, &mut ir)?;

        // 1. Static entrypoint: void @rho_kernel_exec()
        ir.push_str("define void @rho_kernel_exec() #0 {\n");
        ir.push_str("entry:\n");
        self.emit_frame(&mut ir);

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
            let provided = Self::in_out(&in_sym, &out_sym);
            self.emit_given(&mut ir, &|_| None);
            self.emit_entry(&mut ir, &provided, heap);
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
        self.emit_frame(&mut ir);
        ir.push_str("  ; Null pointer safety check\n");
        ir.push_str("  %in_null = icmp eq ptr %in_ptr, null\n");
        ir.push_str("  br i1 %in_null, label %safety_fail, label %exec_start\n\n");
        ir.push_str("safety_fail:\n");
        ir.push_str("  ret void\n\n");
        ir.push_str("exec_start:\n");
        ir.push_str("  %out_null = icmp eq ptr %out_ptr, null\n");
        ir.push_str("  %out_effective = select i1 %out_null, ptr %in_ptr, ptr %out_ptr\n");
        self.emit_given(&mut ir, &|_| None);
        self.emit_entry(&mut ir, &Self::in_out("%in_ptr", "%out_effective"), Vec::new());
        ir.push_str("}\n\n");

        // 3. Length-checked C-ABI entrypoint. The two-argument form has to trust
        // the caller to have allocated element_count() doubles; this one clamps
        // the sweep to whatever the caller actually owns.
        ir.push_str(
            "define void @rho_kernel_exec_bounded(ptr %in_ptr, ptr %out_ptr, i64 %n) #0 {\n",
        );
        ir.push_str("entry:\n");
        self.emit_frame(&mut ir);
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
        self.emit_given(&mut ir, &|_| None);
        self.emit_body(
            block,
            &mut ir,
            &Self::in_out("%in_ptr", "%b_out_effective"),
            "bounded_start",
            Vec::new(),
            "%sweep",
        )?;
        ir.push_str("}\n\n");

        // 4. Every space at call time: void @rho_kernel_exec_spaces(ptr)
        //
        // The two-pointer form can only name INPUT and OUTPUT, so a kernel
        // with two inputs — a matrix product — had to have its addresses
        // baked in with --bind. Here the caller hands over one pointer per
        // space, in the order the metadata lists them, and nothing is baked.
        self.emit_spaces_entrypoint(&mut ir)?;

        // 5. Metadata export: ptr @rho_kernel_metadata()
        ir.push_str("define ptr @rho_kernel_metadata() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  ret ptr @.rho_meta_str\n");
        ir.push_str("}\n\n");

        // 6. Required buffer length: i64 @rho_kernel_element_count()
        ir.push_str("define i64 @rho_kernel_element_count() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str(&format!("  ret i64 {}\n", self.elements));
        ir.push_str("}\n\n");

        // 7. What the last call's iterations did. A kernel without `⇒`
        // reports no sweeps and counts as converged.
        ir.push_str("define i64 @rho_kernel_sweeps() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  %sweeps = load atomic i64, ptr @rho_sweeps monotonic, align 8\n");
        ir.push_str("  ret i64 %sweeps\n");
        ir.push_str("}\n\n");
        ir.push_str("define i64 @rho_kernel_converged() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str("  %converged = load atomic i64, ptr @rho_converged monotonic, align 8\n");
        ir.push_str("  ret i64 %converged\n");
        ir.push_str("}\n\n");

        // 8. Overwrite what the last calls left behind: the scratch kept
        // between calls, and the stack an entrypoint's frame used, measured
        // from where this is called — the same depth as the kernel when it
        // is called the same way. A caller that computed on secrets calls it
        // before letting them go.
        let frame = self.stack_spaces.borrow().values().sum::<usize>() + ZEROIZE_MARGIN;
        ir.push_str("define void @rho_kernel_zeroize() #0 {\n");
        ir.push_str("entry:\n");
        ir.push_str(&format!("  call void @rho_rt_zeroize(i64 {frame})\n"));
        ir.push_str("  ret void\n");
        ir.push_str("}\n\n");

        // The parts the pool runs, and the scratch kept between calls.
        ir.push_str(&self.outlined.borrow());
        ir.push_str(&self.keeps.borrow());
        ir.push('\n');

        // The metadata last: whether the kernel keeps scratch, and so is
        // reentrant, is known only now.
        let (meta_escaped, meta_len) = Self::c_string(&self.generate_metadata_json());
        ir.push_str(&format!(
            "@.rho_meta_str = private unnamed_addr constant [{meta_len} x i8] c\"{meta_escaped}\", align 1\n\n"
        ));

        // No target-cpu pin in the IR: clang is told the machine, or not.
        ir.push_str("attributes #0 = { nounwind uwtable }\n");

        Ok(ir.replace(CTX_SLOTS, &self.slot_count.get().to_string()))
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
        // `X → =` writes OUTPUT too, and a caller that supplies every buffer
        // needs to know that space and its shape.
        let mut flows: Vec<(&Expr, &str)> = Vec::new();
        for stmt in &block.statements {
            match stmt {
                Statement::Flow { src, target } => flows.push((
                    src,
                    match target {
                        FlowTarget::Var(name) => name.as_str(),
                        FlowTarget::Equilibrium => "OUTPUT",
                    },
                )),
                // A loop's prelude writes spaces of its own.
                Statement::Iterate { prelude, .. } => {
                    for flow in prelude {
                        if let Statement::Flow {
                            src,
                            target: FlowTarget::Var(t),
                        } = flow
                        {
                            flows.push((src, t.as_str()));
                        }
                    }
                }
                _ => {}
            }
        }
        for (src, name) in flows {
            if !shapes.contains_key(name) {
                let inferred = Self::expr_shape(src, &shapes)
                    .or_else(|| Self::primary_shape(&shapes))
                    .unwrap_or_else(|| vec![4]);
                shapes.insert(name.to_string(), inferred);
            }
        }

        self.space_shapes = shapes;
    }

    fn expr_shape(expr: &Expr, shapes: &BTreeMap<String, Vec<usize>>) -> Option<Vec<usize>> {
        crate::ast::expr_shape(expr, shapes)
    }

    fn primary_shape(shapes: &BTreeMap<String, Vec<usize>>) -> Option<Vec<usize>> {
        shapes
            .get("INPUT")
            .cloned()
            .or_else(|| shapes.values().next().cloned())
    }

    /// The cells a caller of the two-pointer entrypoint has to provide: the
    /// larger of the input's and the output's. A take past the end, a reshape
    /// that reads round or an outer product writes more cells than it reads,
    /// and a buffer sized to the input alone would be overrun.
    fn resolve_element_count(&self) -> usize {
        let primary = Self::primary_shape(&self.space_shapes)
            .map(|s| s.iter().product::<usize>())
            .filter(|n| *n > 0)
            .unwrap_or(4);
        let output = self
            .space_shapes
            .get("OUTPUT")
            .map(|s| s.iter().product::<usize>())
            .unwrap_or(0);
        primary.max(output)
    }

    // --------------------------------------------------------------- buffers

    /// Position of a space in the order every table uses.
    fn space_slot(&self, name: &str) -> usize {
        self.space_shapes.keys().position(|n| n == name).unwrap_or(0)
    }

    /// Record, for the body, whether the caller passed each fused space:
    /// `null_flag(name)` is the i1 that is true when its pointer was null,
    /// or None for an entrypoint through which no intermediate arrives.
    fn emit_given(&self, ir: &mut String, null_flag: &dyn Fn(&str) -> Option<String>) {
        let n = self.space_shapes.len().max(1);
        for name in self.fused.keys() {
            let slot = self.space_slot(name);
            let ident = self.sanitize_ident(name);
            let value = match null_flag(name) {
                Some(flag) => {
                    ir.push_str(&format!("  %{ident}_have = xor i1 {flag}, true\n"));
                    ir.push_str(&format!("  %{ident}_given = zext i1 %{ident}_have to i8\n"));
                    format!("%{ident}_given")
                }
                None => "0".to_string(),
            };
            ir.push_str(&format!(
                "  %{ident}_gat = getelementptr [{n} x i8], ptr %rho.given.p, i64 0, i64 {slot}\n"
            ));
            ir.push_str(&format!("  store i8 {value}, ptr %{ident}_gat\n"));
        }
    }

    /// A fused space's own sweep, run only when the caller asked for the
    /// space: the test on the call's record and the blocks around the sweep.
    /// Returns the label the sweep starts in and the one that follows it.
    fn open_given(&self, ir: &mut String, name: &str, label: &str) -> (String, String) {
        let n = self.space_shapes.len().max(1);
        let slot = self.space_slot(name);
        let (run, skip) = (format!("{label}.given"), format!("{label}.owned"));
        ir.push_str(&format!("  ; [{name}] is fused into its reader: its own sweep only if the caller passed it\n"));
        ir.push_str(&format!(
            "  %{label}.gat = getelementptr [{n} x i8], ptr %rho.given.p, i64 0, i64 {slot}\n"
        ));
        ir.push_str(&format!("  %{label}.g = load i8, ptr %{label}.gat\n"));
        ir.push_str(&format!("  %{label}.gb = icmp ne i8 %{label}.g, 0\n"));
        ir.push_str(&format!("  br i1 %{label}.gb, label %{run}, label %{skip}\n\n{run}:\n"));
        (run, skip)
    }

    fn close_given(ir: &mut String, skip: &str) -> String {
        ir.push_str(&format!("  br label %{skip}\n\n{skip}:\n"));
        skip.to_string()
    }

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
            // Kept between calls (see rho_rt_scratch): the size is fixed
            // here, and fresh pages cost more than the sweep at a million
            // cells. Nothing goes on the free list.
            let _ = &heap;
            self.kept_spaces
                .borrow_mut()
                .insert(label.to_string(), count * self.precision.bytes());
            let keep = self.keep_count.get();
            self.keep_count.set(keep + 1);
            self.keeps
                .borrow_mut()
                .push_str(&format!("@rho_keep{keep} = internal global ptr null  ; [{label}]\n"));
            ir.push_str(&format!("  ; [{label}] {count} cells -> scratch kept between calls\n"));
            ir.push_str(&format!(
                "  {sym} = call ptr @rho_rt_scratch(ptr @rho_keep{keep}, i64 {})\n",
                count * self.precision.bytes()
            ));
        } else {
            self.stack_spaces
                .borrow_mut()
                .insert(label.to_string(), count * self.precision.bytes());
            ir.push_str(&format!("  ; [{label}] {count} cells -> stack\n"));
            ir.push_str(&format!(
                "  {sym} = alloca [{count} x {}], align 64\n",
                self.precision.llvm_type()
            ));
        }
        sym
    }

    /// Whether the expression reads anything out of order: a rotation or a
    /// reversal wraps at the end of the axis, and a reshape to more cells
    /// reads its operand round again. Such lanes are not contiguous, so the
    /// sweep stays scalar. 📋 A gathered vector path would lift this.
    /// Why a sweep runs one cell at a time, as `explain` says it.
    fn scalar_reason(&self, src: &Expr, result_shape: &[usize], bound: &str) -> String {
        if let Some(turn) = self.turn_kind(src) {
            return turn.to_string();
        }
        if self.needs_broadcast(src, result_shape, &[])
            && !self.stretched_contiguously(src, result_shape, &[], self.lanes(), false)
        {
            return "a stretched read is not contiguous (a column stretched along a row, or under a shift)"
                .to_string();
        }
        if !self.simd {
            return "vectors are off (--no-simd)".to_string();
        }
        if bound.parse::<u64>().is_err() {
            return "its length is known only at run time".to_string();
        }
        "too short for a vector beside its shifts' reach".to_string()
    }

    /// A gather a vector at a time: the positions as the scalar path takes
    /// them (floored, clamped into the space, so a lane never reads outside
    /// it), each lane's cell loaded, and zero where the position named no
    /// cell. The loads are per lane, not `llvm.masked.gather`, whose
    /// signature differs between LLVM versions; the backend may join them.
    fn emit_vector_gather(
        &self,
        ir: &mut String,
        ptr: &str,
        count: usize,
        position: &str,
        w: usize,
        counter: &mut usize,
    ) -> String {
        let mode = Mode::Vector(w, self.precision);
        let (vty, ity, bty) = (mode.ty(), mode.int_ty(), mode.bool_ty());
        let elem = self.precision.llvm_type();
        let floor = mode.unary_intrinsic("floor");
        let floored = Self::fresh(counter);
        let not_before = Self::fresh(counter);
        let not_past = Self::fresh(counter);
        let inside = Self::fresh(counter);
        let safe = Self::fresh(counter);
        let at = Self::fresh(counter);
        ir.push_str(&format!("  {floored} = call {vty} {floor}({vty} {position})\n"));
        ir.push_str(&format!("  {not_before} = fcmp oge {vty} {floored}, {}\n", mode.splat(0.0)));
        ir.push_str(&format!("  {not_past} = fcmp olt {vty} {floored}, {}\n", mode.splat(count as f64)));
        ir.push_str(&format!("  {inside} = and {bty} {not_before}, {not_past}\n"));
        ir.push_str(&format!("  {safe} = select {bty} {inside}, {vty} {floored}, {vty} zeroinitializer\n"));
        // To 32 bits where the space allows, which x86-64 converts four
        // lanes at a time without AVX-512 (to 64 bits it converts one by one).
        let narrow = count < (1usize << 31);
        let ity = if narrow { format!("<{w} x i32>") } else { ity };
        ir.push_str(&format!("  {at} = fptosi {vty} {safe} to {ity}\n"));
        let mut lanes = "poison".to_string();
        for lane in 0..w {
            let (i, gep, val, next) =
                (Self::fresh(counter), Self::fresh(counter), Self::fresh(counter), Self::fresh(counter));
            let lane_ty = if narrow { "i32" } else { "i64" };
            ir.push_str(&format!("  {i} = extractelement {ity} {at}, i64 {lane}\n"));
            ir.push_str(&format!("  {gep} = getelementptr inbounds {elem}, ptr {ptr}, {lane_ty} {i}\n"));
            ir.push_str(&format!("  {val} = load {elem}, ptr {gep}, align {}\n", self.precision.bytes()));
            ir.push_str(&format!("  {next} = insertelement {vty} {lanes}, {elem} {val}, i64 {lane}\n"));
            lanes = next;
        }
        let out = Self::fresh(counter);
        ir.push_str(&format!("  {out} = select {bty} {inside}, {vty} {lanes}, {vty} zeroinitializer\n"));
        out
    }

    /// The first read out of order in an expression, as `explain` says it.
    fn turn_kind(&self, expr: &Expr) -> Option<&'static str> {
        match expr {
            Expr::Gather { index, .. } => self.turn_kind(index),
            Expr::Rotate { .. } => Some("`⌽` wraps at the end of an axis"),
            Expr::Reverse { .. } => Some("`⌽` reverses an axis"),
            Expr::Transpose { .. } => Some("`⍉` reads across rows"),
            Expr::Take { .. } | Expr::Drop { .. } => Some("`↑` or `↓` changes the length of a row"),
            Expr::Reshape { .. } if self.turns(expr) => Some("`⍴` to another count reads its operand round"),
            Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } | Expr::Call { .. } | Expr::Reshape { .. } => None,
            Expr::AuditTrace(inner)
            | Expr::Shift { operand: inner, .. }
            | Expr::Builtin { operand: inner, .. }
            | Expr::Lift { operand: inner, .. }
            | Expr::Scan { operand: inner, .. }
            | Expr::Reduce { operand: inner, .. } => self.turn_kind(inner),
            Expr::BinaryOp { lhs, rhs, .. } => self.turn_kind(lhs).or_else(|| self.turn_kind(rhs)),
        }
    }

    fn turns(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Rotate { .. }
            | Expr::Reverse { .. }
            | Expr::Transpose { .. }
            | Expr::Take { .. }
            | Expr::Drop { .. } => true,
            // A gather's positions are read like anything else, and its
            // space at each lane's own position.
            Expr::Gather { index, .. } => self.turns(index),
            // Same count: a reinterpretation, contiguous as the operand is.
            Expr::Reshape { shape, operand } => {
                match Self::expr_shape(operand, &self.space_shapes) {
                    Some(source) => {
                        source.iter().product::<usize>() != shape.iter().product::<usize>()
                    }
                    None => true,
                }
            }
            Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } | Expr::Call { .. } => false,
            Expr::AuditTrace(inner)
            | Expr::Shift { operand: inner, .. }
            | Expr::Builtin { operand: inner, .. }
            | Expr::Lift { operand: inner, .. }
            | Expr::Scan { operand: inner, .. }
            | Expr::Reduce { operand: inner, .. } => self.turns(inner),
            Expr::BinaryOp { lhs, rhs, .. } => self.turns(lhs) || self.turns(rhs),
        }
    }

    /// The pointer map an entrypoint with an input and an output starts from.
    fn in_out(in_sym: &str, out_sym: &str) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        map.insert("INPUT".to_string(), in_sym.to_string());
        map.insert("OUTPUT".to_string(), out_sym.to_string());
        map
    }

    /// Resolve every space to a pointer: the ones in `provided` came in from
    /// the caller, a bound space is read at its address, and the rest are
    /// scratch the kernel owns for the length of the call.
    fn emit_buffers(
        &self,
        ir: &mut String,
        provided: &BTreeMap<String, String>,
        heap: Vec<String>,
        bound: &str,
    ) -> Buffers {
        let mut map = provided.clone();
        let mut heap = heap;

        for (name, shape) in &self.space_shapes {
            if map.contains_key(name) {
                continue;
            }
            let sanitized = self.sanitize_ident(name);

            // A space bound to an address is the caller's memory, not ours.
            // An entrypoint that only names INPUT and OUTPUT reaches every
            // other space through its binding.
            if let Some(addr) = self.ext_bindings.get(name) {
                let sym = format!("%{sanitized}_ext");
                ir.push_str(&format!("  ; Zero-copy binding for [{name}] at {addr:#X}\n"));
                ir.push_str(&format!("  {sym} = inttoptr i64 {addr} to ptr\n"));
                map.insert(name.clone(), sym);
                continue;
            }

            // A fused space is computed inside its reader; through an
            // entrypoint that cannot pass it, it is never stored.
            if self.fused.contains_key(name) {
                map.insert(name.clone(), "null".to_string());
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
            parallel: !bound.starts_with('%'),
        }
    }

    /// The call's table slot of its partial results (one per part, the
    /// largest move a part of a `⇒` comparison found).
    fn partial_slot(&self) -> usize {
        self.space_shapes.len()
    }

    /// The call's table slot of its record of which fused spaces the caller
    /// passed.
    fn given_slot(&self) -> usize {
        self.space_shapes.len() + 1
    }

    /// The address of slot `slot` of the call's table, as an instruction
    /// appended to `ir` (the table is not a constant, so neither is it).
    fn ctx_slot(&self, ir: &mut String, slot: usize) -> String {
        let n = self.ctx_tmp.get();
        self.ctx_tmp.set(n + 1);
        ir.push_str(&format!("  %ctx.{n} = getelementptr ptr, ptr %rho.ctx, i64 {slot}\n"));
        format!("%ctx.{n}")
    }

    /// The start of every entrypoint: the call's table, its partial results
    /// and its record of passed spaces, on its own stack, so that calls at
    /// once do not share them. The table's length is known when the whole
    /// module is (`CTX_SLOTS` is replaced then).
    fn emit_frame(&self, ir: &mut String) {
        let elem = self.precision.llvm_type();
        let n = self.space_shapes.len().max(1);
        ir.push_str(&format!("  %rho.ctx = alloca [{CTX_SLOTS} x ptr], align 8\n"));
        ir.push_str(&format!("  %rho.partial.p = alloca [{MAX_PARTS} x {elem}], align 64\n"));
        ir.push_str(&format!("  %rho.given.p = alloca [{n} x i8], align 8\n"));
        let at = self.ctx_slot(ir, self.partial_slot());
        ir.push_str(&format!("  store ptr %rho.partial.p, ptr {at}\n"));
        let at = self.ctx_slot(ir, self.given_slot());
        ir.push_str(&format!("  store ptr %rho.given.p, ptr {at}\n"));
    }

    /// Slot of a pointer in the call's table, storing it there the first
    /// time. Every pointer an outlined part reads goes through here.
    fn share(&self, ir: &mut String, sym: &str) -> usize {
        if let Some(&slot) = self.slots.borrow().get(sym) {
            return slot;
        }
        let slot = self.slot_count.get();
        self.slot_count.set(slot + 1);
        self.slots.borrow_mut().insert(sym.to_string(), slot);
        let at = self.ctx_slot(ir, slot);
        ir.push_str(&format!("  store ptr {sym}, ptr {at}\n"));
        slot
    }

    /// Open a part function: `void (i64 lo, i64 hi, i64 part, ptr ctx)`. Its
    /// prologue loads every pointer in `bufs` (and `extra`) from the call's
    /// table, so the body emitted into it reads like the caller's. Returns the function's
    /// name, its text so far, the buffers as seen from inside, and the
    /// translation of every symbol.
    fn open_part(
        &self,
        bufs: &Buffers,
        extra: &[&str],
    ) -> (String, String, Buffers, BTreeMap<String, String>) {
        let id = self.outlined_count.get() + 1;
        self.outlined_count.set(id);
        let name = format!("@rho_part{id}");
        let mut f = format!(
            "define internal void {name}(i64 %lo, i64 %hi, i64 %part, ptr %rho.ctx) #0 {{\nentry:\n"
        );
        let mut translated: BTreeMap<String, String> = BTreeMap::new();
        let mut loaded: BTreeSet<usize> = BTreeSet::new();
        let slots = self.slots.borrow();
        let symbols = bufs
            .map
            .values()
            .chain(bufs.folds.values())
            .map(String::as_str)
            .chain(extra.iter().copied());
        for sym in symbols {
            if translated.contains_key(sym) {
                continue;
            }
            let slot = slots
                .get(sym)
                .copied()
                .unwrap_or_else(|| panic!("pointer {sym} was never shared"));
            loaded.insert(slot);
            translated.insert(sym.to_string(), format!("%c{slot}"));
        }
        // In slot order: the folds are keyed by their node's address, so
        // the order they come in follows the allocator, and the IR must not.
        for slot in &loaded {
            let at = self.ctx_slot(&mut f, *slot);
            f.push_str(&format!("  %c{slot} = load ptr, ptr {at}\n"));
        }
        let at = self.ctx_slot(&mut f, self.partial_slot());
        f.push_str(&format!("  %rho.partial.p = load ptr, ptr {at}\n"));
        let inner = Buffers {
            map: bufs.map.iter().map(|(k, v)| (k.clone(), translated[v].clone())).collect(),
            folds: bufs.folds.iter().map(|(k, v)| (*k, translated[v].clone())).collect(),
            heap: Vec::new(),
            bound: bufs.bound.clone(),
            parallel: false,
        };
        (name, f, inner, translated)
    }

    /// Finish a part function and hand its range to the pool from `ir`.
    fn close_part(&self, ir: &mut String, name: &str, mut f: String, total: &str) {
        f.push_str("  ret void\n}\n\n");
        self.outlined.borrow_mut().push_str(&f);
        ir.push_str(&format!(
            "  call void @rho_rt_run(ptr {name}, ptr %rho.ctx, i64 {total}, i64 {}, i64 {})\n",
            self.threads, self.grain
        ));
    }

    // ----------------------------------------------------------- entrypoints

    /// `void rho_kernel_exec_spaces(void **spaces)`: one pointer per space, in
    /// the order `rho_kernel_metadata()` lists them.
    ///
    /// A null pointer for a space no flow writes makes the call return without
    /// touching memory, as a null input does for the two-pointer form. A null
    /// pointer for a space some flow writes hands that space to the kernel,
    /// which uses scratch of its own — so a caller passes its inputs and its
    /// output, and may leave every intermediate to the kernel.
    fn emit_spaces_entrypoint(&self, ir: &mut String) -> Result<()> {
        ir.push_str("define void @rho_kernel_exec_spaces(ptr %spaces) #0 {\n");
        ir.push_str("entry:\n");
        self.emit_frame(ir);
        ir.push_str("  %spaces_null = icmp eq ptr %spaces, null\n");
        ir.push_str("  br i1 %spaces_null, label %spaces_fail, label %spaces_load\n\n");
        ir.push_str("spaces_fail:\n");
        ir.push_str("  ret void\n\n");
        ir.push_str("spaces_load:\n");

        let mut provided = BTreeMap::new();
        // Null flags of the spaces the kernel only reads; any one set aborts.
        let mut required: Vec<String> = Vec::new();
        // Spaces the kernel writes, which it can also own: (name, arg, flag).
        let mut optional: Vec<(String, String, String)> = Vec::new();

        for (index, name) in self.space_shapes.keys().enumerate() {
            let ident = self.sanitize_ident(name);
            let slot = format!("%{ident}_slot");
            let arg = format!("%{ident}_arg");
            let flag = format!("%{ident}_null");
            ir.push_str(&format!("  ; [{name}] is spaces[{index}]\n"));
            ir.push_str(&format!(
                "  {slot} = getelementptr ptr, ptr %spaces, i64 {index}\n"
            ));
            ir.push_str(&format!("  {arg} = load ptr, ptr {slot}\n"));
            ir.push_str(&format!("  {flag} = icmp eq ptr {arg}, null\n"));
            if self.written_spaces.contains(name) {
                optional.push((name.clone(), arg.clone(), flag));
            } else {
                required.push(flag);
            }
            provided.insert(name.clone(), arg);
        }

        match required.split_first() {
            None => ir.push_str("  br label %spaces_start\n\n"),
            Some((first, rest)) => {
                let mut missing = first.clone();
                for (k, flag) in rest.iter().enumerate() {
                    let next = format!("%missing{k}");
                    ir.push_str(&format!("  {next} = or i1 {missing}, {flag}\n"));
                    missing = next;
                }
                ir.push_str(&format!(
                    "  br i1 {missing}, label %spaces_fail, label %spaces_start\n\n"
                ));
            }
        }

        ir.push_str("spaces_start:\n");
        let mut heap = Vec::new();
        let mut null_flags: BTreeMap<String, String> = BTreeMap::new();
        for (name, arg, flag) in optional {
            // A fused space the caller did not pass is computed inside its
            // reader and never stored: it needs no memory, and gets none.
            if self.fused.contains_key(&name) {
                null_flags.insert(name, flag);
                continue;
            }
            let ident = self.sanitize_ident(&name);
            let count = self.space_shapes[&name].iter().product::<usize>().max(1);
            let own = self.emit_scratch(ir, &format!("{ident}_own"), count, &mut heap);
            let effective = format!("%{ident}_eff");
            ir.push_str(&format!(
                "  {effective} = select i1 {flag}, ptr {own}, ptr {arg}\n"
            ));
            provided.insert(name.clone(), effective);
            null_flags.insert(name, flag);
        }
        // A fused space's own flow runs here only when its pointer came in.
        self.emit_given(ir, &|name: &str| null_flags.get(name).cloned());

        self.emit_entry(ir, &provided, heap);
        ir.push_str("}\n\n");
        Ok(())
    }

    // ---------------------------------------------------------------- bodies

    /// An entrypoint's share of the work: resolve every space to a pointer,
    /// store them for the body and its parts, run the body, free the scratch.
    fn emit_entry(&self, ir: &mut String, provided: &BTreeMap<String, String>, heap: Vec<String>) {
        let bufs = self.emit_buffers(ir, provided, heap, &self.elements.to_string());
        for (slot, name) in self.space_shapes.keys().enumerate() {
            if let Some(sym) = bufs.map.get(name) {
                let at = self.ctx_slot(ir, slot);
                ir.push_str(&format!("  store ptr {sym}, ptr {at}\n"));
            }
        }
        ir.push_str("  call void @rho_body(ptr %rho.ctx)\n");
        for ptr in &bufs.heap {
            ir.push_str(&format!("  call void @free(ptr {ptr})\n"));
        }
        ir.push_str("  ret void\n");
    }

    /// The flows, once, reading the spaces from the call's table. Every
    /// sweep in it is outlined into a part and run on the pool.
    fn emit_shared_body(&self, block: &ToposBlock, ir: &mut String) -> Result<()> {
        ir.push_str("define internal void @rho_body(ptr %rho.ctx) #0 {\n");
        ir.push_str("entry:\n");
        let mut map = BTreeMap::new();
        for (slot, name) in self.space_shapes.keys().enumerate() {
            let sym = format!("%sp{slot}");
            let at = self.ctx_slot(ir, slot);
            ir.push_str(&format!("  {sym} = load ptr, ptr {at}  ; [{name}]\n"));
            self.slots.borrow_mut().insert(sym.clone(), slot);
            map.insert(name.clone(), sym);
        }
        let at = self.ctx_slot(ir, self.partial_slot());
        ir.push_str(&format!("  %rho.partial.p = load ptr, ptr {at}\n"));
        let at = self.ctx_slot(ir, self.given_slot());
        ir.push_str(&format!("  %rho.given.p = load ptr, ptr {at}\n"));
        let mut bufs = Buffers {
            map,
            folds: BTreeMap::new(),
            heap: Vec::new(),
            bound: self.elements.to_string(),
            parallel: true,
        };
        ir.push_str("  store atomic i64 0, ptr @rho_sweeps monotonic, align 8\n");
        ir.push_str("  store atomic i64 1, ptr @rho_converged monotonic, align 8\n");
        let mut counter = 0usize;
        self.notes.borrow_mut().clear();
        self.recording.set(true);
        let emitted = self.emit_flows(&block.statements, ir, &mut bufs, "entry", &mut counter);
        self.recording.set(false);
        emitted?;
        for ptr in &bufs.heap {
            ir.push_str(&format!("  call void @free(ptr {ptr})\n"));
        }
        ir.push_str("  ret void\n");
        ir.push_str("}\n\n");
        Ok(())
    }

    fn emit_body(
        &self,
        block: &ToposBlock,
        ir: &mut String,
        provided: &BTreeMap<String, String>,
        entry_label: &str,
        heap: Vec<String>,
        bound: &str,
    ) -> Result<()> {
        let mut bufs = self.emit_buffers(ir, provided, heap, bound);
        // Every call starts its iteration record afresh.
        ir.push_str("  store atomic i64 0, ptr @rho_sweeps monotonic, align 8\n");
        ir.push_str("  store atomic i64 1, ptr @rho_converged monotonic, align 8\n");
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
            self.current_line.set(self.statement_lines.get(index).copied().unwrap_or(0));
            match stmt {
                Statement::Flow { src, target } => {
                    loop_id += 1;
                    let target_ptr = match target {
                        FlowTarget::Var(name) => self.lookup(bufs, name)?,
                        FlowTarget::Equilibrium => self.lookup(bufs, "OUTPUT")?,
                    };
                    let label = format!("f{loop_id}");
                    let guarded = match target {
                        FlowTarget::Var(name) if self.fused.contains_key(name) => {
                            let (run, skip) = self.open_given(ir, name, &label);
                            pred = run;
                            Some(skip)
                        }
                        _ => None,
                    };
                    let written = self.written_names(target);
                    let mut was = Vec::new();
                    if matches!(target, FlowTarget::Equilibrium)
                        && self.space_shapes.contains_key("INPUT")
                        && Self::reads_elsewhere(src, &["INPUT".to_string()], false, &self.space_shapes)
                    {
                        let (input_was, join) = self.snapshot_if_aliased(ir, bufs, &target_ptr, &label)?;
                        was.extend(input_was);
                        pred = join;
                    }
                    if Self::reads_elsewhere(src, &written, false, &self.space_shapes) {
                        let copy = self.reserve_snapshot(ir, bufs, &written, &label);
                        was.extend(self.take_snapshot(ir, bufs, &written, &target_ptr, &copy));
                    }
                    pred = self.emit_sweep(
                        src,
                        target,
                        &target_ptr,
                        ir,
                        bufs,
                        &pred,
                        &label,
                        &format!("Flow {loop_id}"),
                        counter,
                        None,
                    )?;
                    bufs.map.extend(was);
                    if let Some(skip) = guarded {
                        pred = Self::close_given(ir, &skip);
                    }
                    if matches!(target, FlowTarget::Equilibrium) {
                        break;
                    }
                }
                Statement::Iterate { prelude, src, target } => {
                    loop_id += 1;
                    pred = self.emit_iterate(prelude, src, target, ir, bufs, &pred, loop_id, counter)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Lanes per vector step at this precision.
    fn lanes(&self) -> usize {
        VECTOR_BYTES / self.precision.bytes()
    }

    /// The name a flow's target goes by.
    fn written_names(&self, target: &FlowTarget) -> Vec<String> {
        match target {
            FlowTarget::Var(name) => vec![name.clone()],
            FlowTarget::Equilibrium => vec!["OUTPUT".to_string()],
        }
    }

    /// INPUT may be OUTPUT's own buffer at run time: an unbound OUTPUT
    /// beside a bound INPUT, a null output pointer to the two-pointer form,
    /// or a caller passing one pointer for both. A flow into OUTPUT that
    /// reads INPUT elsewhere than at the cell it writes then reads a copy,
    /// taken only when the two are the same. Returns the bindings to restore
    /// and the block the sweep starts from.
    fn snapshot_if_aliased(
        &self,
        ir: &mut String,
        bufs: &mut Buffers,
        target_ptr: &str,
        label: &str,
    ) -> Result<(Vec<(String, String)>, String)> {
        let input = self.lookup(bufs, "INPUT")?;
        let cells = self.shape_for("INPUT").iter().product::<usize>().max(1);
        let copy = self.emit_scratch(ir, &format!("{label}_in_was"), cells, &mut bufs.heap);
        let align = self.precision.bytes();
        let (copying, join, read) = (format!("{label}.alias_copy"), format!("{label}.alias_join"), format!("%{label}.in"));
        ir.push_str("  ; [INPUT] may be OUTPUT's buffer, written in place: the sweep then reads a copy\n");
        ir.push_str(&format!("  %{label}.alias = icmp eq ptr {input}, {target_ptr}\n"));
        ir.push_str(&format!("  br i1 %{label}.alias, label %{copying}, label %{join}\n\n{copying}:\n"));
        ir.push_str(&format!(
            "  call void @llvm.memcpy.p0.p0.i64(ptr align {align} {copy}, ptr align {align} {input}, i64 {}, i1 false)\n",
            cells * align
        ));
        ir.push_str(&format!("  br label %{join}\n\n{join}:\n"));
        ir.push_str(&format!("  {read} = select i1 %{label}.alias, ptr {copy}, ptr {input}\n"));
        if bufs.parallel {
            self.share(ir, &read);
        }
        let was = bufs.map.insert("INPUT".to_string(), read).map(|old| ("INPUT".to_string(), old));
        Ok((was.into_iter().collect(), join))
    }

    /// Whether `expr` reads one of `names` at a cell other than the one the
    /// sweep writes: shifted, turned, gathered, stretched or reshaped to
    /// another count. A sweep writes its target in place, a cell at a time
    /// (or a vector, or a part on each thread), so such a read would see
    /// cells already overwritten; the interpreter, which is the meaning,
    /// reads the values from before the flow. A fold is computed before the
    /// sweep, and reads them; `⍳` reads no cell.
    fn reads_elsewhere(
        expr: &Expr,
        names: &[String],
        displaced: bool,
        shapes: &BTreeMap<String, Vec<usize>>,
    ) -> bool {
        let again = |e: &Expr, d: bool| Self::reads_elsewhere(e, names, d, shapes);
        match expr {
            Expr::Var(name) => displaced && names.contains(name),
            Expr::Number(_) | Expr::Index { .. } | Expr::Call { .. } => false,
            Expr::Reduce { .. } | Expr::Scan { .. } => false,
            Expr::AuditTrace(inner) | Expr::Builtin { operand: inner, .. } => again(inner, displaced),
            Expr::Shift { operand, .. }
            | Expr::Rotate { operand, .. }
            | Expr::Reverse { operand, .. }
            | Expr::Transpose { operand, .. }
            | Expr::Take { operand, .. }
            | Expr::Drop { operand, .. }
            | Expr::Lift { operand, .. } => again(operand, true),
            Expr::Reshape { shape, operand } => {
                let same = Self::expr_shape(operand, shapes)
                    .is_some_and(|s| s.iter().product::<usize>() == shape.iter().product::<usize>());
                again(operand, displaced || !same)
            }
            Expr::Gather { index, operand } => again(operand, true) || again(index, displaced),
            Expr::BinaryOp { lhs, rhs, .. } => again(lhs, displaced) || again(rhs, displaced),
        }
    }

    /// Room for a copy of a flow's target, as large as the target.
    fn reserve_snapshot(&self, ir: &mut String, bufs: &mut Buffers, written: &[String], label: &str) -> String {
        let cells = self.shape_for(&written[0]).iter().product::<usize>().max(1);
        let copy = self.emit_scratch(ir, &format!("{label}_was"), cells, &mut bufs.heap);
        if bufs.parallel {
            self.share(ir, &copy);
        }
        copy
    }

    /// Copy the target as it is into `copy` and point every name it goes by
    /// at the copy for the sweep's reads; returns the bindings to restore.
    fn take_snapshot(
        &self,
        ir: &mut String,
        bufs: &mut Buffers,
        written: &[String],
        target_ptr: &str,
        copy: &str,
    ) -> Vec<(String, String)> {
        let cells = self.shape_for(&written[0]).iter().product::<usize>().max(1);
        let align = self.precision.bytes();
        ir.push_str(&format!(
            "  ; [{}] is read elsewhere than where it is written: the sweep reads a copy\n",
            written[0]
        ));
        ir.push_str(&format!(
            "  call void @llvm.memcpy.p0.p0.i64(ptr align {align} {copy}, ptr align {align} {target_ptr}, i64 {}, i1 false)\n",
            cells * align
        ));
        let mut was = Vec::new();
        for name in written {
            if let Some(old) = bufs.map.insert(name.clone(), copy.to_string()) {
                was.push((name.clone(), old));
            }
        }
        was
    }

    /// One full sweep of `src` into `target_ptr`: the folds it needs first,
    /// then a head, a vector body and a tail when the length allows it.
    /// Returns the block control flow lands on.
    #[allow(clippy::too_many_arguments)]
    fn emit_sweep(
        &self,
        src: &Expr,
        target: &FlowTarget,
        target_ptr: &str,
        ir: &mut String,
        bufs: &mut Buffers,
        pred: &str,
        label: &str,
        what: &str,
        counter: &mut usize,
        measured: Option<&str>,
    ) -> Result<String> {
        let mut pred = pred.to_string();

        // Fold every reduction in this flow into its own buffer first. The
        // sweep below then reads a plain array, so it stays straight-line.
        pred = self.emit_fold_prepass(src, ir, bufs, &pred, counter)?;

        let sweep = self.sweep_length(src, target, bufs);
        let result_shape = self.sweep_shape(src, target);
        // A read that wraps at the end of an axis is not contiguous, so it
        // cannot be vector loaded; nor is a stretched read, unless every
        // lane of a chunk still lands on the next cell of its source.
        let plan = if self.turns(src) {
            Sweep::AllScalar
        } else if self.needs_broadcast(src, &result_shape, &[]) {
            if self.stretched_contiguously(src, &result_shape, &[], self.lanes(), false) {
                self.plan_sweep(src, &sweep, true)?
            } else {
                Sweep::AllScalar
            }
        } else {
            self.plan_sweep(src, &sweep, false)?
        };
        if self.recording.get() {
            let name = match target {
                FlowTarget::Var(n) => n.clone(),
                FlowTarget::Equilibrium => "OUTPUT".to_string(),
            };
            let plan_text = match &plan {
                Sweep::Split { width, aligned: true, .. } => format!("vector ×{width}, rows stretched over rows"),
                Sweep::Split { width, .. } => format!("vector ×{width}"),
                Sweep::AllScalar => format!("scalar: {}", self.scalar_reason(src, &result_shape, &sweep)),
            };
            self.notes.borrow_mut().push(FlowNote {
                line: self.current_line.get(),
                target: name.clone(),
                cells: sweep.clone(),
                plan: plan_text,
                folds: fold_count(src),
                fused_into: self.fused.get(&name).cloned(),
                round_of: self.in_loop.borrow().clone(),
            });
        }
        ir.push_str(&format!(
            "  ; {what}: sweep {sweep} cells{}\n",
            plan.describe()
        ));

        if bufs.parallel {
            // The sweep as a part over [lo, hi): the pool hands each thread
            // its own range, and this block only makes the call.
            let mut extra = vec![target_ptr];
            extra.extend(measured);
            let (name, mut f, inner, translated) = self.open_part(bufs, &extra);
            let tp = translated[target_ptr].clone();
            // A round's sweep measures its largest move as it goes: one
            // accumulator for the scalar cells, one of lanes for the vector
            // body, both in the part's own frame, reduced into its slot of
            // the call's partial results when the range is done.
            let elem = self.precision.llvm_type();
            let width = match plan {
                Sweep::Split { width, .. } => width,
                Sweep::AllScalar => 1,
            };
            let measure = measured.map(|cur| {
                f.push_str(&format!("  %{label}.macc = alloca {elem}, align 8\n"));
                f.push_str(&format!("  store {elem} {}, ptr %{label}.macc\n", self.f64_literal(0.0)));
                f.push_str(&format!("  %{label}.maccv = alloca <{width} x {elem}>, align 64\n"));
                f.push_str(&format!("  store <{width} x {elem}> zeroinitializer, ptr %{label}.maccv\n"));
                Measure {
                    cur: translated[cur].clone(),
                    acc: format!("%{label}.macc"),
                    acc_vec: format!("%{label}.maccv"),
                }
            });
            let mut p = "entry".to_string();
            match plan {
                Sweep::AllScalar => {
                    p = self.emit_range_loop(
                        &mut f,
                        label,
                        &p,
                        "%lo",
                        "%hi",
                        Mode::Scalar(self.precision),
                        src,
                        &tp,
                        &inner,
                        counter,
                        &result_shape,
                    measure.as_ref(),
                    )?;
                }
                Sweep::Split {
                    vector_start,
                    vector_end,
                    width,
                    aligned,
                    ..
                } => {
                    // The static split, clipped to this part: the head is
                    // what lies below the vector range, the body a whole
                    // number of lanes from there, the tail the rest. An
                    // aligned body starts at the next multiple of the width.
                    if aligned {
                        f.push_str(&format!(
                            "  %{label}.vs0 = call i64 @llvm.umax.i64(i64 %lo, i64 {vector_start})\n"
                        ));
                        f.push_str(&format!("  %{label}.vs1 = add i64 %{label}.vs0, {}\n", width - 1));
                        f.push_str(&format!(
                            "  %{label}.vs = and i64 %{label}.vs1, {}\n",
                            !((width as u64) - 1)
                        ));
                    } else {
                        f.push_str(&format!(
                            "  %{label}.vs = call i64 @llvm.umax.i64(i64 %lo, i64 {vector_start})\n"
                        ));
                    }
                    f.push_str(&format!(
                        "  %{label}.hh = call i64 @llvm.umin.i64(i64 %hi, i64 %{label}.vs)\n"
                    ));
                    f.push_str(&format!(
                        "  %{label}.vl = call i64 @llvm.umin.i64(i64 %hi, i64 {vector_end})\n"
                    ));
                    f.push_str(&format!(
                        "  %{label}.has = icmp ugt i64 %{label}.vl, %{label}.vs\n"
                    ));
                    f.push_str(&format!(
                        "  %{label}.span0 = sub i64 %{label}.vl, %{label}.vs\n"
                    ));
                    f.push_str(&format!(
                        "  %{label}.span = select i1 %{label}.has, i64 %{label}.span0, i64 0\n"
                    ));
                    f.push_str(&format!(
                        "  %{label}.lanes = and i64 %{label}.span, {}\n",
                        !((width as u64) - 1)
                    ));
                    f.push_str(&format!(
                        "  %{label}.ve = add i64 %{label}.vs, %{label}.lanes\n"
                    ));
                    p = self.emit_range_loop(
                        &mut f,
                        &format!("{label}.head"),
                        &p,
                        "%lo",
                        &format!("%{label}.hh"),
                        Mode::Scalar(self.precision),
                        src,
                        &tp,
                        &inner,
                        counter,
                        &result_shape,
                    measure.as_ref(),
                    )?;
                    p = self.emit_range_loop(
                        &mut f,
                        &format!("{label}.vec"),
                        &p,
                        &format!("%{label}.vs"),
                        &format!("%{label}.ve"),
                        Mode::Vector(width, self.precision),
                        src,
                        &tp,
                        &inner,
                        counter,
                        &result_shape,
                    measure.as_ref(),
                    )?;
                    p = self.emit_range_loop(
                        &mut f,
                        &format!("{label}.tail"),
                        &p,
                        &format!("%{label}.ve"),
                        "%hi",
                        Mode::Scalar(self.precision),
                        src,
                        &tp,
                        &inner,
                        counter,
                        &result_shape,
                    measure.as_ref(),
                    )?;
                }
            }
            if let Some(m) = &measure {
                // The largest of the lanes and of the scalar cells, the same
                // rule as between cells, into this part's slot.
                f.push_str(&format!("{p}.measure:\n"));
                let mut acc = format!("%{label}.ms");
                f.push_str(&format!("  {acc} = load {elem}, ptr {}\n", m.acc));
                let lanes = Self::fresh(counter);
                f.push_str(&format!("  {lanes} = load <{width} x {elem}>, ptr {}\n", m.acc_vec));
                for lane in 0..width {
                    let (l, grew, next) = (Self::fresh(counter), Self::fresh(counter), Self::fresh(counter));
                    f.push_str(&format!("  {l} = extractelement <{width} x {elem}> {lanes}, i32 {lane}\n"));
                    f.push_str(&format!("  {grew} = fcmp ogt {elem} {l}, {acc}\n"));
                    f.push_str(&format!("  {next} = select i1 {grew}, {elem} {l}, {elem} {acc}\n"));
                    acc = next;
                }
                f.push_str(&format!(
                    "  %{label}.mslot = getelementptr inbounds {elem}, ptr %rho.partial.p, i64 %part\n"
                ));
                f.push_str(&format!("  store {elem} {acc}, ptr %{label}.mslot, align {}\n", self.precision.bytes()));
                // the block above must be entered from the last loop's end
                let text = format!("{p}.measure:\n");
                let at = f.rfind(&text).unwrap();
                f.insert_str(at, &format!("  br label %{p}.measure\n\n"));
            }
            let _ = p;
            self.close_part(ir, &name, f, &sweep);
            return Ok(pred);
        }

        match plan {
            Sweep::AllScalar => {
                pred = self.emit_range_loop(
                    ir,
                    label,
                    &pred,
                    "0",
                    &sweep,
                    Mode::Scalar(self.precision),
                    src,
                    target_ptr,
                    bufs,
                    counter,
                    &result_shape,
                None,
                )?;
            }
            Sweep::Split {
                vector_start,
                vector_end,
                width,
                total,
                ..
            } => {
                pred = self.emit_range_loop(
                    ir,
                    &format!("{label}.head"),
                    &pred,
                    "0",
                    &vector_start.to_string(),
                    Mode::Scalar(self.precision),
                    src,
                    target_ptr,
                    bufs,
                    counter,
                    &result_shape,
                None,
                )?;
                pred = self.emit_range_loop(
                    ir,
                    &format!("{label}.vec"),
                    &pred,
                    &vector_start.to_string(),
                    &vector_end.to_string(),
                    Mode::Vector(width, self.precision),
                    src,
                    target_ptr,
                    bufs,
                    counter,
                    &result_shape,
                None,
                )?;
                pred = self.emit_range_loop(
                    ir,
                    &format!("{label}.tail"),
                    &pred,
                    &vector_end.to_string(),
                    &total.to_string(),
                    Mode::Scalar(self.precision),
                    src,
                    target_ptr,
                    bufs,
                    counter,
                    &result_shape,
                None,
                )?;
            }
        }
        Ok(pred)
    }

    /// `expr ⇒ target`: the sweep of `expr`, repeated until no cell moves by
    /// more than 𝜏 or the cap is reached.
    ///
    /// Each round sweeps into a second buffer, then measures the largest move
    /// while copying back. So every read inside the sweep sees the finished
    /// previous round — the rule `→` gives a shift, and what keeps the vector
    /// path safe here — and the round is a Jacobi step, never a Gauss–Seidel
    /// one. The buffers the loop needs are allocated once, ahead of it.
    #[allow(clippy::too_many_arguments)]
    fn emit_iterate(
        &self,
        prelude: &[Statement],
        src: &Expr,
        target: &str,
        ir: &mut String,
        bufs: &mut Buffers,
        pred: &str,
        loop_id: usize,
        counter: &mut usize,
    ) -> Result<String> {
        // The sweeps emitted here run every round, which `explain` says.
        *self.in_loop.borrow_mut() = Some(target.to_string());
        let emitted = self.emit_iterate_rounds(prelude, src, target, ir, bufs, pred, loop_id, counter);
        *self.in_loop.borrow_mut() = None;
        emitted
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_iterate_rounds(
        &self,
        prelude: &[Statement],
        src: &Expr,
        target: &str,
        ir: &mut String,
        bufs: &mut Buffers,
        pred: &str,
        loop_id: usize,
        counter: &mut usize,
    ) -> Result<String> {
        let cap = self.max_sweeps.ok_or_else(|| HarmonyDisruption::LoweringErr {
            detail: "this program iterates (⇒); say how many sweeps it may take with --max-iter"
                .to_string(),
            line: self.current_line.get(),
        })?;
        let label = format!("it{loop_id}");
        let flow_target = FlowTarget::Var(target.to_string());
        let held = self.lookup(bufs, target)?;
        let cells = self
            .space_shapes
            .get(target)
            .map(|s| s.iter().product::<usize>())
            .unwrap_or(self.elements)
            .max(1);
        let length = self.sweep_length(src, &flow_target, bufs);
        let elem = self.precision.llvm_type();
        let align = self.precision.bytes();
        let suffix = self.precision.intrinsic_suffix();

        let next = self.emit_scratch(ir, &format!("{label}_next"), cells, &mut bufs.heap);
        if bufs.parallel {
            self.share(ir, &next);
        }
        // A prelude flow that reads its own space elsewhere than at the
        // cell it writes reads a copy taken before it, made here, outside
        // the rounds, so that the stack does not grow with them.
        let mut copies = BTreeMap::new();
        for (i, flow) in prelude.iter().enumerate() {
            if let Statement::Flow { src, target } = flow {
                self.reserve_fold_buffers(src, ir, bufs, counter)?;
                let written = self.written_names(target);
                if Self::reads_elsewhere(src, &written, false, &self.space_shapes) {
                    let copy = self.reserve_snapshot(ir, bufs, &written, &format!("{label}.p{i}"));
                    copies.insert(i, copy);
                }
            }
        }
        self.reserve_fold_buffers(src, ir, bufs, counter)?;

        ir.push_str(&format!(
            "  ; Iterate {loop_id}: sweep into [{target}] until no cell moves by more than {}, at most {cap} times\n",
            self.tau
        ));
        ir.push_str(&format!("  br label %{label}.header\n\n"));
        ir.push_str(&format!("{label}.header:\n"));
        ir.push_str(&format!(
            "  %{label}.k = phi i64 [ 0, %{pred} ], [ %{label}.k.next, %{label}.decide ]\n"
        ));
        // With the pool, the round reads one grid and writes the other, and
        // the two swap roles each round through the pointer table the parts
        // read: no copy back. The sweep measures the largest move as it
        // writes. (Inline, the round is copied back and measured on the way.)
        let (slot_held, slot_next) = if bufs.parallel {
            let slots = self.slots.borrow();
            (slots[&held], slots[&next])
        } else {
            (0, 0)
        };
        if bufs.parallel {
            ir.push_str(&format!(
                "  %{label}.cur = phi ptr [ {held}, %{pred} ], [ %{label}.nxt, %{label}.decide ]\n"
            ));
            ir.push_str(&format!(
                "  %{label}.nxt = phi ptr [ {next}, %{pred} ], [ %{label}.cur, %{label}.decide ]\n"
            ));
            let at = self.ctx_slot(ir, slot_held);
            ir.push_str(&format!("  store ptr %{label}.cur, ptr {at}\n"));
            let at = self.ctx_slot(ir, slot_next);
            ir.push_str(&format!("  store ptr %{label}.nxt, ptr {at}\n"));
        }
        // The prelude's flows first, every round, each a sweep of its own.
        let mut inner_pred = format!("{label}.header");
        for (i, flow) in prelude.iter().enumerate() {
            let Statement::Flow {
                src: flow_src,
                target: FlowTarget::Var(t),
            } = flow
            else {
                continue;
            };
            let ptr = self.lookup(bufs, t)?;
            let plabel = format!("{label}.p{i}");
            let guarded = self.fused.contains_key(t).then(|| {
                let (run, skip) = self.open_given(ir, t, &plabel);
                inner_pred = run;
                skip
            });
            let written = vec![t.clone()];
            let was = copies.get(&i).map(|copy| self.take_snapshot(ir, bufs, &written, &ptr, copy));
            inner_pred = self.emit_sweep(
                flow_src,
                &FlowTarget::Var(t.clone()),
                &ptr,
                ir,
                bufs,
                &inner_pred,
                &plabel,
                &format!("Iterate {loop_id} prelude {i}"),
                counter,
                None,
            )?;
            if let Some(was) = was {
                bufs.map.extend(was);
            }
            if let Some(skip) = guarded {
                inner_pred = Self::close_given(ir, &skip);
            }
        }
        let after_sweep = self.emit_sweep(
            src,
            &flow_target,
            &next,
            ir,
            bufs,
            &inner_pred,
            &format!("{label}.sweep"),
            &format!("Iterate {loop_id} round"),
            counter,
            if bufs.parallel { Some(&held) } else { None },
        )?;

        // The largest move of the round. A NaN move never compares greater,
        // the same rule the reference interpreter follows.
        let (settled_from, largest) = if bufs.parallel {
            // Each part of the sweep left its largest move in its slot; the
            // round's is the largest of those. A maximum is the same
            // whichever order it is taken in, so the answer is the one
            // thread's answer.
            ir.push_str(&format!("  %{label}.parts = call i64 @rho_rt_parts()\n"));
            ir.push_str(&format!("  br label %{label}.red\n\n"));
            ir.push_str(&format!("{label}.red:\n"));
            ir.push_str(&format!(
                "  %{label}.p = phi i64 [ 0, %{after_sweep} ], [ %{label}.p.next, %{label}.red.body ]\n"
            ));
            ir.push_str(&format!(
                "  %{label}.d = phi {elem} [ {}, %{after_sweep} ], [ %{label}.d.next, %{label}.red.body ]\n",
                self.f64_literal(0.0)
            ));
            ir.push_str(&format!(
                "  %{label}.more = icmp ult i64 %{label}.p, %{label}.parts\n"
            ));
            ir.push_str(&format!(
                "  br i1 %{label}.more, label %{label}.red.body, label %{label}.decide\n\n"
            ));
            ir.push_str(&format!("{label}.red.body:\n"));
            ir.push_str(&format!(
                "  %{label}.pp = getelementptr inbounds {elem}, ptr %rho.partial.p, i64 %{label}.p\n"
            ));
            ir.push_str(&format!("  %{label}.pd = load {elem}, ptr %{label}.pp, align {align}\n"));
            ir.push_str(&format!(
                "  %{label}.grew = fcmp ogt {elem} %{label}.pd, %{label}.d\n"
            ));
            ir.push_str(&format!(
                "  %{label}.d.next = select i1 %{label}.grew, {elem} %{label}.pd, {elem} %{label}.d\n"
            ));
            ir.push_str(&format!("  %{label}.p.next = add i64 %{label}.p, 1\n"));
            ir.push_str(&format!("  br label %{label}.red\n\n"));
            (format!("{label}.red"), format!("%{label}.d"))
        } else {
            let (end, d) =
                self.emit_compare_loop(ir, &label, &after_sweep, "0", &length, &next, &held);
            (end, d)
        };
        let _ = suffix;

        // The cap decides on its own; the tolerance is the one decision in a
        // kernel that depends on the data.
        ir.push_str(&format!("{label}.decide:\n"));
        ir.push_str(&format!("  %{label}.k.next = add i64 %{label}.k, 1\n"));
        ir.push_str(&format!(
            "  %{label}.capped = icmp uge i64 %{label}.k.next, {cap}\n"
        ));
        ir.push_str(&format!(
            "  %{label}.settled = fcmp ole {elem} {largest}, {}\n",
            self.f64_literal(self.tau)
        ));
        let _ = settled_from;
        ir.push_str(&format!(
            "  %{label}.done = or i1 %{label}.settled, %{label}.capped\n"
        ));
        ir.push_str(&format!(
            "  br i1 %{label}.done, label %{label}.end, label %{label}.header\n\n"
        ));

        // Record what happened: the sweeps taken, and whether the loop left
        // on the tolerance. Stopping at the cap counts as not converged even
        // if that last sweep happened to settle — the kernel did not check.
        ir.push_str(&format!("{label}.end:\n"));
        let mut landing = format!("{label}.end");
        if bufs.parallel {
            // The last round wrote one of the two grids; the target's own
            // buffer gets it if it was the other, once, and the table goes
            // back to naming the buffers as the rest of the body knows them.
            ir.push_str(&format!("  %{label}.moved = icmp ne ptr %{label}.nxt, {held}\n"));
            ir.push_str(&format!(
                "  br i1 %{label}.moved, label %{label}.copy, label %{label}.fin\n\n"
            ));
            ir.push_str(&format!("{label}.copy:\n"));
            ir.push_str(&format!(
                "  call void @llvm.memcpy.p0.p0.i64(ptr align {align} {held}, ptr align {align} %{label}.nxt, i64 {}, i1 false)\n",
                cells * self.precision.bytes()
            ));
            ir.push_str(&format!("  br label %{label}.fin\n\n"));
            ir.push_str(&format!("{label}.fin:\n"));
            let at = self.ctx_slot(ir, slot_held);
            ir.push_str(&format!("  store ptr {held}, ptr {at}\n"));
            let at = self.ctx_slot(ir, slot_next);
            ir.push_str(&format!("  store ptr {next}, ptr {at}\n"));
            landing = format!("{label}.fin");
        }
        ir.push_str(&format!("  %{label}.s0 = load atomic i64, ptr @rho_sweeps monotonic, align 8\n"));
        ir.push_str(&format!("  %{label}.s1 = add i64 %{label}.s0, %{label}.k.next\n"));
        ir.push_str(&format!("  store atomic i64 %{label}.s1, ptr @rho_sweeps monotonic, align 8\n"));
        ir.push_str(&format!("  %{label}.c0 = load atomic i64, ptr @rho_converged monotonic, align 8\n"));
        ir.push_str(&format!(
            "  %{label}.c1 = select i1 %{label}.capped, i64 0, i64 %{label}.c0\n"
        ));
        ir.push_str(&format!("  store atomic i64 %{label}.c1, ptr @rho_converged monotonic, align 8\n"));
        Ok(landing)
    }

    /// The loop that copies a round back over [lo, hi) and measures the
    /// largest move on the way. Control leaves for `{label}.decide` when
    /// inline; returns the label it branches to on exit and the symbol
    /// holding the largest move there.
    #[allow(clippy::too_many_arguments)]
    fn emit_compare_loop(
        &self,
        ir: &mut String,
        label: &str,
        pred: &str,
        lo: &str,
        hi: &str,
        next: &str,
        held: &str,
    ) -> (String, String) {
        let elem = self.precision.llvm_type();
        let align = self.precision.bytes();
        let suffix = self.precision.intrinsic_suffix();
        let exit = if pred == "entry" {
            format!("{label}.cmp.end")
        } else {
            format!("{label}.decide")
        };
        ir.push_str(&format!("  br label %{label}.cmp\n\n"));
        ir.push_str(&format!("{label}.cmp:\n"));
        ir.push_str(&format!(
            "  %{label}.i = phi i64 [ {lo}, %{pred} ], [ %{label}.i.next, %{label}.cmp.body ]\n"
        ));
        ir.push_str(&format!(
            "  %{label}.d = phi {elem} [ {}, %{pred} ], [ %{label}.d.next, %{label}.cmp.body ]\n",
            self.f64_literal(0.0)
        ));
        ir.push_str(&format!("  %{label}.more = icmp ult i64 %{label}.i, {hi}\n"));
        ir.push_str(&format!(
            "  br i1 %{label}.more, label %{label}.cmp.body, label %{exit}\n\n"
        ));
        ir.push_str(&format!("{label}.cmp.body:\n"));
        ir.push_str(&format!(
            "  %{label}.np = getelementptr inbounds {elem}, ptr {next}, i64 %{label}.i\n"
        ));
        ir.push_str(&format!("  %{label}.new = load {elem}, ptr %{label}.np, align {align}\n"));
        ir.push_str(&format!(
            "  %{label}.op = getelementptr inbounds {elem}, ptr {held}, i64 %{label}.i\n"
        ));
        ir.push_str(&format!("  %{label}.old = load {elem}, ptr %{label}.op, align {align}\n"));
        ir.push_str(&format!(
            "  %{label}.diff = fsub {elem} %{label}.new, %{label}.old\n"
        ));
        ir.push_str(&format!(
            "  %{label}.mag = call {elem} @llvm.fabs.{suffix}({elem} %{label}.diff)\n"
        ));
        ir.push_str(&format!(
            "  %{label}.grew = fcmp ogt {elem} %{label}.mag, %{label}.d\n"
        ));
        ir.push_str(&format!(
            "  %{label}.d.next = select i1 %{label}.grew, {elem} %{label}.mag, {elem} %{label}.d\n"
        ));
        ir.push_str(&format!(
            "  store {elem} %{label}.new, ptr %{label}.op, align {align}\n"
        ));
        ir.push_str(&format!("  %{label}.i.next = add i64 %{label}.i, 1\n"));
        ir.push_str(&format!("  br label %{label}.cmp\n\n"));
        (exit, format!("%{label}.d"))
    }

    /// Allocate the buffer every fold and scan in `expr` will write, so a loop
    /// around the expression allocates once rather than on every round.
    fn reserve_fold_buffers(
        &self,
        expr: &Expr,
        ir: &mut String,
        bufs: &mut Buffers,
        counter: &mut usize,
    ) -> Result<()> {
        match expr {
            Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } | Expr::Call { .. } => {}
            Expr::AuditTrace(inner)
            | Expr::Shift { operand: inner, .. }
            | Expr::Rotate { operand: inner, .. }
            | Expr::Reverse { operand: inner, .. }
            | Expr::Reshape { operand: inner, .. }
            | Expr::Transpose { operand: inner, .. }
            | Expr::Take { operand: inner, .. }
            | Expr::Drop { operand: inner, .. }
            | Expr::Builtin { operand: inner, .. }
            | Expr::Lift { operand: inner, .. } => {
                self.reserve_fold_buffers(inner, ir, bufs, counter)?;
            }
            Expr::Gather { index, operand } => {
                self.reserve_fold_buffers(index, ir, bufs, counter)?;
                self.reserve_fold_buffers(operand, ir, bufs, counter)?;
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                self.reserve_fold_buffers(lhs, ir, bufs, counter)?;
                self.reserve_fold_buffers(rhs, ir, bufs, counter)?;
            }
            Expr::Reduce { axis, operand, .. } | Expr::Scan { axis, operand, .. } => {
                self.reserve_fold_buffers(operand, ir, bufs, counter)?;
                let running = matches!(expr, Expr::Scan { .. });
                let cells = self.fold_cells(*axis, operand, running)?;
                *counter += 1;
                let buffer = self.emit_scratch(
                    ir,
                    &format!("fold{}_buf", *counter),
                    cells,
                    &mut bufs.heap,
                );
                if bufs.parallel {
                    self.share(ir, &buffer);
                }
                bufs.folds.insert(expr as *const Expr as usize, buffer);
            }
        }
        Ok(())
    }

    /// How many cells a fold (or, `running`, a scan) of `operand` along `axis`
    /// writes.
    fn fold_cells(&self, axis: Option<usize>, operand: &Expr, running: bool) -> Result<usize> {
        let shape = Self::expr_shape(operand, &self.space_shapes).ok_or_else(|| {
            HarmonyDisruption::LoweringErr {
                line: self.current_line.get(),
                detail: "a fold needs an operand with a known shape".to_string(),
            }
        })?;
        let a = axis.unwrap_or_else(|| crate::ast::default_axis(&shape));
        if a >= shape.len() {
            return Err(HarmonyDisruption::LoweringErr {
                line: self.current_line.get(),
                detail: format!("axis {a} is past the end of {shape:?}"),
            });
        }
        let extent = shape[a].max(1);
        let inner: usize = shape[a + 1..].iter().product::<usize>().max(1);
        let outer: usize = shape[..a].iter().product::<usize>().max(1);
        Ok(if running {
            (outer * extent * inner).max(1)
        } else {
            (outer * inner).max(1)
        })
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
            Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } | Expr::Call { .. } => {}
            Expr::AuditTrace(inner)
            | Expr::Shift { operand: inner, .. }
            | Expr::Rotate { operand: inner, .. }
            | Expr::Reverse { operand: inner, .. }
            | Expr::Reshape { operand: inner, .. }
            | Expr::Transpose { operand: inner, .. }
            | Expr::Take { operand: inner, .. }
            | Expr::Drop { operand: inner, .. }
            | Expr::Builtin { operand: inner, .. }
            | Expr::Lift { operand: inner, .. } => {
                block = self.emit_fold_prepass(inner, ir, bufs, &block, counter)?;
            }
            Expr::Gather { index, operand } => {
                block = self.emit_fold_prepass(index, ir, bufs, &block, counter)?;
                block = self.emit_fold_prepass(operand, ir, bufs, &block, counter)?;
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                block = self.emit_fold_prepass(lhs, ir, bufs, &block, counter)?;
                block = self.emit_fold_prepass(rhs, ir, bufs, &block, counter)?;
            }
            Expr::Reduce { op, axis, operand } => {
                block = self.emit_fold_prepass(operand, ir, bufs, &block, counter)?;
                block =
                    self.emit_fold(expr, *op, *axis, operand, ir, bufs, &block, counter, false)?;
            }
            Expr::Scan { op, axis, operand } => {
                block = self.emit_fold_prepass(operand, ir, bufs, &block, counter)?;
                block =
                    self.emit_fold(expr, *op, *axis, operand, ir, bufs, &block, counter, true)?;
            }
        }
        Ok(block)
    }

    /// One fold: an outer loop over the cells that survive, and an inner loop
    /// walking the axis being collapsed. The lines are independent, so the
    /// outer loop is what a part runs: on the pool, each thread takes a
    /// range of surviving cells.
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
        running: bool,
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
        // A fold writes one cell per line; a scan writes every cell it walks.
        let out_cells = if running {
            (outer * extent * inner).max(1)
        } else {
            (outer * inner).max(1)
        };
        let lines = (outer * inner).max(1);

        *counter += 1;
        let id = *counter;
        let label = format!("fold{id}");
        let key = node as *const Expr as usize;
        let buffer = match bufs.folds.get(&key) {
            // Reserved ahead of a `⇒` loop, so the loop allocates once.
            Some(reserved) => reserved.clone(),
            None => {
                let b = self.emit_scratch(ir, &format!("{label}_buf"), out_cells, &mut bufs.heap);
                if bufs.parallel {
                    self.share(ir, &b);
                }
                b
            }
        };
        let kind = if running { "running" } else { "total" };
        ir.push_str(&format!(
            "  ; {op} ({kind}) over axis {a} of {shape:?} -> {out_cells} cells\n"
        ));

        let geometry = FoldGeometry {
            op,
            running,
            extent,
            inner,
            axis: a,
            shape: &shape,
        };
        let landed = if bufs.parallel {
            let (name, mut f, inside, translated) = self.open_part(bufs, &[&buffer]);
            self.emit_fold_lines(
                &mut f,
                &inside,
                &label,
                "entry",
                "%lo",
                "%hi",
                &translated[&buffer],
                operand,
                &geometry,
                counter,
            )?;
            f.push_str(&format!("{label}.end:\n"));
            self.close_part(ir, &name, f, &lines.to_string());
            pred.to_string()
        } else {
            self.emit_fold_lines(
                ir,
                bufs,
                &label,
                pred,
                "0",
                &lines.to_string(),
                &buffer,
                operand,
                &geometry,
                counter,
            )?;
            ir.push_str(&format!("{label}.end:\n"));
            format!("{label}.end")
        };

        bufs.folds.insert(key, buffer);
        Ok(landed)
    }

    /// The loops of one fold over the surviving cells `[lo, hi)`. Control
    /// leaves for `{label}.end`, which the caller lays down.
    #[allow(clippy::too_many_arguments)]
    fn emit_fold_lines(
        &self,
        ir: &mut String,
        bufs: &Buffers,
        label: &str,
        pred: &str,
        lo: &str,
        hi: &str,
        buffer: &str,
        operand: &Expr,
        g: &FoldGeometry,
        counter: &mut usize,
    ) -> Result<()> {
        // The lines of a fold are independent, and the chain of one line's
        // steps is what bounds its speed (an fadd waits on the last). So
        // W lines are walked together, each with its own accumulator: the
        // same steps in the same order per line, hence the same bits as
        // one at a time, with W chains in flight instead of one. Lines
        // past the last whole group of W are walked one at a time.
        let w = self.lanes().max(1);
        let group = format!("{label}.g");
        ir.push_str(&format!("  %{label}.span = sub i64 {hi}, {lo}\n"));
        ir.push_str(&format!("  %{label}.whole = udiv i64 %{label}.span, {w}\n"));
        ir.push_str(&format!("  %{label}.grouped = mul i64 %{label}.whole, {w}\n"));
        ir.push_str(&format!("  %{label}.mid = add i64 {lo}, %{label}.grouped\n"));
        self.emit_fold_group(ir, bufs, &group, pred, lo, &format!("%{label}.mid"), buffer, operand, g, w, counter)?;
        ir.push_str(&format!("{group}.end:\n"));
        let single = format!("{label}.s");
        self.emit_fold_group(ir, bufs, &single, &format!("{group}.end"), &format!("%{label}.mid"), hi, buffer, operand, g, 1, counter)?;
        ir.push_str(&format!("{single}.end:\n"));
        ir.push_str(&format!("  br label %{label}.end\n\n"));
        Ok(())
    }

    /// The loops that walk `w` lines at a time over the surviving cells
    /// `[lo, hi)`, which the caller makes a multiple of `w` long. Control
    /// leaves for `{label}.end`, which the caller lays down.
    #[allow(clippy::too_many_arguments)]
    fn emit_fold_group(
        &self,
        ir: &mut String,
        bufs: &Buffers,
        label: &str,
        pred: &str,
        lo: &str,
        hi: &str,
        buffer: &str,
        operand: &Expr,
        g: &FoldGeometry,
        w: usize,
        counter: &mut usize,
    ) -> Result<()> {
        let elem = self.precision.llvm_type();
        let align = self.precision.bytes();
        let (op, running, extent, inner) = (g.op, g.running, g.extent, g.inner);
        ir.push_str(&format!("  br label %{label}.header\n\n"));

        // Outer loop: one iteration per group of w surviving cells.
        ir.push_str(&format!("{label}.header:\n"));
        ir.push_str(&format!(
            "  %{label}.j = phi i64 [ {lo}, %{pred} ], [ %{label}.j.next, %{label}.tail ]\n"
        ));
        ir.push_str(&format!("  %{label}.go = icmp ult i64 %{label}.j, {hi}\n"));
        ir.push_str(&format!(
            "  br i1 %{label}.go, label %{label}.body, label %{label}.end\n\n"
        ));

        // Map each surviving index back to where its line starts, and work
        // out the line's coordinates once: those before the axis from the
        // outer index, those after it from the inner one. The axis's own
        // coordinate is the step.
        let axis = g.axis;
        let outer_strides = crate::ast::strides_of(&g.shape[..axis]);
        let inner_strides = crate::ast::strides_of(&g.shape[axis + 1..]);
        ir.push_str(&format!("{label}.body:\n"));
        for l in 0..w {
            ir.push_str(&format!("  %{label}.j{l} = add i64 %{label}.j, {l}\n"));
            ir.push_str(&format!("  %{label}.o{l} = udiv i64 %{label}.j{l}, {inner}\n"));
            ir.push_str(&format!("  %{label}.t{l} = urem i64 %{label}.j{l}, {inner}\n"));
            ir.push_str(&format!(
                "  %{label}.block{l} = mul i64 %{label}.o{l}, {}\n",
                extent * inner
            ));
            ir.push_str(&format!(
                "  %{label}.base{l} = add i64 %{label}.block{l}, %{label}.t{l}\n"
            ));
            // The outermost axis of each run needs no remainder: the quotient
            // is already below its extent, and saying so (rather than leaving
            // a urem for the optimiser to see through) is what lets the
            // lanes' reads be seen as consecutive.
            for (k, stride) in outer_strides.iter().enumerate() {
                ir.push_str(&format!("  %{label}.c{l}_{k}q = udiv i64 %{label}.o{l}, {stride}\n"));
                if k == 0 {
                    ir.push_str(&format!("  %{label}.c{l}_{k} = add i64 %{label}.c{l}_{k}q, 0\n"));
                } else {
                    ir.push_str(&format!("  %{label}.c{l}_{k} = urem i64 %{label}.c{l}_{k}q, {}\n", g.shape[k]));
                }
            }
            for (k, stride) in inner_strides.iter().enumerate() {
                let ax = axis + 1 + k;
                ir.push_str(&format!("  %{label}.c{l}_{ax}q = udiv i64 %{label}.t{l}, {stride}\n"));
                if k == 0 {
                    ir.push_str(&format!("  %{label}.c{l}_{ax} = add i64 %{label}.c{l}_{ax}q, 0\n"));
                } else {
                    ir.push_str(&format!("  %{label}.c{l}_{ax} = urem i64 %{label}.c{l}_{ax}q, {}\n", g.shape[ax]));
                }
            }
        }
        ir.push_str(&format!("  br label %{label}.inner\n\n"));

        // Inner loop: walk the axis, folding each line as it goes.
        ir.push_str(&format!("{label}.inner:\n"));
        ir.push_str(&format!(
            "  %{label}.m = phi i64 [ 0, %{label}.body ], [ %{label}.m.next, %{label}.step ]\n"
        ));
        for l in 0..w {
            ir.push_str(&format!(
                "  %{label}.acc{l} = phi {elem} [ {}, %{label}.body ], [ %{label}.acc{l}.next, %{label}.step ]\n",
                self.f64_literal(op.identity())
            ));
        }
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
        for l in 0..w {
            ir.push_str(&format!(
                "  %{label}.at{l} = add i64 %{label}.base{l}, %{label}.off\n"
            ));
            let axes: Vec<String> = (0..g.shape.len())
                .map(|k| if k == axis { format!("%{label}.m") } else { format!("%{label}.c{l}_{k}") })
                .collect();
            *self.coords.borrow_mut() = Some(CoordHint {
                idx: format!("%{label}.at{l}"),
                shape: g.shape.to_vec(),
                axes,
            });
            let element = self.emit_expr(
                operand,
                ir,
                bufs,
                &format!("%{label}.at{l}"),
                counter,
                Mode::Scalar(self.precision),
                g.shape,
                &[],
            );
            *self.coords.borrow_mut() = None;
            let element = element?;
            let acc = format!("%{label}.acc{l}");
            match op {
                FoldOp::Sum => ir.push_str(&format!(
                    "  {acc}.next = fadd {elem} {acc}, {element}\n"
                )),
                FoldOp::Product => ir.push_str(&format!(
                    "  {acc}.next = fmul {elem} {acc}, {element}\n"
                )),
                FoldOp::Max | FoldOp::Min => {
                    let pred_op = if matches!(op, FoldOp::Max) { "ogt" } else { "olt" };
                    ir.push_str(&format!(
                        "  {acc}.win = fcmp {pred_op} {elem} {element}, {acc}\n"
                    ));
                    ir.push_str(&format!(
                        "  {acc}.next = select i1 {acc}.win, {elem} {element}, {elem} {acc}\n"
                    ));
                }
            }
            if running {
                // Every step of a scan is an answer, so it is stored as it goes.
                ir.push_str(&format!(
                    "  %{label}.here{l} = getelementptr inbounds {elem}, ptr {buffer}, i64 %{label}.at{l}\n"
                ));
                ir.push_str(&format!(
                    "  store {elem} {acc}.next, ptr %{label}.here{l}, align {align}\n"
                ));
            }
        }
        ir.push_str(&format!("  %{label}.m.next = add i64 %{label}.m, 1\n"));
        ir.push_str(&format!("  br label %{label}.inner\n\n"));

        ir.push_str(&format!("{label}.tail:\n"));
        if !running {
            for l in 0..w {
                ir.push_str(&format!(
                    "  %{label}.slot{l} = getelementptr inbounds {elem}, ptr {buffer}, i64 %{label}.j{l}\n"
                ));
                ir.push_str(&format!(
                    "  store {elem} %{label}.acc{l}, ptr %{label}.slot{l}, align {align}\n"
                ));
            }
        }
        ir.push_str(&format!("  %{label}.j.next = add i64 %{label}.j, {w}\n"));
        ir.push_str(&format!("  br label %{label}.header\n\n"));
        Ok(())
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
        measure: Option<&Measure>,
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
        if let Some(m) = measure {
            // The move this cell makes, kept if it is the largest so far. A
            // NaN move never compares greater, as in the reference.
            let ty = mode.ty();
            let suffix = match mode {
                Mode::Scalar(_) => self.precision.intrinsic_suffix().to_string(),
                Mode::Vector(w, _) => format!("v{w}{}", self.precision.intrinsic_suffix()),
            };
            let acc = match mode {
                Mode::Scalar(_) => &m.acc,
                Mode::Vector(..) => &m.acc_vec,
            };
            let (og, old, diff, mag, was, grew, now) = (
                Self::fresh(counter),
                Self::fresh(counter),
                Self::fresh(counter),
                Self::fresh(counter),
                Self::fresh(counter),
                Self::fresh(counter),
                Self::fresh(counter),
            );
            ir.push_str(&format!(
                "  {og} = getelementptr inbounds {}, ptr {}, i64 {idx}\n",
                self.precision.llvm_type(),
                m.cur
            ));
            ir.push_str(&format!("  {old} = load {ty}, ptr {og}, align {}\n", self.precision.bytes()));
            ir.push_str(&format!("  {diff} = fsub {ty} {value}, {old}\n"));
            ir.push_str(&format!("  {mag} = call {ty} @llvm.fabs.{suffix}({ty} {diff})\n"));
            ir.push_str(&format!("  {was} = load {ty}, ptr {acc}\n"));
            ir.push_str(&format!("  {grew} = fcmp ogt {ty} {mag}, {was}\n"));
            ir.push_str(&format!("  {now} = select {} {grew}, {ty} {mag}, {ty} {was}\n", mode.bool_ty()));
            ir.push_str(&format!("  store {ty} {now}, ptr {acc}\n"));
        }
        let gep = Self::fresh(counter);
        ir.push_str(&format!(
            "  {gep} = getelementptr inbounds {}, ptr {target_ptr}, i64 {idx}\n",
            self.precision.llvm_type()
        ));
        ir.push_str(&format!(
            "  store {} {value}, ptr {gep}, align {}\n",
            mode.ty(),
            self.precision.bytes()
        ));
        ir.push_str(&format!("  {next} = add i64 {idx}, {step}\n"));
        ir.push_str(&format!("  br label %{header}\n\n"));

        ir.push_str(&format!("{end}:\n"));
        Ok(end)
    }

    /// The spaces some flow writes. `X → =` writes OUTPUT.
    fn written_spaces(block: &ToposBlock) -> BTreeSet<String> {
        block
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
                Statement::Iterate { target, .. } => Some(target.clone()),
                _ => None,
            })
            .chain(block.statements.iter().flat_map(|stmt| match stmt {
                Statement::Iterate { prelude, .. } => prelude
                    .iter()
                    .filter_map(|flow| match flow {
                        Statement::Flow {
                            target: FlowTarget::Var(t),
                            ..
                        } => Some(t.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            }))
            .collect()
    }

    /// The declared spaces no flow writes: what a caller has to supply.
    fn source_spaces(&self, block: &ToposBlock) -> Vec<String> {
        block
            .statements
            .iter()
            .filter_map(|stmt| match stmt {
                Statement::SpaceDef(d) => Some(d.name.clone()),
                Statement::ExtBind(b) => Some(b.space.name.clone()),
                _ => None,
            })
            .filter(|name| !self.written_spaces.contains(name))
            .collect()
    }

    /// The spaces a caller has to supply — those no flow writes — in the
    /// order `rho_kernel_exec_spaces` takes them. Meaningful once IR has been
    /// generated.
    pub fn input_spaces(&self) -> Vec<String> {
        self.space_shapes
            .keys()
            .filter(|name| self.role_of(name) == "input")
            .cloned()
            .collect()
    }

    /// What a caller does with a space: supply it, read the result from it,
    /// or leave it to the kernel.
    pub fn role_of(&self, name: &str) -> &'static str {
        if name == "OUTPUT" {
            "output"
        } else if self.written_spaces.contains(name) {
            "internal"
        } else {
            "input"
        }
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

    /// Decide how to split one flow's sweep. An aligned split starts its
    /// vector body at a multiple of the width.
    fn plan_sweep(&self, src: &Expr, bound: &str, aligned: bool) -> Result<Sweep> {
        // Vector loads read a fixed window, so the length has to be known here.
        let Ok(total) = bound.parse::<u64>() else {
            return Ok(Sweep::AllScalar);
        };
        if !self.simd {
            return Ok(Sweep::AllScalar);
        }

        let width = self.lanes() as u64;
        let reach = self.max_shift_stride(src)?;

        // The body may touch [idx - reach, idx + reach + width - 1].
        let start = if aligned { reach.div_ceil(width) * width } else { reach };
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
            aligned,
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
            // None of these reads a neighbour window: a fold has its own
            // buffer, an index reads nothing, and a turn keeps the sweep
            // scalar and addresses each cell itself.
            Expr::Reduce { .. }
            | Expr::Scan { .. }
            | Expr::Index { .. }
            | Expr::Rotate { .. }
            | Expr::Reverse { .. }
            | Expr::Reshape { .. }
            | Expr::Transpose { .. }
            | Expr::Take { .. }
            | Expr::Drop { .. } => 0,
            // A gather keeps the sweep scalar too; its positions may hold
            // shifts, whose reach is theirs.
            Expr::Gather { index, .. } => self.max_shift_stride(index)?,
            Expr::Call { name, .. } => {
                return Err(HarmonyDisruption::LoweringErr {
                    line: self.current_line.get(),
                    detail: format!("the call to `{name}` was not expanded"),
                })
            }
            Expr::Lift { operand, .. } | Expr::Builtin { operand, .. } => {
                self.max_shift_stride(operand)?
            }
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

    /// A literal exponent that is a whole number small enough to unroll.
    fn whole_exponent(rhs: &Expr) -> Option<i32> {
        crate::ast::whole_exponent(rhs)
    }

    /// `x^n` as repeated multiplication, with a reciprocal for a negative n.
    fn emit_integer_power(
        &self,
        base: &str,
        n: i32,
        ir: &mut String,
        counter: &mut usize,
        mode: Mode,
        out: &str,
    ) {
        let ty = mode.ty();
        if n == 0 {
            ir.push_str(&format!("  {out} = fadd {ty} {}, {}\n", mode.zero(), mode.splat(1.0)));
            return;
        }
        let mut acc = base.to_string();
        for _ in 1..n.abs() {
            let next = Self::fresh(counter);
            ir.push_str(&format!("  {next} = fmul {ty} {acc}, {base}\n"));
            acc = next;
        }
        if n < 0 {
            ir.push_str(&format!("  {out} = fdiv {ty} {}, {acc}\n", mode.splat(1.0)));
        } else {
            ir.push_str(&format!("  {out} = fadd {ty} {}, {acc}\n", mode.zero(), ));
        }
    }

    /// The LLVM predicate a comparison lowers to, if it is one.
    fn compare_predicate(op: &BinaryOpKind) -> Option<&'static str> {
        Some(match op {
            BinaryOpKind::Gt => "ogt",
            BinaryOpKind::Lt => "olt",
            BinaryOpKind::Gte => "oge",
            BinaryOpKind::Lte => "ole",
            BinaryOpKind::Eq => "oeq",
            _ => return None,
        })
    }

    /// A flag that holds where the operand is not zero.
    #[allow(clippy::too_many_arguments)]
    fn emit_nonzero(
        &self,
        operand: &Expr,
        ir: &mut String,
        bufs: &Buffers,
        idx: &str,
        counter: &mut usize,
        mode: Mode,
        result_shape: &[usize],
        lifts: &[usize],
    ) -> Result<String> {
        let value = self.emit_expr(operand, ir, bufs, idx, counter, mode, result_shape, lifts)?;
        let flag = Self::fresh(counter);
        // Unordered: NaN is not zero, so `ind` of NaN is 1 — what `!=` says
        // in C and in the reference interpreter. The ordered `one` is false
        // for NaN, and differential testing caught that 0 against the
        // interpreter's 1.
        ir.push_str(&format!(
            "  {flag} = fcmp une {} {value}, {}\n",
            mode.ty(),
            mode.zero()
        ));
        Ok(flag)
    }

    /// The shape of the buffer a fold or scan was precomputed into.
    fn precomputed_shape(&self, expr: &Expr) -> Vec<usize> {
        Self::expr_shape(expr, &self.space_shapes).unwrap_or_else(|| vec![self.elements])
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
        let known = self.known_coords(idx, result_shape);

        let mut terms: Vec<String> = Vec::new();
        for axis in 0..result_shape.len() {
            // A stretched axis holds the same value for every position along it.
            if source[axis] <= 1 {
                continue;
            }
            let coord = Self::fresh(counter);
            let scaled = Self::fresh(counter);
            if let Some(axes) = &known {
                ir.push_str(&format!("  {coord} = add i64 {}, 0\n", axes[axis]));
            } else if res_strides[axis] == 1 {
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
    /// Whether every stretched read in the expression still reads `width`
    /// consecutive cells of its source for `width` consecutive cells of the
    /// result, given a chunk that starts at a multiple of `width`. That holds
    /// when the axes the source shares with the result, counted in from the
    /// innermost, span a multiple of the width: a chunk then stays inside one
    /// run of them, and a stretched axis outside the run is the same for all
    /// its lanes. A table of one row stretched over k rows qualifies; a
    /// column stretched along a row does not, nor does a stretched read
    /// under a shift, whose edge is worked out for the unstretched shape.
    fn stretched_contiguously(
        &self,
        expr: &Expr,
        result_shape: &[usize],
        lifts: &[usize],
        width: usize,
        shifted: bool,
    ) -> bool {
        let fits = |view: Vec<usize>| -> bool {
            if view == result_shape {
                return true;
            }
            if shifted || view.len() != result_shape.len() {
                return false;
            }
            let mut run = 1usize;
            for (v, r) in view.iter().zip(result_shape).rev() {
                if v != r {
                    break;
                }
                run *= v;
            }
            run.is_multiple_of(width)
        };
        match expr {
            Expr::Number(_) => true,
            Expr::Var(name) => fits(Self::lifted_shape(&self.shape_for(name), lifts)),
            Expr::AuditTrace(inner) | Expr::Builtin { operand: inner, .. } => {
                self.stretched_contiguously(inner, result_shape, lifts, width, shifted)
            }
            Expr::Shift { operand, .. } => {
                self.stretched_contiguously(operand, result_shape, lifts, width, true)
            }
            Expr::Lift { axis, operand } => {
                let mut nested = lifts.to_vec();
                nested.push(*axis);
                self.stretched_contiguously(operand, result_shape, &nested, width, shifted)
            }
            Expr::Reduce { .. } | Expr::Scan { .. } => {
                fits(Self::lifted_shape(&self.precomputed_shape(expr), lifts))
            }
            Expr::Index { operand, .. } => match Self::expr_shape(operand, &self.space_shapes) {
                Some(shape) => fits(Self::lifted_shape(&shape, lifts)),
                None => true,
            },
            Expr::Reshape { shape, .. } => fits(Self::lifted_shape(shape, lifts)),
            Expr::BinaryOp { lhs, rhs, .. } => {
                self.stretched_contiguously(lhs, result_shape, lifts, width, shifted)
                    && self.stretched_contiguously(rhs, result_shape, lifts, width, shifted)
            }
            // Turns keep the sweep scalar on their own; a call is expanded.
            Expr::Rotate { .. }
            | Expr::Reverse { .. }
            | Expr::Transpose { .. }
            | Expr::Take { .. }
            | Expr::Drop { .. }
            | Expr::Call { .. } => false,
            Expr::Gather { index, .. } => {
                self.stretched_contiguously(index, result_shape, lifts, width, shifted)
            }
        }
    }

    fn needs_broadcast(&self, expr: &Expr, result_shape: &[usize], lifts: &[usize]) -> bool {
        match expr {
            Expr::Number(_) => false,
            Expr::Var(name) => Self::lifted_shape(&self.shape_for(name), lifts) != result_shape,
            Expr::AuditTrace(inner)
            | Expr::Shift { operand: inner, .. }
            | Expr::Rotate { operand: inner, .. }
            | Expr::Reverse { operand: inner, .. }
            | Expr::Builtin { operand: inner, .. } => {
                self.needs_broadcast(inner, result_shape, lifts)
            }
            // The space is read at an absolute position, never stretched;
            // the positions are read like anything else.
            Expr::Gather { index, .. } => self.needs_broadcast(index, result_shape, lifts),
            Expr::Lift { axis, operand } => {
                let mut nested = lifts.to_vec();
                nested.push(*axis);
                self.needs_broadcast(operand, result_shape, &nested)
            }
            // A fold or scan is read from its own buffer; whether that buffer
            // is stretched is decided by the read below, not by its contents.
            Expr::Reduce { .. } | Expr::Scan { .. } => {
                Self::lifted_shape(&self.precomputed_shape(expr), lifts) != result_shape
            }
            // An index is laid out like the operand it measures; a reshape
            // like the shape it was given.
            Expr::Index { operand, .. } => match Self::expr_shape(operand, &self.space_shapes) {
                Some(shape) => Self::lifted_shape(&shape, lifts) != result_shape,
                None => false,
            },
            Expr::Reshape { shape, .. } => Self::lifted_shape(shape, lifts) != result_shape,
            Expr::Transpose { .. } | Expr::Take { .. } | Expr::Drop { .. } => {
                match Self::expr_shape(expr, &self.space_shapes) {
                    Some(shape) => Self::lifted_shape(&shape, lifts) != result_shape,
                    None => false,
                }
            }
            Expr::Call { .. } => false,
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
                    "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {read_at}\n",
            self.precision.llvm_type()
                ));
                ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                Ok(val)
            }

            Expr::AuditTrace(inner) => {
                self.emit_expr(inner, ir, bufs, idx, counter, mode, result_shape, lifts)
            }

            Expr::Call { name, .. } => Err(HarmonyDisruption::LoweringErr {
                line: self.current_line.get(),
                detail: format!("the call to `{name}` was not expanded"),
            }),

            // `ind` of a comparison is that comparison's truth. Reading the
            // operand's value first would give the mask, which cannot tell a
            // blocked cell from one that passed a zero.
            Expr::Builtin {
                op: BuiltinOp::Indicator,
                operand,
            } => {
                let out = Self::fresh(counter);
                let flag = if let Expr::BinaryOp { op, lhs, rhs } = &**operand {
                    match Self::compare_predicate(op) {
                        Some(predicate) => {
                            let l = self
                                .emit_expr(lhs, ir, bufs, idx, counter, mode, result_shape, lifts)?;
                            let r = self
                                .emit_expr(rhs, ir, bufs, idx, counter, mode, result_shape, lifts)?;
                            let flag = Self::fresh(counter);
                            ir.push_str(&format!(
                                "  {flag} = fcmp {predicate} {ty} {l}, {r}\n"
                            ));
                            flag
                        }
                        None => self.emit_nonzero(operand, ir, bufs, idx, counter, mode, result_shape, lifts)?,
                    }
                } else {
                    self.emit_nonzero(operand, ir, bufs, idx, counter, mode, result_shape, lifts)?
                };
                ir.push_str(&format!(
                    "  {out} = select {} {flag}, {ty} {}, {ty} {}\n",
                    mode.bool_ty(),
                    mode.splat(1.0),
                    mode.zero()
                ));
                Ok(out)
            }

            Expr::Builtin { op, operand } => {
                let value =
                    self.emit_expr(operand, ir, bufs, idx, counter, mode, result_shape, lifts)?;
                let out = Self::fresh(counter);
                match op {
                    // A magnitude and a sign test are exact; the rest are calls.
                    BuiltinOp::Abs => ir.push_str(&format!(
                        "  {out} = call {ty} {}({ty} {value})\n",
                        mode.unary_intrinsic("fabs")
                    )),
                    BuiltinOp::Floor => ir.push_str(&format!(
                        "  {out} = call {ty} {}({ty} {value})\n",
                        mode.unary_intrinsic("floor")
                    )),
                    BuiltinOp::Ceil => ir.push_str(&format!(
                        "  {out} = call {ty} {}({ty} {value})\n",
                        mode.unary_intrinsic("ceil")
                    )),
                    BuiltinOp::Roll => {
                        self.emit_roll(ir, &value, &out, mode, counter);
                    }
                    BuiltinOp::Indicator => unreachable!("handled above"),
                    _ => ir.push_str(&format!(
                        "  {out} = call {ty} {}({ty} {value})\n",
                        mode.unary_intrinsic(match op {
                            BuiltinOp::Exp => "exp",
                            BuiltinOp::Log => "log",
                            BuiltinOp::Sqrt => "sqrt",
                            BuiltinOp::Sin => "sin",
                            _ => "cos",
                        })
                    )),
                }
                Ok(out)
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

            // The fold or scan ran before this sweep started; read its result.
            Expr::Reduce { .. } | Expr::Scan { .. } => {
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
                let view = Self::lifted_shape(&self.precomputed_shape(expr), lifts);
                let read_at = self.emit_index_map(ir, idx, &view, result_shape, counter)?;
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {read_at}\n",
            self.precision.llvm_type()
                ));
                ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                Ok(val)
            }

            // The cell `count` in from the start or the end of the axis, or
            // zero where a take runs past the source. Scalar (see `turns`):
            // the result's row length differs from the source's, so lanes
            // would straddle rows. 📋 A take or drop along the outermost axis
            // is a plain offset and could keep the vector path.
            Expr::Take { count, axis, operand } | Expr::Drop { count, axis, operand } => {
                let dropping = matches!(expr, Expr::Drop { .. });
                let glyph = if dropping { "↓" } else { "↑" };
                let name = Self::place_name(operand).ok_or_else(|| {
                    HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: format!("{glyph} takes from a declared space, so it cannot be applied to a computed value. Flow the sub-expression into its own space first."),
                    }
                })?;
                if matches!(mode, Mode::Vector(..)) {
                    return Err(HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "internal: a take or drop reached the vector path".to_string(),
                    });
                }
                let ptr = self.lookup(bufs, &name)?;
                let source_shape = self.shape_for(&name);
                let a = axis.unwrap_or_else(|| crate::ast::default_axis(&source_shape));
                let taken = crate::ast::taken_shape(&source_shape, Some(a), *count, dropping)
                    .ok_or_else(|| HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: format!("`{count} {glyph}` of {source_shape:?} leaves nothing"),
                    })?;
                let view = Self::lifted_shape(&taken, lifts);
                let read_at = self.emit_index_map(ir, idx, &view, result_shape, counter)?;
                let result_strides = crate::ast::strides_of(&taken);
                let source_strides = crate::ast::strides_of(&source_shape);
                let along = crate::ast::taken_offset(source_shape[a], *count, dropping);
                let extent = source_shape[a] as i64;

                // Coordinates of the result cell, the taken axis moved along.
                let mut acc: Option<String> = None;
                let mut padded: Option<String> = None;
                for b in 0..source_shape.len() {
                    if taken[b] <= 1 && b != a {
                        continue;
                    }
                    let coord = Self::fresh(counter);
                    if result_strides[b] == 1 {
                        ir.push_str(&format!("  {coord} = urem i64 {read_at}, {}\n", taken[b]));
                    } else {
                        let div = Self::fresh(counter);
                        ir.push_str(&format!("  {div} = udiv i64 {read_at}, {}\n", result_strides[b]));
                        ir.push_str(&format!("  {coord} = urem i64 {div}, {}\n", taken[b]));
                    }
                    let position = if b == a {
                        let moved = Self::fresh(counter);
                        ir.push_str(&format!("  {moved} = add i64 {coord}, {along}\n"));
                        // A take longer than the axis reads past it: zero there.
                        if !dropping && count.unsigned_abs() as usize > source_shape[a] {
                            let low = Self::fresh(counter);
                            let high = Self::fresh(counter);
                            let out = Self::fresh(counter);
                            ir.push_str(&format!("  {low} = icmp slt i64 {moved}, 0\n"));
                            ir.push_str(&format!("  {high} = icmp sge i64 {moved}, {extent}\n"));
                            ir.push_str(&format!("  {out} = or i1 {low}, {high}\n"));
                            padded = Some(out);
                        }
                        moved
                    } else {
                        coord
                    };
                    let scaled = Self::fresh(counter);
                    ir.push_str(&format!("  {scaled} = mul i64 {position}, {}\n", source_strides[b]));
                    acc = Some(match acc {
                        None => scaled,
                        Some(previous) => {
                            let next = Self::fresh(counter);
                            ir.push_str(&format!("  {next} = add i64 {previous}, {scaled}\n"));
                            next
                        }
                    });
                }
                let cell = acc.unwrap_or_else(|| "0".to_string());
                // Past the source, read cell 0 and discard it, so no address
                // is ever formed outside the buffer.
                let address = match &padded {
                    Some(flag) => {
                        let safe = Self::fresh(counter);
                        ir.push_str(&format!("  {safe} = select i1 {flag}, i64 0, i64 {cell}\n"));
                        safe
                    }
                    None => cell,
                };
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {address}\n",
                    self.precision.llvm_type()
                ));
                ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                match padded {
                    Some(flag) => {
                        let out = Self::fresh(counter);
                        ir.push_str(&format!(
                            "  {out} = select i1 {flag}, {ty} {}, {ty} {val}\n",
                            mode.zero()
                        ));
                        Ok(out)
                    }
                    None => Ok(val),
                }
            }

            // The cell whose coordinates are this cell's, permuted. The
            // sweep is scalar (see `turns`): each coordinate is taken apart
            // with the result's strides and put back with the source's.
            Expr::Transpose { axes, operand } => {
                let name = Self::place_name(operand).ok_or_else(|| {
                    HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "⍉ transposes a declared space, so it cannot be applied to a computed value. Flow the sub-expression into its own space first.".to_string(),
                    }
                })?;
                if matches!(mode, Mode::Vector(..)) {
                    return Err(HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "internal: a transpose reached the vector path".to_string(),
                    });
                }
                let ptr = self.lookup(bufs, &name)?;
                let source_shape = self.shape_for(&name);
                let perm = crate::ast::transpose_axes(source_shape.len(), axes.as_deref())
                    .ok_or_else(|| HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: format!("⍉ needs a permutation of the axes of {source_shape:?}"),
                    })?;
                let transposed = crate::ast::transposed_shape(&source_shape, Some(&perm))
                    .unwrap_or_else(|| source_shape.clone());
                let view = Self::lifted_shape(&transposed, lifts);
                let read_at = self.emit_index_map(ir, idx, &view, result_shape, counter)?;
                let result_strides = crate::ast::strides_of(&transposed);
                let source_strides = crate::ast::strides_of(&source_shape);
                let mut acc: Option<String> = None;
                for k in 0..source_shape.len() {
                    let a = perm[k];
                    if transposed[a] <= 1 {
                        continue;
                    }
                    let coord = Self::fresh(counter);
                    if result_strides[a] == 1 {
                        ir.push_str(&format!(
                            "  {coord} = urem i64 {read_at}, {}\n",
                            transposed[a]
                        ));
                    } else {
                        let div = Self::fresh(counter);
                        ir.push_str(&format!(
                            "  {div} = udiv i64 {read_at}, {}\n",
                            result_strides[a]
                        ));
                        ir.push_str(&format!("  {coord} = urem i64 {div}, {}\n", transposed[a]));
                    }
                    let scaled = Self::fresh(counter);
                    ir.push_str(&format!(
                        "  {scaled} = mul i64 {coord}, {}\n",
                        source_strides[k]
                    ));
                    acc = Some(match acc {
                        None => scaled,
                        Some(previous) => {
                            let next = Self::fresh(counter);
                            ir.push_str(&format!("  {next} = add i64 {previous}, {scaled}\n"));
                            next
                        }
                    });
                }
                let cell = acc.unwrap_or_else(|| "0".to_string());
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {cell}\n",
                    self.precision.llvm_type()
                ));
                ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                Ok(val)
            }

            // The operand's cells in row-major order, read into a new shape.
            // With the same count it is the operand's buffer read straight
            // through — contiguous, so the vector path stands; with a
            // different count the read goes round, and the sweep is scalar.
            Expr::Reshape { shape, operand } => {
                let name = Self::place_name(operand).ok_or_else(|| {
                    HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "⍴ reshapes a declared space, so it cannot be applied to a computed value. Flow the sub-expression into its own space first.".to_string(),
                    }
                })?;
                let ptr = self.lookup(bufs, &name)?;
                let count = self.shape_for(&name).iter().product::<usize>().max(1);
                let view = Self::lifted_shape(shape, lifts);
                let read_at = self.emit_index_map(ir, idx, &view, result_shape, counter)?;
                let cell = if shape.iter().product::<usize>().max(1) == count {
                    read_at
                } else {
                    if matches!(mode, Mode::Vector(..)) {
                        return Err(HarmonyDisruption::LoweringErr {
                            line: self.current_line.get(),
                            detail: "internal: a cycling reshape reached the vector path".to_string(),
                        });
                    }
                    let wrapped = Self::fresh(counter);
                    ir.push_str(&format!("  {wrapped} = urem i64 {read_at}, {count}\n"));
                    wrapped
                };
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {cell}\n",
                    self.precision.llvm_type()
                ));
                ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                Ok(val)
            }

            // A rotation reads the cell `by` further along the axis, wrapping
            // at its end; a reversal reads the cell at the other end. The
            // sweep is scalar here (see `turns`), so the cell is addressed
            // directly.
            Expr::Rotate { axis, operand, .. } | Expr::Reverse { axis, operand } => {
                let name = Self::place_name(operand).ok_or_else(|| {
                    HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "⌽ turns a declared space, so it cannot be applied to a computed value. Flow the sub-expression into its own space first.".to_string(),
                    }
                })?;
                if matches!(mode, Mode::Vector(..)) {
                    return Err(HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "internal: a turn reached the vector path".to_string(),
                    });
                }
                let ptr = self.lookup(bufs, &name)?;
                let shape = self.shape_for(&name);
                let (stride, extent) = self.axis_geometry(&shape, *axis)?;
                let view = Self::lifted_shape(&shape, lifts);
                let read_at = self.emit_index_map(ir, idx, &view, result_shape, counter)?;

                // Position along the axis, then the position read instead.
                let pos = Self::fresh(counter);
                if stride == 1 {
                    ir.push_str(&format!("  {pos} = urem i64 {read_at}, {extent}\n"));
                } else {
                    let q = Self::fresh(counter);
                    ir.push_str(&format!("  {q} = udiv i64 {read_at}, {stride}\n"));
                    ir.push_str(&format!("  {pos} = urem i64 {q}, {extent}\n"));
                }
                let target = Self::fresh(counter);
                match expr {
                    Expr::Rotate { by, .. } => {
                        let k = by.rem_euclid(extent as i64);
                        let ahead = Self::fresh(counter);
                        ir.push_str(&format!("  {ahead} = add i64 {pos}, {k}\n"));
                        ir.push_str(&format!("  {target} = urem i64 {ahead}, {extent}\n"));
                    }
                    _ => {
                        ir.push_str(&format!("  {target} = sub i64 {}, {pos}\n", extent - 1));
                    }
                }
                let back = Self::fresh(counter);
                let forward = Self::fresh(counter);
                let cell = Self::fresh(counter);
                ir.push_str(&format!("  {back} = mul i64 {pos}, {stride}\n"));
                ir.push_str(&format!("  {forward} = mul i64 {target}, {stride}\n"));
                let line_start = Self::fresh(counter);
                ir.push_str(&format!("  {line_start} = sub i64 {read_at}, {back}\n"));
                ir.push_str(&format!("  {cell} = add i64 {line_start}, {forward}\n"));
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {cell}\n",
                    self.precision.llvm_type()
                ));
                ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                Ok(val)
            }

            // The cell of the space at the position this cell's index names,
            // floored; zero when that is no cell of it. The position is
            // clamped before the load and the value chosen after, so an
            // index that is negative, past the end or NaN touches nothing.
            Expr::Gather { index, operand } => {
                let name = Self::place_name(operand).ok_or_else(|| {
                    HarmonyDisruption::LoweringErr {
                        line: self.current_line.get(),
                        detail: "⌷ reads a declared space, so its right side is a space's name, not a computed value. Flow the sub-expression into its own space first.".to_string(),
                    }
                })?;
                let ptr = self.lookup(bufs, &name)?;
                let count = self.shape_for(&name).iter().product::<usize>().max(1);
                // A position that is a whole-number combination of coordinates
                // (`⍳0N + ⍳1H`, `2 × ⍳0X - 1`) is an integer the interpreter
                // computes exactly in f64 and floors to itself, so it is
                // computed here in integers; and its range is known from the
                // axes' extents, so when it lies inside the space the read is
                // a plain load, with no test and no select per cell.
                if let Some(terms) = Self::affine_position(index, lifts) {
                    return self.emit_affine_gather(ir, &ptr, count, &terms, idx, counter, mode, result_shape);
                }
                let position =
                    self.emit_expr(index, ir, bufs, idx, counter, mode, result_shape, lifts)?;
                if let Mode::Vector(w, _) = mode {
                    return Ok(self.emit_vector_gather(ir, &ptr, count, &position, w, counter));
                }
                let elem = self.precision.llvm_type();
                let suffix = self.precision.intrinsic_suffix();
                let floored = Self::fresh(counter);
                let not_before = Self::fresh(counter);
                let not_past = Self::fresh(counter);
                let inside = Self::fresh(counter);
                let safe = Self::fresh(counter);
                let at = Self::fresh(counter);
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                let out = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {floored} = call {elem} @llvm.floor.{suffix}({elem} {position})\n"
                ));
                ir.push_str(&format!(
                    "  {not_before} = fcmp oge {elem} {floored}, {}\n",
                    self.f64_literal(0.0)
                ));
                ir.push_str(&format!(
                    "  {not_past} = fcmp olt {elem} {floored}, {}\n",
                    self.f64_literal(count as f64)
                ));
                ir.push_str(&format!("  {inside} = and i1 {not_before}, {not_past}\n"));
                ir.push_str(&format!(
                    "  {safe} = select i1 {inside}, {elem} {floored}, {elem} {}\n",
                    self.f64_literal(0.0)
                ));
                ir.push_str(&format!("  {at} = fptosi {elem} {safe} to i64\n"));
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds {elem}, ptr {ptr}, i64 {at}\n"
                ));
                ir.push_str(&format!(
                    "  {val} = load {elem}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                ir.push_str(&format!(
                    "  {out} = select i1 {inside}, {elem} {val}, {elem} {}\n",
                    self.f64_literal(0.0)
                ));
                Ok(out)
            }

            // The coordinate of the cell along one axis of the operand's
            // shape, from zero: the same arithmetic a shift uses to find its
            // boundary, turned into a number. The operand is measured, never
            // read.
            Expr::Index { axis, operand } => {
                let (pos, _) = self.emit_coordinate(ir, *axis, operand, idx, counter, mode, result_shape, lifts)?;
                let out = Self::fresh(counter);
                ir.push_str(&format!("  {out} = uitofp {} {pos} to {ty}\n", mode.int_ty()));
                Ok(out)
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
                    // A whole-number exponent is repeated multiplication, which
                    // is exact and does not depend on a maths library. Leaving
                    // it to pow() made the result differ by an ulp from the
                    // reference interpreter, because the two libm implementations
                    // round the square differently.
                    BinaryOpKind::Pow => match Self::whole_exponent(rhs) {
                        Some(n) => self.emit_integer_power(&l, n, ir, counter, mode, &out),
                        None => ir.push_str(&format!(
                            "  {out} = call {ty} {}({ty} {l}, {ty} {r})\n",
                            mode.pow_intrinsic()
                        )),
                    },
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
                    // The greater and the lesser: IEEE maximum and minimum, so
                    // a NaN propagates as it does through every other operation
                    // and -0 orders below +0. See the declarations for why a
                    // compare-and-select would not do.
                    BinaryOpKind::Max | BinaryOpKind::Min => {
                        let name = if matches!(op, BinaryOpKind::Max) {
                            "maximum"
                        } else {
                            "minimum"
                        };
                        ir.push_str(&format!(
                            "  {out} = call {ty} {}({ty} {l}, {ty} {r})\n",
                            mode.unary_intrinsic(name)
                        ));
                    }
                    // APL's residue, computed as written: B - A × ⌊B ÷ A⌋, with
                    // 0 | B being B. Every step rounds once, as the interpreter's
                    // does; `frem` would round differently and take the sign of
                    // B rather than of A.
                    BinaryOpKind::Residue => {
                        let quotient = Self::fresh(counter);
                        let floored = Self::fresh(counter);
                        let product = Self::fresh(counter);
                        let remainder = Self::fresh(counter);
                        let zero_modulus = Self::fresh(counter);
                        ir.push_str(&format!("  {quotient} = fdiv {ty} {r}, {l}\n"));
                        ir.push_str(&format!(
                            "  {floored} = call {ty} {}({ty} {quotient})\n",
                            mode.unary_intrinsic("floor")
                        ));
                        ir.push_str(&format!("  {product} = fmul {ty} {l}, {floored}\n"));
                        ir.push_str(&format!("  {remainder} = fsub {ty} {r}, {product}\n"));
                        ir.push_str(&format!(
                            "  {zero_modulus} = fcmp oeq {ty} {l}, {}\n",
                            mode.zero()
                        ));
                        ir.push_str(&format!(
                            "  {out} = select {} {zero_modulus}, {ty} {r}, {ty} {remainder}\n",
                            mode.bool_ty()
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

        // Along the outermost axis the boundary cells are the first and the
        // last line of the buffer, whole, and the planner keeps the vector
        // body a full line away from either end: no lane in it can be on
        // the edge, so the load needs no test. Every other axis has edges
        // inside the buffer and keeps the per-lane select.
        let outermost = stride * extent == shape.iter().product::<usize>().max(1);
        if let Mode::Vector(..) = mode {
            if outermost {
                let base = Self::fresh(counter);
                match dir {
                    ShiftDir::Positive => {
                        ir.push_str(&format!("  {base} = sub i64 {idx}, {stride}\n"));
                    }
                    ShiftDir::Negative => {
                        ir.push_str(&format!("  {base} = add i64 {idx}, {stride}\n"));
                    }
                }
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                ir.push_str(&format!(
                    "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {base}\n",
                    self.precision.llvm_type()
                ));
                ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
                return Ok(val);
            }
        }

        // Lane indices: idx for scalar, idx + <0,1,..,W-1> for vector.
        let lane_idx = match mode {
            Mode::Scalar(_) => idx.to_string(),
            Mode::Vector(w, _) => {
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
            Mode::Vector(..) => base.clone(),
            // A scalar step can sit on the very edge, and a bounded call may stop
            // short of the declared grid, so clamp before touching memory.
            Mode::Scalar(_) => {
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
                        "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {clamped}\n",
            self.precision.llvm_type()
                    ));
                    ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
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
            "  {gep} = getelementptr inbounds {}, ptr {ptr}, i64 {safe}\n",
            self.precision.llvm_type()
        ));
        ir.push_str(&format!(
                    "  {val} = load {ty}, ptr {gep}, align {}\n",
                    self.precision.bytes()
                ));
        ir.push_str(&format!(
            "  {out} = select {} {at_edge}, {ty} {}, {ty} {val}\n",
            mode.bool_ty(),
            mode.zero()
        ));

        Ok(out)
    }

    /// `?X`: splitmix64's finaliser over the operand's bits, at double
    /// width whatever the kernel's, the top 53 bits scaled into [0, 1).
    /// Integer arithmetic throughout, so every lane and every thread gets
    /// the number the interpreter gets. A NaN hashes as zero bits, since
    /// its payload is the one thing the two need not agree on.
    fn emit_roll(&self, ir: &mut String, value: &str, out: &str, mode: Mode, counter: &mut usize) {
        let (dbl, int, mask) = match mode {
            Mode::Scalar(_) => ("double".to_string(), "i64".to_string(), "i1".to_string()),
            Mode::Vector(w, _) => (
                format!("<{w} x double>"),
                format!("<{w} x i64>"),
                format!("<{w} x i1>"),
            ),
        };
        // A constant operand: bare for a scalar, one per lane for a vector.
        let splat = |elem: &str, v: String| match mode {
            Mode::Scalar(_) => v,
            Mode::Vector(w, _) => {
                let lanes: Vec<String> = (0..w).map(|_| format!("{elem} {v}")).collect();
                format!("<{}>", lanes.join(", "))
            }
        };
        let ty = mode.ty();
        let nan = Self::fresh(counter);
        let canon = Self::fresh(counter);
        ir.push_str(&format!("  {nan} = fcmp uno {ty} {value}, {value}\n"));
        ir.push_str(&format!(
            "  {canon} = select {mask} {nan}, {ty} {}, {ty} {value}\n",
            mode.zero()
        ));
        let wide = if matches!(self.precision, Precision::F32) {
            let w = Self::fresh(counter);
            ir.push_str(&format!("  {w} = fpext {ty} {canon} to {dbl}\n"));
            w
        } else {
            canon
        };
        let bits = Self::fresh(counter);
        ir.push_str(&format!("  {bits} = bitcast {dbl} {wide} to {int}\n"));
        let lit = |v: u64| splat("i64", (v as i64).to_string());
        let mut z = Self::fresh(counter);
        ir.push_str(&format!(
            "  {z} = add {int} {bits}, {}\n",
            lit(0x9E37_79B9_7F4A_7C15)
        ));
        for (shift, mul) in [
            (30u32, Some(0xBF58_476D_1CE4_E5B9u64)),
            (27, Some(0x94D0_49BB_1331_11EB)),
            (31, None),
        ] {
            let shifted = Self::fresh(counter);
            let mixed = Self::fresh(counter);
            ir.push_str(&format!(
                "  {shifted} = lshr {int} {z}, {}\n",
                splat("i64", shift.to_string())
            ));
            ir.push_str(&format!("  {mixed} = xor {int} {z}, {shifted}\n"));
            z = match mul {
                Some(m) => {
                    let product = Self::fresh(counter);
                    ir.push_str(&format!("  {product} = mul {int} {mixed}, {}\n", lit(m)));
                    product
                }
                None => mixed,
            };
        }
        let top = Self::fresh(counter);
        let real = Self::fresh(counter);
        let scaled = Self::fresh(counter);
        ir.push_str(&format!(
            "  {top} = lshr {int} {z}, {}\n",
            splat("i64", "11".to_string())
        ));
        ir.push_str(&format!("  {real} = uitofp {int} {top} to {dbl}\n"));
        ir.push_str(&format!(
            "  {scaled} = fmul {dbl} {real}, {}\n",
            splat(
                "double",
                format!("0x{:016X}", (1.0f64 / 9_007_199_254_740_992.0).to_bits())
            )
        ));
        if matches!(self.precision, Precision::F32) {
            ir.push_str(&format!("  {out} = fptrunc {dbl} {scaled} to {ty}\n"));
        } else {
            ir.push_str(&format!(
                "  {out} = fadd {dbl} {scaled}, {}\n",
                splat("double", "0.0".to_string())
            ));
        }
    }

    /// The coordinate of the cell along one axis of the operand's shape,
    /// from zero, as an integer of the mode's width — the same arithmetic a
    /// shift uses to find its boundary — and the axis's extent. The operand
    /// is measured, never read.
    #[allow(clippy::too_many_arguments)]
    fn emit_coordinate(
        &self,
        ir: &mut String,
        axis: Option<usize>,
        operand: &Expr,
        idx: &str,
        counter: &mut usize,
        mode: Mode,
        result_shape: &[usize],
        lifts: &[usize],
    ) -> Result<(String, usize)> {
        let shape = Self::expr_shape(operand, &self.space_shapes).ok_or_else(|| {
            HarmonyDisruption::LoweringErr {
                line: self.current_line.get(),
                detail: "⍳ needs an operand with a known shape".to_string(),
            }
        })?;
        let (stride, extent) = self.axis_geometry(&shape, axis)?;
        let int_ty = mode.int_ty();
        // The coordinate is known from the loops when the operand's shape
        // lines up with the result's axis for axis: the cell's own
        // coordinate there, or zero along an axis the operand is one wide.
        if let (Mode::Scalar(_), Some(axes)) = (mode, self.known_coords(idx, result_shape)) {
            let a = axis.unwrap_or_else(|| crate::ast::default_axis(&shape));
            if lifts.is_empty() && shape.len() == result_shape.len() && a < shape.len() {
                if shape[a] == result_shape[a] {
                    let pos = Self::fresh(counter);
                    ir.push_str(&format!("  {pos} = add i64 {}, 0\n", axes[a]));
                    return Ok((pos, extent));
                }
                if shape[a] == 1 {
                    return Ok(("0".to_string(), extent));
                }
            }
        }
        let view = Self::lifted_shape(&shape, lifts);
        let read_at = self.emit_index_map(ir, idx, &view, result_shape, counter)?;
        let lane_idx = match mode {
            Mode::Scalar(_) => read_at,
            Mode::Vector(w, _) => {
                let seed = Self::fresh(counter);
                let splat = Self::fresh(counter);
                let lanes = Self::fresh(counter);
                let offsets: Vec<String> = (0..w).map(|l| format!("i64 {l}")).collect();
                ir.push_str(&format!(
                    "  {seed} = insertelement {int_ty} poison, i64 {read_at}, i64 0\n"
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
        Ok((pos, extent))
    }

    /// The cell's coordinates when the loops around it have said what they
    /// are, for this index into this shape.
    fn known_coords(&self, idx: &str, shape: &[usize]) -> Option<Vec<String>> {
        let hint = self.coords.borrow();
        let h = hint.as_ref()?;
        (h.idx == idx && h.shape == shape).then(|| h.axes.clone())
    }

    /// `index` as a whole-number combination of coordinates: the constant
    /// term and, per `⍳`, its coefficient, axis, operand and the lifts it
    /// sits under. None when anything in it is not a whole number, a `⍳`,
    /// a sum, a difference or a product with a whole number — then the
    /// position is a value the data decides, read as a gather.
    fn affine_position(index: &Expr, lifts: &[usize]) -> Option<AffinePosition> {
        fn whole(v: f64) -> Option<i64> {
            (v == v.trunc() && v.abs() < 9.0e15).then_some(v as i64)
        }
        fn walk(e: &Expr, scale: i64, lifts: &[usize], out: &mut AffinePosition) -> Option<()> {
            match e {
                Expr::Number(v) => {
                    out.constant = out.constant.checked_add(scale.checked_mul(whole(*v)?)?)?;
                    Some(())
                }
                Expr::Index { axis, operand } => {
                    out.terms.push(AffineTerm {
                        coefficient: scale,
                        axis: *axis,
                        operand: (**operand).clone(),
                        lifts: lifts.to_vec(),
                    });
                    Some(())
                }
                Expr::Lift { axis, operand } => {
                    let mut nested = lifts.to_vec();
                    nested.push(*axis);
                    walk(operand, scale, &nested, out)
                }
                Expr::AuditTrace(inner) => walk(inner, scale, lifts, out),
                Expr::BinaryOp { op: BinaryOpKind::Add, lhs, rhs } => {
                    walk(lhs, scale, lifts, out)?;
                    walk(rhs, scale, lifts, out)
                }
                Expr::BinaryOp { op: BinaryOpKind::Sub, lhs, rhs } => {
                    walk(lhs, scale, lifts, out)?;
                    walk(rhs, scale.checked_neg()?, lifts, out)
                }
                Expr::BinaryOp { op: BinaryOpKind::Mul, lhs, rhs } => match (&**lhs, &**rhs) {
                    (Expr::Number(v), other) | (other, Expr::Number(v)) => {
                        walk(other, scale.checked_mul(whole(*v)?)?, lifts, out)
                    }
                    _ => None,
                },
                _ => None,
            }
        }
        let mut out = AffinePosition { constant: 0, terms: Vec::new() };
        walk(index, 1, lifts, &mut out)?;
        // A position with no coordinate in it is a constant the value
        // path reads just as well; the gain here is per-cell arithmetic.
        (!out.terms.is_empty()).then_some(out)
    }

    /// `⌷` at a position that is a whole-number combination of coordinates:
    /// the position in integer arithmetic, and the bounds test only when the
    /// position's range, known from the axes' extents, can leave the space.
    /// The bits are the value path's: the interpreter's f64 arithmetic on
    /// whole numbers below 2^53 is exact, and the floor of an integer is
    /// itself.
    #[allow(clippy::too_many_arguments)]
    fn emit_affine_gather(
        &self,
        ir: &mut String,
        ptr: &str,
        count: usize,
        affine: &AffinePosition,
        idx: &str,
        counter: &mut usize,
        mode: Mode,
        result_shape: &[usize],
    ) -> Result<String> {
        let int_ty = mode.int_ty();
        let elem = self.precision.llvm_type();
        let ty = mode.ty();
        let (mut low, mut high) = (affine.constant as i128, affine.constant as i128);
        let mut acc = mode.int_splat(affine.constant as u64);
        for term in &affine.terms {
            let (coord, extent) =
                self.emit_coordinate(ir, term.axis, &term.operand, idx, counter, mode, result_shape, &term.lifts)?;
            let reach = term.coefficient as i128 * (extent as i128 - 1);
            low += reach.min(0);
            high += reach.max(0);
            let scaled = Self::fresh(counter);
            let next = Self::fresh(counter);
            ir.push_str(&format!(
                "  {scaled} = mul {int_ty} {coord}, {}\n",
                mode.int_splat(term.coefficient as u64)
            ));
            ir.push_str(&format!("  {next} = add {int_ty} {acc}, {scaled}\n"));
            acc = next;
        }
        let inside_always = low >= 0 && high < count as i128;
        match mode {
            Mode::Scalar(_) => {
                let gep = Self::fresh(counter);
                let val = Self::fresh(counter);
                if inside_always {
                    ir.push_str(&format!("  {gep} = getelementptr inbounds {elem}, ptr {ptr}, i64 {acc}\n"));
                    ir.push_str(&format!("  {val} = load {elem}, ptr {gep}, align {}\n", self.precision.bytes()));
                    return Ok(val);
                }
                let not_before = Self::fresh(counter);
                let not_past = Self::fresh(counter);
                let inside = Self::fresh(counter);
                let safe = Self::fresh(counter);
                let out = Self::fresh(counter);
                ir.push_str(&format!("  {not_before} = icmp sge i64 {acc}, 0\n"));
                ir.push_str(&format!("  {not_past} = icmp slt i64 {acc}, {count}\n"));
                ir.push_str(&format!("  {inside} = and i1 {not_before}, {not_past}\n"));
                ir.push_str(&format!("  {safe} = select i1 {inside}, i64 {acc}, i64 0\n"));
                ir.push_str(&format!("  {gep} = getelementptr inbounds {elem}, ptr {ptr}, i64 {safe}\n"));
                ir.push_str(&format!("  {val} = load {elem}, ptr {gep}, align {}\n", self.precision.bytes()));
                ir.push_str(&format!(
                    "  {out} = select i1 {inside}, {elem} {val}, {elem} {}\n",
                    self.f64_literal(0.0)
                ));
                Ok(out)
            }
            Mode::Vector(w, _) => {
                // Lane by lane: the positions need not be consecutive.
                let bty = mode.bool_ty();
                let (inside, safe) = if inside_always {
                    (None, acc.clone())
                } else {
                    let not_before = Self::fresh(counter);
                    let not_past = Self::fresh(counter);
                    let inside = Self::fresh(counter);
                    let safe = Self::fresh(counter);
                    ir.push_str(&format!("  {not_before} = icmp sge {int_ty} {acc}, zeroinitializer\n"));
                    ir.push_str(&format!("  {not_past} = icmp slt {int_ty} {acc}, {}\n", mode.int_splat(count as u64)));
                    ir.push_str(&format!("  {inside} = and {bty} {not_before}, {not_past}\n"));
                    ir.push_str(&format!("  {safe} = select {bty} {inside}, {int_ty} {acc}, {int_ty} zeroinitializer\n"));
                    (Some(inside), safe)
                };
                let mut vec = "poison".to_string();
                for l in 0..w {
                    let at = Self::fresh(counter);
                    let gep = Self::fresh(counter);
                    let val = Self::fresh(counter);
                    let next = Self::fresh(counter);
                    ir.push_str(&format!("  {at} = extractelement {int_ty} {safe}, i32 {l}\n"));
                    ir.push_str(&format!("  {gep} = getelementptr inbounds {elem}, ptr {ptr}, i64 {at}\n"));
                    ir.push_str(&format!("  {val} = load {elem}, ptr {gep}, align {}\n", self.precision.bytes()));
                    ir.push_str(&format!("  {next} = insertelement {ty} {vec}, {elem} {val}, i32 {l}\n"));
                    vec = next;
                }
                match inside {
                    None => Ok(vec),
                    Some(inside) => {
                        let out = Self::fresh(counter);
                        ir.push_str(&format!("  {out} = select {bty} {inside}, {ty} {vec}, {ty} {}\n", mode.zero()));
                        Ok(out)
                    }
                }
            }
        }
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

    /// Exact bit pattern, so literals survive round-tripping and NaN/inf stay
    /// legal. A float literal is still written as a double in LLVM, but of a
    /// value that single precision can hold exactly.
    pub(crate) fn float_literal(value: f64, precision: Precision) -> String {
        let exact = match precision {
            Precision::F64 => value,
            Precision::F32 => value as f32 as f64,
        };
        format!("0x{:016X}", exact.to_bits())
    }

    fn f64_literal(&self, value: f64) -> String {
        Self::float_literal(value, self.precision)
    }

    // -------------------------------------------------------------- metadata

    fn generate_metadata_json(&self) -> String {
        let spaces: Vec<String> = self
            .space_shapes
            .iter()
            .map(|(name, shape)| {
                format!(
                    "{{\"name\":\"{name}\",\"shape\":{shape:?},\"role\":\"{}\"}}",
                    self.role_of(name)
                )
            })
            .collect();
        let bindings: Vec<String> = self
            .ext_bindings
            .iter()
            .map(|(name, addr)| format!("{{\"name\":\"{name}\",\"address\":{addr}}}"))
            .collect();
        // A kernel that iterates says how far it may go and when it stops,
        // since both are part of what it computes.
        let iteration = match (self.iterates, self.max_sweeps) {
            (true, Some(cap)) => format!(
                ",\"iteration\":{{\"max_sweeps\":{cap},\"tolerance\":{}}}",
                self.tau
            ),
            _ => String::new(),
        };
        // Reentrant: two calls may run at once. Every call has its own table
        // and stack; scratch kept between calls is the kernel's one copy.
        format!(
            "{{\"precision\":\"{}\",\"elements\":{},\"threads\":{},\"reentrant\":{},\"spaces\":[{}],\"bindings\":[{}]{}}}",
            self.precision,
            self.elements,
            self.threads,
            self.keep_count.get() == 0,
            spaces.join(","),
            bindings.join(","),
            iteration
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
        // The pool the kernel's parts run on, compiled in the same call.
        let temp_rt = format!("{}.rt.c", output_path);
        fs::write(&temp_rt, RUNTIME_C)?;

        // The module carries no target triple on purpose, so clang supplies the
        // host's. That substitution is what -Woverride-module reports; it is the
        // intended behaviour here, not a problem to surface on every build.
        let mut args: Vec<&str> = vec![
            "-shared",
            "-fPIC",
            "-O3",
            // Contraction — fusing a multiply and an add into one rounding —
            // makes the kernel disagree with the per-operation rounding the
            // solver models, and with the reference interpreter. Differential
            // testing found the divergence as a one-ulp mismatch.
            "-ffp-contract=off",
            "-Wno-override-module",
            "-pthread",
        ];
        // The kernel is built where it runs, so it may use every vector
        // width this machine has; `--portable` gives that up for a .so that
        // runs on any x86-64. Only x86-64 clang takes `-march=native`; on
        // other machines the baseline is what there is.
        if self.native && cfg!(target_arch = "x86_64") {
            args.push("-march=native");
        }
        args.extend([temp_ll.as_str(), temp_rt.as_str(), "-o", output_path]);
        let status = Command::new("clang-22")
            .args(&args)
            .status()
            .or_else(|_| Command::new("clang").args(&args).status())?;

        if status.success() {
            let _ = fs::remove_file(temp_ll);
            let _ = fs::remove_file(temp_rt);
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "clang failed on {temp_ll}; the IR was kept for inspection"
            )))
        }
    }

    /// Compile LLVM IR to a static library (`.a`) whose symbols all start
    /// with `prefix`: `rho_kernel_exec_spaces` is `{prefix}_exec_spaces`, and
    /// so on for every entrypoint, and the runtime's `rho_rt_*` are
    /// `{prefix}_rt_*`. Kernels built with different prefixes link into one
    /// program side by side, which kernels sharing their names cannot; the
    /// runtime's state is its own per kernel, as in a shared library.
    pub fn compile_to_static(&self, ir_content: &str, output_path: &str, prefix: &str) -> std::io::Result<()> {
        let valid = prefix.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && prefix.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid {
            return Err(std::io::Error::other(format!("--symbol-prefix {prefix:?} is not a C identifier")));
        }
        let ir = ir_content
            .replace("@rho_kernel_", &format!("@{prefix}_"))
            .replace("@rho_rt_", &format!("@{prefix}_rt_"));
        let temp_ll = format!("{output_path}.ll");
        let temp_rt = format!("{output_path}.rt.c");
        let (kernel_o, rt_o) = (format!("{output_path}.kernel.o"), format!("{output_path}.rt.o"));
        fs::write(&temp_ll, ir)?;
        let renames: String = ["scratch", "zeroize", "run", "parts"]
            .iter()
            .map(|f| format!("#define rho_rt_{f} {prefix}_rt_{f}\n"))
            .collect();
        fs::write(&temp_rt, format!("{renames}{RUNTIME_C}"))?;
        let mut flags: Vec<&str> = vec!["-c", "-fPIC", "-O3", "-ffp-contract=off", "-Wno-override-module", "-pthread"];
        if self.native && cfg!(target_arch = "x86_64") {
            flags.push("-march=native");
        }
        let clang = |input: &str, out: &str| -> std::io::Result<bool> {
            let mut args = flags.clone();
            args.extend([input, "-o", out]);
            Ok(Command::new("clang-22").args(&args).status().or_else(|_| Command::new("clang").args(&args).status())?.success())
        };
        let built = clang(&temp_ll, &kernel_o)? && clang(&temp_rt, &rt_o)?;
        let _ = fs::remove_file(output_path);
        let archived = built
            && Command::new("ar").args(["rcs", output_path, &kernel_o, &rt_o]).status()?.success();
        let _ = fs::remove_file(&kernel_o);
        let _ = fs::remove_file(&rt_o);
        if archived {
            let _ = fs::remove_file(temp_ll);
            let _ = fs::remove_file(temp_rt);
            Ok(())
        } else {
            Err(std::io::Error::other(format!("building {output_path} failed; the IR was kept in {temp_ll}")))
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

/// Folds and scans in an expression: each is computed as a pass of its own
/// before the sweep that reads it.
fn fold_count(expr: &Expr) -> usize {
    match expr {
        Expr::Reduce { operand, .. } | Expr::Scan { operand, .. } => 1 + fold_count(operand),
        Expr::Var(_) | Expr::Number(_) | Expr::Index { .. } => 0,
        Expr::AuditTrace(inner)
        | Expr::Builtin { operand: inner, .. }
        | Expr::Shift { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. } => fold_count(inner),
        Expr::BinaryOp { lhs, rhs, .. } | Expr::Gather { index: lhs, operand: rhs } => fold_count(lhs) + fold_count(rhs),
        Expr::Call { args, .. } => args.iter().map(fold_count).sum(),
    }
}
