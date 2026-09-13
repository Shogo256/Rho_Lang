//! A program as a circuit: SystemVerilog for a streaming pipeline that takes
//! one cell per clock, in row-major order, and gives one cell per clock back.
//!
//! A ρ program is a static dataflow: every `→` is a function of a cell and
//! its neighbours, and nothing depends on data for its shape or its control.
//! That is what a pipeline is. Each flow becomes a stage: its sources pass
//! through delay lines long enough to hold the furthest neighbour on either
//! side, the stage reads the taps, and the target is one more register. A
//! shift along the innermost axis is a tap one cell away; along an outer
//! axis it is a tap one line away — the line buffer of every stencil
//! accelerator, here because the language said so. Boundary cells read zero,
//! decided by the cell's coordinates, which a counter carries along.
//!
//! Every stream carries a valid bit. A stage's lines advance only on a valid
//! cell, and when a source's cells have all arrived the stage pushes zeros
//! through for as many cells as its reach, so the last cells come out. A
//! fold is a stage with one accumulator per line across the folded axis; it
//! emits a cell whenever a line completes, so its stream is sparse, and what
//! reads it follows its valids. A scan emits on every cell. Streams of one
//! origin — the dense inputs, or one fold's output — are aligned by clocks;
//! two origins in one flow are refused, as is a broadcast.
//!
//! The cells are SystemVerilog `real`: IEEE double in simulation, with the
//! same libm the interpreter and the kernel call, so Verilator's run is held
//! to the interpreter bit for bit. `real` does not synthesise; this is the
//! structure and the timing of the circuit, one cell per cycle, with the
//! arithmetic units left to a later step (fixed point, or floating-point
//! cores). What is in the subset so far: `→`, arithmetic, comparisons as
//! masks, the greater and the lesser, the residue, the named functions,
//! `?`, `⌊` `⌈`, `⍳`, shifts along any axis, chains of flows, folds and
//! scans along any axis, folds of folds, and `⇒`: the spaces a fixed point
//! reads are captured into memories, streamed out again every round through
//! an update stage into a second buffer while the largest move is taken,
//! until the move is within 𝜏 or the cap is reached, and the result is
//! streamed out for what follows. A flow whose sources do not share one
//! timing, or that broadcasts a smaller space (`□`, a length-1 axis), is a
//! replay: its sources are captured into memories and the result's cells
//! are streamed out of them, the broadcast ones read at the mapped place —
//! which is what makes a matrix product a circuit. Not yet: the turns,
//! `⌷`, a fold or a broadcast inside `⇒`.

use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

/// What the cells are made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numbers {
    /// SystemVerilog `real`: a double in simulation, not synthesisable.
    /// The structure and timing of the circuit, with the arithmetic units
    /// left for later.
    Real,
    /// Two's complement of `width` bits with `frac` fraction bits — a
    /// datapath Yosys synthesises. The reference is `numeric::Fixed`: sums
    /// wrap, products floor, quotients truncate toward zero, a division by
    /// zero is zero. exp, log, sqrt, sin, cos and a fractional power are
    /// refused.
    Fixed { width: u32, frac: u32 },
}

impl Numbers {
    /// The SystemVerilog type of a cell.
    fn ty(self) -> String {
        match self {
            Numbers::Real => "real".to_string(),
            Numbers::Fixed { width, .. } => format!("logic signed [{}:0]", width - 1),
        }
    }

    /// A literal cell.
    fn lit(self, v: f64) -> String {
        match self {
            Numbers::Real => format!("$bitstoreal(64'h{:016X})", v.to_bits()),
            Numbers::Fixed { width, frac } => {
                let raw = fixed_raw(v, width, frac);
                format!("{width}'sh{:0w$X}", raw as u64 & mask(width), w = width.div_ceil(4) as usize)
            }
        }
    }
}

/// Bits enough to count from 0 to `n`.
fn bits(n: usize) -> usize {
    (usize::BITS - n.leading_zeros()).max(1) as usize
}

fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// `numeric::Fixed::from_f64` for a width and fraction known at run time.
fn fixed_raw(v: f64, width: u32, frac: u32) -> i64 {
    let max = (1i128 << (width - 1)) - 1;
    let min = -(1i128 << (width - 1));
    let wrap = |x: i128| -> i64 {
        let shift = 128 - width;
        ((x << shift) >> shift) as i64
    };
    if v.is_nan() {
        0
    } else if v == f64::INFINITY {
        max as i64
    } else if v == f64::NEG_INFINITY {
        min as i64
    } else {
        wrap((v * (1u64 << frac) as f64).round() as i128)
    }
}

/// What the emitter produced: the module, a Verilator harness for it, and
/// the numbers a caller needs to drive it.
#[derive(Debug)]
pub struct Circuit {
    pub module: String,
    pub harness: String,
    /// Input spaces in port order, with their cell counts.
    pub inputs: Vec<(String, usize)>,
    /// Cells of OUTPUT.
    pub output_cells: usize,
    /// Clocks from a cell entering to its output cell leaving.
    pub latency: usize,
    pub numbers: Numbers,
}

/// What a simulation produced.
#[derive(Debug)]
pub struct Run {
    pub output: Vec<f64>,
    pub cycles: u64,
    /// Sweeps the `⇒` loops took, in all, and whether every one settled.
    pub sweeps: u64,
    pub converged: bool,
}

/// What a stage does with the cell it computes.
enum Kind {
    /// The cell is the target's cell.
    Flow,
    /// The cell joins an accumulator; a cell of the target leaves when a
    /// line along the axis completes.
    Fold { op: FoldOp, axis: usize, running: bool },
}

struct Stage {
    /// Position in the program: signal names carry it, since a space may be
    /// written by more than one flow (`T → OUTPUT` and then `OUTPUT → =`).
    index: usize,
    /// Target space (a fold's is a name of its own) and the shape the stage
    /// sweeps — the operand's, for a fold.
    target: String,
    shape: Vec<usize>,
    /// Clocks from a source cell of the stage's origin to the stage's cell.
    latency: usize,
    /// How far the furthest neighbour is, in cells.
    reach: usize,
    /// Source space -> (value signal, valid signal, its latency, chain length, centre tap).
    chains: BTreeMap<String, (String, String, usize, usize, usize)>,
    /// The expression, as SystemVerilog over the chains.
    expr: String,
    kind: Kind,
    /// For the update stage of a `⇒`: the signal that starts a round afresh.
    round_reset: Option<String>,
    /// For a stage with no streamed source: the valid that paces it.
    timing: Option<String>,
}

/// A space read at a mapped place rather than streamed: its memory, its own
/// shape, and which axes of the view a lift put there.
#[derive(Clone, Debug)]
struct BroadcastRead {
    memory: String,
    source_shape: Vec<usize>,
    lifted: Vec<bool>,
}

/// A stage whose sources are captured and streamed back in the result's
/// order: the main ones as streams, the broadcast ones read by place.
struct Replay {
    id: usize,
    cells: usize,
    main: Vec<(String, Stream)>,
    broadcast: Vec<(String, Stream)>,
}

/// A `⇒`: which stage updates, what it reads, and where the result goes.
struct Loop {
    id: usize,
    target: String,
    /// Sources of the update, with the streams they are captured from.
    sources: Vec<(String, Stream)>,
    cells: usize,
    update: usize,
    cap: usize,
    tau: f64,
}

/// Where a stream's timing comes from: the dense inputs of one shape, or
/// one fold's completions. Streams of one origin differ by whole clocks.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Origin(String);

/// A stream: the signal, its valid, its latency, its origin, and the cells it carries.
#[derive(Clone, Debug)]
struct Stream {
    value: String,
    valid: String,
    latency: usize,
    origin: Origin,
    cells: usize,
}

fn unsupported(what: &str, line: usize) -> HarmonyDisruption {
    HarmonyDisruption::LoweringErr {
        detail: format!("{what} is not yet in the SystemVerilog subset (--emit-sv)"),
        line,
    }
}


/// The spaces an expression reads, and the furthest a shift reaches along
/// the buffer (in cells): the delay lines have to hold that much either side.
fn sources(expr: &Expr, shapes: &BTreeMap<String, Vec<usize>>, line: usize) -> Result<(BTreeSet<String>, usize)> {
    let mut out = BTreeSet::new();
    let mut reach = 0usize;
    fn walk(
        expr: &Expr,
        shapes: &BTreeMap<String, Vec<usize>>,
        line: usize,
        out: &mut BTreeSet<String>,
        reach: &mut usize,
    ) -> Result<()> {
        match expr {
            Expr::Number(_) => {}
            Expr::Var(name) => {
                if !is_tau(name) {
                    out.insert(name.clone());
                }
            }
            Expr::AuditTrace(inner) | Expr::Builtin { operand: inner, .. } => walk(inner, shapes, line, out, reach)?,
            Expr::BinaryOp { lhs, rhs, .. } => {
                walk(lhs, shapes, line, out, reach)?;
                walk(rhs, shapes, line, out, reach)?;
            }
            Expr::Shift { axis, operand, .. } => {
                let Expr::Var(name) = &**operand else {
                    return Err(unsupported("a shift of a computed value", line));
                };
                let shape = shapes.get(name).ok_or_else(|| unsupported(&format!("the shape of `{name}`"), line))?;
                let (stride, extent) = axis_geometry(shape, *axis).ok_or_else(|| unsupported("an axis past the shape", line))?;
                if extent > 1 {
                    *reach = (*reach).max(stride);
                }
                out.insert(name.clone());
            }
            Expr::Index { operand, .. } => {
                // A coordinate reads no cell, but it takes the space's
                // timing: the stage counts the space's cells to know where
                // it is, so the space is a source of the stage all the same.
                let Expr::Var(name) = &**operand else {
                    return Err(unsupported("`⍳` of a computed value", line));
                };
                out.insert(name.clone());
            }
            // A fold or scan was made a stage of its own before this walk;
            // only its name is left in the expression.
            Expr::Reduce { .. } | Expr::Scan { .. } => return Err(unsupported("a fold inside a fold's own line", line)),
            // A lift reads its space at a mapped place; the place is the
            // stage's business (see `usage`), the space is a source.
            Expr::Lift { operand, .. } => walk(operand, shapes, line, out, reach)?,
            Expr::Rotate { .. } | Expr::Reverse { .. } => return Err(unsupported("a rotation or reversal", line)),
            Expr::Reshape { .. } | Expr::Transpose { .. } => return Err(unsupported("a reshape or transpose", line)),
            Expr::Take { .. } | Expr::Drop { .. } => return Err(unsupported("a take or drop", line)),
            Expr::Gather { .. } => return Err(unsupported("an index by value (`⌷`)", line)),
            Expr::Call { name, .. } => return Err(unsupported(&format!("the call to `{name}`, which was not expanded"), line)),
        }
        Ok(())
    }
    walk(expr, shapes, line, &mut out, &mut reach)?;
    Ok((out, reach))
}

fn is_tau(name: &str) -> bool {
    name == "𝜏" || name == "τ"
}

/// How each space is read: the lifts around it, which must be the same at
/// every read; a shift or `⍳` cannot read a lifted space.
fn usage(expr: &Expr, line: usize, lifts: &mut Vec<usize>, out: &mut BTreeMap<String, Vec<usize>>) -> Result<()> {
    match expr {
        Expr::Var(name) => {
            if is_tau(name) {
                return Ok(());
            }
            match out.get(name) {
                Some(seen) if *seen != *lifts => {
                    return Err(unsupported(&format!("`{name}` read with two different lifts"), line))
                }
                _ => {
                    out.insert(name.clone(), lifts.clone());
                }
            }
        }
        Expr::Lift { axis, operand } => {
            lifts.push(*axis);
            usage(operand, line, lifts, out)?;
            lifts.pop();
        }
        Expr::Shift { operand, .. } | Expr::Index { operand, .. } => {
            if !lifts.is_empty() {
                return Err(unsupported("a shift or `⍳` of a lifted space", line));
            }
            usage(operand, line, lifts, out)?;
        }
        Expr::AuditTrace(inner) | Expr::Builtin { operand: inner, .. } => usage(inner, line, lifts, out)?,
        Expr::BinaryOp { lhs, rhs, .. } => {
            usage(lhs, line, lifts, out)?;
            usage(rhs, line, lifts, out)?;
        }
        _ => {}
    }
    Ok(())
}

/// The shape a space is viewed at under `lifts`, and which of the view's
/// axes the lifts put there.
fn lifted_view(shape: &[usize], lifts: &[usize]) -> Option<(Vec<usize>, Vec<bool>)> {
    let mut view = shape.to_vec();
    let mut marks = vec![false; shape.len()];
    for &axis in lifts.iter().rev() {
        if axis > view.len() {
            return None;
        }
        view.insert(axis, 1);
        marks.insert(axis, true);
    }
    Some((view, marks))
}

/// One stage's expression over its chains, as SystemVerilog `real`.
struct Lowering<'a> {
    shapes: &'a BTreeMap<String, Vec<usize>>,
    chains: &'a BTreeMap<String, (String, String, usize, usize, usize)>,
    /// The result shape, whose coordinates `pos` holds.
    shape: &'a [usize],
    /// The stage's number: its delay lines are `c<stage>_<source>`.
    stage: usize,
    tau: f64,
    line: usize,
    numbers: Numbers,
    /// Spaces read by place out of a memory rather than from a stream.
    broadcast: &'a BTreeMap<String, BroadcastRead>,
}

impl Lowering<'_> {
    fn lower(&self, expr: &Expr) -> Result<String> {
        self.lower_with(expr, &[])
    }

    /// The place in a broadcast source's memory that this cell reads: its
    /// coordinates along the axes the source has, zero where the source is
    /// one wide, nothing for an axis a lift put there.
    fn place(&self, read: &BroadcastRead) -> String {
        let mut terms: Vec<String> = Vec::new();
        let mut j = 0usize;
        for (a, lifted) in read.lifted.iter().enumerate() {
            if *lifted {
                continue;
            }
            let extent = read.source_shape.get(j).copied().unwrap_or(1);
            let stride: usize = read.source_shape[j + 1..].iter().product::<usize>().max(1);
            if extent > 1 && a < self.shape.len() {
                terms.push(if stride == 1 {
                    format!("pos[{a}]")
                } else {
                    format!("(pos[{a}] * {stride})")
                });
            }
            j += 1;
        }
        if terms.is_empty() {
            "0".to_string()
        } else {
            terms.join(" + ")
        }
    }

    fn lower_with(&self, expr: &Expr, lifts: &[usize]) -> Result<String> {
        Ok(match expr {
            Expr::Number(v) => self.numbers.lit(*v),
            Expr::Var(name) if is_tau(name) => self.numbers.lit(self.tau),
            Expr::Var(name) => match self.broadcast.get(name) {
                Some(read) => format!("{}[{}]", read.memory, self.place(read)),
                None => {
                    let (_, _, _, _, centre) = &self.chains[name];
                    format!("c{}_{}[{centre}]", self.stage, ident(name))
                }
            },
            Expr::Lift { axis, operand } => {
                let mut nested = lifts.to_vec();
                nested.push(*axis);
                self.lower_with(operand, &nested)?
            }
            Expr::AuditTrace(inner) => self.lower_with(inner, lifts)?,
            Expr::Shift { dir, axis, operand } => {
                let Expr::Var(name) = &**operand else {
                    return Err(unsupported("a shift of a computed value", self.line));
                };
                let shape = &self.shapes[name];
                let (stride, extent) = axis_geometry(shape, *axis).ok_or_else(|| unsupported("an axis past the shape", self.line))?;
                if extent <= 1 {
                    return Ok(self.numbers.lit(0.0));
                }
                let a = axis.unwrap_or_else(|| default_axis(shape));
                let (_, _, _, _, centre) = &self.chains[name];
                let (tap, edge) = match dir {
                    ShiftDir::Positive => (centre + stride, 0),
                    ShiftDir::Negative => (centre - stride, extent - 1),
                };
                format!(
                    "((pos[{a}] == {edge}) ? {} : c{}_{}[{tap}])",
                    self.numbers.lit(0.0),
                    self.stage,
                    ident(name)
                )
            }
            Expr::Index { axis, operand } => {
                let Expr::Var(name) = &**operand else {
                    return Err(unsupported("`⍳` of a computed value", self.line));
                };
                let shape = &self.shapes[name];
                let a = axis.unwrap_or_else(|| default_axis(shape));
                match self.numbers {
                    Numbers::Real => format!("real'(pos[{a}])"),
                    Numbers::Fixed { .. } => format!("fx_int(pos[{a}])"),
                }
            }
            Expr::Builtin { op, operand } => {
                let x = self.lower(operand)?;
                let fixed = matches!(self.numbers, Numbers::Fixed { .. });
                match op {
                    BuiltinOp::Exp | BuiltinOp::Log | BuiltinOp::Sqrt | BuiltinOp::Sin | BuiltinOp::Cos if fixed => {
                        return Err(unsupported(&format!("`{op}` in fixed point"), self.line))
                    }
                    BuiltinOp::Exp => format!("$exp({x})"),
                    BuiltinOp::Log => format!("$ln({x})"),
                    BuiltinOp::Sqrt => format!("$sqrt({x})"),
                    BuiltinOp::Sin => format!("$sin({x})"),
                    BuiltinOp::Cos => format!("$cos({x})"),
                    BuiltinOp::Abs => format!("rho_abs({x})"),
                    BuiltinOp::Indicator => format!("rho_ind({x})"),
                    BuiltinOp::Roll => format!("rho_roll({x})"),
                    BuiltinOp::Floor if fixed => format!("fx_floor({x})"),
                    BuiltinOp::Ceil if fixed => format!("fx_ceil({x})"),
                    BuiltinOp::Floor => format!("$floor({x})"),
                    BuiltinOp::Ceil => format!("$ceil({x})"),
                }
            }
            Expr::BinaryOp { op, lhs, rhs } => {
                let l = self.lower(lhs)?;
                let r = self.lower(rhs)?;
                let fixed = matches!(self.numbers, Numbers::Fixed { .. });
                let bin = |f: &str, sym: &str, a: &str, b: &str| {
                    if fixed {
                        format!("{f}({a}, {b})")
                    } else {
                        format!("({a} {sym} {b})")
                    }
                };
                // In fixed point a constant that is a power of two is a shift,
                // not a multiplier or a divider: the same bits, since a
                // product floors as an arithmetic shift does, and a quotient
                // truncates toward zero, which `fx_div_pow2` biases for.
                let pow2 = |e: &Expr| -> Option<(u32, bool)> {
                    let Numbers::Fixed { width, frac } = self.numbers else { return None };
                    let Expr::Number(v) = e else { return None };
                    let raw = fixed_raw(*v, width, frac);
                    (raw > 0 && (raw as u64).is_power_of_two()).then(|| {
                        let m = raw.trailing_zeros();
                        if m >= frac { (m - frac, true) } else { (frac - m, false) }
                    })
                };
                match op {
                    BinaryOpKind::Add => bin("fx_add", "+", &l, &r),
                    BinaryOpKind::Sub => bin("fx_sub", "-", &l, &r),
                    BinaryOpKind::Mul => match (pow2(rhs), pow2(lhs)) {
                        (Some((k, up)), _) => format!("{}({l}, {k})", if up { "fx_shl" } else { "fx_shr" }),
                        (None, Some((k, up))) => format!("{}({r}, {k})", if up { "fx_shl" } else { "fx_shr" }),
                        _ => bin("fx_mul", "*", &l, &r),
                    },
                    BinaryOpKind::Div => match pow2(rhs) {
                        Some((k, true)) => format!("fx_div_pow2({l}, {k})"),
                        Some((k, false)) => format!("fx_shl({l}, {k})"),
                        None => bin("fx_div", "/", &l, &r),
                    },
                    BinaryOpKind::Pow => match whole_exponent(rhs) {
                        // Repeated multiplication, as the kernel and the
                        // interpreter do it, from the same rule.
                        Some(n) => {
                            let steps = n.unsigned_abs();
                            let mut acc = self.numbers.lit(1.0);
                            for _ in 0..steps {
                                acc = bin("fx_mul", "*", &acc, &l);
                            }
                            if n < 0 {
                                bin("fx_div", "/", &self.numbers.lit(1.0), &acc)
                            } else {
                                acc
                            }
                        }
                        None if fixed => return Err(unsupported("a fractional power in fixed point", self.line)),
                        None => format!("$pow({l}, {r})"),
                    },
                    BinaryOpKind::Gt => format!("rho_mask({l} > {r}, {l})"),
                    BinaryOpKind::Lt => format!("rho_mask({l} < {r}, {l})"),
                    BinaryOpKind::Gte => format!("rho_mask({l} >= {r}, {l})"),
                    BinaryOpKind::Lte => format!("rho_mask({l} <= {r}, {l})"),
                    BinaryOpKind::Eq => format!("rho_mask({l} == {r}, {l})"),
                    BinaryOpKind::Max => format!("rho_extreme({l}, {r}, 1'b1)"),
                    BinaryOpKind::Min => format!("rho_extreme({l}, {r}, 1'b0)"),
                    BinaryOpKind::Residue => format!("rho_residue({l}, {r})"),
                }
            }
            other => {
                let (_, _) = sources(other, self.shapes, self.line)?;
                return Err(unsupported("this expression", self.line));
            }
        })
    }
}

fn ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Emit the circuit for `block`, or say what in it is not yet a circuit.
pub fn emit(block: &ToposBlock, tau: f64, cap: Option<usize>, numbers: Numbers) -> Result<Circuit> {
    if let Numbers::Fixed { width, frac } = numbers {
        if !(2..=64).contains(&width) || frac >= width {
            return Err(unsupported(&format!("a fixed format of {width} bits with {frac} fraction bits"), 0));
        }
    }
    let ty = numbers.ty();
    let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut inputs: Vec<String> = Vec::new();
    for stmt in &block.statements {
        if let Statement::SpaceDef(decl) = stmt {
            shapes.insert(decl.name.clone(), decl.dimensions.clone());
            inputs.push(decl.name.clone());
        }
    }
    // The stream each space is on right now.
    let mut streams: BTreeMap<String, Stream> = inputs
        .iter()
        .map(|n| {
            let cells = shapes[n].iter().product::<usize>().max(1);
            (
                n.clone(),
                Stream {
                    value: format!("s_in_{}", ident(n)),
                    valid: format!("v_in_{}", ident(n)),
                    latency: 1,
                    origin: Origin(format!("dense{:?}", shapes[n])),
                    cells,
                },
            )
        })
        .collect();
    let mut stages: Vec<Stage> = Vec::new();
    let mut loops: Vec<Loop> = Vec::new();
    let mut replays: Vec<Replay> = Vec::new();
    let mut written: BTreeSet<String> = BTreeSet::new();
    let mut folds_made = 0usize;

    for (index, stmt) in block.statements.iter().enumerate() {
        let line = block.line_of(index);
        match stmt {
            Statement::SpaceDef(_) | Statement::Constraint(_) | Statement::AuditTrace(_) => {}
            Statement::ExtBind(_) => return Err(unsupported("a bound address", line)),
            Statement::Iterate { prelude, src, target } => {
                if !prelude.is_empty() {
                    return Err(unsupported("a call with a body of flows inside `⇒`", line));
                }
                if has_fold(src) {
                    return Err(unsupported("a fold inside `⇒`", line));
                }
                let cap = cap.ok_or_else(|| HarmonyDisruption::LoweringErr {
                    detail: "this program iterates (⇒); say how many sweeps it may take with --max-iter".to_string(),
                    line,
                })?;
                let shape = shapes.get(target).cloned().ok_or_else(|| HarmonyDisruption::SpaceErr {
                    space_name: target.clone(),
                    line,
                })?;
                let cells = shape.iter().product::<usize>().max(1);
                let (srcs, _) = sources(src, &shapes, line)?;
                let id = loops.len();
                // The round streams: every source read from its memory,
                // one cell per clock, all with one timing.
                let mut captured: Vec<(String, Stream)> = Vec::new();
                for s in srcs.iter().chain(std::iter::once(target)) {
                    if captured.iter().any(|(n, _)| n == s) {
                        continue;
                    }
                    let stream = streams.get(s).cloned().ok_or_else(|| HarmonyDisruption::SpaceErr {
                        space_name: s.clone(),
                        line,
                    })?;
                    let alias = format!("{s}⟳{id}");
                    shapes.insert(alias.clone(), shapes[s].clone());
                    streams.insert(
                        alias,
                        Stream {
                            value: format!("rr{id}_{}", ident(s)),
                            valid: format!("rv{id}"),
                            latency: 1,
                            origin: Origin(format!("round{id}")),
                            cells: shapes[s].iter().product::<usize>().max(1),
                        },
                    );
                    captured.push((s.clone(), stream));
                }
                let renamed = rename(src, &|name: &str| format!("{name}⟳{id}"));
                // Inside a loop every source is streamed by the round; a
                // source of another shape would need capturing per round.
                if srcs.iter().any(|s| shapes.get(s) != Some(&shape)) || has_lift(src) {
                    return Err(unsupported("a broadcast inside `⇒`", line));
                }
                let mut stage = plan_stage(
                    &renamed,
                    &format!("{target}⟳{id}"),
                    shape.clone(),
                    Kind::Flow,
                    line,
                    tau,
                    numbers,
                    &mut shapes,
                    &mut streams,
                    stages.len(),
                    &mut replays,
                )?;
                stage.round_reset = Some(format!("rs{id}"));
                let update = stage.index;
                stages.push(stage);
                loops.push(Loop {
                    id,
                    target: target.clone(),
                    sources: captured,
                    cells,
                    update,
                    cap,
                    tau,
                });
                // After the loop, the target is streamed out of its memory.
                streams.insert(
                    target.clone(),
                    Stream {
                        value: format!("px{id}"),
                        valid: format!("pv{id}"),
                        latency: 1,
                        origin: Origin(format!("after{id}")),
                        cells,
                    },
                );
                written.insert(target.clone());
            }
            Statement::Flow { src, target } => {
                let target = match target {
                    FlowTarget::Var(n) => n.clone(),
                    FlowTarget::Equilibrium => "OUTPUT".to_string(),
                };
                // Every fold or scan in the expression becomes a stage first,
                // innermost first, and its name stands in the expression.
                let src = extract_folds(src, line, tau, numbers, &mut shapes, &mut streams, &mut stages, &mut folds_made, &mut replays)?;
                let shape = expr_shape(&src, &shapes)
                    .or_else(|| shapes.get(&target).cloned())
                    .ok_or_else(|| unsupported("a flow with no shape", line))?;
                let stage = plan_stage(
                    &src,
                    &target,
                    shape.clone(),
                    Kind::Flow,
                    line,
                    tau,
                    numbers,
                    &mut shapes,
                    &mut streams,
                    stages.len(),
                    &mut replays,
                )?;
                let cells = shape.iter().product::<usize>().max(1);
                let origin = stage_origin(&stage, &streams);
                streams.insert(
                    target.clone(),
                    Stream {
                        value: format!("st{}_{}", stage.index, ident(&target)),
                        valid: format!("v{}_out", stage.index),
                        latency: stage.latency,
                        origin,
                        cells,
                    },
                );
                shapes.insert(target.clone(), shape);
                written.insert(target.clone());
                stages.push(stage);
                if matches!(stmt, Statement::Flow { target: FlowTarget::Equilibrium, .. }) {
                    break;
                }
            }
        }
    }
    let out_stream = streams
        .get("OUTPUT")
        .cloned()
        .ok_or_else(|| unsupported("a program with no `→ =`", 0))?;
    let out_latency = out_stream.latency;
    let out_cells = out_stream.cells;
    let inputs: Vec<(String, usize)> = inputs
        .into_iter()
        .filter(|n| !written.contains(n))
        .map(|n| {
            let cells = shapes[&n].iter().product::<usize>().max(1);
            (n, cells)
        })
        .collect();
    let budget: usize = inputs.iter().map(|(_, c)| *c).max().unwrap_or(0)
        + out_cells
        + stages.iter().map(|s| s.reach + 3).sum::<usize>()
        + loops
            .iter()
            .map(|l| (l.cap + 1) * (l.cells + stages[l.update].reach + 8) + 2 * l.cells)
            .sum::<usize>()
        + 16;

    // ------------------------------------------------------------ module
    let mut sv = String::new();
    let _ = writeln!(sv, "// ρ kernel as a streaming pipeline: one cell per clock, row-major order.");
    let _ = writeln!(sv, "// Latency {out_latency} clocks from a cell entering to its output cell leaving.");
    let _ = writeln!(sv, "// Cells are `real` (double): the structure and timing of the circuit; the");
    let _ = writeln!(sv, "// arithmetic units are a later step. Generated by rhoc --emit-sv.");
    let _ = writeln!(sv, "`timescale 1ns/1ps");
    let _ = writeln!(sv, "module rho_kernel (");
    let _ = writeln!(sv, "  input  logic clk,");
    let _ = writeln!(sv, "  input  logic rst,");
    let _ = writeln!(sv, "  input  logic in_valid,");
    for (name, _) in &inputs {
        let _ = writeln!(sv, "  input  {ty} in_{},", ident(name));
    }
    let _ = writeln!(sv, "  output logic out_valid,");
    let _ = writeln!(sv, "  output {ty} out_OUTPUT,");
    let _ = writeln!(sv, "  output longint out_sweeps,");
    let _ = writeln!(sv, "  output logic out_converged");
    let _ = writeln!(sv, ");");
    sv.push_str(&functions(numbers));
    for (name, cells) in &inputs {
        let id = ident(name);
        let _ = writeln!(sv, "  // the stream of {name}: {cells} cells, latency 1");
        let _ = writeln!(sv, "  {ty} s_in_{id};");
        let _ = writeln!(sv, "  logic v_in_{id};");
        let _ = writeln!(sv, "  logic [{}:0] n_in_{id};", bits(*cells) - 1);
        let _ = writeln!(sv, "  always_ff @(posedge clk) begin");
        let _ = writeln!(sv, "    if (rst) begin n_in_{id} <= 0; v_in_{id} <= 1'b0; s_in_{id} <= {}; end", numbers.lit(0.0));
        let _ = writeln!(sv, "    else begin");
        let _ = writeln!(sv, "      v_in_{id} <= in_valid && (n_in_{id} < {cells});");
        let _ = writeln!(sv, "      s_in_{id} <= in_valid ? in_{id} : {};", numbers.lit(0.0));
        let _ = writeln!(sv, "      if (in_valid) n_in_{id} <= n_in_{id} + 1;");
        let _ = writeln!(sv, "    end");
        let _ = writeln!(sv, "  end");
    }
    for stage in &stages {
        if let Some(r) = replays.iter().find(|r| r.id == stage.index) {
            emit_replay(&mut sv, r, numbers);
        }
        if let Some(l) = loops.iter().find(|l| l.update == stage.index) {
            emit_loop_declarations(&mut sv, l, numbers);
        }
        emit_stage(&mut sv, stage, &streams, numbers);
        if let Some(l) = loops.iter().find(|l| l.update == stage.index) {
            emit_loop_logic(&mut sv, l, stage, numbers);
        }
    }
    let _ = writeln!(sv);
    let _ = writeln!(sv, "  assign out_OUTPUT = {};", out_stream.value);
    let _ = writeln!(sv, "  assign out_valid = {};", out_stream.valid);
    if loops.is_empty() {
        let _ = writeln!(sv, "  assign out_sweeps = 0;");
        let _ = writeln!(sv, "  assign out_converged = 1'b1;");
    } else {
        let sw: Vec<String> = loops.iter().map(|l| format!("sw{}", l.id)).collect();
        let cv: Vec<String> = loops.iter().map(|l| format!("cv{}", l.id)).collect();
        let _ = writeln!(sv, "  assign out_sweeps = {};", sw.join(" + "));
        let _ = writeln!(sv, "  assign out_converged = {};", cv.join(" && "));
    }
    let _ = writeln!(sv, "endmodule");

    // ------------------------------------------------------------ harness
    let mut h = String::new();
    let _ = writeln!(h, "// Drives rho_kernel through Verilator: reads the input spaces from a file of");
    let _ = writeln!(h, "// doubles (in port order), streams them in, writes OUTPUT's cells to a file.");
    let _ = writeln!(h, "#include \"Vrho_kernel.h\"");
    let _ = writeln!(h, "#include \"verilated.h\"");
    let _ = writeln!(h, "#include <cstdio>");
    let _ = writeln!(h, "#include <cstdlib>");
    let _ = writeln!(h, "#include <cmath>");
    let _ = writeln!(h, "#include <vector>");
    let _ = writeln!(h, "int main(int argc, char** argv) {{");
    let _ = writeln!(h, "  Verilated::commandArgs(argc, argv);");
    let _ = writeln!(h, "  if (argc < 3) {{ std::fprintf(stderr, \"usage: %s inputs.bin output.bin\\n\", argv[0]); return 2; }}");
    let total_in: usize = inputs.iter().map(|(_, c)| c).sum();
    let _ = writeln!(h, "  std::vector<double> in({total_in});");
    let _ = writeln!(h, "  if ({total_in} > 0) {{ FILE* f = std::fopen(argv[1], \"rb\"); if (!f || std::fread(in.data(), sizeof(double), {total_in}, f) != {total_in}) {{ std::fprintf(stderr, \"short input\\n\"); return 3; }} std::fclose(f); }}");
    let _ = writeln!(h, "  Vrho_kernel m;");
    match numbers {
        Numbers::Real => {
            let _ = writeln!(h, "  auto to_cell = [](double v) {{ return v; }};");
            let _ = writeln!(h, "  auto from_cell = [](double v) {{ return v; }};");
        }
        Numbers::Fixed { width, frac } => {
            // The same rounding as numeric::Fixed::from_f64, and the sign
            // extension of a W-bit port read back as an unsigned word.
            let _ = writeln!(h, "  const double scale = (double)(1ULL << {frac});");
            let _ = writeln!(h, "  auto to_cell = [&](double v) -> unsigned long long {{");
            let _ = writeln!(h, "    long long raw;");
            let _ = writeln!(h, "    if (v != v) raw = 0; else if (v == 1.0/0.0) raw = (long long)(((unsigned long long)1 << ({width} - 1)) - 1); else if (v == -1.0/0.0) raw = -(long long)((unsigned long long)1 << ({width} - 1)); else raw = (long long)std::llround(v * scale);");
            let _ = writeln!(h, "    return (unsigned long long)raw & {}ULL;", mask(width));
            let _ = writeln!(h, "  }};");
            let _ = writeln!(h, "  auto from_cell = [&](unsigned long long w) -> double {{");
            let _ = writeln!(h, "    long long raw = (long long)(w << (64 - {width})) >> (64 - {width});");
            let _ = writeln!(h, "    return (double)raw / scale;");
            let _ = writeln!(h, "  }};");
        }
    }
    let _ = writeln!(h, "  auto tick = [&]() {{ m.clk = 0; m.eval(); m.clk = 1; m.eval(); }};");
    let _ = writeln!(h, "  m.rst = 1; m.in_valid = 0; tick(); m.rst = 0;");
    let _ = writeln!(h, "  std::vector<double> out; out.reserve({out_cells});");
    let _ = writeln!(h, "  long cycles = 0;");
    let feed_cells = inputs.iter().map(|(_, c)| *c).max().unwrap_or(0);
    let _ = writeln!(h, "  for (long t = 0; out.size() < {out_cells} && t < {budget}; t++) {{");
    let _ = writeln!(h, "    m.in_valid = t < {feed_cells};");
    let mut offset = 0usize;
    for (name, cells) in &inputs {
        let _ = writeln!(h, "    m.in_{} = to_cell((t < {cells}) ? in[{offset} + t] : 0.0);", ident(name));
        offset += cells;
    }
    let _ = writeln!(h, "    tick(); cycles++;");
    let _ = writeln!(h, "    if (m.out_valid) out.push_back(from_cell(m.out_OUTPUT));");
    let _ = writeln!(h, "  }}");
    let _ = writeln!(h, "  m.final();");
    let _ = writeln!(h, "  FILE* o = std::fopen(argv[2], \"wb\"); if (!o) return 4;");
    let _ = writeln!(h, "  std::fwrite(out.data(), sizeof(double), out.size(), o); std::fclose(o);");
    let _ = writeln!(h, "  std::printf(\"cells %zu cycles %ld latency {out_latency} sweeps %ld converged %d\\n\", out.size(), cycles, (long)m.out_sweeps, (int)m.out_converged);");
    let _ = writeln!(h, "  return out.size() == {out_cells} ? 0 : 5;");
    let _ = writeln!(h, "}}");

    Ok(Circuit {
        module: sv,
        harness: h,
        inputs,
        output_cells: out_cells,
        latency: out_latency,
        numbers,
    })
}

/// Replace every fold and scan in `expr` by a name, making a stage for each,
/// innermost first.
#[allow(clippy::too_many_arguments)]
fn extract_folds(
    expr: &Expr,
    line: usize,
    tau: f64,
    numbers: Numbers,
    shapes: &mut BTreeMap<String, Vec<usize>>,
    streams: &mut BTreeMap<String, Stream>,
    stages: &mut Vec<Stage>,
    made: &mut usize,
    replays: &mut Vec<Replay>,
) -> Result<Expr> {
    let sub = |e: &Expr, shapes: &mut BTreeMap<String, Vec<usize>>, streams: &mut BTreeMap<String, Stream>, stages: &mut Vec<Stage>, made: &mut usize, replays: &mut Vec<Replay>| -> Result<Box<Expr>> {
        Ok(Box::new(extract_folds(e, line, tau, numbers, shapes, streams, stages, made, replays)?))
    };
    Ok(match expr {
        Expr::Reduce { op, axis, operand } | Expr::Scan { op, axis, operand } => {
            let running = matches!(expr, Expr::Scan { .. });
            let operand = extract_folds(operand, line, tau, numbers, shapes, streams, stages, made, replays)?;
            let in_shape = expr_shape(&operand, shapes).ok_or_else(|| unsupported("a fold with no shape", line))?;
            let a = axis.unwrap_or_else(|| default_axis(&in_shape));
            if a >= in_shape.len() {
                return Err(unsupported(&format!("axis {a} of {in_shape:?}"), line));
            }
            *made += 1;
            let name = format!("{}{}{}", if running { "◈" } else { "◇" }, op, made);
            let stage = plan_stage(
                &operand,
                &name,
                in_shape.clone(),
                Kind::Fold { op: *op, axis: a, running },
                line,
                tau,
                numbers,
                shapes,
                streams,
                stages.len(),
                replays,
            )?;
            let out_shape = if running { in_shape.clone() } else { shape_without_axis(&in_shape, a) };
            let cells = out_shape.iter().product::<usize>().max(1);
            let origin = if running {
                stage_origin(&stage, streams)
            } else {
                Origin(format!("fold{}", stage.index))
            };
            streams.insert(
                name.clone(),
                Stream {
                    value: format!("st{}_{}", stage.index, ident(&name)),
                    valid: format!("v{}_out", stage.index),
                    latency: stage.latency,
                    origin,
                    cells,
                },
            );
            shapes.insert(name.clone(), out_shape);
            stages.push(stage);
            Expr::Var(name)
        }
        Expr::Number(_) | Expr::Var(_) => expr.clone(),
        Expr::AuditTrace(inner) => Expr::AuditTrace(sub(inner, shapes, streams, stages, made, replays)?),
        Expr::Builtin { op, operand } => Expr::Builtin { op: *op, operand: sub(operand, shapes, streams, stages, made, replays)? },
        Expr::Lift { axis, operand } => Expr::Lift { axis: *axis, operand: sub(operand, shapes, streams, stages, made, replays)? },
        Expr::BinaryOp { op, lhs, rhs } => Expr::BinaryOp {
            op: op.clone(),
            lhs: sub(lhs, shapes, streams, stages, made, replays)?,
            rhs: sub(rhs, shapes, streams, stages, made, replays)?,
        },
        // Anything else keeps its shape here and is judged by `sources`.
        other => other.clone(),
    })
}

/// The origin a stage's cells have: its sources' (they share one).
fn stage_origin(stage: &Stage, streams: &BTreeMap<String, Stream>) -> Origin {
    stage
        .chains
        .keys()
        .next()
        .map(|s| streams[s].origin.clone())
        .unwrap_or_else(|| Origin("constant".to_string()))
}

/// Lay out one stage: which streams it reads, how deep its lines are, what
/// it computes.
#[allow(clippy::too_many_arguments)]
fn plan_stage(
    src: &Expr,
    target: &str,
    shape: Vec<usize>,
    kind: Kind,
    line: usize,
    tau: f64,
    numbers: Numbers,
    shapes: &mut BTreeMap<String, Vec<usize>>,
    streams: &mut BTreeMap<String, Stream>,
    index: usize,
    replays: &mut Vec<Replay>,
) -> Result<Stage> {
    let (srcs, reach) = sources(src, shapes, line)?;
    if srcs.is_empty() {
        return Err(unsupported("a flow with no space in it", line));
    }
    let mut lifts_of: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    usage(src, line, &mut Vec::new(), &mut lifts_of)?;
    // Main sources have the result's shape and stream; broadcast ones are
    // read by place.
    let mut main: Vec<String> = Vec::new();
    let mut broadcast: BTreeMap<String, BroadcastRead> = BTreeMap::new();
    for s in &srcs {
        let source_shape = shapes.get(s).cloned().ok_or_else(|| HarmonyDisruption::SpaceErr {
            space_name: s.clone(),
            line,
        })?;
        if !streams.contains_key(s) {
            return Err(HarmonyDisruption::SpaceErr {
                space_name: s.clone(),
                line,
            });
        }
        let lifts = lifts_of.get(s).cloned().unwrap_or_default();
        if lifts.is_empty() && source_shape == shape {
            main.push(s.clone());
            continue;
        }
        let (view, lifted) = lifted_view(&source_shape, &lifts)
            .ok_or_else(|| unsupported(&format!("the lift of `{s}`"), line))?;
        let stretches = view.len() == shape.len()
            && view.iter().zip(&shape).all(|(v, r)| v == r || *v == 1);
        if !stretches {
            return Err(unsupported(
                &format!("`{s}` viewed as {view:?} against a flow of {shape:?}"),
                line,
            ));
        }
        broadcast.insert(
            s.clone(),
            BroadcastRead {
                memory: format!("m{index}_{}", ident(s)),
                source_shape,
                lifted,
            },
        );
    }
    let origins: BTreeSet<Origin> = main.iter().map(|s| streams[s].origin.clone()).collect();
    let base = main.iter().map(|s| streams[s].latency).max().unwrap_or(1);
    let sparse = origins.iter().any(|o| o.0.starts_with("fold"));
    let uneven = sparse && main.iter().any(|s| streams[s].latency != base);
    let replay = !broadcast.is_empty() || origins.len() > 1 || uneven;

    // A replay captures every source and streams the main ones back under
    // aliases; the expression reads the aliases.
    type Chains = BTreeMap<String, (String, String, usize, usize, usize)>;
    let (src, chains, timing, base): (Expr, Chains, Option<String>, usize) = if replay {
        let mut captured_main: Vec<(String, Stream)> = Vec::new();
        let mut renamed: BTreeMap<String, String> = BTreeMap::new();
        let cells = shape.iter().product::<usize>().max(1);
        for s in &main {
            let alias = format!("{s}⟳r{index}");
            shapes.insert(alias.clone(), shapes[s].clone());
            streams.insert(
                alias.clone(),
                Stream {
                    value: format!("rr{index}_{}", ident(s)),
                    valid: format!("rv{index}"),
                    latency: 1,
                    origin: Origin(format!("replay{index}")),
                    cells,
                },
            );
            captured_main.push((s.clone(), streams[s].clone()));
            renamed.insert(s.clone(), alias);
        }
        let captured_broadcast: Vec<(String, Stream)> =
            broadcast.keys().map(|s| (s.clone(), streams[s].clone())).collect();
        replays.push(Replay {
            id: index,
            cells,
            main: captured_main,
            broadcast: captured_broadcast,
        });
        let src = rename(src, &|name: &str| renamed.get(name).cloned().unwrap_or_else(|| name.to_string()));
        let chains: Chains = renamed
            .values()
            .map(|alias| {
                let st = &streams[alias];
                (alias.clone(), (st.value.clone(), st.valid.clone(), st.latency, 2 * reach + 1, reach))
            })
            .collect();
        let timing = if renamed.is_empty() { Some(format!("rv{index}")) } else { None };
        (src, chains, timing, 1)
    } else {
        let chains: Chains = main
            .iter()
            .map(|s| {
                let st = &streams[s];
                (s.clone(), (st.value.clone(), st.valid.clone(), st.latency, 2 * reach + 1, reach))
            })
            .collect();
        (src.clone(), chains, None, base)
    };
    let stage_latency = base + reach + 2;
    let lowering = Lowering {
        shapes,
        chains: &chains,
        shape: &shape,
        stage: index,
        tau,
        line,
        numbers,
        broadcast: &broadcast,
    };
    let expr = lowering.lower(&src)?;
    Ok(Stage {
        index,
        target: target.to_string(),
        shape,
        latency: stage_latency,
        reach,
        chains,
        expr,
        kind,
        round_reset: None,
        timing,
    })
}

/// A replay's memories, capture and read-out, ahead of its stage.
fn emit_replay(sv: &mut String, r: &Replay, numbers: Numbers) {
    let k = r.id;
    let ty = numbers.ty();
    let _ = writeln!(sv);
    let _ = writeln!(sv, "  // ---- replay for stage {k}: {} cells streamed back out of memory", r.cells);
    let all: Vec<&(String, Stream)> = r.main.iter().chain(r.broadcast.iter()).collect();
    for (name, stream) in &all {
        let sid = ident(name);
        let _ = writeln!(sv, "  {ty} m{k}_{sid} [0:{}];  // {name}, captured", stream.cells - 1);
        let _ = writeln!(sv, "  logic [{}:0] cap{k}_{sid};", bits(stream.cells) - 1);
    }
    for (name, _) in &r.main {
        let _ = writeln!(sv, "  {ty} rr{k}_{};  // {name} streamed back", ident(name));
    }
    let _ = writeln!(sv, "  logic rv{k};");
    let _ = writeln!(sv, "  logic [1:0] rst{k};  // 0 capture, 1 read, 2 done");
    let _ = writeln!(sv, "  logic [{}:0] ri{k};", bits(r.cells) - 1);
    let _ = writeln!(sv, "  always_ff @(posedge clk) begin : replay{k}");
    let _ = writeln!(sv, "    logic all_captured;");
    let _ = writeln!(sv, "    if (rst) begin");
    let zero: Vec<String> = all.iter().map(|(n, _)| format!("cap{k}_{} <= 0", ident(n))).collect();
    let _ = writeln!(sv, "      {}; rv{k} <= 1'b0; rst{k} <= 0; ri{k} <= 0;", zero.join("; "));
    let _ = writeln!(sv, "    end else begin");
    let _ = writeln!(sv, "      rv{k} <= 1'b0;");
    for (name, stream) in &all {
        let sid = ident(name);
        let _ = writeln!(
            sv,
            "      if (rst{k} == 0 && {} && cap{k}_{sid} < {}) begin m{k}_{sid}[cap{k}_{sid}] <= {}; cap{k}_{sid} <= cap{k}_{sid} + 1; end",
            stream.valid, stream.cells, stream.value
        );
    }
    let done: Vec<String> = all.iter().map(|(n, st)| format!("(cap{k}_{} == {})", ident(n), st.cells)).collect();
    let _ = writeln!(sv, "      all_captured = {};", done.join(" && "));
    let _ = writeln!(sv, "      case (rst{k})");
    let _ = writeln!(sv, "        0: if (all_captured) begin rst{k} <= 1; ri{k} <= 0; end");
    let _ = writeln!(sv, "        1: begin");
    for (name, _) in &r.main {
        let sid = ident(name);
        let _ = writeln!(sv, "          rr{k}_{sid} <= m{k}_{sid}[ri{k}];");
    }
    let _ = writeln!(sv, "          rv{k} <= 1'b1;");
    let _ = writeln!(sv, "          if (ri{k} == {}) rst{k} <= 2; else ri{k} <= ri{k} + 1;", r.cells - 1);
    let _ = writeln!(sv, "        end");
    let _ = writeln!(sv, "        default: ;");
    let _ = writeln!(sv, "      endcase");
    let _ = writeln!(sv, "    end");
    let _ = writeln!(sv, "  end");
}

/// Whether a fold or scan sits anywhere in the expression.
fn has_fold(expr: &Expr) -> bool {
    match expr {
        Expr::Reduce { .. } | Expr::Scan { .. } => true,
        Expr::Number(_) | Expr::Var(_) => false,
        Expr::AuditTrace(inner)
        | Expr::Builtin { operand: inner, .. }
        | Expr::Shift { operand: inner, .. }
        | Expr::Index { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. } => has_fold(inner),
        Expr::BinaryOp { lhs, rhs, .. } | Expr::Gather { index: lhs, operand: rhs } => has_fold(lhs) || has_fold(rhs),
        Expr::Call { args, .. } => args.iter().any(has_fold),
    }
}

/// Whether a lift sits anywhere in the expression.
fn has_lift(expr: &Expr) -> bool {
    match expr {
        Expr::Lift { .. } => true,
        Expr::AuditTrace(inner) | Expr::Builtin { operand: inner, .. } | Expr::Shift { operand: inner, .. } => has_lift(inner),
        Expr::BinaryOp { lhs, rhs, .. } => has_lift(lhs) || has_lift(rhs),
        _ => false,
    }
}

/// The expression with every space name mapped through `f`.
fn rename(expr: &Expr, f: &dyn Fn(&str) -> String) -> Expr {
    let sub = |e: &Expr| Box::new(rename(e, f));
    match expr {
        Expr::Var(name) if !is_tau(name) => Expr::Var(f(name)),
        Expr::Var(_) | Expr::Number(_) => expr.clone(),
        Expr::AuditTrace(inner) => Expr::AuditTrace(sub(inner)),
        Expr::Builtin { op, operand } => Expr::Builtin { op: *op, operand: sub(operand) },
        Expr::Shift { dir, axis, operand } => Expr::Shift { dir: *dir, axis: *axis, operand: sub(operand) },
        Expr::Index { axis, operand } => Expr::Index { axis: *axis, operand: sub(operand) },
        Expr::Lift { axis, operand } => Expr::Lift { axis: *axis, operand: sub(operand) },
        Expr::BinaryOp { op, lhs, rhs } => Expr::BinaryOp { op: op.clone(), lhs: sub(lhs), rhs: sub(rhs) },
        other => other.clone(),
    }
}

/// A loop's memories and the signals its update stage reads, declared ahead
/// of the stage.
fn emit_loop_declarations(sv: &mut String, l: &Loop, numbers: Numbers) {
    let k = l.id;
    let n = l.cells;
    let ty = numbers.ty();
    let _ = writeln!(sv);
    let _ = writeln!(sv, "  // ---- loop {k}: ... ⇒ {} ({n} cells, at most {} sweeps)", l.target, l.cap);
    for (name, _) in &l.sources {
        let sid = ident(name);
        if *name == l.target {
            let _ = writeln!(sv, "  {ty} m{k}_{sid}_a [0:{}];  // {name}: the round's grid", n - 1);
            let _ = writeln!(sv, "  {ty} m{k}_{sid}_b [0:{}];  // {name}: the grid being written", n - 1);
        } else {
            let _ = writeln!(sv, "  {ty} m{k}_{sid} [0:{}];  // {name}, captured", n - 1);
        }
        let _ = writeln!(sv, "  logic [{}:0] cap{k}_{sid};  // cells captured", bits(n) - 1);
        let _ = writeln!(sv, "  {ty} rr{k}_{sid};  // the round stream of {name}");
    }
    let _ = writeln!(sv, "  logic rv{k};  // the round streams' valid");
    let _ = writeln!(sv, "  logic rs{k};  // a round starts: the update stage begins afresh");
    let _ = writeln!(sv, "  logic cur{k};  // which buffer of {} the round reads", l.target);
    let _ = writeln!(sv, "  logic [{}:0] i{k}, w{k};", bits(n) - 1);
    let _ = writeln!(sv, "  logic [{}:0] round{k}, sw{k};", bits(l.cap + 1) - 1);
    let _ = writeln!(sv, "  logic cv{k};");
    let _ = writeln!(sv, "  {ty} d{k};  // the largest move this round");
    let _ = writeln!(sv, "  logic [2:0] ls{k};  // 0 capture, 1 read, 2 drain, 3 emit, 4 done");
    let _ = writeln!(sv, "  {ty} px{k};  // {} after the loop, streamed out", l.target);
    let _ = writeln!(sv, "  logic pv{k};");
}

/// The loop's controller: capture, rounds, the decision, the read-out.
fn emit_loop_logic(sv: &mut String, l: &Loop, update: &Stage, numbers: Numbers) {
    let k = l.id;
    let n = l.cells;
    let ty = numbers.ty();
    let tid = ident(&l.target);
    let upd_value = format!("st{}_{}", update.index, ident(&update.target));
    let upd_valid = format!("v{}_out", update.index);
    let _ = writeln!(sv, "  always_ff @(posedge clk) begin : loop{k}");
    let _ = writeln!(sv, "    {ty} old, diff;");
    let _ = writeln!(sv, "    logic all_captured, settled, capped;");
    let _ = writeln!(sv, "    if (rst) begin");
    for (name, _) in &l.sources {
        let _ = writeln!(sv, "      cap{k}_{} <= 0;", ident(name));
    }
    let _ = writeln!(
        sv,
        "      rv{k} <= 1'b0; rs{k} <= 1'b0; cur{k} <= 1'b0; i{k} <= 0; w{k} <= 0; round{k} <= 0; sw{k} <= 0; cv{k} <= 1'b1; d{k} <= {}; ls{k} <= 0; pv{k} <= 1'b0;",
        numbers.lit(0.0)
    );
    let _ = writeln!(sv, "    end else begin");
    let _ = writeln!(sv, "      rv{k} <= 1'b0; rs{k} <= 1'b0; pv{k} <= 1'b0;");
    // Capture, always on: a source's cells land in its memory as they come.
    for (name, stream) in &l.sources {
        let sid = ident(name);
        let mem = if *name == l.target { format!("m{k}_{sid}_a") } else { format!("m{k}_{sid}") };
        let _ = writeln!(sv, "      if (ls{k} == 0 && {} && cap{k}_{sid} < {n}) begin {mem}[cap{k}_{sid}] <= {}; cap{k}_{sid} <= cap{k}_{sid} + 1; end", stream.valid, stream.value);
    }
    let all: Vec<String> = l.sources.iter().map(|(name, _)| format!("(cap{k}_{} == {n})", ident(name))).collect();
    let _ = writeln!(sv, "      all_captured = {};", all.join(" && "));
    let _ = writeln!(sv, "      case (ls{k})");
    let _ = writeln!(sv, "        0: if (all_captured) begin ls{k} <= 1; i{k} <= 0; w{k} <= 0; d{k} <= {}; rs{k} <= 1'b1; end", numbers.lit(0.0));
    // Read: one cell per clock from every source's memory.
    let _ = writeln!(sv, "        1: begin");
    for (name, _) in &l.sources {
        let sid = ident(name);
        if *name == l.target {
            let _ = writeln!(sv, "          rr{k}_{sid} <= cur{k} ? m{k}_{sid}_b[i{k}] : m{k}_{sid}_a[i{k}];");
        } else {
            let _ = writeln!(sv, "          rr{k}_{sid} <= m{k}_{sid}[i{k}];");
        }
    }
    let _ = writeln!(sv, "          rv{k} <= 1'b1;");
    let _ = writeln!(sv, "          if (i{k} == {}) ls{k} <= 2; else i{k} <= i{k} + 1;", n - 1);
    let _ = writeln!(sv, "        end");
    // Drain: the update stage finishes; when every cell is written, decide.
    let _ = writeln!(sv, "        2: if (w{k} == {n}) begin");
    let _ = writeln!(sv, "          settled = (d{k} <= {});", numbers.lit(l.tau));
    let _ = writeln!(sv, "          capped = (round{k} + 1 >= {});", l.cap);
    let _ = writeln!(sv, "          sw{k} <= sw{k} + 1;");
    let _ = writeln!(sv, "          if (settled || capped) begin");
    let _ = writeln!(sv, "            if (capped) cv{k} <= 1'b0;");
    let _ = writeln!(sv, "            cur{k} <= ~cur{k}; ls{k} <= 3; i{k} <= 0;");
    let _ = writeln!(sv, "          end else begin");
    let _ = writeln!(sv, "            cur{k} <= ~cur{k}; round{k} <= round{k} + 1; i{k} <= 0; w{k} <= 0; d{k} <= {}; rs{k} <= 1'b1; ls{k} <= 1;", numbers.lit(0.0));
    let _ = writeln!(sv, "          end");
    let _ = writeln!(sv, "        end");
    // Emit: the settled grid streamed out for what follows.
    let _ = writeln!(sv, "        3: begin");
    let _ = writeln!(sv, "          px{k} <= cur{k} ? m{k}_{tid}_b[i{k}] : m{k}_{tid}_a[i{k}];");
    let _ = writeln!(sv, "          pv{k} <= 1'b1;");
    let _ = writeln!(sv, "          if (i{k} == {}) ls{k} <= 4; else i{k} <= i{k} + 1;", n - 1);
    let _ = writeln!(sv, "        end");
    let _ = writeln!(sv, "        default: ;");
    let _ = writeln!(sv, "      endcase");
    // The writer: the update's cells into the other buffer, measuring the move.
    let _ = writeln!(sv, "      if ((ls{k} == 1 || ls{k} == 2) && {upd_valid}) begin");
    let _ = writeln!(sv, "        old = cur{k} ? m{k}_{tid}_b[w{k}] : m{k}_{tid}_a[w{k}];");
    let _ = writeln!(sv, "        if (cur{k}) m{k}_{tid}_a[w{k}] <= {upd_value}; else m{k}_{tid}_b[w{k}] <= {upd_value};");
    match numbers {
        Numbers::Real => {
            let _ = writeln!(sv, "        diff = rho_abs({upd_value} - old);");
        }
        Numbers::Fixed { .. } => {
            let _ = writeln!(sv, "        diff = rho_abs(fx_sub({upd_value}, old));");
        }
    }
    let _ = writeln!(sv, "        if (diff > d{k}) d{k} <= diff;");
    let _ = writeln!(sv, "        w{k} <= w{k} + 1;");
    let _ = writeln!(sv, "      end");
    let _ = writeln!(sv, "    end");
    let _ = writeln!(sv, "  end");
}

/// The SystemVerilog of one stage: alignment, lines, counters, flush, the
/// computation, and for a fold the accumulators.
fn emit_stage(sv: &mut String, stage: &Stage, streams: &BTreeMap<String, Stream>, numbers: Numbers) {
    let ty = numbers.ty();
    let k = stage.index;
    let tid = ident(&stage.target);
    let r = stage.reach;
    let cells = stage.shape.iter().product::<usize>().max(1);
    let base = stage.chains.values().map(|c| c.2).max().unwrap_or(1);
    // What starts the stage afresh: reset, and for a loop's update stage
    // every round.
    let clear = match &stage.round_reset {
        Some(rs) => format!("(rst || {rs})"),
        None => "rst".to_string(),
    };
    let _ = writeln!(sv);
    let _ = writeln!(
        sv,
        "  // ---- stage {k}: {} (sweeps {:?}, reach {r}), latency {}",
        match stage.kind {
            Kind::Flow => format!("-> {}", stage.target),
            Kind::Fold { running, .. } => format!("{} {}", if running { "scan" } else { "fold" }, stage.target),
        },
        stage.shape,
        stage.latency
    );
    // Sources aligned to the latest of them, value and valid together.
    let first_src = stage.chains.keys().next().cloned();
    for (src, (value, valid, latency, _, _)) in &stage.chains {
        let sid = ident(src);
        let delay = base - latency;
        if delay == 0 {
            let _ = writeln!(sv, "  {ty} a{k}_{sid};  always_comb a{k}_{sid} = {value};");
            let _ = writeln!(sv, "  logic av{k}_{sid}; assign av{k}_{sid} = {valid};");
        } else {
            let _ = writeln!(sv, "  {ty} ad{k}_{sid} [0:{}];", delay - 1);
            let _ = writeln!(sv, "  logic adv{k}_{sid} [0:{}];", delay - 1);
            let _ = writeln!(sv, "  always_ff @(posedge clk) begin");
            let _ = writeln!(sv, "    ad{k}_{sid}[0] <= {value}; adv{k}_{sid}[0] <= {clear} ? 1'b0 : {valid};");
            let _ = writeln!(sv, "    for (int i = 1; i < {delay}; i++) begin ad{k}_{sid}[i] <= ad{k}_{sid}[i - 1]; adv{k}_{sid}[i] <= {clear} ? 1'b0 : adv{k}_{sid}[i - 1]; end");
            let _ = writeln!(sv, "  end");
            let _ = writeln!(sv, "  {ty} a{k}_{sid};  always_comb a{k}_{sid} = ad{k}_{sid}[{}];", delay - 1);
            let _ = writeln!(sv, "  logic av{k}_{sid}; assign av{k}_{sid} = adv{k}_{sid}[{}];", delay - 1);
        }
    }
    // A stage with no streamed source is paced by its replay's valid.
    let (src_cells, pace) = match &first_src {
        Some(name) => (streams[name].cells, format!("av{k}_{}", ident(name))),
        None => (cells, stage.timing.clone().unwrap_or_else(|| "1'b0".to_string())),
    };
    // Counters as wide as what they count, and the cell's coordinates kept
    // as counters with a carry rather than divided out of an index: what
    // sets the clock rate once the arithmetic is narrow.
    let cw = bits(src_cells + r + 3);
    let _ = writeln!(sv, "  // cells entered, real cells entered, zeros pushed through after the last, cells computed");
    let _ = writeln!(sv, "  logic [{}:0] n{k}_in, n{k}_real, n{k}_flushed, nc{k};", cw - 1);
    let _ = writeln!(sv, "  logic v{k}_src, fl{k}, en{k};");
    let _ = writeln!(sv, "  assign v{k}_src = {pace};");
    let _ = writeln!(
        sv,
        "  assign fl{k} = (n{k}_real == {src_cells}) && (n{k}_flushed < {}) && !v{k}_src;",
        r + 1
    );
    let _ = writeln!(sv, "  assign en{k} = v{k}_src || fl{k};");
    for (src, (_, _, _, length, centre)) in &stage.chains {
        let sid = ident(src);
        let l = format!("c{k}_{sid}");
        let _ = writeln!(sv, "  {ty} {l} [0:{}];  // {src}; centre tap {centre}", length - 1);
        let _ = writeln!(sv, "  always_ff @(posedge clk) if (en{k}) begin");
        let _ = writeln!(sv, "    {l}[0] <= v{k}_src ? a{k}_{sid} : {};", numbers.lit(0.0));
        let _ = writeln!(sv, "    for (int i = 1; i < {length}; i++) {l}[i] <= {l}[i - 1];");
        let _ = writeln!(sv, "  end");
    }
    let rank = stage.shape.len();
    let geometry: Vec<(usize, usize)> = (0..rank)
        .map(|a| axis_geometry(&stage.shape, Some(a)).unwrap_or((1, 1)))
        .collect();
    for (a, (_, extent)) in geometry.iter().enumerate() {
        let _ = writeln!(sv, "  logic [{}:0] pc{k}_{a};  // coordinate along axis {a}, 0..{}", bits(extent.max(&1) - 1) - 1, extent - 1);
    }
    let _ = writeln!(sv, "  {ty} st{k}_{tid};");
    let _ = writeln!(sv, "  logic v{k}_out;");
    if let Kind::Fold { axis, .. } = stage.kind {
        let (stride, _) = geometry[axis];
        let _ = writeln!(sv, "  {ty} acc{k} [0:{}];  // one accumulator per line across axis {axis}", stride - 1);
        let _ = writeln!(sv, "  logic [{}:0] tc{k};  // which line, 0..{}", bits(stride.max(1) - 1) - 1, stride - 1);
    }
    let _ = writeln!(sv, "  always_ff @(posedge clk) begin : stage{k}");
    let _ = writeln!(sv, "    longint pos [0:{}];", rank.max(1) - 1);
    let _ = writeln!(sv, "    {ty} x;");
    let _ = writeln!(sv, "    if ({clear}) begin");
    let mut zeroed: Vec<String> = vec![
        format!("n{k}_in <= 0"),
        format!("n{k}_real <= 0"),
        format!("n{k}_flushed <= 0"),
        format!("nc{k} <= 0"),
        format!("v{k}_out <= 1'b0"),
        format!("st{k}_{tid} <= {}", numbers.lit(0.0)),
    ];
    for a in 0..rank {
        zeroed.push(format!("pc{k}_{a} <= 0"));
    }
    if matches!(stage.kind, Kind::Fold { .. }) {
        zeroed.push(format!("tc{k} <= 0"));
    }
    let _ = writeln!(sv, "      {};", zeroed.join("; "));
    let _ = writeln!(sv, "    end else begin");
    let _ = writeln!(sv, "      v{k}_out <= 1'b0;");
    let _ = writeln!(sv, "      if (v{k}_src) n{k}_real <= n{k}_real + 1;");
    let _ = writeln!(sv, "      if (fl{k}) n{k}_flushed <= n{k}_flushed + 1;");
    let _ = writeln!(sv, "      if (en{k}) begin");
    let _ = writeln!(sv, "        n{k}_in <= n{k}_in + 1;");
    for a in 0..rank {
        let _ = writeln!(sv, "        pos[{a}] = pc{k}_{a};");
    }
    let _ = writeln!(sv, "        if (n{k}_in >= {} && nc{k} < {cells}) begin", r + 1);
    let _ = writeln!(sv, "          x = {};", stage.expr);
    match stage.kind {
        Kind::Flow => {
            let _ = writeln!(sv, "          st{k}_{tid} <= x;");
            let _ = writeln!(sv, "          v{k}_out <= 1'b1;");
        }
        Kind::Fold { op, axis, running } => {
            let (stride, extent) = geometry[axis];
            let fixed = matches!(numbers, Numbers::Fixed { .. });
            let step = match (op, fixed) {
                (FoldOp::Sum, false) => "(a + x)",
                (FoldOp::Product, false) => "(a * x)",
                (FoldOp::Sum, true) => "fx_add(a, x)",
                (FoldOp::Product, true) => "fx_mul(a, x)",
                (FoldOp::Max, _) => "((x > a) ? x : a)",
                (FoldOp::Min, _) => "((x < a) ? x : a)",
            };
            let _ = writeln!(sv, "          begin : accumulate");
            let _ = writeln!(sv, "            {ty} a, next;");
            let _ = writeln!(sv, "            a = (pc{k}_{axis} == 0) ? {} : acc{k}[tc{k}];", numbers.lit(op.identity()));
            let _ = writeln!(sv, "            next = {step};");
            let _ = writeln!(sv, "            acc{k}[tc{k}] <= next;");
            if running {
                let _ = writeln!(sv, "            st{k}_{tid} <= next;");
                let _ = writeln!(sv, "            v{k}_out <= 1'b1;");
            } else {
                let _ = writeln!(sv, "            if (pc{k}_{axis} == {}) begin st{k}_{tid} <= next; v{k}_out <= 1'b1; end", extent - 1);
            }
            let _ = writeln!(sv, "            tc{k} <= (tc{k} == {}) ? 0 : tc{k} + 1;", stride - 1);
            let _ = writeln!(sv, "          end");
        }
    }
    let _ = writeln!(sv, "          nc{k} <= nc{k} + 1;");
    // The coordinates advance with a carry from the innermost axis out.
    if rank > 0 {
        let mut text = String::new();
        let mut indent = "          ".to_string();
        for a in (0..rank).rev() {
            let extent = geometry[a].1;
            let _ = writeln!(text, "{indent}if (pc{k}_{a} != {}) pc{k}_{a} <= pc{k}_{a} + 1;", extent.max(1) - 1);
            let _ = writeln!(text, "{indent}else begin");
            let _ = writeln!(text, "{indent}  pc{k}_{a} <= 0;");
            indent.push_str("  ");
        }
        for _ in 0..rank {
            indent.truncate(indent.len() - 2);
            let _ = writeln!(text, "{indent}end");
        }
        sv.push_str(&text);
    }
    let _ = writeln!(sv, "        end");
    let _ = writeln!(sv, "      end");
    let _ = writeln!(sv, "    end");
    let _ = writeln!(sv, "  end");
}

/// The helpers every module carries: the language's operations that are not
/// one SystemVerilog operator, written to give the reference's bits.
fn functions(numbers: Numbers) -> String {
    match numbers {
        Numbers::Real => REAL_FUNCTIONS.to_string(),
        Numbers::Fixed { width, frac } => {
            let w = width;
            let f = frac;
            let one = numbers.lit(1.0);
            let zero = numbers.lit(0.0);
            let step_mask = format!("{w}'sh{:0wd$X}", ((1u128 << f) - 1) as u64 & mask(w), wd = w.div_ceil(4) as usize);
            // No casts: Yosys and Verilator agree on assignments, which
            // sign-extend into a wider variable and truncate into a narrower.
            format!(
                r#"
  // Fixed point Q{w}.{f}: what an integer datapath does. The reference is
  // numeric::Fixed in the compiler: sums wrap, products floor, quotients
  // truncate toward zero, a division by zero is zero.
  typedef logic signed [{wm}:0] cell_t;
  typedef logic signed [{w2m}:0] wide_t;
  function automatic cell_t fx_add(input cell_t a, input cell_t b);
    fx_add = a + b;
  endfunction
  function automatic cell_t fx_sub(input cell_t a, input cell_t b);
    fx_sub = a - b;
  endfunction
  function automatic cell_t fx_mul(input cell_t a, input cell_t b);
    wide_t x, y, p;
    x = a;
    y = b;
    p = x * y;
    fx_mul = p >>> {f};
  endfunction
  function automatic cell_t fx_div(input cell_t a, input cell_t b);
    wide_t n, d, q;
    if (b == {zero}) fx_div = {zero};
    else begin
      n = a;
      n = n <<< {f};
      d = b;
      q = n / d;
      fx_div = q;
    end
  endfunction
  function automatic cell_t fx_shl(input cell_t a, input integer k);
    fx_shl = a <<< k;
  endfunction
  function automatic cell_t fx_shr(input cell_t a, input integer k);
    fx_shr = a >>> k;
  endfunction
  // a / 2^k truncated toward zero: a negative value is biased up first.
  function automatic cell_t fx_div_pow2(input cell_t a, input integer k);
    cell_t bias, biased;
    bias = ({w}'sh1 <<< k) - {w}'sh1;
    biased = (a < {zero}) ? a + bias : a;
    fx_div_pow2 = biased >>> k;
  endfunction
  function automatic cell_t fx_floor(input cell_t a);
    fx_floor = a & ~{step_mask};
  endfunction
  function automatic cell_t fx_ceil(input cell_t a);
    cell_t r;
    r = a + {step_mask};
    fx_ceil = r & ~{step_mask};
  endfunction
  // A whole number as a cell: the coordinate `⍳` reads.
  function automatic cell_t fx_int(input longint i);
    wide_t x;
    x = i;
    x = x <<< {f};
    fx_int = x;
  endfunction
  function automatic cell_t rho_mask(input logic holds, input cell_t l);
    rho_mask = holds ? l : {zero};
  endfunction
  function automatic cell_t rho_abs(input cell_t x);
    cell_t r;
    r = -x;
    rho_abs = (x < {zero}) ? r : x;
  endfunction
  function automatic cell_t rho_ind(input cell_t x);
    rho_ind = (x != {zero}) ? {one} : {zero};
  endfunction
  function automatic cell_t rho_extreme(input cell_t l, input cell_t r, input logic greater);
    rho_extreme = greater ? ((l > r) ? l : r) : ((l < r) ? l : r);
  endfunction
  // APL's residue: B - A * floor(B / A); 0 | B is B.
  function automatic cell_t rho_residue(input cell_t a, input cell_t b);
    cell_t q, rem;
    q = fx_floor(fx_div(b, a));
    rem = fx_sub(b, fx_mul(a, q));
    rho_residue = (a == {zero}) ? b : rem;
  endfunction
  // The roll over the cell's bits, sign-extended to 64: the top {f} bits of
  // the hash are the fraction of a number in [0, 1).
  function automatic cell_t rho_roll(input cell_t x);
    longint signed xs;
    longint unsigned z;
    xs = x;
    z = xs;
    z = z + 64'h9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) * 64'hBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 64'h94D049BB133111EB;
    z = z ^ (z >> 31);
    rho_roll = z >> {shift};
  endfunction
"#,
                wm = w - 1,
                w2m = 2 * w - 1,
                shift = 64 - f
            )
        }
    }
}

const REAL_FUNCTIONS: &str = r#"
  // A comparison masks: the left value where it holds, zero elsewhere.
  function automatic real rho_mask(input logic holds, input real l);
    return holds ? l : $bitstoreal(64'h0);
  endfunction
  // |x| by clearing the sign, so -0 becomes +0 as fabs has it.
  function automatic real rho_abs(input real x);
    return $bitstoreal($realtobits(x) & 64'h7FFFFFFFFFFFFFFF);
  endfunction
  // 1 where the value is not zero (a NaN is not zero), 0 at zero.
  function automatic real rho_ind(input real x);
    return (x != $bitstoreal(64'h0)) ? $bitstoreal(64'h3FF0000000000000) : $bitstoreal(64'h0);
  endfunction
  // IEEE 754-2019 maximum / minimum: a NaN propagates, -0 orders below +0.
  function automatic real rho_extreme(input real l, input real r, input logic greater);
    real by_value, by_sign, ordered;
    if (l != l) return l;
    if (r != r) return r;
    by_value = greater ? ((l > r) ? l : r) : ((l < r) ? l : r);
    by_sign = greater ? ((1.0 / l > 1.0 / r) ? l : r) : ((1.0 / l < 1.0 / r) ? l : r);
    ordered = (l == r) ? by_sign : by_value;
    return ordered;
  endfunction
  // APL's residue: B - A * floor(B / A); 0 | B is B.
  function automatic real rho_residue(input real a, input real b);
    real q, rem;
    q = $floor(b / a);
    rem = b - a * q;
    return (a == $bitstoreal(64'h0)) ? b : rem;
  endfunction
  // The roll: splitmix64's finaliser over the bits, the top 53 scaled into [0, 1).
  function automatic real rho_roll(input real x);
    longint unsigned z;
    z = (x != x) ? 64'd0 : $realtobits(x);
    z = z + 64'h9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) * 64'hBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 64'h94D049BB133111EB;
    z = z ^ (z >> 31);
    return real'(z >> 11) * $bitstoreal(64'h3CA0000000000000);
  endfunction
"#;

/// Write the circuit to `dir` as rho_kernel.sv and rho_harness.cpp.
pub fn write(circuit: &Circuit, dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("rho_kernel.sv"), &circuit.module)?;
    std::fs::write(dir.join("rho_harness.cpp"), &circuit.harness)?;
    Ok(())
}

/// The verilator to run: $VERILATOR, else `verilator` on the path.
pub fn verilator() -> String {
    std::env::var("VERILATOR").unwrap_or_else(|_| "verilator".to_string())
}

/// What synthesis and place-and-route reported for a circuit.
#[derive(Debug, Default, Clone)]
pub struct Synthesis {
    /// Cell counts from Yosys `stat`, by cell type.
    pub cells: BTreeMap<String, u64>,
    /// nextpnr's maximum clock frequency in MHz, when it ran.
    pub fmax_mhz: Option<f64>,
    pub log: String,
}

/// Synthesise the circuit in `dir` for an FPGA family with Yosys ($YOSYS
/// or `yosys`), and place and route it with nextpnr when that is found
/// ($NEXTPNR_ICE40 / $NEXTPNR_ECP5 or on the path) to get a clock rate.
/// `family` is "ice40" (hx8k) or "ecp5" (85k). Only a fixed-point circuit
/// synthesises; `real` cells stop at Yosys with an error.
pub fn synthesize(dir: &Path, family: &str) -> std::io::Result<Synthesis> {
    let yosys = std::env::var("YOSYS").unwrap_or_else(|_| "yosys".to_string());
    let json = dir.join("rho_kernel.json");
    let script = format!(
        "read_verilog -sv {}; synth_{family} -top rho_kernel -json {}; stat",
        dir.join("rho_kernel.sv").display(),
        json.display()
    );
    // Not quiet: `stat` writes to the log, which is what is parsed.
    let out = Command::new(&yosys).args(["-Q", "-T", "-p", &script]).output()?;
    let mut log = String::from_utf8_lossy(&out.stdout).to_string();
    log.push_str(&String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        return Err(std::io::Error::other(format!("yosys failed:\n{log}")));
    }
    // `stat` lines read `     567   SB_LUT4` (or `TRELLIS_FF`, `$_...`).
    let mut cells = BTreeMap::new();
    for line in log.lines() {
        let mut words = line.split_whitespace();
        if let (Some(count), Some(kind), None) = (words.next(), words.next(), words.next()) {
            if let Ok(n) = count.parse::<u64>() {
                if kind.chars().next().is_some_and(|c| c.is_ascii_uppercase() || c == '$') {
                    *cells.entry(kind.to_string()).or_insert(0) = n;
                }
            }
        }
    }
    let (tool_env, tool, device_args): (&str, &str, Vec<&str>) = match family {
        "ecp5" => ("NEXTPNR_ECP5", "nextpnr-ecp5", vec!["--85k", "--lpf-allow-unconstrained"]),
        _ => ("NEXTPNR_ICE40", "nextpnr-ice40", vec!["--hx8k", "--package", "ct256", "--pcf-allow-unconstrained"]),
    };
    let nextpnr = std::env::var(tool_env).unwrap_or_else(|_| tool.to_string());
    let mut fmax_mhz = None;
    if let Ok(pnr) = Command::new(&nextpnr)
        .args(&device_args)
        .args(["--json", &json.display().to_string(), "--freq", "1"])
        .output()
    {
        let text = format!("{}{}", String::from_utf8_lossy(&pnr.stdout), String::from_utf8_lossy(&pnr.stderr));
        // "Max frequency for clock 'clk...': 61.23 MHz (PASS at 1.00 MHz)"
        for line in text.lines().rev() {
            if line.contains("Max frequency for clock") {
                if let Some((_, rest)) = line.rsplit_once(':') {
                    if let Some(num) = rest.split_whitespace().next() {
                        fmax_mhz = num.parse().ok();
                        break;
                    }
                }
            }
        }
        log.push_str(&text);
    }
    Ok(Synthesis { cells, fmax_mhz, log })
}

/// Build the circuit in `dir` with Verilator and run it over `inputs` (in
/// port order). Returns OUTPUT's cells and the clocks the run took.
/// Contraction is off in the generated C++ as it is in the kernel, so a
/// multiply-add rounds twice here too.
pub fn simulate(circuit: &Circuit, dir: &Path, inputs: &[Vec<f64>]) -> std::io::Result<Run> {
    write(circuit, dir)?;
    let obj = dir.join("obj");
    let status = Command::new(verilator())
        .args([
            "--cc", "--exe", "--build", "-Wno-fatal", "-O2", "-CFLAGS", "-O2 -ffp-contract=off", "--Mdir",
        ])
        .arg(&obj)
        .arg(dir.join("rho_kernel.sv"))
        .arg(dir.join("rho_harness.cpp"))
        .output()?;
    if !status.status.success() {
        return Err(std::io::Error::other(format!(
            "verilator failed:\n{}\n{}",
            String::from_utf8_lossy(&status.stdout),
            String::from_utf8_lossy(&status.stderr)
        )));
    }
    let mut bytes: Vec<u8> = Vec::new();
    for space in inputs {
        for v in space {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let in_path = dir.join("inputs.bin");
    let out_path = dir.join("output.bin");
    std::fs::write(&in_path, bytes)?;
    let run = Command::new(obj.join("Vrho_kernel")).arg(&in_path).arg(&out_path).output()?;
    if !run.status.success() {
        return Err(std::io::Error::other(format!(
            "the simulation failed:\n{}\n{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        )));
    }
    let stdout = String::from_utf8_lossy(&run.stdout);
    let field = |name: &str| -> u64 {
        stdout
            .split_whitespace()
            .skip_while(|w| *w != name)
            .nth(1)
            .and_then(|w| w.parse().ok())
            .unwrap_or(0)
    };
    let raw = std::fs::read(&out_path)?;
    let output = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(Run {
        output,
        cycles: field("cycles"),
        sweeps: field("sweeps"),
        converged: field("converged") != 0,
    })
}
