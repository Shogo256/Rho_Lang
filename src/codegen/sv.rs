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
//! streamed out for what follows. Not yet: lifts, the turns, `⌷`, a fold
//! inside `⇒`, and a flow that mixes a fold's or a loop's stream with another.

use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

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
#[derive(Clone, PartialEq, Eq, Debug)]
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

fn real_literal(v: f64) -> String {
    format!("$bitstoreal(64'h{:016X})", v.to_bits())
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
                // A coordinate reads nothing; the operand only lends its shape.
                let Expr::Var(_) = &**operand else {
                    return Err(unsupported("`⍳` of a computed value", line));
                };
            }
            // A fold or scan was made a stage of its own before this walk;
            // only its name is left in the expression.
            Expr::Reduce { .. } | Expr::Scan { .. } => return Err(unsupported("a fold inside a fold's own line", line)),
            Expr::Lift { .. } => return Err(unsupported("a lift (`□`)", line)),
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
}

impl Lowering<'_> {
    fn lower(&self, expr: &Expr) -> Result<String> {
        Ok(match expr {
            Expr::Number(v) => real_literal(*v),
            Expr::Var(name) if is_tau(name) => real_literal(self.tau),
            Expr::Var(name) => {
                let (_, _, _, _, centre) = &self.chains[name];
                format!("c{}_{}[{centre}]", self.stage, ident(name))
            }
            Expr::AuditTrace(inner) => self.lower(inner)?,
            Expr::Shift { dir, axis, operand } => {
                let Expr::Var(name) = &**operand else {
                    return Err(unsupported("a shift of a computed value", self.line));
                };
                let shape = &self.shapes[name];
                let (stride, extent) = axis_geometry(shape, *axis).ok_or_else(|| unsupported("an axis past the shape", self.line))?;
                if extent <= 1 {
                    return Ok(real_literal(0.0));
                }
                let a = axis.unwrap_or_else(|| default_axis(shape));
                let (_, _, _, _, centre) = &self.chains[name];
                let (tap, edge) = match dir {
                    ShiftDir::Positive => (centre + stride, 0),
                    ShiftDir::Negative => (centre - stride, extent - 1),
                };
                format!(
                    "((pos[{a}] == {edge}) ? {} : c{}_{}[{tap}])",
                    real_literal(0.0),
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
                format!("real'(pos[{a}])")
            }
            Expr::Builtin { op, operand } => {
                let x = self.lower(operand)?;
                match op {
                    BuiltinOp::Exp => format!("$exp({x})"),
                    BuiltinOp::Log => format!("$ln({x})"),
                    BuiltinOp::Sqrt => format!("$sqrt({x})"),
                    BuiltinOp::Sin => format!("$sin({x})"),
                    BuiltinOp::Cos => format!("$cos({x})"),
                    BuiltinOp::Abs => format!("rho_abs({x})"),
                    BuiltinOp::Indicator => format!("rho_ind({x})"),
                    BuiltinOp::Roll => format!("rho_roll({x})"),
                    BuiltinOp::Floor => format!("$floor({x})"),
                    BuiltinOp::Ceil => format!("$ceil({x})"),
                }
            }
            Expr::BinaryOp { op, lhs, rhs } => {
                let l = self.lower(lhs)?;
                let r = self.lower(rhs)?;
                match op {
                    BinaryOpKind::Add => format!("({l} + {r})"),
                    BinaryOpKind::Sub => format!("({l} - {r})"),
                    BinaryOpKind::Mul => format!("({l} * {r})"),
                    BinaryOpKind::Div => format!("({l} / {r})"),
                    BinaryOpKind::Pow => match whole_exponent(rhs) {
                        // Repeated multiplication, as the kernel and the
                        // interpreter do it, from the same rule.
                        Some(n) => {
                            let steps = n.unsigned_abs();
                            let mut acc = real_literal(1.0);
                            for _ in 0..steps {
                                acc = format!("({acc} * {l})");
                            }
                            if n < 0 {
                                format!("({} / {acc})", real_literal(1.0))
                            } else {
                                acc
                            }
                        }
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
pub fn emit(block: &ToposBlock, tau: f64, cap: Option<usize>) -> Result<Circuit> {
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
                let mut stage = plan_stage(
                    &renamed,
                    &format!("{target}⟳{id}"),
                    shape.clone(),
                    Kind::Flow,
                    line,
                    tau,
                    &shapes,
                    &streams,
                    stages.len(),
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
                let src = extract_folds(src, line, tau, &mut shapes, &mut streams, &mut stages, &mut folds_made)?;
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
                    &shapes,
                    &streams,
                    stages.len(),
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
        let _ = writeln!(sv, "  input  real  in_{},", ident(name));
    }
    let _ = writeln!(sv, "  output logic out_valid,");
    let _ = writeln!(sv, "  output real  out_OUTPUT,");
    let _ = writeln!(sv, "  output longint out_sweeps,");
    let _ = writeln!(sv, "  output logic out_converged");
    let _ = writeln!(sv, ");");
    sv.push_str(FUNCTIONS);
    for (name, cells) in &inputs {
        let id = ident(name);
        let _ = writeln!(sv, "  // the stream of {name}: {cells} cells, latency 1");
        let _ = writeln!(sv, "  real  s_in_{id};");
        let _ = writeln!(sv, "  logic v_in_{id};");
        let _ = writeln!(sv, "  longint n_in_{id};");
        let _ = writeln!(sv, "  always_ff @(posedge clk) begin");
        let _ = writeln!(sv, "    if (rst) begin n_in_{id} <= 0; v_in_{id} <= 1'b0; s_in_{id} <= {}; end", real_literal(0.0));
        let _ = writeln!(sv, "    else begin");
        let _ = writeln!(sv, "      v_in_{id} <= in_valid && (n_in_{id} < {cells});");
        let _ = writeln!(sv, "      s_in_{id} <= in_valid ? in_{id} : {};", real_literal(0.0));
        let _ = writeln!(sv, "      if (in_valid) n_in_{id} <= n_in_{id} + 1;");
        let _ = writeln!(sv, "    end");
        let _ = writeln!(sv, "  end");
    }
    for stage in &stages {
        if let Some(l) = loops.iter().find(|l| l.update == stage.index) {
            emit_loop_declarations(&mut sv, l);
        }
        emit_stage(&mut sv, stage, &streams, &shapes);
        if let Some(l) = loops.iter().find(|l| l.update == stage.index) {
            emit_loop_logic(&mut sv, l, stage);
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
    let _ = writeln!(h, "#include <vector>");
    let _ = writeln!(h, "int main(int argc, char** argv) {{");
    let _ = writeln!(h, "  Verilated::commandArgs(argc, argv);");
    let _ = writeln!(h, "  if (argc < 3) {{ std::fprintf(stderr, \"usage: %s inputs.bin output.bin\\n\", argv[0]); return 2; }}");
    let total_in: usize = inputs.iter().map(|(_, c)| c).sum();
    let _ = writeln!(h, "  std::vector<double> in({total_in});");
    let _ = writeln!(h, "  if ({total_in} > 0) {{ FILE* f = std::fopen(argv[1], \"rb\"); if (!f || std::fread(in.data(), sizeof(double), {total_in}, f) != {total_in}) {{ std::fprintf(stderr, \"short input\\n\"); return 3; }} std::fclose(f); }}");
    let _ = writeln!(h, "  Vrho_kernel m;");
    let _ = writeln!(h, "  auto tick = [&]() {{ m.clk = 0; m.eval(); m.clk = 1; m.eval(); }};");
    let _ = writeln!(h, "  m.rst = 1; m.in_valid = 0; tick(); m.rst = 0;");
    let _ = writeln!(h, "  std::vector<double> out; out.reserve({out_cells});");
    let _ = writeln!(h, "  long cycles = 0;");
    let feed_cells = inputs.iter().map(|(_, c)| *c).max().unwrap_or(0);
    let _ = writeln!(h, "  for (long t = 0; out.size() < {out_cells} && t < {budget}; t++) {{");
    let _ = writeln!(h, "    m.in_valid = t < {feed_cells};");
    let mut offset = 0usize;
    for (name, cells) in &inputs {
        let _ = writeln!(h, "    m.in_{} = (t < {cells}) ? in[{offset} + t] : 0.0;", ident(name));
        offset += cells;
    }
    let _ = writeln!(h, "    tick(); cycles++;");
    let _ = writeln!(h, "    if (m.out_valid) out.push_back(m.out_OUTPUT);");
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
    })
}

/// Replace every fold and scan in `expr` by a name, making a stage for each,
/// innermost first.
#[allow(clippy::too_many_arguments)]
fn extract_folds(
    expr: &Expr,
    line: usize,
    tau: f64,
    shapes: &mut BTreeMap<String, Vec<usize>>,
    streams: &mut BTreeMap<String, Stream>,
    stages: &mut Vec<Stage>,
    made: &mut usize,
) -> Result<Expr> {
    let sub = |e: &Expr, shapes: &mut BTreeMap<String, Vec<usize>>, streams: &mut BTreeMap<String, Stream>, stages: &mut Vec<Stage>, made: &mut usize| -> Result<Box<Expr>> {
        Ok(Box::new(extract_folds(e, line, tau, shapes, streams, stages, made)?))
    };
    Ok(match expr {
        Expr::Reduce { op, axis, operand } | Expr::Scan { op, axis, operand } => {
            let running = matches!(expr, Expr::Scan { .. });
            let operand = extract_folds(operand, line, tau, shapes, streams, stages, made)?;
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
                shapes,
                streams,
                stages.len(),
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
        Expr::AuditTrace(inner) => Expr::AuditTrace(sub(inner, shapes, streams, stages, made)?),
        Expr::Builtin { op, operand } => Expr::Builtin { op: *op, operand: sub(operand, shapes, streams, stages, made)? },
        Expr::BinaryOp { op, lhs, rhs } => Expr::BinaryOp {
            op: op.clone(),
            lhs: sub(lhs, shapes, streams, stages, made)?,
            rhs: sub(rhs, shapes, streams, stages, made)?,
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
    shapes: &BTreeMap<String, Vec<usize>>,
    streams: &BTreeMap<String, Stream>,
    index: usize,
) -> Result<Stage> {
    let (srcs, reach) = sources(src, shapes, line)?;
    let mut origin: Option<Origin> = None;
    for s in &srcs {
        if shapes.get(s) != Some(&shape) {
            return Err(unsupported(
                &format!("a broadcast (`{s}` is {:?}, the flow {:?})", shapes.get(s), shape),
                line,
            ));
        }
        let stream = streams.get(s).ok_or_else(|| HarmonyDisruption::SpaceErr {
            space_name: s.clone(),
            line,
        })?;
        match &origin {
            None => origin = Some(stream.origin.clone()),
            Some(o) if *o != stream.origin => {
                return Err(unsupported(
                    &format!("a flow reading two streams of different timing (`{s}` and another)"),
                    line,
                ))
            }
            _ => {}
        }
    }
    if srcs.is_empty() {
        return Err(unsupported("a flow with no space in it", line));
    }
    let base = srcs.iter().map(|s| streams[s].latency).max().unwrap_or(1);
    // A fold's stream is sparse: its cells are aligned by clocks only when
    // they left at the same clock, so two latencies are refused.
    let sparse = origin.as_ref().is_some_and(|o| o.0.starts_with("fold"));
    if sparse && srcs.iter().any(|s| streams[s].latency != base) {
        return Err(unsupported("a flow reading a fold's stream at two latencies", line));
    }
    let stage_latency = base + reach + 2;
    let chains: BTreeMap<String, (String, String, usize, usize, usize)> = srcs
        .iter()
        .map(|s| {
            let st = &streams[s];
            (
                s.clone(),
                (st.value.clone(), st.valid.clone(), st.latency, 2 * reach + 1, reach),
            )
        })
        .collect();
    let lowering = Lowering {
        shapes,
        chains: &chains,
        shape: &shape,
        stage: index,
        tau,
        line,
    };
    let expr = lowering.lower(src)?;
    let _ = lowering.shape;
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
    })
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
        Expr::BinaryOp { op, lhs, rhs } => Expr::BinaryOp { op: op.clone(), lhs: sub(lhs), rhs: sub(rhs) },
        other => other.clone(),
    }
}

/// A loop's memories and the signals its update stage reads, declared ahead
/// of the stage.
fn emit_loop_declarations(sv: &mut String, l: &Loop) {
    let k = l.id;
    let n = l.cells;
    let _ = writeln!(sv);
    let _ = writeln!(sv, "  // ---- loop {k}: ... ⇒ {} ({n} cells, at most {} sweeps)", l.target, l.cap);
    for (name, _) in &l.sources {
        let sid = ident(name);
        if *name == l.target {
            let _ = writeln!(sv, "  real m{k}_{sid}_a [0:{}];  // {name}: the round's grid", n - 1);
            let _ = writeln!(sv, "  real m{k}_{sid}_b [0:{}];  // {name}: the grid being written", n - 1);
        } else {
            let _ = writeln!(sv, "  real m{k}_{sid} [0:{}];  // {name}, captured", n - 1);
        }
        let _ = writeln!(sv, "  longint cap{k}_{sid};  // cells captured");
        let _ = writeln!(sv, "  real  rr{k}_{sid};  // the round stream of {name}");
    }
    let _ = writeln!(sv, "  logic rv{k};  // the round streams' valid");
    let _ = writeln!(sv, "  logic rs{k};  // a round starts: the update stage begins afresh");
    let _ = writeln!(sv, "  logic cur{k};  // which buffer of {} the round reads", l.target);
    let _ = writeln!(sv, "  longint i{k}, w{k}, round{k}, sw{k};");
    let _ = writeln!(sv, "  logic cv{k};");
    let _ = writeln!(sv, "  real  d{k};  // the largest move this round");
    let _ = writeln!(sv, "  int   ls{k};  // 0 capture, 1 read, 2 drain, 3 emit, 4 done");
    let _ = writeln!(sv, "  real  px{k};  // {} after the loop, streamed out", l.target);
    let _ = writeln!(sv, "  logic pv{k};");
}

/// The loop's controller: capture, rounds, the decision, the read-out.
fn emit_loop_logic(sv: &mut String, l: &Loop, update: &Stage) {
    let k = l.id;
    let n = l.cells;
    let tid = ident(&l.target);
    let upd_value = format!("st{}_{}", update.index, ident(&update.target));
    let upd_valid = format!("v{}_out", update.index);
    let _ = writeln!(sv, "  always_ff @(posedge clk) begin : loop{k}");
    let _ = writeln!(sv, "    real old, diff;");
    let _ = writeln!(sv, "    logic all_captured, settled, capped;");
    let _ = writeln!(sv, "    if (rst) begin");
    for (name, _) in &l.sources {
        let _ = writeln!(sv, "      cap{k}_{} <= 0;", ident(name));
    }
    let _ = writeln!(
        sv,
        "      rv{k} <= 1'b0; rs{k} <= 1'b0; cur{k} <= 1'b0; i{k} <= 0; w{k} <= 0; round{k} <= 0; sw{k} <= 0; cv{k} <= 1'b1; d{k} <= {}; ls{k} <= 0; pv{k} <= 1'b0;",
        real_literal(0.0)
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
    let _ = writeln!(sv, "        0: if (all_captured) begin ls{k} <= 1; i{k} <= 0; w{k} <= 0; d{k} <= {}; rs{k} <= 1'b1; end", real_literal(0.0));
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
    let _ = writeln!(sv, "          settled = (d{k} <= {});", real_literal(l.tau));
    let _ = writeln!(sv, "          capped = (round{k} + 1 >= {});", l.cap);
    let _ = writeln!(sv, "          sw{k} <= sw{k} + 1;");
    let _ = writeln!(sv, "          if (settled || capped) begin");
    let _ = writeln!(sv, "            if (capped) cv{k} <= 1'b0;");
    let _ = writeln!(sv, "            cur{k} <= ~cur{k}; ls{k} <= 3; i{k} <= 0;");
    let _ = writeln!(sv, "          end else begin");
    let _ = writeln!(sv, "            cur{k} <= ~cur{k}; round{k} <= round{k} + 1; i{k} <= 0; w{k} <= 0; d{k} <= {}; rs{k} <= 1'b1; ls{k} <= 1;", real_literal(0.0));
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
    let _ = writeln!(sv, "        diff = rho_abs({upd_value} - old);");
    let _ = writeln!(sv, "        if (diff > d{k}) d{k} <= diff;");
    let _ = writeln!(sv, "        w{k} <= w{k} + 1;");
    let _ = writeln!(sv, "      end");
    let _ = writeln!(sv, "    end");
    let _ = writeln!(sv, "  end");
}

/// The SystemVerilog of one stage: alignment, lines, counters, flush, the
/// computation, and for a fold the accumulators.
fn emit_stage(sv: &mut String, stage: &Stage, streams: &BTreeMap<String, Stream>, _shapes: &BTreeMap<String, Vec<usize>>) {
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
    let first_src = stage.chains.keys().next().cloned().unwrap();
    for (src, (value, valid, latency, _, _)) in &stage.chains {
        let sid = ident(src);
        let delay = base - latency;
        if delay == 0 {
            let _ = writeln!(sv, "  real  a{k}_{sid};  always_comb a{k}_{sid} = {value};");
            let _ = writeln!(sv, "  logic av{k}_{sid}; assign av{k}_{sid} = {valid};");
        } else {
            let _ = writeln!(sv, "  real  ad{k}_{sid} [0:{}];", delay - 1);
            let _ = writeln!(sv, "  logic adv{k}_{sid} [0:{}];", delay - 1);
            let _ = writeln!(sv, "  always_ff @(posedge clk) begin");
            let _ = writeln!(sv, "    ad{k}_{sid}[0] <= {value}; adv{k}_{sid}[0] <= {clear} ? 1'b0 : {valid};");
            let _ = writeln!(sv, "    for (int i = 1; i < {delay}; i++) begin ad{k}_{sid}[i] <= ad{k}_{sid}[i - 1]; adv{k}_{sid}[i] <= {clear} ? 1'b0 : adv{k}_{sid}[i - 1]; end");
            let _ = writeln!(sv, "  end");
            let _ = writeln!(sv, "  real  a{k}_{sid};  always_comb a{k}_{sid} = ad{k}_{sid}[{}];", delay - 1);
            let _ = writeln!(sv, "  logic av{k}_{sid}; assign av{k}_{sid} = adv{k}_{sid}[{}];", delay - 1);
        }
    }
    let src_cells = streams[&first_src].cells;
    let fsid = ident(&first_src);
    let _ = writeln!(sv, "  // cells entered, real cells entered, zeros pushed through after the last");
    let _ = writeln!(sv, "  longint n{k}_in, n{k}_real, n{k}_flushed;");
    let _ = writeln!(sv, "  logic v{k}_src, fl{k}, en{k};");
    let _ = writeln!(sv, "  assign v{k}_src = av{k}_{fsid};");
    let _ = writeln!(
        sv,
        "  assign fl{k} = (n{k}_real == {src_cells}) && (n{k}_flushed < {}) && !v{k}_src;",
        r + 1
    );
    let _ = writeln!(sv, "  assign en{k} = v{k}_src || fl{k};");
    for (src, (_, _, _, length, centre)) in &stage.chains {
        let sid = ident(src);
        let l = format!("c{k}_{sid}");
        let _ = writeln!(sv, "  real {l} [0:{}];  // {src}; centre tap {centre}", length - 1);
        let _ = writeln!(sv, "  always_ff @(posedge clk) if (en{k}) begin");
        let _ = writeln!(sv, "    {l}[0] <= v{k}_src ? a{k}_{sid} : {};", real_literal(0.0));
        let _ = writeln!(sv, "    for (int i = 1; i < {length}; i++) {l}[i] <= {l}[i - 1];");
        let _ = writeln!(sv, "  end");
    }
    let rank = stage.shape.len();
    let _ = writeln!(sv, "  real  st{k}_{tid};");
    let _ = writeln!(sv, "  logic v{k}_out;");
    if let Kind::Fold { axis, .. } = stage.kind {
        let (stride, _) = axis_geometry(&stage.shape, Some(axis)).unwrap_or((1, 1));
        let _ = writeln!(sv, "  real acc{k} [0:{}];  // one accumulator per line across axis {axis}", stride - 1);
    }
    let _ = writeln!(sv, "  always_ff @(posedge clk) begin : stage{k}");
    let _ = writeln!(sv, "    longint here;");
    let _ = writeln!(sv, "    longint pos [0:{}];", rank.max(1) - 1);
    let _ = writeln!(sv, "    real x;");
    let _ = writeln!(sv, "    if ({clear}) begin");
    let _ = writeln!(sv, "      n{k}_in <= 0; n{k}_real <= 0; n{k}_flushed <= 0; v{k}_out <= 1'b0; st{k}_{tid} <= {};", real_literal(0.0));
    let _ = writeln!(sv, "    end else begin");
    let _ = writeln!(sv, "      v{k}_out <= 1'b0;");
    let _ = writeln!(sv, "      if (v{k}_src) n{k}_real <= n{k}_real + 1;");
    let _ = writeln!(sv, "      if (fl{k}) n{k}_flushed <= n{k}_flushed + 1;");
    let _ = writeln!(sv, "      if (en{k}) begin");
    let _ = writeln!(sv, "        n{k}_in <= n{k}_in + 1;");
    let _ = writeln!(sv, "        here = n{k}_in - {};", r + 1);
    for a in 0..rank {
        let (stride, extent) = axis_geometry(&stage.shape, Some(a)).unwrap_or((1, 1));
        let _ = writeln!(sv, "        pos[{a}] = (here < 0) ? 0 : ((here / {stride}) % {extent});");
    }
    let _ = writeln!(sv, "        if (here >= 0 && here < {cells}) begin");
    let _ = writeln!(sv, "          x = {};", stage.expr);
    match stage.kind {
        Kind::Flow => {
            let _ = writeln!(sv, "          st{k}_{tid} <= x;");
            let _ = writeln!(sv, "          v{k}_out <= 1'b1;");
        }
        Kind::Fold { op, axis, running } => {
            let (stride, extent) = axis_geometry(&stage.shape, Some(axis)).unwrap_or((1, 1));
            let step = match op {
                FoldOp::Sum => "(a + x)",
                FoldOp::Product => "(a * x)",
                FoldOp::Max => "((x > a) ? x : a)",
                FoldOp::Min => "((x < a) ? x : a)",
            };
            let _ = writeln!(sv, "          begin : accumulate");
            let _ = writeln!(sv, "            real a, next;");
            let _ = writeln!(sv, "            longint t, m;");
            let _ = writeln!(sv, "            t = here % {stride};");
            let _ = writeln!(sv, "            m = (here / {stride}) % {extent};");
            let _ = writeln!(sv, "            a = (m == 0) ? {} : acc{k}[t];", real_literal(op.identity()));
            let _ = writeln!(sv, "            next = {step};");
            let _ = writeln!(sv, "            acc{k}[t] <= next;");
            if running {
                let _ = writeln!(sv, "            st{k}_{tid} <= next;");
                let _ = writeln!(sv, "            v{k}_out <= 1'b1;");
            } else {
                let _ = writeln!(sv, "            if (m == {}) begin st{k}_{tid} <= next; v{k}_out <= 1'b1; end", extent - 1);
            }
            let _ = writeln!(sv, "          end");
        }
    }
    let _ = writeln!(sv, "        end");
    let _ = writeln!(sv, "      end");
    let _ = writeln!(sv, "    end");
    let _ = writeln!(sv, "  end");
}

/// The helpers every module carries: the language's operations that are not
/// one SystemVerilog operator, written to give the interpreter's bits.
const FUNCTIONS: &str = r#"
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
