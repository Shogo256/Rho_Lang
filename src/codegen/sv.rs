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
//! The cells are SystemVerilog `real`: IEEE double in simulation, with the
//! same libm the interpreter and the kernel call, so Verilator's run is held
//! to the interpreter bit for bit. `real` does not synthesise; this is the
//! structure and the timing of the circuit, one cell per cycle, with the
//! arithmetic units left to a later step (fixed point, or floating-point
//! cores). What is in the subset so far: `→`, arithmetic, comparisons as
//! masks, the greater and the lesser, the residue, the named functions,
//! `?`, `⌊` `⌈`, `⍳`, shifts along any axis, chains of flows. Not yet:
//! folds, scans, lifts, the turns, `⌷`, `⇒`.

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

struct Stage {
    /// Position in the program: signal names carry it, since a space may be
    /// written by more than one flow (`T → OUTPUT` and then `OUTPUT → =`).
    index: usize,
    /// Target space, its shape, and the latency its stream has.
    target: String,
    shape: Vec<usize>,
    latency: usize,
    /// Source space -> (the signal its stream is on, chain length, centre tap index).
    chains: BTreeMap<String, (String, usize, usize)>,
    /// The expression, as SystemVerilog over the chains.
    expr: String,
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
            Expr::Reduce { .. } | Expr::Scan { .. } => return Err(unsupported("a fold or scan", line)),
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
    chains: &'a BTreeMap<String, (String, usize, usize)>,
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
                let (_, _, centre) = &self.chains[name];
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
                let (_, _, centre) = &self.chains[name];
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
pub fn emit(block: &ToposBlock, tau: f64) -> Result<Circuit> {
    let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut inputs: Vec<String> = Vec::new();
    for stmt in &block.statements {
        if let Statement::SpaceDef(decl) = stmt {
            shapes.insert(decl.name.clone(), decl.dimensions.clone());
            inputs.push(decl.name.clone());
        }
    }
    // The stream each space is on right now: its signal and its latency.
    let mut streams: BTreeMap<String, (String, usize)> = inputs
        .iter()
        .map(|n| (n.clone(), (format!("s_in_{}", ident(n)), 1)))
        .collect();
    let mut stages: Vec<Stage> = Vec::new();
    let mut written: BTreeSet<String> = BTreeSet::new();

    for (index, stmt) in block.statements.iter().enumerate() {
        let line = block.line_of(index);
        match stmt {
            Statement::SpaceDef(_) | Statement::Constraint(_) | Statement::AuditTrace(_) => {}
            Statement::ExtBind(_) => return Err(unsupported("a bound address", line)),
            Statement::Iterate { .. } => return Err(unsupported("a fixed point (`⇒`)", line)),
            Statement::Flow { src, target } => {
                let target = match target {
                    FlowTarget::Var(n) => n.clone(),
                    FlowTarget::Equilibrium => "OUTPUT".to_string(),
                };
                let (srcs, reach) = sources(src, &shapes, line)?;
                let shape = expr_shape(src, &shapes)
                    .or_else(|| shapes.get(&target).cloned())
                    .ok_or_else(|| unsupported("a flow with no shape", line))?;
                for s in &srcs {
                    if shapes.get(s) != Some(&shape) {
                        return Err(unsupported(
                            &format!("a broadcast (`{s}` is {:?}, the flow {:?})", shapes.get(s), shape),
                            line,
                        ));
                    }
                    if !streams.contains_key(s) {
                        return Err(HarmonyDisruption::SpaceErr {
                            space_name: s.clone(),
                            line,
                        });
                    }
                }
                let stage_index = stages.len();
                let base = srcs.iter().map(|s| streams[s].1).max().unwrap_or(1);
                let stage_latency = base + reach + 2;
                let chains: BTreeMap<String, (String, usize, usize)> = srcs
                    .iter()
                    .map(|s| {
                        let (signal, d) = &streams[s];
                        let centre = stage_latency - d - 2;
                        (s.clone(), (signal.clone(), centre + reach + 1, centre))
                    })
                    .collect();
                let lowering = Lowering {
                    shapes: &shapes,
                    chains: &chains,
                    shape: &shape,
                    stage: stage_index,
                    tau,
                    line,
                };
                let expr = lowering.lower(src)?;
                let _ = lowering.shape;
                shapes.insert(target.clone(), shape.clone());
                streams.insert(target.clone(), (format!("st{stage_index}_{}", ident(&target)), stage_latency));
                written.insert(target.clone());
                stages.push(Stage {
                    index: stage_index,
                    target: target.clone(),
                    shape,
                    latency: stage_latency,
                    chains,
                    expr,
                });
                if matches!(stmt, Statement::Flow { target: FlowTarget::Equilibrium, .. }) {
                    break;
                }
            }
        }
    }
    let (out_signal, out_latency) = streams
        .get("OUTPUT")
        .cloned()
        .ok_or_else(|| unsupported("a program with no `→ =`", 0))?;
    let out_shape = shapes["OUTPUT"].clone();
    let out_cells = out_shape.iter().product::<usize>().max(1);
    let inputs: Vec<(String, usize)> = inputs
        .into_iter()
        .filter(|n| !written.contains(n))
        .map(|n| {
            let cells = shapes[&n].iter().product::<usize>().max(1);
            (n, cells)
        })
        .collect();

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
    let _ = writeln!(sv, "  output real  out_OUTPUT");
    let _ = writeln!(sv, ");");
    sv.push_str(FUNCTIONS);
    let _ = writeln!(sv, "  // Clocks since the first cell; cell k of a stream with latency d is on it at clock k + d.");
    let _ = writeln!(sv, "  longint cyc;");
    let _ = writeln!(sv, "  always_ff @(posedge clk) begin");
    let _ = writeln!(sv, "    if (rst) cyc <= 0; else cyc <= cyc + 1;");
    let _ = writeln!(sv, "  end");
    for (name, _) in &inputs {
        let id = ident(name);
        let _ = writeln!(sv, "  real s_in_{id};  // the stream of {name}, latency 1");
        let _ = writeln!(sv, "  always_ff @(posedge clk) s_in_{id} <= in_valid ? in_{id} : {};", real_literal(0.0));
    }
    for stage in &stages {
        let k = stage.index;
        let tid = ident(&stage.target);
        let _ = writeln!(sv);
        let _ = writeln!(
            sv,
            "  // ---- stage {k}: -> {} (shape {:?}), latency {}",
            stage.target, stage.shape, stage.latency
        );
        for (src, (signal, length, centre)) in &stage.chains {
            let line = format!("c{k}_{}", ident(src));
            let _ = writeln!(
                sv,
                "  real {line} [0:{}];  // {src} delayed for this stage; centre tap {centre}",
                length - 1
            );
            let _ = writeln!(sv, "  always_ff @(posedge clk) begin");
            let _ = writeln!(sv, "    {line}[0] <= {signal};");
            let _ = writeln!(sv, "    for (int i = 1; i < {length}; i++) {line}[i] <= {line}[i - 1];");
            let _ = writeln!(sv, "  end");
        }
        // Coordinates of the centre cell this stage computes at this clock.
        let rank = stage.shape.len();
        let _ = writeln!(sv, "  real st{k}_{tid};");
        let _ = writeln!(sv, "  always_ff @(posedge clk) begin : stage{k}_{tid}");
        let _ = writeln!(sv, "    longint here;  // the cell this clock computes");
        let _ = writeln!(sv, "    longint pos [0:{}];", rank.max(1) - 1);
        let _ = writeln!(sv, "    here = cyc - {};", stage.latency - 1);
        for a in 0..rank {
            let (stride, extent) = axis_geometry(&stage.shape, Some(a)).unwrap_or((1, 1));
            let _ = writeln!(sv, "    pos[{a}] = (here < 0) ? 0 : ((here / {stride}) % {extent});");
        }
        let _ = writeln!(sv, "    st{k}_{tid} <= {};", stage.expr);
        let _ = writeln!(sv, "  end");
    }
    let _ = writeln!(sv);
    let _ = writeln!(sv, "  assign out_OUTPUT = {out_signal};");
    let _ = writeln!(
        sv,
        "  assign out_valid = (cyc >= {out_latency}) && (cyc < {});",
        out_latency + out_cells
    );
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
    let feed_cells = inputs.first().map(|(_, c)| *c).unwrap_or(0);
    let _ = writeln!(h, "  for (long t = 0; out.size() < {out_cells} && t < {}; t++) {{", feed_cells + out_latency + out_cells + 4);
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
    let _ = writeln!(h, "  std::printf(\"cells %zu cycles %ld latency {out_latency}\\n\", out.size(), cycles);");
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
pub fn simulate(circuit: &Circuit, dir: &Path, inputs: &[Vec<f64>]) -> std::io::Result<(Vec<f64>, u64)> {
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
    let cycles = stdout
        .split_whitespace()
        .skip_while(|w| *w != "cycles")
        .nth(1)
        .and_then(|w| w.parse().ok())
        .unwrap_or(0);
    let raw = std::fs::read(&out_path)?;
    let out = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((out, cycles))
}
