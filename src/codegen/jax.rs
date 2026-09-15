// SPDX-License-Identifier: Apache-2.0
//! A program as a JAX function: the same static-shape, no-branch dataflow
//! that XLA compiles for CPU, GPU and TPU, written out as `jax.numpy`.
//!
//! Every construct has a direct counterpart — a shift is a pad and a slice,
//! a fold a reduction, a lift an expanded axis, a rotation a roll, `⌷` a
//! clipped take under a mask, `⇒` a `lax.while_loop` — so the emitted module
//! is short and reads like the program. What it does not promise is the
//! interpreter's bits: XLA orders a reduction as it likes and its
//! transcendental functions are its own, so a fold or an `exp` may differ
//! in the last places. Programs of arithmetic, shifts, masks and turns are
//! bit-identical on CPU; the tests hold the rest to a few ulps.

use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use crate::numeric::Precision;
use std::collections::BTreeMap;
use std::fmt::Write as _;

fn unsupported(what: &str, line: usize) -> HarmonyDisruption {
    HarmonyDisruption::LoweringErr {
        detail: format!("{what} is not in the JAX subset (--emit-jax)"),
        line,
    }
}

fn py_name(name: &str) -> String {
    // A ρ name may carry `·` from a function's expansion or `⟳`; Python
    // identifiers cannot.
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push_str(&format!("_u{:04x}_", c as u32));
        }
    }
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

fn py_float(v: f64) -> String {
    if v.is_nan() {
        "float('nan')".to_string()
    } else if v == f64::INFINITY {
        "float('inf')".to_string()
    } else if v == f64::NEG_INFINITY {
        "float('-inf')".to_string()
    } else {
        // `repr`-style: shortest round-trip, so the literal is the value.
        let s = format!("{v:?}");
        if s.contains('.') || s.contains('e') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

fn is_tau(name: &str) -> bool {
    name == "𝜏" || name == "τ"
}

/// A shape as a Python tuple, which `jax.jit` can hash where a list cannot.
fn py_tuple(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("({},)", dims.join(", "))
}

type Shapes = BTreeMap<String, Vec<usize>>;

struct Emitter {
    tau: f64,
}

impl Emitter {
    fn shape_of(&self, expr: &Expr, line: usize, shapes: &Shapes) -> Result<Vec<usize>> {
        expr_shape(expr, shapes).ok_or_else(|| unsupported("an expression with no shape", line))
    }

    fn lower(&self, expr: &Expr, line: usize, shapes: &Shapes) -> Result<String> {
        Ok(match expr {
            Expr::Number(v) => format!("_c({})", py_float(*v)),
            Expr::Var(name) if is_tau(name) => format!("_c({})", py_float(self.tau)),
            Expr::Var(name) => py_name(name),
            Expr::AuditTrace(inner) => self.lower(inner, line, shapes)?,
            Expr::BinaryOp { op, lhs, rhs } => {
                let l = self.lower(lhs, line, shapes)?;
                let r = self.lower(rhs, line, shapes)?;
                match op {
                    BinaryOpKind::Add => format!("({l} + {r})"),
                    BinaryOpKind::Sub => format!("({l} - {r})"),
                    BinaryOpKind::Mul => format!("({l} * {r})"),
                    BinaryOpKind::Div => format!("({l} / {r})"),
                    BinaryOpKind::Pow => match whole_exponent(rhs) {
                        Some(n) => format!("_ipow({l}, {n})"),
                        None => format!("jnp.power({l}, {r})"),
                    },
                    BinaryOpKind::Gt => format!("_mask({l} > {r}, {l})"),
                    BinaryOpKind::Lt => format!("_mask({l} < {r}, {l})"),
                    BinaryOpKind::Gte => format!("_mask({l} >= {r}, {l})"),
                    BinaryOpKind::Lte => format!("_mask({l} <= {r}, {l})"),
                    BinaryOpKind::Eq => format!("_mask({l} == {r}, {l})"),
                    BinaryOpKind::Max => format!("jnp.maximum({l}, {r})"),
                    BinaryOpKind::Min => format!("jnp.minimum({l}, {r})"),
                    BinaryOpKind::Residue => format!("_residue({l}, {r})"),
                }
            }
            Expr::Builtin { op, operand } => {
                let x = self.lower(operand, line, shapes)?;
                match op {
                    BuiltinOp::Exp => format!("jnp.exp({x})"),
                    BuiltinOp::Log => format!("jnp.log({x})"),
                    BuiltinOp::Sqrt => format!("jnp.sqrt({x})"),
                    BuiltinOp::Sin => format!("jnp.sin({x})"),
                    BuiltinOp::Cos => format!("jnp.cos({x})"),
                    BuiltinOp::Abs => format!("jnp.abs({x})"),
                    BuiltinOp::Indicator => format!("_ind({x})"),
                    BuiltinOp::Roll => format!("_roll({x})"),
                    BuiltinOp::Floor => format!("jnp.floor({x})"),
                    BuiltinOp::Ceil => format!("jnp.ceil({x})"),
                }
            }
            Expr::Shift { dir, axis, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let a = axis.unwrap_or_else(|| default_axis(&shape));
                let x = self.lower(operand, line, shapes)?;
                match dir {
                    ShiftDir::Positive => format!("_prev({x}, {a})"),
                    ShiftDir::Negative => format!("_next({x}, {a})"),
                }
            }
            Expr::Lift { axis, operand } => format!("jnp.expand_dims({}, {axis})", self.lower(operand, line, shapes)?),
            Expr::Reduce { op, axis, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let a = axis.unwrap_or_else(|| default_axis(&shape));
                let x = self.lower(operand, line, shapes)?;
                let f = match op {
                    FoldOp::Sum => "jnp.sum",
                    FoldOp::Product => "jnp.prod",
                    FoldOp::Max => "jnp.max",
                    FoldOp::Min => "jnp.min",
                };
                format!("{f}({x}, axis={a})")
            }
            Expr::Scan { op, axis, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let a = axis.unwrap_or_else(|| default_axis(&shape));
                let x = self.lower(operand, line, shapes)?;
                let f = match op {
                    FoldOp::Sum => "jnp.cumsum",
                    FoldOp::Product => "jnp.cumprod",
                    FoldOp::Max => "lax.cummax",
                    FoldOp::Min => "lax.cummin",
                };
                format!("{f}({x}, axis={a})")
            }
            Expr::Index { axis, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let a = axis.unwrap_or_else(|| default_axis(&shape));
                format!("_coord({}, {a})", py_tuple(&shape))
            }
            Expr::Rotate { by, axis, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let a = axis.unwrap_or_else(|| default_axis(&shape));
                format!("jnp.roll({}, {}, axis={a})", self.lower(operand, line, shapes)?, -by)
            }
            Expr::Reverse { axis, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let a = axis.unwrap_or_else(|| default_axis(&shape));
                format!("jnp.flip({}, axis={a})", self.lower(operand, line, shapes)?)
            }
            Expr::Reshape { shape, operand } => format!("jnp.resize({}, {})", self.lower(operand, line, shapes)?, py_tuple(shape)),
            Expr::Transpose { axes, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let perm = transpose_axes(shape.len(), axes.as_deref())
                    .ok_or_else(|| unsupported("a transpose that is not a permutation of the axes", line))?;
                // numpy's `axes[i]` is the source axis of result axis i; ρ's
                // sends source axis k to result axis perm[k].
                let mut inverse = vec![0usize; perm.len()];
                for (k, &p) in perm.iter().enumerate() {
                    inverse[p] = k;
                }
                format!("jnp.transpose({}, axes={})", self.lower(operand, line, shapes)?, py_tuple(&inverse))
            }
            Expr::Take { count, axis, operand } | Expr::Drop { count, axis, operand } => {
                let shape = self.shape_of(operand, line, shapes)?;
                let a = axis.unwrap_or_else(|| default_axis(&shape));
                let f = if matches!(expr, Expr::Drop { .. }) { "_drop" } else { "_take" };
                format!("{f}({}, {a}, {count})", self.lower(operand, line, shapes)?)
            }
            Expr::Gather { index, operand } => {
                format!("_gather({}, {})", self.lower(index, line, shapes)?, self.lower(operand, line, shapes)?)
            }
            Expr::Call { name, .. } => return Err(unsupported(&format!("the call to `{name}`, which was not expanded"), line)),
        })
    }
}

/// The Python module: `rho(**spaces)` returning every written space, the
/// sweeps and whether every `⇒` settled, and `rho_jit`, the same under
/// `jax.jit`.
pub fn emit(block: &ToposBlock, tau: f64, cap: Option<usize>, precision: Precision) -> Result<String> {
    let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut bound: Vec<(String, u64)> = Vec::new();
    for stmt in &block.statements {
        match stmt {
            Statement::SpaceDef(decl) => {
                shapes.insert(decl.name.clone(), decl.dimensions.clone());
                inputs.push(decl.name.clone());
            }
            // The address is the kernel's business; here the space is an
            // argument like any other.
            Statement::ExtBind(binding) => {
                shapes.insert(binding.space.name.clone(), binding.space.dimensions.clone());
                inputs.push(binding.space.name.clone());
                bound.push((binding.space.name.clone(), binding.address));
            }
            _ => {}
        }
    }
    let dtype = match precision {
        Precision::F64 => "jnp.float64",
        Precision::F32 => "jnp.float32",
    };
    let mut py = String::new();
    let _ = writeln!(py, "# ρ program as JAX. Generated by rhoc --emit-jax; the source is at the end.");
    for (name, address) in &bound {
        let _ = writeln!(py, "# {name} is bound to address {address:#x} for the compiled kernel; here it is an argument.");
    }
    let _ = writeln!(py, "import jax");
    let _ = writeln!(py, "import jax.numpy as jnp");
    let _ = writeln!(py, "from jax import lax");
    let _ = writeln!(py, "jax.config.update('jax_enable_x64', True)");
    let _ = writeln!(py);
    let _ = writeln!(py, "DTYPE = {dtype}");
    let _ = writeln!(py, "SHAPES = {{");
    for (name, shape) in &shapes {
        let _ = writeln!(py, "    {:?}: {},", name, py_tuple(shape));
    }
    let _ = writeln!(py, "}}");
    py.push_str(HELPERS);
    let _ = writeln!(py);
    let params: Vec<String> = inputs.iter().map(|n| py_name(n)).collect();
    let _ = writeln!(py, "def rho({}):", params.join(", "));
    let _ = writeln!(py, "    \"\"\"The program over its declared spaces, as jax arrays of DTYPE.\"\"\"");
    for name in &inputs {
        let p = py_name(name);
        let _ = writeln!(py, "    {p} = jnp.asarray({p}, dtype=DTYPE).reshape({})", py_tuple(&shapes[name]));
    }
    let _ = writeln!(py, "    sweeps = 0");
    let _ = writeln!(py, "    converged = True");
    let emitter = Emitter { tau };
    let mut written: Vec<String> = Vec::new();
    let mut loops = 0usize;
    for (index, stmt) in block.statements.iter().enumerate() {
        let line = block.line_of(index);
        match stmt {
            Statement::SpaceDef(_) | Statement::Constraint(_) | Statement::AuditTrace(_) | Statement::ExtBind(_) => {}
            Statement::Flow { src, target } => {
                let name = match target {
                    FlowTarget::Var(n) => n.clone(),
                    FlowTarget::Equilibrium => "OUTPUT".to_string(),
                };
                let value = emitter.lower(src, line, &shapes)?;
                let _ = writeln!(py, "    {} = {value}", py_name(&name));
                if let Some(shape) = expr_shape(src, &shapes) {
                    shapes.insert(name.clone(), shape);
                }
                if !written.contains(&name) {
                    written.push(name.clone());
                }
                if matches!(target, FlowTarget::Equilibrium) {
                    break;
                }
            }
            Statement::Iterate { prelude, src, target } => {
                let cap = cap.ok_or_else(|| HarmonyDisruption::LoweringErr {
                    detail: "this program iterates (⇒); say how many sweeps it may take with --max-iter".to_string(),
                    line,
                })?;
                loops += 1;
                let t = py_name(target);
                // The round: the prelude's flows, then the update. Its
                // spaces are the target and every space the prelude writes,
                // carried through the loop's state.
                let mut carried: Vec<String> = vec![target.clone()];
                for flow in prelude {
                    if let Statement::Flow { target: FlowTarget::Var(n), .. } = flow {
                        if !carried.contains(n) {
                            carried.push(n.clone());
                        }
                    }
                }
                let _ = writeln!(py, "    def _round{loops}(state):");
                let names: Vec<String> = carried.iter().map(|n| py_name(n)).collect();
                let _ = writeln!(py, "        {}, _k, _settled = state", names.join(", "));
                let _ = writeln!(py, "        _old = {t}");
                for flow in prelude {
                    if let Statement::Flow { src, target: FlowTarget::Var(n) } = flow {
                        let value = emitter.lower(src, line, &shapes)?;
                        let _ = writeln!(py, "        {} = {value}", py_name(n));
                        if let Some(shape) = expr_shape(src, &shapes) {
                            shapes.insert(n.clone(), shape);
                        }
                    }
                }
                let value = emitter.lower(src, line, &shapes)?;
                let _ = writeln!(py, "        {t} = {value}");
                let _ = writeln!(py, "        _moved = jnp.max(jnp.abs({t} - _old))");
                let _ = writeln!(py, "        _settled = _moved <= _c({})", py_float(tau));
                let _ = writeln!(py, "        return ({}, _k + 1, _settled)", names.join(", "));
                let _ = writeln!(py, "    def _go{loops}(state):");
                let _ = writeln!(py, "        _k, _settled = state[-2], state[-1]");
                let _ = writeln!(py, "        return jnp.logical_and(_k < {cap}, jnp.logical_not(_settled))");
                // The prelude's spaces start as zeros of their shape; the
                // first round writes them before anything reads them.
                let mut init: Vec<String> = vec![t.clone()];
                for n in carried.iter().skip(1) {
                    let shape = shapes.get(n).cloned().unwrap_or_else(|| shapes[target].clone());
                    init.push(format!("jnp.zeros({}, dtype=DTYPE)", py_tuple(&shape)));
                }
                let _ = writeln!(py, "    _state{loops} = ({}, jnp.int64(0), jnp.bool_(False))", init.join(", "));
                let _ = writeln!(py, "    _state{loops} = lax.while_loop(_go{loops}, _round{loops}, _state{loops})");
                let _ = writeln!(py, "    {} = _state{loops}[:{}]", names.join(", ") + if names.len() == 1 { "," } else { "" }, names.len());
                let _ = writeln!(py, "    sweeps = sweeps + _state{loops}[-2]");
                // Stopping at the cap counts as not converged even when the
                // last sweep happened to settle, as the kernel has it.
                let _ = writeln!(py, "    converged = jnp.logical_and(converged, _state{loops}[-2] < {cap})");
                for n in &carried {
                    if !written.contains(n) {
                        written.push(n.clone());
                    }
                }
            }
        }
    }
    let _ = writeln!(py, "    return {{");
    for name in &written {
        let _ = writeln!(py, "        {:?}: {},", name, py_name(name));
    }
    let _ = writeln!(py, "        'rho_sweeps': sweeps,");
    let _ = writeln!(py, "        'rho_converged': converged,");
    let _ = writeln!(py, "    }}");
    let _ = writeln!(py);
    let _ = writeln!(py, "rho_jit = jax.jit(rho)");
    let _ = writeln!(py);
    let _ = writeln!(py, "INPUTS = {:?}", inputs);
    let _ = writeln!(py, "OUTPUT = 'OUTPUT'");
    Ok(py)
}

const HELPERS: &str = r#"

def _c(v):
    return jnp.asarray(v, dtype=DTYPE)


def _mask(holds, l):
    """A comparison masks: the left value where it holds, zero elsewhere."""
    return jnp.where(holds, l, _c(0.0))


def _ind(x):
    """1 where the value is not zero (a NaN is not zero), 0 at zero."""
    return jnp.where(x != 0, _c(1.0), _c(0.0))


def _ipow(x, n):
    """A whole power by repeated multiplication, as the kernel does it."""
    acc = _c(1.0)
    for _ in range(abs(n)):
        acc = acc * x
    return _c(1.0) / acc if n < 0 else acc


def _residue(a, b):
    """APL's residue: b - a * floor(b / a); 0 | b is b."""
    return jnp.where(a == 0, b, b - a * jnp.floor(b / a))


def _roll(x):
    """The roll: splitmix64's finaliser over the value's bits (as a double),
    the top 53 bits scaled into [0, 1). A NaN hashes as zero bits."""
    d = jnp.asarray(x, dtype=jnp.float64)
    bits = lax.bitcast_convert_type(jnp.where(jnp.isnan(d), 0.0, d), jnp.uint64)
    z = bits + jnp.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * jnp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * jnp.uint64(0x94D049BB133111EB)
    z = z ^ (z >> 31)
    return jnp.asarray((z >> 11).astype(jnp.float64) * (1.0 / 9007199254740992.0), dtype=DTYPE)


def _prev(x, axis):
    """▷: the preceding cell along the axis, zero at the edge."""
    pad = [(0, 0)] * x.ndim
    pad[axis] = (1, 0)
    padded = jnp.pad(x, pad)
    return lax.slice_in_dim(padded, 0, x.shape[axis], axis=axis)


def _next(x, axis):
    """▽: the following cell along the axis, zero at the edge."""
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, 1)
    padded = jnp.pad(x, pad)
    return lax.slice_in_dim(padded, 1, x.shape[axis] + 1, axis=axis)


def _coord(shape, axis):
    """⍳: the coordinate of each cell along the axis, from zero."""
    n = shape[axis]
    view = [1] * len(shape)
    view[axis] = n
    return jnp.broadcast_to(jnp.arange(n, dtype=DTYPE).reshape(view), shape)


def _take(x, axis, count):
    """k ↑ X: the first k cells along the axis (the last for a negative k),
    padded with zero past the source."""
    n = x.shape[axis]
    k = abs(count)
    if count >= 0:
        piece = lax.slice_in_dim(x, 0, min(k, n), axis=axis)
        pad_after = max(k - n, 0)
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, pad_after)
    else:
        piece = lax.slice_in_dim(x, max(n - k, 0), n, axis=axis)
        pad = [(0, 0)] * x.ndim
        pad[axis] = (max(k - n, 0), 0)
    return jnp.pad(piece, pad)


def _drop(x, axis, count):
    """k ↓ X: without the first k cells along the axis (the last for a negative k)."""
    n = x.shape[axis]
    if count >= 0:
        return lax.slice_in_dim(x, count, n, axis=axis)
    return lax.slice_in_dim(x, 0, n + count, axis=axis)


def _gather(index, x):
    """I ⌷ X: the cell of X at the position each cell of I names, in row-major
    order, floored; zero where that is no position of X."""
    flat = x.reshape(-1)
    n = flat.shape[0]
    place = jnp.floor(index)
    valid = jnp.logical_and(place >= 0, place < n)
    at = jnp.clip(jnp.where(valid, place, 0), 0, n - 1).astype(jnp.int64)
    return jnp.where(valid, flat[at], _c(0.0))
"#;


/// The python to run an emitted module with: $RHO_JAX_PYTHON, else `python3`.
pub fn python() -> String {
    std::env::var("RHO_JAX_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// Whether that python can import jax.
pub fn available() -> bool {
    std::process::Command::new(python())
        .args(["-c", "import jax"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// What a run of an emitted module produced.
#[derive(Debug)]
pub struct Run {
    pub output: Vec<f64>,
    pub sweeps: u64,
    pub converged: bool,
}

/// Run an emitted module over `inputs` (one vector per declared input, in
/// declaration order) on the CPU through `rho_jit`, exchanging the cells
/// as raw doubles so nothing is rounded on the way.
pub fn run(module: &str, dir: &std::path::Path, inputs: &[Vec<f64>]) -> std::io::Result<Run> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("rho_module.py"), module)?;
    let mut bytes = Vec::new();
    for space in inputs {
        for v in space {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("inputs.bin"), bytes)?;
    let driver = r#"
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import rho_module as m
raw = np.fromfile(sys.argv[1] + '/inputs.bin', dtype=np.float64)
args, at = [], 0
for name in m.INPUTS:
    n = int(np.prod(m.SHAPES[name]))
    args.append(raw[at:at + n].reshape(m.SHAPES[name]))
    at += n
out = m.rho_jit(*args)
np.asarray(out[m.OUTPUT], dtype=np.float64).tofile(sys.argv[1] + '/output.bin')
print('sweeps', int(out['rho_sweeps']), 'converged', int(bool(out['rho_converged'])))
"#;
    std::fs::write(dir.join("driver.py"), driver)?;
    let out = std::process::Command::new(python())
        .arg(dir.join("driver.py"))
        .arg(dir)
        .output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "the JAX run failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let field = |name: &str| -> u64 {
        text.split_whitespace().skip_while(|w| *w != name).nth(1).and_then(|w| w.parse().ok()).unwrap_or(0)
    };
    let raw = std::fs::read(dir.join("output.bin"))?;
    let output = raw.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
    Ok(Run {
        output,
        sweeps: field("sweeps"),
        converged: field("converged") != 0,
    })
}
