# Changelog

## 1.0.0 — 2026-09-14

The first release: the language as specified in `docs/SPECIFICATION.md`, and
three things a program becomes, each held to the reference interpreter.

- **Language.** Spaces and shapes, flows, zero-padded shifts along any axis,
  folds and scans, lifts and broadcasts (outer products, matrix products), the
  fixed point `⇒` with a sweep cap, the interval check `!`, coordinates `⍳`,
  the greater and lesser `⌈ ⌊`, the residue `|`, the turns `⌽ ⍴ ⍉ ↑ ↓`, the
  gather `⌷`, the roll `?`, floor and ceiling, and functions with expression
  or flow bodies (a body of flows inside `⇒` runs every round), called prefix
  or, for two arguments, between them.
- **Kernel.** LLVM to a shared library with a C ABI: two-pointer, bounded and
  every-space entrypoints, metadata, sweeps and convergence. Four-lane
  vectors with per-lane boundaries, sweeps split across a thread pool
  (bit-identical to one thread), intermediates fused into their one reader,
  scratch kept between calls, one sweep per `⇒` round.
- **Python.** `RhoEngine`, `Kernel` (compiled and cached per shape),
  `@rho.compile`; wheels built for Linux, macOS and Windows.
- **Command line.** `rhoc` compiles; `--run` and `--write` feed files in and
  out; `--emit-jax` writes a JAX module; `--emit-sv` a SystemVerilog
  pipeline, `--fixed W.F` in fixed point, which Yosys synthesises.
- **Checks.** A reference interpreter generic over the number type; a
  differential test that compiles random programs at f64 and f32, with every
  sweep split, and compares them bit for bit — and holds the interval check's
  claims to the runs; a circuit leg through Verilator; 158 tests.

Numbers are in the specification (§6, §7, §8) and were measured, not
estimated.
