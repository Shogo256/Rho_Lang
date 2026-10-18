# Changelog

## Unreleased

Found while writing the arithmetic of ML-KEM in ρ (the Rhodium project):
seven NTT layers, a matrix-vector product mod q, and the tools that work
needs.

- **The analysis no longer expands a deep chain whole.** A definition whose
  expansion passes 2^14 nodes is read as a free value named for it and the
  cell; seven NTT layers had run `rhoc` out of memory.
- **A row stretched over rows keeps the vector path.** A stretched read
  whose shared innermost axes span a multiple of the vector width now gets
  vector loads, the body aligned to the width; such flows had run one cell
  at a time (an NTT layer's arithmetic: 9.3 → 2.4 us).
- **A fold is not copied into a reader that names it twice.** Each copy had
  run the fold again as a pass of its own (a matrix-vector product: 4.7 ms
  → 82 us, 60 s → 3 s to compile).
- **`ind` of a comparison is its truth in the circuit and in JAX**, as it
  was in the interpreter and the kernel; both had read the comparison's
  mask, which is 0 where it passed a zero. A formal proof of an ML-KEM
  butterfly found it; the integer mode of the differential test now makes
  such comparisons often.
- **`rho_kernel_zeroize`** overwrites the scratch kept between calls and the
  stack an entrypoint's frame used.
- **`rhoc --constant-time`** (with `--public` for tables) refuses division
  and residue by a non-constant, `sqrt` `exp` `log` `sin` `cos` and
  fractional powers, `⇒`, and `⌷` at positions not made of `⍳`, constants
  and public spaces.
- **`DIFFTEST_MODQ=1`**: random integer programs mod q, the kernel against
  the interpreter, 32-bit fixed point against binary64, the constant-time
  rules, and with `DIFFTEST_SV=1` the circuit at `--fixed 32.0`.
- A fixed-point circuit's header says its cells are fixed point.
- **`rhoc --explain`** prints what each flow became: the sweeps a call runs,
  vector or one cell at a time and why, the folds computed as passes of their
  own, which intermediates fused into which sweep, what a `⇒` runs every
  round, and what `rho_kernel_zeroize` overwrites; with `--emit-sv`, the
  circuit's stages, its products (of two variables, by a constant, by one
  bit) and its replay memories. Nothing it prints changes the code.
- **A part's loads come in slot order.** The pointers an outlined sweep loads
  at its entry followed the addresses of the fold nodes they came from, so a
  build that allocated differently could emit other IR for the same source.

## 1.0.0 — 2026-09-14

The first release: the language, three things a program becomes, each held
to the reference interpreter, and the wiki that describes them.

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

Numbers are in the README and the wiki and were measured, not estimated.
