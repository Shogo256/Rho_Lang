# ρ (RHO) — where 1.0 stands, and what is not in it

ρ is an array language in the APL line for grids: spaces with shapes, flows
that sweep them, shifts, folds and scans, lifts and broadcasts, a fixed point,
turns, a gather, a roll, functions, and an interval check. A program becomes
one of three things, and every one of them is held to the reference
interpreter — bit for bit where the target allows, and to a measured margin
where it does not (XLA's own reductions and transcendentals).

## In 1.0

| What | Where it is checked |
|---|---|
| The language: `◯ □ → = ▷ ▽ ◇ ◈ □ ⇒ ! ⍳ ⌈ ⌊ \| ⌽ ⍴ ⍉ ↑ ↓ ⌷ ?`, functions with expression or flow bodies, infix calls of two | SPECIFICATION §3; 158 tests |
| The reference interpreter, generic over the number type (f64, f32, fixed point) | §5; the differential test compares every kernel with it bit for bit |
| The compiled kernel: LLVM, `.so`, C ABI (`exec_with_args`, `exec_spaces`, `exec_bounded`, metadata), SIMD, threads, fused flows, kept scratch, one sweep a round | §3.4, §6; difftest at f64 and f32 with every sweep split across threads |
| Python: `rho.RhoEngine`, `rho.Kernel` (compile-and-cache per shape), `@rho.compile`, wheels on Linux, macOS and Windows | §3.4.2; CI smoke-tests the installed wheel |
| The command line: `rhoc` with `--run` / `--write`, `--emit-jax`, `--emit-sv`, `--fixed` | README; CI compiles the examples and runs one |
| The JAX module: every construct, `⇒` as `lax.while_loop`, bound spaces as arguments | §8; held to the interpreter within one ulp at the scale of the space |
| The circuit: every construct as a one-cell-per-clock SystemVerilog pipeline, `⇒` over block RAM, fixed point with magic-number constant division, Yosys and nextpnr numbers | §7; Verilator against the interpreter, and a circuit leg of difftest |
| The interval check `!`: binary64 rounding modelled, claims held to real runs | §4; difftest refutes, never trusts |

## Decided out (not bugs, not planned)

- The rank operator `⍤`, `f ⇒ U` sugar, compress `/`, grade `⍋`, nested arrays,
  strings, integers as a type, sort, FFT (§3.11, §9).
- A `⇒` inside a function body, and iterating `=` itself.
- Proofs: Z3, translation validation, convergence proofs and the contract in
  the `.so` were removed in favour of the differential test (see the
  specification's history).
- A hand-written GPU backend: the JAX module is the road to GPUs and TPUs.

## After 1.0, if wanted

- Pipelining a circuit stage's expression across clocks, so multipliers and
  constant divisions run at the register-to-register clock on parts without
  DSP slices (an iCE40 multiplier sits at 34 MHz because its adder tree is
  combinational).
- Stencil fusion in the kernel (an intermediate read under a shift), which
  needs a halo of recomputation; only cell-for-cell reads are fused today.
- Widths other than four lanes, chosen per machine, and Arm.
- A published PyPI release (the workflow exists; it needs a `v*` tag and the
  project's token).
