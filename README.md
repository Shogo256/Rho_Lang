# ρ (RHO)

[![CI](https://github.com/Shogo256/Rho_Lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Shogo256/Rho_Lang/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

```
   _ __
  | '_ \      ρ (RHO) 1.0.0
  | |_) |     an array language in the line of APL
  | .__/      one source → a native kernel, a JAX module, a circuit
  |_|
```

**An array language in the line of APL, compiled to native kernels.**

ρ writes computation over whole arrays — no indices, no loops. A shift reads a
neighbour, a fold collapses an axis, a scan runs along one, a lift stretches
one array against another, and `⇒` repeats a flow to a fixed point. Shapes are
declared, so a program becomes a straight native kernel — or a JAX module, or
a circuit — and every one of them is held to a reference interpreter.

```rho
{
    /* Jacobi: relax X until no cell moves by more than 𝜏 */
    INPUT:◯ □ 64 64
    INPUT → X
    ((INPUT / 4.0) + ((▷X + ▽X + ▷0X + ▽0X) / 4.0)) ⇒ X
    X → =
}
```

```bash
rhoc jacobi.rho --tau 1e-12 --max-iter 200 --run INPUT=field.bin --write OUTPUT=out.bin
```

## What a program becomes

| Target | How | Held to the interpreter |
|---|---|---|
| A native kernel (`.so`, C ABI, Python) | `rhoc program.rho` | bit for bit, at f64 and f32, with every sweep split across threads |
| A JAX module for CPU, GPU, TPU | `rhoc program.rho --emit-jax m.py` | bit for bit where XLA allows; within one ulp at the scale of the space where XLA reorders a fold or uses its own `exp` |
| A SystemVerilog pipeline, one cell per clock | `rhoc program.rho --emit-sv dir --fixed 32.16` | bit for bit, in `real` and in fixed point; Yosys and nextpnr synthesise it |

## What ρ can do

**Grids.** A program declares spaces with shapes of any rank and writes flows
over them, whole. Several inputs, several intermediates; a caller may read any
intermediate or leave it to the kernel. Shapes are fixed at compile time, so
every read has a known place and every buffer a known size.

**Neighbourhoods.** `▷` and `▽` read the cell before and after along any axis,
zero at the edge; `⌽` rotates, so a periodic boundary is one glyph; `↑` `↓`
take a window or a tail; `⍳` is each cell's coordinate, so masks, windows,
distances from a centre and Vandermonde matrices are one expression.

**Reductions.** `◇+ ◇× ◇> ◇<` collapse an axis, `◈` runs along it. Any fold
over any operation is an inner product: `◇+1 (A × B)` is a matrix product,
`◇<1 (D + E)` a min-plus product (one step of shortest paths). `□` lifts an
array against another: outer products, broadcasting, row and column
normalisation.

**Reads the data decides.** `I ⌷ X` reads X at the positions the cells of I
name: lookup tables, colour maps, resampling, permutations. `?X` hashes each
cell to a number in [0, 1) with no state, so a coordinate is a seed and a
Monte Carlo run gives the same numbers on any thread count.

**Iteration.** `expr ⇒ X` sweeps until no cell moves by more than `𝜏` or the
cap is reached, and the kernel reports which: Jacobi relaxation, diffusion,
distance fields, cellular automata, power iteration. Every round reads a
finished grid.

**Functions.** `name:{ params body }` with an expression or a body of flows,
called prefix (`smooth INPUT`) or, for two arguments, infix (`A mix B`).
Expanded at compile time and usable at any shape; a body inside `⇒` runs
every round.

**Numbers.** f64, f32, and fixed point `W.F` for circuits. Integers below
2^53 are exact in f64, so modular arithmetic (`q | X`) and counting are exact.

**Checks before running.** Undeclared spaces, shape mismatches and a missing
output are refused with a line. `! (expr)` is checked by interval arithmetic
that models binary64 rounding, so a claim that only holds over the reals is
not accepted.

**Three targets from one source.** A native shared library with a C ABI
(SIMD, threads, fused flows), a JAX module (`jit` and `vmap`; `grad` over
programs without `⇒`, which is a `while_loop`), and a SystemVerilog streaming pipeline
(one cell per clock, fixed point, synthesised by Yosys).

**Programs people write in it.** Image filters and morphology, stencil PDE
relaxation and diffusion, reaction–diffusion and cellular automata, Monte
Carlo and procedural noise, small linear algebra (products, norms, softmax,
layer normalisation), lookup and colour maps, FIR filtering and pulse
shaping, distance fields for path planning — and the same programs as FPGA
datapaths.

**Not in the language, by decision.** Scatter, sort and grade, strings,
nested arrays, data-dependent branching beyond `⇒`, an integer type, FFT —
each would break the one property everything rests on, that a cell is one
expression of other cells with a shape known in advance.

## Quick start

### Install

The Python package ships the compiler; clang (LLVM 15 or newer) must be on
the machine, since `rhoc` asks it to build the shared library.

```bash
pip install rho-lang                         # once a release is on PyPI
maturin build --release && pip install target/wheels/rho_lang-*.whl   # your own build
```

The compiler depends on no crate: its command line, errors, dataflow graph
and library loading are written in `src/`, so building it fetches nothing
but Rust's standard library. To work on the compiler itself (Rust 1.88+
and clang):

```bash
git clone https://github.com/Shogo256/Rho_Lang.git && cd Rho_Lang
cargo build --release && cargo test
```

### Show `.rho` files with ρ's icon

```bash
sh editors/linux/install.sh            # GNOME Files, Dolphin and other freedesktop file managers
```

### Compile and run

```bash
rhoc examples/matrix_add.rho                       # writes libkernel.so
printf '1 2 3 4\n5 6 7 8\n' > in.txt
rhoc examples/matrix_add.rho --run INPUT=in.txt    # runs it: prints OUTPUT, one row per line
```

| Flag | Effect |
|---|---|
| `--run SPACE=FILE` | Run the kernel once with this input read from a file: raw doubles for `.bin`, whitespace-separated numbers otherwise. Repeat per input |
| `--write SPACE=FILE` | After `--run`, write a space to a file; intermediates may be named too |
| `--tau <v>` | Bind the threshold `𝜏` (default `0.0`) |
| `--max-iter <N>` | Cap every `⇒` at `N` sweeps; required by a program that iterates |
| `--f32` | Compute at single precision |
| `--threads <N>` | Split every sweep across `N` threads; `0` (default) is one per CPU. `RHO_THREADS` overrides at run time |
| `--portable` | Build for any x86-64 rather than this machine |
| `--bind NAME=0x…` | Point a space at an address the caller owns |
| `--emit-lean <FILE.lean>`, `--lean-namespace <NAME>`, `--lean-program-only` | Write the source's functions as Lean 4 definitions over the integers, for proofs about them, and stop: sums, products, `⌊` of a quotient by a whole constant, `ind`, `abs`, calls; a function outside that subset is named in a comment and left out, and a program block of such flows becomes `kernel`, one cell of OUTPUT from the same cell of each input (alone with `--lean-program-only`). A source of definitions alone is accepted |
| `--emit-jax <FILE.py>` | Also write the program as a JAX module (`rho`, `rho_jit`) |
| `--emit-static <LIB.a>`, `--symbol-prefix <P>` | Also write the kernel as a static library whose symbols are `P_exec_spaces`, `P_metadata`, ... in place of `rho_kernel_*`, so that several kernels link into one program |
| `--emit-sv <DIR>` | Also write the program as a SystemVerilog pipeline with a Verilator harness |
| `--fixed <W.F>` | Make the circuit's cells fixed point, e.g. `32.16` |
| `--fuse` | With `--emit-sv`, compute each intermediate that one flow reads cell for cell inside that flow's stage, as the kernel does, and a part a stage reads twice once: fewer stages, registers and replay memories, and a longer path through a stage, so a slower clock |
| `--constant-time`, `--public NAME,…` | Refuse the program unless its cells may be secrets: no division or residue by a non-constant, no `sqrt` `exp` `log` `sin` `cos` or fractional power, no `⇒`, and `⌷` only at positions made of `⍳`, constants and the `--public` spaces |
| `--explain` | Say what each flow became: the sweeps a call runs, vector or one cell at a time and why, the folds computed first, what fused into what, a `⇒`'s rounds; with `--emit-sv`, the circuit's stages, products and memories |
| `--dump-llvm`, `--dump-dag`, `--no-simd` | Print the IR, print the dataflow, emit scalar loops only |
| `--lang <en\|ja>` | The language of the hints under a refusal (`RHO_LANG` too; default English) |

### When a line is refused

A refusal names its line and, when the line looks like another notation's
way of writing it — a fold after its operand, APL's `+/`, NumPy's
`np.sum(X)`, an index `X[i+1]`, `S = expr`, a parenthesis left open, a
misspelt name, a space one axis short — says how ρ writes it:

```
3 |     X ◇+1 → =
  |     ^
  = hint: a fold or a scan comes before what it walks: `◇+1 X`, not `X ◇+1`
  = try (3): (◇+1 X) → =
```

Every `try` line has been parsed and shape-checked with the rest of the
program before it is printed, and when two readings both pass, both are
shown. The differential test rewrites the programs it generates in those
habits and holds every hint to giving back a program that computes the
same cells, bit for bit. `--lang ja` prints the hints in Japanese.

### From Python

Write the flows once, without declarations; each shape you call with is
compiled once and cached (`~/.cache/rho-lang`, or `RHO_CACHE_DIR`).

```python
from rho import Kernel

blur = Kernel("((▷0INPUT + ▽0INPUT + ▷1INPUT + ▽1INPUT + INPUT) / 5.0) → =")
out = blur(INPUT=image)                       # nested lists or numpy arrays
mix = Kernel("((A × W) + (B × (1.0 - W))) → =")
out = mix(A=a, B=b, W=w)                      # every named argument is a space

relax = Kernel("INPUT → X\n((INPUT / 4.0) + ((▷X + ▽X) / 4.0)) ⇒ X\nX → =", tau=1e-12, max_iter=200)
x = relax(INPUT=b); relax.sweeps(), relax.converged()
```

Against a compiled `.so` by hand, `rho.RhoEngine` wraps the C ABI and checks
buffer lengths:

```python
from rho import RhoEngine
engine = RhoEngine()
engine.compile_rho_file("examples/matmul.rho")
engine.spaces()      # [('A', [2,3,1], 'input'), ('B', [1,3,4], 'input'), ('OUTPUT', [2,4], 'output')]
engine.execute_spaces({"A": a, "B": b, "OUTPUT": c})
```

## The language, by example

```rho
{
    /* a softmax */
    INPUT:◯ □ 1024
    exp INPUT → E
    (E / (□0 (◇+ E))) → =
}
```

```rho
{
    /* a matrix product: APL's +.× — any fold over any operation is an inner product */
    A:◯ □ 2 3 1
    B:◯ □ 1 3 4
    ◇+1 (A × B) → =
}
```

```rho
step:{ U
    (▷U + ▽U) → S
    (S / 4.0) → =
}
{
    /* a function with a body of flows, run every round of the loop */
    INPUT:◯ □ 1024
    INPUT → X
    ((INPUT / 4.0) + (step X)) ⇒ X
    X → =
}
```

```rho
{
    INPUT:◯ □ 1024 1024
    (▷INPUT - INPUT) → △
    (▽INPUT - INPUT) → ▽
    ((△ - ▽) / (△ + ▽)) ^ 2 → OUTPUT
    ! (OUTPUT >= 0)
    OUTPUT > 𝜏 → =
}
```

`▷` and `▽` read the neighbours along the last axis (`▷0` steps between rows),
zero at the boundary. A comparison passes the left value through where it
holds and gives 0 elsewhere. `!` is checked at compile time by interval
arithmetic that models binary64 rounding; here it proves `OUTPUT >= 0` and
says the denominator ranges over `[-inf, +inf]`, which includes zero — that is
the formula, not a compiler fault.

The glyphs: shifts `▷ ▽`, folds `◇+ ◇× ◇> ◇<`, scans `◈`, lift `□`, fixed
point `⇒`, coordinates `⍳`, the greater, lesser and residue `⌈ ⌊ |`, rotate
`⌽`, reshape `⍴`, transpose `⍉`, take and drop `↑ ↓`, index by value `⌷`,
roll `?`, `exp log sqrt sin cos abs ind`, and functions `name:{ params body }`
called prefix or, for two, infix (`A mix B`). Every glyph has an ASCII alias;
the wiki's Syntax Reference lists them all.

## C ABI

| Symbol | Signature | Purpose |
|---|---|---|
| `rho_kernel_exec_spaces` | `void (void **spaces)` | One pointer per space, in the order `rho_kernel_metadata()` lists them. Inputs must be given; a null intermediate is the kernel's own, and a null fused one is never stored |
| `rho_kernel_exec_with_args` | `void (const double *in, double *out)` | The two-pointer form; buffers hold `rho_kernel_element_count()` values |
| `rho_kernel_exec_bounded` | `void (const double *in, double *out, int64_t n)` | Same, clamped to `n` cells |
| `rho_kernel_metadata` | `const char * (void)` | JSON: every space's name, shape and role, the loop's cap and tolerance, and `reentrant`: whether two threads may call the kernel at once (true unless it keeps scratch between calls) |
| `rho_kernel_sweeps`, `rho_kernel_converged` | `int64_t (void)` | Sweeps the last call's `⇒` took, and whether every one settled (with calls at once, one of them) |
| `rho_kernel_element_count` | `int64_t (void)` | The larger of the input's and the output's cell counts |
| `rho_kernel_exec` | `void (void)` | Zero-copy: runs against the addresses bound with `&[0x…]` or `--bind` |
| `rho_kernel_zeroize` | `void (void)` | Overwrite the scratch kept between calls and the stack an entrypoint's frame used; call it from where the kernel was called, before letting secrets go |

## Numbers

Measured on an i7-8550U, best of 20, one thread, with a C harness calling
the kernel; the circuit numbers come from Yosys 0.67 and nextpnr 0.11.

| | |
|---|---|
| Three flows over a million cells | 1.4 ms (was 11.6 before fusion and kept scratch) |
| Jacobi through a `step` body, four rounds, a million cells | 11 ms (was 51) |
| A 321-tap FIR over 160 samples as a `⌷` window and a fold | 13 µs (was 95; a sliding dot product in plain Rust: 10–11) |
| Threads on cache-resident grids | 3–4x, bit-identical to one thread; a million cells is memory-bound |
| A 16×16 stencil as a circuit, Q16.16 | 401 LUT4 at 75.8 MHz on an iCE40 HX8K |
| A 48-cell Jacobi loop as a circuit, Q16.16 | 642 LUT4, 3 block RAMs at 89 MHz on an ECP5 (104 MHz through the `step` body) |

Two caveats: past the cache the memory bus is the limit, and the `!` check
models round-to-nearest without infinities, subnormals or NaN.

## What ρ claims — and how each claim is checked

1. **One meaning, three targets.** The compiled kernel gives the reference
   interpreter's bits, at f64 and f32, through both entrypoints, with every
   sweep split across threads. The circuit gives them too, in `real` and in
   fixed point. The JAX module gives them where XLA allows, and stays within
   one ulp at the scale of the space where XLA reorders a fold or uses its own
   `exp`. *Checked by:* a differential test that compiles random programs on
   every push and compares them with the interpreter bit for bit; a circuit
   leg through Verilator; a JAX test with measured tolerances.

2. **Determinism.** The result does not depend on the number of threads, the
   grain or the run, nor — with `--portable` and the same C library, whose
   `exp`, `log`, `sin` and `cos` the kernel calls — on the machine. The roll is
   a hash of the cell's value, so randomness is reproducible. The same source produces
   byte-identical IR. *Checked by:* the threads test (1, 4, 7 and all threads
   give the same bits), the determinism step in CI, the differential test's
   split sweeps.

3. **No data-dependent control flow.** Every cell evaluates the same
   expression; there are no branches, so no instruction sequence depends on
   the values. What still can: the address a `⌷` reads (a memory access), the
   number of rounds a `⇒` takes, and the latency of division and square root
   on processors where it varies. A program that must hide its data keeps
   `⌷`'s positions public, has no `⇒`, and divides only by constants.
   *Checked by:* the language itself, which has no branch to test, and
   `rhoc --constant-time`, which refuses a program that does any of the three
   (`--public` names the tables a position may come from).
   `rho_kernel_zeroize` overwrites what a call leaves in memory.

4. **No out-of-bounds access.** Shapes are static, shifts pad zero, `⌷` reads
   zero past the end, and the metadata states every buffer's size — given
   buffers of those sizes, no read or write leaves them. *Checked
   by:* the shape checks at compile time and the differential test at run
   time.

5. **The `!` check does not lie about rounding.** Its intervals model
   round-to-nearest binary64 (or binary32 under `--f32`), so it refuses
   claims that hold only over the reals. It assumes no overflow, subnormals or
   NaN. *Checked by:* the differential test, which holds every claim to what
   the kernel produced (refutation, not proof).

6. **Termination.** A flow is a finite sweep; a `⇒` stops on the tolerance
   or the cap, and the kernel says which. *Checked by:* the cap is required
   at compile time.

7. **Performance is the memory bus.** On grids past the cache a sweep runs at
   the machine's bandwidth; a chain of flows is one sweep, a `⇒` round is one
   sweep, threads give 3–4x in cache. *Checked by:* the numbers above,
   measured with a C harness.

**What it does not claim.** It is not faster than hand-tuned BLAS or cuDNN
and does not try to be; the JAX module is not bit-identical; the checks are
tests and interval arithmetic, not machine-checked proofs; a circuit's clock
on parts without DSP slices is set by combinational multipliers (34 MHz for
a Q16.16 constant division on an iCE40); and the language leaves out what
the list above says it leaves out.

## Read on

- [Wiki](https://github.com/Shogo256/Rho_Lang/wiki) — getting started, the syntax reference, how it is verified
- [CHANGELOG.md](CHANGELOG.md)
- Bugs go to Issues, with the `.rho` that shows them

Apache 2.0.
