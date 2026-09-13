# ρ (RHO)

[![CI](https://github.com/Escoffier1156/Rho_Lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Escoffier1156/Rho_Lang/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

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

## Quick start

### Install

The Python package ships the compiler; clang (LLVM 15 or newer) must be on
the machine, since `rhoc` asks it to build the shared library.

```bash
pip install rho-lang                         # once a release is on PyPI
maturin build --release && pip install target/wheels/rho_lang-*.whl   # your own build
```

To work on the compiler itself (Rust 1.88+ and clang):

```bash
git clone https://github.com/Escoffier1156/Rho_Lang.git && cd Rho_Lang
cargo build --release && cargo test
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
| `--emit-jax <FILE.py>` | Also write the program as a JAX module (`rho`, `rho_jit`) |
| `--emit-sv <DIR>` | Also write the program as a SystemVerilog pipeline with a Verilator harness |
| `--fixed <W.F>` | Make the circuit's cells fixed point, e.g. `32.16` |
| `--dump-llvm`, `--dump-dag`, `--no-simd` | Print the IR, print the dataflow, emit scalar loops only |

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
called prefix or, for two, infix (`A mix B`). Every glyph has an ASCII alias.
The specification has the dictionary.

## C ABI

| Symbol | Signature | Purpose |
|---|---|---|
| `rho_kernel_exec_spaces` | `void (void **spaces)` | One pointer per space, in the order `rho_kernel_metadata()` lists them. Inputs must be given; a null intermediate is the kernel's own |
| `rho_kernel_exec_with_args` | `void (const double *in, double *out)` | The two-pointer form; buffers hold `rho_kernel_element_count()` values |
| `rho_kernel_exec_bounded` | `void (const double *in, double *out, int64_t n)` | Same, clamped to `n` cells |
| `rho_kernel_metadata` | `const char * (void)` | JSON: every space's name, shape and role, and the loop's cap and tolerance |
| `rho_kernel_sweeps`, `rho_kernel_converged` | `int64_t (void)` | Sweeps the last call's `⇒` took, and whether every one settled |
| `rho_kernel_element_count` | `int64_t (void)` | The larger of the input's and the output's cell counts |
| `rho_kernel_exec` | `void (void)` | Zero-copy: runs against the addresses bound with `&[0x…]` or `--bind` |

## Numbers

Measured on an i7-8550U, best of 20, one thread; details and the circuit
numbers are in the specification (§6, §7, §8).

| | |
|---|---|
| Three flows over a million cells | 1.4 ms (was 11.6 before fusion and kept scratch) |
| Jacobi through a `step` body, four rounds, a million cells | 11 ms (was 51) |
| Threads on cache-resident grids | 3–4x, bit-identical to one thread; a million cells is memory-bound |
| A 16×16 stencil as a circuit, Q16.16 | 401 LUT4 at 75.8 MHz on an iCE40 HX8K |
| A 48-cell Jacobi loop as a circuit, Q16.16 | 642 LUT4, 3 block RAMs at 89 MHz on an ECP5 (104 MHz through the `step` body) |

Two caveats: past the cache the memory bus is the limit, and the `!` check
models round-to-nearest without infinities, subnormals or NaN.

## How it stays right

A reference interpreter, written from the specification and generic over the
number type, is the meaning of a program. On every push the differential test
compiles random programs at f64 and f32 with every sweep split across
threads, runs them through both entrypoints, and compares them with the
interpreter bit for bit; whatever the `!` check claimed about them is held to
what the kernel produced; the circuit leg does the same through Verilator.

## Read on

- [docs/SPECIFICATION.md](docs/SPECIFICATION.md) — the language, the entrypoints, the numbers
- [docs/ROADMAP.md](docs/ROADMAP.md) — what 1.0 is, what was decided out, what could follow
- [CHANGELOG.md](CHANGELOG.md)

Apache 2.0.
