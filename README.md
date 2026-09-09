# ρ (RHO) Language Compiler

[![CI](https://github.com/Escoffier1156/Rho_Lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Escoffier1156/Rho_Lang/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![LLVM](https://img.shields.io/badge/LLVM-15%2B-dragon.svg)](https://llvm.org)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)

**An array language in the line of APL, compiled to native kernels.**

ρ (RHO) writes computation over whole arrays — no indices, no loops. Each glyph
is an array operation: a shift reads a neighbour, a fold collapses an axis, a
scan runs along one, a lift stretches one array against another, and `⇒`
repeats a flow to a fixed point. Shapes are declared, so every program compiles
to a straight native kernel: the compiler emits LLVM IR and links a shared
library callable from C or Python.

---

## Current Status

Working prototype. What runs today:

- Parses the RHO symbol set, with ASCII aliases for every glyph
- Static validation: undeclared spaces, shape mismatches, missing equilibrium point
- `!` constraints checked at compile time by interval arithmetic, which models
  binary64 rounding rather than ℝ
- Lowers flows to LLVM IR — one full grid sweep per `→`
- Multi-dimensional shifts `▷` / `▽`, per axis, zero-padded at each axis's boundary
- Folds `◇+` `◇×` `◇>` `◇<` that collapse an axis, so sums, means, dot products
  and norms are one line each
- Scans `◈+` `◈×` `◈>` `◈<` for running totals, which keep the shape they walk
- Named functions `exp` `log` `sqrt` `sin` `cos` `abs`, with their domains
  checked, and `ind` so a program can count
- APL's dyadic `⌈` `⌊` `|` — the greater, the lesser and the residue — so a
  ReLU is `X ⌈ 0.0`, a clamp is `(X ⌊ 1.0) ⌈ -1.0`, and `3.0 | X` is X mod 3
- `--f32` for single precision, with the interpreter and the `!` check
  following the width
- `□` lifting and broadcasting, so an outer product — and a matrix product — is
  one flow
- `⇒`, a flow repeated to a fixed point — APL's `f⍣≡`: Jacobi iteration,
  relaxation and diffusion in one line, capped by `--max-iter`, with the kernel
  reporting how many sweeps it took and whether it settled
- Explicit `<4 x double>` vector lowering, verified bit-identical to the scalar path
- Zero-copy binding: compile a kernel against a buffer the host already owns
- One pointer per space at call time, so a kernel with several inputs — a
  matrix product — runs with nothing baked in
- Emits a native shared library (`.so`) with a documented C ABI
- Deterministic output: the same source always produces byte-identical IR
- A reference interpreter written from the specification; on every push random
  programs are compiled at both widths, run through both C entrypoints, and
  compared with it bit for bit — and whatever the `!` check claimed about them
  is compared with what the kernel actually produced

See [Implementation Status](#implementation-status) for what is designed but not
yet built. The implementation is deliberately a small core, not a full language
runtime.

---

## Quick Start

### 1. Install or build

The Python package ships the compiler, so a Rust toolchain is not needed to
use it. clang (LLVM 15 or newer) is: `rhoc` emits LLVM IR and asks clang to
build the shared library.

```bash
pip install rho-lang          # once a release is published to PyPI
```

Until then, or to install your own build, make the wheel with
[maturin](https://www.maturin.rs) — CI builds it for Linux, macOS and Windows
on every push, and uploads it as an artifact:

```bash
maturin build --release
pip install target/wheels/rho_lang-*.whl
```

To work on the compiler itself, build from source (Rust 1.80+ and clang):

```bash
git clone https://github.com/Escoffier1156/Rho_Lang.git
cd Rho_Lang
cargo build --release
cargo test
```

Or use the container, which pins Rust, LLVM, Clang and Python:

```bash
docker build -t escoffier1156/rho-lang .
docker run -it escoffier1156/rho-lang
```

### 2. Compile a kernel

```bash
cargo run --release --bin rhoc -- examples/matrix_add.rho
```

This writes `libkernel.so` and reports how many cells the kernel sweeps, along
with what the `!` check could and could not settle.

| Flag | Effect |
|---|---|
| `--dump-llvm` | Print the generated IR |
| `--dump-dag` | Print the dataflow trace |
| `--tau <v>` | Bind the threshold symbol `𝜏` (default `0.0`) |
| `--bind NAME=0x…` | Point a space at an address the caller owns |
| `--f32` | Compute at single precision |
| `--max-iter <N>` | Cap every `⇒` at `N` sweeps; required by a program that iterates |
| `--no-simd` | Emit only scalar loops |

### 3. Call it from Python

```python
import ctypes

lib = ctypes.CDLL('./libkernel.so')

# A kernel's grid size comes from its ◯ □ declaration. Ask for it rather than
# guessing — a buffer shorter than this is a memory error, not a short result.
lib.rho_kernel_element_count.restype = ctypes.c_int64
n = lib.rho_kernel_element_count()

Buffer = ctypes.c_double * n
src = Buffer(*[float(i + 1) for i in range(n)])
dst = Buffer()

lib.rho_kernel_exec_with_args.argtypes = [ctypes.POINTER(ctypes.c_double)] * 2
lib.rho_kernel_exec_with_args.restype = None
lib.rho_kernel_exec_with_args(src, dst)

print(list(dst))          # [2.0, 4.0, 6.0, 8.0]
```

Or through the wrapper, which checks buffer lengths for you. Installed from
the wheel it runs the `rhoc` it shipped; imported from a checkout it runs
`cargo run`, so an edit to the compiler is what executes:

```python
from rho import RhoEngine

engine = RhoEngine()
engine.compile_rho_file("examples/matrix_add.rho")
print(engine.element_count(), engine.get_metadata())
engine.execute_kernel_with_args(src, dst)
```

### 4. Or skip the pointers entirely

Compile the kernel against the address of a buffer you already own, and the
call takes no arguments at all:

```python
buf = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)

engine = RhoEngine()
engine.compile_rho_file("examples/matrix_add.rho", bind={"INPUT": buf})
engine.execute_kernel(buf)

print(list(buf))          # [2.0, 4.0, 6.0, 8.0] — written in place
```

### 5. Iterate to a fixed point

`expr ⇒ U` sweeps `expr` into `U` again and again until no cell moves by more
than `𝜏` or `--max-iter` sweeps have run. Jacobi for a tridiagonal system:

```rho
{
    INPUT:◯ □ 64 1
    INPUT → X
    ((INPUT - (▷X + ▽X)) / 4.0) ⇒ X
    X → =
}
```

```python
engine.compile_rho_file("examples/jacobi.rho", tau=1e-12, max_iter=200)
engine.execute_kernel_with_args(b, x)
engine.sweeps(), engine.converged()     # (40, True)
```

Every sweep reads the whole previous grid — a Jacobi step — so a shift inside
the loop sees a finished grid, as it does everywhere else. The cap is required:
it is what makes the kernel terminate, and it changes the answer when the loop
has not settled, so the kernel says which happened.

### 6. Several inputs

A kernel is not limited to one input. `rho_kernel_exec_spaces` takes one
pointer per space, in the order the metadata lists them; the wrapper builds the
table from a dictionary. Inputs must be present, the output is where the result
lands, and any intermediate may be left out for the kernel to own:

```python
a = (ctypes.c_double * 6)(1, 2, 3, 4, 5, 6)        # 2x3
b = (ctypes.c_double * 12)(*range(12))             # 3x4
c = (ctypes.c_double * 8)()                        # 2x4

engine = RhoEngine()
engine.compile_rho_file("examples/matmul.rho")
engine.spaces()                     # [('A', [2,3,1], 'input'), ('B', [1,3,4], 'input'), ('OUTPUT', [2,4], 'output')]
engine.execute_spaces({"A": a, "B": b, "OUTPUT": c})
```

---

## C ABI

| Symbol | Signature | Purpose |
|---|---|---|
| `rho_kernel_element_count` | `int64_t (void)` | Cells the kernel sweeps; the minimum buffer length |
| `rho_kernel_exec_with_args` | `void (const double *in, double *out)` | Run the kernel. Buffers **must** hold `element_count()` doubles |
| `rho_kernel_exec_bounded` | `void (const double *in, double *out, int64_t n)` | Same, but clamps the sweep to `n` cells |
| `rho_kernel_exec_spaces` | `void (void **spaces)` | One pointer per space, in the order `rho_kernel_metadata()` lists them. The way to call a kernel with several inputs |
| `rho_kernel_metadata` | `const char * (void)` | JSON: element count, every space's shape and role, and the iteration cap and tolerance when the kernel iterates |
| `rho_kernel_sweeps` | `int64_t (void)` | Sweeps the `⇒` loops of the most recent call took, in all |
| `rho_kernel_converged` | `int64_t (void)` | 1 if every `⇒` of the most recent call stopped on the tolerance rather than on the cap |
| `rho_kernel_exec` | `void (void)` | Zero-copy: runs against the addresses compiled in via `&[0x…]` or `--bind`. Returns immediately if `INPUT` is unbound |

Passing `NULL` as the output pointer makes the kernel write in place. Passing
`NULL` as the input pointer makes it return without touching memory.

In the `spaces` table, each entry's metadata `role` says what the pointer is
for: an `input` is read and must be non-null, or the call returns without
touching memory; the `output` receives the result; an `internal` space may be
`NULL`, in which case the kernel uses scratch of its own, or supplied, in which
case the caller gets to see the intermediate. Buffers must hold as many values
as the space's shape multiplies out to.

---

## Example Syntax

```rho
{
    /* a softmax */
    INPUT:◯ □ 1024 1
    exp INPUT → E
    (E / (□0 (◇+ E))) → =
}
```

```rho
{
    /* a matrix product, contracted in one flow */
    A:◯ □ 2 3 1
    B:◯ □ 1 3 4
    ◇+1 (A × B) → =
}
```

That is APL's `+.×`, and the shape is the general inner product: any fold over
any operation. Fold with `<` over `+` and the same two lines are the min-plus
product — one step of a shortest-path relaxation:

```rho
{
    /* shortest paths of at most two edges: C[i][j] = min over k of D[i][k] + D[k][j] */
    D:◯ □ 4 4 1
    E:◯ □ 1 4 4
    ◇<1 (D + E) → =
}
```

Likewise `□1A f □0B` is the outer product under any operation `f`, and
`◈+ ((X × 0.0) + 1.0)` counts `1, 2, …` along an axis — an index by scan, which
costs a sweep an index generator would not.

```rho
{
    /* the mean and the L2 norm of a vector */
    INPUT:◯ □ 1024 1
    ((◇+ INPUT) / 1024.0) → MEAN
    ((◇+ (INPUT × INPUT)) ^ 0.5) → NORM
    (NORM - MEAN) → =
}
```

```rho
{
    &[0x7A4F]:INPUT:◯ □ 1024 1024
    (▷INPUT - INPUT) → △
    (▽INPUT - INPUT) → ▽
    ((△ - ▽) / (△ + ▽)) ^ 2 → OUTPUT
    ! (OUTPUT >= 0)
    OUTPUT > 𝜏 → =
}
```

`▷INPUT` reads the cell before the current one and `▽INPUT` the cell after, both
zero at the boundary, so the first two flows are the forward and backward spatial
differences. On this `1024 1024` grid those neighbours are along a row; write
`▷0INPUT` to step between rows instead. `𝜏` is a threshold, 0.0 unless `--tau`
says otherwise; a comparison passes the left value through where it holds and
collapses the cell to 0 elsewhere.

The compiler will tell you that this example can divide by zero:

```
[proved]   (OUTPUT >= 0)
[unproven] ((△ - ▽) / (△ + ▽)) — the denominator ranges over [-inf, +inf], which includes zero
```

That is the formula, not a compiler fault: `△ + ▽` is the discrete Laplacian, so
smooth input drives it to zero and the kernel returns infinities.

---

## Implementation Status

| Area | Status |
|---|---|
| Parser, ASCII aliases, symbol validation | ✅ implemented |
| Shape / flow / undeclared-space checks | ✅ implemented |
| LLVM lowering, one sweep per `→` | ✅ implemented |
| Multi-dimensional indexing and per-axis shifts | ✅ implemented |
| Folds (`◇`) collapsing an axis | ✅ implemented — scalar inner loop, no scan yet |
| Scans (`◈`) keeping the shape | ✅ implemented — scalar, no parallel scan |
| Lifting (`□`) and broadcasting | ✅ implemented — no implicit rank promotion |
| Fixed points (`⇒`) | ✅ implemented — Jacobi sweeps to a tolerance under a cap; a Gauss–Seidel sweep and a multi-flow loop body are not expressible |
| Named functions and their domain checks | ✅ implemented |
| Single precision (`--f32`) | ✅ implemented — 5.5x on a bandwidth-bound kernel |
| Mixed precision, integer types | 📋 not planned — see the specification |
| Explicit `<4 x double>` vector lowering | ✅ implemented — see the note below |
| Zero-copy binding (`&[0x…]`, `--bind`) | ✅ implemented |
| `!` constraint check | ✅ interval arithmetic that models binary64 rounding, not ℝ; its claims are held to real runs by the differential test |
| Diagnostics with source lines | ✅ implemented |
| Reference interpreter + differential testing | ✅ implemented — both widths, both entrypoints, on every push |
| C ABI, JSON metadata, Python FFI | ✅ implemented |
| `pip install rho-lang` | ✅ wheel built in CI for Linux, macOS, Windows; publishing to PyPI waits on a tagged release |
| AVX-512 / NEON width selection, GPU backends | 📋 planned — the vector width is fixed at four lanes |
| Tiling and cache blocking | 📋 planned — a sweep is one linear pass |
| Parallel execution of independent flows | 📋 planned — flows run in source order on one thread |

Two honest caveats on the ✅ rows:

- **Vector lowering is correct, not dramatically faster.** `clang -O3` vectorises
  no loops on the scalar IR by itself — the boundary select defeats it — and the
  explicit path roughly doubles the packed-double instructions emitted. Wall clock
  still only improves 1.0–1.1x on large grids, because streaming megabytes of
  doubles is bound by memory bandwidth. `--no-simd` is verified to produce
  bit-identical results.
- **The `!` check assumes no overflow, underflow or NaN.** It models
  round-to-nearest — every result is the exact one times `(1 ± 2⁻⁵³)` — which is
  what stops it accepting things that only hold over the reals. It does not model
  infinities, subnormals or NaN.

---

## Design Direction

ρ is an array language in the line of APL: matrix and parallel computation
written over whole arrays, with the shapes known at compile time so that every
program becomes a native kernel. What gets added next is judged by one
question — does it make array code shorter and clearer in that sense? The
checks the compiler runs on itself are how it stays correct while that happens;
they are not the product.

See [docs/SPECIFICATION.md](docs/SPECIFICATION.md) for the symbol dictionary and
[docs/ROADMAP.md](docs/ROADMAP.md) for the staged plan.

---

## License

Apache 2.0
