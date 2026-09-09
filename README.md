# ρ (RHO) Language Compiler

[![CI](https://github.com/Escoffier1156/Rho_Lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Escoffier1156/Rho_Lang/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![LLVM](https://img.shields.io/badge/LLVM-15%2B-dragon.svg)](https://llvm.org)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)

**A clockless, topological dataflow language for numeric computation.**

ρ (RHO) is a mathematics-driven programming language and compiler prototype that
explores a spatial dataflow model: computation is written as transformations over
a grid rather than as loops over time. The compiler parses a compact symbolic
syntax, validates shape and flow constraints, emits LLVM IR, and links a native
shared library callable from C or Python.

---

## Current Status

Working prototype. What runs today:

- Parses the RHO symbol set, with ASCII aliases for every glyph
- Static validation: undeclared spaces, shape mismatches, missing equilibrium point
- Constraint solving for `!` — interval arithmetic by default, Z3 with a feature flag
- Lowers flows to LLVM IR — one full grid sweep per `→`
- Multi-dimensional shifts `▷` / `▽`, per axis, zero-padded at each axis's boundary
- Folds `◇+` `◇×` `◇>` `◇<` that collapse an axis, so sums, means, dot products
  and norms are one line each
- Scans `◈+` `◈×` `◈>` `◈<` for running totals, which keep the shape they walk
- Named functions `exp` `log` `sqrt` `sin` `cos` `abs`, with their domains
  checked, and `ind` so a program can count
- `--f32` for single precision, with the rounding model and the contract
  following the width
- `□` lifting and broadcasting, so an outer product — and a matrix product — is
  one flow
- Explicit `<4 x double>` vector lowering, verified bit-identical to the scalar path
- Zero-copy binding: compile a kernel against a buffer the host already owns
- Emits a native shared library (`.so`) with a documented C ABI
- Deterministic output: the same source always produces byte-identical IR
- A machine-readable contract compiled into the `.so`, so a caller can check what
  was proved at load time
- A reference interpreter written from the specification, and a reader that runs
  the emitted IR without clang, so the source, the IR and the `.so` are compared
  against each other on every push
- Translation validation: the emitted IR is *proved* to compute what the source
  means, for every input at a given shape, with a negative control that damages
  the IR to show the check has teeth

See [Implementation Status](#implementation-status) for what is designed but not
yet built. The implementation is deliberately a small verifiable core, not a
full language runtime.

---

## Quick Start

### 1. Build

Local toolchain (Rust 1.80+, and clang with LLVM 15 or newer):

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
with what the constraint solver could prove.

| Flag | Effect |
|---|---|
| `--dump-llvm` | Print the generated IR |
| `--dump-dag` | Print the dataflow trace |
| `--tau <v>` | Bind the threshold symbol `𝜏` (default `0.0`) |
| `--bind NAME=0x…` | Point a space at an address the caller owns |
| `--f32` | Compute at single precision |
| `--no-simd` | Emit only scalar loops |
| `--require-contract` | Refuse to emit a kernel with unproven obligations |

Build with `--features z3-solver` to discharge `!` constraints with SMT instead
of interval arithmetic.

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

Or through the wrapper, which checks buffer lengths for you:

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

---

## The kernel carries its contract

What the solver proves is compiled into the artifact, not just printed during the
build. A host can check it before it runs anything:

```python
engine.compile_rho_file("kernel.rho")
engine.require_contract()          # raises if anything is unproven
```

```json
{
  "backend": "interval",
  "output_range": [0, null],
  "divisions_proven_safe": true,
  "output_proven_finite": false,
  "open_obligations": 0,
  "assumes": ["no overflow to infinity", "no NaN input", ...]
}
```

`rhoc --require-contract` refuses to emit a kernel whose contract is incomplete,
so an unproven kernel cannot reach production by accident.

---

## C ABI

| Symbol | Signature | Purpose |
|---|---|---|
| `rho_kernel_element_count` | `int64_t (void)` | Cells the kernel sweeps; the minimum buffer length |
| `rho_kernel_exec_with_args` | `void (const double *in, double *out)` | Run the kernel. Buffers **must** hold `element_count()` doubles |
| `rho_kernel_exec_bounded` | `void (const double *in, double *out, int64_t n)` | Same, but clamps the sweep to `n` cells |
| `rho_kernel_metadata` | `const char * (void)` | JSON: element count, every space's shape, and the proven contract |
| `rho_kernel_exec` | `void (void)` | Zero-copy: runs against the addresses compiled in via `&[0x…]` or `--bind`. Returns immediately if `INPUT` is unbound |

Passing `NULL` as the output pointer makes the kernel write in place. Passing
`NULL` as the input pointer makes it return without touching memory.

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
| Named functions and their domain checks | ✅ implemented |
| Single precision (`--f32`) | ✅ implemented — 5.5x on a bandwidth-bound kernel |
| Mixed precision, integer types | 📋 not planned — see the specification |
| Explicit `<4 x double>` vector lowering | ✅ implemented — see the note below |
| Zero-copy binding (`&[0x…]`, `--bind`) | ✅ implemented |
| `!` constraint solver | ✅ interval arithmetic; Z3 with `--features z3-solver`. Models binary64 rounding, not ℝ |
| Proven contract embedded in the artifact | ✅ implemented |
| Diagnostics with source lines | ✅ implemented |
| Reference interpreter + differential testing | ✅ implemented |
| IR proved equivalent to the source, per shape | ✅ implemented — `--features z3-solver` |
| C ABI, JSON metadata, Python FFI | ✅ implemented |
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
- **A proof assumes no overflow, underflow or NaN.** The solver models
  round-to-nearest — every result is the exact one times `(1 ± 2⁻⁵³)` — which is
  what stops it proving things that only hold over the reals. It does not model
  infinities, subnormals or NaN.

---

## Design Direction

- a compiler prototype for a clockless, spatial dataflow style
- a research-oriented implementation of topological numeric computation
- a foundation for future work in memory-aware execution, low-power scheduling,
  and domain-specific numeric kernels

The long-term aim is not merely to add syntax, but to build a runtime model that
can express and execute computation in a more memory-aware and flow-oriented way.

See [docs/SPECIFICATION.md](docs/SPECIFICATION.md) for the symbol dictionary and
[docs/ROADMAP.md](docs/ROADMAP.md) for the staged plan.

---

## License

Apache 2.0
