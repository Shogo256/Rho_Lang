# $\rho$ (RHO) Language Compiler

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![LLVM Version](https://img.shields.io/badge/LLVM-22.1.8-dragon.svg)](https://llvm.org)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)

**The Time-Eliminated Topological Dataflow Language for High-Performance Scientific & Financial Computation.**

$\rho$ (RHO) is a mathematics-driven programming language and compiler prototype that explores a clockless, spatial dataflow model for numeric computation. The current implementation focuses on parsing a compact RHO syntax, validating basic flow constraints, generating LLVM IR, and emitting a native shared library.

---

## Current Status

The project is now in a working prototype state:

- Parses a compact RHO syntax with ASCII aliases
- Generates LLVM IR from simple topological flow expressions
- Emits a native shared library (`.so`)
- Includes regression tests for parsing, constraint validation, and code generation

The implementation is intentionally focused on a minimal, verifiable core rather than a full language runtime.

---

## Quick Start

The easiest way to set up the ρ (RHO) compiler development environment with all toolchains (Rust, LLVM, Clang, Python) pre-configured is using the official Docker container.

### 1. Build and Run via Docker
```bash
# Clone repository
git clone https://github.com/Escoffier1156/Rho_Lang.git
cd Rho_Lang

# Build the official Docker container
docker build -t escoffier1156/rho-lang .

# Run the interactive workspace container
docker run -it escoffier1156/rho-lang
```

### 2. Build and Test inside Container
Once inside the container, you can compile and test the compiler directly:
```bash
cargo test
cargo build --release
```

### 4. Run the Example Compiler

```bash
cargo run --release -- examples/teichmuller.rho
```

This produces a shared library named `libkernel.so`.

### 5. Run the Example Kernel from Python

```bash
python3 - <<'PY'
import ctypes
lib = ctypes.CDLL('./libkernel.so')
lib.rho_kernel_exec_with_args.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.rho_kernel_exec_with_args.restype = None

arr = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
out = (ctypes.c_double * 4)(0.0, 0.0, 0.0, 0.0)
lib.rho_kernel_exec_with_args(arr, out)
print(list(out))
PY
```

---

## Example Syntax

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

---

## Design Direction

The project is currently positioned as:

- a compiler prototype for a clockless, spatial dataflow style
- a research-oriented implementation of topological numeric computation
- a foundation for future work in memory-aware execution, low-power scheduling, and domain-specific numeric kernels

The long-term aim is not merely to add syntax, but to build a runtime model that can express and execute computation in a more memory-aware and flow-oriented way.

---

## License

Apache 2.0
