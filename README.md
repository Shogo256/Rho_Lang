# $\rho$ (RHO) Language Compiler

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Escoffier1156/Rho_Lang)
[![LLVM Version](https://img.shields.io/badge/LLVM-22.1.8-dragon.svg)](https://llvm.org)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)

**The Time-Eliminated Topological Dataflow Language for High-Performance Scientific & Financial Computation.**

$\rho$ (RHO) is a mathematics-driven programming language that completely eliminates the concept of temporal loops (`for`, `while`) and computes exclusively through spatial topological transformations, differential shifts, and tensor dataflow propagation.

---

## 🌸 Philosophy: Inspired by *Wasan* (和算)

Unlike Western programming paradigms rooted in sequential temporal execution (the clock cycle `CLK` and loop iterations `for`/`while`), $\rho$ is fundamentally inspired by **Wasan (和算)** — traditional 17th-century Japanese mathematics pioneered by Seki Takakazu (関孝和).

In Wasan, mathematical problems were solved not by step-by-step temporal loops, but by perceiving spatial geometry (*Enri* / 円理), matrix grids (*Sangaku* / 算額), and sliding beads on a counting board (*Soroban* / 算盤) as simultaneous spatial transformations.

```
   Traditional Western Loop (`for`)           ρ (RHO) Wasan Topological Dataflow
 ┌──────────────────────────────────┐        ┌──────────────────────────────────┐
 │ Step 1 -> Step 2 -> Step 3 (CLK) │   vs   │ Spatial Superposition & Flow     │
 │ Sequential Iterative Execution   │        │ Clocks-Eliminated SIMD Shifts (▷)│
 └──────────────────────────────────┘        └──────────────────────────────────┘
```

### 🔣 The Wasan Topological Symbol Mapping

| $\rho$ Symbol | ASCII Alias | Wasan Concept (和算のモチーフ) | Spatial Semantics |
|---|---|---|---|
| **`◯`** | `matrix` | **Enri (円理 - Circle Theory)** | Memory space container maintaining continuous tensor fields. |
| **`□`** | `shape` | **Hojin (方陣 - Matrix Grid)** | Physical memory boundary and spatial shape definition. |
| **`▷`** | `>>` | **Soroban Shift (算盤の右滑り)** | Instantaneous vector shift right across memory space (0 padding). |
| **`▽`** | `<<` | **Soroban Shift (算盤の左滑り)** | Instantaneous vector shift left across memory space (0 padding). |
| **`→`** | `->` / `=>` | **Ruten (流転 - Flow Propagation)** | Asynchronous energy transfer between spatial domains. |
| **`=`** | `==` | **Tou (答 / 均衡 - Equilibrium)** | Final spatial output convergence point. |
| **`&`** | `@` | **Zero-Copy Coupling (直結)** | Direct zero-copy binding to host memory (NumPy / C++). |
| **`!`** | `assert` | **Static Solver (数理検証)** | Mathematical invariant and static constraint verification. |
| **`$`** | `trace` | **Audit Trace (算額鑑識)** | Topological DAG reverse traversal and audit visualization. |

---

## 🌌 Key Features

- ⏱️ **Clockless & Time-Eliminated**: No sequential instruction clock (`CLK`) or loops. Computation flows as continuous spatial data propagation.
- 🔣 **20-Symbol Strict Syntax with ASCII Fallbacks**: Expressive topological operators ($\bigcirc$, $\square$, $\triangleright$, $\nabla$, $\rightarrow$, $=$, $\&$, $!$, $\$$) with full ASCII fallback aliases (`>>`, `<<`, `->`, `@`).
- ⚡ **Zero-Copy Memory Pointer Binding**: Direct memory coupling to C++/Fortran/PyTorch/NumPy buffers without `memcpy` overhead (`&[0x7A4F]:INPUT` or `@[0x7A4F]:INPUT`).
- 🛡️ **Automated TLA+ & Z3 Mathematical Verification**: Auto-generates TLA+ specifications (`rho_harmony.tla`) and statically verifies logic constraints ($!$) for mathematical validity.
- 🚀 **LLVM 22 Native Execution**: Directly maps topological vector shifts ($\triangleright$, $\nabla$) to hardware SIMD instructions and emits native shared libraries (`.so`).

---

## 💻 Syntax Example: Teichmüller Space Analysis (Unicode & ASCII Support)

```rho
{
    /* Map existing tensor from external memory (Zero-Copy) using ASCII or Unicode */
    @[0x7A4F]:INPUT:◯ □ 1024 1024 
    
    /* Spatial gradient extraction via vector shifts */
    (>>INPUT - INPUT) -> △
    (<<INPUT - INPUT) -> ▽
    
    /* Calculate & verify spatial distortion ratio */
    ((△ - ▽) / (△ + ▽)) ^ 2 -> OUTPUT
    ! (OUTPUT >= 0)
    
    /* Stream output exceeding threshold to Equilibrium Point */
    OUTPUT > 𝜏 -> =
}
```

---

## ⚡ Quick Start & Installation

### 1. Clone Repository

```bash
git clone https://github.com/Escoffier1156/Rho_Lang.git
cd Rho_Lang
```

### 2. Prerequisites

- Rust (`cargo 1.80+`)
- LLVM 22 / Clang 22

### 3. Build & Run Compiler CLI (`rhoc`)

```bash
# Build compiler
cargo build --release

# Compile RHO script to native shared library (.so)
cargo run -- examples/teichmuller.rho

# Dump Active Audit DAG Trace ($) & generated LLVM 22 IR
cargo run -- examples/teichmuller.rho --dump-dag --dump-llvm
```

### 4. Advanced Python FFI Pointer Integration & Benchmark

```bash
# Run advanced Python FFI pointer coupling
python3 examples/python_demo.py

# Run performance benchmark suite (Pure Python Loop vs ρ Engine)
python3 benches/benchmark.py
```

### 5. Docker Container Setup

```bash
docker build -t rho-lang .
docker run --rm rho-lang
```

---

## 🧩 VS Code Extension Support

To enable 20-symbol syntax highlighting and autocomplete snippets (`\shiftright` -> `▷`, `\space` -> `◯ □`):
1. Copy or link the `editors/code/` folder into your VS Code extensions directory (`~/.vscode/extensions/rho-lang-vscode`).
2. Open any `.rho` file in VS Code.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
