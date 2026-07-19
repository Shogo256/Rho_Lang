# $\rho$ (RHO) Language Compiler

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/shogo/rho_lang)
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

| $\rho$ Symbol | Wasan Concept (和算のモチーフ) | Spatial Semantics |
|---|---|---|
| **`◯`** | **Enri (円理 - Circle Theory)** | Memory space container maintaining continuous tensor fields. |
| **`□`** | **Hojin (方陣 - Matrix Grid)** | Physical memory boundary and spatial shape definition. |
| **`▷` / `▽`** | **Soroban Shift (算盤の滑り)** | Instantaneous vector shift across memory space without loops. |
| **`→`** | **Ruten (流転 - Flow Propagation)** | Asynchronous energy transfer between spatial domains. |
| **`=`** | **Tou (答 / 均衡 - Equilibrium)** | Final spatial output convergence point. |
| **`&`** | **Zero-Copy Coupling (直結)** | Direct zero-copy binding to host memory (NumPy / C++). |
| **`!`** | **Static Solver (数理検証)** | Mathematical invariant and static constraint verification. |
| **`$`** | **Audit Trace (算額鑑識)** | Topological DAG reverse traversal and audit visualization. |

---

## 🌌 Key Features

- ⏱️ **Clockless & Time-Eliminated**: No sequential instruction clock (`CLK`) or loops. Computation flows as continuous spatial data propagation.
- 🔣 **20-Symbol Strict Syntax**: Expressive topological operators ($\bigcirc$, $\square$, $\triangleright$, $\nabla$, $\rightarrow$, $=$, $\&$, $!$, $\$$).
- ⚡ **Zero-Copy Memory Pointer Binding**: Direct memory coupling to C++/Fortran/PyTorch/NumPy buffers without `memcpy` overhead (`&[0x7A4F]:INPUT`).
- 🛡️ **Automated TLA+ & Z3 Mathematical Verification**: Auto-generates TLA+ specifications (`rho_harmony.tla`) and statically verifies logic constraints ($!$) for mathematical validity.
- 🚀 **LLVM 22 Native Execution**: Directly maps topological vector shifts ($\triangleright$, $\nabla$) to hardware SIMD instructions and emits native shared libraries (`.so`).

---

## 💻 Syntax Example: Teichmüller Space Analysis

```rho
{
    /* Map existing tensor from external memory (Zero-Copy) */
    &[0x7A4F]:INPUT:◯ □ 1024 1024 
    
    /* Spatial gradient extraction via vector shifts */
    (▷INPUT - INPUT) → △
    (▽INPUT - INPUT) → ▽
    
    /* Calculate & verify spatial distortion ratio */
    ((△ - ▽) / (△ + ▽)) ^ 2 → OUTPUT
    ! (OUTPUT >= 0)
    
    /* Stream output exceeding threshold to Equilibrium Point */
    OUTPUT > 𝜏 → =
}
```

---

## ⚡ Quick Start

### Prerequisites

- Rust (`cargo 1.80+`)
- LLVM 22 / Clang 22

### Build Compiler

```bash
cargo build --release
```

### Run Compiler CLI (`rhoc`)

```bash
# Compile RHO source script to shared library (.so)
cargo run -- examples/teichmuller.rho

# Dump Active Audit DAG Trace ($) & generated LLVM 22 IR
cargo run -- examples/teichmuller.rho --dump-dag --dump-llvm
```

### Python Zero-Copy Integration Demo

```bash
python3 examples/python_demo.py
```

### Run Tests

```bash
cargo test
```

---

## 🏗️ Architecture Overview

```
   [ .rho Source File ]
          │
          ▼
   【 Phase 1: Rust/nom Front-End 】 ───> Invalid token check (Disruption: Glyph Error)
          │
          ▼ [ Abstract Syntax Tree (AST) ]
   【 Phase 2: DAG Transformer 】 ───> $ Trace Tree / TLA+ Translation
          │                                  │
          ├──────────────────────────────────┘
          ▼ [ Validated Multi-Dimensional DAG ]
   【 Phase 3: MLIR / LLVM 22 Back-End 】 ───> ! Constraint Check (Disruption: Dimension/Logic Error)
          │
          ▼ [ Zero-Overhead Optimization Passes ]
   【 Phase 4: Native Code Generation 】 ───> Clocks-Eliminated Binary (No CLK / Vector Ops)
```

1. **Phase 1: Nom/Rust 20-Symbol Strict Tokenizer & Boundary Mapper**  
   Validates symbols, eliminates temporal loops, and allocates tensor memory boundaries ($\bigcirc \square$).
2. **Phase 2: Flow-Centered DAG Transformer & TLA+ Verifier**  
   Transforms expressions into Directed Acyclic Graphs (DAG), extracts parallel execution nodes, and generates formal TLA+ verification models.
3. **Phase 3: Static Constraint Solver (Z3) & LLVM 22 Mapping**  
   Verifies mathematical logic ($!$) and maps tensors to zero-copy memory pointers ($\&$).
4. **Phase 4: Clockless Vector Shift Engine & Native Code Emission**  
   Lowers topological shifts ($\triangleright$, $\nabla$) directly to LLVM vector/SIMD instruction passes and emits native `.so` binaries.

---

## 📚 Documentation

- [Compiler Spec & Development Roadmap](docs/ROADMAP.md)
- [Grammar & Mathematical Foundation](docs/SPECIFICATION.md)

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
