# $\rho$ (RHO) Language Specification v1.0

## 1. Overview & Computational Philosophy: Inspired by *Wasan* (和算)

The $\rho$ (RHO) Language is a high-dimensional mathematics-driven programming environment that eliminates temporal loop constructs (`for`, `while`, sequential clock cycles `CLK`). Computations in $\rho$ are modeled strictly as topological spatial transformations, vector shifts, and energy propagation within a bounded computational universe (`{ }`).

### The Wasan Philosophical Foundation
Unlike Western programming paradigms rooted in sequential temporal execution (iterative loop control), $\rho$ is fundamentally inspired by **Wasan (和算)** — traditional 17th-century Japanese mathematics pioneered by Seki Takakazu (関孝和). In Wasan, mathematical problems were solved by perceiving spatial geometry (*Enri* / 円理), matrix grids (*Sangaku* / 算額), and sliding beads on a counting board (*Soroban* / 算盤) as simultaneous spatial transformations rather than step-by-step loops.

---

## 2. The 20-Symbol Mathematical Dictionary

All valid $\rho$ source code consists exclusively of the following 20 core mathematical symbols:

| Symbol | Name | Wasan Concept (和算モチーフ) | Topological Operation / Semantics |
|---|---|---|---|
| `◯` | Space Matrix | Enri (円理 - Circle Theory) | Container maintaining multi-dimensional tensor data. |
| `□` | Operator/Shape | Hojin (方陣 - Matrix Grid) | Defines physical memory boundary and spatial shape. |
| `▷` | Positive Shift | Soroban Shift (算盤の右滑り) | Rightward vector shift across memory space (0 padding). |
| `▽` | Negative Shift | Soroban Shift (算盤の左滑り) | Leftward vector shift across memory space (0 padding). |
| `+` | Addition | Superposition | Spatial superposition of cell values. |
| `-` | Subtraction | Spatial Difference | Spatial difference extraction / Gradient calculation. |
| `×` / `*` | Multiplication | Scaling | Scalar or tensor coefficient scaling. |
| `/` | Division | Distortion Ratio | Spatial distortion ratio computation. |
| `^` | Exponentiation | Dimension Expansion | Dimensional expansion / Power scaling. |
| `→` | Flow / Stream | Ruten (流転 - Energy Flow) | Asynchronous energy transfer from source to target space. |
| `<` / `>` | Threshold Check | Boundary Condition | Spatial boundary condition logic evaluation. |
| `=` | Equilibrium Point | Tou (答 / 均衡 - Equilibrium) | Final convergence output target point. |
| `:` | Bind Operator | Binding | Binds identifiers to memory shapes/types. |
| `{` `}` | Topos Block | Computational Boundary | Computational universe memory boundary. |
| `$` | Audit Trace | Sangaku Inspection (算額鑑識) | Topological DAG reverse traversal & graph tracer. |
| `&` | Zero-Copy Pointer | Direct Coupling (直結) | Direct pointer binding to external memory buffers. |
| `!` | Constraint Solver | Mathematical Invariant (数理検証) | Static logical invariant solver and assertion validator. |

---

## 3. Disruption Types (Compile-Time Errors)

$\rho$ enforces strict mathematical harmony. Deviations trigger immediate compilation termination with the following error classifications:

| Disruption Type | Trigger Cause | Resolution |
|---|---|---|
| **Glyph Error** (`GlyphErr`) | Tokens outside the 20 valid symbols (e.g. `for`) | Purify source code to 20-symbol syntax. |
| **Space Error** (`SpaceErr`) | Operations on undeclared space names | Initialize space boundaries using `◯ □`. |
| **Dimension Error** (`DimensionErr`) | Mismatched matrix shapes | Validate matrix dimensions via `!` operator. |
| **Flow Error** (`FlowErr`) | Missing final equilibrium output (`=`) | Add `=` target flow assignment. |
| **Logic Error** (`LogicErr`) | Static constraint assertion failure | Verify mathematical expressions in `!` solver. |
