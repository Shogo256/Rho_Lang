# $\rho$ (RHO) Language Specification v1.0

## 1. Computational Philosophy: Hardware-Agnostic Parallelism

The $\rho$ (RHO) Language is a high-dimensional, mathematics-driven dataflow DSL. It eliminates the concept of temporal loop control (`for`, `while`, step-by-step clock cycles `CLK`) in favor of **spatial topological transformations**. 

Inspired by **Wasan (和算)** — traditional 17th-century Japanese mathematics pioneered by Seki Takakazu (関孝和) — computations are perceived as simultaneous state transformations over a memory grid rather than sequential loops.

RHO targets heterogeneous hardware (CPUs, GPUs, and accelerators) by mapping topological shifts directly onto physical hardware memory hierarchies (registers, caches, shared memory, and global memory) and execution threads.

---

## 2. The 20-Symbol Mathematical Dictionary (Fixed)

All RHO source code consists exclusively of the following 20 core mathematical symbols:

| Symbol | Name | Wasan Concept | Operational Semantics |
|---|---|---|---|
| `◯` | Space Matrix | Enri (円理) | Allocates a multi-dimensional tensor container. |
| `□` | Operator/Shape | Hojin (方陣) | Defines physical layout, dimensions, and strides. |
| `▷` | Positive Shift | Soroban Shift (右) | Element shift along the primary dimension (0-padded). |
| `▽` | Negative Shift | Soroban Shift (左) | Element shift opposite the primary dimension (0-padded). |
| `+` | Addition | Superposition | Element-wise tensor addition / superposition. |
| `-` | Subtraction | Difference | Element-wise tensor subtraction / spatial difference. |
| `×` / `*` | Multiplication | Scaling | Element-wise scaling or matrix scaling. |
| `/` | Division | Distortion | Element-wise division / distortion ratio. |
| `^` | Exponentiation | Expansion | Element-wise power scaling / dimension expansion. |
| `→` | Flow | Ruten (流転) | Asynchronous energy transfer to target space. |
| `<` / `>` | Threshold Check | Boundary Condition | Logical thresholding; maps to conditional execution. |
| `=` | Equilibrium Point | Tou (答 / 均衡) | Primary output convergence target. |
| `:` | Bind Operator | Binding | Associates identifiers with memory spaces/shapes. |
| `{` `}` | Topos Block | Universe Boundary | Encloses the computation grid/topological space. |
| `$` | Audit Trace | Sangaku (算額鑑識) | Traces DAG data-dependencies backwards for audit. |
| `&` | Zero-Copy Pointer | Direct Coupling | Binds external memory address pointers directly. |
| `!` | Constraint Solver | Invariant Check | Enforces logical constraint verification using SMT/Z3. |

---

## 3. Parallel Execution & Memory Semantics

To execute without sequential CPU clock loops, the compiler parses the AST into a Directed Acyclic Graph (DAG) and applies the following hardware mapping:

### 3.1 Space & Memory Partitioning
* **Space (`◯ □`)**: Maps to a contiguous segment of hardware memory. Large matrices are automatically partitioned into blocks (sub-tiles).
* **Zero-Copy Pointer (`&` or `@`)**: Directly maps an external memory pointer (e.g., PyTorch/NumPy buffer) to the space boundary, eliminating memory copy overhead.

### 3.2 Topological Shift Semantics (`▷`, `▽`)
* Shifts represent neighborhood operations.
* **Hardware Mapping**: Translated to registers, cache lines, or shared memory offsets within execution blocks, bypassing expensive memory round-trips.
* **CPU Mapping**: Translated to SIMD vector shift/shuffle instructions (AVX-512, NEON).

### 3.3 Flow & Convergence (`→`, `=`)
* **`→` (Flow)**: Establishes a producer-consumer boundary. Independent flows are executed in parallel across CPU cores or execution pipelines.
* **`=` (Equilibrium)**: The final output barrier. Forces synchronization across threads and flushes values to the final destination buffer.

---

## 4. Compile-Time Invariant Checking (`!`)

Logical constraints defined by `! (EXPR)` are statically evaluated before code generation. 
* Prevents runtime faults (e.g. division by zero, dimension mismatch, negative values in strict fields).
* Enables compiler optimization passes to drop unnecessary runtime bounds-checking.
