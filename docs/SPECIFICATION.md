# ρ (RHO) Language Specification

Status marks throughout: ✅ implemented and tested · 🚧 partial · 📋 designed, not built.

## 1. Computational Philosophy

ρ (RHO) is a mathematics-driven dataflow DSL. It drops temporal loop control
(`for`, `while`, explicit clocking) in favour of **spatial transformations**: a
program says what every cell of a grid becomes, not how to walk over it.

Inspired by **Wasan (和算)** — traditional Japanese mathematics pioneered by Seki
Takakazu (関孝和) — computations are expressed as simultaneous state
transformations over a memory grid rather than sequential loops.

The long-term target is heterogeneous hardware (CPUs, GPUs, accelerators),
mapping topological shifts onto the physical memory hierarchy. Today the
compiler emits explicit CPU vector IR; GPU backends are still future work.

---

## 2. The 20-Symbol Dictionary

RHO source consists of these symbols plus identifiers, digits, and grouping
punctuation.

| Symbol | Name | Wasan Concept | Semantics | Status |
|---|---|---|---|---|
| `◯` | Space Matrix | Enri (円理) | Declares a tensor container | ✅ |
| `□` | Shape | Hojin (方陣) | Gives its dimensions | ✅ |
| `▷` | Positive Shift | Soroban Shift (右) | Reads the preceding cell along an axis, 0 at the edge | ✅ |
| `▽` | Negative Shift | Soroban Shift (左) | Reads the following cell along an axis, 0 at the edge | ✅ |
| `△` | Space Glyph | — | Usable as a space identifier | ✅ |
| `◇` | Fold | Dasseki (垜積) | Collapses an axis, see §3.5 | ✅ |
| `+` | Addition | Superposition | Element-wise addition | ✅ |
| `-` | Subtraction | Difference | Element-wise subtraction | ✅ |
| `×` / `*` | Multiplication | Scaling | Element-wise product | ✅ |
| `/` | Division | Distortion | Element-wise division | ✅ |
| `^` | Exponentiation | Expansion | Element-wise power (`llvm.pow.f64`) | ✅ |
| `→` | Flow | Ruten (流転) | One full sweep of the grid into the target | ✅ |
| `<` `>` | Threshold | Boundary Condition | Masking compare, see §3.3 | ✅ |
| `=` | Equilibrium | Tou (答 / 均衡) | Final output; ends the pipeline | ✅ |
| `𝜏` / `τ` | Threshold Constant | — | Scalar bound by `--tau`, default `0.0` | ✅ |
| `:` | Bind | Binding | Associates a name with a space and shape | ✅ |
| `{` `}` | Topos Block | Universe Boundary | Encloses the computation grid | ✅ |
| `$` | Audit Trace | Sangaku (算額鑑識) | Traces DAG dependencies (`--dump-dag`) | ✅ |
| `&` / `@` | Zero-Copy Pointer | Direct Coupling | Binds an external address, see §3.4 | ✅ |
| `!` | Constraint | Invariant Check | Statically verified, see §4 | ✅ |
| `→ =` | Convergence | — | Writes the caller's output buffer | ✅ |

ASCII aliases: `->` or `=>` for `→`, `>>` for `▷`, `<<` for `▽`, `@` for `&`.

A shift may name its axis with a digit: `▷0X` shifts along axis 0, `▽1X` along
axis 1. A bare `▷X` uses the innermost axis that has more than one cell. A fold
is written the same way: `◇+1X`. Its ASCII alias is `<>`.

---

## 3. Execution & Memory Semantics

### 3.1 Space & Shape (`◯ □`) ✅

A space maps to a contiguous run of doubles. Every space in a block shares one
flat index range, whose length is the product of the primary space's dimensions
(`INPUT` if declared, otherwise the first space). `rho_kernel_element_count()`
reports it.

Dimensions drive indexing. For a row-major shape `[d0, .., dk]`, axis `a` has
extent `d_a` and stride `product(d_{a+1..k})`, so `◯ □ 1024 1024` and
`◯ □ 1048576 1` traverse the same cells but give shifts different neighbours.

📋 Tiling and cache blocking are future work; a sweep is still one linear pass.

### 3.2 Shifts (`▷`, `▽`) ✅

A shift reads the neighbouring cell along one axis and yields 0 at that axis's
boundary. On a `3 4` grid, `▷X` stops at the start of each row rather than
wrapping into the previous one; `▷0X` reads the row above instead.

The operand must be a declared space — `▷(A + B)` is rejected, because a shift
needs storage to read a neighbour from. Flow the sub-expression into its own
space first.

Both directions are exact at the edges: the out-of-range index is clamped before
the address is formed, so no read ever leaves the buffer.

### 3.2.1 Vector lowering ✅

A sweep of known length is split into a scalar head, a `<4 x double>` vector
body and a scalar tail. The head and tail cover exactly the cells whose
neighbours would fall outside the buffer, so every vector load in the body is in
bounds. Boundary tests are evaluated per lane; the neighbour window itself is one
contiguous load at a shifted base.

`--no-simd` forces the scalar path. The two are verified to agree bit for bit.

What this buys, measured rather than assumed: on a shift kernel `clang -O3`
vectorises **no** loops on its own (the boundary select defeats it), and explicit
lowering roughly doubles the packed-double instructions in the object file. Wall
clock improves only 1.0–1.1x on large grids, because streaming megabytes of
doubles is bound by memory bandwidth, not arithmetic. The gain here is that
vectorisation is guaranteed and visible in the IR, not that it is fast.

📋 AVX-512 and NEON widths, and GPU warp shuffles, are still future work; the
width is fixed at four lanes and clang widens further if the target allows.

### 3.3 Flow & Convergence (`→`, `=`) ✅

Each `→` is a complete, ordered sweep of the grid: every cell of the target is
written before the next flow begins. This is what makes a shift well-defined —
it always reads the previous flow's finished result, never a half-written buffer.

`=` marks the final output and ends the pipeline; statements after it are not
lowered.

A comparison masks rather than yielding a boolean: `A > B` produces `A` where the
predicate holds and `0.0` elsewhere. The same applies to `<`, `>=`, `<=`, `==`.

📋 Flows are executed in source order on one thread. Extracting independent flows
to run in parallel is future work.

### 3.5 Folds (`◇`) ✅

`◇opX` folds the cells along one axis and removes it, so a flow's target is
smaller than its source. This is the only construct in the language that changes
a shape.

```rho
◇+ INPUT              /* sum along the innermost axis with more than one cell */
◇+1 INPUT             /* [3,4] -> [3]: the sum of each row                    */
◇>0 INPUT             /* [3,4] -> [4]: the largest value in each column       */
◇+ (A × B)            /* a dot product: the operand may be computed           */
◇+0 (◇+1 INPUT)       /* the whole grid, folded twice                         */
```

The operator names the fold and must be associative: `+`, `×`, `>` (maximum) and
`<` (minimum). `◇-` and `◇/` are rejected, since their result would depend on
the traversal order. An empty axis yields the operator's identity — `0`, `1`,
`-∞`, `+∞` respectively.

Folds are lowered before the sweep that reads them, into their own buffer. That
keeps the sweep body straight-line and vectorised, and it means a nested fold's
result exists before the fold that consumes it.

📋 A fold's own loop is scalar: the accumulator carries a dependency across
iterations, and splitting it would change the order of the additions and so the
result. `◇` also does not yet have a scan (running total) counterpart.

### 3.4 Zero-Copy Pointer (`&`) ✅

`&[0xADDR]:NAME:◯ □ ...` binds a space to an address. `rhoc --bind NAME=0x...`
overrides the literal, so a host can compile a kernel against the address of a
buffer it already owns and then call `rho_kernel_exec()` with no arguments at
all — no pointer marshalling, no copy:

```python
buf = (ctypes.c_double * 8)(...)
engine.compile_rho_file("k.rho", bind={"INPUT": buf})
engine.execute_kernel(buf)          # writes straight into buf
```

An unbound `OUTPUT` means the grid is transformed in place. A kernel whose
`INPUT` is unbound refuses to run through this entrypoint rather than reading an
address this process does not own.

`rho_kernel_exec_with_args` and `rho_kernel_exec_bounded` take caller-supplied
pointers instead, which suits buffers whose address is not known at compile time.

---

## 4. Compile-Time Checking (`!`) ✅

Enforced by the parser before code generation:

- undeclared spaces
- shape mismatches between a flow's source and target
- a missing equilibrium point (`=`)
- forbidden control-flow keywords (`for`, `while`, `if`, …)
- symbols outside the dictionary

`! (EXPR)` is then verified statically. The program is expanded so that a
constraint about one cell is inlined through every flow that produced it, leaving
only two kinds of free term: cells of spaces nobody writes — the caller's input —
and boundary flags. Both are genuinely free, so a counterexample over the
expansion corresponds to a real input.

Two backends answer the same question:

| Backend | Build | Strength |
|---|---|---|
| Interval arithmetic | default | No dependencies; sound over-approximation |
| Z3 | `--features z3-solver` | Exact; reports a concrete counterexample |

Both only reject a program when a violation is certain; anything else is reported
as unproven and the build continues. For example `! (OUTPUT >= 0)` where
`OUTPUT = INPUT ^ 2` is **proved** by either backend, while
`(INPUT × INPUT) - (INPUT × INPUT) >= 0` needs Z3 — intervals cannot see that the
two products cancel.

Every division in the program is checked the same way. A denominator that is
always zero is a hard error; one that merely can be zero is reported:

```
[unproven] ((△ - ▽) / (△ + ▽)) — the denominator is zero when
           INPUT@-1 = -1, INPUT@0 = -1/2, INPUT@1 = -1, edge5 = true
```

That is exactly why `examples/teichmuller.rho` returns infinities on smooth
input: its denominator is the discrete Laplacian, which vanishes.

### 4.1 What "proved" means

The kernel computes in IEEE-754 binary64, not in ℝ, and the difference is not
academic. Reasoning over the reals proves that `(x*x)/x > x` is impossible, so a
mask on that test looks like a constant zero — but on the machine the quotient
can exceed `x` by an ulp, the mask fires, and a constraint downstream breaks.
A randomised search against real runs found exactly that case.

Both backends therefore carry the standard model for round-to-nearest: every
arithmetic result is the exact result times `(1 + δ)` with `|δ| ≤ 2⁻⁵³`. The
interval backend widens each result outward by that factor; the SMT backend
introduces a bounded `δ` per operation. This keeps the useful proofs — a squared
value stays non-negative through rounding, and `x² + 1` stays clear of zero —
while refusing the ones that only hold over ℝ.

A fold is not expanded term by term — the number of terms is a property of the
grid, not of the cell a constraint speaks about — so it enters the analysis as an
unconstrained value. Proofs about a folded value stay sound; counterexamples
involving one are reported as unproven rather than as violations.

Still outside the model, and so still assumptions on any proof:

- overflow to ±∞ and underflow to subnormals
- NaN inputs and NaN propagation
- the order in which `clang` contracts operations (e.g. into an FMA)


---

## 5. The Kernel's Contract ✅

A proof that only appears in a build log cannot be relied on by whoever loads the
kernel. Everything the solver establishes is therefore compiled into the artifact
and returned by `rho_kernel_metadata()`:

```json
"contract": {
  "backend": "z3",
  "output_range": [0, null],
  "divisions_proven_safe": true,
  "output_proven_finite": false,
  "open_obligations": 0,
  "assumes": ["no overflow to infinity", "no underflow to subnormals",
              "no NaN input", "no operation contraction, e.g. into an FMA"]
}
```

`output_range` bounds every cell the kernel writes; a `null` end means that side
is not bounded. It is always computed by interval arithmetic, which is cheap and
sound — asking an SMT solver for a range needs an optimiser rather than a
decision procedure.

`assumes` is not decoration. It states what the claims rest on, so a reader can
tell a proof from a proof-under-conditions.

`rhoc --require-contract` refuses to emit a kernel with an incomplete contract,
and `RhoEngine.require_contract()` refuses to load one. Together they let a
system draw a line that unproven kernels cannot cross.
