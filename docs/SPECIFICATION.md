# ρ (RHO) Language Specification

Status marks throughout: ✅ implemented and tested · 🚧 partial · 📋 designed, not built.

## 1. Computational Philosophy

ρ (RHO) is an array language in the line of APL. A program is written over
whole arrays — no indices, no loops — and each glyph is an array operation.
It drops temporal loop control (`for`, `while`, explicit clocking) in favour of
**spatial transformations**: a program says what every cell of a grid becomes,
not how to walk over it. Shapes are declared, which is what lets every program
compile to a native kernel; it also makes a program one expression per cell,
which is what the compile-time checks rest on. The checks are a property of the
design, not its purpose.

Inspired by **Wasan (和算)** — traditional Japanese mathematics pioneered by Seki
Takakazu (関孝和) — computations are expressed as simultaneous state
transformations over a memory grid rather than sequential loops.

The long-term target is heterogeneous hardware (CPUs, GPUs, accelerators),
mapping topological shifts onto the physical memory hierarchy. Today the
compiler emits explicit CPU vector IR; GPU backends are still future work.

---

## 2. The Symbol Dictionary

RHO source consists of these symbols, a handful of named functions, identifiers,
digits, and grouping punctuation.

The dictionary was once fixed at twenty. It is not any more, and pretending
otherwise would have cost more than the slogan was worth: folds, scans and the
analytic functions each earn their place, and a count is not a design
principle.

| Symbol | Name | Wasan Concept | Semantics | Status |
|---|---|---|---|---|
| `◯` | Space Matrix | Enri (円理) | Declares a tensor container | ✅ |
| `□` | Shape | Hojin (方陣) | Gives its dimensions | ✅ |
| `▷` | Positive Shift | Soroban Shift (右) | Reads the preceding cell along an axis, 0 at the edge | ✅ |
| `▽` | Negative Shift | Soroban Shift (左) | Reads the following cell along an axis, 0 at the edge | ✅ |
| `△` | Space Glyph | — | Usable as a space identifier | ✅ |
| `◇` | Fold | Dasseki (垜積) | Collapses an axis, see §3.5 | ✅ |
| `◈` | Scan | Dasseki (垜積) | A running fold; keeps the shape, see §3.5 | ✅ |
| `□` | Lift | Hojin (方陣) | In an expression, inserts a length-1 axis, see §3.6 | ✅ |
| `⍳` / `#` | Index | — | The coordinate of each cell along an axis, from zero, see §3.7 | ✅ |
| `⌽` / `%` | Rotate, Reverse | — | `k ⌽ X` reads `k` cells along, wrapping; `⌽X` reads from the other end, see §3.2.2 | ✅ |
| `⍴` / `\` | Reshape | — | `2 3 ⍴ X` reads X's cells, in order, into that shape, see §3.8 | ✅ |
| `⍉` / `'` | Transpose | — | `⍉X` reverses the axes; `1 0 ⍉ X` permutes them, see §3.9 | ✅ |
| `↑` / `^.` | Take | — | `k ↑ X` keeps the first `k` cells along an axis, the last for negative `k`, see §3.10 | ✅ |
| `↓` / `_.` | Drop | — | `k ↓ X` removes them instead, see §3.10 | ✅ |
| `+` | Addition | Superposition | Element-wise addition | ✅ |
| `-` | Subtraction | Difference | Element-wise subtraction | ✅ |
| `×` / `*` | Multiplication | Scaling | Element-wise product | ✅ |
| `/` | Division | Distortion | Element-wise division | ✅ |
| `^` | Exponentiation | Expansion | Element-wise power; a whole exponent is repeated multiplication | ✅ |
| `⌈` / `>.` | Greater | — | The greater of two, element-wise (APL's dyadic `⌈`) | ✅ |
| `⌊` / `<.` | Lesser | — | The lesser of two, element-wise (APL's dyadic `⌊`) | ✅ |
| `\|` | Residue | — | `A \| B` is B modulo A, with the sign of A; `0 \| B` is B (APL's `\|`) | ✅ |
| `→` | Flow | Ruten (流転) | One full sweep of the grid into the target | ✅ |
| `⇒` | Fixed Point | Iteration | Repeats a flow until no cell moves by more than 𝜏, or the cap is reached, see §3.3 | ✅ |
| `<` `>` | Threshold | Boundary Condition | Masking compare, see §3.3 | ✅ |
| `=` | Equilibrium | Tou (答 / 均衡) | Final output; ends the pipeline | ✅ |
| `𝜏` / `τ` | Threshold Constant | — | Scalar bound by `--tau`, default `0.0` | ✅ |
| `:` | Bind | Binding | Associates a name with a space and shape | ✅ |
| `{` `}` | Topos Block | Universe Boundary | Encloses the computation grid | ✅ |
| `$` | Audit Trace | Sangaku (算額鑑識) | Traces DAG dependencies (`--dump-dag`) | ✅ |
| `&` / `@` | Zero-Copy Pointer | Direct Coupling | Binds an external address, see §3.4 | ✅ |
| `!` | Constraint | Invariant Check | Statically verified, see §4 | ✅ |
| `→ =` | Convergence | — | Writes the caller's output buffer | ✅ |
| `name:{ … }` | Function | Jutsu (術) | A named block with parameters, expanded at each call, see §3.11 | ✅ |

ASCII aliases: `->` for `→`, `=>` for `⇒`, `>>` for `▷`, `<<` for `▽`, `>.` for `⌈`, `<.` for `⌊`, `#` for `⍳`, `%` for `⌽`, `\` for `⍴`, `'` for `⍉`, `^.` for `↑`, `_.` for `↓`, `@` for `&`,
`<>` for `◇`, `<.>` for `◈`, `[]` for `□`.

### The greater, the lesser and the residue

`A ⌈ B` and `A ⌊ B` are the greater and the lesser of two cells, so `X ⌈ 0.0`
is a ReLU and `(X ⌊ 1.0) ⌈ -1.0` a clamp. They are written as APL writes them,
and their ASCII forms `>.` and `<.` follow the fold glyphs `◇>` and `◇<`, which
mean the same thing over an axis. Both bind tighter than `+ -` and looser than
`× /`.

They are IEEE 754-2019's `maximum` and `minimum`: a NaN on either side is the
answer, as it is for every other operation, and `-0` orders below `+0`, so
`0.0 ⌈ -0.0` is `+0` and `0.0 ⌊ -0.0` is `-0`. The kernel calls the intrinsics
of that name rather than lowering a compare and a select, because the
optimiser reads a compare-and-select as a NaN-free minimum: at `-O2` it turned
`NaN < 0 ? NaN : 0` into NaN. The differential test found that within a few
hundred programs of the operators arriving.

`A | B` is APL's residue: B modulo A, computed as `B - A × ⌊B ÷ A⌋`, so the
result takes the sign of A — `3 | -7.5` is `1.5`, `-3 | 7` is `-2` — and
`0 | B` is B. The modulus is on the left, as in APL, which is what makes
`n | ⍳X` read as it should. It is computed as written, one rounding per step;
`frem` would round differently and take the sign of B.

### Named functions

| Name | Meaning | Domain |
|---|---|---|
| `exp X` | `e` to the power of each cell | — |
| `log X` | natural logarithm | the argument must be positive |
| `sqrt X` | square root | the argument must not be negative |
| `sin X`, `cos X` | trigonometric, in radians | — |
| `abs X` | magnitude | — |
| `ind X` | 1 where it holds, 0 elsewhere | — |

They are written as names rather than glyphs because they describe a *quantity*
rather than a shape. The board operations move beads; these are the analytic
functions, which Wasan also reached by a named technique — enri (円理), Seki and
Takebe's theory of series.

A space may not take one of these names, since `exp X` would then be ambiguous.
A name that merely begins with one, like `exposure`, is fine.

`ind` is how a program counts. Applied to a comparison it is that comparison's
truth; applied to anything else it asks whether the value is not zero. A mask
cannot do this: one that passes a value which happens to be zero is
indistinguishable from one that blocked it.

NaN is not zero, so `ind` of NaN is 1 — what `!=` says in C and in the
reference interpreter. The compiler once emitted an *ordered* comparison here,
which is false for NaN; differential testing found the kernel's 0 against the
interpreter's 1, and the comparison is now unordered (`fcmp une`). The masks
are unaffected: NaN fails `>`, `<`, `>=`, `<=` and `==` everywhere.

```rho
(ind (INPUT > 5.0)) → ABOVE
◇+ ABOVE → =                 /* how many cells exceed five */
```

The domains are checked the way divisions are. `log` of something the solver
cannot show is positive is reported as open:

```
[proved]   log ((INPUT ^ 2) + 1)
[unproven] log INPUT — the argument must be positive; it ranges over [-inf, +inf]
```

Their ranges also give the interval backend facts it could not otherwise have.
A sine is bounded whatever it was given, so `! ((sin X) + 2.0 > 0)` is proved
outright. And `exp X > 0` is deliberately *not* proved: `exp` underflows to
exactly zero for a large enough negative argument, so a proof there would be a
false one.

A shift may name its axis with a digit: `▷0X` shifts along axis 0, `▽1X` along
axis 1. A bare `▷X` uses the innermost axis that has more than one cell. A fold
is written the same way: `◇+1X`. Its ASCII alias is `<>`.

---

## 3. Execution & Memory Semantics

### 3.1 Space & Shape (`◯ □`) ✅

A space maps to a contiguous run of doubles. Every space in a block shares one
flat index range, whose length is the product of the primary space's dimensions
(`INPUT` if declared, otherwise the first space). `rho_kernel_element_count()`
reports the larger of that and the output's cell count, since a take past the
end, a reshape that reads round or an outer product writes more cells than it
reads, and the two-pointer entrypoint's buffers have to hold both.

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

### 3.2.2 Rotation and reversal (`⌽`) ✅

`k ⌽ X` is APL's rotate: each cell reads the cell `k` further along the axis,
wrapping at the end, so `result[i] = X[(i + k) mod n]`; a negative `k` turns
the other way, and `k` beyond the axis's length goes round. Where `▷` and `▽`
pad with zero at the edge, `⌽` wraps, which is what a periodic boundary is:

```rho
(((1 ⌽ U) + (-1 ⌽ U)) - (2.0 × U)) → LAPLACIAN      /* on a ring */
```

`⌽X` alone is APL's reverse, `result[i] = X[n - 1 - i]`. As with the shifts, a
digit after the glyph names the axis — `1 ⌽0 X`, `⌽0X` — and a bare `⌽` takes
the innermost axis with more than one cell. `%` spells it in ASCII.

The amount is a whole number written as a literal: a rotation is fixed at
compile time, like a shift's direction. `⌽` binds tighter than every arithmetic
operator, as the prefix glyphs do, so `A + 1 ⌽ X` is `A + (1 ⌽ X)`; after an
operator it is the prefix reverse. Like a shift, a turn reads a declared space,
not a computed value: flow the value into a space first.

A turn's reads are not contiguous — they wrap at the end of every row — so a
sweep that contains one stays scalar. 📋 A gathered vector path would lift
this.

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

### 3.3.1 Fixed points (`⇒`) ✅

`expr ⇒ U` is `expr → U` repeated: every sweep reads the whole of the previous
one — a Jacobi step, never a Gauss–Seidel one, so the rule that a shift sees a
finished grid holds inside the loop and the vector lowering stays safe — and
the loop stops when the largest move of any cell is at most `𝜏`, or when
`--max-iter` sweeps have run. `U` must have been written by a flow before: the
starting point is part of what an iteration computes, so the program spells it
out. The body must produce `U`'s shape. `=` is unchanged — the end of the
program — and an iteration is read from afterwards:

```rho
INPUT → X
((INPUT - (▷X + ▽X)) / 4.0) ⇒ X        /* Jacobi for 4x[i] + x[i-1] + x[i+1] = b[i] */
X → =
```

The cap is required, not defaulted, because it is what makes every kernel
terminate and it changes what the kernel computes when the loop has not
settled. A NaN move never compares greater than the largest so far, so a grid
with a NaN never settles and runs to the cap. What happened is readable
afterwards: `rho_kernel_sweeps()` is the total number of sweeps the loops of
the most recent call took, and `rho_kernel_converged()` is 1 when every loop
left on the tolerance. A loop that reaches the cap counts as not converged
even if its last sweep happened to settle — the kernel did not check. Both are
per kernel, not per thread; the language has no concurrency and neither has
this record.

Not expressible: a Gauss–Seidel sweep (in-place, order-dependent), and a loop
whose body is several flows.

### 3.5 Folds and scans (`◇`, `◈`) ✅

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

`◈` is the running form of the same operator. Where `◇+` answers "what is the
total", `◈+` answers "what is the total so far" at every cell, so it keeps the
shape it walks rather than collapsing it:

```rho
◇+ INPUT      /* [1,2,3,4] -> [10]           */
◈+ INPUT      /* [1,2,3,4] -> [1,3,6,10]     */
◈>1 INPUT     /* a running maximum along each row */
```

The two compose. A cumulative distribution is the running total over the total,
and since the fold drops an axis, `□` puts one back to line them up:

```rho
((◈+ INPUT) / (□0 (◇+ INPUT))) → OUTPUT
```

📋 Both loops are scalar: the accumulator carries a dependency across
iterations, and splitting it would reassociate the additions and change the
result. A parallel scan would need to accept that.

### 3.6 Lifting and broadcasting (`□`) ✅

`□aX` views `X` with a length-1 axis inserted at position `a`. The axis stores
nothing; it exists so an element-wise operation can stretch it.

Two operands combine when their ranks match and each axis pair is either equal
or has a 1 on one side. The result takes the longer of each pair. Requiring
equal rank keeps the rule checkable by eye, and `□` is how a shape gains the
axes it needs.

```rho
(□1A) × (□0B)      /* [4] and [3] -> [4,1] and [1,3] -> [4,3]: an outer product */
COL + ROW          /* [3,1] and [1,4] -> [3,4]                                   */
```

Together with a fold this gives a contraction, and so a matrix product:

```rho
A:◯ □ 2 3 1
B:◯ □ 1 3 4
◇+1 (A × B) → OUTPUT      /* [2,3,4] folded on axis 1 -> [2,4] */
```

The intermediate `[2,3,4]` is never materialised: the fold evaluates the product
at each index as it walks the contracted axis, so the work is exactly the
`M·N·K` multiply-adds the product requires.

A stretched read is not contiguous, so a flow that broadcasts is lowered scalar
rather than vectorised.

📋 Ranks are not promoted implicitly, and there is no reshape that moves cells
between axes — `□` only inserts axes of length 1.

### 3.6.1 Inner and outer products in general

Lifting, broadcasting, folds and scans compose, and three things follow that
need no new glyph:

- **Inner products.** `◇+1 (A × B)` over `A:◯ □ m k 1` and `B:◯ □ 1 k n` is
  the matrix product, APL's `+.×`. The fold and the operation are independent:
  `◇<1 (A + B)` is the min-plus product, whose square is a shortest-path
  relaxation step; `◇>1 (A × B)` is max-times. Any fold over any operation.
- **Outer products.** `□1A f □0B` applies `f` to every pair of a cell of A and
  a cell of B, APL's `∘.f`, for any `f` in the language.

### 3.7 Index (`⍳`) ✅

`⍳X` is the coordinate of each cell of X's shape along one axis, counted from
zero: on `◯ □ 3 4`, `⍳X` is `0 1 2 3` down every row and `⍳0X` is the row
number. A bare `⍳` takes the innermost axis with more than one cell, as `▷` and
`◇` do, and a digit names the axis. It is APL's `⍳` with `⎕IO←0`, counting from
zero because the axes are numbered from zero. `#` spells it in ASCII.

The operand is measured, never read: `⍳(A + B)` is the index of that
expression's shape and evaluates nothing. Under broadcasting the index follows
the operand's layout, so `X + ⍳Y` with `Y:◯ □ 3 1` adds each row's number to
every cell of the row.

This is what position-dependent computation is written with, and the language
had no way to write it before:

```rho
(0.5 - (0.5 × (cos ((6.283185307179586 / 1024.0) × ⍳X)))) → W   /* a Hann window */
((⍳X - 1.5) ^ 2) → D                                             /* squared distance from the centre */
((□1 X) ^ (□0 (⍳K))) → V                                          /* a Vandermonde matrix, x^j */
(n | ⍳X) → P                                                      /* a period-n pattern */
```

To the `!` check a coordinate is a value between 0 and the axis's extent less
one, so `! (⍳X >= 0)` and `! (⍳X <= 3.0)` on `◯ □ 3 4` are both settled.

### 3.8 Reshape (`⍴`) ✅

`2 3 ⍴ X` is APL's reshape: X's cells in row-major order, read into the shape
written on the left, which is a list of whole numbers and so known at compile
time like every other shape. With the same number of cells nothing moves — a
vector of twelve becomes the `3 4` matrix a fold can work on, and the vector
path stands. With fewer cells the source is read round again, `result[i] =
X[i mod n]`, as APL does, so `3 4 ⍴ P` tiles a pair across a grid; with more,
the tail is dropped. A reshape that reads round is not contiguous and keeps
its sweep scalar. `\` spells it in ASCII.

```rho
(3 4 ⍴ V) → M            /* twelve cells as three rows of four */
(◇+1 M) → ROWSUMS
```

Like a shift, it reads a declared space, not a computed value, and it binds
as tightly as the prefix glyphs. It is not a transpose: `4 3 ⍴ X` re-cuts the
same run of cells into rows of three. Transposition is `⍉`, see §3.9.

### 3.9 Transpose (`⍉`) ✅

`⍉X` is APL's transpose: the axes in reverse order, so a `3 4` matrix becomes
`4 3` with column `j` as row `j`, and a rank-3 `2 3 4` becomes `4 3 2`. With a
permutation written on the left, `P ⍉ X`, source axis `k` becomes result axis
`P[k]`: `1 0 ⍉ X` is the matrix transpose again and `0 2 1 ⍉ X` swaps the last
two axes of a rank-3 X. `result[j] = X[i]` where `i[k] = j[P[k]]`. The
permutation is checked against the operand's rank where shapes are checked,
and a list that is not a permutation is an error with a line. `'` spells it in
ASCII.

```rho
(◇+1 ((□2 A) × (□0 (⍉B)))) → C      /* A · Bᵀ, for A of m k and B of n k */
```

A transposed read is not contiguous, so a sweep containing one stays scalar
(📋 a gathered vector path); like a shift, it reads a declared space rather
than a computed value, and it binds as tightly as the prefix glyphs.

### 3.10 Take and drop (`↑`, `↓`) ✅

`k ↑ X` is APL's take: the first `k` cells along the axis, or the last `|k|`
for a negative `k`. `k ↓ X` is drop: the same cells removed. The count is a
whole number written as a literal, so the shape that results — `3 4` taken by
two is `3 2`, dropped by one is `3 3` — is known at compile time, as every
shape is; a take or drop that leaves nothing, or names an axis the operand has
not got, is an error with a line. A digit after the glyph names the axis, and
a bare glyph takes the innermost axis with more than one cell. `^.` and `_.`
spell them in ASCII, which is why a literal needs a digit before its point.

A take longer than its axis pads with zero, at the far end for a positive
count and at the near end for a negative one, as a shift pads at the edge:
`6 ↑ X` of a row of four is the row and two zeros. A drop longer than its axis
would leave nothing and is refused.

```rho
((1 ↓ X) - (-1 ↓ X)) → D      /* the forward difference, n - 1 cells, no boundary zero */
(-3 ↑ X) → TAIL                /* the last three */
```

Like a shift, a take or drop reads a declared space rather than a computed
value, and it binds as tightly as the prefix glyphs. Its reads keep their
order but the result's rows are not the source's, so the sweep stays scalar
(📋 along the outermost axis it is a plain offset and could keep the vector
path).

### 3.11 Functions (`name:{ … }`) ✅

A function is a named block with parameters, written before the program:

```rho
smooth:{ X ((▷X + X + ▽X) / 3.0) }        /* an expression body */

norm:{ V                                    /* a body of flows */
    (V × V) → SQ
    (◇+ SQ) → S
    (S ^ 0.5) → =
}

blend:{ A B W ((A × W) + (B × (1.0 - W))) }

{
    INPUT:◯ □ 256 1
    (blend (smooth INPUT) INPUT 0.5) → =
}
```

The parameters are the new names after the brace, and the body follows: one
expression, or flows ending in `→ =`, which names the result. The parameter
list ends at the first token that is not a name or repeats one, so
`id:{ X X }` is the identity; a one-line body that begins with a name nobody
has seen would be read as one more parameter, so it is parenthesised.

A call is the name followed by its arguments — names, numbers or
parenthesised expressions, one after another; an unbracketed sum has no end
the parser could find, and is refused with that hint. A call binds tighter
than the arithmetic around it: `1.0 + blend A B` is `1.0 + (blend A B)`.

**Nothing runs at call time.** The body is copied in with the arguments bound
and its locals renamed apart, so two calls to `norm` keep two `SQ`s, and a
function costs exactly what its body costs. An argument that is not a name or
a number is first flowed into a space of its own, which is what lets the body
shift it. The shapes of the parameters are the shapes of the arguments, at
each call — a function is written once and used at any shape.

A body sees its parameters, `𝜏` and constants, and nothing of the caller's;
a name from outside is an error at the definition. A body does not write to a
parameter: the argument may be the caller's own space. Definitions come before
the program and each before its use, which is what rules recursion out — a
function is not defined while its own body is being read, and neither is the
one it would call back.

An error inside a body points at both places: the body line it arose on and
the call that expanded it, each with a caret.

A call inside a `⇒` expands into the loop. What it would have flowed before
the call — a body's flows, an argument that is an expression — becomes the
loop's prelude: flows that run on every round, before the update, reading
the iterate as it stands. So a step written as flows iterates like one
written as an expression, and costs the same:

```rho
step:{ U
    (▷U + ▽U) → S
    (S / 4.0) → =
}

{
    INPUT:◯ □ 6 1
    INPUT → X
    ((INPUT / 4.0) - (step X)) ⇒ X        /* S is refilled each round */
    X → =
}
```

Not in this version: a `⇒` inside a body, an infix spelling `A blend B`, and
the rank operator `⍤`.

A note on `^`, which an index is often the exponent of: `x ^ 2.0` is `x × x`,
and `x ^ Y` is a library power even where Y's cells happen to be whole. The
rule is by spelling, and the compiler and the interpreter take it from the
same place; the interpreter once decided by value and could differ by an ulp.

### 3.4 Zero-Copy Pointer (`&`) ✅

Any space may be bound, not just `INPUT` and `OUTPUT`, which is what lets a
kernel take more than one input — a matrix product needs two. The zero-copy
entrypoint runs once every space no flow writes has an address.

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

### 3.4.1 Every space at call time ✅

The two-pointer entrypoints can only name `INPUT` and `OUTPUT`. A kernel that
reads two spaces — the matrix product above — used to need its other addresses
baked in with `--bind`, which ties one compiled kernel to one set of buffers.

`void rho_kernel_exec_spaces(void **spaces)` takes one pointer per space
instead. `spaces[i]` is the *i*-th entry of the `spaces` array in
`rho_kernel_metadata()`, which lists every space's name, shape and role:

| role | the caller | a null pointer |
|---|---|---|
| `input` — no flow writes it | supplies it | returns without touching memory |
| `output` — where `=` lands | reads the result from it | the kernel keeps the result to itself |
| `internal` — written by a flow | may supply it to observe the intermediate | the kernel uses scratch of its own |

A null table returns at once. The order is the metadata's, which is fixed by the
names alone, so the same table serves every rebuild of the same source.
`RhoEngine.execute_spaces({"A": a, "B": b, "OUTPUT": c})` builds the table from
a dictionary and checks each buffer's length against its shape first.

---

## 4. Compile-Time Checking (`!`) ✅

Enforced by the parser before code generation:

- undeclared spaces
- shape mismatches between a flow's source and target
- a missing equilibrium point (`=`)
- forbidden control-flow keywords (`for`, `while`, `if`, …)
- symbols outside the dictionary

`! (EXPR)` is then checked statically. The program is expanded so that a
constraint about one cell is inlined through every flow that produced it, leaving
only two kinds of free term: cells of spaces nobody writes — the caller's input —
and boundary flags. Both are genuinely free, so a counterexample over the
expansion corresponds to a real input. The iterate of a `⇒` is a free term too:
nothing is assumed about what a loop leaves behind.

The expansion is evaluated in interval arithmetic, which needs no dependencies
and soundly over-approximates every value. A program is rejected only when a
violation is certain; anything else is reported as unproven and the build
continues. `! (OUTPUT >= 0)` where `OUTPUT = INPUT ^ 2` is settled; a constraint
that needs two products to cancel is not, since intervals treat the two as
independent, and it is reported as open rather than refused.

Every division in the program is checked the same way. A denominator that is
always zero is a hard error; one that merely can be zero is reported:

```
[unproven] ((△ - ▽) / (△ + ▽)) — the denominator is zero when
           INPUT@-1 = -1, INPUT@0 = -1/2, INPUT@1 = -1, edge5 = true
```

That is exactly why `examples/teichmuller.rho` returns infinities on smooth
input: its denominator is the discrete Laplacian, which vanishes.

### 4.1 What "proved" means here

The kernel computes in IEEE-754 binary64, not in ℝ, and the difference is not
academic. Reasoning over the reals proves that `(x*x)/x > x` is impossible, so a
mask on that test looks like a constant zero — but on the machine the quotient
can exceed `x` by an ulp, the mask fires, and a constraint downstream breaks.
A randomised search against real runs found exactly that case.

The intervals therefore carry the standard model for round-to-nearest: every
arithmetic result is the exact result times `(1 + δ)` with `|δ| ≤ 2⁻⁵³`, and
each result is widened outward by that factor. This keeps the useful claims — a
squared value stays non-negative through rounding, and `x² + 1` stays clear of
zero — while refusing the ones that only hold over ℝ.

The claims are held to real runs. The differential test (§5) compiles random
programs, half of them carrying a `!` on their output, and compares the range
the intervals stated and any constraint they called proved with what the kernel
produced, at both widths. That is refutation rather than proof, but the failure
it catches is the one that once happened: a range narrower than the machine's
truth.

A fold is not expanded term by term — the number of terms is a property of the
grid, not of the cell a constraint speaks about — so it enters the analysis as an
unconstrained value. Proofs about a folded value stay sound; counterexamples
involving one are reported as unproven rather than as violations.

Still outside the model, and so still assumptions on any claim:

- overflow to ±∞ and underflow to subnormals
- NaN inputs and NaN propagation

---

## 5. The Reference Interpreter ✅

`src/interp.rs` evaluates a program directly, written from this specification
rather than from the code generator. It exists to be an independent second
opinion: `cargo run --bin difftest` generates random programs, shapes and
inputs — one program in three reads a second input, of `INPUT`'s shape, of one
that stretches against it, or of one an axis shorter; one intermediate in four
is iterated with `⇒`; one program in two carries a `!` — compiles each at both
widths, runs it through both C entrypoints, and reports any cell where the
kernel and the interpreter disagree on the bits.

That is a stronger check than a test suite, because neither implementation was
written to match the other. It has already found five defects:

- the interpreter treated a fold's surviving index as the start of the line that
  cell summarises, so every row after the first folded the wrong values;
- the parser split `A × -3.0` at the minus and left `A ×` behind as a name,
  because it never asked whether a sign was a sign;
- the same, one step further, for an axis index: `◈×0 -3.0`;
- `^` had no pinned meaning, so the compiler and the interpreter rounded a
  square differently;
- and `ind` of NaN was 0 from the kernel and 1 from the interpreter, because
  the compiler emitted an ordered comparison. A reader that ran the emitted IR
  without clang had hidden it by reading `one` and `une` alike — which is one
  reason that reader is gone.

### Two decisions the differential testing forced

**A whole exponent is repeated multiplication.** `x ^ 2` is `x * x`, not a call
to a maths library. Libraries do not agree with each other on the last bit of a
square, and a language that promises byte-identical output cannot inherit that.

**Floating-point contraction is off.** `clang` may fuse a multiply and an add
into a single rounding, which is more accurate but is not what the interpreter
or the `!` check model — both assume every operation rounds once. The kernels
are compiled with `-ffp-contract=off` so the machine agrees with them.

---

## 6. Precision ✅

`rhoc --f32` compiles the kernel at single precision. There is no syntax for it:
the width is a property of the artifact, not of the program, and the same source
compiles either way.

The gain is larger than halving the arithmetic suggests, because these kernels
are bound by memory rather than by the ALU. On a 1024x1024 gradient magnitude:

```
f64   12.36 ms    2.7 GB/s    8.4 MB buffers
f32    2.23 ms    7.5 GB/s    4.2 MB buffers   5.5x
```

Half the traffic is only part of it; at 4.2 MB the working set starts fitting in
cache, which is where the rest comes from.

### What it costs the check

The rounding model follows the width: `|δ| ≤ 2⁻²⁴` instead of `2⁻⁵³`. A bound
is therefore looser at f32, and the report says which width it was made at,
because a bound does not carry across:

```
f64   output ∈ [0.9999999999999999, +inf]
f32   output ∈ [0.9999999403953552, +inf]
```

That is not a defect. Single precision genuinely has less room, and a compiler
that reported the same bound for both would be hiding it.

The reference interpreter is generic over the width too, so a narrow kernel is
checked against narrow arithmetic rather than against a wide answer rounded
down — the differential testing runs both widths.

📋 The width is per kernel, not per space: there is no mixed precision, and no
integer type. Integers were considered and left out. ρ has no indexing and no
bit operations, counting is exact in floating point to 2⁵³, and adding a type
with no use in the language would have been a feature looking for a problem.
