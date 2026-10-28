# Changelog

## Unreleased

Found by a program outside ρ (an echo canceller's learning step) whose
results differed from the same arithmetic in Rust: `(W + (STEP × (□0 G)))`,
STEP a fold's one cell of shape [1] and `□0 G` of shape [1, 2048]. The
compiled kernel multiplied the first weight only and read zero past the
cell; the reference interpreter refuses the program (the ranks differ);
the JAX module stretched it, as numpy does. Three targets, three answers,
and nothing of ρ's own noticed:

- **The shape check knows the intermediates.** It ran over the flows with
  only the declared spaces' shapes, so an operand that was an intermediate
  had no shape and every pairing with it passed. The flows are now checked
  in the order they run, each target taking the shape its flow produces
  before the next is checked: a mismatch of ranks with an intermediate is
  an error with its line, as with a declared space. Lift the one cell
  (`□0 STEP`) and it stretches.
- **A flow of literals alone is refused**, and a fold or a scan of them:
  "an expression with no space in it has no shape", as the interpreter
  says when it runs one. The kernel used to compile `3.0 → T` and give T
  a shape of its own choosing.
- **The differential test counts a program the interpreter refuses as a
  finding**, where it used to skip it: the checks let it through, so a
  target would have compiled it. Against the compiler before these fixes
  it finds 71 in 300 rounds (10 of them ranks that differ, 61 literals).
  It now writes stretching through a lift (`□a X`, a space an axis short),
  and one time in four leaves the lift out, which every target must refuse
  alike. And with `DIFFTEST_JAX` it runs each program compared as the JAX
  module too, held to the interpreter within 2^10 units in the last place
  of the largest cell: XLA's numpy rules are a reading of the program of
  their own, which is how this one showed.

Six programs among ρ's own tests had shapes that did not agree, all in
tests of something else (parsing, the order of loads, kept scratch); they
now agree, and the tests check what they did.

The JAX leg's first run (seed 20260909, 120 rounds) held 85 programs to the
interpreter and found 3 that differ, of two kinds left for after this:

- **XLA divides by a constant as a multiplication by its reciprocal**: -2.5 /
  -6.0 is 0x1.aaaaaaaaaaaaap-2 under JAX, 0x1.aaaaaaaaaaaabp-2 divided. One
  ulp, which a roll (`?` hashes the bits) turns into another number, and a
  division by a cosine near zero into 4e-13 of the result.
- **A NaN in a min or max fold or scan**: the interpreter's step keeps the
  cell it has when the comparison fails, so a NaN is passed over; XLA's
  `cummin` and `cummax` carry it on. One program came out inf in the
  interpreter and NaN in JAX.

Found while timing a 321-tap FIR written as a window read by `⌷`
(`◇+1 (((⍳0N + ⍳1H) ⌷ X) × H)` over 160 samples), which took 95 µs where a
sliding dot product in plain Rust took 11:

- **A fold walks several lines at once.** The lines of a fold are
  independent, and one line's chain of additions was what bounded its
  speed. The kernel now folds W lines together (four at f64, eight at
  f32), each with its own accumulator, in the same step order per line —
  so the bits are unchanged — with W chains in flight instead of one.
  Scans too; lines past the last whole group are walked one at a time.
- **A `⌷` at a whole-number combination of coordinates is a plain read.**
  `⍳0N + ⍳1H`, `2 × ⍳0X - 1`: the position is an integer the interpreter
  computes exactly in f64 and floors to itself, so it is computed in
  integers, and its range is known from the axes' extents, so when it
  cannot leave the space the read carries no bounds test and no select.
  When it can, the test is an integer compare. A position the data
  decides (`I ⌷ X`) is still a gather.
- **A fold's line knows its coordinates.** Inside the fold's innermost
  loop every read used to divide the cell's index back into coordinates
  (a `udiv` and `urem` per axis per read); the loops that reach the cell
  now hand its coordinates down, and the outermost coordinate of each run
  carries no remainder, which is what lets the lanes' reads be seen as
  consecutive.

The FIR: 95 → 13 µs (the plain-Rust sliding dot product: 10–11). The
differential test generates the scaled and moved positions as well.

- **No dependencies.** `rhoc` and the library are built from this
  repository and Rust's standard library alone: the command line is read
  by hand in place of clap (the same flags, `--flag VALUE` and
  `--flag=VALUE`, `--help`, `--version`), errors are plain `Box<dyn Error>`
  and a hand-written `Display` in place of anyhow and thiserror, the audit
  DAG is a small graph of its own in place of petgraph (its trace in the
  same order), kernels are loaded with dlopen (`rho_lang::dl`) in place of
  libloading, and nom, declared but unused, is gone. Every kernel source
  of Basis compiles to the same bytes as before.

Found while writing the arithmetic of ML-KEM in ρ (the Rhodium project):
seven NTT layers, a matrix-vector product mod q, and the tools that work
needs.

- **The analysis no longer expands a deep chain whole.** A definition whose
  expansion passes 2^14 nodes is read as a free value named for it and the
  cell; seven NTT layers had run `rhoc` out of memory.
- **A row stretched over rows keeps the vector path.** A stretched read
  whose shared innermost axes span a multiple of the vector width now gets
  vector loads, the body aligned to the width; such flows had run one cell
  at a time (an NTT layer's arithmetic: 9.3 → 2.4 us).
- **A fold is not copied into a reader that names it twice.** Each copy had
  run the fold again as a pass of its own (a matrix-vector product: 4.7 ms
  → 82 us, 60 s → 3 s to compile).
- **`ind` of a comparison is its truth in the circuit and in JAX**, as it
  was in the interpreter and the kernel; both had read the comparison's
  mask, which is 0 where it passed a zero. A formal proof of an ML-KEM
  butterfly found it; the integer mode of the differential test now makes
  such comparisons often.
- **`rho_kernel_zeroize`** overwrites the scratch kept between calls and the
  stack an entrypoint's frame used.
- **`rhoc --constant-time`** (with `--public` for tables) refuses division
  and residue by a non-constant, `sqrt` `exp` `log` `sin` `cos` and
  fractional powers, `⇒`, and `⌷` at positions not made of `⍳`, constants
  and public spaces.
- **`DIFFTEST_MODQ=1`**: random integer programs mod q, the kernel against
  the interpreter, 32-bit fixed point against binary64, the constant-time
  rules, and with `DIFFTEST_SV=1` the circuit at `--fixed 32.0`.
- A fixed-point circuit's header says its cells are fixed point.
- **`rhoc --explain`** prints what each flow became: the sweeps a call runs,
  vector or one cell at a time and why, the folds computed as passes of their
  own, which intermediates fused into which sweep, what a `⇒` runs every
  round, and what `rho_kernel_zeroize` overwrites; with `--emit-sv`, the
  circuit's stages, its products (of two variables, by a constant, by one
  bit) and its replay memories. Nothing it prints changes the code.
- **A part's loads come in slot order.** The pointers an outlined sweep loads
  at its entry followed the addresses of the fold nodes they came from, so a
  build that allocated differently could emit other IR for the same source.
- **`--explain` counts a part a circuit's text repeats once.** A gather
  writes its place three times and Yosys merges the copies (a gather at
  `X × 3` maps to two DSP blocks, one product); a matrix-vector product's
  circuit had been said to have 35 products by a constant, where it has 33.
- **`rhoc --emit-sv --fuse`** fuses the circuit as the kernel is fused:
  each intermediate that one flow reads cell for cell is computed inside
  that flow's stage, and a part the stage reads more than once is computed
  once, into a local. Seven NTT layers over three polynomials go from 63
  stages and 21 replay memories to 14 and 14; on an ECP5, from 17,211 LUT4,
  9,815 registers and 98 block RAMs to 14,242, 3,795 and 49, the 112 DSP
  blocks unchanged. The clock falls from 21.9 to 7.3 MHz, a stage now
  computing a whole layer's arithmetic in one clock, so fusing is asked for
  until stages are pipelined. Without `--fuse` the circuit is the same,
  byte for byte.
- **The differential test runs the fused circuit too** (`DIFFTEST_SV=1`,
  in both modes), whenever fusing changes the circuit.
- **Two threads may call a kernel at once.** An entrypoint stored the
  spaces' pointers in a table of the library's own, from which its sweeps
  read them, and kept its parts' partial results and its record of passed
  spaces there too; two calls at once read each other's (eight threads on
  one iterating kernel: 988 of 1,600 calls wrong). The table and the rest
  are now on the call's stack and handed down; a sweep split across the
  pool takes the pool for itself until it is done. The metadata says
  `"reentrant"`, true unless the kernel keeps scratch between calls.
- **A fused space takes no memory unless passed.** Every entrypoint gave
  every intermediate a buffer, fused ones included, which are computed
  inside their reader and never stored: ML-KEM's matrix-vector product for
  k = 4 used 2.1 MB of stack, more than a thread is given by default, and
  uses 222 KB.
- **A single-precision kernel runs eight lanes to a vector step**, where
  it ran four, as a double-precision one does: a step is one 32-byte AVX
  register either way. Rhodium's ML-KEM NTT, written to stay exact in
  single precision, takes 17.0 us against 27.5 in double (same machine,
  same conditions). Double-precision kernels are compiled as before, byte
  for byte.
- **`rhoc --emit-lean FILE.lean`** writes a source's functions as Lean 4
  definitions over the integers, for proofs about them, and compiles
  nothing; a file of definitions alone is accepted. The subset is the
  arithmetic of whole numbers a kernel computes exactly: sums, products,
  `⌊` of a quotient by a positive whole constant (Lean's `/`), `ind` of a
  comparison, `abs`, the greater and the lesser, calls. A function outside
  it is named in a comment, with the reason, and left out. Rhodium's proofs
  are now of what this writes from its kernels' sources, not of a
  transcription by hand. A program block whose flows are all in the subset
  becomes `kernel` too: one cell of OUTPUT from the same cell of each space
  no flow writes, a lifted space read as it is; `--lean-program-only`
  writes it alone, below the namespace its functions are in.
- **A flow that reads the space it writes reads it as it was.** The
  interpreter, which is the meaning, computes a flow whole and then stores
  it. The kernel wrote the target as it swept it, so a flow that read its
  target at another cell (shifted, rotated, reversed, gathered, stretched)
  read cells already overwritten, differently on one thread and on several:
  `(▷Y + Y) → Y` gave another Y. Such a flow now reads a copy of its target
  taken before the sweep. So does a flow into OUTPUT that reads INPUT at
  another cell, when the two are one buffer at run time (the two-pointer
  form with a null output, an unbound OUTPUT beside a bound INPUT): the
  pointers are compared and the copy taken only then. A flow that reads its
  target only at the cell it writes, and every other, is compiled as before,
  byte for byte. difftest now writes an intermediate again from itself, one
  in four; on the old compiler its first seed found four disagreements in
  fifteen programs.
- **A gather runs a vector at a time.** A flow with `⌷` in it ran one cell
  at a time, the whole flow, though only the read is out of order. Its
  positions are now computed four lanes at once like anything else, each
  lane loads its own cell, and the rest of the flow stays vector: Rhodium's
  NTT (seven of its fourteen sweeps gather a partner) went from 30 to 24 us,
  its inverse from 29.5 to 22. The positions are
  converted to 32-bit integers, which x86-64 does four at once without
  AVX-512.
- **`rhoc --emit-static LIB.a --symbol-prefix P`** writes the kernel as a
  static library, its entrypoints `P_exec_spaces`, `P_metadata`, ... and
  its runtime `P_rt_*`: every kernel exports the same names, so a program
  could load several only one shared library at a time, by dlopen. Rhodium's
  Rust crate links its 86 kernels this way.
- **A 64-bit fixed format can be used.** The circuit emitter took
  `--fixed 64.F`, but the fixed-point interpreter's `Fixed<64, F>` could not
  be instantiated: the ends of its range, computed in i64, overflowed. They
  are computed in i128. Rhodium's ML-DSA, whose products reach 2^46, is held
  to `Fixed<64, 0>` this way.
- **A roll with no fraction bits is 0** in the interpreter, as in the
  circuit: `?` in a format like `32.0` shifted its hash by 64, a panic in a
  debug build and the whole hash in a release one.

## 1.0.0 — 2026-09-14

The first release: the language, three things a program becomes, each held
to the reference interpreter, and the wiki that describes them.

- **Language.** Spaces and shapes, flows, zero-padded shifts along any axis,
  folds and scans, lifts and broadcasts (outer products, matrix products), the
  fixed point `⇒` with a sweep cap, the interval check `!`, coordinates `⍳`,
  the greater and lesser `⌈ ⌊`, the residue `|`, the turns `⌽ ⍴ ⍉ ↑ ↓`, the
  gather `⌷`, the roll `?`, floor and ceiling, and functions with expression
  or flow bodies (a body of flows inside `⇒` runs every round), called prefix
  or, for two arguments, between them.
- **Kernel.** LLVM to a shared library with a C ABI: two-pointer, bounded and
  every-space entrypoints, metadata, sweeps and convergence. Four-lane
  vectors with per-lane boundaries, sweeps split across a thread pool
  (bit-identical to one thread), intermediates fused into their one reader,
  scratch kept between calls, one sweep per `⇒` round.
- **Python.** `RhoEngine`, `Kernel` (compiled and cached per shape),
  `@rho.compile`; wheels built for Linux, macOS and Windows.
- **Command line.** `rhoc` compiles; `--run` and `--write` feed files in and
  out; `--emit-jax` writes a JAX module; `--emit-sv` a SystemVerilog
  pipeline, `--fixed W.F` in fixed point, which Yosys synthesises.
- **Checks.** A reference interpreter generic over the number type; a
  differential test that compiles random programs at f64 and f32, with every
  sweep split, and compares them bit for bit — and holds the interval check's
  claims to the runs; a circuit leg through Verilator; 158 tests.

Numbers are in the README and the wiki and were measured, not estimated.
