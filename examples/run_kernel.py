"""A kernel at any shape, from Python.

`Kernel` takes the flows without declarations; each named argument becomes a
space with the argument's shape, and each shape is compiled once and kept in
a cache directory, so a second call — or a second process — with the same
shape does not compile. Nested lists are enough; numpy arrays work when numpy
is installed.

    python3 examples/run_kernel.py
"""

import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from rho import Kernel


def main():
    print("=====================================================")
    print("  ρ (RHO) Language - One Kernel, Any Shape")
    print("=====================================================")
    cache = os.path.join(tempfile.gettempdir(), "rho-lang-example-cache")

    # A five-point mean, written once.
    blur = Kernel(
        "((▷0INPUT + ▽0INPUT + ▷1INPUT + ▽1INPUT + INPUT) / 5.0) → =",
        cache_dir=cache,
    )

    small = [[float(i * j % 7) for j in range(8)] for i in range(6)]
    out = blur(INPUT=small)
    # By hand at an interior cell, with zero past the edges.
    i, j = 2, 3
    expect = (small[i - 1][j] + small[i + 1][j] + small[i][j - 1] + small[i][j + 1] + small[i][j]) / 5.0
    assert abs(out[i][j] - expect) < 1e-12, (out[i][j], expect)
    # On a cold cache this compiled one kernel; on a warm one, none.
    after_first = blur.compiled
    print(f"[Step 1] 6x8 grid: compiled {after_first} kernel(s), out[2][3] = {out[i][j]:.4f}")

    again = blur(INPUT=[[1.0] * 8] * 6)
    assert blur.compiled == after_first, "the same shape must not compile twice"
    print(f"[Step 2] 6x8 again: nothing more compiled, centre = {again[3][4]:.4f}")

    big = [[float((i + j) % 5) for j in range(64)] for i in range(48)]
    out = blur(INPUT=big)
    assert blur.compiled <= after_first + 1, "a new shape compiles at most once"
    assert len(out) == 48 and len(out[0]) == 64
    print(f"[Step 3] 48x64 grid: {blur.compiled} kernel(s) compiled in all, out[10][10] = {out[10][10]:.4f}")

    # A second process with the same program finds the kernel on disk.
    fresh = Kernel(
        "((▷0INPUT + ▽0INPUT + ▷1INPUT + ▽1INPUT + INPUT) / 5.0) → =",
        cache_dir=cache,
    )
    fresh(INPUT=small)
    assert fresh.compiled == 0, "the cache on disk serves a new Kernel object"
    print("[Step 4] a fresh Kernel with the same flows compiled nothing")

    # Several inputs, a function, a fixed point.
    relax = Kernel(
        """
        INPUT → X
        ((B / 4.0) - (step X)) ⇒ X
        X → =
        """,
        definitions="""
        step:{ U
            (▷U + ▽U) → S
            (S / 4.0) → =
        }
        """,
        tau=1e-12,
        max_iter=200,
        cache_dir=cache,
    )
    b = [1.0, -2.0, 3.0, 0.5, -1.5, 2.0, 4.0, -0.5]
    x = relax(INPUT=[0.0] * 8, B=b)
    assert relax.converged(), relax.sweeps()
    for k in range(8):
        left = x[k - 1] if k > 0 else 0.0
        right = x[k + 1] if k < 7 else 0.0
        assert abs(4.0 * x[k] + left + right - b[k]) < 1e-9
    print(f"[Step 5] Jacobi through a function, two inputs: settled in {relax.sweeps()} sweeps")

    print("\n=====================================================")
    print("  [SUCCESS] One program, three shapes, at most two compiles")
    print("=====================================================")


if __name__ == "__main__":
    main()
