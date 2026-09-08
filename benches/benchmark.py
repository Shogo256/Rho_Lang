"""Benchmark one ρ kernel against the same expression in pure Python and NumPy.

All three run the identical formula over the identical element count, so the
ratios below mean something. The kernel is compared against an interpreted
Python loop and, when available, NumPy — not against BLAS or a tuned
hand-written kernel.
"""

import ctypes
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rho import RhoEngine

SIZE = 1_000_000

KERNEL_SOURCE = """{{
    INPUT:◯ □ {size} 1
    ((INPUT × 2.0) / (INPUT + 1.0)) → OUTPUT
    OUTPUT → =
}}
"""


def bench(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    print(f"  └─ {label}: {elapsed:.3f} ms")
    return elapsed, result


def main():
    print("=====================================================")
    print("  ρ (RHO) Language Performance Benchmark")
    print("=====================================================")
    print(f"\nExpression : (x × 2) / (x + 1)")
    print(f"Elements   : {SIZE} doubles ({SIZE * 8 / (1024 * 1024):.2f} MB)")

    here = os.path.dirname(os.path.abspath(__file__))
    rho_path = os.path.join(here, "_benchmark_kernel.rho")
    so_path = os.path.join(here, "_benchmark_kernel.so")
    with open(rho_path, "w", encoding="utf-8") as f:
        f.write(KERNEL_SOURCE.format(size=SIZE))

    DoubleArray = ctypes.c_double * SIZE
    source = [float(i) for i in range(SIZE)]

    print("\n1. Pure Python loop")
    py_time, py_result = bench("pure Python", lambda: [(x * 2.0) / (x + 1.0) for x in source])

    np_time = None
    try:
        import numpy as np
        print("\n2. NumPy")
        arr = np.arange(SIZE, dtype=np.float64)
        np_time, _ = bench("NumPy", lambda: (arr * 2.0) / (arr + 1.0))
    except ImportError:
        print("\n2. NumPy not installed, skipping")

    print("\n3. ρ (RHO) compiled kernel")
    engine = RhoEngine(kernel_so_path=so_path)
    engine.compile_rho_file(rho_path)

    input_buf = DoubleArray(*source)
    output_buf = DoubleArray()

    # Warm-up: fault in the output pages so the timed run measures compute.
    engine.execute_kernel_with_args(input_buf, output_buf)
    rho_time, _ = bench("ρ kernel", lambda: engine.execute_kernel_with_args(input_buf, output_buf))

    # Correctness before speed.
    mismatches = [i for i in range(0, SIZE, 9973) if abs(output_buf[i] - py_result[i]) > 1e-12]
    if mismatches:
        print(f"\n[ERROR] kernel disagrees with the Python baseline at {mismatches[:5]}")
        return 1
    print("\n  Verified: kernel output matches the Python baseline")

    print("\n=====================================================")
    print(f"  vs pure Python : {py_time / rho_time:.1f}x")
    if np_time:
        print(f"  vs NumPy       : {np_time / rho_time:.1f}x")
    print("=====================================================")

    for path in (rho_path, so_path):
        if os.path.exists(path):
            os.remove(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
