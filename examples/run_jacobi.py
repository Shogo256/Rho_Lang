import ctypes
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from rho import RhoEngine


def main():
    print("=====================================================")
    print("  ρ (RHO) Language - Jacobi Iteration To A Fixed Point")
    print("=====================================================")

    # 4·x[i] + x[i-1] + x[i+1] = b[i], with x = 0 outside the grid.
    n = 64
    b = (ctypes.c_double * n)(*[math.sin(i * 0.3) for i in range(n)])
    x = (ctypes.c_double * n)()

    # The cap is part of what the kernel computes, so it is passed here;
    # the tolerance is the threshold symbol 𝜏.
    engine = RhoEngine(kernel_so_path="libjacobi.so")
    engine.compile_rho_file("examples/jacobi.rho", tau=1e-12, max_iter=200)
    print("\n[Step 1] Built with", engine.get_metadata().get("iteration"))

    print("\n[Step 2] Running...")
    engine.execute_kernel_with_args(b, x)
    print(f"  sweeps={engine.sweeps()} converged={engine.converged()}")

    residual = max(
        abs(4.0 * x[i] + (x[i - 1] if i > 0 else 0.0) + (x[i + 1] if i < n - 1 else 0.0) - b[i])
        for i in range(n)
    )
    print(f"  largest residual of the system: {residual:.3e}")

    if engine.converged() and residual < 1e-10:
        print("\n=====================================================")
        print("  [SUCCESS] Settled on the tolerance")
        print("=====================================================")
    else:
        print("\n[ERROR] the iteration did not settle")
        sys.exit(1)


if __name__ == "__main__":
    main()
