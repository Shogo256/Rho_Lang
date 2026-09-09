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
    contract = engine.contract()
    print("\n[Step 1] What the compiler proved about the loop:")
    for claim in contract.get("iterations", []):
        print(f"  ⇒ {claim['target']}: converges={claim['converges']} factor={claim['factor']}")

    print("\n[Step 2] Running...")
    engine.execute_kernel_with_args(b, x)
    print(f"  sweeps={engine.sweeps()} converged={engine.converged()}")

    residual = max(
        abs(4.0 * x[i] + (x[i - 1] if i > 0 else 0.0) + (x[i + 1] if i < n - 1 else 0.0) - b[i])
        for i in range(n)
    )
    print(f"  largest residual of the system: {residual:.3e}")

    ok = engine.converged() and residual < 1e-10 and contract["iterations"][0]["converges"]
    if ok:
        print("\n=====================================================")
        print("  [SUCCESS] Settled on the tolerance, as the proof said it would")
        print("=====================================================")
    else:
        print("\n[ERROR] the iteration did not behave as proved")
        sys.exit(1)


if __name__ == "__main__":
    main()
