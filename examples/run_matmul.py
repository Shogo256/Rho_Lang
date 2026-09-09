import ctypes
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from rho import RhoEngine


def main():
    print("=====================================================")
    print("  ρ (RHO) Language - Matrix Product With Two Inputs")
    print("=====================================================")

    # A is 2x3 and B is 3x4. The kernel declares them as [2,3,1] and [1,3,4]
    # so that their shared axis lines up; the product is [2,4].
    a = (ctypes.c_double * 6)(1.0, 2.0, 3.0,
                              4.0, 5.0, 6.0)
    b = (ctypes.c_double * 12)(1.0, 0.0, 2.0, 1.0,
                               0.0, 1.0, 1.0, 2.0,
                               3.0, 1.0, 0.0, 1.0)
    c = (ctypes.c_double * 8)()

    # Nothing about these buffers is known at compile time: the kernel takes
    # one pointer per space when it is called.
    engine = RhoEngine(kernel_so_path="libmatmul.so")
    engine.compile_rho_file("examples/matmul.rho")

    print("\n[Step 1] Spaces, in the order the kernel takes them:")
    for name, shape, role in engine.spaces():
        print(f"  {name:8} shape={shape}  role={role}")

    print("\n[Step 2] Running with A, B and OUTPUT supplied by Python...")
    engine.execute_spaces({"A": a, "B": b, "OUTPUT": c})

    actual = list(c)
    expected = [10.0, 5.0, 4.0, 8.0, 22.0, 11.0, 13.0, 20.0]
    print(f"[Result] {actual[:4]}")
    print(f"         {actual[4:]}")
    if actual == expected:
        print("\n=====================================================")
        print("  [SUCCESS] A × B computed without any address baked in")
        print("=====================================================")
    else:
        print(f"\n[ERROR] Result mismatch: {actual} != {expected}")
        sys.exit(1)


if __name__ == "__main__":
    main()
