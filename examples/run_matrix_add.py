import ctypes
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rho import RhoEngine

def main():
    print("=====================================================")
    print("  ρ (RHO) Language - 2x2 Matrix Addition Verification")
    print("=====================================================")

    # Prepare 2x2 input & output double buffers
    size = 4
    DoubleArrayType = ctypes.c_double * size

    input_buf = DoubleArrayType(1.0, 2.0, 3.0, 4.0)
    output_buf = DoubleArrayType(0.0, 0.0, 0.0, 0.0)

    print(f"\n[Step 1] Input Matrix Values : {[input_buf[i] for i in range(size)]}")
    print(f"[Step 2] Initial Output Values: {[output_buf[i] for i in range(size)]}")

    # Compile kernel
    engine = RhoEngine(kernel_so_path="libkernel.so")
    engine.compile_rho_file("examples/matrix_add.rho")

    # Execute with pointers
    start_time = time.perf_counter()
    engine.execute_kernel_with_args(input_buf, output_buf)
    elapsed = (time.perf_counter() - start_time) * 1000.0

    print(f"\n[Step 3] Executed ρ-Kernel in {elapsed:.3f} ms")
    print(f"[Result] Computed Output Values: {[output_buf[i] for i in range(size)]}")

    # Verify: 1.0+1.0=2.0, 2.0+2.0=4.0, 3.0+3.0=6.0, 4.0+4.0=8.0
    expected = [2.0, 4.0, 6.0, 8.0]
    actual = [output_buf[i] for i in range(size)]
    if actual == expected:
        print("\n=====================================================")
        print("  [SUCCESS] Matrix Addition Computed Correctly: [2, 4, 6, 8]")
        print("=====================================================")
    else:
        print(f"\n[ERROR] Result mismatch: {actual} != {expected}")

if __name__ == "__main__":
    main()
