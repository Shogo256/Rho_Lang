import ctypes
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rho import RhoEngine

def main():
    print("=====================================================")
    print("  ρ (RHO) Language - Advanced Python FFI Integration")
    print("=====================================================")

    # 1. Allocate input and output double tensor buffers in Python memory
    print("\n[Step 1] Allocating input & output double tensor buffers...")
    size = 1024 * 1024
    DoubleArrayType = ctypes.c_double * size

    input_tensor = DoubleArrayType()
    output_tensor = DoubleArrayType()

    for i in range(10):
        input_tensor[i] = 10.0 + i

    in_addr = ctypes.addressof(input_tensor)
    out_addr = ctypes.addressof(output_tensor)

    print(f"  └─ Input Tensor Address : {hex(in_addr)}")
    print(f"  └─ Output Tensor Address: {hex(out_addr)}")

    # 2. Compile .rho script
    print("\n[Step 2] Compiling 'examples/teichmuller.rho' via rhoc driver...")
    engine = RhoEngine(kernel_so_path="libkernel.so")
    engine.compile_rho_file("examples/teichmuller.rho")
    print("  └─ Kernel compiled successfully!")

    # Retrieve C-ABI Metadata
    metadata = engine.get_metadata()
    print(f"  └─ C-ABI Metadata Inspection: {metadata}")

    # 3. Execute with arguments
    print("\n[Step 3] Executing ρ-Kernel with Argument Pointer Coupling...")
    start_time = time.perf_counter()
    engine.execute_kernel_with_args(input_tensor, output_tensor)
    elapsed = (time.perf_counter() - start_time) * 1000.0

    print(f"  └─ Execution Completed in {elapsed:.3f} ms!")
    print(f"  └─ Sample Output Result Values: {[output_tensor[i] for i in range(4)]}")

    print("\n=====================================================")
    print("  [SUCCESS] Advanced FFI Pointer Coupling OK")
    print("=====================================================")

if __name__ == "__main__":
    main()
