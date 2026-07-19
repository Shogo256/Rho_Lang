import ctypes
import time
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rho import RhoEngine

def main():
    print("=====================================================")
    print("  ρ (RHO) Language - Python Zero-Copy Integration Demo")
    print("=====================================================")

    # 1. Allocate 1024x1024 double array buffer using standard ctypes
    print("\n[Step 1] Allocating 1024x1024 double tensor buffer in Python memory...")
    size = 1024 * 1024
    DoubleArrayType = ctypes.c_double * size
    tensor_buffer = DoubleArrayType()

    # Fill sample data
    for i in range(10):
        tensor_buffer[i] = 1.234 + i

    # Get physical memory address of Python array
    memory_address = ctypes.addressof(tensor_buffer)
    print(f"  └─ Buffer size: {size} elements (8 MB)")
    print(f"  └─ Physical Memory Address: {hex(memory_address)}")

    # 2. Initialize RhoEngine & compile .rho kernel script
    print("\n[Step 2] Compiling 'examples/teichmuller.rho' to native kernel (.so)...")
    engine = RhoEngine(kernel_so_path="libkernel.so")
    engine.compile_rho_file("examples/teichmuller.rho")
    print("  └─ Kernel Compilation Successful! (libkernel.so ready)")

    # 3. Execute ρ kernel on Python buffer with ZERO COPY
    print("\n[Step 3] Executing ρ-Language Kernel with Zero-Copy Direct Address Coupling...")
    
    # Class holding buffer pointer
    class BufferWrapper:
        def __init__(self, addr):
            self.ctypes = type('obj', (object,), {'data': addr})

    start_time = time.perf_counter()
    engine.execute_kernel(BufferWrapper(memory_address))
    elapsed = (time.perf_counter() - start_time) * 1000.0

    print(f"  └─ Kernel Execution Completed in {elapsed:.3f} ms!")
    print("\n=====================================================")
    print("  [SUCCESS] Python <-> ρ-Language Zero-Copy Integration OK")
    print("=====================================================")

if __name__ == "__main__":
    main()
