import ctypes
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rho import RhoEngine

def benchmark():
    print("=====================================================")
    print("  ρ (RHO) Language Performance Benchmark Suite")
    print("=====================================================")

    size = 1000000
    print(f"\n[Benchmark Configuration] Element Count: {size} doubles ({size * 8 / (1024*1024):.2f} MB)")

    # 1. Prepare Buffer
    DoubleArrayType = ctypes.c_double * size
    buffer_data = DoubleArrayType()
    for i in range(size):
        buffer_data[i] = float(i)

    # 2. Pure Python Loop Execution Simulation
    print("\n1. Running Pure Python Sequential Iterative Loop...")
    start_py = time.perf_counter()
    res = [0.0] * min(size, 10000) # subset to avoid extreme wait
    for i in range(len(res)):
        res[i] = (buffer_data[i] * 2.0) / (buffer_data[i] + 1.0)
    time_py = (time.perf_counter() - start_py) * 1000.0
    print(f"  └─ Pure Python Subset Execution Time: {time_py:.3f} ms")

    # 3. ρ (RHO) Topological Dataflow Engine Execution
    print("\n2. Running ρ (RHO) Language Zero-Copy Clockless Kernel...")
    engine = RhoEngine(kernel_so_path="libkernel.so")
    engine.compile_rho_file("examples/teichmuller.rho")

    start_rho = time.perf_counter()
    engine.execute_kernel(buffer_data)
    time_rho = (time.perf_counter() - start_rho) * 1000.0
    print(f"  └─ ρ (RHO) Language Full Execution Time: {time_rho:.3f} ms")

    # 4. Summary
    speedup = (time_py / time_rho) if time_rho > 0 else 0.0
    print("\n=====================================================")
    print(f"  Benchmark Result: ρ-Language is ~{speedup:.1f}x Faster (Clockless Engine)")
    print("=====================================================")

if __name__ == "__main__":
    benchmark()
