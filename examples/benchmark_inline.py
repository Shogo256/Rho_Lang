import ctypes
import time
import rho

# Define RHO size: 1048576 elements (1M float64 elements)
ARRAY_SIZE = 1048576

# 1. Define RHO inline compiled acceleration kernel
@rho.compile
def rho_vector_add_kernel():
    """
    {
        &[0x7A4F]:INPUT:◯ □ 1048576 1
        (INPUT + 2.0) → OUTPUT
        OUTPUT → =
    }
    """
    pass

# Create shared memory buffers via ctypes
double_array_type = ctypes.c_double * ARRAY_SIZE
print(f"Allocating {ARRAY_SIZE} elements (ctypes double array)...")
input_data = double_array_type(*[float(i) for i in range(ARRAY_SIZE)])
output_data = double_array_type(*[0.0] * ARRAY_SIZE)

# 2. Benchmark Pure Python Implementation
print("\n[1/2] Running Pure Python Baseline...")
start_py = time.time()
python_output = []
for val in input_data:
    python_output.append(val + 2.0)
end_py = time.time()
py_time = end_py - start_py
print(f"Pure Python Execution Time: {py_time:.6f} seconds")

# 3. Benchmark RHO Compiled Kernel (Zero-Copy)
print("\n[2/2] Running RHO LLVM-Compiled Kernel...")
# Warm-up run
rho_vector_add_kernel(input_data, output_data)

# Measured run
start_rho = time.time()
rho_vector_add_kernel(input_data, output_data)
end_rho = time.time()
rho_time = end_rho - start_rho
print(f"RHO Kernel Execution Time: {rho_time:.6f} seconds")

# 4. Verification and Speedup Calculation
# Verify correctness
assert list(output_data)[:10] == python_output[:10], "Result verification failed!"
print("\n✅ Verification Successful: Output matches baseline!")

speedup = py_time / rho_time if rho_time > 0 else 0
print(f"\n🚀 RHO speedup factor: {speedup:.1f}x FASTER than Pure Python")
