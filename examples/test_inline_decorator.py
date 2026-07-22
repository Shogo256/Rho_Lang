import ctypes
import rho

@rho.compile
def my_matrix_op():
    """
    {
        &[0x7A4F]:INPUT:◯ □ 4 1
        (INPUT + 2.0) → OUTPUT
        OUTPUT → =
    }
    """
    pass

# Create standard ctypes arrays
double_array_4 = ctypes.c_double * 4
input_data = double_array_4(10.0, 20.0, 30.0, 40.0)
output_data = double_array_4(0.0, 0.0, 0.0, 0.0)

# Execute the RHO inline compiled decorator!
my_matrix_op(input_data, output_data)

print("Input data:", list(input_data))
print("Output data (after RHO kernel):", list(output_data))
assert list(output_data) == [12.0, 22.0, 32.0, 42.0]
print("SUCCESS: Python inline @rho.compile decorator is fully working!")
