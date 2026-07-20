import ctypes
import os
import json
import subprocess

class RhoEngine:
    """Python wrapper for ρ (RHO) Language Compiler Kernel with Metadata & Safety Checks"""

    def __init__(self, kernel_so_path="libkernel.so"):
        self.kernel_so_path = os.path.abspath(kernel_so_path)
        self._lib = None

    def compile_rho_file(self, rho_file_path):
        """Compile .rho script using rhoc driver"""
        cmd = ["cargo", "run", "--quiet", "--", rho_file_path, "-o", self.kernel_so_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ρ Compilation Error:\n{res.stderr}")
        self._load_library()

    def _load_library(self):
        if os.path.exists(self.kernel_so_path):
            self._lib = ctypes.CDLL(self.kernel_so_path)
        else:
            raise FileNotFoundError(f"Shared library not found: {self.kernel_so_path}")

    def get_metadata(self):
        """Retrieve C-ABI JSON Metadata from compiled kernel"""
        if self._lib is None:
            self._load_library()

        meta_fn = getattr(self._lib, "rho_kernel_metadata", None)
        if meta_fn is not None:
            meta_fn.restype = ctypes.c_char_p
            raw_str = meta_fn()
            if raw_str:
                return json.loads(raw_str.decode("utf-8"))
        return {"spaces": []}

    def execute_kernel(self, array_or_wrapper):
        """Execute ρ kernel directly on memory array buffer via Zero-Copy address"""
        if self._lib is None:
            self._load_library()

        if hasattr(array_or_wrapper, "ctypes"):
            address = array_or_wrapper.ctypes.data
        elif isinstance(array_or_wrapper, int):
            address = array_or_wrapper
        else:
            address = ctypes.addressof(array_or_wrapper)

        print(f"[RhoEngine] Zero-Copy Memory Address bound: {hex(address)}")

        kernel_fn = getattr(self._lib, "rho_kernel_exec", None)
        if kernel_fn is None:
            raise AttributeError("Function @rho_kernel_exec not found in shared library")

        kernel_fn()
        return True

    def execute_kernel_with_args(self, input_buf, output_buf=None):
        """Execute ρ kernel with direct argument pointers for input & output buffers"""
        if self._lib is None:
            self._load_library()

        in_addr = input_buf.ctypes.data if hasattr(input_buf, "ctypes") else ctypes.addressof(input_buf)
        out_addr = output_buf.ctypes.data if hasattr(output_buf, "ctypes") else (ctypes.addressof(output_buf) if output_buf else in_addr)

        print(f"[RhoEngine] Input Address: {hex(in_addr)}, Output Address: {hex(out_addr)}")

        kernel_fn = getattr(self._lib, "rho_kernel_exec_with_args", None)
        if kernel_fn is not None:
            kernel_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            kernel_fn(ctypes.c_void_p(in_addr), ctypes.c_void_p(out_addr))
        else:
            self.execute_kernel(input_buf)

        return True
