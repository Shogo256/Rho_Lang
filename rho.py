import ctypes
import os
import subprocess

class RhoEngine:
    """Python wrapper for ρ (RHO) Language Compiler Kernel"""

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

        # Call generated C-ABI function @rho_kernel_exec
        kernel_fn = getattr(self._lib, "rho_kernel_exec", None)
        if kernel_fn is None:
            raise AttributeError("Function @rho_kernel_exec not found in shared library")

        kernel_fn()
        return True
