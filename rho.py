import ctypes
import json
import os
import subprocess


def _buffer_length(buf):
    """Element count of a ctypes array or a numpy array, or None if unknown."""
    if buf is None:
        return None
    if hasattr(buf, "size"):          # numpy
        return int(buf.size)
    try:
        return len(buf)               # ctypes array
    except TypeError:
        return None


def _buffer_address(buf):
    if hasattr(buf, "ctypes"):        # numpy
        return buf.ctypes.data
    if isinstance(buf, int):
        return buf
    return ctypes.addressof(buf)


class RhoEngine:
    """Python wrapper for a compiled ρ (RHO) kernel, with metadata and length checks."""

    def __init__(self, kernel_so_path="libkernel.so"):
        self.kernel_so_path = os.path.abspath(kernel_so_path)
        self._lib = None
        self._bound = {}

    def compile_rho_file(self, rho_file_path, bind=None, tau=None):
        """Compile a .rho script with the rhoc driver.

        `bind` maps space names to the buffers (or raw addresses) the kernel
        should read and write directly. Those addresses are baked into the
        kernel, which is what makes rho_kernel_exec() a genuine zero-copy call
        rather than a read of whatever the .rho source happened to write.
        """
        cmd = ["cargo", "run", "--quiet", "--bin", "rhoc", "--",
               rho_file_path, "-o", self.kernel_so_path]
        if tau is not None:
            cmd += ["--tau", str(tau)]
        self._bound = {}
        for name, target in (bind or {}).items():
            address = _buffer_address(target)
            self._bound[name] = address
            cmd += ["--bind", f"{name}={hex(address)}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ρ Compilation Error:\n{res.stderr}")
        self._load_library()

    def _load_library(self):
        if not os.path.exists(self.kernel_so_path):
            raise FileNotFoundError(f"Shared library not found: {self.kernel_so_path}")
        self._lib = ctypes.CDLL(self.kernel_so_path)

    def _lib_or_load(self):
        if self._lib is None:
            self._load_library()
        return self._lib

    def element_count(self):
        """Cells the kernel sweeps. Buffers shorter than this are rejected."""
        lib = self._lib_or_load()
        fn = getattr(lib, "rho_kernel_element_count", None)
        if fn is None:
            # Kernel built before the length export existed.
            return self.get_metadata().get("elements")
        fn.restype = ctypes.c_int64
        return int(fn())

    def get_metadata(self):
        """Retrieve the C-ABI JSON metadata from the compiled kernel."""
        lib = self._lib_or_load()
        meta_fn = getattr(lib, "rho_kernel_metadata", None)
        if meta_fn is not None:
            meta_fn.restype = ctypes.c_char_p
            raw = meta_fn()
            if raw:
                return json.loads(raw.decode("utf-8"))
        return {"spaces": []}

    def execute_kernel(self, expected_buffer=None):
        """Run the zero-copy entrypoint, which uses the addresses compiled in.

        Only meaningful when the kernel was compiled with `bind=`; otherwise the
        `&[0x...]` literal in the source points at an address this process does
        not own, and the kernel refuses to touch it.
        """
        lib = self._lib_or_load()

        bindings = {b["name"]: b["address"] for b in self.get_metadata().get("bindings", [])}
        if "INPUT" not in bindings:
            raise RuntimeError(
                "this kernel has no INPUT binding, so rho_kernel_exec() does nothing. "
                "Compile with bind={'INPUT': buf} or use execute_kernel_with_args()."
            )
        if expected_buffer is not None:
            address = _buffer_address(expected_buffer)
            if bindings["INPUT"] != address:
                raise ValueError(
                    f"kernel is bound to {hex(bindings['INPUT'])} but this buffer lives at "
                    f"{hex(address)}. Recompile with bind={{'INPUT': buf}}."
                )

        fn = getattr(lib, "rho_kernel_exec", None)
        if fn is None:
            raise AttributeError("@rho_kernel_exec not found in shared library")
        fn()
        return True

    def execute_kernel_with_args(self, input_buf, output_buf=None):
        """Run the kernel over caller-owned input/output buffers."""
        lib = self._lib_or_load()

        required = self.element_count()
        in_len = _buffer_length(input_buf)
        out_len = _buffer_length(output_buf)

        for label, length in (("input", in_len), ("output", out_len)):
            if required is not None and length is not None and length < required:
                raise ValueError(
                    f"{label} buffer holds {length} doubles but this kernel sweeps "
                    f"{required}. Allocate at least {required}, or call "
                    f"execute_kernel_bounded() to clamp the sweep."
                )

        in_addr = _buffer_address(input_buf)
        out_addr = _buffer_address(output_buf) if output_buf is not None else in_addr

        fn = getattr(lib, "rho_kernel_exec_with_args", None)
        if fn is None:
            raise AttributeError("@rho_kernel_exec_with_args not found in shared library")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        fn(ctypes.c_void_p(in_addr), ctypes.c_void_p(out_addr))
        return True

    def execute_kernel_bounded(self, input_buf, output_buf=None, count=None):
        """Run the kernel but clamp the sweep to `count` cells.

        Use this when the buffers are deliberately shorter than the declared
        space; the kernel stops at whatever the caller actually owns.
        """
        lib = self._lib_or_load()

        if count is None:
            lengths = [n for n in (_buffer_length(input_buf), _buffer_length(output_buf))
                       if n is not None]
            if not lengths:
                raise ValueError("count is required for buffers of unknown length")
            count = min(lengths)

        fn = getattr(lib, "rho_kernel_exec_bounded", None)
        if fn is None:
            raise AttributeError(
                "@rho_kernel_exec_bounded not found; recompile the kernel with the "
                "current rhoc."
            )
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

        in_addr = _buffer_address(input_buf)
        out_addr = _buffer_address(output_buf) if output_buf is not None else in_addr
        fn(ctypes.c_void_p(in_addr), ctypes.c_void_p(out_addr), ctypes.c_int64(count))
        return True


def compile(func):
    """Decorator that compiles the RHO block in a function's docstring."""
    import tempfile

    doc = func.__doc__
    if not doc:
        raise ValueError("No docstring containing a RHO block found in function")

    start_idx = doc.find('{')
    end_idx = doc.rfind('}')
    if start_idx == -1 or end_idx == -1:
        raise ValueError("RHO code block not found inside docstring (must be enclosed in { ... })")

    rho_code = doc[start_idx:end_idx + 1]

    with tempfile.NamedTemporaryFile(suffix=".rho", delete=False, mode="w", encoding="utf-8") as f:
        f.write(rho_code)
        temp_rho_path = f.name

    so_path = temp_rho_path.replace(".rho", ".so")

    engine = RhoEngine(kernel_so_path=so_path)
    try:
        engine.compile_rho_file(temp_rho_path)
    finally:
        if os.path.exists(temp_rho_path):
            os.remove(temp_rho_path)

    def wrapper(input_buf, output_buf=None):
        return engine.execute_kernel_with_args(input_buf, output_buf)

    wrapper.engine = engine
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
