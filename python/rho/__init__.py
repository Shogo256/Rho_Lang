"""Python side of ρ (Rho): compile a .rho file with rhoc and call the kernel.

The compiler is found in this order: the RHOC environment variable, `cargo run`
when this module is imported from a checkout of the repository, and otherwise
the `rhoc` that `pip install rho-lang` puts next to the interpreter. clang is
still needed at run time — rhoc emits LLVM IR and asks clang to build the
shared library.
"""

import ctypes
import json
import os
import shutil
import subprocess
import sys
import sysconfig


def _checkout_root():
    """The repository this module is imported from, or None once installed."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return root if os.path.isfile(os.path.join(root, "Cargo.toml")) else None


def rhoc_command():
    """The command that runs the compiler, as a list for subprocess.

    `RHOC=/path/to/rhoc` overrides everything. Inside a checkout the compiler
    is built and run with cargo, so an edit to the source is what runs. Once
    installed from a wheel, the `rhoc` binary shipped in it is used.
    """
    explicit = os.environ.get("RHOC")
    if explicit:
        return [explicit]

    root = _checkout_root()
    if root is not None:
        return [
            "cargo", "run", "--quiet",
            "--manifest-path", os.path.join(root, "Cargo.toml"),
            "--bin", "rhoc", "--",
        ]

    suffix = ".exe" if sys.platform == "win32" else ""
    candidates = [
        os.path.join(sysconfig.get_path("scripts"), "rhoc" + suffix),
        os.path.join(os.path.dirname(sys.executable), "rhoc" + suffix),
        shutil.which("rhoc"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return [candidate]

    raise FileNotFoundError(
        "no rhoc compiler found: `pip install rho-lang`, set RHOC=/path/to/rhoc, "
        "or import this module from a checkout of the repository"
    )


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

    def compile_rho_file(self, rho_file_path, bind=None, tau=None, require_contract=False,
                         max_iter=None):
        """Compile a .rho script with the rhoc driver.

        `bind` maps space names to the buffers (or raw addresses) the kernel
        should read and write directly. Those addresses are baked into the
        kernel, which is what makes rho_kernel_exec() a genuine zero-copy call
        rather than a read of whatever the .rho source happened to write.

        `max_iter` caps every `⇒` in the program; a program that iterates
        cannot be compiled without it. `tau` is the tolerance it stops at.
        """
        cmd = rhoc_command() + [rho_file_path, "-o", self.kernel_so_path]
        if tau is not None:
            cmd += ["--tau", str(tau)]
        if max_iter is not None:
            cmd += ["--max-iter", str(max_iter)]
        if require_contract:
            cmd += ["--require-contract"]
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

    def contract(self):
        """What the compiler proved about this kernel.

        The facts travel inside the .so, so a host can check them at load time
        rather than trusting a line that scrolled past during the build.
        Returns None for a kernel built before contracts existed.
        """
        return self.get_metadata().get("contract")

    def require_contract(self, finite_output=False):
        """Raise unless the kernel's contract is complete.

        Use it as an admission check at the boundary of a system that is not
        allowed to run kernels with unproven behaviour.
        """
        c = self.contract()
        if c is None:
            raise RuntimeError(
                "this kernel carries no contract; rebuild it with a current rhoc"
            )
        problems = []
        if not c.get("divisions_proven_safe", False):
            problems.append("a division may divide by zero")
        if c.get("open_obligations", 0):
            problems.append(f"{c['open_obligations']} obligation(s) unproven")
        if finite_output and not c.get("output_proven_finite", False):
            lo, hi = c.get("output_range", [None, None])
            problems.append(f"output range is not bounded on both ends ({lo}, {hi})")
        if problems:
            raise RuntimeError("incomplete contract: " + "; ".join(problems))
        return c

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

    def sweeps(self):
        """How many sweeps the `⇒` loops of the most recent call took, in all.

        A kernel without `⇒` reports 0. The count is per kernel, not per
        thread: the language has no concurrency, and neither has this.
        """
        fn = getattr(self._lib_or_load(), "rho_kernel_sweeps", None)
        if fn is None:
            return 0
        fn.restype = ctypes.c_int64
        return int(fn())

    def converged(self):
        """Whether every `⇒` of the most recent call stopped on the tolerance.

        False means at least one loop ran into its cap. A kernel without `⇒`
        reports True.
        """
        fn = getattr(self._lib_or_load(), "rho_kernel_converged", None)
        if fn is None:
            return True
        fn.restype = ctypes.c_int64
        return bool(fn())

    def spaces(self):
        """Every space of the kernel as (name, shape, role), in table order.

        The order is the one `execute_spaces()` and the C entrypoint
        `rho_kernel_exec_spaces` expect. The role says what the caller does
        with a space: "input" must be supplied, "output" is where the result
        lands, and "internal" may be left to the kernel.
        """
        return [
            (s["name"], list(s["shape"]), s.get("role"))
            for s in self.get_metadata().get("spaces", [])
        ]

    def execute_spaces(self, buffers):
        """Run the kernel with every space supplied at call time.

        `buffers` maps space names to ctypes arrays, numpy arrays or raw
        addresses. Every "input" space must be present. Any other space may be
        omitted, in which case the kernel uses memory of its own for it; an
        intermediate that is supplied is filled in and left for the caller to
        read. Nothing is baked into the kernel, so one compiled kernel serves
        any buffers — this is how a kernel with several inputs, such as a
        matrix product, is meant to be called.
        """
        lib = self._lib_or_load()
        fn = getattr(lib, "rho_kernel_exec_spaces", None)
        if fn is None:
            raise AttributeError(
                "@rho_kernel_exec_spaces not found; recompile the kernel with the "
                "current rhoc."
            )

        spaces = self.get_metadata().get("spaces", [])
        names = [s["name"] for s in spaces]
        unknown = sorted(set(buffers) - set(names))
        if unknown:
            raise ValueError(f"unknown space(s) {unknown}; this kernel has {names}")
        missing = [
            s["name"] for s in spaces
            if s.get("role") == "input" and buffers.get(s["name"]) is None
        ]
        if missing:
            raise ValueError(
                f"input space(s) {missing} need a buffer; without one the kernel "
                f"returns without running"
            )

        table = (ctypes.c_void_p * len(spaces))()
        for index, space in enumerate(spaces):
            buf = buffers.get(space["name"])
            if buf is None:
                table[index] = None
                continue
            cells = 1
            for extent in space["shape"]:
                cells *= extent
            length = _buffer_length(buf)
            if length is not None and length < cells:
                raise ValueError(
                    f"buffer for {space['name']} holds {length} values but its "
                    f"shape {space['shape']} needs {cells}"
                )
            table[index] = _buffer_address(buf)

        fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        fn.restype = None
        fn(table)
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
