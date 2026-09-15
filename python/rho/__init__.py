# SPDX-License-Identifier: Apache-2.0
"""Python side of ρ (Rho): compile a .rho file with rhoc and call the kernel.

The compiler is found in this order: the RHOC environment variable, `cargo run`
when this module is imported from a checkout of the repository, and otherwise
the `rhoc` that `pip install rho-lang` puts next to the interpreter. clang is
still needed at run time — rhoc emits LLVM IR and asks clang to build the
shared library.
"""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap


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

    def compile_rho_file(self, rho_file_path, bind=None, tau=None, max_iter=None, threads=None,
                         portable=False, precision=None, load=True):
        """Compile a .rho script with the rhoc driver.

        `bind` maps space names to the buffers (or raw addresses) the kernel
        should read and write directly. Those addresses are baked into the
        kernel, which is what makes rho_kernel_exec() a genuine zero-copy call
        rather than a read of whatever the .rho source happened to write.

        `max_iter` caps every `⇒` in the program; a program that iterates
        cannot be compiled without it. `tau` is the tolerance it stops at.

        `threads` is how many threads every sweep is split across (None or
        0: one per CPU; the kernel also honours RHO_THREADS when it runs).
        `portable=True` builds for any x86-64 instead of this machine.
        `precision="f32"` computes in single precision. `load=False` leaves
        the library on disk without loading it.
        """
        cmd = rhoc_command() + [rho_file_path, "-o", self.kernel_so_path]
        if tau is not None:
            cmd += ["--tau", str(tau)]
        if max_iter is not None:
            cmd += ["--max-iter", str(max_iter)]
        if threads is not None:
            cmd += ["--threads", str(threads)]
        if portable:
            cmd += ["--portable"]
        if precision == "f32":
            cmd += ["--f32"]
        elif precision not in (None, "f64"):
            raise ValueError(f"precision is 'f64' or 'f32', not {precision!r}")
        self._bound = {}
        for name, target in (bind or {}).items():
            address = _buffer_address(target)
            self._bound[name] = address
            cmd += ["--bind", f"{name}={hex(address)}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ρ Compilation Error:\n{res.stderr}")
        if load:
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


# --------------------------------------------------------------------------
# A kernel at any shape: the flows are written once, the declarations come
# from the arrays passed at call time, and each shape is compiled once and
# kept on disk.
# --------------------------------------------------------------------------


def _default_cache_dir():
    explicit = os.environ.get("RHO_CACHE_DIR")
    if explicit:
        return explicit
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "rho-lang")


_COMPILER_STAMP = None


def _compiler_stamp():
    """Something that changes when the compiler does, so a kernel built by an
    older rhoc is not served for a newer one. Taken once per process: in a
    checkout the first compile is what builds rhoc, and a stamp read before
    and after it would name two compilers for one program."""
    global _COMPILER_STAMP
    if _COMPILER_STAMP is not None:
        return _COMPILER_STAMP
    cmd = rhoc_command()
    root = _checkout_root()
    if cmd[0] == "cargo" and root is not None:
        # Build first, so the binary stamped is the one that will run.
        subprocess.run(
            ["cargo", "build", "--quiet", "--manifest-path", os.path.join(root, "Cargo.toml"), "--bin", "rhoc"],
            check=False, capture_output=True,
        )
    parts = [" ".join(cmd)]
    candidates = [cmd[0]]
    if root is not None:
        candidates += [os.path.join(root, "target", profile, "rhoc") for profile in ("debug", "release")]
    for path in candidates:
        try:
            st = os.stat(path)
        except OSError:
            continue
        parts.append(f"{path}:{st.st_size}:{int(st.st_mtime)}")
    _COMPILER_STAMP = "|".join(parts)
    return _COMPILER_STAMP


def _numpy():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _flatten(value, out):
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, out)
    else:
        out.append(float(value))
    return out


def _shape_of_nested(value):
    shape = []
    probe = value
    while isinstance(probe, (list, tuple)):
        shape.append(len(probe))
        if not probe:
            break
        probe = probe[0]
    return tuple(shape)


def _as_space(value, ctype):
    """(shape, buffer to pass, object to keep alive) for one argument.

    Accepts a numpy array (made contiguous and of the kernel's width if it is
    not), a ctypes array (one axis), a `(buffer, shape)` pair for anything
    else, or nested lists, which are copied into a ctypes array.
    """
    np = _numpy()
    if np is not None and isinstance(value, np.ndarray):
        want = np.float32 if ctype is ctypes.c_float else np.float64
        arr = value
        if arr.dtype != want or not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=want)
        return tuple(int(d) for d in arr.shape), arr, arr
    if isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], (int, float)):
        buf, shape = value
        return tuple(int(d) for d in shape), buf, buf
    if isinstance(value, ctypes.Array):
        return (len(value),), value, value
    if isinstance(value, (list, tuple)):
        shape = _shape_of_nested(value)
        flat = _flatten(value, [])
        cells = 1
        for d in shape:
            cells *= d
        if cells != len(flat):
            raise ValueError(f"a nested list of shape {shape} holds {cells} cells, not {len(flat)}")
        buf = (ctype * len(flat))(*flat)
        return shape, buf, buf
    raise TypeError(
        f"a space is a numpy array, a ctypes array, nested lists or a (buffer, shape) "
        f"pair, not {type(value).__name__}"
    )


def _from_space(buf, shape, ctype):
    """What a call returns for a space the kernel filled: a numpy array when
    numpy is there, nested lists otherwise."""
    np = _numpy()
    if np is not None:
        dtype = np.float32 if ctype is ctypes.c_float else np.float64
        if isinstance(buf, np.ndarray):
            return buf
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
    flat = list(buf)

    def nest(values, dims):
        if len(dims) <= 1:
            return list(values)
        step = len(values) // dims[0]
        return [nest(values[i * step:(i + 1) * step], dims[1:]) for i in range(dims[0])]

    return nest(flat, list(shape))


class Kernel:
    """A ρ kernel written without declarations, compiled for each shape it is
    called with, and cached on disk.

        blur = Kernel("((▷0INPUT + ▽0INPUT + ▷1INPUT + ▽1INPUT + INPUT) / 5.0) → =")
        out = blur(INPUT=image)          # compiles for image's shape once
        out = blur(INPUT=other_image)    # same shape: the cached kernel

    The flows are the inside of the program's braces; `definitions` are the
    functions written before them. Every named argument becomes a space
    declared with the argument's shape, so the program is written once and
    used at any shape. A space the kernel fills and the caller did not pass
    (OUTPUT) is allocated and returned — one value, or a dict of them.

    Kernels live in `cache_dir` (RHO_CACHE_DIR, else ~/.cache/rho-lang),
    keyed by the program, the options and the compiler, so a second process
    with the same program does not compile either.
    """

    def __init__(self, flows, definitions="", *, tau=None, max_iter=None, threads=None,
                 precision="f64", portable=False, cache_dir=None):
        self.flows = textwrap.dedent(flows).strip("\n")
        self.definitions = textwrap.dedent(definitions).strip("\n")
        self.tau = tau
        self.max_iter = max_iter
        self.threads = threads
        if precision not in ("f64", "f32"):
            raise ValueError(f"precision is 'f64' or 'f32', not {precision!r}")
        self.precision = precision
        self.portable = portable
        self.cache_dir = os.path.abspath(cache_dir or _default_cache_dir())
        self._engines = {}
        self._last = None
        self.compiled = 0

    @property
    def _ctype(self):
        return ctypes.c_float if self.precision == "f32" else ctypes.c_double

    def program(self, shapes):
        """The full program for these shapes: definitions, declarations, flows."""
        decls = "".join(
            f"    {name}:◯ □ {' '.join(str(d) for d in shape)}\n" for name, shape in shapes.items()
        )
        body = "".join(f"    {line.strip()}\n" for line in self.flows.splitlines() if line.strip())
        head = f"{self.definitions}\n\n" if self.definitions else ""
        return f"{head}{{\n{decls}{body}}}\n"

    def _key(self, program):
        digest = hashlib.sha256()
        digest.update(program.encode("utf-8"))
        digest.update(repr((self.tau, self.max_iter, self.threads, self.precision, self.portable)).encode())
        digest.update(_compiler_stamp().encode("utf-8"))
        return digest.hexdigest()[:24]

    def engine(self, **shapes):
        """The compiled kernel for these shapes, compiling it if no cache has it."""
        program = self.program({name: tuple(shape) for name, shape in shapes.items()})
        key = self._key(program)
        engine = self._engines.get(key)
        if engine is not None:
            return engine
        os.makedirs(self.cache_dir, exist_ok=True)
        so_path = os.path.join(self.cache_dir, f"{key}.so")
        if not os.path.exists(so_path):
            source = os.path.join(self.cache_dir, f"{key}.rho")
            with open(source, "w", encoding="utf-8") as f:
                f.write(program)
            # Built under a private name and moved into place, so two
            # processes racing for the same kernel never see half a file.
            scratch = RhoEngine(os.path.join(self.cache_dir, f"{key}.{os.getpid()}.building.so"))
            scratch.compile_rho_file(
                source, tau=self.tau, max_iter=self.max_iter, threads=self.threads,
                portable=self.portable, precision=self.precision, load=False,
            )
            os.replace(scratch.kernel_so_path, so_path)
            self.compiled += 1
        engine = RhoEngine(so_path)
        engine._load_library()
        self._engines[key] = engine
        return engine

    def __call__(self, **spaces):
        ctype = self._ctype
        shapes, buffers, alive = {}, {}, []
        for name, value in spaces.items():
            shape, buf, keep = _as_space(value, ctype)
            shapes[name] = shape
            buffers[name] = buf
            alive.append(keep)
        engine = self.engine(**shapes)
        produced = {}
        for name, shape, role in engine.spaces():
            if role == "output" and name not in buffers:
                cells = 1
                for d in shape:
                    cells *= d
                buf = (ctype * max(cells, 1))()
                buffers[name] = buf
                produced[name] = (buf, tuple(shape))
        engine.execute_spaces(buffers)
        self._last = engine
        results = {name: _from_space(buf, shape, ctype) for name, (buf, shape) in produced.items()}
        if len(results) == 1:
            return next(iter(results.values()))
        return results

    def sweeps(self):
        """Sweeps the `⇒` loops of the most recent call took."""
        return self._last.sweeps() if self._last is not None else 0

    def converged(self):
        """Whether every `⇒` of the most recent call stopped on the tolerance."""
        return self._last.converged() if self._last is not None else True
