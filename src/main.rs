// SPDX-License-Identifier: Apache-2.0
use rho_lang::codegen::LlvmCodeGen;
use rho_lang::dag::RhoDag;
use rho_lang::parser::parse_rho_program;
use rho_lang::solver::{ConstraintSolver, Verdict};
use std::fs;
use std::path::{Path, PathBuf};

/// The glyph, as the terminal draws it. Shown by `rhoc` on its own, at the
/// top of `--help`, and at the start of every compile.
/// Errors as `run` returns them: a message, or a disruption that knows its
/// line, told apart by downcasting.
type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Return the formatted message as an error.
macro_rules! bail {
    ($($t:tt)*) => {
        return Err(format!($($t)*).into())
    };
}

const BANNER: &str = r#"
   _ __
  | '_ \      ρ (RHO)
  | |_) |     an array language in the line of APL
  | .__/      one source → a native kernel, a JAX module, a circuit
  |_|
"#;

fn banner() -> String {
    BANNER.replace("ρ (RHO)", concat!("ρ (RHO) ", env!("CARGO_PKG_VERSION")))
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    tau: f64,
    bind: Vec<String>,
    dump_dag: bool,
    f32: bool,
    max_iter: Option<usize>,
    no_simd: bool,
    threads: usize,
    portable: bool,
    dump_llvm: bool,
    emit_static: Option<PathBuf>,
    symbol_prefix: Option<String>,
    emit_lean: Option<PathBuf>,
    lean_namespace: Option<String>,
    lean_program_only: bool,
    emit_sv: Option<PathBuf>,
    fuse: bool,
    fixed: Option<String>,
    emit_jax: Option<PathBuf>,
    run: Vec<String>,
    write: Vec<String>,
    constant_time: bool,
    explain: bool,
    public: Vec<String>,
}

/// The flags, as `--help` lists them.
const OPTIONS: &str = r#"  -o, --output <PATH>
          Output shared library (.so) path [default: libkernel.so]
      --tau <VALUE>
          Value bound to the threshold symbol 𝜏 [default: 0.0]
      --bind <NAME=ADDR>
          Bind a space to a memory address for the zero-copy entrypoint, e.g. --bind INPUT=0x7f2c00000000. Repeatable; overrides &[0x...].
      --dump-dag
          Display Active Audit DAG Trace ($)
      --f32
          Compute at single precision. Halves the memory traffic these kernels are bound by, and widens the doubt every proof carries.
      --max-iter <N>
          The most sweeps a `⇒` may take before it stops regardless of 𝜏. Required by a program that iterates: the cap is what makes the kernel terminate, and it is part of what the kernel computes.
      --no-simd
          Emit only scalar loops, skipping vector lowering
      --threads <N>
          Threads every sweep is split across; 0 takes one per CPU. The kernel still honours RHO_THREADS in the environment when it runs. [default: 0]
      --portable
          Compile for any x86-64 rather than for this machine. By default the kernel uses every vector width the machine it is built on has.
      --dump-llvm
          Display generated LLVM IR
      --emit-static <LIB.a>
          Also write the kernel as a static library (`.a`) to link into a program, its symbols renamed by --symbol-prefix.
      --symbol-prefix <PREFIX>
          With --emit-static, what every exported symbol starts with in place of `rho_kernel` (`PREFIX_exec_spaces`, `PREFIX_metadata`, ...), so that several kernels link into one program.
      --emit-lean <FILE.lean>
          Write the source's functions as Lean 4 definitions over the integers, for proofs about them, and stop: nothing is compiled, and a source of definitions alone is accepted. A function outside the whole-number subset is named in a comment and left out.
      --lean-namespace <NAME>
          With --emit-lean, the namespace of the definitions (default `Rho`).
      --lean-program-only
          With --emit-lean, write the program block alone (as `kernel`), not the functions: its calls then name the functions of an enclosing namespace, written once from the definitions' own file.
      --emit-sv <DIR>
          Also write the program as a SystemVerilog streaming pipeline (rho_kernel.sv and a Verilator harness) into this directory.
      --fuse
          With --emit-sv, fuse every intermediate one flow reads cell for cell into that flow's stage, as the kernel does: fewer stages, counters, registers and replay memories, and a longer path through a stage, so a slower clock.
      --fixed <W.F>
          Make the circuit's cells fixed point, WIDTH.FRAC bits (e.g. 32.16), which Yosys synthesises; without it they are `real`, for simulation.
      --emit-jax <FILE.py>
          Also write the program as a JAX module (a `rho(...)` function over jax arrays, and `rho_jit`), for CPU, GPU and TPU through XLA.
      --run <SPACE=FILE>
          Run the compiled kernel once, with this input space read from a file: raw little-endian doubles when the file ends in .bin, numbers separated by whitespace otherwise. Repeat for every input. OUTPUT is printed as text unless --write names a file for it.
      --write <SPACE=FILE>
          After --run, write this space to a file (.bin for raw doubles, text otherwise, one line per row). Any space a flow writes may be named, intermediates included. Repeatable.
      --constant-time
          Refuse the program unless it keeps the rules for secret cells: no division or residue by a non-constant, no sqrt, exp, log, sin, cos or fractional power, no `⇒`, and `⌷` only at positions made of `⍳`, constants and --public spaces.
      --explain
          Say what the compiler made of each flow: the sweeps a call runs, vector or one cell at a time and why, the folds computed as passes of their own, which intermediates fused into which sweep, what a `⇒` runs every round, and, with --emit-sv, the circuit's stages, memories and multipliers.
      --public <NAME,...>
          Spaces that hold no secret, such as tables, for --constant-time. Repeat, or separate with commas.
  -h, --help
          Print help
  -V, --version
          Print version"#;

fn usage() -> String {
    format!("{}\nCompile a ρ program to a native kernel, and run it, or write it as JAX or a circuit\n\nUsage: rhoc [OPTIONS] <INPUT>\n\nArguments:\n  <INPUT>\n          Input .rho source file\n\nOptions:\n{OPTIONS}\n", banner())
}

/// The command line, read by hand: `--flag VALUE`, `--flag=VALUE`, `-o
/// VALUE`, and the input file as the one positional argument. A fault comes
/// back as a message for `main` to print.
fn parse_args(raw: impl Iterator<Item = String>) -> Result<Args, String> {
    // The value of a flag: after `=`, else the next argument.
    fn value(flag: &str, inline: &mut Option<String>, it: &mut impl Iterator<Item = String>) -> Result<String, String> {
        inline.take().or_else(|| it.next()).ok_or_else(|| format!("{flag} needs a value"))
    }
    fn number<T: std::str::FromStr>(flag: &str, v: String) -> Result<T, String> {
        v.parse().map_err(|_| format!("{flag}: '{v}' is not a number"))
    }
    let mut args = Args {
        input: PathBuf::new(),
        output: PathBuf::from("libkernel.so"),
        tau: 0.0,
        bind: Vec::new(),
        dump_dag: false,
        f32: false,
        max_iter: None,
        no_simd: false,
        threads: 0,
        portable: false,
        dump_llvm: false,
        emit_static: None,
        symbol_prefix: None,
        emit_lean: None,
        lean_namespace: None,
        lean_program_only: false,
        emit_sv: None,
        fuse: false,
        fixed: None,
        emit_jax: None,
        run: Vec::new(),
        write: Vec::new(),
        constant_time: false,
        explain: false,
        public: Vec::new(),
    };
    let mut input = None;
    let mut it = raw.skip(1);
    while let Some(arg) = it.next() {
        if !arg.starts_with('-') || arg == "-" {
            if input.replace(PathBuf::from(&arg)).is_some() {
                return Err(format!("unexpected argument '{arg}': one input file only"));
            }
            continue;
        }
        let (flag, mut inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("rhoc {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--output" | "-o" => args.output = PathBuf::from(value(&flag, &mut inline, &mut it)?),
            "--tau" => args.tau = number(&flag, value(&flag, &mut inline, &mut it)?)?,
            "--bind" => args.bind.push(value(&flag, &mut inline, &mut it)?),
            "--dump-dag" => args.dump_dag = true,
            "--f32" => args.f32 = true,
            "--max-iter" => args.max_iter = Some(number(&flag, value(&flag, &mut inline, &mut it)?)?),
            "--no-simd" => args.no_simd = true,
            "--threads" => args.threads = number(&flag, value(&flag, &mut inline, &mut it)?)?,
            "--portable" => args.portable = true,
            "--dump-llvm" => args.dump_llvm = true,
            "--emit-static" => args.emit_static = Some(PathBuf::from(value(&flag, &mut inline, &mut it)?)),
            "--symbol-prefix" => args.symbol_prefix = Some(value(&flag, &mut inline, &mut it)?),
            "--emit-lean" => args.emit_lean = Some(PathBuf::from(value(&flag, &mut inline, &mut it)?)),
            "--lean-namespace" => args.lean_namespace = Some(value(&flag, &mut inline, &mut it)?),
            "--lean-program-only" => args.lean_program_only = true,
            "--emit-sv" => args.emit_sv = Some(PathBuf::from(value(&flag, &mut inline, &mut it)?)),
            "--fuse" => args.fuse = true,
            "--fixed" => args.fixed = Some(value(&flag, &mut inline, &mut it)?),
            "--emit-jax" => args.emit_jax = Some(PathBuf::from(value(&flag, &mut inline, &mut it)?)),
            "--run" => args.run.push(value(&flag, &mut inline, &mut it)?),
            "--write" => args.write.push(value(&flag, &mut inline, &mut it)?),
            "--constant-time" => args.constant_time = true,
            "--explain" => args.explain = true,
            "--public" => args.public.extend(value(&flag, &mut inline, &mut it)?.split(',').map(String::from)),
            _ => return Err(format!("unexpected argument '{arg}'; rhoc --help lists the flags")),
        }
        if inline.is_some() {
            return Err(format!("{flag} takes no value"));
        }
    }
    args.input = input.ok_or("the input .rho file is required")?;
    if args.emit_static.is_some() && args.symbol_prefix.is_none() {
        return Err("--emit-static needs --symbol-prefix".to_string());
    }
    if args.lean_namespace.is_some() && args.emit_lean.is_none() {
        return Err("--lean-namespace needs --emit-lean".to_string());
    }
    if args.lean_program_only && args.emit_lean.is_none() {
        return Err("--lean-program-only needs --emit-lean".to_string());
    }
    Ok(args)
}

fn main() {
    // `rhoc` alone: the glyph and the one line that gets someone started.
    if std::env::args_os().len() == 1 {
        print!("{}", banner());
        println!("  usage: rhoc program.rho [--run INPUT=file] [--emit-jax m.py] [--emit-sv dir]");
        println!("         rhoc --help for every flag");
        println!();
        return;
    }
    let args = parse_args(std::env::args()).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    match run(&args) {
        Ok(()) => {}
        Err(err) => {
            // A parse or lowering failure knows its line; show it under a caret.
            if let Some(disruption) = err.downcast_ref::<rho_lang::error::HarmonyDisruption>() {
                if let Ok(source) = fs::read_to_string(&args.input) {
                    eprintln!("{}", disruption.render(&source));
                    std::process::exit(1);
                }
            }
            eprintln!("Error: {err}");
            std::process::exit(1);
        }
    }
}

fn fmt_bound(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else if v > 0.0 {
        "+inf".to_string()
    } else {
        "-inf".to_string()
    }
}

fn run(args: &Args) -> AnyResult<()> {

    print!("{}", banner());

    let source_code = fs::read_to_string(&args.input)
        .map_err(|e| format!("Failed to read source file: {}", e))?;

    if let Some(path) = &args.emit_lean {
        let (functions, statements) = rho_lang::parser::parse_unexpanded(&source_code)?;
        let namespace = args.lean_namespace.as_deref().unwrap_or("Rho");
        let source = args.input.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let text = rho_lang::codegen::lean::emit_with(&functions, &statements, namespace, &source, args.lean_program_only)?;
        let left_out = text.matches("` (line ").count();
        let program = if text.contains("def «kernel»") {
            ", the program as `kernel`"
        } else if statements.is_empty() {
            ""
        } else {
            ", the program left out"
        };
        fs::write(path, text)?;
        println!(
            "  └─ Lean definitions emitted to -> {} ({} functions, {left_out} left out{program})",
            path.display(),
            if args.lean_program_only { 0 } else { functions.len() - left_out }
        );
        return Ok(());
    }

    // 1. 20-Symbol Strict Lexer & Parser
    println!("[Phase 1] Parsing & 20-Symbol Strict Tokenizer...");
    let block = parse_rho_program(&source_code)?;
    println!("  └─ Parsing completed successfully (OK)");

    if args.constant_time {
        let public: std::collections::BTreeSet<String> = args.public.iter().cloned().collect();
        let breaches = rho_lang::consttime::check(&block, &public);
        if !breaches.is_empty() {
            for b in &breaches {
                eprintln!("  line {}: {}", b.line, b.detail);
            }
            bail!("not constant time: {} place(s) above", breaches.len());
        }
        println!("  └─ Constant time: no data-dependent latency, rounds or addresses (OK)");
    }

    // 2. DAG Construction & Audit Tracer
    println!("[Phase 2] DAG Transformation...");
    let dag = RhoDag::build(&block)?;

    if args.dump_dag {
        println!("{}", dag.print_audit_trace());
    }


    // 3. Static Constraint Solver !
    println!("[Phase 3] Static Constraint Solver (!) Validation...");
    let precision = if args.f32 {
        rho_lang::numeric::Precision::F32
    } else {
        rho_lang::numeric::Precision::F64
    };
    let report = ConstraintSolver::verify_at(&block, args.tau, precision)
        .map_err(|e| block.attribute(e))?;
    println!("  └─ Backend: {} at {precision}", report.backend);
    let mut open_questions = 0;
    for finding in report
        .constraints
        .iter()
        .chain(report.divisions.iter())
        .chain(report.domains.iter())
    {
        match &finding.verdict {
            Verdict::Proved => println!("  └─ [proved]   {}", finding.subject),
            Verdict::Unproven(why) => {
                open_questions += 1;
                println!("  └─ [unproven] {} — {why}", finding.subject);
            }
            // A violation aborts in verify(), so this arm is unreachable.
            Verdict::Violated(why) => println!("  └─ [VIOLATED] {} — {why}", finding.subject),
        }
    }
    if open_questions == 0 {
        println!("  └─ Static Constraint Check Passed (OK)");
    } else {
        println!("  └─ Passed with {open_questions} unproven obligation(s)");
    }
    println!(
        "  └─ Output ∈ [{}, {}]",
        fmt_bound(report.output_range.lo),
        fmt_bound(report.output_range.hi)
    );

    // 4. CodeGen & Native Compilation
    println!("[Phase 4] LLVM Hardware Mapping...");
    let mut codegen = LlvmCodeGen::new("rho_kernel")
        .with_tau(args.tau)
        .with_precision(precision);
    if args.no_simd {
        codegen = codegen.without_simd();
    }
    codegen = codegen.with_threads(args.threads);
    if args.portable {
        codegen = codegen.portable();
    }
    let iterates = block
        .statements
        .iter()
        .any(|s| matches!(s, rho_lang::ast::Statement::Iterate { .. }));
    match args.max_iter {
        Some(0) => bail!("--max-iter must be at least 1: a ⇒ always sweeps once"),
        Some(cap) => codegen = codegen.with_max_sweeps(cap),
        None if iterates => bail!(
            "this program iterates (⇒); say how many sweeps it may take with --max-iter N"
        ),
        None => {}
    }
    for entry in &args.bind {
        let (name, addr) = entry.split_once('=').ok_or_else(|| {
            format!("--bind expects NAME=ADDR, got '{entry}'")
        })?;
        let digits = addr.trim().trim_start_matches("0x").trim_start_matches("0X");
        let radix = if addr.trim() == digits { 10 } else { 16 };
        let value = u64::from_str_radix(digits, radix)
            .map_err(|e| format!("--bind address '{addr}' is not a number: {e}"))?;
        codegen = codegen.bind(name.trim(), value);
    }
    let llvm_ir = codegen.generate_llvm_ir(&block).map_err(|e| block.attribute(e))?;
    if args.explain {
        println!("[Explain] What each flow became in the kernel");
        print!("{}", codegen.explain());
    }

    if args.dump_llvm {
        println!("-----------------------------------------------------");
        println!("{}", llvm_ir);
        println!("-----------------------------------------------------");
    }

    let out_path = args.output.to_string_lossy();
    codegen.compile_to_so(&llvm_ir, &out_path)?;
    println!("  └─ Compilation Successful: Binary emitted to -> {}", out_path);
    if let (Some(lib), Some(prefix)) = (&args.emit_static, &args.symbol_prefix) {
        codegen.compile_to_static(&llvm_ir, &lib.to_string_lossy(), prefix)?;
        println!("  └─ Static library emitted to -> {} (symbols {prefix}_*)", lib.display());
    }

    if let Some(path) = &args.emit_jax {
        let module = rho_lang::codegen::jax::emit(&block, args.tau, args.max_iter, precision)
            .map_err(|e| block.attribute(e))?;
        let mut with_source = module;
        with_source.push_str("\n# ---- the program this was made from\n");
        for line in source_code.lines() {
            with_source.push_str("# ");
            with_source.push_str(line);
            with_source.push('\n');
        }
        std::fs::write(path, with_source)?;
        println!("  └─ JAX module emitted to -> {}", path.display());
    }

    if let Some(dir) = &args.emit_sv {
        let numbers = match &args.fixed {
            None => rho_lang::codegen::sv::Numbers::Real,
            Some(spec) => {
                let (w, f) = spec
                    .split_once('.')
                    .ok_or_else(|| format!("--fixed expects WIDTH.FRAC, such as 32.16, got '{spec}'"))?;
                rho_lang::codegen::sv::Numbers::Fixed {
                    width: w.parse().map_err(|_| format!("--fixed width '{w}' is not a number"))?,
                    frac: f.parse().map_err(|_| format!("--fixed fraction '{f}' is not a number"))?,
                }
            }
        };
        let emit = if args.fuse { rho_lang::codegen::sv::emit_fused } else { rho_lang::codegen::sv::emit };
        let circuit = emit(&block, args.tau, args.max_iter, numbers).map_err(|e| block.attribute(e))?;
        rho_lang::codegen::sv::write(&circuit, dir)?;
        println!(
            "  └─ Circuit emitted to -> {}/rho_kernel.sv (one cell per clock, latency {} clocks)",
            dir.display(),
            circuit.latency
        );
        if args.explain {
            println!("[Explain] What the circuit is made of");
            print!("{}", circuit.explain);
        }
    }

    println!("=====================================================");
    println!("  [SUCCESS] Harmony Achieved: Zero Errors");
    println!(
        "  Callers must supply buffers of {} {} values",
        codegen.element_count(),
        precision
    );
    println!("=====================================================");

    if !args.run.is_empty() || !args.write.is_empty() {
        run_kernel(args, &codegen, &out_path, precision)?;
    }

    Ok(())
}

/// `NAME=FILE` arguments as pairs.
fn name_file(items: &[String], flag: &str) -> AnyResult<Vec<(String, PathBuf)>> {
    items
        .iter()
        .map(|item| {
            let (name, file) = item
                .split_once('=')
                .ok_or_else(|| format!("{flag} expects SPACE=FILE, got '{item}'"))?;
            Ok((name.trim().to_string(), PathBuf::from(file.trim())))
        })
        .collect()
}

/// A space's cells from a file: raw doubles for `.bin`, text otherwise.
fn read_cells(path: &Path, cells: usize, name: &str) -> AnyResult<Vec<f64>> {
    let values: Vec<f64> = if path.extension().is_some_and(|e| e == "bin") {
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if bytes.len() % 8 != 0 {
            bail!("{} holds {} bytes, not a whole number of doubles", path.display(), bytes.len());
        }
        bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()
    } else {
        let text = fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        text.split_whitespace()
            .map(|w| w.parse::<f64>().map_err(|_| format!("{}: '{w}' is not a number", path.display())))
            .collect::<Result<_, String>>()?
    };
    if values.len() != cells {
        bail!(
            "{} holds {} values, but {name} has {cells} cells",
            path.display(),
            values.len()
        );
    }
    Ok(values)
}

/// A space's cells to a file: raw doubles for `.bin`, else text with one
/// line per row (the last axis across).
fn write_cells(path: &Path, shape: &[usize], values: &[f64]) -> AnyResult<()> {
    if path.extension().is_some_and(|e| e == "bin") {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(path, bytes)?;
    } else {
        fs::write(path, format_rows(shape, values))?;
    }
    Ok(())
}

fn format_rows(shape: &[usize], values: &[f64]) -> String {
    let row = shape.last().copied().unwrap_or(1).max(1);
    let mut text = String::new();
    for line in values.chunks(row) {
        let words: Vec<String> = line.iter().map(|v| format!("{v:?}")).collect();
        text.push_str(&words.join(" "));
        text.push('\n');
    }
    text
}

/// Load the kernel just built and run it once over the files named by
/// --run, writing the spaces named by --write (OUTPUT to stdout when no
/// file is named for it). Every input must be given; intermediates the
/// caller does not ask for are the kernel's own.
fn run_kernel(
    args: &Args,
    codegen: &LlvmCodeGen,
    so_path: &str,
    precision: rho_lang::numeric::Precision,
) -> AnyResult<()> {
    let inputs = name_file(&args.run, "--run")?;
    let outputs = name_file(&args.write, "--write")?;
    let shapes = &codegen.space_shapes;
    for (name, _) in inputs.iter().chain(outputs.iter()) {
        if !shapes.contains_key(name) {
            bail!("there is no space named {name}; the program declares {}", shapes.keys().cloned().collect::<Vec<_>>().join(", "));
        }
    }
    let print_output = shapes.contains_key("OUTPUT") && !outputs.iter().any(|(n, _)| n == "OUTPUT");
    // One buffer per space, in the table's order; f32 kernels take floats.
    let wide = precision == rho_lang::numeric::Precision::F64;
    let mut f64_bufs: Vec<Vec<f64>> = Vec::new();
    let mut f32_bufs: Vec<Vec<f32>> = Vec::new();
    let mut given: Vec<bool> = Vec::new();
    for (name, shape) in shapes {
        let cells = shape.iter().product::<usize>().max(1);
        let role = codegen.role_of(name);
        let values: Option<Vec<f64>> = match role {
            "input" => {
                let (_, path) = inputs.iter().find(|(n, _)| n == name).ok_or_else(|| {
                    format!("--run needs a file for the input {name} (shape {shape:?}): --run {name}=FILE")
                })?;
                Some(read_cells(path, cells, name)?)
            }
            _ if outputs.iter().any(|(n, _)| n == name) || (name == "OUTPUT" && print_output) => Some(vec![0.0; cells]),
            _ => None,
        };
        given.push(values.is_some());
        let values = values.unwrap_or_default();
        if wide {
            f64_bufs.push(values);
        } else {
            f32_bufs.push(values.iter().map(|v| *v as f32).collect());
        }
    }
    let table: Vec<*mut std::ffi::c_void> = (0..shapes.len())
        .map(|k| {
            if !given[k] {
                std::ptr::null_mut()
            } else if wide {
                f64_bufs[k].as_mut_ptr() as *mut std::ffi::c_void
            } else {
                f32_bufs[k].as_mut_ptr() as *mut std::ffi::c_void
            }
        })
        .collect();
    // dlopen searches the library path for a bare name; the kernel is a file.
    let so_file = fs::canonicalize(so_path).unwrap_or_else(|_| PathBuf::from(so_path));
    let lib = rho_lang::dl::Library::open(&so_file).map_err(|e| format!("cannot load {}: {e}", so_file.display()))?;
    unsafe {
        let exec: unsafe extern "C" fn(*const *mut std::ffi::c_void) = lib.get("rho_kernel_exec_spaces")?;
        exec(table.as_ptr());
        let sweeps: unsafe extern "C" fn() -> i64 = lib.get("rho_kernel_sweeps")?;
        let converged: unsafe extern "C" fn() -> i64 = lib.get("rho_kernel_converged")?;
        if codegen.iterates() {
            eprintln!("sweeps {} converged {}", sweeps(), if converged() != 0 { "yes" } else { "no" });
        }
    }
    let cells_of = |k: usize| -> Vec<f64> {
        if wide {
            f64_bufs[k].clone()
        } else {
            f32_bufs[k].iter().map(|v| *v as f64).collect()
        }
    };
    for (name, path) in &outputs {
        let k = shapes.keys().position(|n| n == name).unwrap();
        write_cells(path, &shapes[name], &cells_of(k))?;
        eprintln!("{name} -> {}", path.display());
    }
    if print_output {
        let k = shapes.keys().position(|n| n == "OUTPUT").unwrap();
        print!("{}", format_rows(&shapes["OUTPUT"], &cells_of(k)));
    }
    Ok(())
}
