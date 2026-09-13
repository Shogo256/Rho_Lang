use clap::Parser;
use rho_lang::codegen::LlvmCodeGen;
use rho_lang::dag::RhoDag;
use rho_lang::parser::parse_rho_program;
use rho_lang::solver::{ConstraintSolver, Verdict};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "rhoc")]
#[command(version)]
#[command(about = "ρ (RHO) Language Compiler - Time-eliminated topological dataflow driver", long_about = None)]
struct Args {
    /// Input .rho source file
    #[arg(required = true)]
    input: PathBuf,

    /// Output shared library (.so) path
    #[arg(short, long, default_value = "libkernel.so")]
    output: PathBuf,

    /// Value bound to the threshold symbol 𝜏
    #[arg(long, default_value_t = 0.0)]
    tau: f64,

    /// Bind a space to a memory address for the zero-copy entrypoint,
    /// e.g. --bind INPUT=0x7f2c00000000. Repeatable; overrides &[0x...].
    #[arg(long, value_name = "NAME=ADDR")]
    bind: Vec<String>,

    /// Display Active Audit DAG Trace ($)
    #[arg(long)]
    dump_dag: bool,

    /// Compute at single precision. Halves the memory traffic these kernels
    /// are bound by, and widens the doubt every proof carries.
    #[arg(long)]
    f32: bool,

    /// The most sweeps a `⇒` may take before it stops regardless of 𝜏.
    /// Required by a program that iterates: the cap is what makes the kernel
    /// terminate, and it is part of what the kernel computes.
    #[arg(long, value_name = "N")]
    max_iter: Option<usize>,

    /// Emit only scalar loops, skipping vector lowering
    #[arg(long)]
    no_simd: bool,

    /// Threads every sweep is split across; 0 takes one per CPU. The kernel
    /// still honours RHO_THREADS in the environment when it runs.
    #[arg(long, value_name = "N", default_value_t = 0)]
    threads: usize,

    /// Compile for any x86-64 rather than for this machine. By default the
    /// kernel uses every vector width the machine it is built on has.
    #[arg(long)]
    portable: bool,

    /// Display generated LLVM IR
    #[arg(long)]
    dump_llvm: bool,

    /// Also write the program as a SystemVerilog streaming pipeline
    /// (rho_kernel.sv and a Verilator harness) into this directory.
    #[arg(long, value_name = "DIR")]
    emit_sv: Option<PathBuf>,

    /// Make the circuit's cells fixed point, WIDTH.FRAC bits (e.g. 32.16),
    /// which Yosys synthesises; without it they are `real`, for simulation.
    #[arg(long, value_name = "W.F")]
    fixed: Option<String>,

    /// Also write the program as a JAX module (a `rho(...)` function over
    /// jax arrays, and `rho_jit`), for CPU, GPU and TPU through XLA.
    #[arg(long, value_name = "FILE.py")]
    emit_jax: Option<PathBuf>,

    /// Run the compiled kernel once, with this input space read from a file:
    /// raw little-endian doubles when the file ends in .bin, numbers
    /// separated by whitespace otherwise. Repeat for every input. OUTPUT is
    /// printed as text unless --write names a file for it.
    #[arg(long, value_name = "SPACE=FILE")]
    run: Vec<String>,

    /// After --run, write this space to a file (.bin for raw doubles, text
    /// otherwise, one line per row). Any space a flow writes may be named,
    /// intermediates included. Repeatable.
    #[arg(long, value_name = "SPACE=FILE")]
    write: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => Ok(()),
        Err(err) => {
            // A parse or lowering failure knows its line; show it under a caret.
            if let Some(disruption) = err.downcast_ref::<rho_lang::error::HarmonyDisruption>() {
                if let Ok(source) = fs::read_to_string(&args.input) {
                    eprintln!("{}", disruption.render(&source));
                    std::process::exit(1);
                }
            }
            Err(err)
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

fn run(args: &Args) -> anyhow::Result<()> {

    println!("=====================================================");
    println!("  ρ (RHO) Language Compiler v{}", env!("CARGO_PKG_VERSION"));
    println!("=====================================================");

    let source_code = fs::read_to_string(&args.input)
        .map_err(|e| anyhow::anyhow!("Failed to read source file: {}", e))?;

    // 1. 20-Symbol Strict Lexer & Parser
    println!("[Phase 1] Parsing & 20-Symbol Strict Tokenizer...");
    let block = parse_rho_program(&source_code)?;
    println!("  └─ Parsing completed successfully (OK)");

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
        Some(0) => anyhow::bail!("--max-iter must be at least 1: a ⇒ always sweeps once"),
        Some(cap) => codegen = codegen.with_max_sweeps(cap),
        None if iterates => anyhow::bail!(
            "this program iterates (⇒); say how many sweeps it may take with --max-iter N"
        ),
        None => {}
    }
    for entry in &args.bind {
        let (name, addr) = entry.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("--bind expects NAME=ADDR, got '{entry}'")
        })?;
        let digits = addr.trim().trim_start_matches("0x").trim_start_matches("0X");
        let radix = if addr.trim() == digits { 10 } else { 16 };
        let value = u64::from_str_radix(digits, radix)
            .map_err(|e| anyhow::anyhow!("--bind address '{addr}' is not a number: {e}"))?;
        codegen = codegen.bind(name.trim(), value);
    }
    let llvm_ir = codegen.generate_llvm_ir(&block).map_err(|e| block.attribute(e))?;

    if args.dump_llvm {
        println!("-----------------------------------------------------");
        println!("{}", llvm_ir);
        println!("-----------------------------------------------------");
    }

    let out_path = args.output.to_string_lossy();
    codegen.compile_to_so(&llvm_ir, &out_path)?;
    println!("  └─ Compilation Successful: Binary emitted to -> {}", out_path);

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
                    .ok_or_else(|| anyhow::anyhow!("--fixed expects WIDTH.FRAC, such as 32.16, got '{spec}'"))?;
                rho_lang::codegen::sv::Numbers::Fixed {
                    width: w.parse().map_err(|_| anyhow::anyhow!("--fixed width '{w}' is not a number"))?,
                    frac: f.parse().map_err(|_| anyhow::anyhow!("--fixed fraction '{f}' is not a number"))?,
                }
            }
        };
        let circuit = rho_lang::codegen::sv::emit(&block, args.tau, args.max_iter, numbers)
            .map_err(|e| block.attribute(e))?;
        rho_lang::codegen::sv::write(&circuit, dir)?;
        println!(
            "  └─ Circuit emitted to -> {}/rho_kernel.sv (one cell per clock, latency {} clocks)",
            dir.display(),
            circuit.latency
        );
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
fn name_file(items: &[String], flag: &str) -> anyhow::Result<Vec<(String, PathBuf)>> {
    items
        .iter()
        .map(|item| {
            let (name, file) = item
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("{flag} expects SPACE=FILE, got '{item}'"))?;
            Ok((name.trim().to_string(), PathBuf::from(file.trim())))
        })
        .collect()
}

/// A space's cells from a file: raw doubles for `.bin`, text otherwise.
fn read_cells(path: &Path, cells: usize, name: &str) -> anyhow::Result<Vec<f64>> {
    let values: Vec<f64> = if path.extension().is_some_and(|e| e == "bin") {
        let bytes = fs::read(path).map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        if bytes.len() % 8 != 0 {
            anyhow::bail!("{} holds {} bytes, not a whole number of doubles", path.display(), bytes.len());
        }
        bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()
    } else {
        let text = fs::read_to_string(path).map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        text.split_whitespace()
            .map(|w| w.parse::<f64>().map_err(|_| anyhow::anyhow!("{}: '{w}' is not a number", path.display())))
            .collect::<anyhow::Result<_>>()?
    };
    if values.len() != cells {
        anyhow::bail!(
            "{} holds {} values, but {name} has {cells} cells",
            path.display(),
            values.len()
        );
    }
    Ok(values)
}

/// A space's cells to a file: raw doubles for `.bin`, else text with one
/// line per row (the last axis across).
fn write_cells(path: &Path, shape: &[usize], values: &[f64]) -> anyhow::Result<()> {
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
) -> anyhow::Result<()> {
    let inputs = name_file(&args.run, "--run")?;
    let outputs = name_file(&args.write, "--write")?;
    let shapes = &codegen.space_shapes;
    for (name, _) in inputs.iter().chain(outputs.iter()) {
        if !shapes.contains_key(name) {
            anyhow::bail!("there is no space named {name}; the program declares {}", shapes.keys().cloned().collect::<Vec<_>>().join(", "));
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
                    anyhow::anyhow!("--run needs a file for the input {name} (shape {shape:?}): --run {name}=FILE")
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
    let lib = unsafe { libloading::Library::new(&so_file) }
        .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", so_file.display()))?;
    unsafe {
        let exec: libloading::Symbol<unsafe extern "C" fn(*const *mut std::ffi::c_void)> =
            lib.get(b"rho_kernel_exec_spaces").map_err(|e| anyhow::anyhow!("{e}"))?;
        exec(table.as_ptr());
        let sweeps: libloading::Symbol<unsafe extern "C" fn() -> i64> = lib.get(b"rho_kernel_sweeps").map_err(|e| anyhow::anyhow!("{e}"))?;
        let converged: libloading::Symbol<unsafe extern "C" fn() -> i64> = lib.get(b"rho_kernel_converged").map_err(|e| anyhow::anyhow!("{e}"))?;
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
