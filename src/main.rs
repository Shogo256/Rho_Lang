use clap::Parser;
use rho_lang::codegen::LlvmCodeGen;
use rho_lang::dag::RhoDag;
use rho_lang::parser::parse_rho_program;
use rho_lang::solver::{ConstraintSolver, Verdict};
use std::fs;
use std::path::PathBuf;

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

    /// Refuse to emit a kernel whose contract has unproven obligations
    #[arg(long)]
    require_contract: bool,

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

    /// Display generated LLVM IR
    #[arg(long)]
    dump_llvm: bool,
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
    let report = ConstraintSolver::verify_at(&block, args.tau, precision)?;
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
    // Convergence is reported on its own: a loop that is not shown to
    // converge still ends, at its cap, so nothing unsafe follows.
    for (finding, claim) in report.iterations.iter().zip(&report.contract.iterations) {
        let range = format!(
            "[{}, {}]",
            fmt_bound(claim.invariant.lo),
            fmt_bound(claim.invariant.hi)
        );
        match (&finding.verdict, claim.factor) {
            (Verdict::Proved, Some(c)) => println!(
                "  └─ [proved]   ⇒ {} converges: each sweep contracts by {c} (∞-norm), iterates stay in {range}",
                claim.target
            ),
            (Verdict::Unproven(why), _) => println!(
                "  └─ [open]     ⇒ {} convergence not shown — {why}; iterates stay in {range}",
                claim.target
            ),
            (other, _) => println!("  └─ [{other:?}] {}", finding.subject),
        }
    }

    let contract = &report.contract;
    println!(
        "  └─ Contract: output ∈ [{}, {}], divisions {}",
        fmt_bound(contract.output_range.lo),
        fmt_bound(contract.output_range.hi),
        if contract.divisions_proven_safe {
            "proven safe"
        } else {
            "not proven safe"
        }
    );
    if args.require_contract && !contract.is_complete() {
        anyhow::bail!(
            "--require-contract: {} obligation(s) unproven and divisions {}. \
             Refusing to emit a kernel whose contract is incomplete.",
            contract.open_obligations,
            if contract.divisions_proven_safe { "safe" } else { "unproven" }
        );
    }

    // 4. CodeGen & Native Compilation
    println!("[Phase 4] LLVM Hardware Mapping...");
    let mut codegen = LlvmCodeGen::new("rho_kernel")
        .with_tau(args.tau)
        .with_precision(precision)
        .with_contract(report.contract.to_json());
    if args.no_simd {
        codegen = codegen.without_simd();
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
    let llvm_ir = codegen.generate_llvm_ir(&block)?;

    if args.dump_llvm {
        println!("-----------------------------------------------------");
        println!("{}", llvm_ir);
        println!("-----------------------------------------------------");
    }

    let out_path = args.output.to_string_lossy();
    codegen.compile_to_so(&llvm_ir, &out_path)?;
    println!("  └─ Compilation Successful: Binary emitted to -> {}", out_path);

    println!("=====================================================");
    println!("  [SUCCESS] Harmony Achieved: Zero Errors");
    println!(
        "  Callers must supply buffers of {} {} values",
        codegen.element_count(),
        precision
    );
    println!("=====================================================");

    Ok(())
}
