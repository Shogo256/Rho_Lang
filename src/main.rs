use clap::Parser;
use rho_lang::codegen::LlvmCodeGen;
use rho_lang::dag::RhoDag;
use rho_lang::parser::parse_rho_program;
use rho_lang::solver::ConstraintSolver;
use rho_lang::tla::generate_tla_spec;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "rhoc")]
#[command(about = "ρ (RHO) Language Compiler - Time-eliminated topological dataflow driver", long_about = None)]
struct Args {
    /// Input .rho source file
    #[arg(required = true)]
    input: PathBuf,

    /// Output shared library (.so) path
    #[arg(short, long, default_value = "libkernel.so")]
    output: PathBuf,

    /// Dump TLA+ formal specification (.tla)
    #[arg(long)]
    dump_tla: bool,

    /// Display Active Audit DAG Trace ($)
    #[arg(long)]
    dump_dag: bool,

    /// Display generated LLVM 22 IR
    #[arg(long)]
    dump_llvm: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=====================================================");
    println!("  ρ (RHO) Language Compiler v1.0 [LLVM 22 / Rust]");
    println!("=====================================================");

    let source_code = fs::read_to_string(&args.input)
        .map_err(|e| anyhow::anyhow!("Failed to read source file: {}", e))?;

    // 1. 20-Symbol Strict Lexer & Parser
    println!("[Phase 1] Parsing & 20-Symbol Strict Tokenizer...");
    let block = parse_rho_program(&source_code)?;
    println!("  └─ Parsing completed successfully (OK)");

    // 2. DAG Construction & Audit Tracer & TLA+
    println!("[Phase 2] DAG Transformation & TLA+ Auto-Generation...");
    let dag = RhoDag::build(&block)?;

    if args.dump_dag {
        println!("{}", dag.print_audit_trace());
    }

    let tla_code = generate_tla_spec("rho_harmony", &block);
    fs::write("rho_harmony.tla", &tla_code)?;
    println!("  └─ TLA+ Specification written to 'rho_harmony.tla'");

    // 3. Static Constraint Solver !
    println!("[Phase 3] Static Constraint Solver (!) Validation...");
    ConstraintSolver::verify_constraints(&block)?;
    println!("  └─ Static Constraint Check Passed (OK)");

    // 4. LLVM 22 CodeGen & Native Compilation
    println!("[Phase 4] LLVM 22 Hardware Mapping & Vector Shift Engine...");
    let mut codegen = LlvmCodeGen::new("rho_kernel");
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
    println!("=====================================================");

    Ok(())
}
