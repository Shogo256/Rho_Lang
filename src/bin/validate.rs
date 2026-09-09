//! Prove that the emitted IR computes what the source means, for every input.
//!
//!     cargo run --release --features z3-solver --bin validate -- [seed] [rounds]

use rho_lang::parser::parse_rho_program;
use rho_lang::validate::{validate, Verdict};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20260909);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    // Shapes small enough that a proof finishes, large enough to exercise the
    // head, the vector body and the tail of a sweep.
    let shapes: [(&str, Vec<usize>); 4] = [
        ("6 1", vec![6, 1]),
        ("8 1", vec![8, 1]),
        ("2 3", vec![2, 3]),
        ("3 4", vec![3, 4]),
    ];

    let programs: [&str; 10] = [
        "(INPUT + 1.0) → OUTPUT\n    OUTPUT → =",
        "(▷INPUT - INPUT) → OUTPUT\n    OUTPUT → =",
        "(▽INPUT) → OUTPUT\n    OUTPUT → =",
        "(▷0INPUT + INPUT) → OUTPUT\n    OUTPUT → =",
        "(INPUT > 1.0) → OUTPUT\n    OUTPUT → =",
        "((INPUT ^ 2.0) - INPUT) → OUTPUT\n    OUTPUT → =",
        "◇+ INPUT → OUTPUT\n    OUTPUT → =",
        "◇> INPUT → OUTPUT\n    OUTPUT → =",
        "◈+ INPUT → OUTPUT\n    OUTPUT → =",
        "(▷INPUT - INPUT) → D\n    ((D × D) + 1.0) → OUTPUT\n    OUTPUT → =",
    ];

    let mut proved = 0usize;
    let mut open = 0usize;
    let mut broken = 0usize;
    let mut checked = 0usize;
    let _ = seed;

    for (dims, shape) in shapes.iter() {
        for body in programs.iter().take(rounds) {
            let source = format!("{{\n    INPUT:◯ □ {dims}\n    {body}\n}}\n");
            let Ok(block) = parse_rho_program(&source) else {
                continue;
            };
            checked += 1;
            let result = validate(&block, shape, 0.0);
            let headline = body.lines().next().unwrap_or("").trim();
            match &result.verdict {
                Verdict::Equivalent => {
                    proved += 1;
                    println!(
                        "  proved   [{dims}] {headline}   ({} / {} nodes)",
                        result.source_nodes, result.target_nodes
                    );
                }
                Verdict::Differs { cell, witness } => {
                    broken += 1;
                    println!("  DIFFERS  [{dims}] {headline}");
                    println!("           cell {cell}, {witness}");
                }
                Verdict::Unknown(why) => {
                    open += 1;
                    println!("  unknown  [{dims}] {headline}   {why}");
                }
                Verdict::NotChecked(why) => {
                    open += 1;
                    println!("  skipped  [{dims}] {headline}   {why}");
                }
            }
        }
    }

    println!("\nchecked {checked}: proved {proved}, unsettled {open}, differing {broken}");
    std::process::exit(if broken > 0 { 1 } else { 0 });
}
