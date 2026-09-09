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

    // `{dims}` and `{rows}` are filled in per shape, so a second input can be
    // declared to match INPUT or to stretch against it.
    let programs: [&str; 20] = [
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
        "exp INPUT → OUTPUT\n    OUTPUT → =",
        "(abs INPUT) → OUTPUT\n    OUTPUT → =",
        "(ind (INPUT > 1.0)) → OUTPUT\n    OUTPUT → =",
        "(ind (INPUT > 0.0)) → M\n    ◇+ M → OUTPUT\n    OUTPUT → =",
        // Two inputs: only the table entrypoint can carry these.
        "AUX:◯ □ {dims}\n    ((INPUT - AUX) × AUX) → OUTPUT\n    OUTPUT → =",
        "AUX:◯ □ {rows} 1\n    ((INPUT × AUX) > AUX) → OUTPUT\n    OUTPUT → =",
        "A:◯ □ 2 3 1\n    B:◯ □ 1 3 4\n    ◇+1 (A × B) → OUTPUT\n    OUTPUT → =",
        // Iterations: proved by induction over one round from an arbitrary grid.
        "INPUT → X\n    ((INPUT - (▷X + ▽X)) / 4.0) ⇒ X\n    X → =",
        "INPUT → U\n    ((▷U + ▽U) / 4.0) ⇒ U\n    (U × 2.0) → OUTPUT\n    OUTPUT → =",
        "INPUT → X\n    (X / (□1 (◇+1 (abs X)))) ⇒ X\n    X → =",
    ];

    let mut proved = 0usize;
    let mut open = 0usize;
    let mut broken = 0usize;
    let mut checked = 0usize;
    let _ = seed;

    for (dims, shape) in shapes.iter() {
        for body in programs.iter().take(rounds) {
            let body = body
                .replace("{dims}", dims)
                .replace("{rows}", &shape[0].to_string());
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
                        "  proved   [{dims}] {headline}   ({} / {} nodes, {} entrypoint{})",
                        result.source_nodes,
                        result.target_nodes,
                        result.entrypoints,
                        if result.entrypoints == 1 { "" } else { "s" }
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
