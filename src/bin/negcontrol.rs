//! A negative control for the translation validator.
//!
//! A checker that never fails proves nothing about the thing it checks, so this
//! damages the emitted IR in a few small ways and insists the validator notices.
//! Each entrypoint is damaged and checked on its own: a proof of the two-pointer
//! form says nothing about the plumbing of the table form, and the other way
//! round.

use rho_lang::codegen::LlvmCodeGen;
use rho_lang::parser::parse_rho_program;
use rho_lang::validate::{compare, expressions_via, Entrypoint, Verdict, VALIDATION_SWEEPS};

/// Apply one replacement inside the body of `entry` only, or None if the text
/// does not occur there.
fn damage_in(ir: &str, entry: Entrypoint, from: &str, to: &str) -> Option<String> {
    let marker = format!("define void @{}(", entry.symbol());
    let start = ir.find(&marker)?;
    let end = start + ir[start..].find("\n}\n")?;
    let body = &ir[start..end];
    if !body.contains(from) {
        return None;
    }
    Some(format!("{}{}{}", &ir[..start], body.replacen(from, to, 1), &ir[end..]))
}

/// One way to break the IR: a label, the text to find, what to put instead.
type Damage = (&'static str, &'static str, &'static str);

fn main() {
    // A straight-line kernel, and one that iterates: the loop's plumbing —
    // the copy back, the largest move, the exit test — is damaged on its own.
    let kernels: [(&str, &str, [Damage; 5]); 2] = [
        (
            "straight-line",
            "{\n    INPUT:◯ □ 6 1\n    (▷INPUT - INPUT) → D\n    ((D × D) + 1.0) → OUTPUT\n    OUTPUT → =\n}\n",
            [
                ("an add became a subtract", "fadd double", "fsub double"),
                ("a multiply became a divide", "fmul double", "fdiv double"),
                ("a sweep stops one cell early", "icmp ult i64 %f1.idx, 6", "icmp ult i64 %f1.idx, 5"),
                ("a boundary test flipped", "icmp eq i64 %v1, 0", "icmp ne i64 %v1, 0"),
                ("a constant shifted", "0x3FF0000000000000", "0x4000000000000000"),
            ],
        ),
        (
            "iterating",
            "{\n    INPUT:◯ □ 6 1\n    INPUT → X\n    ((INPUT - (▷X + ▽X)) / 4.0) ⇒ X\n    X → =\n}\n",
            [
                ("the loop body's subtract became an add", "fsub double", "fadd double"),
                ("the exit test became strict", "fcmp ole double %it2.d", "fcmp olt double %it2.d"),
                ("the largest move became the smallest", "fcmp ogt double %it2.mag, %it2.d", "fcmp olt double %it2.mag, %it2.d"),
                ("the cap dropped to one sweep", "icmp uge i64 %it2.k.next, 1000", "icmp uge i64 %it2.k.next, 1"),
                ("the copy back stops one cell early", "icmp ult i64 %it2.i, 6", "icmp ult i64 %it2.i, 5"),
            ],
        ),
    ];
    let shape = vec![6usize, 1];
    let entrypoints = [Entrypoint::WithArgs, Entrypoint::Spaces];

    let mut caught = 0usize;
    let mut missed = 0usize;
    let mut clean = true;

    for (kind, source, damage) in kernels {
        let block = parse_rho_program(source).unwrap();
        let ir = LlvmCodeGen::new("negative")
            .with_max_sweeps(VALIDATION_SWEEPS)
            .generate_llvm_ir(&block)
            .unwrap();

        for (label, from, to) in damage {
            for entry in entrypoints {
                let Some(broken) = damage_in(&ir, entry, from, to) else {
                    println!("  n/a      [{kind}] {label} (no `{from}` in {})", entry.symbol());
                    continue;
                };
                match expressions_via(&block, &shape, 0.0, &broken, entry) {
                    Ok((left, right)) => match compare(&left, &right) {
                        Verdict::Differs { cell, .. } => {
                            caught += 1;
                            println!("  caught   [{kind}] {label} in {} (cell {cell})", entry.symbol());
                        }
                        other => {
                            missed += 1;
                            println!("  MISSED   [{kind}] {label} in {} -> {other:?}", entry.symbol());
                        }
                    },
                    Err(why) => {
                        missed += 1;
                        println!("  MISSED   [{kind}] {label} in {} (IR unreadable: {why})", entry.symbol());
                    }
                }
            }
        }

        // And the undamaged kernel must still come out clean, both ways in.
        for entry in entrypoints {
            let (left, right) = expressions_via(&block, &shape, 0.0, &ir, entry).unwrap();
            let verdict = compare(&left, &right);
            println!("\n  undamaged [{kind}] kernel via {}: {verdict:?}", entry.symbol());
            clean &= verdict == Verdict::Equivalent;
        }
    }

    println!("caught {caught}, missed {missed}");
    std::process::exit(if missed == 0 && clean { 0 } else { 1 });
}
