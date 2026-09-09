//! A negative control for the translation validator.
//!
//! A checker that never fails proves nothing about the thing it checks, so this
//! damages the emitted IR in a few small ways and insists the validator notices.
//! Each entrypoint is damaged and checked on its own: a proof of the two-pointer
//! form says nothing about the plumbing of the table form, and the other way
//! round.

use rho_lang::codegen::LlvmCodeGen;
use rho_lang::parser::parse_rho_program;
use rho_lang::validate::{compare, expressions_via, Entrypoint, Verdict};

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

fn main() {
    let source = "{\n    INPUT:◯ □ 6 1\n    (▷INPUT - INPUT) → D\n    ((D × D) + 1.0) → OUTPUT\n    OUTPUT → =\n}\n";
    let block = parse_rho_program(source).unwrap();
    let shape = vec![6usize, 1];
    let ir = LlvmCodeGen::new("negative")
        .generate_llvm_ir(&block)
        .unwrap();

    // Each entry breaks the IR in one small way that a careless generator
    // might plausibly produce.
    let damage: [(&str, &str, &str); 5] = [
        ("an add became a subtract", "fadd double", "fsub double"),
        ("a multiply became a divide", "fmul double", "fdiv double"),
        ("a sweep stops one cell early", "icmp ult i64 %f1.idx, 6", "icmp ult i64 %f1.idx, 5"),
        ("a boundary test flipped", "icmp eq i64 %v1, 0", "icmp ne i64 %v1, 0"),
        ("a constant shifted", "0x3FF0000000000000", "0x4000000000000000"),
    ];
    let entrypoints = [Entrypoint::WithArgs, Entrypoint::Spaces];

    let mut caught = 0usize;
    let mut missed = 0usize;

    for (label, from, to) in damage {
        for entry in entrypoints {
            let Some(broken) = damage_in(&ir, entry, from, to) else {
                println!("  n/a      {label} (no `{from}` in {})", entry.symbol());
                continue;
            };
            match expressions_via(&block, &shape, 0.0, &broken, entry) {
                Ok((left, right)) => match compare(&left, &right) {
                    Verdict::Differs { cell, .. } => {
                        caught += 1;
                        println!("  caught   {label} in {} (cell {cell})", entry.symbol());
                    }
                    other => {
                        missed += 1;
                        println!("  MISSED   {label} in {} -> {other:?}", entry.symbol());
                    }
                },
                Err(why) => {
                    missed += 1;
                    println!("  MISSED   {label} in {} (IR unreadable: {why})", entry.symbol());
                }
            }
        }
    }

    // And the undamaged kernel must still come out clean, both ways in.
    let mut clean = true;
    for entry in entrypoints {
        let (left, right) = expressions_via(&block, &shape, 0.0, &ir, entry).unwrap();
        let verdict = compare(&left, &right);
        println!("\n  undamaged kernel via {}: {verdict:?}", entry.symbol());
        clean &= verdict == Verdict::Equivalent;
    }

    println!("caught {caught}, missed {missed}");
    std::process::exit(if missed == 0 && clean { 0 } else { 1 });
}
