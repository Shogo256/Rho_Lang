//! A negative control for the translation validator.
//!
//! A checker that never fails proves nothing about the thing it checks, so this
//! damages the emitted IR in a few small ways and insists the validator notices.

use rho_lang::parser::parse_rho_program;
use rho_lang::validate::{compare, expressions_of, Verdict};
use rho_lang::codegen::LlvmCodeGen;

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

    let mut caught = 0usize;
    let mut missed = 0usize;

    for (label, from, to) in damage {
        if !ir.contains(from) {
            println!("  n/a      {label} (no `{from}` in this kernel)");
            continue;
        }
        let broken = ir.replacen(from, to, 1);
        match expressions_of(&block, &shape, 0.0, &broken) {
            Ok((left, right)) => match compare(&left, &right) {
                Verdict::Differs { cell, .. } => {
                    caught += 1;
                    println!("  caught   {label} (cell {cell})");
                }
                other => {
                    missed += 1;
                    println!("  MISSED   {label} -> {other:?}");
                }
            },
            Err(why) => {
                missed += 1;
                println!("  MISSED   {label} (IR unreadable: {why})");
            }
        }
    }

    // And the undamaged kernel must still come out clean.
    let (left, right) = expressions_of(&block, &shape, 0.0, &ir).unwrap();
    let clean = compare(&left, &right);
    println!("\n  undamaged kernel: {clean:?}");

    println!("caught {caught}, missed {missed}");
    let sound = missed == 0 && clean == Verdict::Equivalent;
    std::process::exit(if sound { 0 } else { 1 });
}
