//! Differential testing of the compiler against the reference interpreter.
//!
//! Random programs, random shapes, random inputs. The compiler and the
//! interpreter are written from the same specification but share no evaluation
//! code, so a disagreement means one of them is wrong and points at where.
//!
//!     cargo run --release --bin difftest -- [seed] [rounds]

use rho_lang::codegen::LlvmCodeGen;
use rho_lang::interp::{interpret, Env, Grid};
use rho_lang::irvm::{parse_module, Machine, Value};
use rho_lang::parser::parse_rho_program;

/// Run the emitted IR directly, without going through clang.
///
/// This is the middle link of the chain: the interpreter says what the source
/// means, the IR says what the generator decided, and the .so says what clang
/// built. Checking the IR separately tells the two kinds of mistake apart.
fn run_ir(ir: &str, input: &[f64], cells: usize) -> Result<Vec<f64>, String> {
    let functions = parse_module(ir);
    let entry = functions
        .iter()
        .find(|f| f.name == "rho_kernel_exec_with_args")
        .ok_or("no rho_kernel_exec_with_args in the module")?;

    let mut machine = Machine::new();
    let source = machine.add_buffer(input.to_vec());
    let target = machine.add_buffer(vec![0.0; cells]);
    machine.run(entry, &[Value::P(source, 0), Value::P(target, 0)])?;
    Ok(machine.buffer(target).to_vec())
}

/// Deterministic xorshift, so any failure is reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn value(&mut self) -> f64 {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (unit - 0.5) * 20.0
    }
}

/// Build an expression that produces exactly `want`.
///
/// Generating blindly and discarding whatever fails to typecheck wastes most of
/// the search; steering by the wanted shape keeps nearly every program valid, so
/// the rounds are spent comparing rather than rejecting.
fn expression(
    rng: &mut Rng,
    depth: usize,
    spaces: &[(String, Vec<usize>)],
    want: &[usize],
) -> String {
    let matching: Vec<&(String, Vec<usize>)> =
        spaces.iter().filter(|(_, shape)| shape == want).collect();

    if depth == 0 || rng.below(4) == 0 {
        // A literal-only expression has no shape and would be discarded, so a
        // leaf reaches for a space whenever one fits — and folds one down to
        // size when none does.
        if matching.is_empty() {
            if let Some(folded) = fold_down_to(rng, spaces, want) {
                return folded;
            }
            return format!("{:.1}", rng.value().trunc());
        }
        if rng.below(6) == 0 {
            return format!("{:.1}", rng.value().trunc());
        }
        return matching[rng.below(matching.len())].0.clone();
    }

    match rng.below(10) {
        // A shift keeps its shape, but needs a space to read a neighbour from.
        0 | 1 if !matching.is_empty() => {
            let glyph = ["▷", "▽"][rng.below(2)];
            let axis = if want.len() > 1 && rng.below(2) == 0 {
                rng.below(want.len()).to_string()
            } else {
                String::new()
            };
            format!("({glyph}{axis}{})", matching[rng.below(matching.len())].0)
        }

        // A fold needs an operand whose shape collapses to the wanted one.
        2 => match fold_down_to(rng, spaces, want) {
            Some(folded) => folded,
            None => expression(rng, depth - 1, spaces, want),
        },

        // A scan keeps its shape.
        3 => {
            let glyph = ["◈+", "◈×", "◈>", "◈<"][rng.below(4)];
            let axis = if want.len() > 1 && rng.below(2) == 0 {
                rng.below(want.len()).to_string()
            } else {
                String::new()
            };
            format!("({glyph}{axis} {})", expression(rng, depth - 1, spaces, want))
        }

        4 => format!("({} ^ 2.0)", expression(rng, depth - 1, spaces, want)),

        _ => {
            let op = ["+", "-", "×", "/", ">", "<"][rng.below(6)];
            let lhs = expression(rng, depth - 1, spaces, want);
            let rhs = expression(rng, depth - 1, spaces, want);
            // Anchor the pair to a space if neither side reached one.
            if !matching.is_empty()
                && !spaces.iter().any(|(n, _)| lhs.contains(n.as_str()) || rhs.contains(n.as_str()))
            {
                return format!("({} {op} {})", matching[rng.below(matching.len())].0, rhs);
            }
            format!("({lhs} {op} {rhs})")
        }
    }
}

/// A fold of some declared space whose shape collapses to `want`.
fn fold_down_to(rng: &mut Rng, spaces: &[(String, Vec<usize>)], want: &[usize]) -> Option<String> {
    let candidates: Vec<&(String, Vec<usize>)> = spaces
        .iter()
        .filter(|(_, shape)| {
            shape.len() == want.len() + 1
                && (0..shape.len()).any(|a| rho_lang::ast::shape_without_axis(shape, a) == want)
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let (name, shape) = candidates[rng.below(candidates.len())];
    let axis = (0..shape.len())
        .find(|&a| rho_lang::ast::shape_without_axis(shape, a) == want)?;
    let glyph = ["◇+", "◇×", "◇>", "◇<"][rng.below(4)];
    Some(format!("({glyph}{axis} {name})"))
}

fn program(rng: &mut Rng, dims: &str, shape: &[usize]) -> String {
    let mut spaces: Vec<(String, Vec<usize>)> = vec![("INPUT".to_string(), shape.to_vec())];
    let mut body = String::new();

    for k in 0..=rng.below(3) {
        // Half the intermediates keep the grid; the rest collapse an axis, so
        // both the shape-preserving and the shape-changing paths get exercised.
        let target: Vec<usize> = if shape.len() > 1 && rng.below(2) == 0 {
            rho_lang::ast::shape_without_axis(shape, rng.below(shape.len()))
        } else {
            shape.to_vec()
        };
        let name = format!("T{k}");
        body.push_str(&format!(
            "    {} → {name}\n",
            expression(rng, 3, &spaces, &target)
        ));
        spaces.push((name, target));
    }

    let final_shape = spaces[rng.below(spaces.len())].1.clone();
    body.push_str(&format!(
        "    {} → OUTPUT\n",
        expression(rng, 3, &spaces, &final_shape)
    ));
    body.push_str("    OUTPUT → =\n");
    format!("{{\n    INPUT:◯ □ {dims}\n{body}}}\n")
}

/// The first line of a diagnostic, for tallying why programs were skipped.
fn short(e: &rho_lang::error::HarmonyDisruption) -> String {
    let text = e.to_string();
    text.split(']').nth(1).unwrap_or(&text).trim().chars().take(48).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20260909u64);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);

    let shapes: [(&str, Vec<usize>); 6] = [
        ("8 1", vec![8, 1]),
        ("12 1", vec![12, 1]),
        ("3 4", vec![3, 4]),
        ("4 4", vec![4, 4]),
        ("5 3", vec![5, 3]),
        ("2 3 2", vec![2, 3, 2]),
    ];

    let mut rng = Rng(seed);
    let (mut compared, mut skipped, mut mismatches) = (0usize, 0usize, 0usize);
    let (mut ir_mismatches, mut ir_unsupported) = (0usize, 0usize);
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();

    for round in 0..rounds {
        let (dims, shape) = &shapes[rng.below(shapes.len())];
        let source = program(&mut rng, dims, shape);

        let block = match parse_rho_program(&source) {
            Ok(b) => b,
            Err(e) => {
                skipped += 1;
                *reasons.entry(format!("parse: {}", short(&e))).or_insert(0) += 1;
                continue;
            }
        };

        let cells: usize = shape.iter().product();
        let input: Vec<f64> = (0..cells).map(|_| rng.value()).collect();

        let mut env = Env::new();
        env.insert("INPUT".to_string(), Grid::from(shape.clone(), input.clone()));
        let interpreted = match interpret(&block, &env, 0.0) {
            Ok(i) => i,
            Err(e) => {
                skipped += 1;
                *reasons.entry(format!("interp: {}", short(&e))).or_insert(0) += 1;
                continue;
            }
        };
        let Some(expected) = interpreted.get("OUTPUT") else {
            skipped += 1;
            *reasons.entry("no OUTPUT".to_string()).or_insert(0) += 1;
            continue;
        };

        let mut codegen = LlvmCodeGen::new(&format!("diff{round}"));
        let ir = match codegen.generate_llvm_ir(&block) {
            Ok(ir) => ir,
            Err(e) => {
                skipped += 1;
                *reasons.entry(format!("codegen: {}", short(&e))).or_insert(0) += 1;
                continue;
            }
        };
        let so = format!("target/diff{round}.so");
        if codegen.compile_to_so(&ir, &so).is_err() {
            skipped += 1;
            *reasons.entry("clang".to_string()).or_insert(0) += 1;
            continue;
        }

        // The IR, read back and run without clang.
        match run_ir(&ir, &input, cells.max(expected.len())) {
            Ok(from_ir) => {
                let gap = expected
                    .cells
                    .iter()
                    .zip(&from_ir)
                    .position(|(a, b)| !(a.is_nan() && b.is_nan()) && a.to_bits() != b.to_bits());
                if let Some(cell) = gap {
                    ir_mismatches += 1;
                    println!("IR MISMATCH at cell {cell} (round {round}, shape {shape:?})");
                    println!("{source}");
                    println!("  input       {input:?}");
                    println!("  interpreted {:?}", expected.cells[cell]);
                    println!("  from IR     {:?}\n", from_ir[cell]);
                }
            }
            Err(why) => {
                ir_unsupported += 1;
                if ir_unsupported <= 2 {
                    println!("IR NOT READ (round {round}): {why}");
                }
            }
        }

        let mut output = vec![0.0f64; cells.max(expected.len())];
        unsafe {
            let lib = libloading::Library::new(&so).unwrap();
            let run: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
                lib.get(b"rho_kernel_exec_with_args").unwrap();
            run(input.as_ptr(), output.as_mut_ptr());
        }
        let _ = std::fs::remove_file(&so);

        compared += 1;
        // NaN compares unequal to itself, so agreement is judged on the bits —
        // except that the payload of a propagated NaN is not architecturally
        // fixed, so two NaNs count as agreeing however they are spelled.
        let disagreement = expected.cells.iter().zip(&output).position(|(a, b)| {
            if a.is_nan() && b.is_nan() {
                false
            } else {
                a.to_bits() != b.to_bits()
            }
        });

        if let Some(cell) = disagreement {
            mismatches += 1;
            let (a, b) = (expected.cells[cell], output[cell]);
            println!("MISMATCH at cell {cell} (round {round}, shape {shape:?})");
            println!("{source}");
            println!("  input       {input:?}");
            println!("  interpreted {a:?}  bits {:#018x}", a.to_bits());
            println!("  compiled    {b:?}  bits {:#018x}", b.to_bits());
            println!("  ulp gap     {}\n", (a.to_bits() as i64 - b.to_bits() as i64).abs());
            if mismatches >= 3 {
                break;
            }
        }
    }

    println!(
        "seed {seed}: compared {compared}, skipped {skipped}, \
         mismatches {mismatches}, ir mismatches {ir_mismatches}, ir unread {ir_unsupported}"
    );
    if std::env::var("DIFFTEST_VERBOSE").is_ok() {
        for (reason, count) in &reasons {
            println!("  skipped {count:4} x {reason}");
        }
    }
    std::process::exit(if mismatches > 0 || ir_mismatches > 0 { 1 } else { 0 });
}
