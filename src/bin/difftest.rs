//! Differential testing of the compiler against the reference interpreter.
//!
//! Random programs, random shapes, random inputs. The compiler and the
//! interpreter are written from the same specification but share no evaluation
//! code, so a disagreement means one of them is wrong and points at where.
//!
//!     cargo run --release --bin difftest -- [seed] [rounds]

use rho_lang::codegen::LlvmCodeGen;
use rho_lang::interp::{interpret_with, ByValue, Env, Grid, Options};
use rho_lang::irvm::{parse_module, Machine, Value};
use rho_lang::numeric::{Numeric, Precision};
use rho_lang::parser::parse_rho_program;
use std::collections::BTreeMap;

/// The cap on every `⇒` a generated program contains. Small, so a loop that
/// blows up to infinity or never settles costs little; the point is that all
/// three representations stop at the same sweep with the same bits.
const SWEEPS: usize = 8;

/// A tolerance of zero: a loop settles only when a sweep changes nothing.
const RUN: Options = Options {
    tau: 0.0,
    max_sweeps: SWEEPS,
};


/// Run the emitted IR directly, without going through clang.
///
/// This is the middle link of the chain: the interpreter says what the source
/// means, the IR says what the generator decided, and the .so says what clang
/// built. Checking the IR separately tells the two kinds of mistake apart.
fn run_ir<S: Numeric>(
    ir: &str,
    input: &[S],
    cells: usize,
    widen: impl Fn(&S) -> f64,
) -> Result<Vec<f64>, String> {
    let functions = parse_module(ir);
    let entry = functions
        .iter()
        .find(|f| f.name == "rho_kernel_exec_with_args")
        .ok_or("no rho_kernel_exec_with_args in the module")?;

    let mut machine: Machine<S> = Machine::new();
    let source = machine.add_buffer(input.to_vec());
    let target = machine.add_buffer(vec![S::constant(0.0); cells]);
    machine.run(entry, &[Value::P(source, 0), Value::P(target, 0)])?;
    Ok(machine.buffer(target).iter().map(&widen).collect())
}

/// As [`run_ir`], through `rho_kernel_exec_spaces`: one pointer per space in
/// `order`, the caller's inputs and the output filled in and every space the
/// kernel owns left null.
fn run_ir_spaces<S: Numeric>(
    ir: &str,
    order: &[String],
    inputs: &BTreeMap<String, Vec<S>>,
    cells: usize,
    widen: impl Fn(&S) -> f64,
) -> Result<Vec<f64>, String> {
    let functions = parse_module(ir);
    let entry = functions
        .iter()
        .find(|f| f.name == "rho_kernel_exec_spaces")
        .ok_or("no rho_kernel_exec_spaces in the module")?;

    let mut machine: Machine<S> = Machine::new();
    let target = machine.add_buffer(vec![S::constant(0.0); cells]);
    let mut table = Vec::new();
    for name in order {
        table.push(match inputs.get(name) {
            Some(data) => Value::P(machine.add_buffer(data.clone()), 0),
            None if name == "OUTPUT" => Value::P(target, 0),
            None => Value::null(),
        });
    }
    let handle = machine.add_table(table);
    machine.run(entry, &[Value::T(handle, 0)])?;
    Ok(machine.buffer(target).iter().map(&widen).collect())
}

/// Run the shared object through the table entrypoint and, when the program
/// reads INPUT alone, through the two-pointer one as well.
fn run_so(
    so: &str,
    order: &[String],
    inputs: &BTreeMap<String, Vec<f64>>,
    cells: usize,
    single_input: bool,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let mut copies = inputs.clone();
    let mut output = vec![0.0f64; cells];
    let mut table: Vec<*mut f64> = Vec::new();
    for name in order {
        table.push(match copies.get_mut(name) {
            Some(buf) => buf.as_mut_ptr(),
            None if name == "OUTPUT" => output.as_mut_ptr(),
            None => std::ptr::null_mut(),
        });
    }
    let mut via_args = single_input.then(|| vec![0.0f64; cells]);
    unsafe {
        let lib = libloading::Library::new(so).unwrap();
        let spaces: libloading::Symbol<unsafe extern "C" fn(*const *mut f64)> =
            lib.get(b"rho_kernel_exec_spaces").unwrap();
        spaces(table.as_ptr());
        if let Some(out) = via_args.as_mut() {
            let run: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
                lib.get(b"rho_kernel_exec_with_args").unwrap();
            run(inputs["INPUT"].as_ptr(), out.as_mut_ptr());
        }
    }
    (output, via_args)
}

/// The first cell where two results disagree on the bits. NaN compares unequal
/// to itself and the payload of a propagated NaN is not architecturally fixed,
/// so two NaNs count as agreeing however they are spelled.
fn first_gap(expected: &[f64], actual: &[f64]) -> Option<usize> {
    expected
        .iter()
        .zip(actual)
        .position(|(a, b)| !(a.is_nan() && b.is_nan()) && a.to_bits() != b.to_bits())
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

        // Named functions keep their shape, so they compose anywhere.
        5 => {
            let name = ["exp", "sqrt", "sin", "cos", "abs", "ind"][rng.below(6)];
            format!("({name} {})", expression(rng, depth - 1, spaces, want))
        }

        _ => {
            let op = ["+", "-", "×", "/", ">", "<"][rng.below(6)];
            let lhs = expression(rng, depth - 1, spaces, want);
            // A space whose shape stretches against `want` — a length-1 axis
            // against a longer one — exercises broadcasting. The other side
            // has to carry the wanted shape itself, so a bare literal is
            // replaced by a space that does.
            let stretching: Vec<&(String, Vec<usize>)> = spaces
                .iter()
                .filter(|(_, shape)| {
                    shape != want
                        && rho_lang::ast::broadcast_shapes(shape, want).as_deref() == Some(want)
                })
                .collect();
            if !stretching.is_empty() && !matching.is_empty() && rng.below(2) == 0 {
                let anchored = if spaces.iter().any(|(n, _)| lhs.contains(n.as_str())) {
                    lhs
                } else {
                    matching[rng.below(matching.len())].0.clone()
                };
                return format!("({anchored} {op} {})", stretching[rng.below(stretching.len())].0);
            }
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

/// Spell a shape as a declaration's dimension list.
fn dims_of(shape: &[usize]) -> String {
    shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" ")
}

/// A random program over INPUT and, in one round of three, a second input
/// AUX. Returns the source and AUX's shape when it has one.
fn program(rng: &mut Rng, dims: &str, shape: &[usize]) -> (String, Option<Vec<usize>>) {
    let mut spaces: Vec<(String, Vec<usize>)> = vec![("INPUT".to_string(), shape.to_vec())];
    let mut decls = format!("    INPUT:◯ □ {dims}\n");

    // The second input takes INPUT's shape, one that stretches against it —
    // a length-1 axis broadcasts — or one an axis shorter, which a fold of
    // INPUT can meet.
    let aux = if rng.below(3) == 0 {
        let aux_shape = match rng.below(3) {
            1 if shape.len() > 1 => {
                let mut stretched = shape.to_vec();
                let axis = rng.below(stretched.len());
                stretched[axis] = 1;
                stretched
            }
            2 if shape.len() > 1 => {
                rho_lang::ast::shape_without_axis(shape, rng.below(shape.len()))
            }
            _ => shape.to_vec(),
        };
        decls.push_str(&format!("    AUX:◯ □ {}\n", dims_of(&aux_shape)));
        spaces.push(("AUX".to_string(), aux_shape.clone()));
        Some(aux_shape)
    } else {
        None
    };

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
        spaces.push((name.clone(), target.clone()));

        // One intermediate in four is then iterated: a body of its own shape,
        // anchored to a space so that it has one.
        if rng.below(4) == 0 {
            let mut update = expression(rng, 2, &spaces, &target);
            if !spaces.iter().any(|(n, _)| update.contains(n.as_str())) {
                update = format!("({update} + {name})");
            }
            body.push_str(&format!("    {update} ⇒ {name}\n"));
        }
    }

    let final_shape = spaces[rng.below(spaces.len())].1.clone();
    body.push_str(&format!(
        "    {} → OUTPUT\n",
        expression(rng, 3, &spaces, &final_shape)
    ));
    body.push_str("    OUTPUT → =\n");
    (format!("{{\n{decls}{body}}}\n"), aux)
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
    let mut two_inputs = 0usize;
    let mut iterating = 0usize;
    let (mut ir_mismatches, mut ir_unsupported) = (0usize, 0usize);
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();

    for round in 0..rounds {
        let (dims, shape) = &shapes[rng.below(shapes.len())];
        let (source, aux) = program(&mut rng, dims, shape);

        let block = match parse_rho_program(&source) {
            Ok(b) => b,
            Err(e) => {
                skipped += 1;
                *reasons.entry(format!("parse: {}", short(&e))).or_insert(0) += 1;
                continue;
            }
        };

        let cells: usize = shape.iter().product();
        let single_input = aux.is_none();
        let mut inputs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        inputs.insert("INPUT".to_string(), (0..cells).map(|_| rng.value()).collect());
        if let Some(aux_shape) = &aux {
            let n: usize = aux_shape.iter().product();
            inputs.insert("AUX".to_string(), (0..n).map(|_| rng.value()).collect());
        }

        let mut env = Env::new();
        env.insert("INPUT".to_string(), Grid::from(shape.clone(), inputs["INPUT"].clone()));
        if let Some(aux_shape) = &aux {
            env.insert("AUX".to_string(), Grid::from(aux_shape.clone(), inputs["AUX"].clone()));
        }
        let interpreted = match interpret_with(&block, &env, &RUN, &mut ByValue) {
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

        let mut codegen = LlvmCodeGen::new(&format!("diff{round}")).with_max_sweeps(SWEEPS);
        let ir = match codegen.generate_llvm_ir(&block) {
            Ok(ir) => ir,
            Err(e) => {
                skipped += 1;
                *reasons.entry(format!("codegen: {}", short(&e))).or_insert(0) += 1;
                continue;
            }
        };
        // The table entrypoint takes the spaces in this order.
        let order: Vec<String> = codegen.space_shapes.keys().cloned().collect();
        let so = format!("target/diff{round}.so");
        if codegen.compile_to_so(&ir, &so).is_err() {
            skipped += 1;
            *reasons.entry("clang".to_string()).or_insert(0) += 1;
            continue;
        }
        let out_cells = cells.max(expected.len());

        // The IR, read back and run without clang — through the table of
        // spaces always, and through the two-pointer form when it applies.
        let mut ir_runs = vec![(
            "exec_spaces",
            run_ir_spaces(&ir, &order, &inputs, out_cells, |v: &f64| *v),
        )];
        if single_input {
            ir_runs.push((
                "exec_with_args",
                run_ir(&ir, &inputs["INPUT"], out_cells, |v: &f64| *v),
            ));
        }
        for (entry, outcome) in ir_runs {
            match outcome {
                Ok(from_ir) => {
                    if let Some(cell) = first_gap(&expected.cells, &from_ir) {
                        ir_mismatches += 1;
                        println!(
                            "IR MISMATCH at cell {cell} via {entry} (round {round}, shape {shape:?})"
                        );
                        println!("{source}");
                        println!("  inputs      {inputs:?}");
                        println!("  interpreted {:?}", expected.cells[cell]);
                        println!("  from IR     {:?}\n", from_ir[cell]);
                    }
                }
                Err(why) => {
                    ir_unsupported += 1;
                    if ir_unsupported <= 2 {
                        println!("IR NOT READ via {entry} (round {round}): {why}");
                    }
                }
            }
        }

        // The shared object, through the same entrypoints.
        let (output, output_args) = run_so(&so, &order, &inputs, out_cells, single_input);
        let _ = std::fs::remove_file(&so);

        // Every other round is repeated at single precision, where the same
        // three representations must still agree with one another.
        if round % 2 == 0 {
            let narrow: BTreeMap<String, Vec<f32>> = inputs
                .iter()
                .map(|(name, data)| (name.clone(), data.iter().map(|v| *v as f32).collect()))
                .collect();
            let mut narrow_env: rho_lang::interp::Env<f32> = rho_lang::interp::Env::new();
            narrow_env.insert(
                "INPUT".to_string(),
                Grid::from(shape.clone(), narrow["INPUT"].clone()),
            );
            if let Some(aux_shape) = &aux {
                narrow_env.insert(
                    "AUX".to_string(),
                    Grid::from(aux_shape.clone(), narrow["AUX"].clone()),
                );
            }
            if let Ok(narrow_out) = interpret_with(&block, &narrow_env, &RUN, &mut ByValue) {
                if let Some(meant) = narrow_out.get("OUTPUT") {
                    let mut narrow_codegen = LlvmCodeGen::new(&format!("diff{round}f32"))
                        .with_precision(Precision::F32)
                        .with_max_sweeps(SWEEPS);
                    if let Ok(narrow_ir) = narrow_codegen.generate_llvm_ir(&block) {
                        let meant_wide: Vec<f64> = meant.cells.iter().map(|v| *v as f64).collect();
                        let mut runs = vec![(
                            "exec_spaces",
                            run_ir_spaces(&narrow_ir, &order, &narrow, meant.len(), |v: &f32| {
                                *v as f64
                            }),
                        )];
                        if single_input {
                            runs.push((
                                "exec_with_args",
                                run_ir(&narrow_ir, &narrow["INPUT"], meant.len(), |v: &f32| {
                                    *v as f64
                                }),
                            ));
                        }
                        for (entry, outcome) in runs {
                            match outcome {
                                Ok(from_ir) => {
                                    if let Some(cell) = first_gap(&meant_wide, &from_ir) {
                                        ir_mismatches += 1;
                                        println!(
                                            "F32 IR MISMATCH at cell {cell} via {entry} (round {round})"
                                        );
                                        println!("{source}");
                                    }
                                }
                                Err(why) => {
                                    ir_unsupported += 1;
                                    if ir_unsupported <= 2 {
                                        println!("F32 IR NOT READ via {entry} (round {round}): {why}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        compared += 1;
        if !single_input {
            two_inputs += 1;
        }
        if source.contains('⇒') {
            iterating += 1;
        }
        let mut so_runs = vec![("exec_spaces", output)];
        if let Some(via_args) = output_args {
            so_runs.push(("exec_with_args", via_args));
        }
        for (entry, actual) in so_runs {
            if let Some(cell) = first_gap(&expected.cells, &actual) {
                mismatches += 1;
                let (a, b) = (expected.cells[cell], actual[cell]);
                println!("MISMATCH at cell {cell} via {entry} (round {round}, shape {shape:?})");
                println!("{source}");
                println!("  inputs      {inputs:?}");
                println!("  interpreted {a:?}  bits {:#018x}", a.to_bits());
                println!("  compiled    {b:?}  bits {:#018x}", b.to_bits());
                println!("  ulp gap     {}\n", (a.to_bits() as i64 - b.to_bits() as i64).abs());
            }
        }
        if mismatches >= 3 {
            break;
        }
    }

    println!(
        "seed {seed}: compared {compared} ({two_inputs} with two inputs, {iterating} iterating), \
         skipped {skipped}, mismatches {mismatches}, ir mismatches {ir_mismatches}, \
         ir unread {ir_unsupported}"
    );
    if std::env::var("DIFFTEST_VERBOSE").is_ok() {
        for (reason, count) in &reasons {
            println!("  skipped {count:4} x {reason}");
        }
    }
    std::process::exit(if mismatches > 0 || ir_mismatches > 0 { 1 } else { 0 });
}
