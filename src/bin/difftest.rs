//! Differential testing of the compiler against the reference interpreter.
//!
//! Random programs, random shapes, random inputs. The compiler and the
//! interpreter are written from the same specification but share no evaluation
//! code, so a disagreement means one of them is wrong and points at where.
//! Every program is compiled and run at both widths, through both C
//! entrypoints, and the bits are compared.
//!
//! The same runs check the `!` analysis. Whatever the intervals claimed —
//! the range of the output, a constraint they called proved — is compared
//! with what the kernel actually produced. That is refutation, not proof,
//! but it catches the failure that matters: a range narrower than the truth.
//!
//!     cargo run --release --bin difftest -- [seed] [rounds]

use rho_lang::codegen::LlvmCodeGen;
use rho_lang::interp::{interpret_with, Env, Grid, Options};
use rho_lang::numeric::Precision;
use rho_lang::parser::parse_rho_program;
use rho_lang::solver::{ConstraintSolver, Verdict};
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


/// Run the shared object through the table entrypoint and, when the program
/// reads INPUT alone, through the two-pointer one as well.
fn run_so<T: Copy + Default>(
    so: &str,
    order: &[String],
    inputs: &BTreeMap<String, Vec<T>>,
    cells: usize,
    single_input: bool,
) -> (Vec<T>, Option<Vec<T>>) {
    let mut copies = inputs.clone();
    let mut output = vec![T::default(); cells];
    let mut table: Vec<*mut T> = Vec::new();
    for name in order {
        table.push(match copies.get_mut(name) {
            Some(buf) => buf.as_mut_ptr(),
            None if name == "OUTPUT" => output.as_mut_ptr(),
            None => std::ptr::null_mut(),
        });
    }
    let mut via_args = single_input.then(|| vec![T::default(); cells]);
    unsafe {
        let lib = libloading::Library::new(so).unwrap();
        let spaces: libloading::Symbol<unsafe extern "C" fn(*const *mut T)> =
            lib.get(b"rho_kernel_exec_spaces").unwrap();
        spaces(table.as_ptr());
        if let Some(out) = via_args.as_mut() {
            let run: libloading::Symbol<unsafe extern "C" fn(*const T, *mut T)> =
                lib.get(b"rho_kernel_exec_with_args").unwrap();
            run(inputs["INPUT"].as_ptr(), out.as_mut_ptr());
        }
    }
    (output, via_args)
}

/// A number whose agreement is judged on its bits.
trait Bits: Copy {
    fn bits(self) -> u64;
    fn nan(self) -> bool;
}

impl Bits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
}

/// The first cell where two results disagree on the bits. NaN compares unequal
/// to itself and the payload of a propagated NaN is not architecturally fixed,
/// so two NaNs count as agreeing however they are spelled.
fn first_gap<T: Bits>(expected: &[T], actual: &[T]) -> Option<usize> {
    expected
        .iter()
        .zip(actual)
        .position(|(a, b)| !(a.nan() && b.nan()) && a.bits() != b.bits())
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

/// A constraint the generator put on OUTPUT: the comparison and the bound,
/// and whether it was derived from the range the analysis believed in.
#[derive(Clone, Copy)]
struct Claim {
    op: &'static str,
    bound: f64,
    derived: bool,
}

impl Claim {
    /// Whether every cell satisfies it.
    fn holds(&self, cells: &[f64]) -> bool {
        cells.iter().all(|v| match self.op {
            ">=" => *v >= self.bound,
            "<=" => *v <= self.bound,
            ">" => *v > self.bound,
            _ => *v < self.bound,
        })
    }
}

/// A random program over INPUT and, in one round of three, a second input
/// AUX. Returns the source and AUX's shape when it has one. A `!` on OUTPUT is
/// added afterwards, once the analysis has said what range it believes in.
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

/// A `!` on OUTPUT the analysis ought to settle, and a program carrying it.
///
/// A constraint picked blindly is almost never proved, and an unproved claim
/// says nothing when it turns out true. So the analysis is asked first what
/// range it believes OUTPUT has, and the constraint is placed just outside
/// that range: `OUTPUT >= lo - margin` when the low end is known, `<= hi +
/// margin` when the high end is. The analysis then has to prove it through
/// the constraint's own expansion — a different path from the range — and the
/// run has to bear it out. One program in five keeps a blind constraint, so
/// the unproved and the false are still exercised.
fn constrain(rng: &mut Rng, source: &str, range: (f64, f64)) -> Option<(String, Claim)> {
    let (lo, hi) = range;
    let blind = rng.below(5) == 0;
    let claim = if !blind && lo.is_finite() && (hi.is_infinite() || rng.below(2) == 0) {
        Claim {
            op: ">=",
            bound: (lo - lo.abs() * 0.25 - 1.0).floor(),
            derived: true,
        }
    } else if !blind && hi.is_finite() {
        Claim {
            op: "<=",
            bound: (hi + hi.abs() * 0.25 + 1.0).ceil(),
            derived: true,
        }
    } else {
        Claim {
            op: [">=", "<=", ">", "<"][rng.below(4)],
            bound: (rng.value() / 4.0).trunc(),
            derived: false,
        }
    };
    let line = format!("    ! (OUTPUT {} {:.1})\n    OUTPUT → =\n", claim.op, claim.bound);
    let constrained = source.replacen("    OUTPUT → =\n", &line, 1);
    (constrained != source).then_some((constrained, claim))
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

    let shapes: [(&str, Vec<usize>); 9] = [
        ("8 1", vec![8, 1]),
        ("12 1", vec![12, 1]),
        ("16 1", vec![16, 1]),
        ("3 4", vec![3, 4]),
        ("4 4", vec![4, 4]),
        ("5 3", vec![5, 3]),
        ("7 3", vec![7, 3]),
        ("2 3 2", vec![2, 3, 2]),
        ("4 4 4", vec![4, 4, 4]),
    ];

    let mut rng = Rng(seed);
    let (mut compared, mut skipped, mut mismatches) = (0usize, 0usize, 0usize);
    let mut two_inputs = 0usize;
    let mut iterating = 0usize;
    let mut narrow_mismatches = 0usize;
    let mut unsound = 0usize;
    let mut claims_checked = 0usize;
    let mut proofs_checked = 0usize;
    // Constraints placed just outside the believed range, and how many the
    // analysis then failed to prove through the constraint's own expansion.
    let (mut derived, mut derived_unproved) = (0usize, 0usize);
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

        // Ask the analysis what it believes, then make it commit to a `!`.
        let believed = ConstraintSolver::analyze_at(&block, RUN.tau, Precision::F64).output_range;
        let (source, block, claim) = match constrain(&mut rng, &source, (believed.lo, believed.hi)) {
            Some((constrained, claim)) => match parse_rho_program(&constrained) {
                Ok(b) => (constrained, b, Some(claim)),
                Err(_) => (source, block, None),
            },
            None => (source, block, None),
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
        let interpreted = match interpret_with(&block, &env, &RUN) {
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

        // The shared object, through both entrypoints.
        let (output, output_args) = run_so(&so, &order, &inputs, out_cells, single_input);
        let _ = std::fs::remove_file(&so);

        // What the intervals claimed about this program, held to this run.
        // The claims assume no overflow and no NaN, so a run that produced
        // either is not a counterexample to anything.
        let report = ConstraintSolver::analyze_at(&block, RUN.tau, Precision::F64);
        let produced = &output[..expected.len()];
        if produced.iter().all(|v| v.is_finite()) {
            claims_checked += 1;
            let (lo, hi) = (report.output_range.lo, report.output_range.hi);
            if let Some(cell) = produced.iter().position(|v| *v < lo || *v > hi) {
                unsound += 1;
                println!("UNSOUND RANGE at cell {cell} (round {round}, shape {shape:?})");
                println!("{source}");
                println!("  claimed     [{lo}, {hi}]");
                println!("  produced    {:?}\n", produced[cell]);
            }
            if let (Some(claim), Some(finding)) = (claim, report.constraints.first()) {
                if claim.derived {
                    derived += 1;
                    if finding.verdict != Verdict::Proved {
                        derived_unproved += 1;
                    }
                }
                if finding.verdict == Verdict::Proved {
                    proofs_checked += 1;
                    if !claim.holds(produced) {
                        unsound += 1;
                        println!(
                            "UNSOUND PROOF (round {round}, shape {shape:?}): {}",
                            finding.subject
                        );
                        println!("{source}");
                        println!("  produced    {produced:?}\n");
                    }
                }
            }
        }

        // Every other round is repeated at single precision: the interpreter
        // narrowed to f32 against a kernel compiled at f32.
        if round % 2 == 0 {
            let narrow: BTreeMap<String, Vec<f32>> = inputs
                .iter()
                .map(|(name, data)| (name.clone(), data.iter().map(|v| *v as f32).collect()))
                .collect();
            let mut narrow_env: Env<f32> = Env::new();
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
            if let Ok(narrow_out) = interpret_with(&block, &narrow_env, &RUN) {
                if let Some(meant) = narrow_out.get("OUTPUT") {
                    let mut narrow_codegen = LlvmCodeGen::new(&format!("diff{round}f32"))
                        .with_precision(Precision::F32)
                        .with_max_sweeps(SWEEPS);
                    if let Ok(narrow_ir) = narrow_codegen.generate_llvm_ir(&block) {
                        let narrow_so = format!("target/diff{round}f32.so");
                        if narrow_codegen.compile_to_so(&narrow_ir, &narrow_so).is_ok() {
                            let (got, got_args) = run_so(
                                &narrow_so,
                                &order,
                                &narrow,
                                cells.max(meant.len()),
                                single_input,
                            );
                            let _ = std::fs::remove_file(&narrow_so);
                            // The analysis at f32 widens by 2^-24; the narrow
                            // kernel must stay inside what it claimed too.
                            let narrow_report =
                                ConstraintSolver::analyze_at(&block, RUN.tau, Precision::F32);
                            let produced: Vec<f64> =
                                got[..meant.len()].iter().map(|v| *v as f64).collect();
                            if produced.iter().all(|v| v.is_finite()) {
                                let (lo, hi) =
                                    (narrow_report.output_range.lo, narrow_report.output_range.hi);
                                if let Some(cell) =
                                    produced.iter().position(|v| *v < lo || *v > hi)
                                {
                                    unsound += 1;
                                    println!(
                                        "UNSOUND F32 RANGE at cell {cell} (round {round}, shape {shape:?})"
                                    );
                                    println!("{source}");
                                    println!("  claimed     [{lo}, {hi}]");
                                    println!("  produced    {:?}\n", produced[cell]);
                                }
                            }
                            let mut runs = vec![("exec_spaces", got)];
                            if let Some(g) = got_args {
                                runs.push(("exec_with_args", g));
                            }
                            for (entry, actual) in runs {
                                if let Some(cell) = first_gap(&meant.cells, &actual) {
                                    narrow_mismatches += 1;
                                    println!(
                                        "F32 MISMATCH at cell {cell} via {entry} (round {round}, shape {shape:?})"
                                    );
                                    println!("{source}");
                                    println!("  inputs      {narrow:?}");
                                    println!("  interpreted {:?}", meant.cells[cell]);
                                    println!("  compiled    {:?}\n", actual[cell]);
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
         skipped {skipped}, mismatches {mismatches}, f32 mismatches {narrow_mismatches}, \
         claims checked {claims_checked} ({proofs_checked} proved constraints, \
         {derived_unproved} of {derived} derived ones left open), unsound {unsound}"
    );
    if std::env::var("DIFFTEST_VERBOSE").is_ok() {
        for (reason, count) in &reasons {
            println!("  skipped {count:4} x {reason}");
        }
    }
    std::process::exit(if mismatches > 0 || narrow_mismatches > 0 || unsound > 0 {
        1
    } else {
        0
    });
}
