// SPDX-License-Identifier: Apache-2.0
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

use rho_lang::ast::BuiltinOp;
use rho_lang::codegen::LlvmCodeGen;
use rho_lang::interp::{interpret_with, Env, Grid, Options};
use rho_lang::numeric::{Compare, Numeric, Precision};
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
        let lib = rho_lang::dl::Library::open(std::path::Path::new(so)).unwrap();
        let spaces: unsafe extern "C" fn(*const *mut T) = lib.get("rho_kernel_exec_spaces").unwrap();
        spaces(table.as_ptr());
        if let Some(out) = via_args.as_mut() {
            let run: unsafe extern "C" fn(*const T, *mut T) = lib.get("rho_kernel_exec_with_args").unwrap();
            run(inputs["INPUT"].as_ptr(), out.as_mut_ptr());
        }
    }
    (output, via_args)
}

/// A number that remembers whether anything non-finite went into it.
///
/// The `!` analysis assumes no overflow and no NaN. A run that broke that
/// assumption is not a counterexample to anything the analysis said — but
/// whether it broke it cannot be read off the output, since a mask, a `⌈`
/// or an `ind` turns a NaN back into a finite number. So the interpreter is
/// run once more on values that carry a taint: any operation that sees or
/// produces an infinity or a NaN poisons its result, and poison flows through
/// every operation including a select, because the kernel computes both arms.
#[derive(Clone, Copy, Debug)]
struct Traced<S> {
    value: S,
    poisoned: bool,
}

impl<S: Numeric + Copy> Traced<S> {
    fn of(value: S, from: &[bool]) -> Traced<S> {
        let finite = value.as_constant().is_some_and(f64::is_finite);
        Traced {
            value,
            poisoned: !finite || from.iter().any(|p| *p),
        }
    }
}

impl<S: Numeric + Copy> Numeric for Traced<S> {
    type Bool = bool;

    fn constant(value: f64) -> Self {
        Traced::of(S::constant(value), &[])
    }
    fn as_constant(&self) -> Option<f64> {
        self.value.as_constant()
    }
    fn add(&self, o: &Self) -> Self {
        Traced::of(self.value.add(&o.value), &[self.poisoned, o.poisoned])
    }
    fn sub(&self, o: &Self) -> Self {
        Traced::of(self.value.sub(&o.value), &[self.poisoned, o.poisoned])
    }
    fn mul(&self, o: &Self) -> Self {
        Traced::of(self.value.mul(&o.value), &[self.poisoned, o.poisoned])
    }
    fn div(&self, o: &Self) -> Self {
        Traced::of(self.value.div(&o.value), &[self.poisoned, o.poisoned])
    }
    fn power(&self, o: &Self) -> Self {
        Traced::of(self.value.power(&o.value), &[self.poisoned, o.poisoned])
    }
    fn unary(&self, op: BuiltinOp) -> Self {
        Traced::of(self.value.unary(op), &[self.poisoned])
    }
    fn floor(&self) -> Self {
        Traced::of(self.value.floor(), &[self.poisoned])
    }
    fn compare(&self, o: &Self, how: Compare) -> bool {
        S::truth(&self.value.compare(&o.value, how))
    }
    fn select(condition: &bool, a: &Self, b: &Self) -> Self {
        let chosen = if *condition { a.value } else { b.value };
        Traced::of(chosen, &[a.poisoned, b.poisoned])
    }
    fn truth(flag: &bool) -> bool {
        *flag
    }
    fn carrying(&self, other: &Self) -> Self {
        Traced::of(self.value, &[self.poisoned, other.poisoned])
    }
}

/// Whether the interpreter, run on tainted values at width `S`, reaches the
/// output without any operation on its way having touched a non-finite
/// number. `None` when the run fails outright.
fn stays_within_assumptions<S: Numeric + Copy>(
    block: &rho_lang::ast::ToposBlock,
    inputs: &BTreeMap<String, Vec<S>>,
    shapes: &BTreeMap<String, Vec<usize>>,
) -> Option<bool> {
    let mut env: Env<Traced<S>> = Env::new();
    for (name, data) in inputs {
        let cells = data.iter().map(|v| Traced::of(*v, &[])).collect();
        env.insert(name.clone(), Grid::from(shapes[name].clone(), cells));
    }
    let out = interpret_with(block, &env, &RUN).ok()?;
    Some(out.get("OUTPUT")?.cells.iter().all(|c| !c.poisoned))
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
            // Any space reshapes to the wanted shape, reading round if short.
            if !spaces.is_empty() && rng.below(2) == 0 {
                let (name, _) = &spaces[rng.below(spaces.len())];
                return format!("({} ⍴ {name})", dims_of(want));
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
        // So do an index, a rotation and a reversal, which take their turns.
        0 | 1 if !matching.is_empty() => {
            let name = &matching[rng.below(matching.len())].0;
            let axis = if want.len() > 1 && rng.below(2) == 0 {
                rng.below(want.len()).to_string()
            } else {
                String::new()
            };
            // A take or a drop fits where some space differs from the wanted
            // shape along exactly one axis: shorter is taken or dropped down
            // to, longer is over-taken and padded with zero.
            let trimmable: Vec<(&String, usize, usize)> = spaces
                .iter()
                .filter(|(_, shape)| shape.len() == want.len())
                .filter_map(|(n, shape)| {
                    let differing: Vec<usize> =
                        (0..want.len()).filter(|&b| shape[b] != want[b]).collect();
                    match differing.as_slice() {
                        [b] => Some((n, *b, shape[*b])),
                        _ => None,
                    }
                })
                .collect();
            if !trimmable.is_empty() && rng.below(3) == 0 {
                let (n, b, have) = trimmable[rng.below(trimmable.len())];
                let need = want[b] as i64;
                let have = have as i64;
                let (glyph, count) = if have > need {
                    match rng.below(4) {
                        0 => ("↑", need),
                        1 => ("↑", -need),
                        2 => ("↓", have - need),
                        _ => ("↓", -(have - need)),
                    }
                } else if rng.below(2) == 0 {
                    ("↑", need)
                } else {
                    ("↑", -need)
                };
                return format!("({count} {glyph}{b} {n})");
            }
            // A transpose fits where some space's reversed shape is the
            // wanted one; a square or palindromic grid is its own.
            let transposable: Vec<&String> = spaces
                .iter()
                .filter(|(_, shape)| shape.iter().rev().copied().collect::<Vec<_>>() == want)
                .map(|(n, _)| n)
                .collect();
            if !transposable.is_empty() && rng.below(4) == 0 {
                return format!("(⍉{})", transposable[rng.below(transposable.len())]);
            }
            match rng.below(10) {
                0 => format!("(⍳{axis}{name})"),
                1 => format!("({} ⌽{axis} {name})", [-3i64, -2, -1, 1, 2, 3, 5][rng.below(7)]),
                2 => format!("(⌽{axis}{name})"),
                // A read at a position the data names: the coordinate moved
                // by a whole number, which runs past either end, or the
                // cell's own value scaled, which is fractional, negative or
                // far out as often as not; the space read is any at all,
                // since the position is a place in its row-major order.
                7 => format!(
                    "((⍳{axis}{name}) + {}) ⌷ {}",
                    [-2i64, -1, 0, 1, 2, 5][rng.below(6)],
                    spaces[rng.below(spaces.len())].0
                ),
                8 => format!(
                    "(({name} × {}) ⌷ {})",
                    ["0.5", "1.0", "3.0", "-2.0"][rng.below(4)],
                    spaces[rng.below(spaces.len())].0
                ),
                // The coordinate scaled and moved: still a whole number the
                // coordinates fix, read without a gather, and as often as
                // not past an end of the space read.
                9 => format!(
                    "((({}) × (⍳{axis}{name})) + {}) ⌷ {}",
                    ["2.0", "3.0", "-1.0"][rng.below(3)],
                    [-3i64, 0, 1, 4][rng.below(4)],
                    spaces[rng.below(spaces.len())].0
                ),
                // Any space at all reshapes to the wanted shape, reading
                // round when it is short and cut short when it is long.
                3 => format!("({} ⍴ {})", dims_of(want), spaces[rng.below(spaces.len())].0),
                _ => format!("({}{axis}{name})", ["▷", "▽"][rng.below(2)]),
            }
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
            let name = ["exp", "sqrt", "sin", "cos", "abs", "ind", "?", "⌊", "⌈"][rng.below(9)];
            format!("({name} {})", expression(rng, depth - 1, spaces, want))
        }

        _ => {
            let op = ["+", "-", "×", "/", ">", "<", "⌈", "⌊", "|"][rng.below(9)];
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
            // A space an axis short of `want` (a fold's result, a second
            // input, one cell) stretches once `□` gives it a unit axis where
            // the axis is missing. One time in four the lift is left out:
            // the ranks then differ, and every target must refuse the program
            // alike, as the interpreter does.
            let liftable: Vec<(&String, usize)> = spaces
                .iter()
                .filter(|(_, shape)| shape.len() + 1 == want.len())
                .flat_map(|(n, shape)| {
                    (0..=shape.len())
                        .filter(|&a| {
                            rho_lang::ast::shape_with_unit_axis(shape, a).is_some_and(|lifted| {
                                rho_lang::ast::broadcast_shapes(&lifted, want).as_deref() == Some(want)
                            })
                        })
                        .map(move |a| (n, a))
                })
                .collect();
            if !liftable.is_empty() && !matching.is_empty() && rng.below(3) == 0 {
                let anchored = if spaces.iter().any(|(n, _)| lhs.contains(n.as_str())) {
                    lhs
                } else {
                    matching[rng.below(matching.len())].0.clone()
                };
                let (n, a) = liftable[rng.below(liftable.len())];
                return if rng.below(4) == 0 {
                    format!("({anchored} {op} {n})")
                } else {
                    format!("({anchored} {op} (□{a} {n}))")
                };
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
    // Definitions, which go before the program.
    let mut defs = String::new();

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

        // One in four is then written again from itself, read at the cell
        // before: every cell must read the value from before this flow,
        // though the kernel writes the space as it sweeps it.
        if rng.below(4) == 0 {
            body.push_str(&format!(
                "    ({} + (▷{name})) → {name}\n",
                expression(rng, 2, &spaces, &target)
            ));
        }

        // One intermediate in four is then iterated: a body of its own shape,
        // anchored to a space so that it has one.
        if rng.below(4) == 0 {
            let mut update = expression(rng, 2, &spaces, &target);
            // Half of the loops step through a function with a body of
            // flows, applied to the iterate: the flows become the loop's
            // prelude, which the interpreter and the kernel each run every
            // round on their own.
            if rng.below(2) == 0 {
                let f = format!("step{}", ["A", "B", "C"][k]);
                let mut params = vec![("U".to_string(), target.clone())];
                // Half of them take two, and are then written between their
                // arguments; the second is any space of the same shape.
                let partner = if rng.below(2) == 0 {
                    let same: Vec<&(String, Vec<usize>)> =
                        spaces.iter().filter(|(_, s)| *s == target).collect();
                    params.push(("V".to_string(), target.clone()));
                    Some(same[rng.below(same.len())].0.clone())
                } else {
                    None
                };
                let first = expression(rng, 2, &params, &target);
                let mut locals = params.clone();
                locals.push(("S".to_string(), target.clone()));
                let second = expression(rng, 2, &locals, &target);
                let header: Vec<&str> = params.iter().map(|(p, _)| p.as_str()).collect();
                defs.push_str(&format!(
                    "{f}:{{ {}\n    {first} → S\n    {second} → =\n}}\n",
                    header.join(" ")
                ));
                let call = match partner {
                    Some(v) => format!("({name} {f} {v})"),
                    None => format!("({f} {name})"),
                };
                update = format!("({update} + {call})");
            }
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
    (format!("{defs}{{\n{decls}{body}}}\n"), aux)
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

/// The inputs a program declares, in the order the JAX module takes them
/// (as the program declares them).
fn jax_inputs(block: &rho_lang::ast::ToposBlock) -> Vec<String> {
    block
        .statements
        .iter()
        .filter_map(|s| match s {
            rho_lang::ast::Statement::SpaceDef(d) => Some(d.name.clone()),
            rho_lang::ast::Statement::ExtBind(b) => Some(b.space.name.clone()),
            _ => None,
        })
        .collect()
}

/// The first cell where the JAX run is not the interpreter's within rounding:
/// 2^10 units in the last place of the largest finite cell, so that XLA's own
/// order of a fold and its own `exp` pass and a cell stretched otherwise, or
/// left at zero, does not. NaN matches NaN, an infinity itself.
fn jax_gap(expected: &[f64], actual: &[f64]) -> Option<usize> {
    if expected.len() != actual.len() {
        return Some(expected.len().min(actual.len()));
    }
    let scale = expected
        .iter()
        .fold(0.0f64, |m, x| if x.is_finite() { m.max(x.abs()) } else { m });
    let ulp = f64::from_bits(scale.to_bits() + 1) - scale;
    expected.iter().zip(actual).position(|(e, a)| {
        if e.is_nan() || a.is_nan() {
            e.is_nan() != a.is_nan()
        } else if e.is_infinite() || a.is_infinite() {
            e != a
        } else {
            (e - a).abs() > 1024.0 * ulp
        }
    })
}

/// The first line of a diagnostic, for tallying why programs were skipped.
fn short(e: &rho_lang::error::HarmonyDisruption) -> String {
    let text = e.to_string();
    text.split(']').nth(1).unwrap_or(&text).trim().chars().take(48).collect()
}

/// q, the modulus of ML-KEM, for the integer mode.
const Q: u64 = 3329;

/// No value the integer mode writes reaches this: a 32-bit two's complement
/// cell holds it, and binary64 holds it exactly.
const INTEGER_LIMIT: u64 = 1 << 31;

/// A random term over integers below `INTEGER_LIMIT`, made of what an ML-KEM
/// kernel is made of, with the largest value it can take. An operation
/// whose bound would pass the limit is not taken. The bounds are this
/// generator's belief; where a run breaks one, fixed point and binary64
/// part ways and the comparison says so.
fn integer_term(rng: &mut Rng, depth: usize, spaces: &[(String, u64)]) -> (String, u64) {
    if depth == 0 || rng.below(4) == 0 {
        if rng.below(3) == 0 {
            let c = rng.below(Q as usize) as u64;
            return (format!("{c}.0"), c);
        }
        return spaces[rng.below(spaces.len())].clone();
    }
    let (a, ba) = integer_term(rng, depth - 1, spaces);
    let fits = |b: u64| b < INTEGER_LIMIT;
    match rng.below(9) {
        0..=2 => {
            let (b, bb) = integer_term(rng, depth - 1, spaces);
            match rng.below(3) {
                0 if fits(ba + bb) => (format!("({a} + {b})"), ba + bb),
                1 if ba.checked_mul(bb).is_some_and(fits) => (format!("({a} × {b})"), ba * bb),
                // A difference lifted by the subtrahend's bound stays whole.
                2 if fits(ba + bb) => (format!("(({a} - {b}) + {bb}.0)"), ba + bb),
                _ => (a, ba),
            }
        }
        3 => {
            let k = 1 + rng.below(16) as u32;
            (format!("(⌊({a} / {}.0))", 1u64 << k), ba >> k)
        }
        4 => (format!("(⌊({a} / 3329.0))"), ba / Q),
        // 2^16 ≡ 2285 (mod q): fold the high half down.
        5 if ba < (1 << 24) => (
            format!("(((⌊({a} / 65536.0)) × 2285.0) + ({a} - ((⌊({a} / 65536.0)) × 65536.0)))"),
            (ba >> 16) * 2285 + ba.min(65535),
        ),
        6 if ba < 2 * Q => (format!("({a} - (3329.0 × (ind ({a} >= 3329.0))))"), ba.min(Q - 1)),
        // A mask from a comparison's truth. Small constants and `<`, `<=`,
        // `==` make a comparison that holds on a zero common: `ind` must give
        // its truth there, not the zero the mask passed through.
        7 => {
            let (b, _) = integer_term(rng, depth - 1, spaces);
            // Half the time a high part of b, zero on most cells.
            let b = if rng.below(2) == 0 { format!("(⌊({b} / {}.0))", 1u64 << (10 + rng.below(7))) } else { b };
            let c = if rng.below(2) == 0 { rng.below(2) } else { rng.below(Q as usize) };
            let cmp = [">=", ">", "<", "<=", "=="][rng.below(5)];
            (format!("({a} × (ind ({b} {cmp} {c}.0)))"), ba)
        }
        _ => (a, ba),
    }
}

/// A program over integers mod q: INPUT and AUX hold coefficients, PERM a
/// permutation of the cells (a public table) that a `⌷` reads by, and two
/// to four flows of terms.
fn integer_program(rng: &mut Rng, dims: &str) -> String {
    let mut lines = vec![
        format!("    INPUT:◯ □ {dims}"),
        format!("    AUX:◯ □ {dims}"),
        format!("    PERM:◯ □ {dims}"),
        "    (PERM ⌷ INPUT) → G".to_string(),
    ];
    let mut spaces = vec![("INPUT".to_string(), Q - 1), ("AUX".to_string(), Q - 1), ("G".to_string(), Q - 1)];
    let flows = 2 + rng.below(3);
    for f in 0..flows {
        let (mut term, bound) = integer_term(rng, 3, &spaces);
        // A flow of constants alone has no shape; give it G's.
        if !term.chars().any(|c| c.is_ascii_uppercase()) {
            term = format!("({term} + (G × 0.0))");
        }
        if f + 1 == flows {
            lines.push(format!("    {term} → ="));
        } else {
            lines.push(format!("    {term} → F{f}"));
            spaces.push((format!("F{f}"), bound));
        }
    }
    format!("{{\n{}\n}}\n", lines.join("\n"))
}

/// DIFFTEST_MODQ=1: the integer mode the ML-KEM core needs (its spec, §8).
/// Each program is compiled and run at binary64 against the interpreter,
/// the interpreter in 32-bit fixed point is held to the one in binary64 (a
/// value past 2^31 would wrap in one and not the other), the constant-time
/// rules are checked with PERM public, and with DIFFTEST_SV the circuit at
/// `--fixed 32.0` is simulated against the fixed-point interpreter.
fn integer_main(seed: u64, rounds: usize) -> ! {
    use rho_lang::numeric::Fixed;
    type Q32 = Fixed<32, 0>;
    let shapes: [(&str, Vec<usize>); 5] =
        [("1 8", vec![1, 8]), ("2 8", vec![2, 8]), ("3 16", vec![3, 16]), ("4 4", vec![4, 4]), ("2 3 8", vec![2, 3, 8])];
    let circuit_leg = std::env::var("DIFFTEST_SV").is_ok();
    let mut rng = Rng(seed);
    let (mut compared, mut skipped, mut mismatches, mut wrapped, mut lint) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut circuit_compared, mut circuit_mismatches, mut circuit_failed) = (0usize, 0usize, 0usize);
    let mut fused_compared = 0usize;
    let public: std::collections::BTreeSet<String> = ["PERM".to_string()].into_iter().collect();
    for round in 0..rounds {
        let (dims, shape) = &shapes[rng.below(shapes.len())];
        let cells: usize = shape.iter().product();
        let source = integer_program(&mut rng, dims);
        let block = match parse_rho_program(&source) {
            Ok(b) => b,
            Err(e) => {
                skipped += 1;
                if std::env::var("DIFFTEST_VERBOSE").is_ok() {
                    println!("skipped (parse: {}):\n{source}", short(&e));
                }
                continue;
            }
        };
        if !rho_lang::consttime::check(&block, &public).is_empty() {
            lint += 1;
            println!("CONSTANT-TIME REFUSAL (round {round})\n{source}");
        }
        let mut perm: Vec<usize> = (0..cells).collect();
        for i in (1..cells).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        let mut inputs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        inputs.insert("INPUT".to_string(), (0..cells).map(|_| rng.below(Q as usize) as f64).collect());
        inputs.insert("AUX".to_string(), (0..cells).map(|_| rng.below(Q as usize) as f64).collect());
        inputs.insert("PERM".to_string(), perm.iter().map(|&p| p as f64).collect());
        let mut env: Env<f64> = Env::new();
        let mut fenv: Env<Q32> = Env::new();
        for (name, values) in &inputs {
            env.insert(name.clone(), Grid::from(shape.clone(), values.clone()));
            fenv.insert(name.clone(), Grid::from(shape.clone(), values.iter().map(|&v| Q32::from_f64(v)).collect()));
        }
        let (meant, fixed) = match (interpret_with(&block, &env, &RUN), interpret_with(&block, &fenv, &RUN)) {
            (Ok(m), Ok(f)) => (m, f),
            (Err(e), _) | (_, Err(e)) => {
                skipped += 1;
                if std::env::var("DIFFTEST_VERBOSE").is_ok() {
                    println!("skipped (interp: {}):\n{source}", short(&e));
                }
                continue;
            }
        };
        let expected = meant["OUTPUT"].cells.clone();
        let fixed: Vec<f64> = fixed["OUTPUT"].cells.iter().map(|q| q.to_f64()).collect();
        if first_gap(&expected, &fixed).is_some() {
            wrapped += 1;
            println!("FIXED POINT PARTS FROM BINARY64 (round {round})\n{source}");
        }
        let mut codegen = LlvmCodeGen::new(&format!("diffq{round}")).with_threads(3).with_grain(1);
        let ir = match codegen.generate_llvm_ir(&block) {
            Ok(ir) => ir,
            Err(e) => {
                skipped += 1;
                if std::env::var("DIFFTEST_VERBOSE").is_ok() {
                    println!("skipped (codegen: {}):\n{source}", short(&e));
                }
                continue;
            }
        };
        let order: Vec<String> = codegen.space_shapes.keys().cloned().collect();
        let so = format!("target/diffq{round}.so");
        if codegen.compile_to_so(&ir, &so).is_err() {
            skipped += 1;
            continue;
        }
        let (output, _) = run_so(&so, &order, &inputs, cells, false);
        let _ = std::fs::remove_file(&so);
        compared += 1;
        if let Some(cell) = first_gap(&expected, &output[..expected.len()]) {
            mismatches += 1;
            println!("MISMATCH at cell {cell} (round {round})\n{source}  interpreted {} compiled {}", expected[cell], output[cell]);
        }
        if circuit_leg {
            let numbers = rho_lang::codegen::sv::Numbers::Fixed { width: 32, frac: 0 };
            match rho_lang::codegen::sv::emit(&block, 0.0, None, numbers) {
                Ok(circuit) => {
                    // The circuit as written, and fused when fusing changes it.
                    let fused = rho_lang::codegen::sv::emit_fused(&block, 0.0, None, numbers)
                        .ok()
                        .filter(|f| f.module != circuit.module);
                    let variants = [Some(("", circuit)), fused.map(|f| (" (fused)", f))];
                    for (what, circuit) in variants.into_iter().flatten() {
                        let feed: Vec<Vec<f64>> = circuit.inputs.iter().map(|(n, _)| inputs[n].clone()).collect();
                        let dir = std::path::PathBuf::from(format!("target/diffqsv{round}"));
                        match rho_lang::codegen::sv::simulate(&circuit, &dir, &feed) {
                            Ok(run) => {
                                circuit_compared += 1;
                                fused_compared += usize::from(!what.is_empty());
                                if let Some(cell) = first_gap(&fixed, &run.output) {
                                    circuit_mismatches += 1;
                                    println!("CIRCUIT{what} MISMATCH at cell {cell} (round {round})\n{source}");
                                }
                            }
                            Err(e) => {
                                circuit_failed += 1;
                                println!("CIRCUIT{what} BUILD FAILED (round {round})\n{source}\n{e}");
                            }
                        }
                        let _ = std::fs::remove_dir_all(&dir);
                    }
                }
                Err(e) => {
                    circuit_failed += 1;
                    println!("CIRCUIT REFUSED (round {round})\n{source}\n{e}");
                }
            }
        }
    }
    println!(
        "seed {seed} (integers mod q): compared {compared}, skipped {skipped}, mismatches {mismatches}, \
         fixed point parted from binary64 {wrapped}, constant-time refusals {lint}"
    );
    if circuit_leg {
        println!("  circuit at 32.0: {circuit_compared} compared ({fused_compared} of them fused), {circuit_mismatches} mismatches, {circuit_failed} failed");
    }
    let bad = mismatches + wrapped + lint + circuit_mismatches + circuit_failed;
    std::process::exit(if bad > 0 { 1 } else { 0 })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20260909u64);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    if std::env::var("DIFFTEST_MODQ").is_ok() {
        integer_main(seed, rounds);
    }

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
    // The circuit leg: every program the SystemVerilog emitter accepts is
    // also simulated with Verilator and held to the interpreter. Off unless
    // asked for, since each build takes seconds.
    let circuit_leg = std::env::var("DIFFTEST_SV").is_ok();
    let (mut circuit_compared, mut circuit_mismatches, mut circuit_failed) = (0usize, 0usize, 0usize);
    let mut fused_compared = 0usize;
    let mut stepping = 0usize;
    let mut narrow_mismatches = 0usize;
    let mut refused = 0usize;
    let mut unsound = 0usize;
    let mut lifting = 0usize;
    // The JAX leg: every program compared is also run as the JAX module and
    // held to the interpreter, within rounding (XLA orders some folds its own
    // way and has its own `exp`). Off unless asked for, and skipped without a
    // python that imports jax ($RHO_JAX_PYTHON or python3): each run starts one.
    let jax_leg = std::env::var("DIFFTEST_JAX").is_ok() && rho_lang::codegen::jax::available();
    let (mut jax_compared, mut jax_mismatches, mut jax_failed) = (0usize, 0usize, 0usize);
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
        // The checks let it through, so the interpreter must run it: a
        // program it refuses while a target would compile it is one the
        // targets disagree on (a fold's one cell against a lifted grid
        // compiled to a kernel reading past the cell, where the interpreter
        // refused it and JAX stretched it).
        let interpreted = match interpret_with(&block, &env, &RUN) {
            Ok(i) => i,
            Err(e) => {
                refused += 1;
                println!("REFUSED BY THE INTERPRETER ALONE (round {round}, shape {shape:?}): {}", short(&e));
                println!("{source}\n");
                *reasons.entry(format!("interp: {}", short(&e))).or_insert(0) += 1;
                continue;
            }
        };
        let Some(expected) = interpreted.get("OUTPUT") else {
            skipped += 1;
            *reasons.entry("no OUTPUT".to_string()).or_insert(0) += 1;
            continue;
        };

        // Three threads and a grain of one cell: every sweep, however small,
        // is split into parts, so the head/body/tail of each part and the
        // partial maxima of a `⇒` are what the interpreter is held against.
        let mut codegen = LlvmCodeGen::new(&format!("diff{round}"))
            .with_max_sweeps(SWEEPS)
            .with_threads(3)
            .with_grain(1);
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
        let mut input_shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        input_shapes.insert("INPUT".to_string(), shape.clone());
        if let Some(aux_shape) = &aux {
            input_shapes.insert("AUX".to_string(), aux_shape.clone());
        }
        if stays_within_assumptions(&block, &inputs, &input_shapes) == Some(true) {
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
                        .with_threads(3)
                        .with_grain(1)
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
                            if stays_within_assumptions(&block, &narrow, &input_shapes)
                                == Some(true)
                            {
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
        if source.contains(":{") {
            stepping += 1;
        }
        // DIFFTEST_ROUND=n simulates the circuit of one round only; every
        // round is still compiled and run, so the random stream is the same.
        let this_round = std::env::var("DIFFTEST_ROUND")
            .ok()
            .and_then(|r| r.parse::<usize>().ok())
            .is_none_or(|only| only == round);
        if circuit_leg && this_round {
            let numbers = rho_lang::codegen::sv::Numbers::Real;
            if let Ok(circuit) = rho_lang::codegen::sv::emit(&block, RUN.tau, Some(SWEEPS), numbers) {
                // The circuit as written, and fused when fusing changes it.
                let fused = rho_lang::codegen::sv::emit_fused(&block, RUN.tau, Some(SWEEPS), numbers)
                    .ok()
                    .filter(|f| f.module != circuit.module);
                let variants = [Some(("", circuit)), fused.map(|f| (" (fused)", f))];
                for (what, circuit) in variants.into_iter().flatten() {
                    let feed: Vec<Vec<f64>> = circuit.inputs.iter().map(|(n, _)| inputs[n].clone()).collect();
                    let dir = std::path::PathBuf::from(format!("target/diffsv{round}"));
                    let fused = !what.is_empty();
                    match rho_lang::codegen::sv::simulate(&circuit, &dir, &feed) {
                        Ok(run) => {
                            circuit_compared += 1;
                            fused_compared += usize::from(fused);
                            if let Some(cell) = first_gap(&expected.cells, &run.output) {
                                circuit_mismatches += 1;
                                println!("CIRCUIT{what} MISMATCH at cell {cell} (round {round}, shape {shape:?})");
                                println!("{source}");
                                println!("  interpreted {:?}", expected.cells[cell]);
                                println!("  circuit     {:?}\n", run.output.get(cell));
                            }
                        }
                        Err(e) => {
                            circuit_failed += 1;
                            println!("CIRCUIT{what} BUILD FAILED (round {round}, shape {shape:?})\n{source}\n{e}\n");
                        }
                    }
                    let _ = std::fs::remove_dir_all(&dir);
                }
            }
        }
        // `□` also declares a space; as a lift it follows a parenthesis.
        if source.contains("(□") {
            lifting += 1;
        }
        if jax_leg {
            let feed: Vec<Vec<f64>> = jax_inputs(&block)
                .iter()
                .map(|n| inputs[n].clone())
                .collect();
            let dir = std::path::PathBuf::from(format!("target/diffjax{round}"));
            match rho_lang::codegen::jax::emit(&block, RUN.tau, Some(SWEEPS), Precision::F64)
                .map_err(|e| e.to_string())
                .and_then(|module| {
                    rho_lang::codegen::jax::run(&module, &dir, &feed).map_err(|e| e.to_string())
                }) {
                Ok(run) => {
                    jax_compared += 1;
                    if let Some(cell) = jax_gap(&expected.cells, &run.output) {
                        jax_mismatches += 1;
                        println!("JAX MISMATCH at cell {cell} (round {round}, shape {shape:?})");
                        println!("{source}");
                        println!("  interpreted {:?}", expected.cells[cell]);
                        println!("  jax         {:?}\n", run.output.get(cell));
                    }
                }
                Err(e) => {
                    jax_failed += 1;
                    println!("JAX FAILED (round {round}, shape {shape:?})\n{source}\n{e}\n");
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
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
        "seed {seed}: compared {compared} ({two_inputs} with two inputs, {iterating} iterating, \
         {stepping} through a function, \
         {lifting} stretching through a lift), skipped {skipped}, refused by the interpreter \
         alone {refused}, mismatches {mismatches}, f32 mismatches {narrow_mismatches}, \
         claims checked {claims_checked} ({proofs_checked} proved constraints, \
         {derived_unproved} of {derived} derived ones left open), unsound {unsound}"
    );
    if circuit_leg {
        println!(
            "  circuit: {circuit_compared} compared ({fused_compared} of them fused), {circuit_mismatches} mismatches, {circuit_failed} failed to build"
        );
    }
    if jax_leg {
        println!("  jax: {jax_compared} compared, {jax_mismatches} mismatches, {jax_failed} failed to run");
    }
    if std::env::var("DIFFTEST_VERBOSE").is_ok() {
        for (reason, count) in &reasons {
            println!("  skipped {count:4} x {reason}");
        }
    }
    std::process::exit(
        if mismatches > 0
            || narrow_mismatches > 0
            || refused > 0
            || unsound > 0
            || circuit_mismatches > 0
            || circuit_failed > 0
            || jax_mismatches > 0
            || jax_failed > 0
        {
            1
        } else {
            0
        },
    );
}
