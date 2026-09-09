//! Translation validation: proving that the emitted IR computes what the
//! source means, for every input rather than for the ones we happened to try.
//!
//! Both sides are already written against `Numeric`, so running them on symbols
//! instead of numbers costs nothing but a type parameter — and it is the same
//! code that the differential testing exercises on numbers, so a bug in one is
//! a bug in both.
//!
//! The question handed to the solver is the negation: *is there an input for
//! which some cell differs?* `unsat` is the proof.

use crate::ast::ToposBlock;
use crate::codegen::LlvmCodeGen;
use crate::interp::{interpret, Env, Grid};
use crate::irvm::{parse_module, Machine, Value};
use crate::numeric::{Numeric, Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The two agree for every input.
    Equivalent,
    /// They differ, with the cell and an input that shows it.
    Differs { cell: usize, witness: String },
    /// The solver could not settle it.
    Unknown(String),
    /// Something upstream stopped the comparison from happening.
    NotChecked(String),
}

pub struct Validation {
    pub verdict: Verdict,
    /// Nodes in the two expression graphs, as a sense of the problem's size.
    pub source_nodes: usize,
    pub target_nodes: usize,
}

/// Build the source and IR expression graphs for a program at one shape.
pub fn expressions(
    block: &ToposBlock,
    shape: &[usize],
    tau: f64,
) -> Result<(Vec<Term>, Vec<Term>), String> {
    let ir = LlvmCodeGen::new("validate")
        .with_tau(tau)
        .generate_llvm_ir(block)
        .map_err(|e| e.to_string())?;
    expressions_of(block, shape, tau, &ir)
}

/// As [`expressions`], against IR supplied by the caller.
///
/// A validator has to be shown catching something, so a test needs a way to
/// hand it IR that is deliberately wrong.
pub fn expressions_of(
    block: &ToposBlock,
    shape: &[usize],
    tau: f64,
    ir: &str,
) -> Result<(Vec<Term>, Vec<Term>), String> {
    let cells: usize = shape.iter().product::<usize>().max(1);
    let inputs: Vec<Term> = (0..cells).map(Term::input).collect();

    // What the source means.
    let mut env: Env<Term> = Env::new();
    env.insert("INPUT".to_string(), Grid::from(shape.to_vec(), inputs.clone()));
    let interpreted = interpret(block, &env, tau).map_err(|e| e.to_string())?;
    let source = interpreted
        .get("OUTPUT")
        .ok_or("the program produces no OUTPUT")?
        .cells
        .clone();

    // What the generator emitted.
    let functions = parse_module(ir);
    let entry = functions
        .iter()
        .find(|f| f.name == "rho_kernel_exec_with_args")
        .ok_or("no rho_kernel_exec_with_args in the module")?;

    let mut machine: Machine<Term> = Machine::new();
    let in_handle = machine.add_buffer(inputs);
    let out_handle = machine.add_buffer(vec![Term::constant(0.0); source.len().max(cells)]);
    machine.run(entry, &[Value::P(in_handle, 0), Value::P(out_handle, 0)])?;
    let target = machine.buffer(out_handle)[..source.len()].to_vec();

    Ok((source, target))
}

#[cfg(feature = "z3-solver")]
mod smt {
    use super::*;
    use crate::numeric::{BoolNode, Compare, Node, Predicate};
    use std::collections::BTreeMap;
    use z3::ast::{Ast, Bool, Int, Real};
    use z3::{Config, Context, FuncDecl, SatResult, Solver, Sort};

    /// Give up on one program rather than stalling a build.
    const TIMEOUT_MS: u64 = 20_000;

    struct Encoder<'ctx> {
        ctx: &'ctx Context,
        inputs: BTreeMap<usize, Real<'ctx>>,
        terms: BTreeMap<usize, Real<'ctx>>,
        /// A power with a non-whole exponent, left uninterpreted. Both sides
        /// apply the same function to the same arguments, so equality still
        /// follows without anyone reasoning about real exponentiation.
        power: FuncDecl<'ctx>,
        /// The same treatment for the named functions: an exponential is a
        /// symbol here, and equivalence needs only that both sides use it.
        named: BTreeMap<&'static str, FuncDecl<'ctx>>,
    }

    impl<'ctx> Encoder<'ctx> {
        fn new(ctx: &'ctx Context) -> Encoder<'ctx> {
            let real = Sort::real(ctx);
            Encoder {
                ctx,
                inputs: BTreeMap::new(),
                terms: BTreeMap::new(),
                power: FuncDecl::new(ctx, "rho_power", &[&real, &real], &real),
                named: crate::ast::BuiltinOp::ALL
                    .iter()
                    .map(|name| {
                        (
                            *name,
                            FuncDecl::new(ctx, format!("rho_{name}"), &[&real], &real),
                        )
                    })
                    .collect(),
            }
        }

        fn rational(&self, value: f64) -> Real<'ctx> {
            if value == value.trunc() && value.abs() < 9.0e15 {
                return Real::from_int(&Int::from_i64(self.ctx, value as i64));
            }
            let text = format!("{value}");
            if let Some((whole, fraction)) = text.split_once('.') {
                if !text.contains('e') && fraction.len() <= 17 {
                    let digits = format!("{whole}{fraction}");
                    if let (Ok(num), Some(den)) =
                        (digits.parse::<i64>(), 10i64.checked_pow(fraction.len() as u32))
                    {
                        return Real::from_int(&Int::from_i64(self.ctx, num))
                            .div(&Real::from_int(&Int::from_i64(self.ctx, den)));
                    }
                }
            }
            // A literal this awkward becomes its own constant; both sides get
            // the same one, which is all the comparison needs.
            Real::new_const(self.ctx, format!("literal_{}", value.to_bits()))
        }

        fn term(&mut self, term: &Term) -> Real<'ctx> {
            if let Some(existing) = self.terms.get(&term.id()) {
                return existing.clone();
            }
            let built = match &*term.0 {
                Node::Input(index) => self
                    .inputs
                    .entry(*index)
                    .or_insert_with(|| Real::new_const(self.ctx, format!("in{index}")))
                    .clone(),
                Node::Const(v) => self.rational(*v),
                Node::Add(a, b) => {
                    let (x, y) = (self.term(a), self.term(b));
                    Real::add(self.ctx, &[&x, &y])
                }
                Node::Sub(a, b) => {
                    let (x, y) = (self.term(a), self.term(b));
                    Real::sub(self.ctx, &[&x, &y])
                }
                Node::Mul(a, b) => {
                    let (x, y) = (self.term(a), self.term(b));
                    Real::mul(self.ctx, &[&x, &y])
                }
                Node::Div(a, b) => {
                    let (x, y) = (self.term(a), self.term(b));
                    x.div(&y)
                }
                Node::Power(a, b) => {
                    let (x, y) = (self.term(a), self.term(b));
                    self.power
                        .apply(&[&x, &y])
                        .as_real()
                        .expect("rho_power returns a real")
                }
                Node::Unary(op, a) => {
                    let x = self.term(a);
                    self.named[op.name()]
                        .apply(&[&x])
                        .as_real()
                        .expect("a named function returns a real")
                }
                Node::Select(condition, a, b) => {
                    let flag = self.predicate(condition);
                    let (x, y) = (self.term(a), self.term(b));
                    flag.ite(&x, &y)
                }
            };
            self.terms.insert(term.id(), built.clone());
            built
        }

        fn predicate(&mut self, predicate: &Predicate) -> Bool<'ctx> {
            match &*predicate.0 {
                BoolNode::Const(v) => Bool::from_bool(self.ctx, *v),
                BoolNode::Compare(how, a, b) => {
                    let (x, y) = (self.term(a), self.term(b));
                    match how {
                        Compare::Gt => x.gt(&y),
                        Compare::Lt => x.lt(&y),
                        Compare::Gte => x.ge(&y),
                        Compare::Lte => x.le(&y),
                        Compare::Eq => x._eq(&y),
                        Compare::Ne => x._eq(&y).not(),
                    }
                }
                BoolNode::Or(a, b) => {
                    let (x, y) = (self.predicate(a), self.predicate(b));
                    Bool::or(self.ctx, &[&x, &y])
                }
            }
        }
    }

    /// Ask whether any cell can differ. `unsat` means none can.
    pub fn check(source: &[Term], target: &[Term]) -> Verdict {
        let mut config = Config::new();
        config.set_timeout_msec(TIMEOUT_MS);
        let ctx = Context::new(&config);

        for (cell, (a, b)) in source.iter().zip(target).enumerate() {
            let mut encoder = Encoder::new(&ctx);
            let left = encoder.term(a);
            let right = encoder.term(b);

            let solver = Solver::new(&ctx);
            solver.assert(&left._eq(&right).not());

            match solver.check() {
                SatResult::Unsat => {}
                SatResult::Sat => {
                    let witness = solver
                        .get_model()
                        .map(|model| {
                            encoder
                                .inputs
                                .iter()
                                .filter_map(|(index, var)| {
                                    model.eval(var, true).map(|v| format!("in{index} = {v}"))
                                })
                                .take(8)
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    return Verdict::Differs { cell, witness };
                }
                SatResult::Unknown => {
                    return Verdict::Unknown(
                        solver
                            .get_reason_unknown()
                            .unwrap_or_else(|| "the solver returned unknown".to_string()),
                    )
                }
            }
        }
        Verdict::Equivalent
    }
}

/// Compare two expression graphs that were built elsewhere.
pub fn compare(source: &[Term], target: &[Term]) -> Verdict {
    #[cfg(feature = "z3-solver")]
    {
        smt::check(source, target)
    }
    #[cfg(not(feature = "z3-solver"))]
    {
        let _ = (source, target);
        Verdict::NotChecked(
            "translation validation needs the SMT backend; build with --features z3-solver"
                .to_string(),
        )
    }
}

/// Prove that the emitted IR agrees with the source for every input at `shape`.
pub fn validate(block: &ToposBlock, shape: &[usize], tau: f64) -> Validation {
    let (source, target) = match expressions(block, shape, tau) {
        Ok(pair) => pair,
        Err(why) => {
            return Validation {
                verdict: Verdict::NotChecked(why),
                source_nodes: 0,
                target_nodes: 0,
            }
        }
    };

    let source_nodes = source.iter().map(Term::size).sum();
    let target_nodes = target.iter().map(Term::size).sum();

    let verdict = compare(&source, &target);

    Validation {
        verdict,
        source_nodes,
        target_nodes,
    }
}
