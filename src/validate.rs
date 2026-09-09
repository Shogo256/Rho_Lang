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
    /// How many entrypoints were proved: the two-pointer form and the table
    /// form for a program that reads INPUT alone, the table form otherwise.
    pub entrypoints: usize,
}

/// Which C entrypoint the IR is read through. The two share the body a flow
/// lowers to but not the plumbing that hands it its buffers, so a proof of one
/// says nothing about the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entrypoint {
    /// `rho_kernel_exec_with_args(in, out)`: INPUT and OUTPUT only.
    WithArgs,
    /// `rho_kernel_exec_spaces(void **)`: one pointer per space.
    Spaces,
}

impl Entrypoint {
    pub fn symbol(self) -> &'static str {
        match self {
            Entrypoint::WithArgs => "rho_kernel_exec_with_args",
            Entrypoint::Spaces => "rho_kernel_exec_spaces",
        }
    }
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

/// As [`expressions`], against IR supplied by the caller, read through the
/// two-pointer entrypoint.
///
/// A validator has to be shown catching something, so a test needs a way to
/// hand it IR that is deliberately wrong.
pub fn expressions_of(
    block: &ToposBlock,
    shape: &[usize],
    tau: f64,
    ir: &str,
) -> Result<(Vec<Term>, Vec<Term>), String> {
    expressions_via(block, shape, tau, ir, Entrypoint::WithArgs)
}

/// As [`expressions_of`], through the entrypoint of the caller's choice.
///
/// Every space the kernel reads gets one symbol per cell, so a program with a
/// second input is as provable as one with INPUT alone — through the table
/// entrypoint, the only one that can carry it.
pub fn expressions_via(
    block: &ToposBlock,
    shape: &[usize],
    tau: f64,
    ir: &str,
    entry: Entrypoint,
) -> Result<(Vec<Term>, Vec<Term>), String> {
    // How the generator lays the spaces out. Generation is deterministic, so
    // this is the layout of the IR the caller hands in, damaged or not.
    let mut layout = LlvmCodeGen::new("layout").with_tau(tau);
    layout.generate_llvm_ir(block).map_err(|e| e.to_string())?;

    // One symbol per cell of every space the kernel reads, numbered across
    // the spaces so that no two cells share a name.
    let mut env: Env<Term> = Env::new();
    let mut symbols: std::collections::BTreeMap<String, Vec<Term>> = Default::default();
    let mut next = 0usize;
    for name in layout.input_spaces() {
        let space_shape = if name == "INPUT" {
            shape.to_vec()
        } else {
            layout.space_shapes[&name].clone()
        };
        let count = space_shape.iter().product::<usize>().max(1);
        let terms: Vec<Term> = (next..next + count).map(Term::input).collect();
        next += count;
        env.insert(name.clone(), Grid::from(space_shape, terms.clone()));
        symbols.insert(name, terms);
    }

    // What the source means.
    let interpreted = interpret(block, &env, tau).map_err(|e| e.to_string())?;
    let source = interpreted
        .get("OUTPUT")
        .ok_or("the program produces no OUTPUT")?
        .cells
        .clone();

    // What the generator emitted.
    let functions = parse_module(ir);
    let function = functions
        .iter()
        .find(|f| f.name == entry.symbol())
        .ok_or_else(|| format!("no {} in the module", entry.symbol()))?;

    let mut machine: Machine<Term> = Machine::new();
    let out_handle = machine.add_buffer(vec![Term::constant(0.0); source.len().max(1)]);
    match entry {
        Entrypoint::WithArgs => {
            if symbols.len() != 1 || !symbols.contains_key("INPUT") {
                return Err(format!(
                    "the program reads {:?}; {} carries INPUT alone",
                    symbols.keys().collect::<Vec<_>>(),
                    entry.symbol()
                ));
            }
            let in_handle = machine.add_buffer(symbols["INPUT"].clone());
            machine.run(function, &[Value::P(in_handle, 0), Value::P(out_handle, 0)])?;
        }
        Entrypoint::Spaces => {
            let mut table = Vec::new();
            for name in layout.space_shapes.keys() {
                table.push(match symbols.get(name) {
                    Some(terms) => Value::P(machine.add_buffer(terms.clone()), 0),
                    None if name == "OUTPUT" => Value::P(out_handle, 0),
                    // An intermediate the kernel owns.
                    None => Value::null(),
                });
            }
            let handle = machine.add_table(table);
            machine.run(function, &[Value::T(handle, 0)])?;
        }
    }
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

/// Prove that the emitted IR agrees with the source for every input at `shape`,
/// through every entrypoint that can carry the program.
pub fn validate(block: &ToposBlock, shape: &[usize], tau: f64) -> Validation {
    let not_checked = |why: String| Validation {
        verdict: Verdict::NotChecked(why),
        source_nodes: 0,
        target_nodes: 0,
        entrypoints: 0,
    };

    let mut codegen = LlvmCodeGen::new("validate").with_tau(tau);
    let ir = match codegen.generate_llvm_ir(block) {
        Ok(ir) => ir,
        Err(e) => return not_checked(e.to_string()),
    };
    // The two-pointer form cannot carry a second input.
    let entries: &[Entrypoint] = if codegen.input_spaces() == ["INPUT"] {
        &[Entrypoint::WithArgs, Entrypoint::Spaces]
    } else {
        &[Entrypoint::Spaces]
    };

    let mut source_nodes = 0usize;
    let mut target_nodes = 0usize;
    let mut entrypoints = 0usize;
    for &entry in entries {
        let (source, target) = match expressions_via(block, shape, tau, &ir, entry) {
            Ok(pair) => pair,
            Err(why) => return not_checked(why),
        };
        source_nodes = source.iter().map(Term::size).sum();
        target_nodes = target_nodes.max(target.iter().map(Term::size).sum());

        match compare(&source, &target) {
            Verdict::Equivalent => entrypoints += 1,
            Verdict::Differs { cell, witness } => {
                return Validation {
                    verdict: Verdict::Differs {
                        cell,
                        witness: format!("{}: {witness}", entry.symbol()),
                    },
                    source_nodes,
                    target_nodes,
                    entrypoints,
                }
            }
            other => {
                return Validation {
                    verdict: other,
                    source_nodes,
                    target_nodes,
                    entrypoints,
                }
            }
        }
    }

    Validation {
        verdict: Verdict::Equivalent,
        source_nodes,
        target_nodes,
        entrypoints,
    }
}
