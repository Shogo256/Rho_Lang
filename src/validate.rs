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

use crate::ast::{Statement, ToposBlock};
use crate::codegen::LlvmCodeGen;
use crate::interp::{interpret_with, Env, Grid, Options};
use crate::irvm::{parse_module, Machine, Value};
use crate::numeric::{Numeric, Predicate, Term};

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
        .with_max_sweeps(VALIDATION_SWEEPS)
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

/// The cap every validation run compiles with. Its value never matters to
/// the proof — the decisions are forced — as long as it leaves room for the
/// two rounds the induction takes.
pub const VALIDATION_SWEEPS: usize = 1000;

/// As [`expressions_of`], through the entrypoint of the caller's choice.
///
/// Every space the kernel reads gets one symbol per cell, so a program with a
/// second input is as provable as one with INPUT alone — through the table
/// entrypoint, the only one that can carry it.
///
/// A `⇒` loop is proved by induction rather than unrolled. Both sides are run
/// with the loop's exit forced the same way — once round from the start, once
/// more from a grid of fresh symbols, then out — and the test each side made
/// at each exit is recorded. Agreement on the second round's output says the
/// sweep agrees for *any* iterate, agreement on the recorded tests says both
/// sides leave at the same moment, and together with the first round that
/// covers every run of any length, whatever the tolerance decides.
pub fn expressions_via(
    block: &ToposBlock,
    shape: &[usize],
    tau: f64,
    ir: &str,
    entry: Entrypoint,
) -> Result<(Vec<Term>, Vec<Term>), String> {
    // How the generator lays the spaces out. Generation is deterministic, so
    // this is the layout of the IR the caller hands in, damaged or not.
    let mut layout = LlvmCodeGen::new("layout")
        .with_tau(tau)
        .with_max_sweeps(VALIDATION_SWEEPS);
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

    // The loops, in order, and the names their targets' buffers go by in
    // the IR, so the reader can hand a fresh grid to the right one.
    let loops: Vec<String> = block
        .statements
        .iter()
        .filter_map(|s| match s {
            Statement::Iterate { target, .. } => Some(target.clone()),
            _ => None,
        })
        .collect();
    let mut on_source = Induction::new(&loops, next);
    let mut on_ir = Induction::new(&loops, next);

    // What the source means.
    let options = Options {
        tau,
        max_sweeps: VALIDATION_SWEEPS,
    };
    let interpreted =
        interpret_with(block, &env, &options, &mut on_source).map_err(|e| e.to_string())?;
    let mut source = interpreted
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
            machine.run_with(
                function,
                &[Value::P(in_handle, 0), Value::P(out_handle, 0)],
                &mut on_ir,
            )?;
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
            machine.run_with(function, &[Value::T(handle, 0)], &mut on_ir)?;
        }
    }
    if let Some(name) = on_ir.lost.take() {
        return Err(format!(
            "the reader could not find the buffer of the iterate {name} in {}",
            entry.symbol()
        ));
    }
    let mut target = machine.buffer(out_handle)[..source.len()].to_vec();

    // The decisions each side made, as cells of their own: how many, then
    // each test as a 1-or-0 value. A kernel that left its loop at a different
    // moment, or tested a different thing, differs here.
    source.push(Term::constant(on_source.recorded.len() as f64));
    target.push(Term::constant(on_ir.recorded.len() as f64));
    let (one, zero) = (Term::constant(1.0), Term::constant(0.0));
    for (a, b) in on_source.recorded.iter().zip(&on_ir.recorded) {
        source.push(Term::select(a, &one, &zero));
        target.push(Term::select(b, &one, &zero));
    }

    Ok((source, target))
}

/// The policy both sides of a proof follow through a `⇒`: force each loop
/// round once from its start and once from a grid of fresh symbols, record
/// every exit test, and hand out the same fresh symbols in the same order.
struct Induction {
    forced: std::collections::VecDeque<bool>,
    recorded: Vec<Predicate>,
    next_symbol: usize,
    /// The loops in order; each round of fresh symbols goes to the next.
    targets: Vec<String>,
    rounds: usize,
    /// A target whose buffer the reader could not locate.
    lost: Option<String>,
}

impl Induction {
    fn new(targets: &[String], first_symbol: usize) -> Induction {
        Induction {
            forced: targets.iter().flat_map(|_| [false, true]).collect(),
            recorded: Vec::new(),
            next_symbol: first_symbol,
            targets: targets.to_vec(),
            rounds: 0,
            lost: None,
        }
    }

    fn fresh(&mut self) -> Term {
        let term = Term::input(self.next_symbol);
        self.next_symbol += 1;
        term
    }

    /// The loop whose round is about to start.
    fn current_target(&mut self) -> String {
        let target = self.targets[self.rounds.min(self.targets.len() - 1)].clone();
        self.rounds += 1;
        target
    }
}

impl crate::interp::Decider<Term> for Induction {
    fn done(&mut self, settled: &Predicate) -> Option<bool> {
        self.recorded.push(settled.clone());
        self.forced.pop_front()
    }
    fn next_round(&mut self, cells: &mut [Term]) {
        let _ = self.current_target();
        for cell in cells {
            *cell = self.fresh();
        }
    }
}

impl crate::irvm::Decider<Term> for Induction {
    fn done(&mut self, settled: &Predicate) -> Option<bool> {
        self.recorded.push(settled.clone());
        self.forced.pop_front()
    }
    fn next_round(&mut self, machine: &mut Machine<Term>) {
        let target = self.current_target();
        // The names a space's buffer goes by, by entrypoint: chosen at call
        // time, scratch, or the argument itself.
        let candidates = [
            format!("%{target}_eff"),
            format!("%{target}_buf"),
            match target.as_str() {
                "INPUT" => "%in_ptr".to_string(),
                "OUTPUT" => "%out_effective".to_string(),
                _ => String::new(),
            },
            if target == "OUTPUT" {
                "%b_out_effective".to_string()
            } else {
                String::new()
            },
        ];
        for name in candidates.iter().filter(|n| !n.is_empty()) {
            if let Some((handle, _)) = machine.pointer(name) {
                let cells = machine.cells_mut(handle);
                for cell in cells.iter_mut() {
                    *cell = self.fresh();
                }
                return;
            }
        }
        self.lost = Some(target);
    }
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

    let mut codegen = LlvmCodeGen::new("validate")
        .with_tau(tau)
        .with_max_sweeps(VALIDATION_SWEEPS);
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
