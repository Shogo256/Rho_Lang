//! The value a program is evaluated over.
//!
//! Both evaluators — the reference interpreter and the reader for the emitted
//! IR — are written against this trait rather than against `f64` directly. That
//! lets the same, already differential-tested code run twice: once on numbers,
//! to compare answers, and once on symbols, to compare *functions*.
//!
//! Only the data is abstracted. Indices, addresses and control flow stay
//! concrete, because nothing `rhoc` emits ever branches on a value.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// The comparisons the language and the IR both need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
}

pub trait Numeric: Clone + fmt::Debug {
    type Bool: Clone + fmt::Debug;

    fn constant(value: f64) -> Self;
    /// The value when it is known, which is what lets `^` recognise a whole
    /// exponent whether it is running on numbers or on symbols.
    fn as_constant(&self) -> Option<f64>;

    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    /// Exponentiation with an exponent that is not a whole number.
    fn power(&self, other: &Self) -> Self;

    fn compare(&self, other: &Self, how: Compare) -> Self::Bool;
    fn select(condition: &Self::Bool, when_true: &Self, when_false: &Self) -> Self;

    fn boolean(value: bool) -> Self::Bool;
    fn or(a: &Self::Bool, b: &Self::Bool) -> Self::Bool;
}

impl Numeric for f64 {
    type Bool = bool;

    fn constant(value: f64) -> Self {
        value
    }
    fn as_constant(&self) -> Option<f64> {
        Some(*self)
    }
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn mul(&self, other: &Self) -> Self {
        self * other
    }
    fn div(&self, other: &Self) -> Self {
        self / other
    }
    fn power(&self, other: &Self) -> Self {
        f64::powf(*self, *other)
    }
    fn compare(&self, other: &Self, how: Compare) -> bool {
        match how {
            Compare::Gt => self > other,
            Compare::Lt => self < other,
            Compare::Gte => self >= other,
            Compare::Lte => self <= other,
            Compare::Eq => self == other,
        }
    }
    fn select(condition: &bool, when_true: &Self, when_false: &Self) -> Self {
        if *condition {
            *when_true
        } else {
            *when_false
        }
    }
    fn boolean(value: bool) -> bool {
        value
    }
    fn or(a: &bool, b: &bool) -> bool {
        *a || *b
    }
}

// ------------------------------------------------------------------ symbols

/// A value built from the inputs rather than computed from them.
#[derive(Debug)]
pub enum Node {
    /// Cell `index` of the caller's input.
    Input(usize),
    Const(f64),
    Add(Term, Term),
    Sub(Term, Term),
    Mul(Term, Term),
    Div(Term, Term),
    /// A power with an exponent that is not a whole number. Left uninterpreted:
    /// both sides apply the same function to the same arguments, so equivalence
    /// still follows, and no solver has to reason about real exponentiation.
    Power(Term, Term),
    Select(Predicate, Term, Term),
}

#[derive(Debug)]
pub enum BoolNode {
    Const(bool),
    Compare(Compare, Term, Term),
    Or(Predicate, Predicate),
}

/// A shared subexpression. Sharing matters: an unrolled sweep reuses the same
/// values many times, and walking them again for each use would not finish.
#[derive(Debug, Clone)]
pub struct Term(pub Rc<Node>);

#[derive(Debug, Clone)]
pub struct Predicate(pub Rc<BoolNode>);

impl Term {
    fn of(node: Node) -> Term {
        Term(Rc::new(node))
    }

    /// A stable identity for memoising a walk over the shared graph.
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn input(index: usize) -> Term {
        Term::of(Node::Input(index))
    }

    /// How many distinct nodes this term is built from.
    pub fn size(&self) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        let mut stack = vec![self.clone()];
        while let Some(term) = stack.pop() {
            if !seen.insert(term.id()) {
                continue;
            }
            match &*term.0 {
                Node::Input(_) | Node::Const(_) => {}
                Node::Add(a, b)
                | Node::Sub(a, b)
                | Node::Mul(a, b)
                | Node::Div(a, b)
                | Node::Power(a, b) => {
                    stack.push(a.clone());
                    stack.push(b.clone());
                }
                Node::Select(p, a, b) => {
                    stack.push(a.clone());
                    stack.push(b.clone());
                    collect_predicate(p, &mut stack);
                }
            }
        }
        seen.len()
    }
}

fn collect_predicate(predicate: &Predicate, stack: &mut Vec<Term>) {
    match &*predicate.0 {
        BoolNode::Const(_) => {}
        BoolNode::Compare(_, a, b) => {
            stack.push(a.clone());
            stack.push(b.clone());
        }
        BoolNode::Or(a, b) => {
            collect_predicate(a, stack);
            collect_predicate(b, stack);
        }
    }
}

thread_local! {
    /// Constants are shared so that the many zeros and ones an unrolled sweep
    /// produces do not each become a separate node.
    static CONSTANTS: RefCell<std::collections::BTreeMap<u64, Term>> =
        const { RefCell::new(std::collections::BTreeMap::new()) };
}

impl Numeric for Term {
    type Bool = Predicate;

    fn constant(value: f64) -> Term {
        CONSTANTS.with(|cache| {
            cache
                .borrow_mut()
                .entry(value.to_bits())
                .or_insert_with(|| Term::of(Node::Const(value)))
                .clone()
        })
    }

    fn as_constant(&self) -> Option<f64> {
        match &*self.0 {
            Node::Const(v) => Some(*v),
            _ => None,
        }
    }

    fn add(&self, other: &Term) -> Term {
        Term::of(Node::Add(self.clone(), other.clone()))
    }
    fn sub(&self, other: &Term) -> Term {
        Term::of(Node::Sub(self.clone(), other.clone()))
    }
    fn mul(&self, other: &Term) -> Term {
        Term::of(Node::Mul(self.clone(), other.clone()))
    }
    fn div(&self, other: &Term) -> Term {
        Term::of(Node::Div(self.clone(), other.clone()))
    }
    fn power(&self, other: &Term) -> Term {
        Term::of(Node::Power(self.clone(), other.clone()))
    }

    fn compare(&self, other: &Term, how: Compare) -> Predicate {
        Predicate(Rc::new(BoolNode::Compare(how, self.clone(), other.clone())))
    }

    fn select(condition: &Predicate, when_true: &Term, when_false: &Term) -> Term {
        Term::of(Node::Select(
            condition.clone(),
            when_true.clone(),
            when_false.clone(),
        ))
    }

    fn boolean(value: bool) -> Predicate {
        Predicate(Rc::new(BoolNode::Const(value)))
    }

    fn or(a: &Predicate, b: &Predicate) -> Predicate {
        Predicate(Rc::new(BoolNode::Or(a.clone(), b.clone())))
    }
}

/// `x^n` as repeated multiplication when `n` is a small whole number.
///
/// The definition lives here so the interpreter, the IR reader and the solver
/// all inherit exactly the same one. Leaving it to a maths library made two
/// implementations round a square differently.
pub fn integer_power<S: Numeric>(base: &S, exponent: &S) -> S {
    let Some(n) = exponent.as_constant() else {
        return base.power(exponent);
    };
    if n != n.trunc() || n.abs() > 64.0 {
        return base.power(exponent);
    }
    let steps = n.abs() as u32;
    let mut acc = S::constant(1.0);
    for _ in 0..steps {
        acc = acc.mul(base);
    }
    if n < 0.0 {
        S::constant(1.0).div(&acc)
    } else {
        acc
    }
}
