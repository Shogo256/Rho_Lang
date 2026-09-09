//! A reader and evaluator for the LLVM IR that `rhoc` emits.
//!
//! This exists to check the code generator's output against the language's
//! semantics without going through clang: the IR is the contract with the
//! backend, so validating it separates "did we mean the right thing" from "did
//! clang do what we said".
//!
//! It handles the subset `rhoc` produces and nothing else. That is not a
//! limitation to apologise for — a validator that accepted more would be harder
//! to trust, and anything outside the subset is a bug in the generator worth
//! hearing about.
//!
//! One property of that subset makes this tractable: control flow never depends
//! on data. Loop bounds are literals, `br` only ever tests an integer
//! comparison, and every floating-point decision goes through `select`. So the
//! whole program unrolls with concrete indices, and only the values are in
//! question.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    F(f64),
    I(i64),
    B(bool),
    /// A buffer and an element offset into it.
    P(usize, i64),
    VF(Vec<f64>),
    VI(Vec<i64>),
    VB(Vec<bool>),
}

impl Value {
    fn f(&self) -> f64 {
        match self {
            Value::F(v) => *v,
            other => panic!("expected a double, found {other:?}"),
        }
    }
    fn i(&self) -> i64 {
        match self {
            Value::I(v) => *v,
            other => panic!("expected an i64, found {other:?}"),
        }
    }
    fn b(&self) -> bool {
        match self {
            Value::B(v) => *v,
            other => panic!("expected an i1, found {other:?}"),
        }
    }
    fn vf(&self) -> Vec<f64> {
        match self {
            Value::VF(v) => v.clone(),
            Value::F(v) => vec![*v],
            other => panic!("expected a double vector, found {other:?}"),
        }
    }
    fn vi(&self) -> Vec<i64> {
        match self {
            Value::VI(v) => v.clone(),
            Value::I(v) => vec![*v],
            other => panic!("expected an i64 vector, found {other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    label: String,
    instructions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    blocks: Vec<Block>,
}

/// Split a module into its functions, keeping each block's instructions in order.
pub fn parse_module(ir: &str) -> Vec<Function> {
    let mut functions = Vec::new();
    let mut current: Option<Function> = None;

    for raw in ir.lines() {
        let line = raw.split(';').next().unwrap_or("").trim().to_string();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("define ") {
            let name = rest
                .split('@')
                .nth(1)
                .and_then(|s| s.split('(').next())
                .unwrap_or("")
                .to_string();
            let params = rest
                .split('(')
                .nth(1)
                .and_then(|s| s.split(')').next())
                .map(|s| {
                    s.split(',')
                        .filter_map(|p| p.split_whitespace().nth(1).map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            // Blocks come from label lines. An unlabelled entry block is
            // created on demand, so an explicit `entry:` does not open a second
            // one that nothing ever branches to.
            current = Some(Function {
                name,
                params,
                blocks: Vec::new(),
            });
            continue;
        }

        if line == "}" {
            if let Some(f) = current.take() {
                functions.push(f);
            }
            continue;
        }

        let Some(f) = current.as_mut() else { continue };

        // `label:` opens a block; anything else is an instruction in it.
        if line.ends_with(':') && !line.contains(' ') {
            f.blocks.push(Block {
                label: format!("%{}", line.trim_end_matches(':')),
                instructions: Vec::new(),
            });
        } else {
            if f.blocks.is_empty() {
                f.blocks.push(Block {
                    label: "%entry".to_string(),
                    instructions: Vec::new(),
                });
            }
            f.blocks.last_mut().unwrap().instructions.push(line);
        }
    }

    functions
}

/// The state one function body runs against.
pub struct Machine {
    /// Element-addressed storage, one entry per buffer.
    buffers: Vec<Vec<f64>>,
    /// Buffers reached through `inttoptr`, keyed by the address in the source.
    external: BTreeMap<i64, usize>,
    names: BTreeMap<String, Value>,
    /// A stop so a generator bug cannot hang the validator.
    budget: usize,
}

impl Machine {
    pub fn new() -> Machine {
        Machine {
            buffers: Vec::new(),
            external: BTreeMap::new(),
            names: BTreeMap::new(),
            budget: 20_000_000,
        }
    }

    /// Register a buffer the caller owns, returning the handle to bind to a
    /// parameter or an address.
    pub fn add_buffer(&mut self, cells: Vec<f64>) -> usize {
        self.buffers.push(cells);
        self.buffers.len() - 1
    }

    pub fn bind_address(&mut self, address: i64, buffer: usize) {
        self.external.insert(address, buffer);
    }

    pub fn buffer(&self, handle: usize) -> &[f64] {
        &self.buffers[handle]
    }

    fn value(&self, token: &str) -> Value {
        self.try_value(token)
            .unwrap_or_else(|why| panic!("{why}"))
    }

    fn try_value(&self, token: &str) -> Result<Value, String> {
        if let Some(v) = self.names.get(token) {
            return Ok(v.clone());
        }
        parse_literal(token).ok_or_else(|| format!("unknown operand `{token}`"))
    }

    fn store(&mut self, handle: usize, offset: i64, value: f64) {
        let buffer = &mut self.buffers[handle];
        let index = offset.max(0) as usize;
        if index >= buffer.len() {
            buffer.resize(index + 1, 0.0);
        }
        buffer[index] = value;
    }

    fn read(&self, handle: usize, offset: i64) -> f64 {
        self.buffers[handle]
            .get(offset.max(0) as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// Run one function to completion.
    pub fn run(&mut self, function: &Function, arguments: &[Value]) -> Result<(), String> {
        for (name, value) in function.params.iter().zip(arguments) {
            self.names.insert(name.clone(), value.clone());
        }

        let mut current = 0usize;
        let mut previous = String::new();

        loop {
            if self.budget == 0 {
                return Err("instruction budget exhausted".to_string());
            }
            let block = function
                .blocks
                .get(current)
                .ok_or_else(|| "fell off the end of the function".to_string())?;
            let label = block.label.clone();
            let mut next: Option<String> = None;

            for instruction in &block.instructions {
                self.budget -= 1;
                match self.step(instruction, &previous)? {
                    Flow::Next => {}
                    Flow::Jump(target) => {
                        next = Some(target);
                        break;
                    }
                    Flow::Return => return Ok(()),
                }
            }

            let Some(target) = next else {
                return Err(format!("block {label} ran off its end"));
            };
            previous = label;
            current = function
                .blocks
                .iter()
                .position(|b| b.label == target)
                .ok_or_else(|| format!("no such block {target}"))?;
        }
    }

    fn step(&mut self, instruction: &str, previous: &str) -> Result<Flow, String> {
        // Terminators first: they decide where control goes next.
        if instruction == "ret void" {
            return Ok(Flow::Return);
        }
        if let Some(rest) = instruction.strip_prefix("br ") {
            if let Some(target) = rest.strip_prefix("label ") {
                return Ok(Flow::Jump(target.trim().to_string()));
            }
            let parts = split_fields(rest);
            let condition = parts[0].trim_start_matches("i1 ").trim();
            let taken = self.value(condition).b();
            let target = if taken { parts[1] } else { parts[2] };
            return Ok(Flow::Jump(
                target.trim_start_matches("label ").trim().to_string(),
            ));
        }
        if instruction.starts_with("call void @free") {
            return Ok(Flow::Next);
        }

        let (name, body) = match instruction.split_once(" = ") {
            Some((n, b)) => (n.trim().to_string(), b.trim()),
            None => {
                // A store is the only instruction here that yields nothing.
                if let Some(rest) = instruction.strip_prefix("store ") {
                    self.store_instruction(rest)?;
                    return Ok(Flow::Next);
                }
                return Err(format!("unsupported instruction: {instruction}"));
            }
        };

        let value = self.evaluate(body, previous)?;
        self.names.insert(name, value);
        Ok(Flow::Next)
    }

    fn store_instruction(&mut self, rest: &str) -> Result<(), String> {
        // `<ty> <value>, ptr <dest>, align N`
        let fields = split_fields(rest);
        let value_part = fields
            .first()
            .ok_or_else(|| format!("malformed store: {rest}"))?;
        let dest = fields
            .get(1)
            .and_then(|f| f.strip_prefix("ptr "))
            .ok_or_else(|| format!("malformed store: {rest}"))?
            .trim();
        let Value::P(handle, offset) = self.value(dest) else {
            return Err(format!("store to a non-pointer: {dest}"));
        };

        let (ty, token) = split_typed(value_part);
        let value = self.value(token);
        if ty.starts_with('<') {
            for (lane, v) in value.vf().iter().enumerate() {
                self.store(handle, offset + lane as i64, *v);
            }
        } else {
            self.store(handle, offset, value.f());
        }
        Ok(())
    }

    fn evaluate(&mut self, body: &str, previous: &str) -> Result<Value, String> {
        let opcode = body.split_whitespace().next().unwrap_or("");

        match opcode {
            "alloca" => {
                let count = body
                    .split('[')
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                Ok(Value::P(self.add_buffer(vec![0.0; count]), 0))
            }

            "call" => {
                if body.contains("@malloc") {
                    let bytes = body
                        .split("i64 ")
                        .nth(1)
                        .and_then(|s| s.trim_end_matches(')').trim().parse::<i64>().ok())
                        .unwrap_or(0);
                    let cells = (bytes / 8).max(0) as usize;
                    return Ok(Value::P(self.add_buffer(vec![0.0; cells]), 0));
                }
                if body.contains("@llvm.pow") {
                    let args = call_arguments(body);
                    let base = self.value(&args[0]);
                    let exponent = self.value(&args[1]);
                    return Ok(if matches!(base, Value::VF(_)) {
                        Value::VF(
                            base.vf()
                                .iter()
                                .zip(exponent.vf())
                                .map(|(b, e)| b.powf(e))
                                .collect(),
                        )
                    } else {
                        Value::F(base.f().powf(exponent.f()))
                    });
                }
                Err(format!("unsupported call: {body}"))
            }

            "inttoptr" => {
                let address = body
                    .split_whitespace()
                    .nth(2)
                    .and_then(|s| s.parse::<i64>().ok())
                    .ok_or_else(|| format!("malformed inttoptr: {body}"))?;
                let handle = *self
                    .external
                    .get(&address)
                    .ok_or_else(|| format!("no buffer bound at address {address}"))?;
                Ok(Value::P(handle, 0))
            }

            "getelementptr" => {
                // `getelementptr inbounds <ty>, ptr <base>, i64 <index>`.
                // The opcode itself ends in "ptr", so the fields are split off
                // by comma rather than by searching for a "ptr " that the
                // mnemonic also contains.
                let fields = split_fields(&body["getelementptr".len()..]);
                let base = fields
                    .get(1)
                    .and_then(|f| f.strip_prefix("ptr "))
                    .ok_or_else(|| format!("malformed gep: {body}"))?
                    .trim();
                let index = fields
                    .get(2)
                    .map(|f| split_typed(f).1)
                    .ok_or_else(|| format!("malformed gep: {body}"))?
                    .trim();
                let Value::P(handle, offset) = self.value(base) else {
                    return Err(format!("gep from a non-pointer: {base}"));
                };
                Ok(Value::P(handle, offset + self.value(index).i()))
            }

            "load" => {
                let ty = body["load ".len()..]
                    .split(',')
                    .next()
                    .unwrap_or("double")
                    .trim()
                    .to_string();
                let source = body
                    .split("ptr ")
                    .nth(1)
                    .and_then(|s| s.split(',').next())
                    .ok_or_else(|| format!("malformed load: {body}"))?
                    .trim();
                let Value::P(handle, offset) = self.value(source) else {
                    return Err(format!("load from a non-pointer: {source}"));
                };
                if let Some(lanes) = vector_width(&ty) {
                    Ok(Value::VF(
                        (0..lanes).map(|l| self.read(handle, offset + l as i64)).collect(),
                    ))
                } else {
                    Ok(Value::F(self.read(handle, offset)))
                }
            }

            "phi" => {
                // `phi <ty> [ v, %a ], [ v, %b ]` — take the arm we arrived from.
                for arm in body.split('[').skip(1) {
                    let inner = arm.split(']').next().unwrap_or("");
                    let mut halves = inner.split(',');
                    let value = halves.next().unwrap_or("").trim();
                    let from = halves.next().unwrap_or("").trim();
                    if from == previous {
                        return Ok(self.value(value));
                    }
                }
                Err(format!("no phi arm for {previous}: {body}"))
            }

            "select" => {
                let parts: Vec<&str> = split_fields(&body["select ".len()..]);
                let condition = self.value(split_typed(parts[0]).1);
                let (then_ty, then_token) = split_typed(parts[1]);
                let then_value = self.value(then_token);
                let else_value = self.value(split_typed(parts[2]).1);
                Ok(if vector_width(then_ty).is_some() {
                    let flags = match condition {
                        Value::VB(f) => f,
                        Value::B(b) => vec![b; then_value.vf().len()],
                        other => return Err(format!("select on {other:?}")),
                    };
                    Value::VF(
                        flags
                            .iter()
                            .zip(then_value.vf())
                            .zip(else_value.vf())
                            .map(|((c, a), b)| if *c { a } else { b })
                            .collect(),
                    )
                } else if matches!(then_value, Value::P(..)) {
                    // `out_effective` picks a buffer when the caller passed none.
                    if condition.b() {
                        then_value
                    } else {
                        else_value
                    }
                } else if matches!(then_value, Value::I(_)) {
                    Value::I(if condition.b() {
                        then_value.i()
                    } else {
                        else_value.i()
                    })
                } else {
                    Value::F(if condition.b() {
                        then_value.f()
                    } else {
                        else_value.f()
                    })
                })
            }

            "insertelement" => {
                // `insertelement <ty> <base>, <ty> <scalar>, <ty> <lane>`
                let fields = split_fields(&body["insertelement".len()..]);
                let lanes = vector_width(split_typed(fields[0]).0).unwrap_or(1);
                let scalar = split_typed(
                    fields.get(1).ok_or_else(|| format!("malformed: {body}"))?,
                )
                .1;
                let mut cells = vec![0i64; lanes];
                cells[0] = self.value(scalar).i();
                Ok(Value::VI(cells))
            }

            "shufflevector" => {
                // Only the splat form is emitted: every lane takes lane 0. The
                // fields are separated by commas; splitting on whitespace lands
                // inside the `<4 x i64>` type instead.
                let fields = split_fields(&body["shufflevector".len()..]);
                let (ty, source) = split_typed(fields[0]);
                let lanes = vector_width(ty).unwrap_or(1);
                let first = self.value(source).vi()[0];
                Ok(Value::VI(vec![first; lanes]))
            }

            "fadd" | "fsub" | "fmul" | "fdiv" => {
                let (ty, a, b) = binary_operands(body, opcode);
                let (x, y) = (self.value(&a), self.value(&b));
                let apply = |p: f64, q: f64| match opcode {
                    "fadd" => p + q,
                    "fsub" => p - q,
                    "fmul" => p * q,
                    _ => p / q,
                };
                Ok(if vector_width(&ty).is_some() {
                    Value::VF(x.vf().iter().zip(y.vf()).map(|(p, q)| apply(*p, q)).collect())
                } else {
                    Value::F(apply(x.f(), y.f()))
                })
            }

            "add" | "sub" | "mul" | "udiv" | "urem" => {
                let (ty, a, b) = binary_operands(body, opcode);
                let (x, y) = (self.value(&a), self.value(&b));
                let apply = |p: i64, q: i64| match opcode {
                    "add" => p.wrapping_add(q),
                    "sub" => p.wrapping_sub(q),
                    "mul" => p.wrapping_mul(q),
                    "udiv" => {
                        if q == 0 {
                            0
                        } else {
                            ((p as u64) / (q as u64)) as i64
                        }
                    }
                    _ => {
                        if q == 0 {
                            0
                        } else {
                            ((p as u64) % (q as u64)) as i64
                        }
                    }
                };
                Ok(if vector_width(&ty).is_some() {
                    Value::VI(x.vi().iter().zip(y.vi()).map(|(p, q)| apply(*p, q)).collect())
                } else {
                    Value::I(apply(x.i(), y.i()))
                })
            }

            "or" => {
                let (_, a, b) = binary_operands(body, opcode);
                Ok(Value::B(self.value(&a).b() || self.value(&b).b()))
            }

            "fcmp" | "icmp" => {
                let mut words = body.split_whitespace();
                words.next();
                let predicate = words.next().unwrap_or("").to_string();
                let rest = body
                    .split_once(&predicate)
                    .map(|(_, r)| r.trim())
                    .unwrap_or("");
                let (ty, a, b) = typed_pair(rest);
                let (x, y) = (self.value(&a), self.value(&b));

                if opcode == "fcmp" {
                    let apply = |p: f64, q: f64| match predicate.as_str() {
                        "ogt" => p > q,
                        "olt" => p < q,
                        "oge" => p >= q,
                        "ole" => p <= q,
                        "oeq" => p == q,
                        other => panic!("unsupported fcmp {other}"),
                    };
                    Ok(if vector_width(&ty).is_some() {
                        Value::VB(x.vf().iter().zip(y.vf()).map(|(p, q)| apply(*p, q)).collect())
                    } else {
                        Value::B(apply(x.f(), y.f()))
                    })
                } else {
                    // Pointer comparisons only ever test for null, and the
                    // machine always passes buffers it owns.
                    if a.trim() == "null" || b.trim() == "null" {
                        let null = |v: &Value| matches!(v, Value::P(h, _) if *h == usize::MAX);
                        let equal = null(&x) == null(&y);
                        return Ok(Value::B(match predicate.as_str() {
                            "eq" => equal,
                            _ => !equal,
                        }));
                    }
                    let apply = |p: i64, q: i64| match predicate.as_str() {
                        "ult" => (p as u64) < (q as u64),
                        "ule" => (p as u64) <= (q as u64),
                        "uge" => (p as u64) >= (q as u64),
                        "ugt" => (p as u64) > (q as u64),
                        "eq" => p == q,
                        "ne" => p != q,
                        other => panic!("unsupported icmp {other}"),
                    };
                    Ok(if vector_width(&ty).is_some() {
                        Value::VB(x.vi().iter().zip(y.vi()).map(|(p, q)| apply(*p, q)).collect())
                    } else {
                        Value::B(apply(x.i(), y.i()))
                    })
                }
            }

            other => Err(format!("unsupported opcode `{other}` in: {body}")),
        }
    }
}

impl Default for Machine {
    fn default() -> Self {
        Machine::new()
    }
}

enum Flow {
    Next,
    Jump(String),
    Return,
}

/// Split on commas that are not inside a `<...>`, `[...]` or `(...)`.
///
/// A vector literal is spelled `<double 0x..., double 0x...>`, so a plain
/// `split(',')` tears it in half and hands the pieces on as operands.
fn split_fields(text: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                fields.push(text[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    fields.push(text[start..].trim());
    fields
}

fn vector_width(ty: &str) -> Option<usize> {
    let ty = ty.trim();
    if !ty.starts_with('<') {
        return None;
    }
    ty.trim_start_matches('<')
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
}

/// Split `<type> <token>` where the type may itself contain spaces.
fn split_typed(text: &str) -> (&str, &str) {
    let text = text.trim();
    if let Some(end) = text.find('>') {
        let (ty, rest) = text.split_at(end + 1);
        return (ty.trim(), rest.trim());
    }
    match text.split_once(' ') {
        Some((ty, token)) => (ty.trim(), token.trim()),
        None => ("", text),
    }
}

/// `<op> <ty> <a>, <b>` — the type is shared by both operands.
fn binary_operands(body: &str, opcode: &str) -> (String, String, String) {
    let rest = body[opcode.len()..].trim();
    let (ty, a, b) = typed_pair(rest);
    (ty, a, b)
}

fn typed_pair(rest: &str) -> (String, String, String) {
    let (ty, tail) = split_typed(rest);
    let fields = split_fields(tail);
    (
        ty.to_string(),
        fields.first().unwrap_or(&"").to_string(),
        fields.get(1).unwrap_or(&"").to_string(),
    )
}

fn call_arguments(body: &str) -> Vec<String> {
    let inner = body
        .split_once('(')
        .and_then(|(_, r)| r.rsplit_once(')'))
        .map(|(a, _)| a)
        .unwrap_or("");
    split_fields(inner)
        .iter()
        .map(|part| split_typed(part).1.to_string())
        .collect()
}

fn parse_literal(token: &str) -> Option<Value> {
    let token = token.trim();

    if token == "zeroinitializer" {
        return Some(Value::VF(vec![0.0; 4]));
    }
    if token == "poison" || token == "undef" {
        return Some(Value::VI(vec![0; 4]));
    }
    if token == "null" {
        // Never a real buffer: the machine always passes pointers it owns.
        return Some(Value::P(usize::MAX, 0));
    }
    if token == "true" {
        return Some(Value::B(true));
    }
    if token == "false" {
        return Some(Value::B(false));
    }

    // A vector literal: `<double 0x..., double 0x...>`
    if let Some(inner) = token.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        let parts = split_fields(inner);
        if parts.first().is_some_and(|p| p.starts_with("double")) {
            return Some(Value::VF(
                parts
                    .iter()
                    .filter_map(|p| parse_double(split_typed(p).1))
                    .collect(),
            ));
        }
        return Some(Value::VI(
            parts
                .iter()
                .filter_map(|p| split_typed(p).1.parse::<i64>().ok())
                .collect(),
        ));
    }

    if let Some(v) = parse_double(token) {
        return Some(Value::F(v));
    }
    token.parse::<i64>().ok().map(Value::I)
}

/// `0x...` is the exact bit pattern; anything else is read as a decimal.
fn parse_double(token: &str) -> Option<f64> {
    let token = token.trim();
    if let Some(hex) = token.strip_prefix("0x") {
        return u64::from_str_radix(hex, 16).ok().map(f64::from_bits);
    }
    if token.contains('.') || token.contains('e') {
        return token.parse::<f64>().ok();
    }
    None
}
