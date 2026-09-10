use crate::ast::*;
use crate::error::{HarmonyDisruption, Result};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};

/// The functions a program has defined so far, by name.
pub type Functions = BTreeMap<String, Function>;

thread_local! {
    /// The functions in scope while a program is being parsed. `parse_expr`
    /// is recursive and called from many places, so the table travels here
    /// rather than through every signature; it is set for the length of one
    /// `parse_rho_program` and holds, while a definition is being read, only
    /// the definitions before it — which is what rules recursion out.
    static FUNCTIONS: RefCell<Functions> = const { RefCell::new(BTreeMap::new()) };
}

fn with_functions<T>(functions: Functions, f: impl FnOnce() -> T) -> T {
    let previous = FUNCTIONS.with(|cell| cell.replace(functions));
    let result = f();
    FUNCTIONS.with(|cell| cell.replace(previous));
    result
}

fn function_named(name: &str) -> Option<Function> {
    FUNCTIONS.with(|cell| cell.borrow().get(name).cloned())
}

/// Validate allowed RHO symbols and character set
pub fn validate_symbols(input: &str) -> Result<()> {
    let code_only = remove_comments(input);

    let allowed_unicode: HashSet<char> = [
        '◯', '□', '▷', '▽', '△', '◇', '◈', '⍳', '⌽', '⍴', '⍉', '↑', '↓', '+', '-', '×', '*', '/', '^', '⌈', '⌊', '|', '→', '⇒', '<', '>', '=', ':', '{', '}', '$', '&', '!',
        '(', ')', '[', ']', ';', ',', '.', ' ', '\t', '\r', '\n', '_', '𝜏', 'τ'
    ].iter().cloned().collect();

    for (line_idx, line) in code_only.lines().enumerate() {
        let line_num = line_idx + 1;
        for (col_idx, ch) in line.char_indices() {
            let col_num = col_idx + 1;

            if ch.is_ascii_alphanumeric() {
                continue;
            }

            if !allowed_unicode.contains(&ch) {
                return Err(HarmonyDisruption::GlyphErr {
                    symbol: ch.to_string(),
                    line: line_num,
                    column: col_num,
                });
            }
        }
    }
    Ok(())
}

/// Check for forbidden control flow keywords (for, while, if, etc.)
pub fn check_forbidden_keywords(input: &str) -> Result<()> {
    let forbidden = ["for", "while", "if", "else", "function", "fn", "var", "let", "const", "class", "return"];
    let code_only = remove_comments(input);

    for (line_idx, line) in code_only.lines().enumerate() {
        let words: Vec<&str> = line.split_whitespace().collect();
        for word in words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
            if forbidden.contains(&clean_word) {
                return Err(HarmonyDisruption::GlyphErr {
                    symbol: clean_word.to_string(),
                    line: line_idx + 1,
                    column: line.find(clean_word).unwrap_or(0) + 1,
                });
            }
        }
    }
    Ok(())
}

/// Normalize ASCII symbol aliases to Unicode RHO topological symbols
pub fn normalize_ascii_aliases(input: &str) -> String {
    input
        // <.> before <>, and both before << and >>, so a scan is not read as a
        // fold and neither is read as two shifts.
        .replace("<.>", "◈")
        // The greater and the lesser of two, spelled after the fold glyphs
        // `◇>` and `◇<` that mean the same thing.
        .replace(">.", "⌈")
        .replace("<.", "⌊")
        .replace("<>", "◇")
        // [] never appears in an address binding, which is always &[0x...].
        .replace("[]", "□")
        .replace("->", "→")
        .replace("=>", "⇒")
        .replace(">>", "▷")
        .replace("<<", "▽")
        .replace("#", "⍳")
        .replace("%", "⌽")
        .replace("\\", "⍴")
        .replace("'", "⍉")
        // Take and drop; a literal therefore needs a digit before its point,
        // as `X ^ .5` would otherwise read as a take.
        .replace("^.", "↑")
        .replace("_.", "↓")
        .replace("@", "&")
}

/// Top-level parser for ρ (RHO) Language source code
pub fn parse_rho_program(input: &str) -> Result<ToposBlock> {
    let normalized_input = normalize_ascii_aliases(input);
    validate_symbols(&normalized_input)?;
    check_forbidden_keywords(&normalized_input)?;
    let clean_code = remove_comments(&normalized_input);

    // Definitions come first and each sees only the ones before it.
    let (raw_definitions, program_lines) = split_definitions(&clean_code)?;
    let mut functions = Functions::new();
    for raw in raw_definitions {
        let function = with_functions(functions.clone(), || parse_definition(&raw))?;
        functions.insert(function.name.clone(), function);
    }

    let (statements, lines) = with_functions(functions.clone(), || {
        let mut statements = Vec::new();
        let mut lines = Vec::new();
        for (line_no, raw) in &program_lines {
            // The topos braces may share a line with a statement.
            let text = raw.trim().trim_start_matches('{').trim_end_matches('}').trim();
            if text.is_empty() {
                continue;
            }
            let parsed = parse_line(text).map_err(|e| match e {
                HarmonyDisruption::IterateErr { detail, line: 0 } => {
                    HarmonyDisruption::IterateErr {
                        detail,
                        line: *line_no,
                    }
                }
                HarmonyDisruption::LoweringErr { detail, line: 0 } => {
                    HarmonyDisruption::LoweringErr {
                        detail,
                        line: *line_no,
                    }
                }
                other => other,
            })?;
            if let Some(stmt) = parsed {
                statements.push(stmt);
                lines.push(*line_no);
            }
        }
        Ok::<_, HarmonyDisruption>((statements, lines))
    })?;

    // A space may not take a function's name: `smooth` alone would then be
    // a space in one place and a function in another.
    for (stmt, line) in statements.iter().zip(&lines) {
        let named = match stmt {
            Statement::SpaceDef(d) => Some(&d.name),
            Statement::ExtBind(b) => Some(&b.space.name),
            Statement::Flow {
                target: FlowTarget::Var(n),
                ..
            } => Some(n),
            _ => None,
        };
        if let Some(name) = named {
            if functions.contains_key(name) {
                return Err(HarmonyDisruption::LoweringErr {
                    detail: format!("`{name}` is a function defined above and cannot also be a space"),
                    line: *line,
                });
            }
        }
    }

    let (statements, lines, origins) = expand_calls(statements, lines, &functions)?;
    let block = ToposBlock {
        statements,
        lines,
        origins,
    };
    let checked = validate_space_declarations(&block)
        .and_then(|_| validate_dimension_shapes(&block))
        .and_then(|_| validate_flow_equilibrium(&block));
    checked.map_err(|e| block.attribute(e))?;

    Ok(block)
}

/// A definition's text before it is parsed: its name, header line, what
/// followed the brace on that line, and the lines of its body.
struct RawDefinition {
    name: String,
    line: usize,
    header_rest: String,
    body: Vec<(usize, String)>,
}

/// `NAME:{ rest`, or None for a line that is not a definition header.
fn definition_header(text: &str, line: usize) -> Option<RawDefinition> {
    let (name, rest) = text.split_once(':')?;
    let name = name.trim();
    if name.is_empty()
        || !name.chars().all(|c| c.is_alphanumeric() || c == '_')
        || name.starts_with(|c: char| c.is_ascii_digit())
    {
        return None;
    }
    let rest = rest.trim_start().strip_prefix('{')?;
    Some(RawDefinition {
        name: name.to_string(),
        line,
        header_rest: rest.to_string(),
        body: Vec::new(),
    })
}

/// The program's lines with their numbers, once the definitions are out.
type ProgramLines = Vec<(usize, String)>;

/// Separate the function definitions from the program that follows them,
/// keeping every line's number for diagnostics.
fn split_definitions(code: &str) -> Result<(Vec<RawDefinition>, ProgramLines)> {
    let mut definitions = Vec::new();
    let mut program = Vec::new();
    let mut current: Option<RawDefinition> = None;

    for (index, raw) in code.lines().enumerate() {
        let line_no = index + 1;
        let text = raw.trim();

        if let Some(def) = current.as_mut() {
            if let Some(pos) = text.find('}') {
                let before = text[..pos].trim();
                if !before.is_empty() {
                    def.body.push((line_no, before.to_string()));
                }
                if !text[pos + 1..].trim().is_empty() {
                    return Err(HarmonyDisruption::LoweringErr {
                        detail: "nothing may follow a definition's closing brace on its line"
                            .to_string(),
                        line: line_no,
                    });
                }
                definitions.push(current.take().unwrap());
            } else if !text.is_empty() {
                def.body.push((line_no, text.to_string()));
            }
            continue;
        }

        if let Some(mut def) = definition_header(text, line_no) {
            // Definitions come before the program, and each before its use.
            if program.iter().any(|(_, l): &(usize, String)| !l.trim().is_empty()) {
                return Err(HarmonyDisruption::LoweringErr {
                    detail: format!(
                        "`{}` is defined after the program began; definitions come first",
                        def.name
                    ),
                    line: line_no,
                });
            }
            if let Some(pos) = def.header_rest.find('}') {
                if !def.header_rest[pos + 1..].trim().is_empty() {
                    return Err(HarmonyDisruption::LoweringErr {
                        detail: "nothing may follow a definition's closing brace on its line"
                            .to_string(),
                        line: line_no,
                    });
                }
                def.header_rest = def.header_rest[..pos].to_string();
                definitions.push(def);
            } else {
                current = Some(def);
            }
            continue;
        }

        program.push((line_no, raw.to_string()));
    }

    if let Some(def) = current {
        return Err(HarmonyDisruption::LoweringErr {
            detail: format!("the definition of `{}` has no closing brace", def.name),
            line: def.line,
        });
    }
    Ok((definitions, program))
}

fn is_identifier(token: &str) -> bool {
    !token.is_empty()
        && !token.starts_with(|c: char| c.is_ascii_digit())
        && token.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Parse one definition. The parameters are the names at the start; what
/// follows on the header line, and the lines below, are the body. A body
/// without a `→` is one expression; otherwise it is flows ending in `→ =`.
fn parse_definition(raw: &RawDefinition) -> Result<Function> {
    let at = |detail: String| HarmonyDisruption::LoweringErr {
        detail,
        line: raw.line,
    };
    if BuiltinOp::ALL.contains(&raw.name.as_str()) || is_tau(&raw.name) {
        return Err(at(format!("`{}` is a function of the language already", raw.name)));
    }
    if function_named(&raw.name).is_some() {
        return Err(at(format!("`{}` is defined twice", raw.name)));
    }

    // Parameters: the leading new names. The list ends at the first token
    // that is not a name, or names a parameter again — so `id:{ X X }` is
    // the identity — and a one-line body that begins with a name nobody has
    // seen would be read as one more parameter, so it needs parentheses.
    let mut rest = raw.header_rest.trim_start();
    let mut params: Vec<String> = Vec::new();
    loop {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let token = &rest[..end];
        if !is_identifier(token) || params.contains(&token.to_string()) {
            break;
        }
        if BuiltinOp::ALL.contains(&token) || is_tau(token) || function_named(token).is_some() {
            return Err(at(format!("a parameter cannot be called `{token}`")));
        }
        params.push(token.to_string());
        rest = rest[end..].trim_start();
    }
    if params.is_empty() {
        return Err(at(format!("`{}` needs at least one parameter", raw.name)));
    }

    let mut body_lines: Vec<(usize, String)> = Vec::new();
    if !rest.is_empty() {
        body_lines.push((raw.line, rest.to_string()));
    }
    body_lines.extend(raw.body.iter().cloned());
    if body_lines.is_empty() {
        return Err(at(format!(
            "`{}` has no body; a one-line body that begins with a name must be parenthesised",
            raw.name
        )));
    }

    let mut visible: HashSet<String> = params.iter().cloned().collect();
    visible.insert("𝜏".to_string());
    visible.insert("τ".to_string());

    let flows = body_lines.iter().any(|(_, text)| text.contains('→') || text.contains('⇒'));
    if !flows {
        if body_lines.len() != 1 {
            return Err(at(format!(
                "`{}` has several lines but no `→`: a body is one expression or flows ending in `→ =`",
                raw.name
            )));
        }
        let (line, text) = &body_lines[0];
        let expr = parse_expr(text).map_err(|e| relined(e, *line))?;
        check_expr_spaces(&expr, &visible, *line).map_err(|e| scoped(e, &raw.name))?;
        return Ok(Function {
            name: raw.name.clone(),
            params,
            body: FunctionBody::Expression(expr, *line),
            line: raw.line,
        });
    }

    let mut statements: Vec<(Statement, usize)> = Vec::new();
    for (line, text) in &body_lines {
        let parsed = parse_line(text).map_err(|e| relined(e, *line))?;
        let Some(stmt) = parsed else {
            return Err(HarmonyDisruption::LoweringErr {
                detail: format!("`{}`: this line is not a flow", raw.name),
                line: *line,
            });
        };
        match &stmt {
            Statement::Flow { src, target } => {
                check_expr_spaces(src, &visible, *line).map_err(|e| scoped(e, &raw.name))?;
                if let FlowTarget::Var(t) = target {
                    if params.contains(t) {
                        return Err(HarmonyDisruption::LoweringErr {
                            detail: format!("`{}` writes to its parameter `{t}`; a function leaves its arguments as they were", raw.name),
                            line: *line,
                        });
                    }
                    visible.insert(t.clone());
                }
            }
            Statement::Iterate { .. } => {
                return Err(HarmonyDisruption::LoweringErr {
                    detail: format!("`{}`: a `⇒` inside a function is not allowed yet", raw.name),
                    line: *line,
                })
            }
            _ => {
                return Err(HarmonyDisruption::LoweringErr {
                    detail: format!("`{}`: only flows may appear in a function's body", raw.name),
                    line: *line,
                })
            }
        }
        statements.push((stmt, *line));
    }
    let ends_well = matches!(
        statements.last(),
        Some((Statement::Flow { target: FlowTarget::Equilibrium, .. }, _))
    );
    let early_end = statements[..statements.len().saturating_sub(1)]
        .iter()
        .any(|(s, _)| matches!(s, Statement::Flow { target: FlowTarget::Equilibrium, .. }));
    if !ends_well || early_end {
        return Err(at(format!(
            "`{}`: a body of flows ends with `→ =`, once, naming its result",
            raw.name
        )));
    }
    Ok(Function {
        name: raw.name.clone(),
        params,
        body: FunctionBody::Flows(statements),
        line: raw.line,
    })
}

/// Give a line to an error raised before one was known.
fn relined(error: HarmonyDisruption, line: usize) -> HarmonyDisruption {
    match error {
        HarmonyDisruption::LoweringErr { detail, line: 0 } => HarmonyDisruption::LoweringErr { detail, line },
        HarmonyDisruption::IterateErr { detail, line: 0 } => HarmonyDisruption::IterateErr { detail, line },
        HarmonyDisruption::GlyphErr { symbol, line: 0, column } => HarmonyDisruption::GlyphErr { symbol, line, column },
        other => other,
    }
}

/// Say why a name is unknown inside a function: it sees nothing of the caller.
fn scoped(error: HarmonyDisruption, function: &str) -> HarmonyDisruption {
    match error {
        HarmonyDisruption::SpaceErr { space_name, line } => HarmonyDisruption::LoweringErr {
            detail: format!(
                "`{space_name}` is not visible inside `{function}`: a function sees its parameters, 𝜏 and constants, nothing of the caller's"
            ),
            line,
        },
        other => other,
    }
}

// ---------------------------------------------------------------- expansion

/// Replace every call with the function's body, the arguments bound. An
/// argument that is not a name or a number is flowed into a space of its own
/// first, so the body may shift it; a body of flows is copied out with its
/// locals renamed apart, its `=` becoming the value of the call. Each copied
/// statement remembers its origin for diagnostics.
struct Expander<'a> {
    functions: &'a Functions,
    calls: usize,
    /// Set when an expression body was inlined into the statement being
    /// expanded, so that statement can carry the origin.
    inlined: Option<Origin>,
}

type Expanded = (Vec<Statement>, Vec<usize>, Vec<Option<Origin>>);

fn expand_calls(statements: Vec<Statement>, lines: Vec<usize>, functions: &Functions) -> Result<Expanded> {
    let mut expander = Expander {
        functions,
        calls: 0,
        inlined: None,
    };
    let mut out: Vec<(Statement, usize, Option<Origin>)> = Vec::new();
    for (stmt, line) in statements.into_iter().zip(lines) {
        expander.inlined = None;
        let rewritten = match stmt {
            Statement::Flow { src, target } => Statement::Flow {
                src: expander.expression(&src, line, &mut out)?,
                target,
            },
            Statement::Constraint(expr) => Statement::Constraint(expander.expression(&expr, line, &mut out)?),
            Statement::AuditTrace(expr) => Statement::AuditTrace(expander.expression(&expr, line, &mut out)?),
            Statement::Iterate { src, target } => {
                // Anything hoisted would run once, before the loop, rather
                // than on every round: not yet.
                let before = out.len();
                let src = expander.expression(&src, line, &mut out)?;
                if out.len() != before {
                    return Err(HarmonyDisruption::IterateErr {
                        detail: "a call inside `⇒` may only be to a function with an expression body, with names or numbers as arguments, in this version".to_string(),
                        line,
                    });
                }
                Statement::Iterate { src, target }
            }
            other => other,
        };
        let origin = expander.inlined.take();
        out.push((rewritten, line, origin));
    }
    let mut statements = Vec::new();
    let mut lines = Vec::new();
    let mut origins = Vec::new();
    for (s, l, o) in out {
        statements.push(s);
        lines.push(l);
        origins.push(o);
    }
    Ok((statements, lines, origins))
}

impl Expander<'_> {
    fn expression(
        &mut self,
        expr: &Expr,
        call_line: usize,
        out: &mut Vec<(Statement, usize, Option<Origin>)>,
    ) -> Result<Expr> {
        Ok(match expr {
            Expr::Call { name, args } => {
                let function = self.functions.get(name).cloned().ok_or_else(|| {
                    HarmonyDisruption::LoweringErr {
                        detail: format!("`{name}` is not a function defined above this point"),
                        line: call_line,
                    }
                })?;
                if args.len() != function.params.len() {
                    return Err(HarmonyDisruption::LoweringErr {
                        detail: format!(
                            "`{name}` takes {} argument(s), not {}; an argument that is a sum or a product needs parentheses",
                            function.params.len(),
                            args.len()
                        ),
                        line: call_line,
                    });
                }
                self.calls += 1;
                let call = self.calls;

                let mut binding: BTreeMap<String, Expr> = BTreeMap::new();
                for (param, arg) in function.params.iter().zip(args) {
                    let arg = self.expression(arg, call_line, out)?;
                    match arg {
                        Expr::Var(_) | Expr::Number(_) => {
                            binding.insert(param.clone(), arg);
                        }
                        other => {
                            let bound = format!("{name}·{call}·{param}");
                            out.push((
                                Statement::Flow {
                                    src: other,
                                    target: FlowTarget::Var(bound.clone()),
                                },
                                call_line,
                                Some(Origin {
                                    function: name.clone(),
                                    body_line: function.line,
                                }),
                            ));
                            binding.insert(param.clone(), Expr::Var(bound));
                        }
                    }
                }

                match &function.body {
                    FunctionBody::Expression(body, body_line) => {
                        let substituted = substitute(body, &binding, &BTreeMap::new());
                        self.inlined = Some(Origin {
                            function: name.clone(),
                            body_line: *body_line,
                        });
                        self.expression(&substituted, call_line, out)?
                    }
                    FunctionBody::Flows(flows) => {
                        let result = format!("{name}·{call}·=");
                        let mut locals: BTreeMap<String, String> = BTreeMap::new();
                        for (stmt, body_line) in flows {
                            let Statement::Flow { src, target } = stmt else {
                                continue;
                            };
                            let src = substitute(src, &binding, &locals);
                            let src = self.expression(&src, call_line, out)?;
                            let renamed = match target {
                                FlowTarget::Var(t) => {
                                    let fresh = format!("{name}·{call}·{t}");
                                    locals.insert(t.clone(), fresh.clone());
                                    fresh
                                }
                                FlowTarget::Equilibrium => result.clone(),
                            };
                            out.push((
                                Statement::Flow {
                                    src,
                                    target: FlowTarget::Var(renamed),
                                },
                                call_line,
                                Some(Origin {
                                    function: name.clone(),
                                    body_line: *body_line,
                                }),
                            ));
                        }
                        Expr::Var(result)
                    }
                }
            }
            Expr::Var(_) | Expr::Number(_) => expr.clone(),
            Expr::Shift { dir, axis, operand } => Expr::Shift {
                dir: *dir,
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::BinaryOp { op, lhs, rhs } => Expr::BinaryOp {
                op: op.clone(),
                lhs: Box::new(self.expression(lhs, call_line, out)?),
                rhs: Box::new(self.expression(rhs, call_line, out)?),
            },
            Expr::Builtin { op, operand } => Expr::Builtin {
                op: *op,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Lift { axis, operand } => Expr::Lift {
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Scan { op, axis, operand } => Expr::Scan {
                op: *op,
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Reduce { op, axis, operand } => Expr::Reduce {
                op: *op,
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Rotate { by, axis, operand } => Expr::Rotate {
                by: *by,
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Reverse { axis, operand } => Expr::Reverse {
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Reshape { shape, operand } => Expr::Reshape {
                shape: shape.clone(),
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Transpose { axes, operand } => Expr::Transpose {
                axes: axes.clone(),
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Take { count, axis, operand } => Expr::Take {
                count: *count,
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Drop { count, axis, operand } => Expr::Drop {
                count: *count,
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::Index { axis, operand } => Expr::Index {
                axis: *axis,
                operand: Box::new(self.expression(operand, call_line, out)?),
            },
            Expr::AuditTrace(inner) => {
                Expr::AuditTrace(Box::new(self.expression(inner, call_line, out)?))
            }
        })
    }
}

/// A body with its parameters replaced by what they were bound to and its
/// locals by their renamed selves. Nothing else in a body is a name of the
/// caller's, since the definition was checked to see only its own.
fn substitute(expr: &Expr, binding: &BTreeMap<String, Expr>, locals: &BTreeMap<String, String>) -> Expr {
    let sub = |e: &Expr| Box::new(substitute(e, binding, locals));
    match expr {
        Expr::Var(name) => {
            if let Some(bound) = binding.get(name) {
                bound.clone()
            } else if let Some(renamed) = locals.get(name) {
                Expr::Var(renamed.clone())
            } else {
                expr.clone()
            }
        }
        Expr::Number(_) => expr.clone(),
        Expr::Call { name, args } => Expr::Call {
            name: name.clone(),
            args: args.iter().map(|a| substitute(a, binding, locals)).collect(),
        },
        Expr::Shift { dir, axis, operand } => Expr::Shift { dir: *dir, axis: *axis, operand: sub(operand) },
        Expr::BinaryOp { op, lhs, rhs } => Expr::BinaryOp { op: op.clone(), lhs: sub(lhs), rhs: sub(rhs) },
        Expr::Builtin { op, operand } => Expr::Builtin { op: *op, operand: sub(operand) },
        Expr::Lift { axis, operand } => Expr::Lift { axis: *axis, operand: sub(operand) },
        Expr::Scan { op, axis, operand } => Expr::Scan { op: *op, axis: *axis, operand: sub(operand) },
        Expr::Reduce { op, axis, operand } => Expr::Reduce { op: *op, axis: *axis, operand: sub(operand) },
        Expr::Rotate { by, axis, operand } => Expr::Rotate { by: *by, axis: *axis, operand: sub(operand) },
        Expr::Reverse { axis, operand } => Expr::Reverse { axis: *axis, operand: sub(operand) },
        Expr::Reshape { shape, operand } => Expr::Reshape { shape: shape.clone(), operand: sub(operand) },
        Expr::Transpose { axes, operand } => Expr::Transpose { axes: axes.clone(), operand: sub(operand) },
        Expr::Take { count, axis, operand } => Expr::Take { count: *count, axis: *axis, operand: sub(operand) },
        Expr::Drop { count, axis, operand } => Expr::Drop { count: *count, axis: *axis, operand: sub(operand) },
        Expr::Index { axis, operand } => Expr::Index { axis: *axis, operand: sub(operand) },
        Expr::AuditTrace(inner) => Expr::AuditTrace(sub(inner)),
    }
}

/// Split a call's arguments: names, numbers and parenthesised expressions,
/// one after another.
fn split_operands(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in text.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            c if c.is_whitespace() && depth == 0 => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn is_tau(name: &str) -> bool {
    name == "𝜏" || name == "τ"
}

/// The word an expression starts with, and what follows it.
fn leading_word(text: &str) -> Option<(&str, &str)> {
    let end = text
        .char_indices()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let word = &text[..end];
    if !is_identifier(word) {
        return None;
    }
    Some((word, &text[end..]))
}

fn remove_comments(input: &str) -> String {
    let mut result = String::new();
    let mut in_block_comment = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if in_block_comment {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                in_block_comment = false;
            } else if c == '\n' {
                // Keep the line structure intact for diagnostics.
                result.push(c);
            }
            continue;
        }

        if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            in_block_comment = true;
            continue;
        }

        if c == '/' && chars.peek() == Some(&'/') {
            while let Some(&nc) = chars.peek() {
                if nc == '\n' {
                    break;
                }
                chars.next();
            }
            continue;
        }

        result.push(c);
    }
    result
}

fn parse_line(line: &str) -> Result<Option<Statement>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }

    // 1. External Binding: &[0x7A4F]:INPUT:◯ □ 1024 1024
    if line.starts_with('&') {
        return Ok(Some(Statement::ExtBind(parse_ext_binding(line)?)));
    }

    // 2. Constraint Solver: ! (OUTPUT >= 0)
    if let Some(rest) = line.strip_prefix('!') {
        let expr_str = rest.trim();
        let expr_str = expr_str.trim_matches(|c| c == '(' || c == ')');
        let expr = parse_expr(expr_str)?;
        return Ok(Some(Statement::Constraint(expr)));
    }

    // 3. Audit Trace: $ Expr
    if let Some(rest) = line.strip_prefix('$') {
        let expr_str = rest.trim();
        let expr = parse_expr(expr_str)?;
        return Ok(Some(Statement::AuditTrace(expr)));
    }

    // 4. Space Declaration: [Name]:◯ □ 1024 1024
    if line.contains(":◯ □") || line.contains(":◯□") {
        return Ok(Some(Statement::SpaceDef(parse_space_decl(line)?)));
    }

    // 5. Fixed point: Expr ⇒ Name. The target is a space, never `=`: the
    // equilibrium point is where a program ends, and an iteration has to be
    // read from afterwards.
    if line.contains('⇒') {
        let parts: Vec<&str> = line.split('⇒').collect();
        if parts.len() != 2 {
            return Err(HarmonyDisruption::IterateErr {
                detail: "a line iterates one expression into one space: `expr ⇒ NAME`"
                    .to_string(),
                line: 0,
            });
        }
        let src = parse_expr(parts[0].trim())?;
        let target = parts[1].trim();
        if target == "=" {
            return Err(HarmonyDisruption::IterateErr {
                detail: "`⇒` cannot flow into `=`; iterate into a space and then send it on with `NAME → =`"
                    .to_string(),
                line: 0,
            });
        }
        if target.is_empty() || !target.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(HarmonyDisruption::IterateErr {
                detail: format!("`⇒` needs a space to iterate, not '{target}'"),
                line: 0,
            });
        }
        return Ok(Some(Statement::Iterate {
            src,
            target: target.to_string(),
        }));
    }

    // 6. Flow Statement: Expr → Target
    if line.contains('→') {
        let parts: Vec<&str> = line.split('→').collect();
        if parts.len() == 2 {
            let src_expr = parse_expr(parts[0].trim())?;
            let target_str = parts[1].trim();
            let target = if target_str == "=" {
                FlowTarget::Equilibrium
            } else {
                FlowTarget::Var(target_str.to_string())
            };
            return Ok(Some(Statement::Flow {
                src: src_expr,
                target,
            }));
        }
    }

    Ok(None)
}

fn parse_ext_binding(line: &str) -> Result<ExternalBinding> {
    let parts: Vec<&str> = line.split(':').collect();
    if parts.len() < 3 {
        return Err(HarmonyDisruption::GlyphErr {
            symbol: line.to_string(),
            line: 1,
            column: 1,
        });
    }

    let addr_part = parts[0].trim().trim_start_matches("&[").trim_end_matches(']');
    let addr = u64::from_str_radix(addr_part.trim_start_matches("0x"), 16).unwrap_or(0);

    let space_name = parts[1].trim().to_string();

    let dims_part = parts[2].replace("◯", "").replace("□", "");
    let dimensions: Vec<usize> = dims_part
        .split_whitespace()
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();

    Ok(ExternalBinding {
        address: addr,
        space: SpaceDecl {
            name: space_name,
            dimensions,
        },
    })
}

fn parse_space_decl(line: &str) -> Result<SpaceDecl> {
    let parts: Vec<&str> = line.split(':').collect();
    let name = parts[0].trim().to_string();
    let rest = parts[1].replace("◯", "").replace("□", "");
    let dimensions: Vec<usize> = rest
        .split_whitespace()
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();

    Ok(SpaceDecl { name, dimensions })
}

/// Topological Expression Parser
pub fn parse_expr(expr_str: &str) -> Result<Expr> {
    let expr_str = expr_str.trim();

    if is_enclosed_by_outer_parens(expr_str) {
        let inner = &expr_str[1..expr_str.len() - 1];
        return parse_expr(inner);
    }

    let operators = [
        (">=", BinaryOpKind::Gte),
        ("<=", BinaryOpKind::Lte),
        ("==", BinaryOpKind::Eq),
        (">", BinaryOpKind::Gt),
        ("<", BinaryOpKind::Lt),
        ("+", BinaryOpKind::Add),
        ("-", BinaryOpKind::Sub),
        // Tighter than a sum, looser than a product: `A + B ⌈ C × D` is
        // `A + (B ⌈ (C × D))`.
        ("⌈", BinaryOpKind::Max),
        ("⌊", BinaryOpKind::Min),
        ("|", BinaryOpKind::Residue),
        ("×", BinaryOpKind::Mul),
        ("*", BinaryOpKind::Mul),
        ("/", BinaryOpKind::Div),
        ("^", BinaryOpKind::Pow),
    ];

    for (op_str, op_kind) in &operators {
        if let Some(pos) = find_binary_op_position(expr_str, op_str) {
            let lhs_str = expr_str[..pos].trim();
            let rhs_str = expr_str[pos + op_str.len()..].trim();
            
            let lhs = parse_expr(lhs_str)?;
            let rhs = parse_expr(rhs_str)?;
            return Ok(Expr::BinaryOp {
                op: op_kind.clone(),
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });
        }
    }

    // Rotate: `k ⌽ X`, `k ⌽0 X`. Binds tighter than every arithmetic operator,
    // as the prefix glyphs do, so `A + 1 ⌽ X` is `A + (1 ⌽ X)`. The amount is
    // a whole number written as a literal: a rotation is fixed at compile
    // time, like a shift's direction.
    if let Some(pos) = find_binary_op_position(expr_str, "⌽") {
        let amount = parse_expr(expr_str[..pos].trim())?;
        let by = match amount {
            Expr::Number(v) if v == v.trunc() && v.abs() <= 1e9 => v as i64,
            _ => {
                return Err(HarmonyDisruption::LoweringErr {
                    detail: format!(
                        "`⌽` rotates by a whole number written as a literal, not by `{}`",
                        expr_str[..pos].trim()
                    ),
                    line: 0,
                })
            }
        };
        let after = &expr_str[pos + '⌽'.len_utf8()..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        let operand_str = after[digits.len()..].trim();
        if operand_str.is_empty() {
            return Err(HarmonyDisruption::LoweringErr {
                detail: "`⌽` needs a space to rotate".to_string(),
                line: 0,
            });
        }
        return Ok(Expr::Rotate {
            by,
            axis: if digits.is_empty() { None } else { digits.parse().ok() },
            operand: Box::new(parse_expr(operand_str)?),
        });
    }

    // A named function: `exp X`, `ind (A > B)`. The name has to be followed by
    // whitespace or a bracket, so a space called `expansion` stays a space.
    for name in BuiltinOp::ALL {
        let Some(rest) = expr_str.strip_prefix(name) else {
            continue;
        };
        if !rest.starts_with(|c: char| c.is_whitespace() || c == '(') {
            continue;
        }
        let operand = rest.trim();
        if operand.is_empty() {
            continue;
        }
        return Ok(Expr::Builtin {
            op: BuiltinOp::from_name(name).unwrap(),
            operand: Box::new(parse_expr(operand)?),
        });
    }

    // A function of the program's own: `smooth X`, `blend A (B + C)`. The
    // arguments are names, numbers or parenthesised expressions, one after
    // another; an unbracketed sum would have no end the parser could find.
    if let Some((word, rest)) = leading_word(expr_str) {
        if let Some(function) = function_named(word) {
            let operands = split_operands(rest);
            if operands.len() != function.params.len() {
                return Err(HarmonyDisruption::LoweringErr {
                    detail: format!(
                        "`{word}` takes {} argument(s), not {}; an argument that is a sum or a product needs parentheses",
                        function.params.len(),
                        operands.len()
                    ),
                    line: 0,
                });
            }
            let args = operands
                .iter()
                .map(|o| parse_expr(o))
                .collect::<Result<Vec<Expr>>>()?;
            return Ok(Expr::Call {
                name: word.to_string(),
                args,
            });
        }
        // A word applied to something is a call, and this one is to nothing
        // defined above: say so, rather than reporting a space named `f X`.
        if !rest.trim().is_empty()
            && !BuiltinOp::ALL.contains(&word)
            && !is_tau(word)
            && rest.starts_with(|c: char| c.is_whitespace() || c == '(')
        {
            return Err(HarmonyDisruption::LoweringErr {
                detail: format!(
                    "`{word}` is not a function defined above this point (definitions come before their use)"
                ),
                line: 0,
            });
        }
    }

    // Lift: □2X views X with a length-1 axis inserted at position 2.
    if expr_str.starts_with('□') && expr_str.chars().count() > 1 {
        let rest = &expr_str['□'.len_utf8()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let operand_str = rest[digits.len()..].trim();
        if !digits.is_empty() && !operand_str.is_empty() {
            return Ok(Expr::Lift {
                axis: digits.parse().unwrap_or(0),
                operand: Box::new(parse_expr(operand_str)?),
            });
        }
    }

    // Fold and scan share a shape: glyph, operator, optional axis, operand.
    for (glyph, scanning) in [('◇', false), ('◈', true)] {
        if !expr_str.starts_with(glyph) || expr_str.chars().count() <= 2 {
            continue;
        }
        let rest = &expr_str[glyph.len_utf8()..];
        let mut chars = rest.chars();
        let op_char = chars.next().unwrap();
        let op = match op_char {
            '+' => BinaryOpKind::Add,
            '×' | '*' => BinaryOpKind::Mul,
            '>' => BinaryOpKind::Gt,
            '<' => BinaryOpKind::Lt,
            other => {
                return Err(HarmonyDisruption::GlyphErr {
                    symbol: format!("◇{other}"),
                    line: 0,
                    column: 0,
                })
            }
        };
        let fold = FoldOp::from_op(&op).ok_or_else(|| HarmonyDisruption::GlyphErr {
            symbol: format!("◇{op_char}"),
            line: 0,
            column: 0,
        })?;
        let after_op = &rest[op_char.len_utf8()..];
        let digits: String = after_op.chars().take_while(char::is_ascii_digit).collect();
        let operand_str = after_op[digits.len()..].trim();
        if !operand_str.is_empty() {
            let axis = if digits.is_empty() { None } else { digits.parse().ok() };
            let operand = Box::new(parse_expr(operand_str)?);
            return Ok(if scanning {
                Expr::Scan { op: fold, axis, operand }
            } else {
                Expr::Reduce { op: fold, axis, operand }
            });
        }
    }

    // Unary shift operators ▷, ▽, optionally pinned to an axis: ▷0X, ▽1X.
    // Bare ▷X shifts along the last (contiguous) axis.
    for (glyph, dir) in [('▷', ShiftDir::Positive), ('▽', ShiftDir::Negative)] {
        if !expr_str.starts_with(glyph) || expr_str.chars().count() <= 1 {
            continue;
        }
        let rest = &expr_str[glyph.len_utf8()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let operand_str = rest[digits.len()..].trim();
        if operand_str.is_empty() {
            break;
        }
        return Ok(Expr::Shift {
            dir,
            axis: if digits.is_empty() { None } else { digits.parse().ok() },
            operand: Box::new(parse_expr(operand_str)?),
        });
    }

    // Take and drop: `2 ↑ X`, `-1 ↓0 X`. The count is a whole number written
    // as a literal, so the shape that results is known where every shape is.
    for (glyph, dropping) in [('↑', false), ('↓', true)] {
        let Some(pos) = find_binary_op_position(expr_str, &glyph.to_string()) else {
            continue;
        };
        let written = expr_str[..pos].trim();
        let count = match parse_expr(written)? {
            Expr::Number(v) if v == v.trunc() && v != 0.0 && v.abs() <= 1e9 => v as i64,
            _ => {
                return Err(HarmonyDisruption::LoweringErr {
                    detail: format!(
                        "`{glyph}` takes a whole number of cells written as a literal, not `{written}`"
                    ),
                    line: 0,
                })
            }
        };
        let after = &expr_str[pos + glyph.len_utf8()..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        let operand_str = after[digits.len()..].trim();
        if operand_str.is_empty() {
            return Err(HarmonyDisruption::LoweringErr {
                detail: format!("`{glyph}` needs a space on its right"),
                line: 0,
            });
        }
        let axis = if digits.is_empty() { None } else { digits.parse().ok() };
        let operand = Box::new(parse_expr(operand_str)?);
        return Ok(if dropping {
            Expr::Drop { count, axis, operand }
        } else {
            Expr::Take { count, axis, operand }
        });
    }
    if expr_str.starts_with('↑') || expr_str.starts_with('↓') {
        return Err(HarmonyDisruption::LoweringErr {
            detail: "`↑` and `↓` take a count on their left: `2 ↑ X`, `-1 ↓ X`".to_string(),
            line: 0,
        });
    }

    // Reshape: `2 3 ⍴ X`. The shape is a list of whole numbers written as
    // literals, so the result's shape is known where every shape is: at
    // compile time. Binds as tightly as the prefix glyphs.
    if let Some(pos) = find_binary_op_position(expr_str, "⍴") {
        let written = expr_str[..pos].trim();
        let dims: Option<Vec<usize>> = written
            .split_whitespace()
            .map(|t| t.parse::<usize>().ok().filter(|d| *d >= 1))
            .collect();
        let Some(shape) = dims.filter(|d| !d.is_empty()) else {
            return Err(HarmonyDisruption::LoweringErr {
                detail: format!(
                    "`⍴` reshapes to a list of whole numbers written as literals, not to `{written}`"
                ),
                line: 0,
            });
        };
        let operand_str = expr_str[pos + '⍴'.len_utf8()..].trim();
        if operand_str.is_empty() {
            return Err(HarmonyDisruption::LoweringErr {
                detail: "`⍴` needs a space to reshape".to_string(),
                line: 0,
            });
        }
        return Ok(Expr::Reshape {
            shape,
            operand: Box::new(parse_expr(operand_str)?),
        });
    }

    // Transpose with a permutation: `1 0 ⍉ X`, source axis k to result axis
    // P[k]. Checked against the operand's rank once shapes are known.
    if let Some(pos) = find_binary_op_position(expr_str, "⍉") {
        let written = expr_str[..pos].trim();
        let axes: Option<Vec<usize>> = written
            .split_whitespace()
            .map(|t| t.parse::<usize>().ok())
            .collect();
        let Some(axes) = axes.filter(|a| !a.is_empty()) else {
            return Err(HarmonyDisruption::LoweringErr {
                detail: format!(
                    "`⍉` takes a permutation of the axes written as literals, not `{written}`"
                ),
                line: 0,
            });
        };
        let operand_str = expr_str[pos + '⍉'.len_utf8()..].trim();
        if operand_str.is_empty() {
            return Err(HarmonyDisruption::LoweringErr {
                detail: "`⍉` needs a space to transpose".to_string(),
                line: 0,
            });
        }
        return Ok(Expr::Transpose {
            axes: Some(axes),
            operand: Box::new(parse_expr(operand_str)?),
        });
    }

    // Transpose alone: ⍉X reverses the axes.
    if expr_str.starts_with('⍉') && expr_str.chars().count() > 1 {
        let operand_str = expr_str['⍉'.len_utf8()..].trim();
        if !operand_str.is_empty() {
            return Ok(Expr::Transpose {
                axes: None,
                operand: Box::new(parse_expr(operand_str)?),
            });
        }
    }

    // Reverse: ⌽X reads the cell at the other end of the axis; ⌽0X names it.
    if expr_str.starts_with('⌽') && expr_str.chars().count() > 1 {
        let rest = &expr_str['⌽'.len_utf8()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let operand_str = rest[digits.len()..].trim();
        if !operand_str.is_empty() {
            return Ok(Expr::Reverse {
                axis: if digits.is_empty() { None } else { digits.parse().ok() },
                operand: Box::new(parse_expr(operand_str)?),
            });
        }
    }

    // Index: ⍳X is the coordinate of each cell of X along an axis, from
    // zero; ⍳0X names the axis.
    if expr_str.starts_with('⍳') && expr_str.chars().count() > 1 {
        let rest = &expr_str['⍳'.len_utf8()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let operand_str = rest[digits.len()..].trim();
        if !operand_str.is_empty() {
            return Ok(Expr::Index {
                axis: if digits.is_empty() { None } else { digits.parse().ok() },
                operand: Box::new(parse_expr(operand_str)?),
            });
        }
    }

    // Unary audit tracer $
    if expr_str.starts_with('$') {
        let sub = parse_expr(expr_str['$'.len_utf8()..].trim())?;
        return Ok(Expr::AuditTrace(Box::new(sub)));
    }

    // Numeric literals
    if let Ok(num) = expr_str.parse::<f64>() {
        return Ok(Expr::Number(num));
    }

    // Variable / Space identifier
    if !expr_str.is_empty() {
        return Ok(Expr::Var(expr_str.to_string()));
    }

    Err(HarmonyDisruption::GlyphErr {
        symbol: expr_str.to_string(),
        line: 1,
        column: 1,
    })
}

fn find_binary_op_position(s: &str, op: &str) -> Option<usize> {
    let mut depth = 0;
    let op_bytes = op.as_bytes();

    for (i, c) in s.char_indices().rev() {
        if c == ')' {
            depth += 1;
        } else if c == '(' {
            depth -= 1;
        } else if depth == 0 && s.as_bytes()[i..].starts_with(op_bytes) {
            // The operator right after ◇ or ◈ names the fold, so it is part of
            // the glyph rather than a binary operator splitting the expression.
            if s[..i].ends_with('◇') || s[..i].ends_with('◈') {
                continue;
            }
            // A + or - that follows another operator is a sign on the number to
            // its right, not a split point. Without this, `A × -3.0` breaks at
            // the minus and leaves `A ×` behind as if it were a name.
            if matches!(op, "+" | "-" | "⌽" | "⍉") && is_sign_position(&s[..i]) {
                continue;
            }
            let lhs = s[..i].trim();
            let rhs = s[i + op_bytes.len()..].trim();
            if !lhs.is_empty() && !rhs.is_empty() {
                return Some(i);
            }
        }
    }
    None
}

/// Whether a `+` or `-` at the end of `before` would be a sign rather than a
/// binary operator: nothing precedes it, or what does is itself an operator.
fn is_sign_position(before: &str) -> bool {
    let trimmed = before.trim_end();

    // An axis index can sit between a glyph and the sign, as in `◈×0 -3.0`.
    // Digits only follow a glyph there, so stepping over them and finding one
    // settles it.
    let stripped = trimmed.trim_end_matches(|c: char| c.is_ascii_digit());
    if stripped.len() < trimmed.len() {
        let mut tail = stripped.chars();
        let last = tail.next_back();
        let before_last = tail.next_back();
        if matches!(last, Some('▷' | '▽' | '□' | '⍳' | '⌽' | '↑' | '↓'))
            || (matches!(last, Some('+' | '-' | '×' | '*' | '>' | '<'))
                && matches!(before_last, Some('◇' | '◈')))
        {
            return true;
        }
    }

    match trimmed.chars().next_back() {
        None => true,
        Some(c) => matches!(
            c,
            '+' | '-' | '×' | '*' | '/' | '^' | '⌈' | '⌊' | '|' | '>' | '<' | '=' | '('
                | ':' | '→' | '◇' | '◈' | '▷' | '▽' | '□' | '⍳' | '⌽' | '⍴' | '⍉' | '↑' | '↓'
                | '!' | '$'
        ),
    }
}

fn is_enclosed_by_outer_parens(s: &str) -> bool {
    if !s.starts_with('(') || !s.ends_with(')') {
        return false;
    }
    let mut depth = 0;
    let char_count = s.chars().count();
    for (i, c) in s.chars().enumerate() {
        if c == '(' {
            depth += 1;
        } else if c == ')' {
            depth -= 1;
            if depth == 0 && i < char_count - 1 {
                return false;
            }
        }
    }
    depth == 0
}

/// Static space declaration validation (Undeclared space throws HarmonyDisruption::SpaceErr)
pub fn validate_space_declarations(block: &ToposBlock) -> Result<()> {
    let mut declared_spaces = HashSet::new();

    // A space that shadowed a function name would make `exp X` ambiguous.
    for (index, stmt) in block.statements.iter().enumerate() {
        let name = match stmt {
            Statement::SpaceDef(d) => Some(&d.name),
            Statement::ExtBind(b) => Some(&b.space.name),
            Statement::Flow {
                target: FlowTarget::Var(n),
                ..
            }
            | Statement::Iterate { target: n, .. } => Some(n),
            _ => None,
        };
        if let Some(name) = name {
            if BuiltinOp::ALL.contains(&name.as_str()) {
                return Err(HarmonyDisruption::GlyphErr {
                    symbol: name.clone(),
                    line: block.line_of(index),
                    column: 1,
                });
            }
        }
    }

    for stmt in &block.statements {
        match stmt {
            Statement::SpaceDef(decl) => {
                declared_spaces.insert(decl.name.clone());
            }
            Statement::ExtBind(bind) => {
                declared_spaces.insert(bind.space.name.clone());
            }
            _ => {}
        }
    }

    // Spaces some flow has written so far. A `⇒` may only iterate one of
    // these: its starting value is part of what it computes, so the program
    // has to have spelled that value out.
    let mut written: HashSet<String> = HashSet::new();

    for (index, stmt) in block.statements.iter().enumerate() {
        let line = block.line_of(index);
        match stmt {
            Statement::Flow { src, target } => {
                check_expr_spaces(src, &declared_spaces, line)?;
                match target {
                    FlowTarget::Var(var_name) => {
                        declared_spaces.insert(var_name.clone());
                        written.insert(var_name.clone());
                    }
                    FlowTarget::Equilibrium => {
                        written.insert("OUTPUT".to_string());
                    }
                }
            }
            Statement::Iterate { src, target } => {
                check_expr_spaces(src, &declared_spaces, line)?;
                if !written.contains(target) {
                    return Err(HarmonyDisruption::IterateErr {
                        detail: format!(
                            "`⇒ {target}` needs a starting value: write {target} with a flow first, \
                             e.g. `INPUT → {target}`"
                        ),
                        line,
                    });
                }
            }
            Statement::Constraint(expr) | Statement::AuditTrace(expr) => {
                check_expr_spaces(expr, &declared_spaces, line)?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn check_expr_spaces(expr: &Expr, declared: &HashSet<String>, line: usize) -> Result<()> {
    match expr {
        Expr::Var(name) => {
            if name != "𝜏" && name != "τ" && !declared.contains(name) {
                return Err(HarmonyDisruption::SpaceErr {
                    space_name: name.clone(),
                    line,
                });
            }
        }
        Expr::Shift { operand: inner, .. }
        | Expr::Reduce { operand: inner, .. }
        | Expr::Scan { operand: inner, .. }
        | Expr::Builtin { operand: inner, .. }
        | Expr::Lift { operand: inner, .. }
        | Expr::Index { operand: inner, .. }
        | Expr::Rotate { operand: inner, .. }
        | Expr::Reverse { operand: inner, .. }
        | Expr::Reshape { operand: inner, .. }
        | Expr::Transpose { operand: inner, .. }
        | Expr::Take { operand: inner, .. }
        | Expr::Drop { operand: inner, .. }
        | Expr::AuditTrace(inner) => {
            check_expr_spaces(inner, declared, line)?;
        }
        Expr::BinaryOp { lhs, rhs, .. } => {
            check_expr_spaces(lhs, declared, line)?;
            check_expr_spaces(rhs, declared, line)?;
        }
        Expr::Call { args, .. } => {
            for arg in args {
                check_expr_spaces(arg, declared, line)?;
            }
        }
        Expr::Number(_) => {}
    }
    Ok(())
}

/// Static equilibrium point validation (Missing = throws HarmonyDisruption::FlowErr)
pub fn validate_flow_equilibrium(block: &ToposBlock) -> Result<()> {
    let has_eq = block.statements.iter().any(|stmt| match stmt {
        Statement::Flow { target, .. } => *target == FlowTarget::Equilibrium,
        _ => false,
    });

    if !has_eq {
        return Err(HarmonyDisruption::FlowErr);
    }
    Ok(())
}

/// Static shape and dimension mismatch check
pub fn validate_dimension_shapes(block: &ToposBlock) -> Result<()> {
    use std::collections::BTreeMap;

    let mut space_shapes = BTreeMap::new();

    // 1. Gather initial declared shapes
    for stmt in &block.statements {
        match stmt {
            Statement::SpaceDef(decl) => {
                space_shapes.insert(decl.name.clone(), decl.dimensions.clone());
            }
            Statement::ExtBind(bind) => {
                space_shapes.insert(bind.space.name.clone(), bind.space.dimensions.clone());
            }
            _ => {}
        }
    }

    // The one shape inference, shared with the compiler and the analysis.
    fn get_expr_shape(expr: &Expr, shapes: &BTreeMap<String, Vec<usize>>) -> Option<Vec<usize>> {
        expr_shape(expr, shapes)
    }

    // 2. Operands that cannot stretch against one another are an error, not an
    // unknown shape. Reporting it here names the line rather than letting the
    // mismatch surface as a silently skipped check.
    fn check_broadcast(
        expr: &Expr,
        shapes: &BTreeMap<String, Vec<usize>>,
        line: usize,
    ) -> Result<()> {
        match expr {
            Expr::BinaryOp { lhs, rhs, .. } => {
                check_broadcast(lhs, shapes, line)?;
                check_broadcast(rhs, shapes, line)?;
                if let (Some(l), Some(r)) =
                    (get_expr_shape(lhs, shapes), get_expr_shape(rhs, shapes))
                {
                    if broadcast_shapes(&l, &r).is_none() {
                        return Err(HarmonyDisruption::DimensionErr {
                            space_a: format!("{}", crate::symbolic::ExprGlyphs(lhs)),
                            shape_a: l,
                            space_b: format!("{}", crate::symbolic::ExprGlyphs(rhs)),
                            shape_b: r,
                            line,
                        });
                    }
                }
            }
            Expr::Shift { operand: inner, .. }
            | Expr::Reduce { operand: inner, .. }
            | Expr::Scan { operand: inner, .. }
            | Expr::Builtin { operand: inner, .. }
            | Expr::Lift { operand: inner, .. }
            | Expr::Index { operand: inner, .. }
            | Expr::Rotate { operand: inner, .. }
            | Expr::Reverse { operand: inner, .. }
            | Expr::Reshape { operand: inner, .. }
            | Expr::AuditTrace(inner) => check_broadcast(inner, shapes, line)?,
            Expr::Call { args, .. } => {
                for arg in args {
                    check_broadcast(arg, shapes, line)?;
                }
            }
            // A take or drop that leaves nothing, or names an axis the
            // operand has not got, is an error here with a line.
            Expr::Take { count, axis, operand } | Expr::Drop { count, axis, operand } => {
                check_broadcast(operand, shapes, line)?;
                let dropping = matches!(expr, Expr::Drop { .. });
                if let Some(inner) = get_expr_shape(operand, shapes) {
                    if taken_shape(&inner, *axis, *count, dropping).is_none() {
                        return Err(HarmonyDisruption::LoweringErr {
                            detail: format!(
                                "`{} {}` of {inner:?} leaves nothing, or names an axis it has not got",
                                count,
                                if dropping { "↓" } else { "↑" }
                            ),
                            line,
                        });
                    }
                }
            }
            // A permutation that does not fit the operand is an error here,
            // with a line, rather than an unknown shape later.
            Expr::Transpose { axes, operand } => {
                check_broadcast(operand, shapes, line)?;
                if let Some(inner) = get_expr_shape(operand, shapes) {
                    if transpose_axes(inner.len(), axes.as_deref()).is_none() {
                        return Err(HarmonyDisruption::LoweringErr {
                            detail: format!(
                                "`⍉` needs a permutation of the {} axes of {:?}, not {:?}",
                                inner.len(),
                                inner,
                                axes.clone().unwrap_or_default()
                            ),
                            line,
                        });
                    }
                }
            }
            Expr::Var(_) | Expr::Number(_) => {}
        }
        Ok(())
    }

    for (index, stmt) in block.statements.iter().enumerate() {
        match stmt {
            Statement::Flow { src, .. }
            | Statement::Iterate { src, .. }
            | Statement::Constraint(src) => {
                check_broadcast(src, &space_shapes, block.line_of(index))?
            }
            _ => {}
        }
    }

    // 3. Validate shapes across flows
    for (index, stmt) in block.statements.iter().enumerate() {
        // An iteration writes back into the space it reads, so the body has
        // to produce exactly that space's shape — nothing to infer here.
        if let Statement::Iterate { src, target } = stmt {
            let body = get_expr_shape(src, &space_shapes);
            let held = space_shapes.get(target).cloned();
            if let (Some(body), Some(held)) = (body, held) {
                if body != held {
                    return Err(HarmonyDisruption::DimensionErr {
                        space_a: format!("{} ⇒", crate::symbolic::ExprGlyphs(src)),
                        shape_a: body,
                        space_b: target.clone(),
                        shape_b: held,
                        line: block.line_of(index),
                    });
                }
            }
            continue;
        }
        if let Statement::Flow { src, target } = stmt {
            let src_shape = get_expr_shape(src, &space_shapes);
            match target {
                FlowTarget::Var(var_name) => {
                    if let Some(target_shape) = space_shapes.get(var_name).cloned() {
                        if let Some(ref s_shape) = src_shape {
                            if *s_shape != target_shape {
                                return Err(HarmonyDisruption::DimensionErr {
                                    space_a: "INPUT".to_string(), // Keep name compatible with error tests
                                    shape_a: s_shape.clone(),
                                    space_b: var_name.clone(),
                                    shape_b: target_shape,
                                    line: block.line_of(index),
                                });
                            }
                        }
                    } else if let Some(ref s_shape) = src_shape {
                        space_shapes.insert(var_name.clone(), s_shape.clone());
                    }
                }
                FlowTarget::Equilibrium => {
                    // Output verification if needed
                }
            }
        }
    }

    Ok(())
}

