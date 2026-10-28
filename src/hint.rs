// SPDX-License-Identifier: Apache-2.0
//! Hints: when a program is refused, how it is written in ρ.
//!
//! A refusal points at a line. Each rule here looks at that line for a habit
//! from another notation — a fold written after its operand, APL's `+/`,
//! NumPy's `np.sum(X)`, an index `X[i+1]`, an assignment `S = expr`, a
//! parenthesis left open, a misspelt name, a space one axis short — and
//! proposes the line as ρ writes it. A proposal is shown only if the program
//! with it applied gets past that line: every hint printed has been parsed
//! and shape-checked before anyone reads it. When two proposals both pass,
//! both are shown; the compiler does not guess which one was meant.
//!
//! The text of every hint is in one table (`MESSAGES`), in English and in
//! Japanese.

use crate::error::HarmonyDisruption;
use crate::parser::parse_rho_program;

/// The language hints are written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Ja,
}

impl Lang {
    /// `ja` or `en`, from a flag's value or `RHO_LANG`.
    pub fn parse(s: &str) -> Option<Lang> {
        match s.trim().to_ascii_lowercase().as_str() {
            "en" | "english" => Some(Lang::En),
            "ja" | "jp" | "japanese" | "日本語" => Some(Lang::Ja),
            _ => None,
        }
    }

    /// The flag if given, else `RHO_LANG`, else English.
    pub fn resolve(flag: Option<&str>) -> Lang {
        flag.and_then(Lang::parse)
            .or_else(|| std::env::var("RHO_LANG").ok().as_deref().and_then(Lang::parse))
            .unwrap_or(Lang::En)
    }
}

/// A change to the source: a line replaced, or a line put before one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    Replace { line: usize, text: String },
    InsertBefore { line: usize, text: String },
    InsertAfter { line: usize, text: String },
}

impl Edit {
    fn apply(&self, source: &str) -> String {
        let mut lines: Vec<String> = source.lines().map(String::from).collect();
        match self {
            Edit::Replace { line, text } => {
                if let Some(l) = lines.get_mut(line - 1) {
                    *l = text.clone();
                }
            }
            Edit::InsertBefore { line, text } => lines.insert((line - 1).min(lines.len()), text.clone()),
            Edit::InsertAfter { line, text } => lines.insert((*line).min(lines.len()), text.clone()),
        }
        let mut out = lines.join("\n");
        if source.ends_with('\n') {
            out.push('\n');
        }
        out
    }

    /// The last line of the edited source that this edit accounts for.
    fn reach(&self) -> usize {
        match self {
            Edit::Replace { line, .. } => *line,
            Edit::InsertBefore { line, .. } | Edit::InsertAfter { line, .. } => line + 1,
        }
    }

    pub fn text(&self) -> &str {
        match self {
            Edit::Replace { text, .. } | Edit::InsertBefore { text, .. } | Edit::InsertAfter { text, .. } => text,
        }
    }
}

/// One hint: what to say, and the change that says it, when there is one.
#[derive(Debug, Clone)]
pub struct Hint {
    /// Message keys with their arguments, in the order they apply.
    pub notes: Vec<(&'static str, Vec<String>)>,
    /// The checked change as it is shown; None for a hint that only explains.
    pub edit: Option<Edit>,
    /// The whole source with the change applied, as it was checked.
    pub fixed: Option<String>,
}

impl Hint {
    /// The source with the hint's change applied.
    pub fn apply(&self, _source: &str) -> Option<String> {
        self.fixed.clone()
    }
}

/// Every message a hint can carry: key, English, Japanese. `{0}`, `{1}`, …
/// are the hint's arguments.
const MESSAGES: &[(&str, &str, &str)] = &[
    ("postfix_fold", "a fold or a scan comes before what it walks: `{0} X`, not `X {0}`", "畳み込み・走査は対象の前に置きます。`X {0}` ではなく `{0} X`"),
    ("apl_reduce", "APL's `{0}/` is `◇{1}` in ρ", "APL の `{0}/` は ρ では `◇{1}` です"),
    ("apl_scan", "APL's `{0}\\` is `◈{1}` in ρ", "APL の `{0}\\` は ρ では `◈{1}` です"),
    ("outer", "an outer product lifts each side: `□1` on the left, `□0` on the right", "外積は両辺を持ち上げて書きます。左に `□1`、右に `□0`"),
    ("index_shift", "ρ has no indices: the next cell is `▽{0}`, the previous one `▷{0}`", "ρ では添字を書きません。次のセルは `▽{0}`、前のセルは `▷{0}` です"),
    ("call_fold", "`{0}(…)` is the fold `◇{1}` in ρ", "`{0}(…)` は ρ では畳み込み `◇{1}` です"),
    ("call_scan", "`{0}(…)` is the scan `◈{1}` in ρ", "`{0}(…)` は ρ では走査 `◈{1}` です"),
    ("call_builtin", "`{0}(…)` is written `({1} …)` in ρ", "`{0}(…)` は ρ では `({1} …)` と書きます"),
    ("call_binary", "`{0}(A, B)` is `A {1} B` in ρ", "`{0}(A, B)` は ρ では `A {1} B` です"),
    ("transpose", "a transpose is `⍉X` in ρ", "転置は ρ では `⍉X` です"),
    ("assign", "a flow names its target on the right: `expr → {0}`", "流れ(→)の行き先は右に書きます: `式 → {0}`"),
    ("assign_output", "the output is the flow into `=`: `expr → =`", "出力は `=` への流れです: `式 → =`"),
    ("power", "a power is `^` in ρ", "べき乗は ρ では `^` です"),
    ("residue", "`A mod B` is the residue `B | A` in ρ (APL's order)", "`A mod B` は ρ では剰余 `B | A` です(APL の順)"),
    ("sigma", "a sum along an axis is the fold `◇+`", "軸に沿った和は畳み込み `◇+` です"),
    ("paren_open", "a parenthesis is left open", "閉じ括弧が足りません"),
    ("paren_close", "a closing parenthesis has nothing to close", "対応する開き括弧の無い閉じ括弧があります"),
    ("nearest", "`{0}` is not declared or defined; `{1}` is", "`{0}` は宣言も定義もされていません。`{1}` のことですか"),
    ("lift", "`{0}` has shape {1} against {2}: lift it with `□{3}` and it stretches along that axis", "`{0}` の形は {1}、相手は {2} です。`□{3}` で持ち上げるとその軸に引き伸ばされます"),
    ("iterate_init", "`{0}` needs a starting value before `⇒`: flow one into it first", "`⇒` の前に `{0}` の初期値が要ります。先に流し込みます"),
    ("output", "no flow ends in `=`: say which space is the output", "`=` で終わる流れがありません。出力にする空間を書きます"),
    ("decl", "a space is declared `NAME:◯ □ dims`", "空間の宣言は `名前:◯ □ 寸法` の順です"),
    ("shift_computed", "a shift reads a declared space: flow the expression into one first", "シフトは宣言された空間から読みます。式をいったん空間に流してから"),
    ("no_loops", "ρ has no `{0}`: write what each cell becomes; repeat to a fixed point with `⇒`, choose per cell with a comparison", "ρ に `{0}` はありません。各セルが何になるかを書き、繰り返しは `⇒`、セルごとの選択は比較で書きます"),
];

fn say(key: &str, args: &[String], lang: Lang) -> String {
    let (_, en, ja) = MESSAGES
        .iter()
        .find(|(k, _, _)| *k == key)
        .copied()
        .unwrap_or((key, key, key));
    let mut text = match lang {
        Lang::En => en,
        Lang::Ja => ja,
    }
    .to_string();
    for (i, a) in args.iter().enumerate() {
        text = text.replace(&format!("{{{i}}}"), a);
    }
    text
}

/// The hints as they are printed under a refusal.
pub fn render(hints: &[Hint], lang: Lang) -> String {
    let (hint, try_) = match lang {
        Lang::En => ("hint", "try"),
        Lang::Ja => ("ヒント", "こう書く"),
    };
    let mut out = String::new();
    for h in hints {
        for (key, args) in &h.notes {
            out.push_str(&format!("  = {hint}: {}\n", say(key, args, lang)));
        }
        if let Some(edit) = &h.edit {
            let where_ = match edit {
                Edit::Replace { line, .. } => format!("{line}"),
                Edit::InsertBefore { line, .. } => format!("+{line}"),
                Edit::InsertAfter { line, .. } => format!("+{}", line + 1),
            };
            out.push_str(&format!("  = {try_} ({where_}): {}\n", edit.text().trim()));
        }
    }
    out
}

// ---------------------------------------------------------------- reading

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '·'
}

/// The operand that ends just before byte `end`: a parenthesised group or a
/// name with any shift or coordinate glyphs before it. Returns its start.
fn operand_before(s: &str, end: usize) -> Option<usize> {
    let head = &s[..end];
    let trimmed = head.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let mut i = trimmed.len();
    if trimmed.ends_with(')') {
        let mut depth = 0i32;
        for (k, c) in trimmed.char_indices().rev() {
            match c {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(k);
                    }
                }
                _ => {}
            }
        }
        return None;
    }
    for (k, c) in trimmed.char_indices().rev() {
        if is_ident_char(c) || (c == '.' && trimmed[..k].chars().next_back().is_some_and(|d| d.is_ascii_digit())) {
            i = k;
        } else {
            break;
        }
    }
    if i == trimmed.len() {
        return None;
    }
    // glyphs that bind tight to a name: ▷ ▽ ⍳ ⌽ ⍉ with an optional digit
    let mut start = i;
    let before: Vec<(usize, char)> = trimmed[..i].char_indices().collect();
    let mut j = before.len();
    while j > 0 {
        let (k, c) = before[j - 1];
        if c.is_ascii_digit() && j >= 2 && "▷▽⍳⌽⍉".contains(before[j - 2].1) {
            start = before[j - 2].0;
            j -= 2;
        } else if "▷▽⍳⌽⍉".contains(c) {
            start = k;
            j -= 1;
        } else {
            break;
        }
    }
    Some(start)
}

/// The operand that starts at or after byte `from`: a parenthesised group or
/// a name. Returns (start, end).
fn operand_after(s: &str, from: usize) -> Option<(usize, usize)> {
    let rest = &s[from..];
    let skip = rest.len() - rest.trim_start().len();
    let start = from + skip;
    let body = &s[start..];
    if body.starts_with('(') {
        let mut depth = 0i32;
        for (k, c) in body.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((start, start + k + 1));
                    }
                }
                _ => {}
            }
        }
        return None;
    }
    // a number: digits, a point, an exponent, a sign in front
    let number: usize = {
        let b = body.as_bytes();
        let mut k = 0;
        if k < b.len() && b[k] == b'-' {
            k += 1;
        }
        let digits_from = k;
        while k < b.len() && (b[k].is_ascii_digit() || b[k] == b'.') {
            k += 1;
        }
        if k > digits_from && body[digits_from..k].chars().next().is_some_and(|c| c.is_ascii_digit()) {
            k
        } else {
            0
        }
    };
    if number > 0 {
        return Some((start, start + number));
    }
    let mut end = start;
    for (k, c) in body.char_indices() {
        if is_ident_char(c) || ("▷▽⍳⌽⍉".contains(c) && k == 0) || (c.is_ascii_digit() && end == start + k) {
            end = start + k + c.len_utf8();
        } else {
            break;
        }
    }
    (end > start).then_some((start, end))
}

fn paren(s: &str) -> String {
    let t = s.trim();
    if t.chars().all(is_ident_char) || (t.starts_with('(') && t.ends_with(')') && balanced_whole(t)) {
        t.to_string()
    } else {
        format!("({t})")
    }
}

fn balanced_whole(t: &str) -> bool {
    let mut depth = 0i32;
    let n = t.chars().count();
    for (i, c) in t.chars().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 && i + 1 < n {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// Every identifier in the source that names a space or a function, with
/// the built-in names: what a misspelt name could have meant.
fn known_names(source: &str) -> Vec<String> {
    let mut names: Vec<String> = ["exp", "log", "sqrt", "sin", "cos", "abs", "ind"].iter().map(|s| s.to_string()).collect();
    for line in source.lines() {
        let t = line.trim();
        // `NAME:◯ □ …`, `&[0x…]:NAME:◯ □ …`, `name:{ …`
        for marker in [":◯", ":□", ":{"] {
            if let Some(at) = t.find(marker) {
                let head = t[..at].rsplit(':').next().unwrap_or("").trim();
                if !head.is_empty() && head.chars().all(is_ident_char) {
                    names.push(head.to_string());
                }
            }
        }
        for arrow in ["→", "⇒", "->", "=>"] {
            if let Some(p) = t.rfind(arrow) {
                let target = t[p + arrow.len()..].trim();
                if !target.is_empty() && target.chars().all(is_ident_char) {
                    names.push(target.to_string());
                }
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// A transposition of two neighbouring letters counts as one edit.
fn close_enough(typed: &str, known: &str) -> bool {
    if typed == known {
        return false;
    }
    let a: Vec<char> = typed.chars().collect();
    let b: Vec<char> = known.chars().collect();
    if a.len() == b.len() {
        let diff: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
        if diff.len() == 2 && diff[1] == diff[0] + 1 && a[diff[0]] == b[diff[1]] && a[diff[1]] == b[diff[0]] {
            return true;
        }
    }
    if typed.to_lowercase() == known.to_lowercase() {
        return true;
    }
    if a.len() < 3 || b.len() < 3 {
        return false;
    }
    let d = edit_distance(typed, known);
    d <= 1 || (d == 2 && b.len() >= 5)
}

/// Replace every whole-word occurrence of `from` in `line` with `to`.
fn replace_word(line: &str, from: &str, to: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    while let Some(p) = line[i..].find(from) {
        let at = i + p;
        let before_ok = line[..at].chars().next_back().is_none_or(|c| !is_ident_char(c));
        let after_ok = line[at + from.len()..].chars().next().is_none_or(|c| !is_ident_char(c));
        out.push_str(&line[i..at]);
        if before_ok && after_ok {
            out.push_str(to);
        } else {
            out.push_str(from);
        }
        i = at + from.len();
    }
    out.push_str(&line[i..]);
    out
}

// ---------------------------------------------------------------- rules

type Rewrite = (&'static str, Vec<String>, String);

/// Rewrites of one line that need nothing but the line.
fn line_rules(line: &str) -> Vec<Rewrite> {
    let mut out: Vec<Rewrite> = Vec::new();

    // A fold or a scan after its operand: `X ◇+1 → …`, `(X × X) ◇+ → …`.
    for glyph in ['◇', '◈'] {
        for (at, _) in line.match_indices(glyph) {
            let after = &line[at + glyph.len_utf8()..];
            let mut op_end = at + glyph.len_utf8();
            let mut chars = after.chars();
            let Some(op) = chars.next() else { continue };
            if !"+×*><".contains(op) {
                continue;
            }
            op_end += op.len_utf8();
            let digits: String = line[op_end..].chars().take_while(|c| c.is_ascii_digit()).collect();
            op_end += digits.len();
            let next = line[op_end..].trim_start();
            // prefix when an operand follows; postfix when an operator,
            // a closing parenthesis, an arrow or the end of the line does
            let operand_follows = next.chars().next().is_some_and(|c| {
                c == '(' || is_ident_char(c) || "▷▽⍳⌽⍉◇◈□?⌊⌈𝜏τ-".contains(c)
            });
            let postfix = !operand_follows;
            if !postfix {
                continue;
            }
            let Some(start) = operand_before(line, at) else { continue };
            let operand = line[start..at].trim();
            if operand.is_empty() {
                continue;
            }
            let fold = format!("{glyph}{op}{digits}");
            let fixed = format!("{}({fold} {operand}){}", &line[..start], &line[op_end..]);
            out.push(("postfix_fold", vec![fold], fixed));
        }
    }

    // APL's reductions and scans: `+/X`, `⌈\X`.
    for (apl, rho) in [("+", "+"), ("×", "×"), ("*", "×"), ("⌈", ">"), ("⌊", "<")] {
        for (slash, key, glyph) in [("/", "apl_reduce", '◇'), ("\\", "apl_scan", '◈')] {
            let pat = format!("{apl}{slash}");
            if let Some(at) = line.find(&pat) {
                let after = at + pat.len();
                if let Some((s, e)) = operand_after(line, after) {
                    let fixed = format!("{}({glyph}{rho} {}){}", &line[..at], &line[s..e], &line[e..]);
                    out.push((key, vec![apl.to_string(), rho.to_string()], fixed));
                }
            }
        }
    }

    // APL's outer product: `A ∘.× B`.
    if let Some(at) = line.find("∘.") {
        let op_at = at + "∘.".len();
        if let Some(op) = line[op_at..].chars().next() {
            let op_end = op_at + op.len_utf8();
            if let (Some(ls), Some((rs, re))) = (operand_before(line, at), operand_after(line, op_end)) {
                let left = line[ls..at].trim();
                let right = &line[rs..re];
                let fixed = format!("{}((□1 {left}) {op} (□0 {right})){}", &line[..ls], &line[re..]);
                out.push(("outer", vec![], fixed));
            }
        }
    }

    // Indices: `X[i+1]`, `X[i-1]`, `X[i]`.
    if line.contains('[') && !line.contains("&[") {
        let mut fixed = line.to_string();
        let mut names: Vec<String> = Vec::new();
        while let Some(open) = fixed.find('[') {
            let Some(close_rel) = fixed[open..].find(']') else { break };
            let close = open + close_rel;
            let Some(name_start) = operand_before(&fixed, open) else { break };
            let name = fixed[name_start..open].trim().to_string();
            if name.is_empty() || !name.chars().all(is_ident_char) {
                break;
            }
            let inside: String = fixed[open + 1..close].chars().filter(|c| !c.is_whitespace()).collect();
            let glyph = if inside.ends_with("+1") {
                "▽"
            } else if inside.ends_with("-1") {
                "▷"
            } else if inside.chars().all(is_ident_char) {
                ""
            } else {
                break;
            };
            fixed = format!("{}{glyph}{name}{}", &fixed[..name_start], &fixed[close + 1..]);
            names.push(name);
        }
        if fixed != line && !fixed.contains('[') {
            out.push(("index_shift", vec![names.first().cloned().unwrap_or_default()], fixed));
        }
    }

    // Calls in the style of NumPy or Python: `np.sum(X)`, `exp(X)`,
    // `max(A, B)`, `X.T`.
    let calls: &[(&str, &str, &str)] = &[
        ("sum", "call_fold", "+"),
        ("prod", "call_fold", "×"),
        ("product", "call_fold", "×"),
        ("max", "call_fold", ">"),
        ("amax", "call_fold", ">"),
        ("min", "call_fold", "<"),
        ("amin", "call_fold", "<"),
        ("cumsum", "call_scan", "+"),
        ("cumprod", "call_scan", "×"),
        ("exp", "call_builtin", "exp"),
        ("log", "call_builtin", "log"),
        ("sqrt", "call_builtin", "sqrt"),
        ("sin", "call_builtin", "sin"),
        ("cos", "call_builtin", "cos"),
        ("abs", "call_builtin", "abs"),
        ("fabs", "call_builtin", "abs"),
        ("maximum", "call_binary", "⌈"),
        ("minimum", "call_binary", "⌊"),
        ("transpose", "transpose", "⍉"),
    ];
    for (name, key, rho) in calls {
        for prefix in ["np.", "numpy.", "jnp.", "math.", ""] {
            let call = format!("{prefix}{name}(");
            let Some(at) = line.find(&call) else { continue };
            if prefix.is_empty() && line[..at].chars().next_back().is_some_and(|c| is_ident_char(c) || c == '.') {
                continue;
            }
            let open = at + call.len() - 1;
            let Some((s, e)) = operand_after(line, open) else { continue };
            let inner = &line[s + 1..e - 1];
            let parts: Vec<&str> = split_top(inner);
            let shown = format!("{prefix}{name}");
            let fixed = match (*key, parts.len()) {
                ("call_fold", 1) => format!("{}(◇{rho} {}){}", &line[..at], paren(parts[0]), &line[e..]),
                ("call_fold", 2) if *name == "max" || *name == "min" => {
                    let g = if *name == "max" { "⌈" } else { "⌊" };
                    out.push(("call_binary", vec![shown.clone(), g.to_string()], format!("{}({} {g} {}){}", &line[..at], parts[0].trim(), parts[1].trim(), &line[e..])));
                    continue;
                }
                ("call_scan", 1) => format!("{}(◈{rho} {}){}", &line[..at], paren(parts[0]), &line[e..]),
                ("call_builtin", 1) => format!("{}({rho} {}){}", &line[..at], paren(parts[0]), &line[e..]),
                ("call_binary", 2) => format!("{}({} {rho} {}){}", &line[..at], parts[0].trim(), parts[1].trim(), &line[e..]),
                ("transpose", 1) => format!("{}(⍉{}){}", &line[..at], paren(parts[0]), &line[e..]),
                _ => continue,
            };
            out.push((key, vec![shown, rho.to_string()], fixed));
        }
    }
    if let Some(at) = line.find(".T") {
        let after_ok = line[at + 2..].chars().next().is_none_or(|c| !is_ident_char(c));
        if after_ok {
            if let Some(start) = operand_before(line, at) {
                let fixed = format!("{}(⍉{}){}", &line[..start], line[start..at].trim(), &line[at + 2..]);
                out.push(("transpose", vec![], fixed));
            }
        }
    }
    if let Some(at) = line.find("reduce(") {
        let open = at + "reduce".len();
        if let Some((s, e)) = operand_after(line, open) {
            let parts = split_top(&line[s + 1..e - 1]);
            if parts.len() == 2 {
                let op = parts[0].trim();
                let rho = match op {
                    "+" | "add" | "np.add" | "operator.add" => Some("+"),
                    "*" | "×" | "mul" | "np.multiply" | "operator.mul" => Some("×"),
                    "max" => Some(">"),
                    "min" => Some("<"),
                    _ => None,
                };
                if let Some(rho) = rho {
                    let fixed = format!("{}(◇{rho} {}){}", &line[..at], paren(parts[1]), &line[e..]);
                    out.push(("call_fold", vec!["reduce".to_string(), rho.to_string()], fixed));
                }
            }
        }
    }

    // Σ X: the sum along the default axis.
    for sigma in ["Σ", "∑"] {
        if let Some(at) = line.find(sigma) {
            if let Some((s, e)) = operand_after(line, at + sigma.len()) {
                let fixed = format!("{}(◇+ {}){}", &line[..at], &line[s..e], &line[e..]);
                out.push(("sigma", vec![], fixed));
            }
        }
    }

    // An assignment: `S = expr`, `S ← expr`, `S := expr`, `= ← expr`.
    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
    let body = line.trim();
    for sign in ["←", ":=", "="] {
        let Some(at) = body.find(sign) else { continue };
        if sign == "=" && (body[at + 1..].starts_with('=') || body[..at].ends_with(['=', '!', '<', '>', ':', '-', '→'])) {
            continue;
        }
        let target = body[..at].trim();
        let expr = body[at + sign.len()..].trim();
        if expr.is_empty() || expr.contains('→') || expr.contains('⇒') {
            continue;
        }
        if target == "OUTPUT" {
            // the output itself, or a space named OUTPUT read later: both
            out.push(("assign_output", vec![], format!("{indent}{expr} → =")));
            out.push(("assign", vec![target.to_string()], format!("{indent}{expr} → {target}")));
        } else if target == "=" || target.is_empty() && sign == "←" {
            out.push(("assign_output", vec![], format!("{indent}{expr} → =")));
        } else if !target.is_empty() && target.chars().all(is_ident_char) {
            out.push(("assign", vec![target.to_string()], format!("{indent}{expr} → {target}")));
        }
        break;
    }

    // `**` for a power; `mod` for the residue.
    if line.contains("**") {
        out.push(("power", vec![], line.replace("**", "^")));
    }
    if let Some(at) = line.find(" mod ") {
        if let (Some(ls), Some((rs, re))) = (operand_before(line, at), operand_after(line, at + " mod ".len())) {
            let left = line[ls..at].trim();
            let fixed = format!("{}({} | {left}){}", &line[..ls], &line[rs..re], &line[re..]);
            out.push(("residue", vec![], fixed));
        }
    }

    // A declaration with its glyphs out of order or one missing.
    if let Some(colon) = body.rfind(':') {
        let head = &body[..colon];
        let rest = body[colon + 1..].trim();
        let has_circle = rest.contains('◯');
        let has_square = rest.contains('□');
        let wrong = (has_circle || has_square) && !rest.starts_with("◯ □") && !rest.starts_with('{');
        if wrong && !head.contains('{') && !head.ends_with('}') {
            let dims: String = rest.replace(['◯', '□'], " ").split_whitespace().collect::<Vec<_>>().join(" ");
            if !dims.is_empty() && dims.split(' ').all(|d| d.chars().all(|c| c.is_ascii_digit())) {
                out.push(("decl", vec![], format!("{indent}{head}:◯ □ {dims}")));
            }
        }
    }

    // A parenthesis left open, or one too many closed, before the arrow.
    let arrow = ["→", "⇒", "->", "=>"].iter().filter_map(|a| line.find(a)).min();
    let (expr_end, tail) = match arrow {
        Some(a) => (a, &line[a..]),
        None => (line.len(), ""),
    };
    let expr = &line[..expr_end];
    let opens = expr.matches('(').count();
    let closes = expr.matches(')').count();
    if opens > closes {
        let fixed = format!("{}{}{}{}", expr.trim_end(), ")".repeat(opens - closes), if tail.is_empty() { "" } else { " " }, tail);
        out.push(("paren_open", vec![], fixed));
    } else if closes > opens {
        // drop the last unmatched ')'
        let mut depth = 0i32;
        let mut drop_at = None;
        for (k, c) in expr.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    if depth == 0 {
                        drop_at = Some(k);
                    } else {
                        depth -= 1;
                    }
                }
                _ => {}
            }
        }
        if let Some(k) = drop_at {
            let fixed = format!("{}{}{}", &expr[..k], &expr[k + 1..], tail);
            out.push(("paren_close", vec![], fixed));
        }
    }

    out
}

/// A line without a doubled pair of parentheses around one group:
/// `((A × B))` reads as `(A × B)`.
fn tidy(line: &str) -> String {
    let mut s = line.to_string();
    loop {
        let chars: Vec<(usize, char)> = s.char_indices().collect();
        let mut changed = false;
        for w in 0..chars.len().saturating_sub(1) {
            if chars[w].1 == '(' && chars[w + 1].1 == '(' {
                // the close of each
                let close_of = |from: usize| -> Option<usize> {
                    let mut depth = 0i32;
                    for (idx, &(_, c)) in chars.iter().enumerate().skip(from) {
                        match c {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    return Some(idx);
                                }
                            }
                            _ => {}
                        }
                    }
                    None
                };
                if let (Some(outer), Some(inner)) = (close_of(w), close_of(w + 1)) {
                    if outer == inner + 1 {
                        let (o, i) = (chars[w].0, chars[outer].0);
                        s = format!("{}{}{}", &s[..o], &s[o + 1..i], &s[i + 1..]);
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed {
            return s;
        }
    }
}

/// Split at commas not inside parentheses.
fn split_top(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (k, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..k]);
                start = k + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts.into_iter().filter(|p| !p.trim().is_empty()).collect()
}

/// Rewrites that need the refusal: a misspelt name, a space one axis short,
/// a `⇒` with nothing in its target, no output, a shift of an expression.
fn error_rules(source: &str, err: &HarmonyDisruption, line_no: usize) -> Vec<(&'static str, Vec<String>, Edit)> {
    let mut out = Vec::new();
    let Some(line) = source.lines().nth(line_no - 1) else { return out };
    let names = known_names(source);

    let innermost = match err {
        HarmonyDisruption::InFunction { inner, .. } => inner.as_ref(),
        other => other,
    };
    let unknown: Option<String> = match innermost {
        HarmonyDisruption::SpaceErr { space_name, .. } => Some(space_name.clone()),
        HarmonyDisruption::LoweringErr { detail, .. } if detail.contains("is not a function defined") => {
            detail.split('`').nth(1).map(String::from)
        }
        _ => None,
    };
    if let Some(name) = &unknown {
        if name.chars().all(is_ident_char) {
            for known in names.iter().filter(|k| close_enough(name, k)) {
                for (k, l) in source.lines().enumerate() {
                    let replaced = replace_word(l, name, known);
                    if replaced != l {
                        out.push((
                            "nearest",
                            vec![name.clone(), known.clone()],
                            Edit::Replace { line: k + 1, text: replaced },
                        ));
                    }
                }
            }
            // a `⇒` into a space nothing has flowed into yet
            let iterates_into = line.split('⇒').nth(1).map(|t| t.trim().to_string());
            if iterates_into.as_deref() == Some(name.as_str()) {
                for start in names.iter().filter(|k| *k != name && source.contains(&format!("{k}:"))) {
                    out.push((
                        "iterate_init",
                        vec![name.clone()],
                        Edit::InsertBefore {
                            line: line_no,
                            text: format!("{}{start} → {name}", line.chars().take_while(|c| c.is_whitespace()).collect::<String>()),
                        },
                    ));
                }
            }
        }
    }

    if let HarmonyDisruption::DimensionErr { space_a, shape_a, space_b, shape_b, .. } = err {
        for (name, short, long) in [(space_a, shape_a, shape_b), (space_b, shape_b, shape_a)] {
            if !name.chars().all(is_ident_char) || short.len() + 1 != long.len() {
                continue;
            }
            for axis in 0..=short.len() {
                let mut lifted = short.clone();
                lifted.insert(axis, 1);
                let fits = lifted.iter().zip(long.iter()).all(|(a, b)| a == b || *a == 1);
                if fits {
                    out.push((
                        "lift",
                        vec![name.clone(), format!("{short:?}"), format!("{long:?}"), axis.to_string()],
                        Edit::Replace { line: line_no, text: replace_word(line, name, &format!("(□{axis} {name})")) },
                    ));
                }
            }
        }
    }

    if let HarmonyDisruption::FlowErr { .. } = err {
        // the last flow's target becomes the output
        let all: Vec<&str> = source.lines().collect();
        for (k, l) in all.iter().enumerate().rev() {
            if let Some(p) = l.rfind('→') {
                let target = l[p + '→'.len_utf8()..].trim();
                if !target.is_empty() && target.chars().all(is_ident_char) {
                    let indent: String = l.chars().take_while(|c| c.is_whitespace()).collect();
                    out.push(("output", vec![], Edit::InsertAfter { line: k + 1, text: format!("{indent}{target} → =") }));
                    break;
                }
            }
        }
    }

    if let HarmonyDisruption::LoweringErr { detail, .. } = err {
        if detail.contains("cannot be applied to a computed value") {
            let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
            for glyph in ["▷", "▽", "⌽", "↑", "↓"] {
                for (at, _) in line.match_indices(glyph) {
                    let mut from = at + glyph.len();
                    let digits: String = line[from..].chars().take_while(|c| c.is_ascii_digit()).collect();
                    from += digits.len();
                    if let Some((s, e)) = operand_after(line, from) {
                        if line[s..].starts_with('(') {
                            let name = fresh_name(source);
                            let inner = &line[s..e];
                            // two edits in one: the new line, and the use
                            out.push((
                                "shift_computed",
                                vec![],
                                Edit::InsertBefore { line: line_no, text: format!("{indent}{inner} → {name}\u{1}{}", format_args!("{}{name}{}", &line[..s], &line[e..])) },
                            ));
                        }
                    }
                }
            }
        }
    }

    out
}

/// A name no line uses yet, for a space a hint introduces.
fn fresh_name(source: &str) -> String {
    for n in ["T", "U", "V", "W", "T1", "T2", "T3"] {
        if !source.split(|c: char| !is_ident_char(c)).any(|w| w == n) {
            return n.to_string();
        }
    }
    "TMP".to_string()
}

/// Apply an edit whose text carries a second line for the line it was put
/// before (`\u{1}` separates them): the hint that introduces a space.
fn apply_compound(edit: &Edit, source: &str) -> String {
    if let Edit::InsertBefore { line, text } = edit {
        if let Some((first, second)) = text.split_once('\u{1}') {
            let mut lines: Vec<String> = source.lines().map(String::from).collect();
            if let Some(l) = lines.get_mut(line - 1) {
                *l = second.to_string();
            }
            lines.insert(line - 1, first.to_string());
            let mut out = lines.join("\n");
            if source.ends_with('\n') {
                out.push('\n');
            }
            return out;
        }
    }
    edit.apply(source)
}

/// Whether the source with an edit gets past the line the refusal was on
/// (and past the edit's own lines).
fn passes(edited: &str, edit: &Edit, refused: usize) -> bool {
    match parse_rho_program(edited) {
        Ok(_) => true,
        Err(e) => {
            // an inserted line moves the refused one down by one
            let moved = refused + usize::from(!matches!(edit, Edit::Replace { .. }));
            e.line().is_some_and(|l| l > moved.max(edit.reach()))
        }
    }
}

/// Hints for a refused program: each one checked by parsing the program
/// with its change applied. At most a few, the ones that make the whole
/// program parse first.
pub fn suggest(source: &str, err: &HarmonyDisruption) -> Vec<Hint> {
    let line_no = match err {
        HarmonyDisruption::InFunction { inner, .. } => inner.line().or(err.line()),
        _ => err.line(),
    };
    let Some(line_no) = line_no else { return Vec::new() };
    let Some(line) = source.lines().nth(line_no - 1) else { return Vec::new() };

    let mut found: Vec<(bool, Hint, String)> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut consider = |notes: Vec<(&'static str, Vec<String>)>, edit: Edit, found: &mut Vec<(bool, Hint, String)>| {
        let edited = apply_compound(&edit, source);
        if seen.contains(&edited) || edited == source {
            return;
        }
        seen.push(edited.clone());
        if passes(&edited, &edit, line_no) {
            let whole = parse_rho_program(&edited).is_ok();
            let shown = match &edit {
                Edit::InsertBefore { line, text } if text.contains('\u{1}') => {
                    let (a, b) = text.split_once('\u{1}').unwrap();
                    Edit::InsertBefore { line: *line, text: format!("{}\n      {}", a.trim(), b.trim()) }
                }
                other => other.clone(),
            };
            found.push((whole, Hint { notes, edit: Some(shown), fixed: Some(edited.clone()) }, edited));
        }
    };

    // One rewrite of a line, then a second on its result: a line can carry
    // two habits at once (`np.sum(X[i+1])`). The refused line first, then
    // the lines above it: a habit can pass its own line unnoticed and be
    // refused where its result is used (`S = expr`, then `S` undeclared).
    let _ = line;
    let all: Vec<&str> = source.lines().collect();
    let order = std::iter::once(line_no)
        .chain((1..line_no).rev())
        .chain(line_no + 1..=all.len());
    for at in order {
        let Some(text_at) = all.get(at - 1) else { continue };
        for (key, args, text) in line_rules(text_at) {
            let text = tidy(&text);
            let edit = Edit::Replace { line: at, text: text.clone() };
            consider(vec![(key, args.clone())], edit, &mut found);
            for (key2, args2, text2) in line_rules(&text) {
                let edit2 = Edit::Replace { line: at, text: tidy(&text2) };
                consider(vec![(key, args.clone()), (key2, args2)], edit2, &mut found);
            }
        }
    }
    for (key, args, edit) in error_rules(source, err, line_no) {
        consider(vec![(key, args)], edit, &mut found);
    }

    // A keyword from a language with loops: say how ρ writes it instead.
    let mut explain: Vec<Hint> = Vec::new();
    if let HarmonyDisruption::GlyphErr { symbol, .. } = err {
        if ["for", "while", "if", "else", "return", "let", "var", "const", "fn", "function", "class"].contains(&symbol.as_str()) {
            explain.push(Hint { notes: vec![("no_loops", vec![symbol.clone()])], edit: None, fixed: None });
        }
    }

    // Whole fixes first, then the shortest; at most three.
    found.sort_by_key(|(whole, h, _)| (!*whole, h.notes.len()));
    let best_whole = found.iter().any(|(w, _, _)| *w);
    let mut hints: Vec<Hint> = found
        .into_iter()
        .filter(|(w, _, _)| *w || !best_whole)
        .map(|(_, h, _)| h)
        .take(3)
        .collect();
    hints.extend(explain);
    hints
}

/// The habits a program can be written with by someone used to another
/// notation: each a valid program's line rewritten the way that notation
/// writes it. A test applies them to programs that compile and checks that
/// the refusal's hint gives back a program that computes the same thing.
pub fn mutations(source: &str) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let rebuild = |k: usize, text: String| -> String {
        let mut ls: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        ls[k] = text;
        let mut s = ls.join("\n");
        if source.ends_with('\n') {
            s.push('\n');
        }
        s
    };
    for (k, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with("/*") || t.starts_with("//") || t.is_empty() || t == "{" || t == "}" {
            continue;
        }
        // fold before → fold after
        for glyph in ['◇', '◈'] {
            if let Some(at) = line.find(glyph) {
                let mut end = at + glyph.len_utf8();
                if let Some(op) = line[end..].chars().next() {
                    if "+×><".contains(op) {
                        end += op.len_utf8();
                        let digits: String = line[end..].chars().take_while(|c| c.is_ascii_digit()).collect();
                        end += digits.len();
                        if let Some((s, e)) = operand_after(line, end) {
                            let fold = &line[at..end];
                            let operand = &line[s..e];
                            out.push(("postfix_fold", rebuild(k, format!("{}{operand} {fold}{}", &line[..at], &line[e..]))));
                            // and as APL writes it, when the axis is the default
                            if digits.is_empty() && glyph == '◇' {
                                let apl = match op { '+' => "+", '×' => "×", '>' => "⌈", _ => "⌊" };
                                out.push(("apl_reduce", rebuild(k, format!("{}{apl}/{operand}{}", &line[..at], &line[e..]))));
                                let np = match op { '+' => "sum", '×' => "prod", '>' => "max", _ => "min" };
                                let arg = operand.strip_prefix('(').and_then(|o| o.strip_suffix(')')).filter(|o| balanced_whole(&format!("({o})"))).unwrap_or(operand);
                                out.push(("numpy", rebuild(k, format!("{}np.{np}({arg}){}", &line[..at], &line[e..]))));
                            }
                        }
                    }
                }
            }
        }
        // a flow into a name → an assignment
        if let Some(p) = line.rfind('→') {
            let target = line[p + '→'.len_utf8()..].trim();
            let expr = line[..p].trim();
            let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
            if !expr.is_empty() && !expr.contains('⇒') {
                if target == "=" {
                    out.push(("assign_output", rebuild(k, format!("{indent}OUTPUT = {expr}"))));
                } else if target.chars().all(is_ident_char) && !target.is_empty() {
                    out.push(("assign", rebuild(k, format!("{indent}{target} = {expr}"))));
                }
            }
            // the last ')' before the arrow left out
            if let Some(close) = line[..p].rfind(')') {
                if line[..p].trim_end().ends_with(')') {
                    out.push(("paren_open", rebuild(k, format!("{}{}", &line[..close], &line[close + 1..]))));
                }
            }
        }
        // the next and the previous cell as indices
        for (glyph, idx) in [("▽", "[i+1]"), ("▷", "[i-1]")] {
            if let Some(at) = line.find(glyph) {
                let after = at + glyph.len();
                if line[after..].chars().next().is_some_and(|c| c.is_alphabetic()) {
                    if let Some((s, e)) = operand_after(line, after) {
                        let name = &line[s..e];
                        if name.chars().all(is_ident_char) {
                            out.push(("index_shift", rebuild(k, format!("{}{name}{idx}{}", &line[..at], &line[e..]))));
                        }
                    }
                }
            }
        }
        // a built-in called with parentheses
        for f in ["exp", "log", "sqrt", "sin", "cos", "abs"] {
            if let Some(at) = line.find(&format!("{f} ")) {
                if line[..at].chars().next_back().is_none_or(|c| !is_ident_char(c)) {
                    if let Some((s, e)) = operand_after(line, at + f.len() + 1) {
                        let arg = &line[s..e];
                        let arg = arg.strip_prefix('(').and_then(|o| o.strip_suffix(')')).filter(|o| balanced_whole(&format!("({o})"))).unwrap_or(arg);
                        out.push(("numpy", rebuild(k, format!("{}np.{f}({arg}){}", &line[..at], &line[e..]))));
                    }
                }
            }
        }
        // a declaration with its glyphs swapped
        if line.contains("◯ □") {
            out.push(("decl", rebuild(k, line.replacen("◯ □", "□ ◯", 1))));
        }
        // a name misspelt where it is read: two letters swapped
        if let Some(p) = line.find('→') {
            let expr = &line[..p];
            let decl_or_def = expr.contains(':') || expr.contains('{');
            if !decl_or_def {
                let words: Vec<&str> = expr.split(|c: char| !is_ident_char(c)).filter(|w| w.chars().count() >= 4 && w.chars().all(|c| c.is_alphabetic())).collect();
                if let Some(w) = words.first() {
                    let mut cs: Vec<char> = w.chars().collect();
                    let n = cs.len();
                    cs.swap(n - 2, n - 3);
                    let typo: String = cs.into_iter().collect();
                    if typo != *w {
                        out.push(("nearest", rebuild(k, replace_word(line, w, &typo))));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_fix(src: &str) -> Option<String> {
        let err = parse_rho_program(src).err()?;
        let hints = suggest(src, &err);
        hints.iter().find_map(|h| h.apply(src))
    }

    #[test]
    fn a_fold_after_its_operand_is_put_before_it() {
        let src = "{\n    X:◯ □ 4 8\n    X ◇+1 → =\n}\n";
        let err = parse_rho_program(src).unwrap_err();
        let hints = suggest(src, &err);
        assert!(hints.iter().any(|h| h.edit.as_ref().unwrap().text().contains("(◇+1 X) → =")), "{hints:?}");
    }

    #[test]
    fn every_hint_shown_parses() {
        let src = "{\n    X:◯ □ 8\n    np.sum(X[i+1]) → =\n}\n";
        let err = parse_rho_program(src).unwrap_err();
        let hints = suggest(src, &err);
        assert!(!hints.is_empty());
        let fixed = first_fix(src).unwrap();
        assert!(parse_rho_program(&fixed).is_ok(), "{fixed}");
    }
}
