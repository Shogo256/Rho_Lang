use thiserror::Error;

/// Computational disruption error states in ρ (RHO) Language
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum HarmonyDisruption {
    /// Glyph Disruption: Unregistered symbol/character detected
    #[error("[Harmony Disruption: Glyph Failure] Invalid character/symbol detected outside dictionary: '{symbol}' (Line {line}, Column {column})")]
    GlyphErr {
        symbol: String,
        line: usize,
        column: usize,
    },

    /// Space Disruption: Operation on undeclared space
    #[error("[Harmony Disruption: Space Failure] Operation on undeclared space '{space_name}'. Verify '◯ □' initialization. (Line {line})")]
    SpaceErr {
        space_name: String,
        line: usize,
    },

    /// Dimension Disruption: Shape mismatch between spaces
    #[error("[Harmony Disruption: Dimension Failure] Shape mismatch between '{space_a}' ({shape_a:?}) and '{space_b}' ({shape_b:?}). (Line {line})")]
    DimensionErr {
        space_a: String,
        shape_a: Vec<usize>,
        space_b: String,
        shape_b: Vec<usize>,
        line: usize,
    },

    /// Flow Disruption: Missing equilibrium output point (=)
    #[error("[Harmony Disruption: Flow Failure] Missing final equilibrium output point (=) in computational universe.")]
    FlowErr,

    /// Logic Disruption: Static constraint solver (!) failure
    #[error("[Harmony Disruption: Logic Failure] Static constraint expression '{expr}' failed validation. (Line {line})")]
    LogicErr {
        expr: String,
        line: usize,
    },

    /// Iteration Disruption: a `⇒` whose starting point or target is not sound
    #[error("[Harmony Disruption: Iteration Failure] {detail} (Line {line})")]
    IterateErr {
        detail: String,
        line: usize,
    },

    /// A disruption inside a function's body, reached through a call
    #[error("{inner}\n  in `{function}`, whose body is at line {defined}; expanded at line {line}")]
    InFunction {
        function: String,
        defined: usize,
        line: usize,
        inner: Box<HarmonyDisruption>,
    },

    /// Lowering Disruption: construct cannot be mapped to hardware
    #[error("[Harmony Disruption: Lowering Failure] {detail} (Line {line})")]
    LoweringErr {
        detail: String,
        line: usize,
    },
}

impl HarmonyDisruption {
    /// The source line this diagnostic points at, when it has one.
    pub fn line(&self) -> Option<usize> {
        let line = match self {
            HarmonyDisruption::GlyphErr { line, .. } => *line,
            HarmonyDisruption::SpaceErr { line, .. } => *line,
            HarmonyDisruption::DimensionErr { line, .. } => *line,
            HarmonyDisruption::LogicErr { line, .. } => *line,
            HarmonyDisruption::LoweringErr { line, .. } => *line,
            HarmonyDisruption::IterateErr { line, .. } => *line,
            HarmonyDisruption::InFunction { line, .. } => *line,
            HarmonyDisruption::FlowErr => 0,
        };
        (line > 0).then_some(line)
    }

    /// The diagnostic with the offending source line under a caret. An error
    /// inside a function shows the body line it arose on and the call that
    /// expanded it, both with carets.
    pub fn render(&self, source: &str) -> String {
        if let HarmonyDisruption::InFunction {
            function,
            defined,
            line,
            inner,
        } = self
        {
            let show = |n: usize| {
                let text = source.lines().nth(n.saturating_sub(1)).unwrap_or("");
                let column = text.len() - text.trim_start().len() + 1;
                let gutter = format!("{n}");
                let pad = " ".repeat(gutter.len());
                format!(
                    "{pad} |\n{gutter} | {text}\n{pad} | {}^",
                    " ".repeat(column.saturating_sub(1))
                )
            };
            return format!(
                "{inner}\n{}\n  expanded from the call to `{function}` at line {line}:\n{}",
                show(*defined),
                show(*line)
            );
        }
        let Some(line) = self.line() else {
            return self.to_string();
        };
        let Some(text) = source.lines().nth(line - 1) else {
            return self.to_string();
        };
        let column = match self {
            HarmonyDisruption::GlyphErr { column, .. } => *column,
            _ => text.len() - text.trim_start().len() + 1,
        };
        let gutter = format!("{line}");
        let pad = " ".repeat(gutter.len());
        format!(
            "{self}\n{pad} |\n{gutter} | {text}\n{pad} | {}^",
            " ".repeat(column.saturating_sub(1))
        )
    }
}

pub type Result<T> = std::result::Result<T, HarmonyDisruption>;
