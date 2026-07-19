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
    #[error("[Harmony Disruption: Space Failure] Operation on undeclared space '{space_name}'. Verify '◯ □' initialization.")]
    SpaceErr {
        space_name: String,
    },

    /// Dimension Disruption: Shape mismatch between spaces
    #[error("[Harmony Disruption: Dimension Failure] Shape mismatch between '{space_a}' ({shape_a:?}) and '{space_b}' ({shape_b:?}).")]
    DimensionErr {
        space_a: String,
        shape_a: Vec<usize>,
        space_b: String,
        shape_b: Vec<usize>,
    },

    /// Flow Disruption: Missing equilibrium output point (=)
    #[error("[Harmony Disruption: Flow Failure] Missing final equilibrium output point (=) in computational universe.")]
    FlowErr,

    /// Logic Disruption: Static constraint solver (!) failure
    #[error("[Harmony Disruption: Logic Failure] Static constraint expression '{expr}' failed validation.")]
    LogicErr {
        expr: String,
    },
}

pub type Result<T> = std::result::Result<T, HarmonyDisruption>;
