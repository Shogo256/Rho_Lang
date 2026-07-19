/// Clockless Shift Engine Optimization (▷, ▽)
/// Eliminates loop (CLK) cycles, converting spatial shifts directly to vector/SIMD IR instructions
pub struct VectorShiftEngine;

impl VectorShiftEngine {
    /// ▷ (Shift Right): Shift elements right, padding 0 at index 0
    pub fn emit_shift_right_ir(var_name: &str, size: usize) -> String {
        format!(
            "  ; Vector Shift Right (▷) for {size} elements (Clockless)\n\
               %{var_name}_shifted = call <{size} x double> @llvm.rho.shift_right.<{size} x double>(<{size} x double> %{var_name})\n"
        )
    }

    /// ▽ (Shift Left): Shift elements left, padding 0 at final index
    pub fn emit_shift_left_ir(var_name: &str, size: usize) -> String {
        format!(
            "  ; Vector Shift Left (▽) for {size} elements (Clockless)\n\
               %{var_name}_shifted = call <{size} x double> @llvm.rho.shift_left.<{size} x double>(<{size} x double> %{var_name})\n"
        )
    }
}
