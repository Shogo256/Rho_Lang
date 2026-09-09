use rho_lang::codegen::LlvmCodeGen;
use rho_lang::error::HarmonyDisruption;
use rho_lang::parser::parse_rho_program;

#[test]
fn test_parse_teichmuller() {
    let source = r#"{
        &[0x7A4F]:INPUT:◯ □ 1024 1024 
        (▷INPUT - INPUT) → △
        (▽INPUT - INPUT) → ▽
        ((△ - ▽) / (△ + ▽)) ^ 2 → OUTPUT
        ! (OUTPUT >= 0)
        OUTPUT > 𝜏 → =
    }"#;

    let res = parse_rho_program(source);
    assert!(res.is_ok(), "Teichmuller script should parse successfully");

    let block = res.unwrap();
    assert_eq!(block.statements.len(), 6);
}

#[test]
fn test_glyph_disruption_err() {
    let source = r#"{
        for i in 0..10 { }
        OUTPUT → =
    }"#;

    let res = parse_rho_program(source);
    assert!(res.is_err());
    match res.err().unwrap() {
        HarmonyDisruption::GlyphErr { symbol, .. } => {
            assert_eq!(symbol, "for");
        }
        err => panic!("Unexpected error: {:?}", err),
    }
}

#[test]
fn test_space_disruption_err() {
    let source = r#"{
        UNDECLARED_SPACE → OUTPUT
        OUTPUT → =
    }"#;

    let res = parse_rho_program(source);
    assert!(res.is_err());
    match res.err().unwrap() {
        HarmonyDisruption::SpaceErr { space_name, line } => {
            assert_eq!(space_name, "UNDECLARED_SPACE");
            assert_eq!(line, 2, "the diagnostic should point at the offending flow");
        }
        err => panic!("Unexpected error: {:?}", err),
    }
}

#[test]
fn test_flow_disruption_err() {
    let source = r#"{
        INPUT:◯ □ 10 10
        INPUT → OUTPUT
    }"#;

    let res = parse_rho_program(source);
    assert!(res.is_err());
    match res.err().unwrap() {
        HarmonyDisruption::FlowErr => {}
        err => panic!("Unexpected error: {:?}", err),
    }
}

#[test]
fn test_parse_ascii_aliases() {
    let source = r#"{
        @[0x7A4F]:INPUT:◯ □ 1024 1024 
        (>>INPUT - INPUT) -> △
        (<<INPUT - INPUT) -> ▽
        ((△ - ▽) / (△ + ▽)) ^ 2 -> OUTPUT
        ! (OUTPUT >= 0)
        OUTPUT > 𝜏 -> =
    }"#;

    let res = parse_rho_program(source);
    assert!(res.is_ok(), "ASCII alias script should parse successfully");
}

#[test]
fn test_end_to_end_kernel_codegen() {
    let source = r#"{
        &[0x7A4F]:INPUT:◯ □ 4 4
        (INPUT + INPUT) → OUTPUT
        OUTPUT → =
    }"#;

    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new("test_kernel");
    let ir = codegen.generate_llvm_ir(&block).unwrap();

    assert!(ir.contains("define void @rho_kernel_exec()"));
    assert!(ir.contains("define void @rho_kernel_exec_with_args"));
    assert!(ir.contains("define ptr @rho_kernel_metadata()"));
    assert!(ir.contains("fadd double"));
    assert!(ir.contains("icmp eq ptr %in_ptr, null"));

    let so_result = codegen.compile_to_so(&ir, "target/test_kernel.so");
    assert!(so_result.is_ok(), "LLVM to Native .so compilation should succeed");
}

#[test]
fn test_generated_kernel_writes_output_values() {
    let source = r#"{
        &[0x7A4F]:INPUT:◯ □ 4 1
        (INPUT + 1.0) → OUTPUT
        OUTPUT → =
    }"#;

    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new("runtime_kernel");
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/runtime_kernel.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok());

    let lib = unsafe { libloading::Library::new(so_path).unwrap() };
    let func: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> = unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
    unsafe { func(input.as_ptr(), output.as_mut_ptr()) };

    assert_eq!(output, [2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_dimension_disruption_for_shape_mismatch() {
    let source = r#"{
        INPUT:◯ □ 2 2
        TEMP:◯ □ 3 3
        INPUT → TEMP
        TEMP → =
    }"#;

    let res = parse_rho_program(source);
    assert!(res.is_err());
    if let Err(e) = res {
        assert!(matches!(e, HarmonyDisruption::DimensionErr { .. }));
    }
}

#[test]
fn test_codegen_uses_declared_shape_for_loop_bound() {
    let source = r#"{
        INPUT:◯ □ 8 1
        (INPUT + 1.0) → OUTPUT
        OUTPUT → =
    }"#;

    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new("dynamic_bound_kernel");
    let ir = codegen.generate_llvm_ir(&block).unwrap();

    // The primary space has shape [8, 1], so the sweep is 8 cells — taken from
    // the declaration rather than a fixed guess.
    assert_eq!(codegen.element_count(), 8);
    assert!(ir.contains("sweep 8 cells"), "{ir}");
}


// --------------------------------------------------------------------------
// Regression tests for the lowering rewrite.
// --------------------------------------------------------------------------

/// Compile a block and run it over `input`, returning the output buffer.
fn run_kernel(name: &str, source: &str, input: &[f64]) -> Vec<f64> {
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new(name);
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = format!("target/{name}.so");
    assert!(codegen.compile_to_so(&ir, &so_path).is_ok(), "{name} should link");

    let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
    let func: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
        unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
    let mut output = vec![0.0; input.len()];
    unsafe { func(input.as_ptr(), output.as_mut_ptr()) };
    output
}

#[test]
fn test_shift_right_reads_the_previous_cell() {
    // ▷ used to lower to `fmul x, 1.0`, i.e. the identity.
    let source = r#"{
        INPUT:◯ □ 8 1
        (▷INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("shift_right_kernel", source, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_shift_left_reads_the_next_cell() {
    let source = r#"{
        INPUT:◯ □ 8 1
        (▽INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("shift_left_kernel", source, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0]);
}

#[test]
fn test_each_flow_reads_its_own_source() {
    // Statements used to share one running scalar, so INPUT in the second flow
    // silently meant "whatever the first flow produced".
    let source = r#"{
        INPUT:◯ □ 4 1
        (INPUT + 10.0) → A
        (INPUT + 100.0) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("flow_isolation_kernel", source, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out, vec![101.0, 102.0, 103.0, 104.0]);
}

#[test]
fn test_comparison_masks_instead_of_adding() {
    // >=, <= and == fell through to a catch-all that emitted fadd.
    let source = r#"{
        INPUT:◯ □ 4 1
        (INPUT >= 3.0) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("compare_kernel", source, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out, vec![0.0, 0.0, 3.0, 4.0]);
}

#[test]
fn test_codegen_is_deterministic() {
    // Shapes lived in a HashMap, so allocation order — and which buffer became
    // OUTPUT — changed between runs of the same source.
    let source = r#"{
        &[0x7A4F]:INPUT:◯ □ 4 4
        (▷INPUT - INPUT) → △
        (▽INPUT - INPUT) → ▽
        ((△ - ▽) / (△ + ▽)) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();

    let first = LlvmCodeGen::new("determinism").generate_llvm_ir(&block).unwrap();
    for _ in 0..8 {
        let again = LlvmCodeGen::new("determinism").generate_llvm_ir(&block).unwrap();
        assert_eq!(first, again, "identical source must produce identical IR");
    }
}

#[test]
fn test_shift_of_a_computed_value_is_rejected() {
    let source = r#"{
        INPUT:◯ □ 4 1
        (▷(INPUT + 1.0)) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let err = LlvmCodeGen::new("bad_shift").generate_llvm_ir(&block).unwrap_err();
    assert!(
        matches!(err, HarmonyDisruption::LoweringErr { .. }),
        "expected a lowering failure, got {err:?}"
    );
}

#[test]
fn test_temporaries_are_sized_for_the_whole_sweep() {
    // Flow targets used to default to a 4x4 shape regardless of the grid, so a
    // 1024-cell sweep wrote a long way past a 16-element allocation.
    let source = r#"{
        INPUT:◯ □ 1024 1
        (INPUT + 1.0) → SCRATCH
        (SCRATCH + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new("sizing");
    let ir = codegen.generate_llvm_ir(&block).unwrap();

    assert_eq!(codegen.element_count(), 1024);
    assert!(
        !ir.contains("alloca [16 x double]"),
        "temporary must cover the sweep, not a fixed guess:\n{ir}"
    );

    let input: Vec<f64> = (0..1024).map(|i| i as f64).collect();
    let out = run_kernel("sizing_kernel", source, &input);
    assert_eq!(out[0], 2.0);
    assert_eq!(out[1023], 1025.0);
}

#[test]
fn test_bounded_entrypoint_clamps_the_sweep() {
    let source = r#"{
        INPUT:◯ □ 8 1
        (INPUT + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new("bounded");
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/bounded_kernel.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok());

    let lib = unsafe { libloading::Library::new(so_path).unwrap() };
    let func: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64, i64)> =
        unsafe { lib.get(b"rho_kernel_exec_bounded").unwrap() };

    // The kernel declares 8 cells; the caller owns 4 and says so.
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
    unsafe { func(input.as_ptr(), output.as_mut_ptr(), 4) };
    assert_eq!(output, [2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_metadata_reports_element_count_and_shapes() {
    let source = r#"{
        INPUT:◯ □ 4 4
        (INPUT + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new("metadata");
    let ir = codegen.generate_llvm_ir(&block).unwrap();

    assert_eq!(codegen.element_count(), 16);
    assert!(ir.contains("\\22elements\\22:16"), "metadata must carry the sweep length");
    assert!(ir.contains("define i64 @rho_kernel_element_count()"));
}

#[test]
fn test_shift_respects_row_boundaries_on_a_2d_grid() {
    // A 3x4 grid: a bare ▷ shifts within each row and zeroes the first column,
    // rather than wrapping into the tail of the previous row.
    let source = r#"{
        INPUT:◯ □ 3 4
        (▷INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let input: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let out = run_kernel("grid_row_shift", source, &input);
    assert_eq!(
        out,
        vec![0.0, 1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn test_explicit_axis_shifts_across_rows() {
    // ▷0 pins the shift to axis 0, so each cell reads the row above it.
    let source = r#"{
        INPUT:◯ □ 3 4
        (▷0INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let input: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let out = run_kernel("grid_col_shift", source, &input);
    assert_eq!(
        out,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn test_negative_axis_shift_zeroes_the_far_edge() {
    let source = r#"{
        INPUT:◯ □ 3 4
        (▽0INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let input: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let out = run_kernel("grid_col_shift_neg", source, &input);
    assert_eq!(
        out,
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn test_axis_out_of_range_is_rejected() {
    let source = r#"{
        INPUT:◯ □ 3 4
        (▷7INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let err = LlvmCodeGen::new("bad_axis").generate_llvm_ir(&block).unwrap_err();
    assert!(matches!(err, HarmonyDisruption::LoweringErr { .. }), "got {err:?}");
}



// --------------------------------------------------------------------------
// Constraint solver. These assertions hold for both backends: the interval
// analysis by default, Z3 under --features z3-solver.
// --------------------------------------------------------------------------

use rho_lang::solver::{ConstraintSolver, Verdict};

fn analyze(source: &str) -> rho_lang::solver::Report {
    let block = parse_rho_program(source).unwrap();
    ConstraintSolver::analyze(&block, 0.0)
}

#[test]
fn test_constraint_over_a_square_is_proved() {
    // The old solver accepted this without looking at it.
    let report = analyze(
        r#"{
        INPUT:◯ □ 4 1
        (INPUT ^ 2) → OUTPUT
        ! (OUTPUT >= 0)
        OUTPUT → =
    }"#,
    );
    assert_eq!(report.constraints.len(), 1);
    assert_eq!(
        report.constraints[0].verdict,
        Verdict::Proved,
        "backend {} left it open",
        report.backend
    );
}

#[test]
fn test_unsatisfiable_constraint_is_rejected() {
    let block = parse_rho_program(
        r#"{
        INPUT:◯ □ 4 1
        ((INPUT ^ 2) + 1.0) → OUTPUT
        ! (OUTPUT < 0)
        OUTPUT → =
    }"#,
    )
    .unwrap();
    let err = ConstraintSolver::verify(&block, 0.0).unwrap_err();
    assert!(matches!(err, HarmonyDisruption::LogicErr { .. }), "got {err:?}");
}

#[test]
fn test_constraint_reads_through_earlier_flows() {
    // OUTPUT is defined two flows back; the solver has to inline both to know
    // its sign.
    let report = analyze(
        r#"{
        INPUT:◯ □ 4 1
        (INPUT ^ 2) → A
        (A + 1.0) → OUTPUT
        ! (OUTPUT > 0)
        OUTPUT → =
    }"#,
    );
    assert_eq!(report.constraints[0].verdict, Verdict::Proved);
}

#[test]
fn test_certain_division_by_zero_is_rejected() {
    let block = parse_rho_program(
        r#"{
        INPUT:◯ □ 4 1
        (INPUT / 0.0) → OUTPUT
        OUTPUT → =
    }"#,
    )
    .unwrap();
    let err = ConstraintSolver::verify(&block, 0.0).unwrap_err();
    assert!(matches!(err, HarmonyDisruption::LogicErr { .. }), "got {err:?}");
}

#[test]
fn test_possible_division_by_zero_is_reported_once() {
    // This is why examples/teichmuller.rho returns infinities on smooth input:
    // the denominator is the discrete Laplacian and it can vanish.
    let report = analyze(
        r#"{
        INPUT:◯ □ 8 1
        (▷INPUT - INPUT) → △
        (▽INPUT - INPUT) → ▽
        ((△ - ▽) / (△ + ▽)) → OUTPUT
        OUTPUT → =
    }"#,
    );
    assert_eq!(
        report.divisions.len(),
        1,
        "one division in the source, one finding: {:?}",
        report.divisions
    );
    assert!(
        matches!(report.divisions[0].verdict, Verdict::Unproven(_)),
        "expected an open question, got {:?}",
        report.divisions[0].verdict
    );
}

#[test]
fn test_safe_denominator_is_proved() {
    let report = analyze(
        r#"{
        INPUT:◯ □ 4 1
        (INPUT / ((INPUT ^ 2) + 1.0)) → OUTPUT
        OUTPUT → =
    }"#,
    );
    assert_eq!(report.divisions.len(), 1);
    assert_eq!(
        report.divisions[0].verdict,
        Verdict::Proved,
        "x^2+1 is never zero; backend {} disagreed",
        report.backend
    );
}

// --------------------------------------------------------------------------
// Vector lowering. The scalar path is the reference: SIMD must agree with it
// bit for bit, since both compute the same IEEE operations in the same order.
// --------------------------------------------------------------------------

fn run_variant(name: &str, source: &str, input: &[f64], simd: bool) -> Vec<f64> {
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new(name);
    if !simd {
        codegen = codegen.without_simd();
    }
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = format!("target/{name}.so");
    assert!(codegen.compile_to_so(&ir, &so_path).is_ok(), "{name} should link");

    let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
    let func: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
        unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
    let mut output = vec![0.0; input.len()];
    unsafe { func(input.as_ptr(), output.as_mut_ptr()) };
    output
}

#[test]
fn test_vector_and_scalar_lowering_agree() {
    let programs = [
        ("v_shift", "(▷INPUT) → OUTPUT\n        OUTPUT → ="),
        ("v_neg", "(▽INPUT) → OUTPUT\n        OUTPUT → ="),
        ("v_axis0", "(▷0INPUT - INPUT) → OUTPUT\n        OUTPUT → ="),
        (
            "v_grad",
            "(▷INPUT - INPUT) → GX\n        (▽INPUT - INPUT) → GY\n        \
             ((GX × GX) + (GY × GY)) → OUTPUT\n        OUTPUT → =",
        ),
        (
            "v_mask",
            "(INPUT > 0.5) → A\n        ((A ^ 2) + 1.0) → OUTPUT\n        OUTPUT → =",
        ),
    ];
    let shapes: [&[usize]; 4] = [&[8, 1], &[33, 1], &[5, 7], &[16, 16]];

    for (name, body) in programs {
        for (si, shape) in shapes.iter().enumerate() {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            let n: usize = shape.iter().product();
            let source = format!(
                "{{\n        INPUT:◯ □ {}\n        {body}\n    }}",
                dims.join(" ")
            );

            // A ramp with a couple of awkward values mixed in.
            let input: Vec<f64> = (0..n)
                .map(|i| match i % 5 {
                    0 => i as f64 * 0.5 - 3.0,
                    1 => 0.0,
                    2 => -(i as f64),
                    3 => 1e-9,
                    _ => i as f64 * 1.25,
                })
                .collect();

            let scalar = run_variant(&format!("{name}_s{si}_scalar"), &source, &input, false);
            let vector = run_variant(&format!("{name}_s{si}_simd"), &source, &input, true);

            assert_eq!(
                scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name} on shape {shape:?}: SIMD disagrees with the scalar reference"
            );
        }
    }
}

#[test]
fn test_vector_loop_is_actually_emitted() {
    // A grid with room for whole vector chunks must produce vector IR, not just
    // a scalar loop clang might or might not widen.
    let source = r#"{
        INPUT:◯ □ 64 64
        (▷INPUT - INPUT) → GX
        (GX × GX) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let ir = LlvmCodeGen::new("vector_shape")
        .generate_llvm_ir(&block)
        .unwrap();

    assert!(ir.contains("<4 x double>"), "expected vector types:\n{ir}");
    assert!(ir.contains("shufflevector"), "expected a lane-index splat");
    assert!(
        ir.contains("icmp eq <4 x i64>"),
        "boundary test should be per lane"
    );
}

#[test]
fn test_bounded_entrypoint_stays_scalar() {
    // The bounded call may sweep fewer cells than the grid declares, so a vector
    // window could read past what the caller owns. It is deliberately scalar.
    let source = r#"{
        INPUT:◯ □ 64 64
        (▽INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let ir = LlvmCodeGen::new("bounded_scalar")
        .generate_llvm_ir(&block)
        .unwrap();

    let bounded = ir
        .split("define void @rho_kernel_exec_bounded")
        .nth(1)
        .expect("bounded entrypoint");
    let body = bounded.split("\n}").next().unwrap();
    assert!(
        !body.contains("<4 x double>"),
        "bounded entrypoint must not vectorise:\n{body}"
    );
}

#[test]
fn test_no_simd_flag_produces_only_scalar_loops() {
    let source = r#"{
        INPUT:◯ □ 64 64
        (INPUT + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let ir = LlvmCodeGen::new("scalar_only")
        .without_simd()
        .generate_llvm_ir(&block)
        .unwrap();
    assert!(!ir.contains("<4 x double>"), "--no-simd must stay scalar:\n{ir}");
}

// --------------------------------------------------------------------------
// Solver soundness. The dangerous failure is a proof that is not true, so every
// obligation the solver calls Proved is also checked against the kernel running
// on real inputs. A violation found here must never come from a Proved verdict.
// --------------------------------------------------------------------------

/// Deterministic xorshift, so a failure is reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Spread over a range wide enough to reach overflow and cancellation.
        let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        (unit - 0.5) * 2.0e3
    }
}

/// Inputs worth trying beyond random ones: zeros, ramps, and awkward magnitudes.
fn sample_inputs(n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut sets = vec![
        vec![0.0; n],
        (0..n).map(|i| i as f64).collect(),
        (0..n).map(|i| -(i as f64)).collect(),
        (0..n)
            .map(|i| match i % 4 {
                0 => 1e-300,
                1 => 1e300,
                2 => -1e-300,
                _ => 0.0,
            })
            .collect(),
    ];
    for _ in 0..24 {
        sets.push((0..n).map(|_| rng.next_f64()).collect());
    }
    sets
}

#[test]
fn test_proved_constraints_survive_real_inputs() {
    // Each program constrains OUTPUT, so the equilibrium buffer is exactly what
    // the obligation talks about and can be checked cell by cell.
    // (name, program body, the property OUTPUT must satisfy)
    type ConstraintCase = (&'static str, &'static str, fn(f64) -> bool);
    let cases: [ConstraintCase; 6] = [
        (
            "sound_square",
            "(INPUT ^ 2) → OUTPUT\n        ! (OUTPUT >= 0)\n        OUTPUT → =",
            |v| v >= 0.0,
        ),
        (
            "sound_shifted_square",
            "(▷INPUT - INPUT) → D\n        (D ^ 2) → OUTPUT\n        \
             ! (OUTPUT >= 0)\n        OUTPUT → =",
            |v| v >= 0.0,
        ),
        (
            "sound_positive",
            "((INPUT × INPUT) + 1.0) → OUTPUT\n        ! (OUTPUT > 0)\n        OUTPUT → =",
            |v| v > 0.0,
        ),
        (
            "sound_chained",
            "(INPUT ^ 2) → A\n        (A + 1.0) → OUTPUT\n        \
             ! (OUTPUT > 0)\n        OUTPUT → =",
            |v| v > 0.0,
        ),
        (
            "sound_mask",
            "(INPUT > 0.0) → OUTPUT\n        ! (OUTPUT >= 0)\n        OUTPUT → =",
            |v| v >= 0.0,
        ),
        (
            "sound_open",
            // Deliberately not provable: the solver must not claim otherwise.
            "(▷INPUT - INPUT) → OUTPUT\n        ! (OUTPUT >= 0)\n        OUTPUT → =",
            |v| v >= 0.0,
        ),
    ];

    let mut rng = Rng(0x9E3779B97F4A7C15);
    let n = 32;

    for (name, body, holds) in cases {
        let source = format!("{{\n        INPUT:◯ □ 32 1\n        {body}\n    }}");
        let block = parse_rho_program(&source).unwrap();
        let report = ConstraintSolver::analyze(&block, 0.0);
        assert_eq!(report.constraints.len(), 1, "{name}");
        let verdict = report.constraints[0].verdict.clone();

        let mut counterexample = None;
        for input in sample_inputs(n, &mut rng) {
            let out = run_kernel(name, &source, &input);
            if let Some(bad) = out.iter().copied().find(|v| !holds(*v) && !v.is_nan()) {
                counterexample = Some(bad);
                break;
            }
        }

        match (&verdict, counterexample) {
            // The whole point: a proof must never be contradicted by a run.
            (Verdict::Proved, Some(bad)) => panic!(
                "{name}: backend {} proved the constraint, but the kernel produced {bad}",
                report.backend
            ),
            (Verdict::Proved, None) => {}
            (other, _) => {
                // Everything else is allowed to be imprecise, but the five cases
                // above the last one are simple enough that both backends settle.
                if name != "sound_open" {
                    panic!("{name}: expected a proof from backend {}, got {other:?}", report.backend);
                }
            }
        }
    }
}

#[test]
fn test_proved_denominator_never_divides_by_zero() {
    let source = r#"{
        INPUT:◯ □ 32 1
        (INPUT / ((INPUT ^ 2) + 1.0)) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let report = ConstraintSolver::analyze(&block, 0.0);
    assert_eq!(report.divisions.len(), 1);
    assert_eq!(
        report.divisions[0].verdict,
        Verdict::Proved,
        "x^2+1 is never zero; backend {}",
        report.backend
    );

    // x / (x^2 + 1) is bounded by 1/2, so a proof here means finite output.
    let mut rng = Rng(0xDEADBEEFCAFEF00D);
    for input in sample_inputs(32, &mut rng) {
        let out = run_kernel("sound_denominator", source, &input);
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "denominator was proved non-zero but cell {i} came out {v} for input {:?}",
                input[i]
            );
        }
    }
}

#[test]
fn test_solver_models_binary64_rounding_not_real_arithmetic() {
    // Found by fuzzing the solver against real runs. Over ℝ, (x*x)/x is exactly
    // x, so `(T0 / INPUT) > INPUT` is never true, the mask always yields 0, and
    // `OUTPUT >= 0` looks provable. In binary64 (x*x)/x can exceed x by an ulp,
    // the mask fires, and cubing a negative x breaks the constraint.
    //
    // Both backends now carry a rounding term per operation, so neither may
    // claim this holds.
    let source = r#"{
        INPUT:◯ □ 6 4
        (INPUT × INPUT) → T0
        (((T0 / INPUT) > INPUT) ^ 3.0) → OUTPUT
        ! (OUTPUT >= 0)
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let report = ConstraintSolver::analyze(&block, 0.0);
    assert_ne!(
        report.constraints[0].verdict,
        Verdict::Proved,
        "backend {} proved a constraint that binary64 rounding breaks",
        report.backend
    );
}

#[test]
fn test_rounding_model_still_admits_the_obvious_proofs() {
    // The rounding term must not be so pessimistic that nothing is provable:
    // a squared value stays non-negative through rounding, and x^2 + 1 stays
    // clear of zero.
    for (name, body) in [
        ("round_square", "(INPUT ^ 2) → OUTPUT\n        ! (OUTPUT >= 0)"),
        (
            "round_product",
            "(INPUT × INPUT) → OUTPUT\n        ! (OUTPUT >= 0)",
        ),
        (
            "round_offset",
            "((INPUT ^ 2) + 1.0) → OUTPUT\n        ! (OUTPUT > 0)",
        ),
    ] {
        let source = format!("{{\n        INPUT:◯ □ 4 1\n        {body}\n        OUTPUT → =\n    }}");
        let block = parse_rho_program(&source).unwrap();
        let report = ConstraintSolver::analyze(&block, 0.0);
        assert_eq!(
            report.constraints[0].verdict,
            Verdict::Proved,
            "{name}: backend {} lost a proof it should keep",
            report.backend
        );
    }
}

#[test]
fn test_repeated_shift_shares_one_boundary_condition() {
    // Both reads of GX are the same cell of the same space, so they sit on the
    // row boundary together or not at all. Giving each occurrence its own
    // boundary flag let the solver pick `edge = false` for one and `true` for
    // the other, invent a negative square, and reject a valid program.
    let source = r#"{
        INPUT:◯ □ 2 3
        (▷INPUT - INPUT) → GX
        (GX × GX) → OUTPUT
        ! (OUTPUT >= 0)
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let report = ConstraintSolver::analyze(&block, 0.0);
    assert_eq!(
        report.constraints[0].verdict,
        Verdict::Proved,
        "a square is non-negative; backend {} disagreed",
        report.backend
    );

    // And the program must still compile.
    assert!(ConstraintSolver::verify(&block, 0.0).is_ok());
}

#[test]
fn test_distinct_shifts_keep_distinct_boundaries() {
    // The flags are keyed by the boundary condition, so a forward and a backward
    // shift must not collapse onto the same flag: at the first column ▷ is on a
    // boundary and ▽ is not.
    let source = r#"{
        INPUT:◯ □ 8 1
        (▷INPUT) → A
        (▽INPUT) → B
        (A - B) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("distinct_boundaries", source, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    // A = [0,1,2,3,4,5,6,7], B = [2,3,4,5,6,7,8,0]
    assert_eq!(out, vec![-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 7.0]);
}

#[test]
fn test_zero_copy_binding_writes_the_supplied_address() {
    // --bind bakes a caller's address into the kernel, which is what makes
    // rho_kernel_exec() — the no-argument entrypoint — reachable at all.
    let source = r#"{
        INPUT:◯ □ 4 1
        (INPUT + 100.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();

    let mut grid: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    // as_mut_ptr, not as_ptr: the kernel writes through this address, so the
    // borrow that hands it out has to be the mutable one.
    let address = grid.as_mut_ptr() as u64;

    let mut codegen = LlvmCodeGen::new("zero_copy").bind("INPUT", address);
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/zero_copy.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok());

    let lib = unsafe { libloading::Library::new(so_path).unwrap() };
    let func: libloading::Symbol<unsafe extern "C" fn()> =
        unsafe { lib.get(b"rho_kernel_exec").unwrap() };
    unsafe { func() };

    // No pointer was passed, yet the caller's buffer changed in place.
    assert_eq!(grid, vec![101.0, 102.0, 103.0, 104.0]);
}

#[test]
fn test_unbound_input_makes_the_zero_copy_entrypoint_inert() {
    // Without an address there is nothing to read, so the entrypoint must
    // return rather than dereference whatever the source literal happened to be.
    let source = r#"{
        &[0x7A4F]:INPUT:◯ □ 4 1
        (INPUT + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let ir = LlvmCodeGen::new("literal_addr")
        .generate_llvm_ir(&block)
        .unwrap();

    // The literal from the source is still honoured; only an unbound INPUT is inert.
    assert!(ir.contains("inttoptr i64 31311"), "{ir}");

    let no_binding = r#"{
        INPUT:◯ □ 4 1
        (INPUT + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(no_binding).unwrap();
    let ir = LlvmCodeGen::new("no_addr").generate_llvm_ir(&block).unwrap();
    let exec = ir
        .split("define void @rho_kernel_exec()")
        .nth(1)
        .and_then(|s| s.split("\n}").next())
        .expect("exec entrypoint");
    assert!(
        exec.contains("[INPUT] is unbound"),
        "an unbound source space should make the entrypoint inert:\n{exec}"
    );
}

#[test]
fn test_tau_binds_the_threshold_symbol() {
    // 𝜏 defaults to 0.0; --tau moves the threshold the mask compares against.
    let source = r#"{
        INPUT:◯ □ 4 1
        INPUT > 𝜏 → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let input = [1.0, 2.0, 3.0, 4.0];

    for (tau, expected) in [
        (0.0, vec![1.0, 2.0, 3.0, 4.0]),
        (2.0, vec![0.0, 0.0, 3.0, 4.0]),
        (10.0, vec![0.0, 0.0, 0.0, 0.0]),
    ] {
        let name = format!("tau_{}", tau as i64);
        let mut codegen = LlvmCodeGen::new(&name).with_tau(tau);
        let ir = codegen.generate_llvm_ir(&block).unwrap();
        let so_path = format!("target/{name}.so");
        assert!(codegen.compile_to_so(&ir, &so_path).is_ok());

        let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
        let func: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
            unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
        let mut out = vec![0.0; 4];
        unsafe { func(input.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(out, expected, "tau = {tau}");
    }
}

// --------------------------------------------------------------------------
// Diagnostics. An error that cannot say where it happened is hard to act on.
// --------------------------------------------------------------------------

#[test]
fn test_diagnostics_point_at_the_offending_line() {
    // The block comment spans two lines: line numbering has to survive it.
    let source = "{\n    INPUT:◯ □ 4 1\n\n    /* a comment\n       over two lines */\n    (INPUT + 1.0) → A\n    (UNDECLARD + A) → OUTPUT\n    OUTPUT → =\n}";
    let err = parse_rho_program(source).unwrap_err();
    assert_eq!(err.line(), Some(7), "got {err:?}");

    let rendered = err.render(source);
    assert!(rendered.contains("7 |     (UNDECLARD + A) → OUTPUT"), "{rendered}");
    assert!(rendered.contains('^'), "{rendered}");
}

#[test]
fn test_shape_mismatch_names_its_line() {
    let source = "{\n    INPUT:◯ □ 2 2\n    TEMP:◯ □ 3 3\n    INPUT → TEMP\n    TEMP → =\n}";
    let err = parse_rho_program(source).unwrap_err();
    assert!(matches!(err, HarmonyDisruption::DimensionErr { .. }), "{err:?}");
    assert_eq!(err.line(), Some(4));
}

#[test]
fn test_lowering_failure_names_its_line() {
    let source = "{\n    INPUT:◯ □ 8 1\n    (INPUT + 1.0) → A\n    (▷(A + INPUT)) → OUTPUT\n    OUTPUT → =\n}";
    let block = parse_rho_program(source).unwrap();
    let err = LlvmCodeGen::new("span_lowering")
        .generate_llvm_ir(&block)
        .unwrap_err();
    assert!(matches!(err, HarmonyDisruption::LoweringErr { .. }), "{err:?}");
    assert_eq!(err.line(), Some(4));
}

#[test]
fn test_constraint_failure_names_its_line() {
    let source = "{\n    INPUT:◯ □ 4 1\n    ((INPUT ^ 2) + 1.0) → OUTPUT\n    ! (OUTPUT < 0)\n    OUTPUT → =\n}";
    let block = parse_rho_program(source).unwrap();
    let err = ConstraintSolver::verify(&block, 0.0).unwrap_err();
    assert!(matches!(err, HarmonyDisruption::LogicErr { .. }), "{err:?}");
    assert_eq!(err.line(), Some(4));
}

#[test]
fn test_statements_record_their_source_line() {
    let source = "{\n    INPUT:◯ □ 4 1\n\n    (INPUT + 1.0) → OUTPUT\n\n    OUTPUT → =\n}";
    let block = parse_rho_program(source).unwrap();
    assert_eq!(block.lines, vec![2, 4, 6]);
}

// --------------------------------------------------------------------------
// Contracts. What the solver proves has to reach whoever loads the kernel,
// not just whoever watched the build.
// --------------------------------------------------------------------------

#[test]
fn test_contract_states_the_output_range() {
    let source = r#"{
        INPUT:◯ □ 4 1
        (INPUT ^ 2) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let contract = ConstraintSolver::analyze(&block, 0.0).contract;

    assert_eq!(contract.output_range.lo, 0.0, "a square is never negative");
    assert!(contract.output_range.hi.is_infinite());
    assert!(!contract.output_proven_finite);
    assert!(contract.divisions_proven_safe, "there are no divisions to fail");
}

#[test]
fn test_contract_flags_a_division_that_can_vanish() {
    let source = r#"{
        INPUT:◯ □ 8 1
        (▷INPUT - INPUT) → D
        (INPUT / D) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let contract = ConstraintSolver::analyze(&block, 0.0).contract;

    assert!(!contract.divisions_proven_safe);
    assert!(!contract.is_complete());
    assert!(contract.open_obligations >= 1);
}

#[test]
fn test_contract_is_complete_when_everything_is_proved() {
    let source = r#"{
        INPUT:◯ □ 4 1
        (INPUT ^ 2) → SQ
        (SQ / ((INPUT ^ 2) + 1.0)) → OUTPUT
        ! (OUTPUT >= 0)
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let contract = ConstraintSolver::analyze(&block, 0.0).contract;

    assert!(contract.divisions_proven_safe, "x^2 + 1 is never zero");
    assert_eq!(contract.open_obligations, 0);
    assert!(contract.is_complete());
    assert_eq!(contract.output_range.lo, 0.0);
}

#[test]
fn test_contract_travels_inside_the_shared_library() {
    let source = r#"{
        INPUT:◯ □ 4 1
        (INPUT ^ 2) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let contract = ConstraintSolver::analyze(&block, 0.0).contract;

    let mut codegen = LlvmCodeGen::new("contract_abi").with_contract(contract.to_json());
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/contract_abi.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok());

    let lib = unsafe { libloading::Library::new(so_path).unwrap() };
    let meta: libloading::Symbol<unsafe extern "C" fn() -> *const std::os::raw::c_char> =
        unsafe { lib.get(b"rho_kernel_metadata").unwrap() };
    let json = unsafe { std::ffi::CStr::from_ptr(meta()) }
        .to_str()
        .unwrap()
        .to_string();

    // A caller can read the guarantee without having watched the build.
    assert!(json.contains("\"contract\""), "{json}");
    assert!(json.contains("\"divisions_proven_safe\":true"), "{json}");
    assert!(json.contains("\"output_range\":[0,null]"), "{json}");
    assert!(json.contains("no NaN input"), "assumptions must ship too: {json}");
}

#[test]
fn test_division_by_a_square_plus_one_is_proved_safe() {
    // inf / inf is NaN, so endpoint arithmetic alone gave up here and reported
    // an unbounded range. The sign of the quotient is still known.
    let source = r#"{
        INPUT:◯ □ 4 1
        ((INPUT ^ 2) / ((INPUT ^ 2) + 1.0)) → OUTPUT
        ! (OUTPUT >= 0)
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let report = ConstraintSolver::analyze(&block, 0.0);
    assert_eq!(report.constraints[0].verdict, Verdict::Proved);
    assert_eq!(report.contract.output_range.lo, 0.0);
}

// --------------------------------------------------------------------------
// Reduction. A fold collapses an axis, so for the first time a flow's output
// has a different shape from its input.
// --------------------------------------------------------------------------

#[test]
fn test_fold_sums_an_axis() {
    let source = r#"{
        INPUT:◯ □ 8 1
        ◇+ INPUT → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("fold_sum", source, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(out[0], 36.0);
}

#[test]
fn test_fold_over_a_product_is_a_dot_product() {
    // Not expressible before: it needs the fold to see a computed operand.
    let source = r#"{
        INPUT:◯ □ 4 1
        ◇+ (INPUT × INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("fold_dot", source, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out[0], 30.0);
}

#[test]
fn test_fold_collapses_the_chosen_axis() {
    // A 3x4 grid: axis 1 sums each row, axis 0 sums each column.
    let grid: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, //
        10.0, 20.0, 30.0, 40.0, //
        100.0, 200.0, 300.0, 400.0,
    ];

    let rows = run_kernel(
        "fold_rows",
        "{\n        INPUT:◯ □ 3 4\n        ◇+1 INPUT → OUTPUT\n        OUTPUT → =\n    }",
        &grid,
    );
    assert_eq!(&rows[..3], &[10.0, 100.0, 1000.0]);

    let cols = run_kernel(
        "fold_cols",
        "{\n        INPUT:◯ □ 3 4\n        ◇+0 INPUT → OUTPUT\n        OUTPUT → =\n    }",
        &grid,
    );
    assert_eq!(&cols[..4], &[111.0, 222.0, 333.0, 444.0]);
}

#[test]
fn test_fold_operators_max_min_and_product() {
    let data = [5.0, 3.0, -2.0, 8.0, 1.0, 4.0];
    for (name, glyph, expected) in [
        ("fold_max", "◇>", 8.0),
        ("fold_min", "◇<", -2.0),
        ("fold_prod", "◇×", -960.0),
    ] {
        let source = format!(
            "{{\n        INPUT:◯ □ 6 1\n        {glyph} INPUT → OUTPUT\n        OUTPUT → =\n    }}"
        );
        let out = run_kernel(name, &source, &data);
        assert_eq!(out[0], expected, "{name}");
    }
}

#[test]
fn test_nested_folds_reduce_the_whole_grid() {
    // The inner fold must be lowered before the outer one reads it.
    let source = r#"{
        INPUT:◯ □ 3 4
        ◇+0 (◇+1 INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let grid: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let out = run_kernel("fold_nested", source, &grid);
    assert_eq!(out[0], 78.0, "1..12 sums to 78");
}

#[test]
fn test_fold_composes_with_arithmetic_and_shifts() {
    // Mean of the forward differences.
    let source = r#"{
        INPUT:◯ □ 8 1
        (▽INPUT - INPUT) → D
        ((◇+ D) / 8.0) → OUTPUT
        OUTPUT → =
    }"#;
    let input: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let out = run_kernel("fold_with_shift", source, &input);
    // D = [1,1,1,1,1,1,1,-7]; the sum is 0, so the mean is 0.
    assert_eq!(out[0], 0.0);
}

#[test]
fn test_ascii_alias_for_the_fold_glyph() {
    let source = r#"{
        INPUT:◯ □ 4 1
        <>+ INPUT -> OUTPUT
        OUTPUT -> =
    }"#;
    let out = run_kernel("fold_ascii", source, &[10.0, 20.0, 30.0, 40.0]);
    assert_eq!(out[0], 100.0);
}

#[test]
fn test_a_non_associative_fold_is_rejected() {
    // Folding with - or / has no order-independent meaning.
    for glyph in ["◇-", "◇/"] {
        let source = format!(
            "{{\n        INPUT:◯ □ 4 1\n        {glyph} INPUT → OUTPUT\n        OUTPUT → =\n    }}"
        );
        assert!(
            parse_rho_program(&source).is_err(),
            "{glyph} should not parse as a fold"
        );
    }
}

#[test]
fn test_fold_shrinks_the_declared_shape() {
    let source = r#"{
        INPUT:◯ □ 3 4
        ◇+1 INPUT → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new("fold_shape");
    let ir = codegen.generate_llvm_ir(&block).unwrap();

    // The sweep that writes OUTPUT covers 3 cells, not 12.
    assert!(ir.contains("sweep 3 cells"), "{ir}");
    assert!(ir.contains("over axis 1 of [3, 4] -> 3 cells"), "{ir}");
}

// --------------------------------------------------------------------------
// Lifting and broadcasting. `□aX` inserts a length-1 axis; an element-wise
// operation stretches such an axis against a longer one. Together with a fold
// this is what makes a contraction — and so a matrix product — expressible.
// --------------------------------------------------------------------------

/// Run a kernel whose spaces are bound to caller memory, and return OUTPUT.
fn run_bound(name: &str, source: &str, inputs: &[(&str, &[f64])], out_len: usize) -> Vec<f64> {
    let block = parse_rho_program(source).unwrap();
    let mut buffers: Vec<Vec<f64>> = inputs.iter().map(|(_, data)| data.to_vec()).collect();
    let mut output = vec![0.0f64; out_len];

    let mut codegen = LlvmCodeGen::new(name);
    for ((space, _), buffer) in inputs.iter().zip(buffers.iter_mut()) {
        codegen = codegen.bind(space, buffer.as_mut_ptr() as u64);
    }
    codegen = codegen.bind("OUTPUT", output.as_mut_ptr() as u64);

    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = format!("target/{name}.so");
    assert!(codegen.compile_to_so(&ir, &so_path).is_ok(), "{name} should link:\n{ir}");

    let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
    let func: libloading::Symbol<unsafe extern "C" fn()> =
        unsafe { lib.get(b"rho_kernel_exec").unwrap() };
    unsafe { func() };
    output
}

#[test]
fn test_matrix_multiply_in_one_flow() {
    // A is [2,3] viewed as [2,3,1] and B is [3,4] viewed as [1,3,4]; they
    // stretch to [2,3,4], and folding the shared axis contracts them to [2,4].
    let source = r#"{
        A:◯ □ 2 3 1
        B:◯ □ 1 3 4
        ◇+1 (A × B) → OUTPUT
        OUTPUT → =
    }"#;
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0];

    let out = run_bound("matmul", source, &[("A", &a), ("B", &b)], 8);
    assert_eq!(out, vec![10.0, 5.0, 4.0, 8.0, 22.0, 11.0, 13.0, 20.0]);
}

#[test]
fn test_matrix_multiply_agrees_with_a_reference() {
    for (m, k, n) in [(1, 1, 1), (3, 3, 3), (4, 2, 5), (5, 7, 3)] {
        let source = format!(
            "{{\n    A:◯ □ {m} {k} 1\n    B:◯ □ 1 {k} {n}\n    ◇+1 (A × B) → OUTPUT\n    OUTPUT → =\n}}"
        );
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.75) - 2.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| 1.5 - (i as f64 * 0.25)).collect();

        let out = run_bound(&format!("matmul_{m}_{k}_{n}"), &source, &[("A", &a), ("B", &b)], m * n);

        let mut expected = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                expected.push((0..k).map(|p| a[i * k + p] * b[p * n + j]).sum::<f64>());
            }
        }
        assert_eq!(out, expected, "{m}x{k} times {k}x{n}");
    }
}

#[test]
fn test_lift_makes_an_outer_product() {
    // [4] and [3] lifted to [4,1] and [1,3] stretch to [4,3].
    let source = r#"{
        A:◯ □ 4
        B:◯ □ 3
        ((□1A) × (□0B)) → OUTPUT
        OUTPUT → =
    }"#;
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [10.0, 20.0, 30.0];

    let out = run_bound("outer", source, &[("A", &a), ("B", &b)], 12);
    assert_eq!(
        out,
        vec![
            10.0, 20.0, 30.0, //
            20.0, 40.0, 60.0, //
            30.0, 60.0, 90.0, //
            40.0, 80.0, 120.0,
        ]
    );
}

#[test]
fn test_broadcast_stretches_a_unit_axis() {
    // A column [3,1] plus a row [1,4] fills a [3,4] grid.
    let source = r#"{
        COL:◯ □ 3 1
        ROW:◯ □ 1 4
        (COL + ROW) → OUTPUT
        OUTPUT → =
    }"#;
    let col = [1.0, 2.0, 3.0];
    let row = [10.0, 20.0, 30.0, 40.0];

    let out = run_bound("broadcast", source, &[("COL", &col), ("ROW", &row)], 12);
    assert_eq!(
        out,
        vec![
            11.0, 21.0, 31.0, 41.0, //
            12.0, 22.0, 32.0, 42.0, //
            13.0, 23.0, 33.0, 43.0,
        ]
    );
}

#[test]
fn test_shapes_that_cannot_broadcast_are_rejected() {
    // Ranks must match, and a mismatched axis must be 1 on one side.
    for (a, b) in [("3 4", "5 4"), ("3 4", "4")] {
        let source = format!(
            "{{\n    A:◯ □ {a}\n    B:◯ □ {b}\n    (A + B) → OUTPUT\n    OUTPUT → =\n}}"
        );
        assert!(
            parse_rho_program(&source).is_err(),
            "shapes [{a}] and [{b}] should not broadcast"
        );
    }
}

#[test]
fn test_zero_copy_entrypoint_needs_every_source_bound() {
    // Two inputs and no INPUT space at all: the gate is whether the spaces the
    // kernel reads are bound, not whether one of them happens to be called
    // INPUT.
    let source = r#"{
        A:◯ □ 4
        B:◯ □ 4
        (A + B) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();

    let ir = LlvmCodeGen::new("half_bound")
        .bind("A", 0x1000)
        .generate_llvm_ir(&block)
        .unwrap();
    let exec = ir
        .split("define void @rho_kernel_exec()")
        .nth(1)
        .and_then(|s| s.split("\n}").next())
        .unwrap();
    assert!(exec.contains("[B] is unbound"), "{exec}");

    let ir = LlvmCodeGen::new("all_bound")
        .bind("A", 0x1000)
        .bind("B", 0x2000)
        .generate_llvm_ir(&block)
        .unwrap();
    let exec = ir
        .split("define void @rho_kernel_exec()")
        .nth(1)
        .and_then(|s| s.split("\n}").next())
        .unwrap();
    assert!(!exec.contains("is unbound"), "{exec}");
    assert!(exec.contains("inttoptr i64 8192"), "B should be read from its binding");
}
