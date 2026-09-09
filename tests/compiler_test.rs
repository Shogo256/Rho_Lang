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

// --------------------------------------------------------------------------
// Scan. Where a fold answers "what is the total", a scan answers "what is the
// total so far" for every cell, and so keeps the shape it walks.
// --------------------------------------------------------------------------

#[test]
fn test_scan_accumulates_along_an_axis() {
    let source = r#"{
        INPUT:◯ □ 6 1
        ◈+ INPUT → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("scan_sum", source, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0]);
}

#[test]
fn test_scan_operators_product_max_and_min() {
    for (name, glyph, data, expected) in [
        (
            "scan_prod",
            "◈×",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
            vec![1.0, 2.0, 6.0, 24.0, 120.0, 120.0],
        ),
        (
            "scan_max",
            "◈>",
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 2.0],
            vec![3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        ),
        (
            "scan_min",
            "◈<",
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 2.0],
            vec![3.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ] {
        let source = format!(
            "{{\n        INPUT:◯ □ 6 1\n        {glyph} INPUT → OUTPUT\n        OUTPUT → =\n    }}"
        );
        assert_eq!(run_kernel(name, &source, &data), expected, "{name}");
    }
}

#[test]
fn test_scan_keeps_its_shape_and_respects_the_axis() {
    let grid: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, //
        10.0, 20.0, 30.0, 40.0, //
        100.0, 200.0, 300.0, 400.0,
    ];

    // Along a row: each row accumulates on its own.
    let rows = run_kernel(
        "scan_rows",
        "{\n        INPUT:◯ □ 3 4\n        ◈+1 INPUT → OUTPUT\n        OUTPUT → =\n    }",
        &grid,
    );
    assert_eq!(
        rows,
        vec![1.0, 3.0, 6.0, 10.0, 10.0, 30.0, 60.0, 100.0, 100.0, 300.0, 600.0, 1000.0]
    );

    // Down a column: the totals run between rows instead.
    let cols = run_kernel(
        "scan_cols",
        "{\n        INPUT:◯ □ 3 4\n        ◈+0 INPUT → OUTPUT\n        OUTPUT → =\n    }",
        &grid,
    );
    assert_eq!(
        cols,
        vec![1.0, 2.0, 3.0, 4.0, 11.0, 22.0, 33.0, 44.0, 111.0, 222.0, 333.0, 444.0]
    );
}

#[test]
fn test_scan_composes_with_a_fold() {
    // A cumulative distribution: the running total over the total. The fold
    // drops an axis, so `□` puts one back before the division.
    let source = r#"{
        INPUT:◯ □ 4 1
        ((◈+ INPUT) / (□0 (◇+ INPUT))) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("scan_cdf", source, &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(out, vec![0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn test_ascii_alias_for_the_scan_glyph() {
    let source = r#"{
        INPUT:◯ □ 4 1
        <.>+ INPUT -> OUTPUT
        OUTPUT -> =
    }"#;
    let out = run_kernel("scan_ascii", source, &[5.0, 5.0, 5.0, 5.0]);
    assert_eq!(out, vec![5.0, 10.0, 15.0, 20.0]);
}

#[test]
fn test_scan_and_fold_write_different_amounts() {
    // The same expression under each glyph: one keeps the grid, one collapses it.
    let scan = LlvmCodeGen::new("scan_extent")
        .generate_llvm_ir(
            &parse_rho_program(
                "{\n    INPUT:◯ □ 3 4\n    ◈+1 INPUT → OUTPUT\n    OUTPUT → =\n}",
            )
            .unwrap(),
        )
        .unwrap();
    assert!(scan.contains("(running) over axis 1 of [3, 4] -> 12 cells"), "{scan}");
    assert!(scan.contains("sweep 12 cells"), "{scan}");

    let fold = LlvmCodeGen::new("fold_extent")
        .generate_llvm_ir(
            &parse_rho_program(
                "{\n    INPUT:◯ □ 3 4\n    ◇+1 INPUT → OUTPUT\n    OUTPUT → =\n}",
            )
            .unwrap(),
        )
        .unwrap();
    assert!(fold.contains("(total) over axis 1 of [3, 4] -> 3 cells"), "{fold}");
    assert!(fold.contains("sweep 3 cells"), "{fold}");
}

#[test]
fn test_a_non_associative_scan_is_rejected() {
    for glyph in ["◈-", "◈/"] {
        let source = format!(
            "{{\n        INPUT:◯ □ 4 1\n        {glyph} INPUT → OUTPUT\n        OUTPUT → =\n    }}"
        );
        assert!(parse_rho_program(&source).is_err(), "{glyph} should not parse");
    }
}

// --------------------------------------------------------------------------
// The reference interpreter. It is written from the semantics rather than from
// the code generator, so a disagreement between the two means one of them is
// wrong — and says where to look.
// --------------------------------------------------------------------------

use rho_lang::interp::{interpret, Env, Grid};

/// Run a program both ways and require the compiled kernel to match the
/// interpreter cell for cell.
fn agree(name: &str, source: &str, shape: &[usize], input: &[f64]) {
    let block = parse_rho_program(source).unwrap();

    let mut env: Env = Env::new();
    env.insert("INPUT".to_string(), Grid::from(shape.to_vec(), input.to_vec()));
    let interpreted = interpret(&block, &env, 0.0).unwrap();
    let expected = &interpreted.get("OUTPUT").expect("OUTPUT").cells;

    let compiled = run_kernel(name, source, input);

    assert_eq!(
        compiled[..expected.len()]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{name}: the compiler and the interpreter disagree\n  compiled    {:?}\n  interpreted {:?}",
        &compiled[..expected.len()],
        expected
    );
}

#[test]
fn test_compiler_agrees_with_the_reference_interpreter() {
    let ramp: Vec<f64> = (0..12).map(|i| i as f64 - 5.0).collect();
    let mixed = vec![
        2.0, -4.0, 4.0, 0.5, 5.0, -5.0, 7.0, 9.0, -1.0, 3.0, 0.0, 6.0,
    ];

    let cases: [(&str, &str, &[usize]); 12] = [
        ("ref_add", "(INPUT + 1.0) → OUTPUT\n        OUTPUT → =", &[12, 1]),
        ("ref_shift", "(▷INPUT) → OUTPUT\n        OUTPUT → =", &[12, 1]),
        ("ref_shift_neg", "(▽INPUT) → OUTPUT\n        OUTPUT → =", &[12, 1]),
        ("ref_grid_shift", "(▷INPUT - INPUT) → OUTPUT\n        OUTPUT → =", &[3, 4]),
        ("ref_axis0", "(▽0INPUT + INPUT) → OUTPUT\n        OUTPUT → =", &[3, 4]),
        ("ref_mask", "(INPUT > 1.0) → OUTPUT\n        OUTPUT → =", &[12, 1]),
        ("ref_pow", "((INPUT ^ 2) + 1.0) → OUTPUT\n        OUTPUT → =", &[12, 1]),
        ("ref_fold", "◇+ INPUT → OUTPUT\n        OUTPUT → =", &[12, 1]),
        ("ref_fold_axis", "◇>1 INPUT → OUTPUT\n        OUTPUT → =", &[3, 4]),
        ("ref_scan", "◈+ INPUT → OUTPUT\n        OUTPUT → =", &[12, 1]),
        ("ref_scan_axis", "◈+0 INPUT → OUTPUT\n        OUTPUT → =", &[3, 4]),
        (
            "ref_chain",
            "(▷INPUT - INPUT) → D\n        (◈+ (D × D)) → S\n        (S / 2.0) → OUTPUT\n        OUTPUT → =",
            &[12, 1],
        ),
    ];

    for (name, body, shape) in cases {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let source = format!(
            "{{\n        INPUT:◯ □ {}\n        {body}\n    }}",
            dims.join(" ")
        );
        agree(name, &source, shape, &ramp);
        agree(&format!("{name}_b"), &source, shape, &mixed);
    }
}

#[test]
fn test_a_sign_is_not_a_binary_operator() {
    // Found by differential testing. The expression parser split at the
    // rightmost operator without asking whether a + or - was a sign, so
    // `A × -3.0` broke at the minus and left `A ×` behind as if it were a name.
    for (index, (body, expected)) in [
        ("(INPUT × -3.0)", vec![-3.0, -6.0, -9.0, -12.0]),
        ("(INPUT - -2.0)", vec![3.0, 4.0, 5.0, 6.0]),
        ("(INPUT + -1.0)", vec![0.0, 1.0, 2.0, 3.0]),
        ("(INPUT / -2.0)", vec![-0.5, -1.0, -1.5, -2.0]),
        ("((-2.0) × INPUT)", vec![-2.0, -4.0, -6.0, -8.0]),
        ("(INPUT ^ -1.0)", vec![1.0, 0.5, 1.0 / 3.0, 0.25]),
    ]
    .into_iter()
    .enumerate()
    {
        let source = format!(
            "{{\n        INPUT:◯ □ 4 1\n        {body} → OUTPUT\n        OUTPUT → =\n    }}"
        );
        let name = format!("sign_case_{index}");
        let out = run_kernel(&name, &source, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out, expected, "{body}");
    }
}

#[test]
fn test_interpreter_folds_the_right_line() {
    // Also found by differential testing: the interpreter treated the index of
    // a surviving cell as if it were the start of the line that cell
    // summarises, so every row after the first folded the wrong values.
    let source = r#"{
        INPUT:◯ □ 3 4
        ◇>1 INPUT → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let grid: Vec<f64> = (0..12).map(|i| i as f64 - 5.0).collect();

    let mut env = Env::new();
    env.insert("INPUT".to_string(), Grid::from(vec![3, 4], grid));
    let out = interpret(&block, &env, 0.0).unwrap();

    // Row maxima of [-5..-2], [-1..2], [3..6].
    assert_eq!(out["OUTPUT"].cells, vec![-2.0, 2.0, 6.0]);
}

#[test]
fn test_a_whole_exponent_is_repeated_multiplication() {
    // Found by differential testing. Leaving `^` to a maths library made the
    // compiler and the interpreter disagree by an ulp, because the two round a
    // square differently. Pinning whole exponents to repeated multiplication
    // makes the result exact and the same everywhere.
    let x = 3.6295812865328214f64;
    let source = r#"{
        INPUT:◯ □ 4 1
        (INPUT ^ 2.0) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("whole_power", source, &[x, 2.0, -3.0, 0.5]);

    assert_eq!(out[0], x * x, "a square must be exactly the product");
    assert_ne!(out[0], x.powf(2.0), "and not what powf returns for it");
    assert_eq!(out[1], 4.0);
    assert_eq!(out[2], 9.0);
    assert_eq!(out[3], 0.25);

    // Zero and negative exponents keep the same definition.
    let zero = run_kernel(
        "whole_power_zero",
        "{\n        INPUT:◯ □ 4 1\n        (INPUT ^ 0.0) → OUTPUT\n        OUTPUT → =\n    }",
        &[5.0, -2.0, 0.0, 1.0],
    );
    assert_eq!(zero, vec![1.0, 1.0, 1.0, 1.0]);
}

// --------------------------------------------------------------------------
// The emitted IR, read back and run without clang. This separates "did the
// generator mean the right thing" from "did clang build what it was told".
// --------------------------------------------------------------------------

use rho_lang::irvm::{parse_module, Machine, Value};

fn run_emitted_ir(name: &str, source: &str, input: &[f64], out_cells: usize) -> Vec<f64> {
    let block = parse_rho_program(source).unwrap();
    let ir = LlvmCodeGen::new(name).generate_llvm_ir(&block).unwrap();

    let functions = parse_module(&ir);
    let entry = functions
        .iter()
        .find(|f| f.name == "rho_kernel_exec_with_args")
        .expect("the argument entrypoint");

    let mut machine = Machine::new();
    let src = machine.add_buffer(input.to_vec());
    let dst = machine.add_buffer(vec![0.0; out_cells]);
    machine
        .run(entry, &[Value::P(src, 0), Value::P(dst, 0)])
        .unwrap_or_else(|why| panic!("{name}: {why}\n{ir}"));
    machine.buffer(dst).to_vec()
}

#[test]
fn test_the_emitted_ir_computes_what_the_source_means() {
    let input: Vec<f64> = vec![2.0, -4.0, 4.0, 0.5, 5.0, -5.0, 7.0, 9.0, -1.0, 3.0, 0.0, 6.0];

    let cases: [(&str, &str, &[usize], usize); 7] = [
        ("ir_add", "(INPUT + 1.0) → OUTPUT\n        OUTPUT → =", &[12, 1], 12),
        ("ir_shift", "(▷INPUT - INPUT) → OUTPUT\n        OUTPUT → =", &[3, 4], 12),
        ("ir_axis", "(▽0INPUT) → OUTPUT\n        OUTPUT → =", &[3, 4], 12),
        ("ir_mask", "(INPUT > 1.0) → OUTPUT\n        OUTPUT → =", &[12, 1], 12),
        ("ir_fold", "◇+1 INPUT → OUTPUT\n        OUTPUT → =", &[3, 4], 3),
        ("ir_scan", "◈+ INPUT → OUTPUT\n        OUTPUT → =", &[12, 1], 12),
        (
            "ir_chain",
            "(▷INPUT - INPUT) → D\n        ((D ^ 2.0) + 1.0) → OUTPUT\n        OUTPUT → =",
            &[12, 1],
            12,
        ),
    ];

    for (name, body, shape, out_cells) in cases {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let source = format!(
            "{{\n        INPUT:◯ □ {}\n        {body}\n    }}",
            dims.join(" ")
        );

        // What the source means, per the reference interpreter.
        let block = parse_rho_program(&source).unwrap();
        let mut env = Env::new();
        env.insert("INPUT".to_string(), Grid::from(shape.to_vec(), input.clone()));
        let meant = interpret(&block, &env, 0.0).unwrap()["OUTPUT"].cells.clone();

        // What the generator emitted, run on its own.
        let emitted = run_emitted_ir(name, &source, &input, out_cells);

        assert_eq!(
            meant.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            emitted[..meant.len()].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}: the emitted IR does not compute what the source means"
        );
    }
}

#[test]
fn test_the_ir_reader_refuses_what_it_does_not_understand() {
    // A validator that quietly ignored an instruction would be worse than one
    // that stops, so an unknown opcode has to be an error rather than a shrug.
    let functions = parse_module(
        "define void @rho_kernel_exec_with_args(ptr %in_ptr, ptr %out_ptr) #0 {\nentry:\n  %x = frobnicate double 1.0, 2.0\n  ret void\n}\n",
    );
    let mut machine = Machine::new();
    let a = machine.add_buffer(vec![0.0; 4]);
    let b = machine.add_buffer(vec![0.0; 4]);
    let outcome = machine.run(&functions[0], &[Value::P(a, 0), Value::P(b, 0)]);
    assert!(outcome.is_err(), "an unknown opcode must be reported");
}

// --------------------------------------------------------------------------
// Translation validation. Testing says the two agree on the inputs we tried;
// this says they agree on every input, for a given program and shape.
// --------------------------------------------------------------------------

use rho_lang::validate::{compare, expressions, expressions_of, Verdict as Equivalence};

#[test]
fn test_the_emitted_ir_is_equivalent_to_the_source() {
    let cases: [(&str, &[usize]); 6] = [
        ("(INPUT + 1.0) → OUTPUT\n        OUTPUT → =", &[6, 1]),
        ("(▷INPUT - INPUT) → OUTPUT\n        OUTPUT → =", &[2, 3]),
        ("(INPUT > 1.0) → OUTPUT\n        OUTPUT → =", &[6, 1]),
        ("◇+ INPUT → OUTPUT\n        OUTPUT → =", &[6, 1]),
        ("◈+ INPUT → OUTPUT\n        OUTPUT → =", &[6, 1]),
        (
            "(▷INPUT - INPUT) → D\n        ((D × D) + 1.0) → OUTPUT\n        OUTPUT → =",
            &[2, 3],
        ),
    ];

    for (body, shape) in cases {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let source = format!(
            "{{\n        INPUT:◯ □ {}\n        {body}\n    }}",
            dims.join(" ")
        );
        let block = parse_rho_program(&source).unwrap();
        let (left, right) = expressions(&block, shape, 0.0).unwrap();

        // Without the SMT backend this reports NotChecked rather than failing:
        // the default build still gets the graphs, just not the proof.
        match compare(&left, &right) {
            Equivalence::Equivalent | Equivalence::NotChecked(_) => {}
            other => panic!("{body}: {other:?}"),
        }
    }
}

#[test]
fn test_the_validator_catches_a_damaged_kernel() {
    // A checker that never fails proves nothing about the thing it checks.
    let source = "{\n    INPUT:◯ □ 6 1\n    (▷INPUT - INPUT) → D\n    ((D × D) + 1.0) → OUTPUT\n    OUTPUT → =\n}";
    let block = parse_rho_program(source).unwrap();
    let shape = vec![6usize, 1];
    let ir = LlvmCodeGen::new("validator_control")
        .generate_llvm_ir(&block)
        .unwrap();

    for (from, to) in [
        ("fadd double", "fsub double"),
        ("fmul double", "fdiv double"),
        ("icmp ult i64 %f1.idx, 6", "icmp ult i64 %f1.idx, 5"),
    ] {
        if !ir.contains(from) {
            continue;
        }
        let broken = ir.replacen(from, to, 1);
        let (left, right) = expressions_of(&block, &shape, 0.0, &broken).unwrap();
        match compare(&left, &right) {
            Equivalence::Differs { .. } => {}
            // The default build cannot prove anything, so it cannot refute either.
            Equivalence::NotChecked(_) => {}
            other => panic!("damaging `{from}` went unnoticed: {other:?}"),
        }
    }

    // And the undamaged kernel must still come out clean.
    let (left, right) = expressions_of(&block, &shape, 0.0, &ir).unwrap();
    assert!(
        matches!(
            compare(&left, &right),
            Equivalence::Equivalent | Equivalence::NotChecked(_)
        ),
        "the untouched kernel should validate"
    );
}

// --------------------------------------------------------------------------
// Named functions. The board operations are glyphs because they describe a
// shape; these are named because they describe a quantity.
// --------------------------------------------------------------------------

#[test]
fn test_named_functions_compute_what_they_say() {
    for (name, body, input, expected) in [
        (
            "fn_exp",
            "exp INPUT",
            vec![0.0, 1.0, 2.0, -1.0],
            vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2), 1.0 / std::f64::consts::E],
        ),
        (
            "fn_log",
            "log INPUT",
            vec![1.0, std::f64::consts::E, 10.0, 0.5],
            vec![0.0, 1.0, 10f64.ln(), 0.5f64.ln()],
        ),
        ("fn_sqrt", "sqrt INPUT", vec![0.0, 1.0, 4.0, 9.0], vec![0.0, 1.0, 2.0, 3.0]),
        ("fn_abs", "abs INPUT", vec![-3.0, 2.0, -1.0, 0.0], vec![3.0, 2.0, 1.0, 0.0]),
    ] {
        let source = format!(
            "{{\n        INPUT:◯ □ 4 1\n        {body} → OUTPUT\n        OUTPUT → =\n    }}"
        );
        let out = run_kernel(name, &source, &input);
        for (got, want) in out.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{name}: {got} vs {want}");
        }
    }
}

#[test]
fn test_the_indicator_makes_counting_possible() {
    // A mask cannot count: one that passes a value which happens to be zero is
    // indistinguishable from one that blocked it. `ind` answers the predicate.
    let source = r#"{
        INPUT:◯ □ 8 1
        (ind (INPUT > 5.0)) → M
        ◇+ M → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("count_above", source, &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert_eq!(out[0], 2.0, "two cells are above five");

    // The case a mask gets wrong: a value of zero that passes the test.
    let source = r#"{
        INPUT:◯ □ 4 1
        (ind (INPUT > -1.0)) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("count_zero_passes", source, &[0.0, -2.0, 0.0, 3.0]);
    assert_eq!(out, vec![1.0, 0.0, 1.0, 1.0], "zero is greater than minus one");
}

#[test]
fn test_softmax_is_expressible() {
    // Needs exp, a fold, and a lift to line the total up with the vector.
    let source = r#"{
        INPUT:◯ □ 4 1
        exp INPUT → E
        (E / (□0 (◇+ E))) → OUTPUT
        OUTPUT → =
    }"#;
    let out = run_kernel("softmax", source, &[1.0, 2.0, 3.0, 4.0]);
    let total: f64 = out.iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "a softmax sums to one, got {total}");
    assert!(out.windows(2).all(|w| w[0] < w[1]), "and preserves the order");
}

#[test]
fn test_the_solver_learned_what_these_functions_do() {
    // A sine is bounded whatever it was given, which is exactly the kind of
    // fact interval arithmetic can carry and an SMT solver cannot.
    let bounded = analyze(
        r#"{
        INPUT:◯ □ 4 1
        ((sin INPUT) + 2.0) → OUTPUT
        ! (OUTPUT > 0.0)
        OUTPUT → =
    }"#,
    );
    assert_eq!(bounded.constraints[0].verdict, Verdict::Proved);

    // An indicator is zero or one.
    let indicator = analyze(
        r#"{
        INPUT:◯ □ 4 1
        (ind (INPUT > 0.0)) → OUTPUT
        ! (OUTPUT >= 0.0)
        OUTPUT → =
    }"#,
    );
    assert_eq!(indicator.constraints[0].verdict, Verdict::Proved);
    assert_eq!(indicator.contract.output_range.hi, 1.0);
}

#[test]
fn test_a_functions_domain_is_an_obligation() {
    // log of an unknown value is an open question...
    let open = analyze(
        r#"{
        INPUT:◯ □ 4 1
        (log INPUT) → OUTPUT
        OUTPUT → =
    }"#,
    );
    assert_eq!(open.domains.len(), 1);
    assert!(matches!(open.domains[0].verdict, Verdict::Unproven(_)));

    // ...and log of something provably positive is not.
    let settled = analyze(
        r#"{
        INPUT:◯ □ 4 1
        (log ((INPUT ^ 2) + 1.0)) → OUTPUT
        OUTPUT → =
    }"#,
    );
    assert_eq!(settled.domains[0].verdict, Verdict::Proved);
}

#[test]
fn test_a_function_name_cannot_also_be_a_space() {
    // `exp X` would be ambiguous if a space could be called `exp`.
    for name in ["exp", "log", "ind"] {
        let source = format!(
            "{{\n        {name}:◯ □ 4 1\n        {name} → OUTPUT\n        OUTPUT → =\n    }}"
        );
        assert!(parse_rho_program(&source).is_err(), "a space named {name}");
    }

    // A name that merely starts with one is fine.
    let source = r#"{
        exposure:◯ □ 4 1
        (exposure + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    assert!(parse_rho_program(source).is_ok());
}

// --------------------------------------------------------------------------
// Single precision. Narrower numbers halve the memory traffic these kernels
// are bound by, and widen the doubt every proof has to carry.
// --------------------------------------------------------------------------

use rho_lang::numeric::Precision;

/// Compile at the given width and run it, returning the cells widened to f64.
fn run_at(name: &str, source: &str, input: &[f64], precision: Precision) -> Vec<f64> {
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new(name).with_precision(precision);
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = format!("target/{name}.so");
    assert!(codegen.compile_to_so(&ir, &so_path).is_ok(), "{name} should link");

    let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
    match precision {
        Precision::F64 => {
            let func: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
                unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
            let mut out = vec![0.0f64; input.len()];
            unsafe { func(input.as_ptr(), out.as_mut_ptr()) };
            out
        }
        Precision::F32 => {
            let narrow: Vec<f32> = input.iter().map(|v| *v as f32).collect();
            let func: libloading::Symbol<unsafe extern "C" fn(*const f32, *mut f32)> =
                unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
            let mut out = vec![0.0f32; narrow.len()];
            unsafe { func(narrow.as_ptr(), out.as_mut_ptr()) };
            out.iter().map(|v| *v as f64).collect()
        }
    }
}

#[test]
fn test_a_narrow_kernel_computes_in_narrow_arithmetic() {
    let source = r#"{
        INPUT:◯ □ 8 1
        (▷INPUT - INPUT) → D
        ((D × D) + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    // Values single precision holds exactly, so both widths must agree.
    let exact = [1.5, 2.25, 3.125, 4.0, 5.5, 6.75, 7.0, 8.5];
    assert_eq!(
        run_at("prec_exact_64", source, &exact, Precision::F64),
        run_at("prec_exact_32", source, &exact, Precision::F32),
    );

    // And the emitted IR really is single precision, not a double one rounded.
    let block = parse_rho_program(source).unwrap();
    let ir = LlvmCodeGen::new("prec_ir")
        .with_precision(Precision::F32)
        .generate_llvm_ir(&block)
        .unwrap();
    assert!(ir.contains("load float"), "{ir}");
    assert!(ir.contains("<4 x float>"), "{ir}");
    assert!(!ir.contains("double"), "no double should survive:\n{ir}");
    assert!(
        ir.contains("precision\\22:\\22f32"),
        "the metadata should record the width"
    );
}

#[test]
fn test_the_interpreter_follows_the_kernel_into_single_precision() {
    // A value f32 cannot hold exactly: the two widths must now differ, and the
    // compiled kernel must match the narrow interpreter rather than the wide one.
    let source = r#"{
        INPUT:◯ □ 4 1
        ((INPUT × INPUT) + INPUT) → OUTPUT
        OUTPUT → =
    }"#;
    let awkward = [0.1, 1.0 / 3.0, 1e-8, 12345.678];

    let compiled = run_at("prec_narrow", source, &awkward, Precision::F32);

    let block = parse_rho_program(source).unwrap();
    let narrow: Vec<f32> = awkward.iter().map(|v| *v as f32).collect();
    let mut env: rho_lang::interp::Env<f32> = rho_lang::interp::Env::new();
    env.insert("INPUT".to_string(), Grid::from(vec![4, 1], narrow));
    let meant = interpret(&block, &env, 0.0).unwrap()["OUTPUT"].cells.clone();

    assert_eq!(
        compiled.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        meant,
        "a narrow kernel is checked against narrow arithmetic"
    );
}

#[test]
fn test_a_proof_carries_the_width_it_was_made_at() {
    // The rounding model follows the precision: 2^-24 instead of 2^-53. A
    // bound proved at f64 is not the same bound at f32, and the contract says
    // which one it is.
    let source = r#"{
        INPUT:◯ □ 4 1
        ((INPUT ^ 2) + 1.0) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();

    let wide = ConstraintSolver::analyze_at(&block, 0.0, Precision::F64).contract;
    let narrow = ConstraintSolver::analyze_at(&block, 0.0, Precision::F32).contract;

    assert_eq!(wide.precision, Precision::F64);
    assert_eq!(narrow.precision, Precision::F32);
    assert!(
        narrow.output_range.lo < wide.output_range.lo,
        "narrower numbers admit a wider range: {} vs {}",
        narrow.output_range.lo,
        wide.output_range.lo
    );
    assert!(narrow.to_json().contains("\"precision\":\"f32\""));
}

// --------------------------------------------------------------------------
// Every space at call time. `rho_kernel_exec_spaces` takes one pointer per
// space in metadata order, so a kernel with two inputs runs without any
// address baked in at compile time.
// --------------------------------------------------------------------------

/// The spaces a kernel lists in its metadata, in table order: (name, cells, role).
fn kernel_spaces(lib: &libloading::Library) -> Vec<(String, usize, String)> {
    let meta: libloading::Symbol<unsafe extern "C" fn() -> *const std::os::raw::c_char> =
        unsafe { lib.get(b"rho_kernel_metadata").unwrap() };
    let json = unsafe { std::ffi::CStr::from_ptr(meta()) }
        .to_str()
        .unwrap()
        .to_string();
    let spaces = json
        .split("\"spaces\":[")
        .nth(1)
        .and_then(|s| s.split("],\"bindings\"").next())
        .unwrap_or_else(|| panic!("no spaces array in {json}"));
    spaces
        .split("{\"name\":\"")
        .skip(1)
        .map(|entry| {
            let name = entry.split('"').next().unwrap().to_string();
            let shape = entry.split("\"shape\":[").nth(1).unwrap().split(']').next().unwrap();
            let cells: usize = shape
                .split(',')
                .map(|d| d.trim().parse::<usize>().unwrap())
                .product();
            let role = entry.split("\"role\":\"").nth(1).unwrap().split('"').next().unwrap();
            (name, cells, role.to_string())
        })
        .collect()
}

/// Compile `source` and run it through `rho_kernel_exec_spaces` with the
/// buffers named in `supplied`; every other space is left to the kernel.
/// Returns each supplied buffer after the call.
fn run_spaces(
    name: &str,
    source: &str,
    supplied: &[(&str, Vec<f64>)],
) -> std::collections::BTreeMap<String, Vec<f64>> {
    let block = parse_rho_program(source).unwrap();
    let mut codegen = LlvmCodeGen::new(name);
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = format!("target/{name}.so");
    assert!(codegen.compile_to_so(&ir, &so_path).is_ok(), "{name} should link:\n{ir}");

    let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
    let spaces = kernel_spaces(&lib);
    let mut buffers: std::collections::BTreeMap<String, Vec<f64>> = supplied
        .iter()
        .map(|(n, data)| (n.to_string(), data.clone()))
        .collect();
    for supplied_name in buffers.keys() {
        assert!(
            spaces.iter().any(|(n, _, _)| n == supplied_name),
            "{supplied_name} is not a space of this kernel: {spaces:?}"
        );
    }
    let table: Vec<*mut f64> = spaces
        .iter()
        .map(|(n, _, _)| {
            buffers
                .get_mut(n)
                .map(|b| b.as_mut_ptr())
                .unwrap_or(std::ptr::null_mut())
        })
        .collect();

    let func: libloading::Symbol<unsafe extern "C" fn(*const *mut f64)> =
        unsafe { lib.get(b"rho_kernel_exec_spaces").unwrap() };
    unsafe { func(table.as_ptr()) };
    buffers
}

const MATMUL_2_3_4: &str = r#"{
    A:◯ □ 2 3 1
    B:◯ □ 1 3 4
    ◇+1 (A × B) → OUTPUT
    OUTPUT → =
}"#;

const GRADIENT_4_4: &str = r#"{
    INPUT:◯ □ 4 4
    (▷INPUT - INPUT) → GX
    (▽0INPUT - INPUT) → GY
    ((GX × GX) + (GY × GY)) → OUTPUT
    OUTPUT → =
}"#;

#[test]
fn test_exec_spaces_runs_a_matrix_product_without_bindings() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0];
    let out = run_spaces(
        "spaces_matmul",
        MATMUL_2_3_4,
        &[("A", a), ("B", b), ("OUTPUT", vec![0.0; 8])],
    );
    assert_eq!(out["OUTPUT"], vec![10.0, 5.0, 4.0, 8.0, 22.0, 11.0, 13.0, 20.0]);
}

#[test]
fn test_metadata_lists_every_space_with_its_role_in_table_order() {
    let block = parse_rho_program(GRADIENT_4_4).unwrap();
    let mut codegen = LlvmCodeGen::new("spaces_roles");
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/spaces_roles.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok());
    let lib = unsafe { libloading::Library::new(so_path).unwrap() };

    // The order is the table order, and the role says what a caller does
    // with each entry: supply it, read it back, or leave it to the kernel.
    let spaces = kernel_spaces(&lib);
    let expected: Vec<(String, usize, String)> = [
        ("GX", 16, "internal"),
        ("GY", 16, "internal"),
        ("INPUT", 16, "input"),
        ("OUTPUT", 16, "output"),
    ]
    .iter()
    .map(|(n, c, r)| (n.to_string(), *c, r.to_string()))
    .collect();
    assert_eq!(spaces, expected);

    // The IR says the same thing next to each load.
    assert!(ir.contains("; [GX] is spaces[0]"), "{ir}");
    assert!(ir.contains("; [OUTPUT] is spaces[3]"), "{ir}");
}

#[test]
fn test_exec_spaces_leaves_an_unsupplied_intermediate_to_the_kernel() {
    let input: Vec<f64> = (0..16).map(|i| (i as f64 * 0.7).sin() * 3.0).collect();

    // What the two-pointer entrypoint computes.
    let block = parse_rho_program(GRADIENT_4_4).unwrap();
    let mut codegen = LlvmCodeGen::new("spaces_gradient_ref");
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/spaces_gradient_ref.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok());
    let lib = unsafe { libloading::Library::new(so_path).unwrap() };
    let with_args: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
        unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
    let mut reference = vec![0.0f64; 16];
    unsafe { with_args(input.as_ptr(), reference.as_mut_ptr()) };

    // GX and GY are not supplied: the kernel owns them for the call.
    let out = run_spaces(
        "spaces_gradient",
        GRADIENT_4_4,
        &[("INPUT", input.clone()), ("OUTPUT", vec![0.0; 16])],
    );
    assert_eq!(out["OUTPUT"], reference);

    // Supplied, an intermediate becomes visible to the caller.
    let out = run_spaces(
        "spaces_gradient_gx",
        GRADIENT_4_4,
        &[
            ("INPUT", input.clone()),
            ("GX", vec![0.0; 16]),
            ("OUTPUT", vec![0.0; 16]),
        ],
    );
    assert_eq!(out["OUTPUT"], reference);
    // ▷ reads the preceding cell along the axis, zero past the boundary.
    let gx: Vec<f64> = (0..16)
        .map(|i| {
            let previous = if i % 4 == 0 { 0.0 } else { input[i - 1] };
            previous - input[i]
        })
        .collect();
    assert_eq!(out["GX"], gx);
}

#[test]
fn test_exec_spaces_returns_without_writing_when_an_input_is_null() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sentinel = vec![-1.0; 8];

    // B is missing, so the kernel must not run at all.
    let out = run_spaces(
        "spaces_missing_input",
        MATMUL_2_3_4,
        &[("A", a), ("OUTPUT", sentinel.clone())],
    );
    assert_eq!(out["OUTPUT"], sentinel);

    // A null table is refused the same way.
    let lib = unsafe { libloading::Library::new("target/spaces_missing_input.so").unwrap() };
    let func: libloading::Symbol<unsafe extern "C" fn(*const *mut f64)> =
        unsafe { lib.get(b"rho_kernel_exec_spaces").unwrap() };
    unsafe { func(std::ptr::null()) };
}

#[test]
fn test_exec_spaces_and_exec_with_args_agree_bit_for_bit() {
    // A shift, a fold and a scan, so the head, the vector body, the tail and
    // the prepasses all run through the new entrypoint's buffers.
    let source = r#"{
        INPUT:◯ □ 3 4
        (▷INPUT - INPUT) → D
        (◈+ D) → S
        ((S × S) + (□1 (◇+1 INPUT))) → OUTPUT
        OUTPUT → =
    }"#;
    let block = parse_rho_program(source).unwrap();
    let input: Vec<f64> = (0..12).map(|i| (i as f64 * 1.3).cos() * 5.0).collect();

    let mut codegen = LlvmCodeGen::new("spaces_agree");
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/spaces_agree.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok(), "{ir}");
    let lib = unsafe { libloading::Library::new(so_path).unwrap() };

    let with_args: libloading::Symbol<unsafe extern "C" fn(*const f64, *mut f64)> =
        unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
    let mut expected = vec![0.0f64; 12];
    unsafe { with_args(input.as_ptr(), expected.as_mut_ptr()) };

    let spaces = kernel_spaces(&lib);
    assert_eq!(
        spaces.iter().map(|(n, _, _)| n.as_str()).collect::<Vec<_>>(),
        vec!["D", "INPUT", "OUTPUT", "S"]
    );
    let mut input_copy = input.clone();
    let mut actual = vec![0.0f64; 12];
    let table: Vec<*mut f64> = vec![
        std::ptr::null_mut(),
        input_copy.as_mut_ptr(),
        actual.as_mut_ptr(),
        std::ptr::null_mut(),
    ];
    let exec_spaces: libloading::Symbol<unsafe extern "C" fn(*const *mut f64)> =
        unsafe { lib.get(b"rho_kernel_exec_spaces").unwrap() };
    unsafe { exec_spaces(table.as_ptr()) };

    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&actual), bits(&expected));

    // The same holds at single precision: the table is of float buffers then.
    let mut codegen = LlvmCodeGen::new("spaces_agree_f32")
        .with_precision(rho_lang::numeric::Precision::F32);
    let ir = codegen.generate_llvm_ir(&block).unwrap();
    let so_path = "target/spaces_agree_f32.so";
    assert!(codegen.compile_to_so(&ir, so_path).is_ok(), "{ir}");
    let lib = unsafe { libloading::Library::new(so_path).unwrap() };

    let narrow: Vec<f32> = input.iter().map(|x| *x as f32).collect();
    let with_args: libloading::Symbol<unsafe extern "C" fn(*const f32, *mut f32)> =
        unsafe { lib.get(b"rho_kernel_exec_with_args").unwrap() };
    let mut expected = vec![0.0f32; 12];
    unsafe { with_args(narrow.as_ptr(), expected.as_mut_ptr()) };

    let mut narrow_copy = narrow.clone();
    let mut actual = vec![0.0f32; 12];
    let table: Vec<*mut f32> = vec![
        std::ptr::null_mut(),
        narrow_copy.as_mut_ptr(),
        actual.as_mut_ptr(),
        std::ptr::null_mut(),
    ];
    let exec_spaces: libloading::Symbol<unsafe extern "C" fn(*const *mut f32)> =
        unsafe { lib.get(b"rho_kernel_exec_spaces").unwrap() };
    unsafe { exec_spaces(table.as_ptr()) };
    let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits32(&actual), bits32(&expected));
}

#[test]
fn test_equilibrium_target_declares_output_in_the_metadata() {
    // `X → =` writes OUTPUT without ever naming it, and a caller that supplies
    // every buffer has to be told that space exists and how large it is.
    let source = r#"{
        INPUT:◯ □ 2 3
        ◇+1 INPUT → =
    }"#;
    let out = run_spaces(
        "spaces_equilibrium",
        source,
        &[
            ("INPUT", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("OUTPUT", vec![0.0; 2]),
        ],
    );
    assert_eq!(out["OUTPUT"], vec![6.0, 15.0]);

    let lib = unsafe { libloading::Library::new("target/spaces_equilibrium.so").unwrap() };
    let spaces = kernel_spaces(&lib);
    assert_eq!(
        spaces,
        vec![
            ("INPUT".to_string(), 6, "input".to_string()),
            ("OUTPUT".to_string(), 2, "output".to_string()),
        ]
    );
}
