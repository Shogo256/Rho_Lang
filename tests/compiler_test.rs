use rho_lang::codegen::LlvmCodeGen;
use rho_lang::error::HarmonyDisruption;
use rho_lang::parser::parse_rho_program;
use rho_lang::tla::generate_tla_spec;

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

    let tla = generate_tla_spec("teichmuller_test", &block);
    assert!(tla.contains("MODULE teichmuller_test"));
    assert!(tla.contains("ShiftRight"));
    assert!(tla.contains("ShiftLeft"));
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
        HarmonyDisruption::SpaceErr { space_name } => {
            assert_eq!(space_name, "UNDECLARED_SPACE");
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
