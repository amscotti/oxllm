//! Pins the production behavior of the `environment_variables` config block.
//!
//! `std::env::set_var` is `unsafe` in edition 2024, and `oxllm-server` is
//! `#![cfg_attr(not(test), forbid(unsafe_code))]`. Because `cfg(test)` is set
//! only for the crate currently being compiled, the writing arm of
//! `apply_env_entries` runs inside the library's own unit tests but NOT here
//! (integration tests compile the library as a dependency, so `cfg(test)` is
//! false for it). This test asserts that reality: a config declaring an
//! `environment_variables` entry is parsed without error, but the value is
//! deliberately NOT written to the process environment.
//!
//! See `config::loader::apply_environment_variables` for the full rationale.

use oxllm_server::parse_config_str;

const VAR: &str = "OXLLM_INTEG_ENV_NOOP_SHIPPED";

#[test]
fn environment_variables_block_is_not_applied_in_integration_build() {
    // Best-effort cleanup; the var is not expected to exist. Wrapped in
    // `unsafe` because `remove_var` is `unsafe` in edition 2024.
    unsafe {
        std::env::remove_var(VAR);
    }

    let yaml = "\
model_list: []

environment_variables:
  OXLLM_INTEG_ENV_NOOP_SHIPPED: from-config
";
    // Parsing succeeds: the block is validated (string values, mapping shape).
    let _cfg = parse_config_str(yaml).expect("config with env block should parse");

    // But the value is NOT applied to the process environment in this build,
    // because the library is compiled without its own `cfg(test)` here.
    let result = std::env::var(VAR);
    assert!(
        result.is_err(),
        "environment_variables must be a documented no-op outside the \
         crate's own unit tests, but the var was set to: {:?}",
        result.ok()
    );
}
