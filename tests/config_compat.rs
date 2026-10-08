//! Parse legacy YAML fixtures and example configs (unknown keys tolerated).

use std::path::PathBuf;

use oxllm_server::parse_config_str;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/litellm_configs")
}

fn read_fixture(name: &str) -> String {
    std::fs::read_to_string(fixtures_dir().join(name))
        .unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
}

/// Configure only a child's environment. A serial-test lock cannot make
/// process-global set_var/remove_var safe while unrelated tests or libraries
/// read the environment. Command::env also preserves every parent value on
/// success, panic, or failed assertions without any cleanup mutation.
fn in_env_subprocess(test_name: &str, vars: &[(&str, &str)]) -> bool {
    const CHILD_TEST: &str = "OXLLM_CONFIG_COMPAT_CHILD_TEST";
    if std::env::var(CHILD_TEST).ok().as_deref() == Some(test_name) {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(CHILD_TEST, test_name)
        .envs(vars.iter().copied())
        .output()
        .expect("run environment-isolated config test");
    assert!(
        output.status.success(),
        "child test {test_name} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
        "child filter must execute exactly one test; check the name {test_name}"
    );
    false
}

#[test]
fn simple_config_parses() {
    parse_config_str(&read_fixture("simple_config.yaml")).expect("simple_config");
}

#[test]
fn load_balancer_parses() {
    let cfg = parse_config_str(&read_fixture("load_balancer.yaml")).expect("load_balancer");
    assert_eq!(cfg.model_list.len(), 4);
    assert_eq!(cfg.oxllm_settings.drop_params, Some(true));
    assert!(
        cfg.model_list[0].extra.contains_key("rpm"),
        "row-level rpm is captured on the model entry"
    );
    let dep: oxllm_core::Deployment = cfg.model_list[0].clone().into();
    assert_eq!(
        dep.oxllm_params.rpm,
        Some(3),
        "row-level rpm must become a deployment cap"
    );
    assert_eq!(dep.oxllm_params.tpm, Some(20_000));
}

#[test]
fn aliases_config_parses() {
    let cfg = parse_config_str(&read_fixture("aliases_config.yaml")).expect("aliases");
    assert_eq!(cfg.model_list.len(), 3);
    assert_eq!(cfg.model_list[1].oxllm_params.model, "ollama/llama2");
}

/// The legacy settings key must keep parsing after the
/// rename to `oxllm_settings` (configs written before the rename, and
/// legacy configs generally, use the old spelling).
#[test]
fn legacy_litellm_settings_key_still_parses() {
    let legacy = r#"
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
      api_key: sk-test
litellm_settings:
  drop_params: true
  num_retries: 3
  request_timeout: 42
general_settings:
  master_key: sk-master-key-test-000000
"#;
    let cfg = parse_config_str(legacy).expect("legacy litellm_settings key must parse");
    assert_eq!(
        cfg.oxllm_settings.drop_params,
        Some(true),
        "legacy key must populate the renamed settings block"
    );
    assert_eq!(cfg.oxllm_settings.routing.num_retries, Some(3));
    assert_eq!(cfg.oxllm_settings.request_timeout, Some(42));
    // The legacy key must not be captured as an unknown `extra` key either.
    assert!(
        cfg.extra.get("litellm_settings").is_none(),
        "legacy key should be consumed by the alias, not captured by the flatten"
    );
}

/// A config carrying BOTH spellings is ambiguous, and serde's standard alias
/// semantics reject it with a duplicate-field error rather than guessing
/// which block wins. Pin that: a silent pick between two conflicting blocks
/// would be a config footgun.
#[test]
fn both_settings_spellings_rejected_as_ambiguous() {
    let both = r#"
model_list: []
oxllm_settings:
  request_timeout: 10
litellm_settings:
  request_timeout: 20
"#;
    let err = parse_config_str(both).expect_err("both spellings must be rejected");
    let chain = format!("{err:#}");
    assert!(
        chain.contains("duplicate field"),
        "expected a duplicate-field error in the chain, got: {chain}"
    );
}

#[test]
fn router_settings_parses_and_preserves_extra() {
    let cfg = parse_config_str(&read_fixture("router_settings.yaml")).expect("router_settings");
    assert_eq!(cfg.model_list.len(), 2);
    // The `router_settings` block is typed for the common routing knobs
    // (routing_strategy, cooldown_time, num_retries, fallbacks, …). They land
    // on typed fields; truly unknown keys still bubble to
    // `router_settings.extra`.
    assert_eq!(
        cfg.router_settings.routing.routing_strategy.as_deref(),
        Some("simple-shuffle")
    );
    assert_eq!(cfg.router_settings.routing.cooldown_time, Some(120));
    assert_eq!(cfg.router_settings.routing.num_retries, Some(2));
}

#[test]
fn cache_with_params_resolves_env() {
    if !in_env_subprocess(
        "cache_with_params_resolves_env",
        &[("OXLLM_TEST_REDIS_HOST", "redis.env-resolved.test")],
    ) {
        return;
    }
    const YAML: &str = r#"
model_list:
  - model_name: openai-model
    oxllm_params:
      model: gpt-3.5-turbo
oxllm_settings:
  cache:
    type: redis
    host: os.environ/OXLLM_TEST_REDIS_HOST
general_settings:
  allow_private_api_base: true
"#;
    let cfg = parse_config_str(YAML).expect("cache_with_params");
    let cache = cfg.oxllm_settings.cache.as_ref().expect("cache block");
    assert_eq!(cache.cache_type.as_deref(), Some("redis"));
    assert_eq!(
        cache.host.as_deref(),
        Some("redis.env-resolved.test"),
        "os.environ/ should resolve at load"
    );
}

#[test]
fn cache_with_params_fixture_parses() {
    let cfg = parse_config_str(&read_fixture("cache_with_params.yaml")).expect("cache_with_params");
    let cache = cfg.oxllm_settings.cache.as_ref().expect("cache block");
    assert_eq!(cache.cache_type.as_deref(), Some("redis"));
    assert_eq!(cache.host.as_deref(), Some("redis.example.test"));
}

#[test]
fn all_vendored_config_fixtures_parse() {
    let dir = fixtures_dir();
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .expect("fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".yaml") || n.ends_with(".yml"))
        .collect();
    names.sort();
    assert!(!names.is_empty(), "expected at least one fixture");
    for name in names {
        let yaml = read_fixture(&name);
        parse_config_str(&yaml).unwrap_or_else(|e| panic!("fixture {name} should parse: {e}"));
    }
}

// ---------------------------------------------------------------------------
// Typed parsing of router settings and advanced oxllm_settings.
// ---------------------------------------------------------------------------

#[test]
fn content_policy_fallbacks_and_retry_policy_parse() {
    let yaml = r#"
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o

oxllm_settings:
  content_policy_fallbacks:
    - gpt-4o: [claude-guarded, llama-guarded]
  retry_policy:
    "429": 5
    "500": 3
"#;
    let cfg = parse_config_str(yaml).expect("oxllm_settings should parse");
    let cpf = cfg
        .oxllm_settings
        .routing
        .content_policy_fallbacks
        .as_ref()
        .expect("content_policy_fallbacks");
    assert_eq!(cpf[0]["gpt-4o"][0], "claude-guarded");
    let policy = cfg
        .oxllm_settings
        .routing
        .retry_policy
        .as_ref()
        .expect("retry_policy");
    assert_eq!(policy.get("429"), Some(&5));
    assert_eq!(policy.get("500"), Some(&3));
}

#[test]
fn model_group_alias_parses() {
    let yaml = r#"
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
  - model_name: gpt-4-turbo
    oxllm_params:
      model: openai/gpt-4-turbo

router_settings:
  model_group_alias:
    gpt-4: [gpt-4o, gpt-4-turbo]
"#;
    let cfg = parse_config_str(yaml).expect("router_settings should parse");
    let alias = cfg
        .router_settings
        .model_group_alias
        .as_ref()
        .expect("model_group_alias");
    assert_eq!(
        alias.get("gpt-4"),
        Some(&vec!["gpt-4o".to_string(), "gpt-4-turbo".to_string()])
    );
}

#[test]
fn example_config_parses() {
    // The example config references several env vars. Provide stubs so the
    // os.environ/ resolver succeeds.
    let keys = [
        "OPENAI_API_KEY",
        "OPENAI_API_KEY_BACKUP",
        "AZURE_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "MISTRAL_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
        "OLLAMA_API_KEY",
        "COHERE_API_KEY",
        "GEMINI_API_KEY",
        "GCP_ACCESS_TOKEN",
        "VERTEX_AI_PROJECT",
        "OXLLM_MASTER_KEY",
    ];
    let vars: Vec<_> = keys
        .into_iter()
        .map(|key| (key, "config-compat-test-credential"))
        .collect();
    if !in_env_subprocess("example_config_parses", &vars) {
        return;
    }
    let yaml = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/config.example.yaml"),
    )
    .expect("examples/config.example.yaml readable");
    let cfg = parse_config_str(&yaml).expect("examples/config.example.yaml should parse");
    // router_settings is now a typed block (possibly empty).
    assert!(
        cfg.router_settings.model_group_alias.is_none(),
        "example intentionally leaves model_group_alias commented out"
    );
}
