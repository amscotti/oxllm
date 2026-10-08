//! Inline unit tests for the config serde types. Imported via
//! `#[cfg(test)] mod tests;` from `types/mod.rs`.

use super::*;

#[test]
fn parse_minimal_config() {
    let yaml = "model_list:\n  - model_name: gpt-4o\n    oxllm_params:\n      model: openai/gpt-4o\n      api_key: sk-test\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("minimal config should parse");
    assert_eq!(cfg.model_list.len(), 1);
    assert_eq!(cfg.model_list[0].model_name, "gpt-4o");
    assert_eq!(cfg.model_list[0].oxllm_params.model, "openai/gpt-4o");
}

#[test]
fn unknown_top_level_keys_captured() {
    let yaml = "model_list: []\nfuture_section:\n foo: bar\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("unknown keys should not fail");
    assert!(cfg.extra.contains_key("future_section"));
}

#[test]
fn unknown_model_keys_captured() {
    let yaml = "model_list:\n  - model_name: gpt-4o\n    oxllm_params:\n      model: openai/gpt-4o\n    custom_field: 42\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("unknown model keys should not fail");
    assert_eq!(
        cfg.model_list[0]
            .extra
            .get("custom_field")
            .and_then(|v| v.as_i64()),
        Some(42)
    );
}

#[test]
fn full_config_parse() {
    let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
      api_key: sk-test
      timeout: 300
      rpm: 480

oxllm_settings:
  drop_params: true
  num_retries: 3
  request_timeout: 600
  routing_strategy: simple_shuffle

general_settings:
  master_key: sk-master-test
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("full config should parse");
    assert_eq!(cfg.oxllm_settings.drop_params, Some(true));
    assert_eq!(cfg.oxllm_settings.routing.num_retries, Some(3));
    assert_eq!(
        cfg.general_settings.master_key.as_deref(),
        Some("sk-master-test")
    );
    assert_eq!(cfg.model_list[0].oxllm_params.timeout, Some(300.0));
}

#[test]
fn empty_config_uses_defaults() {
    let yaml = "model_list: []\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("empty config should parse");
    assert!(cfg.model_list.is_empty());
    assert!(cfg.oxllm_settings.drop_params.is_none());
    assert!(cfg.general_settings.master_key.is_none());
}

// Bug #49 — explicit `null` for a struct-valued settings block must
// parse as the default rather than being rejected as `invalid type:
// null`. Some YAML emitters and hand-written configs that have a
// top-level key with only comments underneath produce this shape.
#[test]
fn explicit_null_settings_parses_as_default() {
    let yaml = "model_list: []\noxllm_settings:\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("null oxllm_settings should parse");
    // Assert against the meaningful fields rather than the whole struct
    // (these types don't derive `PartialEq`).
    assert!(cfg.oxllm_settings.drop_params.is_none());
    assert!(cfg.oxllm_settings.request_timeout.is_none());
}

#[test]
fn explicit_null_general_settings_parses_as_default() {
    let yaml = "model_list: []\ngeneral_settings:\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("null general_settings should parse");
    assert!(cfg.general_settings.master_key.is_none());
    assert!(cfg.general_settings.database_url.is_none());
}

#[test]
fn explicit_null_router_settings_parses_as_default() {
    let yaml = "model_list: []\nrouter_settings:\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("null router_settings should parse");
    assert!(cfg.router_settings.routing.routing_strategy.is_none());
    assert!(cfg.router_settings.routing.num_retries.is_none());
}

#[test]
fn router_settings_optional_pre_call_checks_parse() {
    let yaml = "\
model_list: []
router_settings:
  optional_pre_call_checks: [prompt_caching, budget_limiter, health_check]
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("pre-call checks should parse");
    assert_eq!(
        cfg.router_settings.optional_pre_call_checks,
        Some(vec![
            "prompt_caching".to_string(),
            "budget_limiter".to_string(),
            "health_check".to_string()
        ])
    );
}

#[test]
fn router_settings_complexity_router_config_parses() {
    let yaml = "\
model_list: []
router_settings:
  routing_strategy: complexity_router
  complexity_router_config:
    tiers:
      simple: [gpt-4o-mini]
      complex: [gpt-4o]
      reasoning: [o3]
    default_model: gpt-4o-mini
    tier_boundaries:
      simple: 0.2
      complex: 0.7
    escalation_keywords: [prove, theorem]
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("complexity router config should parse");
    assert_eq!(
        cfg.router_settings.routing.routing_strategy.as_deref(),
        Some("complexity_router")
    );
    let crc = cfg
        .router_settings
        .complexity_router_config
        .expect("complexity_router_config present");
    assert_eq!(
        crc.tiers.get("simple"),
        Some(&vec!["gpt-4o-mini".to_string()])
    );
    assert_eq!(crc.tiers.get("reasoning"), Some(&vec!["o3".to_string()]));
    assert_eq!(crc.default_model.as_deref(), Some("gpt-4o-mini"));
    let boundaries = crc.tier_boundaries.expect("boundaries present");
    assert_eq!(boundaries.get("simple"), Some(&0.2));
    assert_eq!(boundaries.get("complex"), Some(&0.7));
    assert_eq!(
        crc.escalation_keywords,
        Some(vec!["prove".to_string(), "theorem".to_string()])
    );
}

#[test]
fn router_settings_adaptive_router_config_parses() {
    let yaml = "\
model_list: []
router_settings:
  routing_strategy: adaptive_router
  adaptive_router_config:
    quality_weight: 1.0
    cost_weight: 0.25
    epsilon: 0.05
    cold_start_min_observations: 10
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("adaptive router config should parse");
    assert_eq!(
        cfg.router_settings.routing.routing_strategy.as_deref(),
        Some("adaptive_router")
    );
    let arc = cfg
        .router_settings
        .adaptive_router_config
        .expect("adaptive_router_config present");
    assert_eq!(arc.quality_weight, 1.0);
    assert_eq!(arc.cost_weight, 0.25);
    assert_eq!(arc.epsilon, 0.05);
    assert_eq!(arc.cold_start_min_observations, 10);
}

#[test]
fn router_settings_adaptive_router_config_defaults_when_partial() {
    // A partial block fills the rest from the documented defaults.
    let yaml = "\
model_list: []
router_settings:
  routing_strategy: adaptive-router
  adaptive_router_config:
    epsilon: 0.2
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("partial adaptive config should parse");
    assert_eq!(
        cfg.router_settings.routing.routing_strategy.as_deref(),
        Some("adaptive-router")
    );
    let arc = cfg
        .router_settings
        .adaptive_router_config
        .expect("adaptive_router_config present");
    assert_eq!(arc.epsilon, 0.2);
    assert_eq!(arc.quality_weight, 1.0);
    assert_eq!(arc.cost_weight, 0.0);
    assert_eq!(arc.cold_start_min_observations, 20);
}

#[test]
fn settings_adaptive_strategy_parses() {
    // Routing knobs (incl. the strategy name) are accepted under
    // `oxllm_settings` too, for config compatibility.
    let yaml = "\
model_list: []
oxllm_settings:
  routing_strategy: adaptive_router
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("oxllm_settings strategy should parse");
    assert_eq!(
        cfg.oxllm_settings.routing.routing_strategy.as_deref(),
        Some("adaptive_router")
    );
    assert!(cfg.router_settings.adaptive_router_config.is_none());
}

// Bug #49 — `oxllm_params.timeout` accepts fractional seconds (e.g.
// `timeout: 0.5`), which existing configs use legitimately. Stored as
// `f64` rather than `u64`.
#[test]
fn params_timeout_accepts_fractional_seconds() {
    let yaml = "model_list:\n  - model_name: m\n    oxllm_params:\n      model: openai/gpt-4o\n      timeout: 0.5\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("fractional timeout parses");
    assert_eq!(cfg.model_list[0].oxllm_params.timeout, Some(0.5));
}

#[test]
fn semantic_cache_config_parses() {
    let yaml = "\
model_list: []
oxllm_settings:
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      similarity_threshold: 0.92
      max_entries: 5000
      ttl_secs: 7200
      api_base: https://api.openai.com/v1
      api_key: sk-embed
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("semantic cache yaml parses");
    let sc = cfg
        .oxllm_settings
        .cache
        .as_ref()
        .and_then(|c| c.semantic_cache.as_ref())
        .expect("semantic_cache present");
    assert!(sc.enabled);
    assert_eq!(sc.embedding_model, "text-embedding-3-small");
    assert_eq!(sc.similarity_threshold, Some(0.92));
    assert_eq!(sc.max_entries, Some(5000));
    assert_eq!(sc.ttl_secs, Some(7200));
    assert_eq!(sc.api_base.as_deref(), Some("https://api.openai.com/v1"));
    assert_eq!(sc.api_key.as_deref(), Some("sk-embed"));
}

#[test]
fn cache_config_without_semantic_cache_still_parses() {
    let yaml = "model_list: []\noxllm_settings:\n  cache:\n    type: memory\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("cache without semantic parses");
    let cache = cfg.oxllm_settings.cache.as_ref().expect("cache present");
    assert!(cache.semantic_cache.is_none());
}

// ---- M8: failure_callback / callbacks --------------------------------

#[test]
fn failure_and_callbacks_lists_parse() {
    let yaml = "\
model_list: []
oxllm_settings:
  success_callback:
    - generic
  failure_callback:
    - langfuse
    - datadog
  callbacks:
    - helicone
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("callback lists should parse");
    assert_eq!(
        cfg.oxllm_settings.success_callback.as_deref(),
        Some(&["generic".to_string()][..])
    );
    assert_eq!(
        cfg.oxllm_settings.failure_callback.as_deref(),
        Some(&["langfuse".to_string(), "datadog".to_string()][..])
    );
    assert_eq!(
        cfg.oxllm_settings.callbacks.as_deref(),
        Some(&["helicone".to_string()][..])
    );
}

#[test]
fn callback_lists_default_to_none() {
    let yaml = "model_list: []\n";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("empty config parses");
    assert!(cfg.oxllm_settings.success_callback.is_none());
    assert!(cfg.oxllm_settings.failure_callback.is_none());
    assert!(cfg.oxllm_settings.callbacks.is_none());
}

#[test]
fn callback_lists_roundtrip_through_serialize() {
    // Serializing and re-parsing must preserve the three lists so the
    // /config/update write-back (YAML round-trip) keeps them intact.
    let original = OxllmSettings {
        success_callback: Some(vec!["generic".to_string()]),
        failure_callback: Some(vec!["langfuse".to_string()]),
        callbacks: Some(vec!["helicone".to_string(), "datadog".to_string()]),
        ..Default::default()
    };
    let s = serde_yml::to_string(&original).expect("serialize");
    let back: OxllmSettings = serde_yml::from_str(&s).expect("deserialize");
    assert_eq!(back.success_callback, original.success_callback);
    assert_eq!(back.failure_callback, original.failure_callback);
    assert_eq!(back.callbacks, original.callbacks);
}

#[test]
fn model_max_budget_parses() {
    let yaml = "\
model_list: []
oxllm_settings:
  model_max_budget:
    gpt-4o: 100.0
    claude-3: 50.5
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("model_max_budget yaml should parse");
    let mmb = cfg
        .oxllm_settings
        .model_max_budget
        .as_ref()
        .expect("model_max_budget present");
    assert_eq!(mmb.len(), 2);
    assert_eq!(mmb.get("gpt-4o").copied(), Some(100.0));
    assert_eq!(mmb.get("claude-3").copied(), Some(50.5));
}

#[test]
fn prompt_caching_min_tokens_parses() {
    let yaml = "\
model_list:
  - model_name: claude
    oxllm_params:
      model: anthropic/claude-3-5-sonnet
      prompt_caching_min_tokens: 2048
oxllm_settings:
  prompt_caching_min_tokens: 1024
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("prompt_caching_min_tokens yaml should parse");
    assert_eq!(cfg.oxllm_settings.prompt_caching_min_tokens, Some(1024));
    assert_eq!(
        cfg.model_list[0].oxllm_params.prompt_caching_min_tokens,
        Some(2048)
    );
}

#[test]
fn allowed_fails_policy_parses() {
    let yaml = "\
model_list: []
router_settings:
  allowed_fails: 2
  allowed_fails_policy:
    \"429\": 3
    \"401\": 1
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("allowed_fails yaml should parse");
    assert_eq!(cfg.router_settings.routing.allowed_fails, Some(2));
    let policy = cfg
        .router_settings
        .routing
        .allowed_fails_policy
        .as_ref()
        .expect("allowed_fails_policy present");
    assert_eq!(policy.get("429").copied(), Some(3));
    assert_eq!(policy.get("401").copied(), Some(1));
}

#[test]
fn allowed_fails_accepted_in_settings() {
    // Existing configs carry allowed_fails under oxllm_settings;
    // they must not be silently dropped.
    let yaml = "\
model_list: []
oxllm_settings:
  allowed_fails: 3
  allowed_fails_policy:
    \"500\": 2
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("allowed_fails in oxllm_settings should parse");
    assert_eq!(cfg.oxllm_settings.routing.allowed_fails, Some(3));
    let policy = cfg
        .oxllm_settings
        .routing
        .allowed_fails_policy
        .as_ref()
        .expect("allowed_fails_policy present");
    assert_eq!(policy.get("500").copied(), Some(2));
}

#[test]
fn prompt_caching_enabled_parses() {
    let yaml = "\
model_list: []
oxllm_settings:
  prompt_caching_enabled: true
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("prompt_caching_enabled yaml should parse");
    assert_eq!(cfg.oxllm_settings.prompt_caching_enabled, Some(true));
}

#[test]
fn provider_budget_config_parses() {
    let yaml = "\
model_list: []
router_settings:
  provider_budget_config:
    openai:
      budget_limit: 100.0
      time_period: 1d
    anthropic:
      budget_limit: 50.5
      time_period: 1h
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("provider_budget_config yaml should parse");
    let budgets = cfg
        .router_settings
        .provider_budget_config
        .as_ref()
        .expect("provider_budget_config present");
    assert_eq!(budgets.len(), 2);
    let openai = budgets.get("openai").expect("openai budget");
    assert_eq!(openai.budget_limit, 100.0);
    assert_eq!(openai.time_period, "1d");
    let anthropic = budgets.get("anthropic").expect("anthropic budget");
    assert_eq!(anthropic.budget_limit, 50.5);
    assert_eq!(anthropic.time_period, "1h");
}

#[test]
fn health_check_routing_config_parses() {
    let yaml = "\
model_list: []
router_settings:
  enable_health_check_routing: true
  health_check_staleness_threshold_secs: 30
  health_check_failure_threshold: 5
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("health-check routing yaml should parse");
    assert_eq!(cfg.router_settings.enable_health_check_routing, Some(true));
    assert_eq!(
        cfg.router_settings.health_check_staleness_threshold_secs,
        Some(30)
    );
    assert_eq!(cfg.router_settings.health_check_failure_threshold, Some(5));
}

#[test]
fn router_settings_default_fallbacks_parses() {
    let yaml = "\
model_list: []
router_settings:
  default_fallbacks: [model-a, model-b]
";
    let cfg: ProxyConfig = serde_yml::from_str(yaml).expect("default_fallbacks should parse");
    assert_eq!(
        cfg.router_settings.routing.default_fallbacks,
        Some(serde_json::json!(["model-a", "model-b"]))
    );
}

#[test]
fn settings_default_fallbacks_dict_form_parses() {
    let yaml = "\
model_list: []
oxllm_settings:
  default_fallbacks:
    - original-model: [model-a]
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("dict-form default_fallbacks should parse");
    assert_eq!(
        cfg.oxllm_settings.routing.default_fallbacks,
        Some(serde_json::json!([{"original-model": ["model-a"]}]))
    );
}

#[test]
fn health_check_routing_config_defaults_to_none() {
    let cfg: ProxyConfig =
        serde_yml::from_str("model_list: []\n").expect("empty config should parse");
    assert!(cfg.router_settings.enable_health_check_routing.is_none());
    assert!(
        cfg.router_settings
            .health_check_staleness_threshold_secs
            .is_none()
    );
    assert!(cfg.router_settings.health_check_failure_threshold.is_none());
    assert!(cfg.router_settings.provider_budget_config.is_none());
}

#[test]
fn deployment_max_budget_parses_in_params() {
    let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
      max_budget: 10.0
      budget_duration: 1d
";
    let cfg: ProxyConfig =
        serde_yml::from_str(yaml).expect("deployment max_budget yaml should parse");
    assert_eq!(cfg.model_list[0].oxllm_params.max_budget, Some(10.0));
    assert_eq!(
        cfg.model_list[0].oxllm_params.budget_duration.as_deref(),
        Some("1d")
    );
}
