//! `RouterConfig` assembly from [`ProxyConfig`].
//!
//! Routing knobs are accepted in either `oxllm_settings` or
//! `router_settings` (the latter is the canonical location in newer
//! configs). [`build_router_config`] consults both and prefers
//! `router_settings` when a field is set in both places.

use oxllm_core::router::{
    ComplexityRouterConfig, FallbackRule, RouterConfig, RouterGroupOverride, RoutingStrategyKind,
};

use crate::config::ProxyConfig;
use crate::config::types::RoutingKnobs;

pub(crate) fn build_router_config(config: &ProxyConfig) -> anyhow::Result<RouterConfig> {
    let settings = &config.oxllm_settings;
    let router = &config.router_settings;

    if let Some(complexity) = &router.complexity_router_config {
        validate_tier_boundaries(complexity)?;
    }
    if let Some(adaptive) = &router.adaptive_router_config {
        adaptive.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    if let Some(scores) = &settings.quality_scores {
        validate_quality_scores(scores)?;
    }

    // Routing knobs are accepted in either `oxllm_settings` or
    // `router_settings` (the latter is the canonical location in newer
    // configs). Prefer `router_settings` field-by-field so both shapes work.
    let knobs = RoutingKnobs::merge(&settings.routing, &router.routing);

    if let Some(threshold) = knobs.failure_threshold_percent {
        if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
            return Err(anyhow::anyhow!(
                "failure_threshold_percent must be a number in [0, 1], got {threshold}"
            ));
        }
    }
    if let Some(min_requests) = knobs.failure_threshold_min_requests {
        if min_requests == 0 {
            return Err(anyhow::anyhow!(
                "failure_threshold_min_requests must be at least 1 \
                 (a 0 threshold would trip percentage cooldown on the first failure)"
            ));
        }
    }
    if let Some(pct) = knobs.budget_throttle_pct {
        if !(pct.is_finite() && (0.0..=1.0).contains(&pct)) {
            return Err(anyhow::anyhow!(
                "budget_throttle_pct must be a number in [0, 1], got {pct}"
            ));
        }
    }

    let routing_strategy = parse_strategy_kind(knobs.routing_strategy.as_deref().unwrap_or(
        if crate::pre_call_checks::session_affinity_requested(config) {
            "session_affinity"
        } else {
            "simple_shuffle"
        },
    ));

    let retry_policy = filter_status_policy(knobs.retry_policy.as_ref(), "retry_policy");

    let routing_groups = config
        .router_settings
        .routing_groups
        .as_ref()
        .map(|groups| {
            groups
                .iter()
                .map(|(model, g)| {
                    // Group knobs are passed through as `Option` overrides:
                    // per-model precedence over the merged globals is
                    // resolved at request time inside `oxllm-core`'s router.
                    let g = &g.routing;
                    let override_kind = g.routing_strategy.as_deref().map(parse_strategy_kind);
                    let group_fallbacks = parse_fallback_rules(g.fallbacks.as_ref());
                    let group_ctx_fallbacks =
                        parse_fallback_rules(g.context_window_fallbacks.as_ref());
                    let group_cp_fallbacks =
                        parse_fallback_rules(g.content_policy_fallbacks.as_ref());
                    let group_retry_policy = filter_status_policy(
                        g.retry_policy.as_ref(),
                        "routing_groups.retry_policy",
                    );
                    (
                        model.clone(),
                        RouterGroupOverride {
                            routing_strategy: override_kind,
                            num_retries: g.num_retries,
                            cooldown_time_secs: g.cooldown_time,
                            retry_policy: group_retry_policy,
                            fallbacks: group_fallbacks,
                            context_window_fallbacks: group_ctx_fallbacks,
                            content_policy_fallbacks: group_cp_fallbacks,
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(RouterConfig {
        routing_strategy,
        num_retries: knobs.num_retries.unwrap_or(2),
        max_fallbacks: knobs.max_fallbacks.unwrap_or(5),
        fallbacks: parse_fallback_rules(knobs.fallbacks.as_ref()),
        context_window_fallbacks: parse_fallback_rules(knobs.context_window_fallbacks.as_ref()),
        content_policy_fallbacks: parse_fallback_rules(knobs.content_policy_fallbacks.as_ref()),
        default_fallbacks: parse_default_fallbacks(knobs.default_fallbacks.as_ref()),
        cooldown_time_secs: knobs.cooldown_time.unwrap_or(60),
        quality_scores: settings.quality_scores.clone().unwrap_or_default(),
        retry_policy,
        model_group_alias: config
            .router_settings
            .model_group_alias
            .clone()
            .unwrap_or_default(),
        routing_groups,
        allowed_fails: knobs.allowed_fails,
        default_tag: knobs.default_tag.clone(),
        enable_weighted_failover: knobs.enable_weighted_failover.unwrap_or(false),
        exception_retry_policy: knobs.exception_retry_policy.clone(),
        failure_threshold_percent: knobs.failure_threshold_percent,
        failure_threshold_min_requests: knobs.failure_threshold_min_requests,
        budget_throttle_pct: knobs.budget_throttle_pct,
        allowed_fails_policy: filter_status_policy_opt(
            knobs.allowed_fails_policy.as_ref(),
            "allowed_fails_policy",
        ),
        provider_budget_config: router.provider_budget_config.clone(),
        enable_health_check_routing: router.enable_health_check_routing.unwrap_or(false),
        health_check_staleness_threshold_secs: router
            .health_check_staleness_threshold_secs
            .unwrap_or(60),
        health_check_failure_threshold: router.health_check_failure_threshold.unwrap_or(3),
        optional_pre_call_checks: crate::pre_call_checks::configured_filter_names(config),
        complexity_router_config: router.complexity_router_config.clone(),
        adaptive_router_config: router.adaptive_router_config.clone(),
    })
}

/// Validate `complexity_router_config.tier_boundaries`. Unknown tier keys
/// are warned about here (the classifier ignores them); everything else —
/// finite values in `[0, 1]` and strictly increasing effective ceilings
/// `simple < medium < complex` (unset keys fall back to the classifier
/// defaults) — is enforced by [`ComplexityRouterConfig::validate`].
/// Returns an error for invalid boundaries.
fn validate_tier_boundaries(config: &ComplexityRouterConfig) -> anyhow::Result<()> {
    if let Some(boundaries) = &config.tier_boundaries {
        const KNOWN_TIERS: [&str; 3] = ["simple", "medium", "complex"];
        for tier in boundaries.keys() {
            if !KNOWN_TIERS.contains(&tier.as_str()) {
                tracing::warn!(
                    tier = %tier,
                    "unknown complexity_router_config tier_boundaries key; it is ignored"
                );
            }
        }
    }
    config.validate().map_err(|e| anyhow::anyhow!("{e}"))
}

/// Validate `quality_scores`: every value must be finite and non-negative.
/// A non-finite score (NaN, ±∞) would poison the argmax comparison in
/// `QualityRouter::pick`; a negative score is almost always a config typo
/// (the strategy silently reads it as `0.0` and treats the deployment as
/// unscored). Reject both at load so a typo fails loudly instead of
/// silently degrading to "no quality routing" for the affected model.
fn validate_quality_scores(scores: &std::collections::HashMap<String, f64>) -> anyhow::Result<()> {
    for (model, score) in scores {
        if !score.is_finite() {
            return Err(anyhow::anyhow!(
                "quality_scores[{model}] must be a finite number, got {score}"
            ));
        }
        if *score < 0.0 {
            return Err(anyhow::anyhow!(
                "quality_scores[{model}] must be non-negative, got {score}"
            ));
        }
    }
    Ok(())
}

/// Parse a routing-strategy name from config into a [`RoutingStrategyKind`].
/// Accepts both `snake_case` and `kebab-case`. Unknown names log a warning
/// and fall back to [`RoutingStrategyKind::SimpleShuffle`]. Callers should
/// pass `None` for an unset strategy and propagate the resulting default.
fn parse_strategy_kind(name: &str) -> RoutingStrategyKind {
    match name {
        "simple_shuffle" | "simple-shuffle" => RoutingStrategyKind::SimpleShuffle,
        "lowest_cost" | "lowest-cost" | "cost-based-routing" | "cost_based_routing" => {
            RoutingStrategyKind::LowestCost
        }
        "lowest_latency" | "lowest-latency" => RoutingStrategyKind::LowestLatency,
        "usage_based"
        | "usage-based"
        | "usage-based-routing"
        | "usage_based_routing"
        | "usage-based-routing-v2"
        | "usage_based_routing_v2" => RoutingStrategyKind::UsageBased,
        "latency_based" | "latency-based" | "latency-based-routing" | "latency_based_routing" => {
            RoutingStrategyKind::LatencyBased
        }
        "quality_router" | "quality-router" => RoutingStrategyKind::QualityRouter,
        "tag_based" | "tag-based" => RoutingStrategyKind::TagBased,
        "least_busy" | "least-busy" => RoutingStrategyKind::LeastBusy,
        "session_affinity" | "session-affinity" => RoutingStrategyKind::SessionAffinity,
        "complexity_router" | "complexity-router" => RoutingStrategyKind::ComplexityRouter,
        "adaptive_router" | "adaptive-router" => RoutingStrategyKind::AdaptiveRouter,
        "round_robin" | "round-robin" => RoutingStrategyKind::RoundRobin,
        other => {
            tracing::warn!(
                strategy = %other,
                "unknown routing strategy, defaulting to simple_shuffle"
            );
            RoutingStrategyKind::SimpleShuffle
        }
    }
}

/// Filter a per-HTTP-status policy map (`retry_policy` /
/// `allowed_fails_policy`) to entries whose key parses as a valid HTTP
/// status code (`u16` in `100..=599`). Invalid keys (e.g. `"4290"`,
/// `"50a"`, a stray typo) would never match a real upstream status, so the
/// operator's intended policy for that status is silently ineffective;
/// surface the typo via a `warn!` and drop the entry rather than failing
/// startup — consistent with how unknown routing-strategy names and
/// malformed fallback entries are handled.
///
/// Returns an empty map when `policy` is `None` (so callers that store the
/// map directly, like `RouterConfig::retry_policy`, get an empty default).
/// Callers that store `Option<HashMap>` (like `allowed_fails_policy`)
/// should use [`filter_status_policy_opt`] to preserve the
/// `None`-when-unset semantics that downstream code distinguishes from an
/// explicit empty map.
fn filter_status_policy(
    policy: Option<&std::collections::HashMap<String, u32>>,
    field: &str,
) -> std::collections::HashMap<String, u32> {
    let Some(map) = policy else {
        return std::collections::HashMap::new();
    };
    filter_status_policy_inner(map, field)
}

/// Same as [`filter_status_policy`] but preserves `None`: returns `None`
/// only when the input is `None`, so an unset policy stays distinct from an
/// explicit empty one.
fn filter_status_policy_opt(
    policy: Option<&std::collections::HashMap<String, u32>>,
    field: &str,
) -> Option<std::collections::HashMap<String, u32>> {
    let map = policy?;
    Some(filter_status_policy_inner(map, field))
}

fn filter_status_policy_inner(
    map: &std::collections::HashMap<String, u32>,
    field: &str,
) -> std::collections::HashMap<String, u32> {
    let mut out = std::collections::HashMap::with_capacity(map.len());
    for (key, val) in map {
        match key.parse::<u16>() {
            Ok(code) if (100..=599).contains(&code) => {
                out.insert(key.clone(), *val);
            }
            _ => {
                tracing::warn!(
                    field,
                    key = %key,
                    "{field} key is not a valid HTTP status code (100-599); \
                     entry is ignored and will never match any upstream status"
                );
            }
        }
    }
    out
}

fn parse_fallback_rules(value: Option<&serde_json::Value>) -> Vec<FallbackRule> {
    let Some(serde_json::Value::Array(arr)) = value else {
        return Vec::new();
    };
    arr.iter()
        .flat_map(|item| {
            let serde_json::Value::Object(map) = item else {
                tracing::warn!(
                    value = ?item,
                    "fallbacks entry is not an object (expected {{\"model\": [\"fb\"]}}); ignoring"
                );
                return Vec::new();
            };
            map.iter()
                .map(|(primary, fallbacks)| {
                    // Fallbacks allow either an array of fallback names or a
                    // bare string shorthand: `{"gpt-4o": "claude"}` is treated
                    // as `{"gpt-4o": ["claude"]}`. `null` means "no fallbacks"
                    // for this primary and is honored silently.
                    let fb_list: Vec<String> = match fallbacks {
                        serde_json::Value::Array(a) => a
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect(),
                        serde_json::Value::String(s) => vec![s.clone()],
                        serde_json::Value::Null => Vec::new(),
                        _ => {
                            tracing::warn!(
                                primary = %primary,
                                value = ?fallbacks,
                                "fallbacks value for model is neither a string nor an array \
                                 of strings; produced a no-op rule with zero fallbacks"
                            );
                            Vec::new()
                        }
                    };
                    FallbackRule {
                        primary: primary.clone(),
                        fallbacks: fb_list,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Parse `default_fallbacks`. Two shapes are accepted:
///
/// - the list form — `default_fallbacks: ["model-a", "model-b"]`: one
///   chain applied to ANY unknown model group, stored as a single rule
///   with the wildcard primary `"*"`;
/// - the dict form — `default_fallbacks: [{"original-model": ["a"]}]`:
///   per-model chains, same shape as `fallbacks` (parsed by
///   [`parse_fallback_rules`]).
fn parse_default_fallbacks(value: Option<&serde_json::Value>) -> Vec<FallbackRule> {
    let Some(serde_json::Value::Array(arr)) = value else {
        return Vec::new();
    };
    if arr.iter().all(|v| v.is_string()) {
        let chain: Vec<String> = arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        return if chain.is_empty() {
            Vec::new()
        } else {
            vec![FallbackRule {
                primary: "*".to_string(),
                fallbacks: chain,
            }]
        };
    }
    parse_fallback_rules(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{OxllmSettings, RouterSettings};
    use serde_json::json;

    /// Test fixtures are valid configs; unwrap the (now fallible)
    /// builder so existing call sites stay unchanged. Shadows the
    /// glob-imported `super::build_router_config` inside this module.
    fn build_router_config(config: &ProxyConfig) -> RouterConfig {
        super::build_router_config(config).expect("test config must be valid")
    }

    fn proxy_with_settings(settings: OxllmSettings) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings,
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        }
    }

    /// `OxllmSettings` carrying only the given routing knobs.
    fn settings_with(routing: RoutingKnobs) -> OxllmSettings {
        OxllmSettings {
            routing,
            ..Default::default()
        }
    }

    #[test]
    fn build_router_config_cooldown_time_default() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert_eq!(cfg.cooldown_time_secs, 60);
    }

    #[test]
    fn build_router_config_cooldown_time_configured() {
        let settings = settings_with(RoutingKnobs {
            cooldown_time: Some(120),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.cooldown_time_secs, 120);
    }

    #[test]
    fn parse_fallback_rules_from_yaml_shape() {
        let val = json!([
        {"gpt-4o": ["gpt-4o-mini", "claude"]}
        ]);
        let rules = parse_fallback_rules(Some(&val));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].primary, "gpt-4o");
        assert_eq!(rules[0].fallbacks, vec!["gpt-4o-mini", "claude"]);
    }

    #[test]
    fn parse_fallback_rules_empty() {
        assert!(parse_fallback_rules(None).is_empty());
        assert!(parse_fallback_rules(Some(&json!(null))).is_empty());
        assert!(parse_fallback_rules(Some(&json!("string"))).is_empty());
    }

    #[test]
    fn parse_fallback_rules_string_shorthand() {
        // A bare string is allowed: `{"gpt-4o": "claude"}` is equivalent
        // to `{"gpt-4o": ["claude"]}`.
        let val = json!([{"gpt-4o": "claude"}]);
        let rules = parse_fallback_rules(Some(&val));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].primary, "gpt-4o");
        assert_eq!(rules[0].fallbacks, vec!["claude"]);
    }

    #[test]
    fn parse_fallback_rules_null_value_yields_empty_silently() {
        // `{"gpt-4o": null}` is an intentional "no fallbacks" marker: it
        // produces a rule with zero fallbacks without a warning.
        let val = json!([{"gpt-4o": null}]);
        let rules = parse_fallback_rules(Some(&val));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].primary, "gpt-4o");
        assert!(rules[0].fallbacks.is_empty());
    }

    #[test]
    fn parse_fallback_rules_malformed_value_yields_noop_rule() {
        // A value that is neither string, array, nor null (e.g. a number or
        // bool) is a likely config typo: the rule is kept (so the primary is
        // still discoverable) but carries zero fallbacks and a warning is
        // emitted so the operator can spot the mistake instead of silently
        // getting a no-op.
        let val = json!([{"gpt-4o": 123}]);
        let rules = parse_fallback_rules(Some(&val));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].primary, "gpt-4o");
        assert!(rules[0].fallbacks.is_empty());
    }

    #[test]
    fn parse_fallback_rules_non_object_array_element_dropped() {
        // A bare string/number as an array element (instead of an object) is
        // dropped and a warning is emitted; it must not panic or produce a
        // malformed rule.
        let val = json!(["bare-string", 42]);
        let rules = parse_fallback_rules(Some(&val));
        assert!(rules.is_empty());
    }

    #[test]
    fn parse_fallback_rules_mixed_string_and_array() {
        let val = json!([
        {"gpt-4o": "claude"},
        {"claude": ["llama", "mistake"]}
        ]);
        let rules = parse_fallback_rules(Some(&val));
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].primary, "gpt-4o");
        assert_eq!(rules[0].fallbacks, vec!["claude"]);
        assert_eq!(rules[1].primary, "claude");
        assert_eq!(rules[1].fallbacks, vec!["llama", "mistake"]);
    }

    #[test]
    fn build_router_config_defaults() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert_eq!(cfg.num_retries, 2);
        assert_eq!(cfg.cooldown_time_secs, 60);
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::SimpleShuffle);
        assert!(cfg.fallbacks.is_empty());
        assert!(cfg.context_window_fallbacks.is_empty());
        assert!(cfg.content_policy_fallbacks.is_empty());
        assert!(cfg.retry_policy.is_empty());
        assert!(cfg.model_group_alias.is_empty());
    }

    #[test]
    fn build_router_config_with_settings() {
        let settings = settings_with(RoutingKnobs {
            num_retries: Some(5),
            routing_strategy: Some("simple_shuffle".to_string()),
            fallbacks: Some(json!([{"gpt-4o": ["claude"]}])),
            ..Default::default()
        });

        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.num_retries, 5);
        assert_eq!(cfg.fallbacks.len(), 1);
        assert_eq!(cfg.fallbacks[0].primary, "gpt-4o");
    }

    #[test]
    fn build_router_config_unknown_strategy_warns_and_defaults() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("unknown-strategy".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::SimpleShuffle);
    }

    #[test]
    fn build_router_config_lowest_cost() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("lowest_cost".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::LowestCost);
    }

    #[test]
    fn build_router_config_lowest_latency() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("lowest_latency".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::LowestLatency);
    }

    #[test]
    fn build_router_config_usage_based() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("usage_based".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::UsageBased);
    }

    #[test]
    fn build_router_config_latency_based() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("latency_based".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::LatencyBased);
    }

    #[test]
    fn build_router_config_quality_router() {
        let mut scores = std::collections::HashMap::new();
        scores.insert("openai/gpt-4o".to_string(), 0.9);
        let settings = OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("quality_router".to_string()),
                ..Default::default()
            },
            quality_scores: Some(scores.clone()),
            ..Default::default()
        };
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::QualityRouter);
        assert_eq!(cfg.quality_scores, scores);
    }

    #[test]
    fn build_router_config_rejects_negative_quality_score() {
        // A negative score is almost always a config typo; the strategy
        // silently reads it as `0.0` and treats the deployment as unscored,
        // so reject at load to surface the typo.
        let mut scores = std::collections::HashMap::new();
        scores.insert("openai/gpt-4o".to_string(), -0.1);
        let settings = OxllmSettings {
            quality_scores: Some(scores),
            ..Default::default()
        };
        let err = super::build_router_config(&proxy_with_settings(settings))
            .expect_err("negative score must be rejected");
        assert!(
            err.to_string().contains("non-negative"),
            "error must explain the constraint: {err}"
        );
    }

    #[test]
    fn build_router_config_rejects_non_finite_quality_score() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut scores = std::collections::HashMap::new();
            scores.insert("openai/gpt-4o".to_string(), bad);
            let settings = OxllmSettings {
                quality_scores: Some(scores),
                ..Default::default()
            };
            let err = super::build_router_config(&proxy_with_settings(settings))
                .expect_err("non-finite score must be rejected");
            assert!(
                err.to_string().contains("finite"),
                "error must explain the constraint for {bad}: {err}"
            );
        }
    }

    #[test]
    fn build_router_config_accepts_zero_quality_score() {
        // `0.0` is a valid configured pin (distinct from "no score
        // configured") and must pass validation.
        let mut scores = std::collections::HashMap::new();
        scores.insert("openai/gpt-4o".to_string(), 0.0);
        let settings = OxllmSettings {
            quality_scores: Some(scores.clone()),
            ..Default::default()
        };
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.quality_scores, scores);
    }

    #[test]
    fn build_router_config_tag_based() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("tag_based".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::TagBased);
    }

    #[test]
    fn build_router_config_least_busy() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("least_busy".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::LeastBusy);
    }

    #[test]
    fn build_router_config_session_affinity() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("session_affinity".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::SessionAffinity);
    }

    #[test]
    fn build_router_config_content_policy_fallbacks_parsed() {
        let settings = settings_with(RoutingKnobs {
            content_policy_fallbacks: Some(json!([{"gpt-4o": ["claude", "llama-guarded"]}])),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.content_policy_fallbacks.len(), 1);
        assert_eq!(cfg.content_policy_fallbacks[0].primary, "gpt-4o");
        assert_eq!(
            cfg.content_policy_fallbacks[0].fallbacks,
            vec!["claude", "llama-guarded"]
        );
    }

    #[test]
    fn build_router_config_retry_policy_parsed() {
        let mut policy = std::collections::HashMap::new();
        policy.insert("429".to_string(), 5);
        policy.insert("500".to_string(), 3);
        let settings = settings_with(RoutingKnobs {
            retry_policy: Some(policy.clone()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.retry_policy, policy);
    }

    #[test]
    fn build_router_config_warns_on_invalid_retry_policy_status_key() {
        // Keys must be valid HTTP status codes (100-599). Typos like "4290"
        // (extra digit) or "abc" are dropped and warned about; valid keys
        // in the same map are still applied. An out-of-range-but-parseable
        // code (e.g. 99 or 600) is also dropped.
        let mut policy = std::collections::HashMap::new();
        policy.insert("429".to_string(), 5); // valid
        policy.insert("4290".to_string(), 9); // extra digit — invalid
        policy.insert("abc".to_string(), 1); // non-numeric — invalid
        policy.insert("99".to_string(), 2); // parseable but out of range
        policy.insert("600".to_string(), 7); // parseable but out of range
        let settings = settings_with(RoutingKnobs {
            retry_policy: Some(policy),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(
            cfg.retry_policy.len(),
            1,
            "only the valid status code key should survive; got {:?}",
            cfg.retry_policy
        );
        assert_eq!(cfg.retry_policy.get("429"), Some(&5));
    }

    #[test]
    fn build_router_config_warns_on_invalid_allowed_fails_policy_key() {
        // Same filtering applies to allowed_fails_policy.
        let mut policy = std::collections::HashMap::new();
        policy.insert("401".to_string(), 1); // valid
        policy.insert("50a".to_string(), 2); // non-numeric — invalid
        let settings = settings_with(RoutingKnobs {
            allowed_fails_policy: Some(policy),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        let filtered = cfg
            .allowed_fails_policy
            .as_ref()
            .expect("allowed_fails_policy should be Some after filtering");
        assert_eq!(filtered.len(), 1, "only valid keys should survive");
        assert_eq!(filtered.get("401"), Some(&1));
    }

    #[test]
    fn build_router_config_allowed_fails_defaults_to_none() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert_eq!(cfg.allowed_fails, None);
        assert_eq!(cfg.allowed_fails_policy, None);
    }

    #[test]
    fn build_router_config_allowed_fails_from_router_settings() {
        let policy =
            std::collections::HashMap::from([("429".to_string(), 3u32), ("401".to_string(), 1u32)]);
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    allowed_fails: Some(2),
                    allowed_fails_policy: Some(policy.clone()),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.allowed_fails, Some(2));
        assert_eq!(cfg.allowed_fails_policy, Some(policy));
    }

    #[test]
    fn build_router_config_allowed_fails_from_settings() {
        // Existing configs put allowed_fails under oxllm_settings;
        // they must not be silently dropped.
        let policy = std::collections::HashMap::from([("500".to_string(), 4u32)]);
        let settings = settings_with(RoutingKnobs {
            allowed_fails: Some(3),
            allowed_fails_policy: Some(policy.clone()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.allowed_fails, Some(3));
        assert_eq!(cfg.allowed_fails_policy, Some(policy));
    }

    #[test]
    fn build_router_config_allowed_fails_router_settings_wins() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings_with(RoutingKnobs {
                allowed_fails: Some(9),
                ..Default::default()
            }),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    allowed_fails: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(
            cfg.allowed_fails,
            Some(1),
            "router_settings takes precedence over oxllm_settings"
        );
    }

    #[test]
    fn build_router_config_model_group_alias_from_router_settings() {
        let mut aliases = std::collections::HashMap::new();
        aliases.insert(
            "gpt-4".to_string(),
            vec!["gpt-4o".to_string(), "gpt-4-turbo".to_string()],
        );
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                model_group_alias: Some(aliases.clone()),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.model_group_alias, aliases);
    }

    #[test]
    fn build_router_config_router_settings_takes_precedence() {
        // Both oxllm_settings and router_settings carry a value; the latter
        // wins (reference semantics).
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings_with(RoutingKnobs {
                routing_strategy: Some("simple_shuffle".to_string()),
                num_retries: Some(1),
                cooldown_time: Some(30),
                ..Default::default()
            }),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    routing_strategy: Some("lowest_cost".to_string()),
                    num_retries: Some(7),
                    cooldown_time: Some(180),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::LowestCost);
        assert_eq!(cfg.num_retries, 7);
        assert_eq!(cfg.cooldown_time_secs, 180);
    }

    #[test]
    fn build_router_config_falls_back_to_settings() {
        // router_settings empty → oxllm_settings is consulted.
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings_with(RoutingKnobs {
                routing_strategy: Some("usage_based".to_string()),
                num_retries: Some(5),
                cooldown_time: Some(90),
                ..Default::default()
            }),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::UsageBased);
        assert_eq!(cfg.num_retries, 5);
        assert_eq!(cfg.cooldown_time_secs, 90);
    }

    #[test]
    fn parse_strategy_kind_accepts_legacy_proxy_names() {
        assert_eq!(
            parse_strategy_kind("usage-based-routing-v2"),
            RoutingStrategyKind::UsageBased
        );
        assert_eq!(
            parse_strategy_kind("latency-based-routing"),
            RoutingStrategyKind::LatencyBased
        );
        assert_eq!(
            parse_strategy_kind("cost-based-routing"),
            RoutingStrategyKind::LowestCost
        );
    }

    #[test]
    fn routing_knobs_merge_prefers_group_field_by_field() {
        // Fields set in `group` win; fields left unset fall back to `global`.
        let global = RoutingKnobs {
            routing_strategy: Some("simple_shuffle".to_string()),
            num_retries: Some(3),
            cooldown_time: Some(60),
            fallbacks: Some(json!([{"a": ["b"]}])),
            ..Default::default()
        };
        let mut group_policy = std::collections::HashMap::new();
        group_policy.insert("429".to_string(), 9);
        let group = RoutingKnobs {
            num_retries: Some(7),
            retry_policy: Some(group_policy.clone()),
            ..Default::default()
        };
        let merged = RoutingKnobs::merge(&global, &group);
        assert_eq!(
            merged.routing_strategy.as_deref(),
            Some("simple_shuffle"),
            "unset in group → falls back to global"
        );
        assert_eq!(merged.num_retries, Some(7), "set in group → group wins");
        assert_eq!(merged.cooldown_time, Some(60));
        assert!(merged.fallbacks.is_some());
        assert_eq!(merged.retry_policy, Some(group_policy));
    }

    // -----------------------------------------------------------------------
    // routing_groups
    // -----------------------------------------------------------------------

    #[test]
    fn build_router_config_no_routing_groups_defaults_empty() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert!(
            cfg.routing_groups.is_empty(),
            "absence of routing_groups must yield an empty map"
        );
    }

    #[test]
    fn build_router_config_routing_group_strategy_parsed() {
        let mut groups = std::collections::HashMap::new();
        groups.insert(
            "gpt-4o".to_string(),
            crate::config::types::RoutingGroupConfig {
                routing: RoutingKnobs {
                    routing_strategy: Some("lowest_latency".to_string()),
                    num_retries: Some(5),
                    cooldown_time: Some(120),
                    ..Default::default()
                },
            },
        );
        groups.insert(
            "claude-3".to_string(),
            crate::config::types::RoutingGroupConfig {
                routing: RoutingKnobs {
                    routing_strategy: Some("kebab-lowest-cost".to_string()),
                    ..Default::default()
                },
            },
        );
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings_with(RoutingKnobs {
                routing_strategy: Some("simple_shuffle".to_string()),
                ..Default::default()
            }),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing_groups: Some(groups),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.routing_groups.len(), 2);
        let gpt = cfg.routing_groups.get("gpt-4o").expect("gpt-4o group");
        assert_eq!(
            gpt.routing_strategy,
            Some(RoutingStrategyKind::LowestLatency)
        );
        assert_eq!(gpt.num_retries, Some(5));
        assert_eq!(gpt.cooldown_time_secs, Some(120));
        let claude = cfg.routing_groups.get("claude-3").expect("claude-3 group");
        // Unknown kebab-strategy name should fall back to SimpleShuffle with a warn.
        assert_eq!(
            claude.routing_strategy,
            Some(RoutingStrategyKind::SimpleShuffle)
        );
    }

    #[test]
    fn build_router_config_routing_group_fallbacks_parsed() {
        let mut groups = std::collections::HashMap::new();
        groups.insert(
            "gpt-4o".to_string(),
            crate::config::types::RoutingGroupConfig {
                routing: RoutingKnobs {
                    fallbacks: Some(json!([{"gpt-4o": ["claude-3"]}])),
                    context_window_fallbacks: Some(json!([{"gpt-4o": ["gpt-4o-128k"]}])),
                    content_policy_fallbacks: Some(json!([{"gpt-4o": ["llama-guarded"]}])),
                    ..Default::default()
                },
            },
        );
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing_groups: Some(groups),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        let gpt = cfg.routing_groups.get("gpt-4o").expect("gpt-4o group");
        assert_eq!(gpt.fallbacks.len(), 1);
        assert_eq!(gpt.fallbacks[0].primary, "gpt-4o");
        assert_eq!(gpt.fallbacks[0].fallbacks, vec!["claude-3"]);
        assert_eq!(
            gpt.context_window_fallbacks[0].fallbacks,
            vec!["gpt-4o-128k"]
        );
        assert_eq!(
            gpt.content_policy_fallbacks[0].fallbacks,
            vec!["llama-guarded"]
        );
    }

    #[test]
    fn build_router_config_routing_group_retry_policy_parsed() {
        let mut policy = std::collections::HashMap::new();
        policy.insert("429".to_string(), 8);
        let mut groups = std::collections::HashMap::new();
        groups.insert(
            "gpt-4o".to_string(),
            crate::config::types::RoutingGroupConfig {
                routing: RoutingKnobs {
                    retry_policy: Some(policy.clone()),
                    ..Default::default()
                },
            },
        );
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing_groups: Some(groups),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        let gpt = cfg.routing_groups.get("gpt-4o").expect("gpt-4o group");
        assert_eq!(gpt.retry_policy, policy);
    }

    #[test]
    fn build_router_config_routing_group_unknown_strategy_defaults() {
        let mut groups = std::collections::HashMap::new();
        groups.insert(
            "gpt-4o".to_string(),
            crate::config::types::RoutingGroupConfig {
                routing: RoutingKnobs {
                    routing_strategy: Some("totally-bogus".to_string()),
                    ..Default::default()
                },
            },
        );
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing_groups: Some(groups),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        let gpt = cfg.routing_groups.get("gpt-4o").expect("gpt-4o group");
        assert_eq!(
            gpt.routing_strategy,
            Some(RoutingStrategyKind::SimpleShuffle),
            "unknown strategy name must fall back to SimpleShuffle"
        );
    }

    #[test]
    fn build_router_config_provider_budget_config_wired() {
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                provider_budget_config: Some(budgets.clone()),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.provider_budget_config, Some(budgets));
    }

    #[test]
    fn build_router_config_health_check_defaults() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert!(!cfg.enable_health_check_routing);
        assert_eq!(cfg.health_check_staleness_threshold_secs, 60);
        assert_eq!(cfg.health_check_failure_threshold, 3);
        assert!(cfg.provider_budget_config.is_none());
    }

    #[test]
    fn build_router_config_health_check_from_router_settings() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                enable_health_check_routing: Some(true),
                health_check_staleness_threshold_secs: Some(30),
                health_check_failure_threshold: Some(5),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert!(cfg.enable_health_check_routing);
        assert_eq!(cfg.health_check_staleness_threshold_secs, 30);
        assert_eq!(cfg.health_check_failure_threshold, 5);
    }

    #[test]
    fn build_router_config_complexity_router_strategy_parsed() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("complexity_router".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::ComplexityRouter);
    }

    #[test]
    fn build_router_config_complexity_router_strategy_kebab_parsed() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("complexity-router".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::ComplexityRouter);
    }

    #[test]
    fn build_router_config_optional_pre_call_checks_default_empty() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert!(cfg.optional_pre_call_checks.is_empty());
    }

    #[test]
    fn build_router_config_optional_pre_call_checks_wired() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings {
                prompt_caching_enabled: Some(true),
                ..Default::default()
            },
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                enable_health_check_routing: Some(true),
                provider_budget_config: Some(std::collections::HashMap::from([(
                    "openai".into(),
                    oxllm_core::router::ProviderBudgetConfig {
                        budget_limit: 100.0,
                        time_period: "1d".into(),
                    },
                )])),
                optional_pre_call_checks: Some(vec![
                    "prompt_caching".to_string(),
                    "budget_limiter".to_string(),
                    "health_check".to_string(),
                ]),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(
            cfg.optional_pre_call_checks,
            vec!["prompt_caching", "budget_limiter", "health_check"]
        );
    }

    #[test]
    fn session_pre_call_alias_selects_affinity_and_preserves_explicit_strategy_precedence() {
        let mut config = proxy_with_settings(OxllmSettings::default());
        config.router_settings.optional_pre_call_checks = Some(vec![
            "session-affinity".into(),
            "unknown_check".into(),
            "health_check".into(),
        ]);
        let router = build_router_config(&config);
        assert_eq!(
            router.routing_strategy,
            RoutingStrategyKind::SessionAffinity
        );
        assert!(router.optional_pre_call_checks.is_empty());
        config.router_settings.routing.routing_strategy = Some("round_robin".into());
        assert_eq!(
            build_router_config(&config).routing_strategy,
            RoutingStrategyKind::RoundRobin
        );
    }

    #[test]
    fn build_router_config_complexity_router_config_wired() {
        let complexity = oxllm_core::router::ComplexityRouterConfig {
            tiers: std::collections::HashMap::from([
                ("simple".to_string(), vec!["gpt-4o-mini".to_string()]),
                ("complex".to_string(), vec!["gpt-4o".to_string()]),
            ]),
            default_model: Some("gpt-4o-mini".to_string()),
            tier_boundaries: Some(std::collections::HashMap::from([(
                "simple".to_string(),
                0.2,
            )])),
            escalation_keywords: Some(vec!["prove".to_string()]),
        };
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                complexity_router_config: Some(complexity.clone()),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.complexity_router_config, Some(complexity));
    }

    #[test]
    fn build_router_config_complexity_router_config_defaults_none() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert!(cfg.complexity_router_config.is_none());
    }

    // -----------------------------------------------------------------------
    // adaptive_router
    // -----------------------------------------------------------------------

    #[test]
    fn build_router_config_adaptive_router_strategy_parsed() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("adaptive_router".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::AdaptiveRouter);
    }

    #[test]
    fn build_router_config_adaptive_router_strategy_kebab_parsed() {
        let settings = settings_with(RoutingKnobs {
            routing_strategy: Some("adaptive-router".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.routing_strategy, RoutingStrategyKind::AdaptiveRouter);
    }

    #[test]
    fn build_router_config_adaptive_router_config_wired() {
        let adaptive = oxllm_core::router::AdaptiveRouterConfig {
            quality_weight: 0.8,
            cost_weight: 0.3,
            epsilon: 0.05,
            cold_start_min_observations: 10,
        };
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                adaptive_router_config: Some(adaptive.clone()),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.adaptive_router_config, Some(adaptive));
    }

    #[test]
    fn build_router_config_adaptive_router_config_defaults_none() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert!(cfg.adaptive_router_config.is_none());
    }

    #[test]
    fn build_router_config_rejects_invalid_adaptive_epsilon() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                adaptive_router_config: Some(oxllm_core::router::AdaptiveRouterConfig {
                    epsilon: 1.5,
                    ..Default::default()
                }),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let err =
            super::build_router_config(&config).expect_err("out-of-range epsilon must be rejected");
        assert!(
            err.to_string().contains("epsilon"),
            "error must name the offending knob: {err}"
        );
    }

    #[test]
    fn build_router_config_rejects_non_finite_adaptive_weights() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                adaptive_router_config: Some(oxllm_core::router::AdaptiveRouterConfig {
                    quality_weight: f64::NAN,
                    ..Default::default()
                }),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        assert!(super::build_router_config(&config).is_err());
    }

    #[test]
    fn build_router_config_adaptive_routing_group_strategy_parsed() {
        let mut groups = std::collections::HashMap::new();
        groups.insert(
            "gpt-4o".to_string(),
            crate::config::types::RoutingGroupConfig {
                routing: RoutingKnobs {
                    routing_strategy: Some("adaptive_router".to_string()),
                    ..Default::default()
                },
            },
        );
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing_groups: Some(groups),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        let gpt = cfg.routing_groups.get("gpt-4o").expect("gpt-4o group");
        assert_eq!(
            gpt.routing_strategy,
            Some(RoutingStrategyKind::AdaptiveRouter)
        );
    }

    // -----------------------------------------------------------------------
    // complexity_router_config tier_boundaries validation
    // -----------------------------------------------------------------------

    fn proxy_with_tier_boundaries(boundaries: &[(&str, f64)]) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                complexity_router_config: Some(oxllm_core::router::ComplexityRouterConfig {
                    tier_boundaries: Some(
                        boundaries
                            .iter()
                            .map(|(k, v)| (k.to_string(), *v))
                            .collect(),
                    ),
                    ..Default::default()
                }),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn build_router_config_rejects_inverted_tier_boundaries() {
        let config = proxy_with_tier_boundaries(&[("simple", 0.9), ("medium", 0.2)]);
        let err =
            super::build_router_config(&config).expect_err("inverted boundaries must be rejected");
        assert!(
            err.to_string().contains("simple < medium < complex"),
            "error must explain the ordering requirement: {err}"
        );
    }

    #[test]
    fn build_router_config_rejects_out_of_range_tier_boundaries() {
        let config = proxy_with_tier_boundaries(&[("simple", 1.5)]);
        assert!(super::build_router_config(&config).is_err());
        let config = proxy_with_tier_boundaries(&[("medium", -0.1)]);
        assert!(super::build_router_config(&config).is_err());
    }

    #[test]
    fn build_router_config_rejects_nan_tier_boundary() {
        let config = proxy_with_tier_boundaries(&[("complex", f64::NAN)]);
        assert!(super::build_router_config(&config).is_err());
    }

    #[test]
    fn build_router_config_rejects_partial_boundaries_inverting_defaults() {
        // Only "simple" set, above the default medium ceiling (0.35) →
        // the medium tier would be unreachable → rejected.
        let config = proxy_with_tier_boundaries(&[("simple", 0.5)]);
        assert!(super::build_router_config(&config).is_err());
    }

    #[test]
    fn build_router_config_accepts_valid_partial_boundaries() {
        let config = proxy_with_tier_boundaries(&[("simple", 0.10)]);
        let cfg = super::build_router_config(&config).expect("valid boundaries");
        assert!(cfg.complexity_router_config.is_some());
    }

    #[test]
    fn build_router_config_warns_on_unknown_tier_key_but_accepts() {
        let config = proxy_with_tier_boundaries(&[("trivial", 0.1)]);
        let cfg = super::build_router_config(&config).expect("unknown keys only warn");
        assert!(cfg.complexity_router_config.is_some());
    }

    // -----------------------------------------------------------------------
    // default_fallbacks
    // -----------------------------------------------------------------------

    #[test]
    fn parse_default_fallbacks_list_form_becomes_wildcard_rule() {
        let val = json!(["model-a", "model-b"]);
        let rules = parse_default_fallbacks(Some(&val));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].primary, "*");
        assert_eq!(rules[0].fallbacks, vec!["model-a", "model-b"]);
    }

    #[test]
    fn parse_default_fallbacks_dict_form_mirrors_fallbacks() {
        let val = json!([{"original-model": ["model-a", "model-b"]}]);
        let rules = parse_default_fallbacks(Some(&val));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].primary, "original-model");
        assert_eq!(rules[0].fallbacks, vec!["model-a", "model-b"]);
    }

    #[test]
    fn parse_default_fallbacks_empty_and_invalid() {
        assert!(parse_default_fallbacks(None).is_empty());
        assert!(parse_default_fallbacks(Some(&json!(null))).is_empty());
        assert!(parse_default_fallbacks(Some(&json!("model-a"))).is_empty());
        assert!(parse_default_fallbacks(Some(&json!([]))).is_empty());
    }

    #[test]
    fn build_router_config_default_fallbacks_defaults_empty() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert!(cfg.default_fallbacks.is_empty());
    }

    #[test]
    fn build_router_config_default_fallbacks_list_wired() {
        let settings = settings_with(RoutingKnobs {
            default_fallbacks: Some(json!(["model-a", "model-b"])),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.default_fallbacks.len(), 1);
        assert_eq!(cfg.default_fallbacks[0].primary, "*");
        assert_eq!(
            cfg.default_fallbacks[0].fallbacks,
            vec!["model-a", "model-b"]
        );
    }

    #[test]
    fn build_router_config_default_fallbacks_router_settings_wins() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings_with(RoutingKnobs {
                default_fallbacks: Some(json!(["settings-model"])),
                ..Default::default()
            }),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    default_fallbacks: Some(json!(["router-model"])),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(
            cfg.default_fallbacks[0].fallbacks,
            vec!["router-model"],
            "router_settings takes precedence over oxllm_settings"
        );
    }

    // -----------------------------------------------------------------------
    // default_tag / enable_weighted_failover / exception_retry_policy /
    // failure_threshold_percent / failure_threshold_min_requests wiring
    // -----------------------------------------------------------------------

    #[test]
    fn build_router_config_wires_default_tag() {
        let settings = settings_with(RoutingKnobs {
            default_tag: Some("canary".to_string()),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.default_tag.as_deref(), Some("canary"));
    }

    #[test]
    fn build_router_config_wires_enable_weighted_failover() {
        let settings = settings_with(RoutingKnobs {
            enable_weighted_failover: Some(true),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert!(cfg.enable_weighted_failover);
    }

    #[test]
    fn build_router_config_wires_exception_retry_policy() {
        let settings = settings_with(RoutingKnobs {
            exception_retry_policy: Some(oxllm_core::router::RetryPolicy {
                rate_limit_retries: Some(5),
                timeout_retries: Some(3),
                internal_server_error_retries: Some(2),
                content_policy_retries: Some(1),
            }),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(
            cfg.exception_retry_policy,
            Some(oxllm_core::router::RetryPolicy {
                rate_limit_retries: Some(5),
                timeout_retries: Some(3),
                internal_server_error_retries: Some(2),
                content_policy_retries: Some(1),
            })
        );
    }

    #[test]
    fn build_router_config_wires_failure_threshold_percent_and_min_requests() {
        let settings = settings_with(RoutingKnobs {
            failure_threshold_percent: Some(0.8),
            failure_threshold_min_requests: Some(5),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.failure_threshold_percent, Some(0.8));
        assert_eq!(cfg.failure_threshold_min_requests, Some(5));
    }

    #[test]
    fn build_router_config_rejects_out_of_range_failure_threshold_percent() {
        for bad in [1.5, -0.1, f64::NAN, f64::INFINITY] {
            let settings = settings_with(RoutingKnobs {
                failure_threshold_percent: Some(bad),
                ..Default::default()
            });
            let err = super::build_router_config(&proxy_with_settings(settings))
                .expect_err("out-of-range threshold must be rejected");
            assert!(
                err.to_string().contains("failure_threshold_percent"),
                "error must name the knob: {err}"
            );
        }
    }

    #[test]
    fn build_router_config_rejects_zero_failure_threshold_min_requests() {
        let settings = settings_with(RoutingKnobs {
            failure_threshold_percent: Some(0.5),
            failure_threshold_min_requests: Some(0),
            ..Default::default()
        });
        assert!(super::build_router_config(&proxy_with_settings(settings)).is_err());
    }

    #[test]
    fn build_router_config_dead_knobs_reachable_from_yaml() {
        // These knobs were dead config: any YAML setting them was silently
        // swallowed into the `extra` flatten map. Parse a real YAML config and
        // assert every one lands in `RouterConfig`.
        let yaml = r#"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
router_settings:
  default_tag: canary
  enable_weighted_failover: true
  exception_retry_policy:
    rate_limit_retries: 4
    timeout_retries: 2
  failure_threshold_percent: 0.75
  failure_threshold_min_requests: 10
"#;
        let config =
            crate::config::loader::parse_config_str(yaml).expect("config must parse from YAML");
        let cfg = super::build_router_config(&config).expect("config must be valid");
        assert_eq!(cfg.default_tag.as_deref(), Some("canary"));
        assert!(cfg.enable_weighted_failover);
        assert_eq!(
            cfg.exception_retry_policy
                .as_ref()
                .and_then(|p| p.rate_limit_retries),
            Some(4)
        );
        assert_eq!(cfg.failure_threshold_percent, Some(0.75));
        assert_eq!(cfg.failure_threshold_min_requests, Some(10));
        assert_eq!(cfg.max_fallbacks, 5);
    }

    #[test]
    fn build_router_config_out_of_range_threshold_fails_yaml_load() {
        let yaml = r#"
model_list: []
router_settings:
  failure_threshold_percent: 1.5
"#;
        let config =
            crate::config::loader::parse_config_str(yaml).expect("config must parse from YAML");
        assert!(
            super::build_router_config(&config).is_err(),
            "failure_threshold_percent: 1.5 must fail config validation"
        );
    }

    #[test]
    fn percentage_cooldown_engages_via_yaml_built_config() {
        // Exercise the percentage-cooldown path through a `RouterConfig` built
        // from YAML (not only direct `RouterConfig` construction), so the
        // operator-visible knob is proven live end to end.
        use oxllm_core::router::{AllowedFailDecision, Router};

        let yaml = r#"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o-mini
router_settings:
  failure_threshold_percent: 0.5
  failure_threshold_min_requests: 3
  cooldown_time: 60
"#;
        let config =
            crate::config::loader::parse_config_str(yaml).expect("config must parse from YAML");
        let cfg = super::build_router_config(&config).expect("config must be valid");
        let router = Router::new(cfg);
        let dep_key = "gpt-4o:unit-test-dep";

        // Warm the group size (set on every pick); two deployments make the
        // single-deployment exemption inapplicable.
        router.cooldown_store().set_group_size("gpt-4o", 2);

        // 3 failures (win_total=3, win_failures=3 → 100% > 50%) trip the
        // threshold with min_requests=3.
        let mut seen_limit = false;
        for _ in 0..3 {
            let decision = router.record_failure_and_check(dep_key, 500, "gpt-4o");
            if matches!(decision, AllowedFailDecision::LimitReached { .. }) {
                seen_limit = true;
            }
        }
        assert!(
            seen_limit,
            "percentage cooldown must engage after failures exceed the threshold"
        );
        // Dilute with 4 successes: ratio drops to 3/7 ≈ 0.43, so a further
        // single failure (4/8 = 0.5, not strictly > 0.5) must NOT re-cool.
        for _ in 0..4 {
            router.clear_failure_counts(dep_key);
        }
        let decision = router.record_failure_and_check(dep_key, 500, "gpt-4o");
        assert!(
            matches!(decision, AllowedFailDecision::Tolerated { .. })
                || matches!(decision, AllowedFailDecision::NoPolicy),
            "healthy traffic must dilute the failure ratio below the threshold"
        );
    }

    // -----------------------------------------------------------------------
    // budget_throttle_pct
    // -----------------------------------------------------------------------

    #[test]
    fn build_router_config_budget_throttle_pct_defaults_none() {
        let cfg = build_router_config(&proxy_with_settings(OxllmSettings::default()));
        assert_eq!(cfg.budget_throttle_pct, None);
    }

    #[test]
    fn build_router_config_budget_throttle_pct_from_oxllm_settings() {
        let settings = settings_with(RoutingKnobs {
            budget_throttle_pct: Some(0.8),
            ..Default::default()
        });
        let cfg = build_router_config(&proxy_with_settings(settings));
        assert_eq!(cfg.budget_throttle_pct, Some(0.8));
    }

    #[test]
    fn build_router_config_budget_throttle_pct_from_router_settings() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    budget_throttle_pct: Some(0.5),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(cfg.budget_throttle_pct, Some(0.5));
    }

    #[test]
    fn build_router_config_router_settings_budget_throttle_wins() {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings_with(RoutingKnobs {
                budget_throttle_pct: Some(0.3),
                ..Default::default()
            }),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    budget_throttle_pct: Some(0.7),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let cfg = build_router_config(&config);
        assert_eq!(
            cfg.budget_throttle_pct,
            Some(0.7),
            "router_settings takes precedence over oxllm_settings"
        );
    }

    #[test]
    fn build_router_config_rejects_budget_throttle_pct_above_one() {
        let settings = settings_with(RoutingKnobs {
            budget_throttle_pct: Some(1.5),
            ..Default::default()
        });
        let err = super::build_router_config(&proxy_with_settings(settings))
            .expect_err(">1 must be rejected");
        assert!(
            err.to_string().contains("budget_throttle_pct"),
            "error must name the knob: {err}"
        );
    }

    #[test]
    fn build_router_config_rejects_budget_throttle_pct_negative() {
        let settings = settings_with(RoutingKnobs {
            budget_throttle_pct: Some(-0.1),
            ..Default::default()
        });
        assert!(super::build_router_config(&proxy_with_settings(settings)).is_err());
    }

    #[test]
    fn build_router_config_rejects_budget_throttle_pct_nan() {
        let settings = settings_with(RoutingKnobs {
            budget_throttle_pct: Some(f64::NAN),
            ..Default::default()
        });
        assert!(super::build_router_config(&proxy_with_settings(settings)).is_err());
    }

    #[test]
    fn build_router_config_accepts_boundary_budget_throttle_pct() {
        for valid in [0.0, 1.0] {
            let settings = settings_with(RoutingKnobs {
                budget_throttle_pct: Some(valid),
                ..Default::default()
            });
            let cfg = build_router_config(&proxy_with_settings(settings));
            assert_eq!(cfg.budget_throttle_pct, Some(valid));
        }
    }
}
