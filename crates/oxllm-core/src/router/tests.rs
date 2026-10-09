use super::*;
use crate::deployment::{Deployment, OxllmParams};

fn make_dep(name: &str) -> Deployment {
    Deployment {
        model_name: name.to_string(),
        oxllm_params: OxllmParams {
            model: format!("openai/{name}"),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_dep_with_model(name: &str, model: &str) -> Deployment {
    let mut d = make_dep(name);
    d.oxllm_params.model = model.to_string();
    d
}

fn default_router() -> Router {
    Router::new(RouterConfig::default())
}

// ---------------------------------------------------------------------------
// Allowed-fails policy
// ---------------------------------------------------------------------------

fn router_with_allowed_fails(
    allowed_fails: Option<u32>,
    policy: Option<HashMap<String, u32>>,
) -> Router {
    Router::new(RouterConfig {
        allowed_fails,
        allowed_fails_policy: policy,
        ..RouterConfig::default()
    })
}

#[test]
fn allowed_fails_no_policy_is_not_counted() {
    let router = default_router();
    let decision = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(decision, AllowedFailDecision::NoPolicy);
    assert_eq!(
        router.cooldown_store().failure_count("gpt-4o:0", 500),
        0,
        "without a policy, failures must not be counted"
    );
}

#[test]
fn allowed_fails_global_threshold() {
    // Failure-policy parity: `allowed_fails: 2` tolerates 2 failures, cools on the
    // 3rd (`updated_fails > allowed_fails`).
    let router = router_with_allowed_fails(Some(2), None);
    let d1 = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(d1, AllowedFailDecision::Tolerated { count: 1, limit: 2 });
    let d2 = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(d2, AllowedFailDecision::Tolerated { count: 2, limit: 2 });
    let d3 = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(d3, AllowedFailDecision::LimitReached { count: 3, limit: 2 });
}

#[test]
fn allowed_fails_policy_per_status_thresholds() {
    // 429 tolerated three times and cools on the 4th; 401 tolerated once
    // and cools on the 2nd (exclusive `>` semantics).
    let policy = HashMap::from([("429".to_string(), 3u32), ("401".to_string(), 1u32)]);
    let router = router_with_allowed_fails(None, Some(policy));

    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o"),
        AllowedFailDecision::Tolerated { count: 1, limit: 3 }
    );
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o"),
        AllowedFailDecision::Tolerated { count: 2, limit: 3 }
    );
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o"),
        AllowedFailDecision::Tolerated { count: 3, limit: 3 }
    );
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o"),
        AllowedFailDecision::LimitReached { count: 4, limit: 3 }
    );

    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 401, "gpt-4o"),
        AllowedFailDecision::Tolerated { count: 1, limit: 1 }
    );
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 401, "gpt-4o"),
        AllowedFailDecision::LimitReached { count: 2, limit: 1 }
    );
}

#[test]
fn allowed_fails_policy_falls_back_to_global_for_unlisted_status() {
    let policy = HashMap::from([("429".to_string(), 5u32)]);
    let router = router_with_allowed_fails(Some(1), Some(policy));
    // 500 is not in the per-status policy → the global limit (1) applies.
    // Tolerated on the 1st, cooled on the 2nd.
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o"),
        AllowedFailDecision::Tolerated { count: 1, limit: 1 }
    );
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o"),
        AllowedFailDecision::LimitReached { count: 2, limit: 1 }
    );
    // 429 uses its own higher limit, not the global.
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o"),
        AllowedFailDecision::Tolerated { count: 1, limit: 5 }
    );
}

#[test]
fn allowed_fails_counts_cleared_on_success() {
    let router = router_with_allowed_fails(Some(3), None);
    router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    router.clear_failure_counts("gpt-4o:0");
    assert_eq!(
        router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o"),
        AllowedFailDecision::Tolerated { count: 1, limit: 3 },
        "after a success the count restarts from zero"
    );
}

#[test]
fn allowed_fails_zero_cooldown_uses_default_failure_window() {
    // `cooldown_time_secs: 0` must not silently disable allowed-fails
    // counting: the failure window falls back to the 60s default, so
    // consecutive failures accumulate toward the limit.
    let router = Router::new(RouterConfig {
        allowed_fails: Some(2),
        cooldown_time_secs: 0,
        ..RouterConfig::default()
    });
    let first = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(first, AllowedFailDecision::Tolerated { count: 1, limit: 2 });
    let second = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(
        second,
        AllowedFailDecision::Tolerated { count: 2, limit: 2 },
        "a zero cooldown must fall back to the default failure-counting window"
    );
    let third = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(
        third,
        AllowedFailDecision::LimitReached { count: 3, limit: 2 },
        "third failure cools the deployment under LiteLLM `>` semantics"
    );
}

#[test]
fn allowed_fails_limit_for_resolution() {
    let router = router_with_allowed_fails(Some(4), None);
    assert_eq!(router.allowed_fails_limit_for(500), Some(4));

    let router =
        router_with_allowed_fails(Some(4), Some(HashMap::from([("429".to_string(), 9u32)])));
    assert_eq!(router.allowed_fails_limit_for(429), Some(9));
    assert_eq!(router.allowed_fails_limit_for(500), Some(4));

    let router = default_router();
    assert_eq!(router.allowed_fails_limit_for(500), None);
}

/// Failure-policy parity: pin the exact (count, limit) transition points for a
/// range of `allowed_fails` values against the upstream `updated_fails >
/// allowed_fails` semantics. A future "simplification" of the comparison
/// back to `>=` would flip one of these and trip this test.
#[test]
fn allowed_fails_litellm_parity_transition_points() {
    for limit in [0u32, 1, 2, 3] {
        let router = router_with_allowed_fails(Some(limit), None);
        for count in 0..=(limit + 1) {
            let decision = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
            let observed_count = match decision {
                AllowedFailDecision::Tolerated { count, .. } => count,
                AllowedFailDecision::LimitReached { count, .. } => count,
                AllowedFailDecision::NoPolicy => unreachable!(),
            };
            // `record_failure_and_check` increments then evaluates, so the
            // first call observes count=1 (not 0).
            let expected_count = count + 1;
            assert_eq!(
                observed_count, expected_count,
                "limit={limit} call#{expected_count}: count mismatch"
            );
            let expected_reached = expected_count > limit;
            let actually_reached = matches!(decision, AllowedFailDecision::LimitReached { .. });
            assert_eq!(
                actually_reached, expected_reached,
                "limit={limit} call#{expected_count}: expected LimitReached={expected_reached} \
                 (count > limit), got {actually_reached}"
            );
        }
    }
}

/// Client errors must not mark cooldown.
/// oxllm only calls `mark_cooldown` after retryable exhaustion — never on 4xx paths.
#[test]
fn cooldown_not_marked_until_explicit_mark() {
    let router = default_router();
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
    assert!(
        !router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&deps[0])),
        "upstream 4xx paths should not auto-cooldown"
    );
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(
        router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&deps[0]))
    );
}

#[test]
fn picks_from_available_deployments() {
    let router = default_router();
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    let picked = router.pick_deployment(&deps, &ctx);
    assert!(picked.is_some());
    assert_eq!(picked.unwrap().model_name, "gpt-4o");
}

#[test]
fn cooled_deployment_is_excluded() {
    let router = default_router();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    for _ in 0..50 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.model_name, "gpt-4o",
            "should still be able to pick the non-cooled deployment"
        );
        let idx = deps.iter().position(|d| std::ptr::eq(d, picked)).unwrap();
        assert_eq!(idx, 1, "cooled deployment (index 0) should be excluded");
    }
}

#[test]
fn all_cooled_returns_none() {
    let router = default_router();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    router.mark_cooldown(&router.cooldown_key(&deps[1]));
    assert!(router.pick_deployment(&deps, &ctx).is_none());
}

#[test]
fn cooldown_expires_after_duration() {
    let config = RouterConfig {
        cooldown_time_secs: 0,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "cooldown_time_secs=0 means deployment is immediately available again"
    );
}

#[test]
fn exact_match_preferred_over_wildcard() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/*"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    let picked = router.pick_deployment(&deps, &ctx).unwrap();
    assert_eq!(picked.model_name, "gpt-4o");
}

#[test]
fn wildcard_star_matches_unconfigured_model() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "*")];
    let ctx = RequestContext::new("anything-goes");
    let picked = router.pick_deployment(&deps, &ctx);
    assert!(picked.is_some());
    assert_eq!(picked.unwrap().model_name, "*");
}

#[test]
fn openai_slash_wildcard_matches_openai_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/*")];
    let ctx = RequestContext::new("openai/gpt-4o-mini");
    let picked = router.pick_deployment(&deps, &ctx);
    assert!(picked.is_some());
}

#[test]
fn anthropic_slash_wildcard_matches_anthropic_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "anthropic/*")];
    let ctx = RequestContext::new("anthropic/claude-3-5-sonnet");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn bedrock_slash_wildcard_matches_bedrock_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "bedrock/*")];
    let ctx = RequestContext::new("bedrock/anthropic.claude-3-5-sonnet");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn gemini_slash_wildcard_matches_gemini_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "gemini/*")];
    let ctx = RequestContext::new("gemini/gemini-1.5-pro");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn groq_slash_wildcard_matches_groq_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "groq/*")];
    let ctx = RequestContext::new("groq/llama-3.1-70b");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn together_ai_slash_wildcard_matches_together_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "together_ai/*")];
    let ctx = RequestContext::new("together_ai/llama-v3-70b");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn openrouter_slash_wildcard_matches_openrouter_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "openrouter/*")];
    let ctx = RequestContext::new("openrouter/openai/gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn xai_slash_wildcard_matches_xai_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "xai/*")];
    let ctx = RequestContext::new("xai/grok-2");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn mistral_slash_wildcard_matches_mistral_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "mistral/*")];
    let ctx = RequestContext::new("mistral/mistral-large-latest");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn cohere_slash_wildcard_matches_cohere_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "cohere/*")];
    let ctx = RequestContext::new("cohere/command-r-plus");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn azure_slash_wildcard_matches_azure_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "azure/*")];
    let ctx = RequestContext::new("azure/my-gpt4-deployment");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn vertex_ai_slash_wildcard_matches_vertex_gemini_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "vertex_ai/*")];
    let ctx = RequestContext::new("vertex_ai/gemini-1.5-pro");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn vertex_ai_slash_wildcard_matches_vertex_anthropic_models() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "vertex_ai/*")];
    let ctx = RequestContext::new("vertex_ai/claude-3-5-sonnet@20241022");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn vertex_ai_slash_wildcard_does_not_match_other_providers() {
    let router = default_router();
    let deps = [make_dep_with_model("*", "vertex_ai/*")];
    let ctx = RequestContext::new("openai/gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "vertex_ai/* wildcard must not match an openai/* model"
    );
    let ctx2 = RequestContext::new("anthropic/claude-3-5-sonnet");
    assert!(
        router.pick_deployment(&deps, &ctx2).is_none(),
        "vertex_ai/* wildcard must not match an anthropic/* model"
    );
}

#[test]
fn wildcard_matches_bare_model_name_without_provider_prefix() {
    // A bare model name (no `/`) defaults to the `openai` provider, so it is
    // served by a bare-`*` wildcard (which matches anything). A `provider/*`
    // wildcard would only match it when that provider is `openai`.
    let router = default_router();
    let deps = [make_dep_with_model("*", "*")];
    let ctx = RequestContext::new("claude-fable-5");
    let picked = router.pick_deployment(&deps, &ctx);
    assert!(
        picked.is_some(),
        "bare wildcard must match a bare model name"
    );
    assert_eq!(picked.unwrap().model_name, "*");
}

#[test]
fn bare_non_wildcard_model_does_not_match_other_providers() {
    // A wildcard deployment configured with a bare non-`*` model (no `/` —
    // e.g. a config typo dropping the provider prefix) must NOT become a
    // global catch-all: it serves only an identically-named requested model.
    let router = default_router();
    let deps = [make_dep_with_model("*", "gpt-4o")];
    let ctx = RequestContext::new("anthropic/gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "bare 'gpt-4o' must not match an anthropic/gpt-4o request"
    );
    let ctx = RequestContext::new("gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "bare 'gpt-4o' must match an identical requested model"
    );
}

#[test]
fn wildcard_fully_qualified_model_does_not_match_other_model() {
    // A wildcard deployment configured with a fully-qualified provider model
    // carrying NO `*` (e.g. a config typo for `openai/*`, or a copy-paste from
    // a non-wildcard deployment) must be an exact provider-model match only —
    // it must NOT silently become a provider catch-all that serves every
    // same-provider request (rewriting the upstream body and billing at the
    // configured model's rates).
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/gpt-4o")];
    let ctx = RequestContext::new("openai/gpt-3.5-turbo");
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "openai/gpt-4o on a wildcard deployment must not serve an openai/gpt-3.5-turbo request"
    );
    let ctx = RequestContext::new("openai/gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "openai/gpt-4o must still match an identical openai/gpt-4o request"
    );
    // A bare requested model defaults to openai and matches the exact
    // provider-model pattern, but not other bare names.
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
    let ctx = RequestContext::new("gpt-3.5-turbo");
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "openai/gpt-4o must not match a bare gpt-3.5-turbo request"
    );
}

#[test]
fn openai_wildcard_matches_bare_model_name_defaulting_to_openai() {
    // Bare requested models default to the openai provider, so an openai/*
    // wildcard serves them.
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/*")];
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn openai_wildcard_does_not_match_anthropic_prefixed_model() {
    // Bug #7 regression: an `openai/*` wildcard must NOT serve a request for
    // an `anthropic/...` model. Previously the bare-`*` model_name fallback
    // matched any provider, causing cross-provider misrouting.
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/*")];
    let ctx = RequestContext::new("anthropic/claude-3-5-sonnet");
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "openai/* wildcard must not match an anthropic/* model"
    );
    // And symmetrically the other direction.
    let deps2 = [make_dep_with_model("*", "anthropic/*")];
    let ctx2 = RequestContext::new("openai/gpt-5");
    assert!(
        router.pick_deployment(&deps2, &ctx2).is_none(),
        "anthropic/* wildcard must not match an openai/* model"
    );
}

#[test]
fn anthropic_wildcard_does_not_match_bare_model_name() {
    // Bare requested models default to openai, so an anthropic/* wildcard
    // does not serve them (previously it did).
    let router = default_router();
    let deps = [make_dep_with_model("*", "anthropic/*")];
    let ctx = RequestContext::new("claude-fable-5");
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "anthropic/* wildcard must not match a bare (openai-default) model"
    );
}

#[test]
fn bare_wildcard_matches_any_provider() {
    // A bare `*` configured model (no provider prefix) matches everything.
    let router = default_router();
    let deps = [make_dep_with_model("*", "*")];
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("anthropic/claude-3-5-sonnet"))
            .is_some()
    );
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("openai/gpt-5"))
            .is_some()
    );
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("claude-fable-5"))
            .is_some()
    );
}

// ---- partial-pattern wildcards (suffix glob in the model name) ----------

#[test]
fn partial_wildcard_suffix_matches_same_provider_prefix() {
    // `groq/llama-*` matches `groq/llama-3.1-70b`.
    let router = default_router();
    let deps = [make_dep_with_model("*", "groq/llama-*")];
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("groq/llama-3.1-70b"))
            .is_some(),
        "groq/llama-* must match groq/llama-3.1-70b"
    );
}

#[test]
fn partial_wildcard_suffix_rejects_different_model_same_provider() {
    // `groq/llama-*` must NOT match `groq/mistral-7b`.
    let router = default_router();
    let deps = [make_dep_with_model("*", "groq/llama-*")];
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("groq/mistral-7b"))
            .is_none(),
        "groq/llama-* must not match groq/mistral-7b"
    );
}

#[test]
fn partial_wildcard_rejects_different_provider() {
    // Provider prefix must still match: `openai/gpt-4*` must NOT serve an
    // anthropic model even if its name started with `gpt-4`.
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/gpt-4*")];
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("anthropic/gpt-4o"))
            .is_none(),
        "openai/gpt-4* must not match an anthropic/* model"
    );
}

#[test]
fn partial_wildcard_matches_bare_model_defaulting_to_openai() {
    // A bare requested model defaults to the openai provider, so
    // `openai/gpt-4*` serves `gpt-4o`.
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/gpt-4*")];
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("gpt-4o-mini"))
            .is_some(),
        "openai/gpt-4* must match bare gpt-4o-mini (defaults to openai)"
    );
}

#[test]
fn partial_wildcard_middle_glob_matches() {
    // A `*` in the middle of the model part works like the full glob.
    let router = default_router();
    let deps = [make_dep_with_model("*", "openai/gpt-*-mini")];
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("openai/gpt-4o-mini"))
            .is_some()
    );
    assert!(
        router
            .pick_deployment(&deps, &RequestContext::new("openai/gpt-4o"))
            .is_none(),
        "openai/gpt-*-mini must not match openai/gpt-4o"
    );
}

#[test]
fn glob_model_match_exact_and_wildcard() {
    assert!(glob_model_match("llama-3.1-70b", "llama-3.1-70b"));
    assert!(!glob_model_match("llama-3.1-70b", "llama-3.1-8b"));
    assert!(glob_model_match("llama-*", "llama-3.1-70b"));
    assert!(!glob_model_match("llama-*", "mistral-7b"));
    assert!(glob_model_match("gpt-*-mini", "gpt-4o-mini"));
    assert!(!glob_model_match("gpt-*-mini", "gpt-4o"));
    assert!(glob_model_match("*-mini", "claude-mini"));
    assert!(glob_model_match("*", "anything"));
    assert!(glob_model_match("gpt-4*", "gpt-4"));
    // Star backtracking: greedy-leftmost middle placement must be undone.
    // Middle segments must appear in order without overlap: "ab" then "ba".
    assert!(!glob_model_match("*ab*ba", "aba")); // overlap at 'b' is not allowed
    assert!(glob_model_match("*ab*ba", "abba"));
    assert!(glob_model_match("a*b*c", "abc"));
    assert!(glob_model_match("a*b*c", "axxbyyc"));
    assert!(!glob_model_match("a*b*c", "acb"));
    assert!(glob_model_match("*ee*", "ee"));
    assert!(!glob_model_match("*ee*", "e"));
}

#[test]
fn glob_model_match_multibyte_utf8() {
    // Pin multi-byte (UTF-8) safety: the matcher slices on byte offsets
    // derived from `str::find`/`starts_with`/`ends_with`, which always yield
    // char-aligned offsets for `&str` needles. "é" is two bytes in UTF-8.
    // Prefix segment is multi-byte.
    assert!(glob_model_match("café-*", "café-noir"));
    assert!(!glob_model_match("café-*", "cafe-noir"));
    // Suffix segment is multi-byte.
    assert!(glob_model_match("*-café", "noir-café"));
    assert!(!glob_model_match("*-café", "noir-cafe"));
    // Middle segment is multi-byte (exercises `pos += idx + part.len()`).
    assert!(glob_model_match("*-café-*", "pre-café-post"));
    assert!(glob_model_match("a-*-z", "a-café-z"));
    // `café` (two bytes) is distinct from `cafe` (one byte) at every position.
    assert!(!glob_model_match("*café*", "a-cafe-z"));
    assert!(!glob_model_match("a-*-café-z", "a-cafe-z"));
    // Exact match across multi-byte text.
    assert!(glob_model_match("café", "café"));
    assert!(!glob_model_match("café", "cafe"));
}

#[test]
fn empty_configured_model_wildcard_is_not_a_global_catch_all() {
    // A `model_name: "*"` deployment with an empty `oxllm_params.model` is
    // near-certain config breakage; it must not silently serve (and bill
    // at its configured model's rates) every request, including
    // cross-provider ones. The surrounding guards already fail closed on
    // bare non-`*` names and fully-qualified names — the empty string is
    // the same class of typo.
    let router = default_router();
    let dep = make_dep_with_model("*", "");
    for requested in ["anthropic/claude-3", "openai/gpt-4o", "gpt-4o"] {
        assert!(
            router
                .deployments_for_model(std::slice::from_ref(&dep), requested)
                .is_empty(),
            "empty configured model must not serve {requested}"
        );
    }
}

#[test]
fn star_slash_wildcard_matches_nothing() {
    // `*/*` reads like "match everything" but the provider prefix must be
    // a literal: the pattern matches no requested model (fail-closed).
    // Pin it so the config footgun is at least documented behavior.
    // (Requesting literally "*" is NOT served by the pattern — it only
    // resolves through the exact `model_name == "*"` branch.)
    let router = default_router();
    let dep = make_dep_with_model("*", "*/*");
    for requested in ["openai/gpt-4o", "anthropic/claude-3", "gpt-4o"] {
        assert!(
            router
                .deployments_for_model(std::slice::from_ref(&dep), requested)
                .is_empty(),
            "`*/*` must not serve {requested}"
        );
    }
}

#[test]
fn wildcard_spray_does_not_grow_group_sizes_or_round_robin_counters() {
    // A wildcard deployment resolves every distinct client-requested model
    // string, so per-group state (cooldown `group_sizes`, round-robin
    // counters) must key by the wildcard group name `"*"`, never the raw
    // requested string — otherwise an attacker spraying
    // `model: "x-<random>"` grows both maps without bound between config
    // reloads (the only sweep, `prune_stale_store_keys`, runs at reload).
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::RoundRobin,
        ..RouterConfig::default()
    });
    let mut a = make_dep_with_model("*", "*");
    a.oxllm_params.api_key = Some("sk-aaaaaaaaaaaaaaaaaaaaaaaa".to_string());
    let mut b = make_dep_with_model("*", "*");
    b.oxllm_params.api_key = Some("sk-bbbbbbbbbbbbbbbbbbbbbbbb".to_string());
    let deps = [a, b];
    for i in 0..10_000 {
        let ctx = RequestContext::new(format!("x-{i}"));
        assert!(
            router.pick_deployment(&deps, &ctx).is_some(),
            "wildcard must still serve request {i}"
        );
    }
    assert_eq!(
        router.cooldown_store().group_size_key_count(),
        1,
        "group sizes must aggregate wildcard-served picks under \"*\""
    );
    assert_eq!(
        router
            .round_robin_counters_arc()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len(),
        1,
        "round-robin counters must aggregate wildcard-served picks under \"*\""
    );
    // The aggregated "*" group size is the wildcard deployment count, not
    // the per-request candidate count of any single request.
    assert_eq!(router.cooldown_store().group_size_for("*"), 2);
}

#[test]
fn wildcard_solo_deployment_percentage_safety_engages() {
    // The group-size READ path must mirror the "*" aggregation: a model
    // with no exact deployments (wildcard-served) falls back to the "*"
    // group size, so the single-deployment percentage safety still
    // protects a solo wildcard deployment — the recorded failure's `model`
    // argument is the raw requested string, which has no entry of its own.
    let router = router_with_percentage_threshold(0.1, 1);
    let dep = make_dep_with_model("*", "*");
    let ctx = RequestContext::new("random-wildcard-model");
    let _ = router.pick_deployment(std::slice::from_ref(&dep), &ctx);
    let key = router.cooldown_key(&dep);
    for _ in 0..20 {
        let d = router.record_failure_and_check(&key, 500, "random-wildcard-model");
        assert!(
            !matches!(d, AllowedFailDecision::LimitReached { .. }),
            "solo wildcard deployment must be protected by the single-deployment \
             percentage safety via the \"*\" group-size fallback, got {d:?}"
        );
    }
}

#[test]
fn no_deployments_returns_none() {
    let router = default_router();
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&[], &ctx).is_none());
}

#[test]
fn unknown_model_with_no_wildcard_returns_none() {
    let router = default_router();
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("unknown-model");
    assert!(router.pick_deployment(&deps, &ctx).is_none());
}

#[test]
fn fallbacks_returned_correctly() {
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["gpt-4o-mini".to_string(), "claude".to_string()],
        }],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let fbs = router.fallbacks_for("gpt-4o");
    assert_eq!(fbs, vec!["gpt-4o-mini", "claude"]);
}

#[test]
fn no_fallbacks_for_unknown_model() {
    let router = default_router();
    assert!(router.fallbacks_for("unknown").is_empty());
    assert!(router.context_window_fallbacks_for("unknown").is_empty());
}

#[test]
fn routing_group_fallbacks_with_non_matching_primary_falls_through_to_global() {
    // A routing group for "gpt-4o" lists a fallback rule whose `primary` is
    // "other" (a config typo). The router must fall through to the global
    // fallback chain for "gpt-4o" rather than silently returning empty.
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            fallbacks: vec![FallbackRule {
                primary: "other".to_string(),
                fallbacks: vec!["wrong".to_string()],
            }],
            context_window_fallbacks: vec![FallbackRule {
                primary: "other".to_string(),
                fallbacks: vec!["wrong-cw".to_string()],
            }],
            content_policy_fallbacks: vec![FallbackRule {
                primary: "other".to_string(),
                fallbacks: vec!["wrong-cp".to_string()],
            }],
            ..Default::default()
        },
    );
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["gpt-4o-mini".to_string(), "claude".to_string()],
        }],
        context_window_fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["gpt-4o-cw".to_string()],
        }],
        content_policy_fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["gpt-4o-cp".to_string()],
        }],
        routing_groups: groups,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(
        router.fallbacks_for("gpt-4o"),
        vec!["gpt-4o-mini", "claude"],
        "non-matching group fallback rules must fall through to global"
    );
    assert_eq!(
        router.context_window_fallbacks_for("gpt-4o"),
        vec!["gpt-4o-cw"],
    );
    assert_eq!(
        router.content_policy_fallbacks_for("gpt-4o"),
        vec!["gpt-4o-cp"],
    );
}

#[test]
fn routing_group_fallbacks_with_matching_primary_override_global() {
    // When a group's fallback rule matches the requested primary, the group
    // override wins over the global chain.
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            fallbacks: vec![FallbackRule {
                primary: "gpt-4o".to_string(),
                fallbacks: vec!["group-fb".to_string()],
            }],
            ..Default::default()
        },
    );
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["global-fb".to_string()],
        }],
        routing_groups: groups,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(router.fallbacks_for("gpt-4o"), vec!["group-fb"]);
}

#[test]
fn context_window_fallbacks_returned_correctly() {
    let config = RouterConfig {
        context_window_fallbacks: vec![FallbackRule {
            primary: "gpt-3.5-turbo".to_string(),
            fallbacks: vec!["gpt-4o".to_string()],
        }],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let cwf = router.context_window_fallbacks_for("gpt-3.5-turbo");
    assert_eq!(cwf, vec!["gpt-4o"]);
}

#[test]
fn pick_deployment_index_returns_correct_index() {
    let router = default_router();
    let deps = [make_dep("other"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    let idx = router.pick_deployment_index(&deps, &ctx);
    assert_eq!(idx, Some(1));
}

#[test]
fn cooldown_key_is_stable_deployment_key() {
    let router = default_router();
    let dep = make_dep("gpt-4o");
    assert_eq!(router.cooldown_key(&dep), stable_key(&dep));
    // Stable: the same deployment always yields the same key.
    assert_eq!(router.cooldown_key(&dep), router.cooldown_key(&dep));
}

#[test]
fn cooldown_key_is_a_pure_function_of_deployment_identity() {
    // The cooldown key is a pure function of the deployment — so a hot-reload
    // that reorders or inserts deployments keeps each deployment's
    // cooldown/health/bandit state attached to the deployment, never to a
    // list slot. There is no positional index and no `__unknown__` sentinel.
    let a = make_dep_with_model("gpt-4o", "openai/gpt-4o-a");
    let b = make_dep_with_model("gpt-4o", "openai/gpt-4o-b");
    let outsider = make_dep_with_model("gpt-4o", "openai/gpt-4o-a");
    assert_ne!(
        stable_key(&a),
        stable_key(&b),
        "distinct provider models must key distinct deployments"
    );
    assert_eq!(
        stable_key(&a),
        stable_key(&outsider),
        "the same deployment identity keys the same cell whether in-list or not"
    );
    let router = default_router();
    assert_eq!(router.cooldown_key(&a), router.cooldown_key(&outsider));
    assert_ne!(router.cooldown_key(&a), router.cooldown_key(&b));
}

#[test]
fn cooldown_key_distinguishes_deployments_sharing_model_and_provider() {
    // Two deployments of the same model_name and provider model but different
    // endpoints/credentials must NOT share a cooldown cell: cooling one must
    // never cool its sibling backend (the failover case the positional index
    // scheme handled, and a plain `deployment_key` would break).
    let router = default_router();
    let mut a = make_dep_with_model("gpt-4o", "openai/gpt-4o");
    a.oxllm_params.api_base = Some("https://a.example.test".to_string());
    a.oxllm_params.api_key = Some("key-a".to_string());
    let mut b = make_dep_with_model("gpt-4o", "openai/gpt-4o");
    b.oxllm_params.api_base = Some("https://b.example.test".to_string());
    b.oxllm_params.api_key = Some("key-b".to_string());
    assert_ne!(
        router.cooldown_key(&a),
        router.cooldown_key(&b),
        "endpoint/credential differences must key distinct deployments"
    );
}

#[test]
fn proptest_router_never_panics() {
    use proptest::prelude::*;

    proptest!(|(model in "[a-zA-Z0-9/\\*_-]{0,20}", n in 0usize..50)| {
    let deps: Vec<Deployment> = (0..n)
    .map(|i| {
    let name = if i % 5 == 0 { "*".to_string() } else { format!("model-{i}") };
    make_dep(&name)
    })
    .collect();
    let router = default_router();
    let ctx = RequestContext::new(&model);
    let _ = router.pick_deployment(&deps, &ctx);
    let _ = router.deployments_for_model(&deps, &model);
    let _ = router.pick_deployment_index(&deps, &ctx);
    });
}

// ---------------------------------------------------------------------------
// least_busy strategy, weighted shuffle, model-group alias,
// content-policy fallbacks, retry policy.
// ---------------------------------------------------------------------------

#[test]
fn least_busy_picks_idle_deployment_via_router() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::LeastBusy,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    // Mark the first deployment as having 3 in-flight requests (keyed by the
    // stable per-deployment key, as the strategies and host now do).
    router.record_in_flight(&stable_key(&deps[0]), 3);
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-b",
            "least_busy should prefer the idle deployment"
        );
    }
}

#[test]
fn session_affinity_sticks_via_router() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SessionAffinity,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o"),
        make_dep_with_model("gpt-4o", "xai/grok-4"),
        make_dep_with_model("gpt-4o", "openrouter/openai/gpt-4o"),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.session_id = Some("thread-abc".to_string());
    let first = router
        .pick_deployment(&deps, &ctx)
        .expect("should pick")
        .oxllm_params
        .model
        .clone();
    for _ in 0..10 {
        let again = router
            .pick_deployment(&deps, &ctx)
            .expect("should pick")
            .oxllm_params
            .model
            .clone();
        assert_eq!(first, again);
    }
}

#[test]
fn record_in_flight_increments_and_decrements() {
    let router = default_router();
    router.record_in_flight("k", 1);
    router.record_in_flight("k", 1);
    assert_eq!(router.usage_store().in_flight("k"), 2);
    router.record_in_flight("k", -1);
    assert_eq!(router.usage_store().in_flight("k"), 1);
}

#[test]
fn model_group_alias_expands_to_aliased_deployments() {
    let mut aliases = std::collections::HashMap::new();
    aliases.insert(
        "gpt-4".to_string(),
        vec!["gpt-4o".to_string(), "gpt-4-turbo".to_string()],
    );
    let config = RouterConfig {
        model_group_alias: aliases,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep("gpt-4o"),
        make_dep("gpt-4-turbo"),
        make_dep("claude"),
    ];
    let picked = router.deployments_for_model(&deps, "gpt-4");
    assert_eq!(picked.len(), 2, "alias should expand to two deployments");
    let names: Vec<&str> = picked.iter().map(|d| d.model_name.as_str()).collect();
    assert!(names.contains(&"gpt-4o"));
    assert!(names.contains(&"gpt-4-turbo"));
    assert!(!names.contains(&"claude"));
}

#[test]
fn model_group_alias_falls_through_when_alias_empty() {
    let mut aliases = std::collections::HashMap::new();
    aliases.insert("gpt-4".to_string(), vec!["nonexistent".to_string()]);
    let config = RouterConfig {
        model_group_alias: aliases,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    // The alias group references no real deployment; the requested model
    // "gpt-4" has no exact match either. Result should be empty.
    let deps = [make_dep("gpt-4o")];
    let picked = router.deployments_for_model(&deps, "gpt-4");
    assert!(
        picked.is_empty(),
        "alias group with no matches and no exact match should yield no deployments"
    );
}

#[test]
fn model_group_alias_falls_through_to_exact_when_alias_resolves_nothing() {
    let mut aliases = std::collections::HashMap::new();
    aliases.insert("gpt-4".to_string(), vec!["nonexistent".to_string()]);
    let config = RouterConfig {
        model_group_alias: aliases,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    // Alias resolves nothing, but an exact match for the requested model
    // exists. The router should fall through to the exact match.
    let deps = [make_dep("gpt-4")];
    let picked = router.deployments_for_model(&deps, "gpt-4");
    assert_eq!(picked.len(), 1);
    assert_eq!(picked[0].model_name, "gpt-4");
}

#[test]
fn content_policy_fallbacks_returned_correctly() {
    let config = RouterConfig {
        content_policy_fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["llama-guarded".to_string(), "claude-guarded".to_string()],
        }],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let cpf = router.content_policy_fallbacks_for("gpt-4o");
    assert_eq!(cpf, vec!["llama-guarded", "claude-guarded"]);
}

#[test]
fn no_content_policy_fallbacks_for_unknown_model() {
    let router = default_router();
    assert!(router.content_policy_fallbacks_for("unknown").is_empty());
}

#[test]
fn retries_for_status_uses_policy_when_set() {
    let mut policy = std::collections::HashMap::new();
    policy.insert("429".to_string(), 5);
    policy.insert("503".to_string(), 2);
    let config = RouterConfig {
        num_retries: 3,
        retry_policy: policy,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(router.retries_for_status(429), 5);
    assert_eq!(router.retries_for_status(503), 2);
    assert_eq!(
        router.retries_for_status(500),
        3,
        "missing statuses fall back to num_retries"
    );
}

#[test]
fn retries_for_status_defaults_to_num_retries() {
    let config = RouterConfig {
        num_retries: 4,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(router.retries_for_status(429), 4);
    assert_eq!(router.retries_for_status(500), 4);
}

#[test]
fn weighted_router_can_pick_high_weight_deployment() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SimpleShuffle,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let mut heavy = make_dep("gpt-4o");
    heavy.oxllm_params.weight = Some(99);
    let mut light = make_dep("gpt-4o");
    light.oxllm_params.weight = Some(1);
    let deps = [heavy, light];
    let ctx = RequestContext::new("gpt-4o");
    let mut heavy_hits = 0;
    for _ in 0..500 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        if picked.oxllm_params.weight == Some(99) {
            heavy_hits += 1;
        }
    }
    assert!(
        heavy_hits > 400,
        "weight=99 should win the vast majority of picks, got {heavy_hits}/500"
    );
}

fn make_dep_with_tags(name: &str, model: &str, tags: Vec<String>) -> Deployment {
    use crate::deployment::ModelInfo;
    let mut d = make_dep_with_model(name, model);
    d.model_info = Some(ModelInfo {
        tags,
        ..Default::default()
    });
    d
}

#[test]
fn tag_based_selects_tagged_deployment_via_router() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-canary", vec!["canary".to_string()]),
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-stable", vec!["stable".to_string()]),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("canary".to_string());
    // Every pick must hit the canary-tagged deployment.
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-canary",
            "tag_based must select the deployment whose tags contain the request tag"
        );
    }
}

#[test]
fn tag_based_falls_back_to_shuffle_when_no_tag_in_context() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-canary", vec!["canary".to_string()]),
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-stable", vec!["stable".to_string()]),
    ];
    let ctx = RequestContext::new("gpt-4o");
    // With no tag set, the router should still pick *something* (shuffle fallback).
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

// ---------------------------------------------------------------------------
// routing_groups — per-model strategy / fallback / retry overrides
// ---------------------------------------------------------------------------

fn make_dep_with_latency_model(name: &str, model: &str) -> Deployment {
    make_dep_with_model(name, model)
}

#[test]
fn routing_group_uses_group_strategy_for_grouped_model() {
    // Global default: simple_shuffle. Routing group "gpt-4o": lowest_latency.
    // A latency-aware pick must always land on the faster deployment, while
    // simple_shuffle would not be deterministic.
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            routing_strategy: Some(RoutingStrategyKind::LowestLatency),
            ..Default::default()
        },
    );
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SimpleShuffle,
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);
    router.record_latency(
        &stable_key(&make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-slow")),
        500,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-fast")),
        50,
    );
    let deps = [
        make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-slow"),
        make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-fast"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-fast",
            "lowest_latency group override should always pick the fast deployment"
        );
    }
}

#[test]
fn routing_group_without_override_falls_back_to_default_strategy() {
    // Default strategy: LowestLatency. Model "claude-3" has no group override;
    // it should use the default strategy (lowest latency).
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::LowestLatency,
        ..Default::default()
    };
    let router = Router::new(config);
    router.record_latency(
        &stable_key(&make_dep_with_latency_model(
            "claude-3",
            "anthropic/claude-3-slow",
        )),
        500,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model(
            "claude-3",
            "anthropic/claude-3-fast",
        )),
        50,
    );
    let deps = [
        make_dep_with_latency_model("claude-3", "anthropic/claude-3-slow"),
        make_dep_with_latency_model("claude-3", "anthropic/claude-3-fast"),
    ];
    let ctx = RequestContext::new("claude-3");
    let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
    assert_eq!(
        picked.oxllm_params.model, "anthropic/claude-3-fast",
        "ungrouped model should use default lowest_latency strategy"
    );
}

#[test]
fn routing_group_one_model_override_does_not_affect_other_models() {
    // "gpt-4o" gets lowest_latency override. "claude-3" has no group; with a
    // simple_shuffle default it may pick either deployment. But "gpt-4o" must
    // consistently pick the fast one.
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            routing_strategy: Some(RoutingStrategyKind::LowestLatency),
            ..Default::default()
        },
    );
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SimpleShuffle,
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);
    router.record_latency(
        &stable_key(&make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-slow")),
        500,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-fast")),
        50,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model(
            "claude-3",
            "anthropic/claude-3-slow",
        )),
        500,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model(
            "claude-3",
            "anthropic/claude-3-fast",
        )),
        50,
    );

    let gpt_deps = [
        make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-slow"),
        make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-fast"),
    ];
    let gpt_ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        let picked = router
            .pick_deployment(&gpt_deps, &gpt_ctx)
            .expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-fast");
    }

    // claude-3 has no override; default is simple_shuffle. Just verify it
    // picks *something* — strategy is random so we cannot assert which one.
    let claude_deps = [
        make_dep_with_latency_model("claude-3", "anthropic/claude-3-slow"),
        make_dep_with_latency_model("claude-3", "anthropic/claude-3-fast"),
    ];
    let claude_ctx = RequestContext::new("claude-3");
    assert!(router.pick_deployment(&claude_deps, &claude_ctx).is_some());
}

#[test]
fn routing_group_fallbacks_override_global_fallbacks() {
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            fallbacks: vec![FallbackRule {
                primary: "gpt-4o".to_string(),
                fallbacks: vec!["claude-group".to_string()],
            }],
            ..Default::default()
        },
    );
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["claude-global".to_string()],
        }],
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);
    let fbs = router.fallbacks_for("gpt-4o");
    assert_eq!(
        fbs,
        vec!["claude-group"],
        "group fallback should override global fallback for grouped model"
    );

    // Ungrouped model with global fallback still resolves.
    let config2 = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "claude-3".to_string(),
            fallbacks: vec!["llama".to_string()],
        }],
        ..Default::default()
    };
    let router2 = Router::new(config2);
    assert_eq!(router2.fallbacks_for("claude-3"), vec!["llama"]);
}

#[test]
fn routing_group_context_window_fallbacks_override_global() {
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            context_window_fallbacks: vec![FallbackRule {
                primary: "gpt-4o".to_string(),
                fallbacks: vec!["gpt-4o-128k".to_string()],
            }],
            ..Default::default()
        },
    );
    let config = RouterConfig {
        context_window_fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["gpt-4o-global".to_string()],
        }],
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);
    assert_eq!(
        router.context_window_fallbacks_for("gpt-4o"),
        vec!["gpt-4o-128k"]
    );
}

#[test]
fn routing_group_content_policy_fallbacks_override_global() {
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            content_policy_fallbacks: vec![FallbackRule {
                primary: "gpt-4o".to_string(),
                fallbacks: vec!["llama-guard-group".to_string()],
            }],
            ..Default::default()
        },
    );
    let config = RouterConfig {
        content_policy_fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["llama-guard-global".to_string()],
        }],
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);
    assert_eq!(
        router.content_policy_fallbacks_for("gpt-4o"),
        vec!["llama-guard-group"]
    );
}

#[test]
fn routing_group_retry_policy_overrides_global() {
    let mut group_policy = std::collections::HashMap::new();
    group_policy.insert("429".to_string(), 9);
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            retry_policy: group_policy,
            num_retries: Some(7),
            ..Default::default()
        },
    );
    let mut global_policy = std::collections::HashMap::new();
    global_policy.insert("429".to_string(), 3);
    global_policy.insert("500".to_string(), 2);
    let config = RouterConfig {
        num_retries: 1,
        retry_policy: global_policy,
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);

    // 429 — present in group override: wins.
    assert_eq!(router.retries_for_status_model(429, "gpt-4o"), 9);
    // 500 — missing in group override, falls back to global policy.
    assert_eq!(router.retries_for_status_model(500, "gpt-4o"), 2);
    // 503 — missing in both policies, falls back to group's num_retries.
    assert_eq!(router.retries_for_status_model(503, "gpt-4o"), 7);
    // Ungrouped model — uses global policy + global num_retries.
    assert_eq!(router.retries_for_status_model(429, "claude-3"), 3);
    assert_eq!(router.retries_for_status_model(503, "claude-3"), 1);
}

#[test]
fn routing_group_cooldown_time_overrides_global() {
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            cooldown_time_secs: Some(120),
            ..Default::default()
        },
    );
    let config = RouterConfig {
        cooldown_time_secs: 30,
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);
    assert_eq!(
        router.cooldown_time_secs_for_model("gpt-4o"),
        120,
        "group override cooldown must take precedence"
    );
    assert_eq!(
        router.cooldown_time_secs_for_model("claude-3"),
        30,
        "ungrouped model must use the global cooldown"
    );
}

#[test]
fn routing_group_two_models_two_strategies_route_differently() {
    // Two model groups with different strategies sharing one router.
    // gpt-4o → lowest_latency (deterministic: always picks fast).
    // claude-3 → simple_shuffle (random: not deterministic).
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            routing_strategy: Some(RoutingStrategyKind::LowestLatency),
            ..Default::default()
        },
    );
    groups.insert(
        "claude-3".to_string(),
        RouterGroupOverride {
            routing_strategy: Some(RoutingStrategyKind::SimpleShuffle),
            ..Default::default()
        },
    );
    let config = RouterConfig {
        routing_groups: groups,
        ..Default::default()
    };
    let router = Router::new(config);

    router.record_latency(
        &stable_key(&make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-slow")),
        500,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-fast")),
        50,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model(
            "claude-3",
            "anthropic/claude-3-slow",
        )),
        500,
    );
    router.record_latency(
        &stable_key(&make_dep_with_latency_model(
            "claude-3",
            "anthropic/claude-3-fast",
        )),
        50,
    );

    let gpt_deps = [
        make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-slow"),
        make_dep_with_latency_model("gpt-4o", "openai/gpt-4o-fast"),
    ];
    let claude_deps = [
        make_dep_with_latency_model("claude-3", "anthropic/claude-3-slow"),
        make_dep_with_latency_model("claude-3", "anthropic/claude-3-fast"),
    ];

    // gpt-4o's lowest_latency must always pick the fast one.
    let gpt_ctx = RequestContext::new("gpt-4o");
    for _ in 0..30 {
        let picked = router
            .pick_deployment(&gpt_deps, &gpt_ctx)
            .expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-fast");
    }

    // claude-3's simple_shuffle should eventually produce *both* picks across
    // many trials (sanity check that the shuffle strategy is wired up here,
    // not the latency strategy from the other group).
    let claude_ctx = RequestContext::new("claude-3");
    let mut saw_slow = false;
    let mut saw_fast = false;
    for _ in 0..200 {
        let picked = router
            .pick_deployment(&claude_deps, &claude_ctx)
            .expect("should pick");
        match picked.oxllm_params.model.as_str() {
            "anthropic/claude-3-slow" => saw_slow = true,
            "anthropic/claude-3-fast" => saw_fast = true,
            _ => {}
        }
    }
    assert!(
        saw_slow && saw_fast,
        "claude-3 with simple_shuffle should pick both deployments across trials"
    );
}

// ---------------------------------------------------------------------------
// Twin-inspired tests: cooldown isolation, empty deployment lists, fallback
// chain resolution, model alias edge cases.
// ---------------------------------------------------------------------------

#[test]
fn deployments_for_model_empty_slice() {
    let router = default_router();
    let result = router.deployments_for_model(&[], "gpt-4o");
    assert!(result.is_empty(), "empty slice must yield empty result");
}

#[test]
fn cooldown_isolation_across_models() {
    // Cooldown keys are the stable deployment key, so cooling claude's
    // deployment must not affect gpt-4o routing. The first assertion proves
    // the cooldown actually took effect; the second proves it is scoped to
    // claude.
    let router = default_router();
    let deps = [make_dep("gpt-4o"), make_dep("claude")];
    router.mark_cooldown(&router.cooldown_key(&deps[1]));

    let claude_ctx = RequestContext::new("claude");
    assert!(
        router.pick_deployment(&deps, &claude_ctx).is_none(),
        "claude must be unroutable while its only deployment is cooled"
    );

    let gpt_ctx = RequestContext::new("gpt-4o");
    let picked = router
        .pick_deployment(&deps, &gpt_ctx)
        .expect("gpt-4o must still route when claude is cooled");
    assert_eq!(picked.model_name, "gpt-4o");
}

#[test]
fn pick_deployment_with_cooled_warm_mix() {
    let router = default_router();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-canary"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-stable"),
    ];
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    let ctx = RequestContext::new("gpt-4o");
    let picked = router
        .pick_deployment(&deps, &ctx)
        .expect("should pick stable");
    assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-stable");
}

#[test]
fn fallbacks_for_multiple_returns_all_in_order() {
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec![
                "gpt-4o-mini".to_string(),
                "claude".to_string(),
                "gemini".to_string(),
            ],
        }],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let fbs = router.fallbacks_for("gpt-4o");
    assert_eq!(fbs, vec!["gpt-4o-mini", "claude", "gemini"]);
}

#[test]
fn context_window_fallbacks_for_multiple_returns_all() {
    let config = RouterConfig {
        context_window_fallbacks: vec![FallbackRule {
            primary: "gpt-3.5-turbo".to_string(),
            fallbacks: vec!["gpt-4o".to_string(), "claude-haiku".to_string()],
        }],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let cwf = router.context_window_fallbacks_for("gpt-3.5-turbo");
    assert_eq!(cwf, vec!["gpt-4o", "claude-haiku"]);
}

#[test]
fn model_group_alias_expands_with_duplicates_removes_them() {
    let mut aliases = std::collections::HashMap::new();
    aliases.insert(
        "gpt-4".to_string(),
        vec!["gpt-4o".to_string(), "gpt-4o".to_string()],
    );
    let config = RouterConfig {
        model_group_alias: aliases,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [make_dep("gpt-4o")];
    let picked = router.deployments_for_model(&deps, "gpt-4");
    assert_eq!(
        picked.len(),
        1,
        "duplicate alias targets must produce one deployment"
    );
}

// ===========================================================================
// Reference router behavior tests.
//
// The reference router decides cooldowns per-deployment via traffic-threshold
// heuristics (`_should_cooldown_deployment`) and ships prompt-caching routing.
// oxllm's router is intentionally simpler: cooldowns are explicit
// (`mark_cooldown`) and decided by the caller, retry policy is per-status,
// and prompt-caching's "stick to one deployment" property is served by the
// `SessionAffinity` strategy. These tests port the reference *behaviors* that
// map onto oxllm's actual surface, cross-referencing the source test.
// ===========================================================================

// --- Cooldown edge cases: ports of test_router_cooldown_utils.py ----------

/// Rate-limit cooldown behavior:
/// a 429 rate-limit error triggers cooldown, and the cooled deployment is
/// excluded from routing until the cooldown expires.
#[test]
fn cooldown_429_rate_limit_excludes_deployment() {
    let router = default_router();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    // Simulate a 429 by marking the first deployment cooled.
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(
        router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&deps[0])),
        "429 should leave the deployment cooling"
    );
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        let idx = deps.iter().position(|d| std::ptr::eq(d, picked)).unwrap();
        assert_eq!(idx, 1, "cooled (rate-limited) deployment must be skipped");
    }
}

/// Authentication-failure cooldown behavior:
/// a 401 auth error can trigger cooldown just like a rate-limit error, and
/// the cooled deployment stays out of rotation.
#[test]
fn cooldown_401_auth_error_excludes_deployment() {
    let router = default_router();
    let deps = [
        make_dep_with_model("claude", "anthropic/claude-3-a"),
        make_dep_with_model("claude", "anthropic/claude-3-b"),
    ];
    let ctx = RequestContext::new("claude");
    // Simulate a 401 (bad/expired key) by marking the second deployment.
    router.mark_cooldown(&router.cooldown_key(&deps[1]));
    assert!(
        router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&deps[1]))
    );
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        let idx = deps.iter().position(|d| std::ptr::eq(d, picked)).unwrap();
        assert_eq!(idx, 0, "auth-failed (cooled) deployment must be skipped");
    }
}

/// Rate-limit and low-traffic cooldown behavior:
/// cooldown is an explicit decision in oxllm — a transient 500 on a healthy
/// deployment must NOT auto-cool it. Only an explicit `mark_cooldown` does.
/// This pins the contract that oxllm never silently cools on upstream errors.
#[test]
fn server_error_500_does_not_auto_cooldown() {
    let router = default_router();
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    // No explicit mark → the deployment stays available despite a "500".
    assert!(
        !router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&deps[0]))
    );
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

/// Status-code conversion behavior: status codes are
/// normalized so the retry policy can be keyed by status. oxllm keys
/// `retry_policy` by the decimal status string; configured statuses resolve,
/// and unconfigured/invalid ones fall back to `num_retries` (the analog of
/// the reference casting `"invalid"` → `500` → default behavior).
#[test]
fn retry_policy_status_lookup_429_500_401() {
    let mut policy = std::collections::HashMap::new();
    policy.insert("429".to_string(), 5);
    policy.insert("500".to_string(), 3);
    policy.insert("401".to_string(), 0);
    let config = RouterConfig {
        num_retries: 2,
        retry_policy: policy,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    // 429 (rate limit) and 500 (server error) are retryable.
    assert_eq!(router.retries_for_status(429), 5);
    assert_eq!(router.retries_for_status(500), 3);
    // 401 (auth) is configured as non-retryable.
    assert_eq!(router.retries_for_status(401), 0);
    // An unconfigured status (e.g. 404) falls back to num_retries — the
    // analog of the reference defaulting an unrecognized status.
    assert_eq!(router.retries_for_status(404), 2);
}

/// Cooldown-policy activation behavior: when cooldowns are
/// "disabled" (zero cooldown window), a marked deployment is immediately
/// available again. oxllm expresses this via `cooldown_time_secs == 0`.
#[test]
fn cooldown_disabled_zero_window_no_exclusion() {
    let config = RouterConfig {
        cooldown_time_secs: 0,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    // With a zero window, the cooldown is effectively a no-op, so BOTH
    // deployments must remain routable.
    let mut saw_first = false;
    let mut saw_second = false;
    for _ in 0..100 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        let idx = deps.iter().position(|d| std::ptr::eq(d, picked)).unwrap();
        if idx == 0 {
            saw_first = true;
        }
        if idx == 1 {
            saw_second = true;
        }
    }
    assert!(
        saw_first && saw_second,
        "cooldown_time_secs=0 must not exclude any deployment"
    );
}

/// High-traffic failure cooldown behavior: when a
/// deployment is cooled after sustained failures, the router reports it as
/// cooling and routes around it.
#[test]
fn cooldown_marked_then_routed_around() {
    let router = default_router();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(
        router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&deps[0]))
    );
    // Every pick must land on the healthy sibling.
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-b");
    }
}

// --- Router error handling: ports of test_router_handle_error.py ----------

/// No-deployment error behavior: when every
/// deployment for a model is cooling (e.g. all hit a 429 rate limit), the
/// router must yield `None` — the "no deployment available" outcome.
#[test]
fn all_rate_limited_deployments_yield_none() {
    let router = default_router();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-c"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    // Every deployment has been rate-limited into cooldown.
    for d in &deps {
        router.mark_cooldown(&router.cooldown_key(d));
    }
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "all-cooled model group must be unroutable"
    );
}

/// No-deployment error behavior with an empty cooldown list:
/// when NO deployments are cooling but the model simply has no deployments,
/// the router also yields `None` (the empty-cooldown-list scenario).
#[test]
fn no_deployments_yields_none_empty_cooldown() {
    let router = default_router();
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&[], &ctx).is_none());
}

/// Fallback model-group lookup behavior +
/// mock-testing fallback error types: a failing primary resolves to its
/// configured fallback model group, with the right chain selected per error
/// kind (regular vs context-window vs content-policy).
#[test]
fn fallback_resolution_for_rate_limited_primary() {
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["claude".to_string(), "gemini".to_string()],
        }],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    // After gpt-4o is rate-limited, the router offers its fallback chain.
    assert_eq!(
        router.fallbacks_for("gpt-4o"),
        vec!["claude", "gemini"],
        "rate-limited primary must resolve its fallback chain in order"
    );
    // An unknown primary has no fallbacks.
    assert!(router.fallbacks_for("unknown").is_empty());
}

/// Fallback dispatch behavior: the three distinct
/// fallback kinds (regular, context-window, content-policy) each resolve
/// independently. oxllm surfaces them as separate accessor methods.
#[test]
fn three_fallback_kinds_resolve_independently() {
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["claude".to_string()],
        }],
        context_window_fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["gpt-4o-long".to_string()],
        }],
        content_policy_fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["gpt-4o-guarded".to_string()],
        }],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(router.fallbacks_for("gpt-4o"), vec!["claude"]);
    assert_eq!(
        router.context_window_fallbacks_for("gpt-4o"),
        vec!["gpt-4o-long"]
    );
    assert_eq!(
        router.content_policy_fallbacks_for("gpt-4o"),
        vec!["gpt-4o-guarded"]
    );
}

/// No-deployment exception cooldown-list
/// shape: cooling deployments are identified by their cooldown key, and the
/// router can report each one's cooling state (analog of the cooldown_list).
#[test]
fn cooldown_list_per_deployment_key() {
    let router = default_router();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    // Only the second deployment is rate-limited.
    let key = router.cooldown_key(&deps[1]);
    router.mark_cooldown(&key);
    assert!(router.cooldown_store().is_cooling(&key));
    assert!(
        !router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&deps[0]))
    );
    // Routing skips the cooled one.
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        let idx = deps.iter().position(|d| std::ptr::eq(d, picked)).unwrap();
        assert_eq!(idx, 0);
    }
}

/// Cooldown expiry semantics: once the cooldown window
/// elapses, the deployment rejoins rotation. oxllm uses a zero window to
/// express "immediately available again" without flaky real-time waits.
#[test]
fn cooldown_expires_and_rejoins_rotation() {
    let config = RouterConfig {
        cooldown_time_secs: 0,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "expired cooldown must let the deployment rejoin rotation"
    );
}

// --- Prompt-caching decisions: ports of test_router_prompt_caching.py -----
//
// oxllm has no prompt-caching router; the equivalent "keep a conversation on
// one deployment to warm the provider's prompt cache" property is provided by
// the `SessionAffinity` strategy (consistent hash of the session id). These
// tests port the "same cacheable prefix → same deployment" invariant
// onto session affinity.

/// Matching cacheable prefixes route to the same deployment:
/// requests that share a stable affinity key (session id — the analog of the
/// cacheable prefix) MUST route to the same deployment across many calls, so
/// the upstream provider's prompt cache stays warm.
#[test]
fn prompt_caching_same_session_same_deployment() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SessionAffinity,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-c"),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.session_id = Some("prompt-cache-thread-1".to_string());
    let first = router
        .pick_deployment(&deps, &ctx)
        .expect("should pick")
        .oxllm_params
        .model
        .clone();
    // The same session id must keep hitting the same deployment.
    for _ in 0..50 {
        let again = router
            .pick_deployment(&deps, &ctx)
            .expect("should pick")
            .oxllm_params
            .model
            .clone();
        assert_eq!(
            first, again,
            "same session (cacheable prefix) must route to the same deployment"
        );
    }
}

/// Distinct-cache-key cases: DIFFERENT affinity keys (analog
/// of different cacheable prefixes) are allowed to route to different
/// deployments — i.e. session affinity does not collapse all traffic onto one
/// node. With multiple sessions across many trials, more than one deployment
/// must be selected.
#[test]
fn prompt_caching_distinct_sessions_can_split() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SessionAffinity,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let mut distinct_targets = std::collections::HashSet::new();
    for i in 0..200u32 {
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.session_id = Some(format!("session-{i}"));
        let picked = router
            .pick_deployment(&deps, &ctx)
            .expect("should pick")
            .oxllm_params
            .model
            .clone();
        distinct_targets.insert(picked);
    }
    assert!(
        distinct_targets.len() > 1,
        "distinct affinity keys should spread across deployments, got {:?}",
        distinct_targets
    );
}

/// Cache-key serialization stability: the affinity
/// decision must be deterministic for a given session id — repeated identical
/// inputs never flip between deployments (stable hashing, not random).
#[test]
fn prompt_caching_affinity_is_deterministic() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SessionAffinity,
        ..RouterConfig::default()
    };
    let r1 = Router::new(config.clone());
    let r2 = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.session_id = Some("deterministic-thread".to_string());
    let a = r1
        .pick_deployment(&deps, &ctx)
        .expect("should pick")
        .oxllm_params
        .model
        .clone();
    let b = r2
        .pick_deployment(&deps, &ctx)
        .expect("should pick")
        .oxllm_params
        .model
        .clone();
    assert_eq!(
        a, b,
        "session affinity must be deterministic across router instances"
    );
}

/// Cacheable-prefix extraction from string content without cache controls:
/// when there is no affinity key (no session id — the analog of "no
/// cache_control"), session affinity falls back to the shuffle strategy and
/// still picks *some* deployment rather than failing.
#[test]
fn prompt_caching_no_session_still_routes() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::SessionAffinity,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o"); // no session_id set
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "no affinity key must fall back to shuffle, not fail"
    );
}

// ---------------------------------------------------------------------------
// Health-check routing
// ---------------------------------------------------------------------------

fn router_with_health_check() -> Router {
    Router::new(RouterConfig {
        enable_health_check_routing: true,
        health_check_staleness_threshold_secs: 60,
        health_check_failure_threshold: 3,
        ..RouterConfig::default()
    })
}

#[test]
fn pre_call_check_names_reflects_registered_checks() {
    let router = default_router();
    assert!(
        router.pre_call_check_names().is_empty(),
        "no checks registered and health routing off → no names"
    );
    let router = router_with_health_check();
    assert_eq!(
        router.pre_call_check_names(),
        vec!["health_check".to_string()],
        "the built-in health check is listed when health routing is on"
    );
}

#[test]
fn health_state_shares_the_routers_health_store() {
    let router = router_with_health_check();
    router
        .health_state()
        .mark_unhealthy("gpt-4o:0", Some("dns".to_string()));
    assert!(
        !router
            .health_store()
            .is_healthy("gpt-4o:0", Duration::from_secs(60)),
        "the Arc handle and the direct view see the same entries"
    );
    assert_eq!(
        router.health_state().reason("gpt-4o:0").as_deref(),
        Some("dns")
    );
}

#[test]
fn health_check_disabled_by_default_does_not_filter() {
    // Existing behavior preserved: without `enable_health_check_routing`,
    // even an explicitly unhealthy deployment stays in the pool.
    let router = default_router();
    let deps = [make_dep("gpt-4o")];
    router.health_store().mark_unhealthy("gpt-4o:0", None);
    let ctx = RequestContext::new("gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "health state must be ignored when health-check routing is disabled"
    );
}

#[test]
fn health_check_unhealthy_deployment_filtered_from_pick() {
    let router = router_with_health_check();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    router.health_store().mark_unhealthy(
        &router.cooldown_key(&deps[0]),
        Some("connect refused".to_string()),
    );
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..10 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-b",
            "unhealthy deployment must never be picked"
        );
    }
}

#[test]
fn health_check_all_unhealthy_safety_net_returns_candidates() {
    let router = router_with_health_check();
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    router
        .health_store()
        .mark_unhealthy(&router.cooldown_key(&deps[0]), None);
    router
        .health_store()
        .mark_unhealthy(&router.cooldown_key(&deps[1]), None);
    let ctx = RequestContext::new("gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "all-unhealthy must fall back to the full candidate list"
    );
}

#[test]
fn health_check_stale_entry_routes_again() {
    let mut config = RouterConfig {
        enable_health_check_routing: true,
        health_check_failure_threshold: 3,
        ..RouterConfig::default()
    };
    // Zero staleness: every entry is instantly stale → fail-open.
    config.health_check_staleness_threshold_secs = 0;
    let router = Router::new(config);
    let deps = [make_dep("gpt-4o")];
    router.health_store().mark_unhealthy("gpt-4o:0", None);
    let ctx = RequestContext::new("gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "stale health entries must be treated as healthy (fail-open)"
    );
}

#[test]
fn health_check_failure_threshold_marks_unhealthy() {
    let router = router_with_health_check();
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");

    assert!(!router.record_connection_failure("gpt-4o:0", "timeout"));
    assert!(!router.record_connection_failure("gpt-4o:0", "timeout"));
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "below the failure threshold the deployment stays in rotation"
    );

    assert!(
        router.record_connection_failure("gpt-4o:0", "timeout"),
        "third consecutive failure must mark unhealthy"
    );
    // The safety net still returns the only candidate, so assert on the
    // health state directly.
    assert!(
        !router
            .health_store()
            .is_healthy("gpt-4o:0", Duration::from_secs(60)),
        "deployment must be unhealthy after the failure threshold"
    );
}

#[test]
fn record_connection_failure_reports_newly_unhealthy_only_on_transition() {
    let router = router_with_health_check();
    assert!(!router.record_connection_failure("gpt-4o:0", "timeout"));
    assert!(!router.record_connection_failure("gpt-4o:0", "timeout"));
    assert!(
        router.record_connection_failure("gpt-4o:0", "timeout"),
        "the threshold-crossing call reports the transition"
    );
    assert!(
        !router.record_connection_failure("gpt-4o:0", "timeout"),
        "already-unhealthy failures must not report newly unhealthy"
    );
    // Recovery resets the episode: the next crossing reports again.
    router.record_connection_success("gpt-4o:0");
    assert!(!router.record_connection_failure("gpt-4o:0", "timeout"));
    assert!(!router.record_connection_failure("gpt-4o:0", "timeout"));
    assert!(
        router.record_connection_failure("gpt-4o:0", "timeout"),
        "a new episode reports the transition again"
    );
}

#[test]
fn health_check_auto_recovers_on_success() {
    let router = router_with_health_check();
    assert!(!router.record_connection_failure("gpt-4o:0", "dns failure"));
    assert!(!router.record_connection_failure("gpt-4o:0", "dns failure"));
    assert!(
        router.record_connection_failure("gpt-4o:0", "dns failure"),
        "threshold of 3 reached → unhealthy"
    );
    assert!(
        !router
            .health_store()
            .is_healthy("gpt-4o:0", Duration::from_secs(60))
    );
    router.record_connection_success("gpt-4o:0");
    assert!(
        router
            .health_store()
            .is_healthy("gpt-4o:0", Duration::from_secs(60)),
        "a successful connection must restore health"
    );
}

#[test]
fn health_check_recording_is_noop_when_disabled() {
    let router = default_router();
    assert!(
        !router.record_connection_failure("gpt-4o:0", "timeout"),
        "disabled health routing must not record failures"
    );
    router.record_connection_success("gpt-4o:0");
    assert!(
        router.health_store().is_empty(),
        "disabled health routing must keep no state"
    );
}

#[test]
fn health_check_state_preserved_across_rebuild() {
    let router = router_with_health_check();
    router.health_store().mark_unhealthy("gpt-4o:0", None);
    let rebuilt = Router::rebuild_from_stores(
        RouterConfig {
            enable_health_check_routing: true,
            ..RouterConfig::default()
        },
        router.cooldown_arc(),
        router.latency_store_arc(),
        router.usage_store_arc(),
        router.health_store_arc(),
        router.bandit_store_arc(),
        router.round_robin_counters_arc(),
    );
    assert!(
        !rebuilt
            .health_store()
            .is_healthy("gpt-4o:0", Duration::from_secs(60)),
        "health state must survive a config hot-reload rebuild"
    );
}

// ---------------------------------------------------------------------------
// Pre-call checks framework
// ---------------------------------------------------------------------------

/// A test check that records the candidate counts it was called with and
/// applies a scripted filter.
#[derive(Debug)]
struct ScriptedCheck {
    check_name: &'static str,
    /// Indices (into the received candidates) to DROP.
    drop_indices: Vec<usize>,
    calls: std::sync::Mutex<Vec<usize>>,
}

impl ScriptedCheck {
    fn new(check_name: &'static str, drop_indices: Vec<usize>) -> Self {
        Self {
            check_name,
            drop_indices,
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn call_log(&self) -> Vec<usize> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl PreCallCheck for ScriptedCheck {
    fn filter_deployments<'a>(
        &self,
        candidates: &[&'a Deployment],
        _ctx: &RequestContext,
    ) -> Result<Vec<&'a Deployment>, crate::Error> {
        self.calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(candidates.len());
        Ok(candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.drop_indices.contains(i))
            .map(|(_, d)| *d)
            .collect())
    }

    fn name(&self) -> &'static str {
        self.check_name
    }
}

/// A test check that always fails.
#[derive(Debug)]
struct FailingCheck;

impl PreCallCheck for FailingCheck {
    fn filter_deployments<'a>(
        &self,
        _candidates: &[&'a Deployment],
        _ctx: &RequestContext,
    ) -> Result<Vec<&'a Deployment>, crate::Error> {
        Err(crate::Error::PreCallCheckFailed(
            "check exploded".to_string(),
        ))
    }

    fn name(&self) -> &'static str {
        "failing_check"
    }
}

#[test]
fn health_check_listed_without_flag_is_silent_noop() {
    // Misconfiguration guard: listing `health_check` in
    // `optional_pre_call_checks` without `enable_health_check_routing: true`
    // makes the check a silent no-op (build emits a warn). Routing must
    // still proceed, and an unhealthy deployment must NOT be filtered out
    // (the flag is off, so health routing is inert).
    let config = RouterConfig {
        optional_pre_call_checks: vec!["health_check".to_string()],
        enable_health_check_routing: false,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    // Even with a health entry marked unhealthy, the disabled flag means
    // no filtering occurs: the deployment remains pickable.
    router
        .health_store()
        .mark_unhealthy("gpt-4o:0", Some("dns failure".to_string()));
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "misconfigured health_check must not break routing"
    );
}

#[test]
fn pre_call_checks_run_in_configured_order() {
    // The pipeline must run prompt_caching → budget_limiter → health_check
    // when listed in that order. Health (router-managed) participates in
    // the ordering; here we verify with two host checks that list order is
    // respected, and that the health check runs at its listed position.
    let recorder = Arc::new(ScriptedCheck::new("prompt_caching", vec![]));
    let budget = Arc::new(ScriptedCheck::new("budget_limiter", vec![]));
    let config = RouterConfig {
        optional_pre_call_checks: vec!["prompt_caching".to_string(), "budget_limiter".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![recorder.clone(), budget.clone()]);
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
    assert_eq!(
        recorder.call_log(),
        vec![2],
        "first check sees all candidates"
    );
    assert_eq!(
        budget.call_log(),
        vec![2],
        "second check sees the first check's output"
    );
}

#[test]
fn duplicate_check_listing_runs_once() {
    // Listing the same check twice in `optional_pre_call_checks` (here via
    // two spellings that normalize to the same name) must execute the
    // check exactly once, at its first listed position — previously a
    // duplicated name ran the (idempotent but wasted) check once per
    // listing on every pick.
    let recorder = Arc::new(ScriptedCheck::new("prompt_caching", vec![]));
    let config = RouterConfig {
        optional_pre_call_checks: vec!["prompt_caching".to_string(), "prompt-caching".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![recorder.clone()]);
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
    assert_eq!(
        recorder.call_log(),
        vec![2],
        "duplicate listing (kebab + snake spellings) must run the check exactly once"
    );
}

#[test]
fn pre_call_check_filters_candidates() {
    // A check that drops the first candidate must steer every pick to the
    // survivor.
    let filter = Arc::new(ScriptedCheck::new("prompt_caching", vec![0]));
    let config = RouterConfig {
        optional_pre_call_checks: vec!["prompt_caching".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![filter]);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..10 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-b");
    }
}

#[test]
fn pre_call_checks_compose_sequentially() {
    // Two checks: the first drops candidate 0, the second drops candidate 0
    // OF ITS INPUT (originally candidate 1) — only candidate 2 survives.
    let first = Arc::new(ScriptedCheck::new("prompt_caching", vec![0]));
    let second = Arc::new(ScriptedCheck::new("budget_limiter", vec![0]));
    let config = RouterConfig {
        optional_pre_call_checks: vec!["prompt_caching".to_string(), "budget_limiter".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![first, second.clone()]);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/a"),
        make_dep_with_model("gpt-4o", "openai/b"),
        make_dep_with_model("gpt-4o", "openai/c"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..10 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/c");
    }
    assert_eq!(
        second.call_log(),
        vec![2; 10],
        "the second check must see the first check's filtered output"
    );
}

#[test]
fn pre_call_checks_absent_is_passthrough() {
    // No checks configured and none injected: every candidate stays
    // eligible (existing behavior).
    let router = default_router();
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    let mut saw_first = false;
    let mut saw_second = false;
    for _ in 0..100 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        if std::ptr::eq(picked, &deps[0]) {
            saw_first = true;
        } else {
            saw_second = true;
        }
    }
    assert!(saw_first && saw_second);
}

#[test]
fn pre_call_check_error_is_fail_open() {
    // A failing check must not break routing: its input flows through
    // unchanged and the pick succeeds.
    let config = RouterConfig {
        optional_pre_call_checks: vec!["failing_check".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![Arc::new(FailingCheck)]);
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    assert!(
        router.pick_deployment(&deps, &ctx).is_some(),
        "a check error must be skipped (fail-open), not propagated"
    );
}

#[test]
fn pre_call_check_filtering_everything_hits_safety_net() {
    // A check that filters out every candidate is treated as a no-op:
    // routing must go on with the check's input. The check records its
    // invocations so we can prove it actually ran (and still didn't
    // empty the pool).
    let filter_all = Arc::new(ScriptedCheck::new("filter_all", vec![0, 1]));
    let config = RouterConfig {
        optional_pre_call_checks: vec!["filter_all".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![filter_all.clone()]);
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    let picked = router
        .pick_deployment(&deps, &ctx)
        .expect("all-filtered must restore the check's input (safety net)");
    assert!(
        deps.iter().any(|d| std::ptr::eq(d, picked)),
        "the picked deployment must come from the restored input list"
    );
    assert_eq!(
        filter_all.call_log(),
        vec![2],
        "the check must have run once over both candidates before the safety net restored them"
    );
}

#[test]
fn empty_pool_with_checks_configured_returns_none() {
    // No candidates at all, with pre-call checks configured: the pipeline
    // must not panic and must yield no pick.
    let check = Arc::new(ScriptedCheck::new("prompt_caching", vec![]));
    let config = RouterConfig {
        enable_health_check_routing: true,
        optional_pre_call_checks: vec!["prompt_caching".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![check]);
    let ctx = RequestContext::new("gpt-4o");
    assert!(
        router.pick_deployment(&[], &ctx).is_none(),
        "empty deployment pool must yield None, not a panic"
    );
}

#[test]
fn pre_call_check_unlisted_name_is_skipped() {
    // A listed name with no matching injected check is skipped silently;
    // the pick proceeds.
    let config = RouterConfig {
        optional_pre_call_checks: vec!["budget_limiter".to_string()],
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
}

#[test]
fn pre_call_check_health_runs_at_listed_position() {
    // With health listed BEFORE a recording check, the recorder sees the
    // health-filtered count; with health listed after, it sees the
    // unfiltered count.
    let recorder_first = Arc::new(ScriptedCheck::new("prompt_caching", vec![]));
    let mut config = RouterConfig {
        enable_health_check_routing: true,
        optional_pre_call_checks: vec!["health_check".to_string(), "prompt_caching".to_string()],
        ..RouterConfig::default()
    };
    config.health_check_staleness_threshold_secs = 60;
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![recorder_first.clone()]);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/a"),
        make_dep_with_model("gpt-4o", "openai/b"),
    ];
    router
        .health_store()
        .mark_unhealthy(&router.cooldown_key(&deps[0]), None);
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.pick_deployment(&deps, &ctx).is_some());
    assert_eq!(
        recorder_first.call_log(),
        vec![1],
        "health listed first → later check sees the health-filtered list"
    );

    let recorder_second = Arc::new(ScriptedCheck::new("prompt_caching", vec![]));
    let config = RouterConfig {
        enable_health_check_routing: true,
        health_check_staleness_threshold_secs: 60,
        optional_pre_call_checks: vec!["prompt_caching".to_string(), "health_check".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![recorder_second.clone()]);
    router
        .health_store()
        .mark_unhealthy(&router.cooldown_key(&deps[0]), None);
    assert!(router.pick_deployment(&deps, &ctx).is_some());
    assert_eq!(
        recorder_second.call_log(),
        vec![2],
        "health listed second → earlier check sees the unfiltered list"
    );
}

#[test]
fn pre_call_check_kebab_names_match_snake_checks() {
    // `budget-limiter` in config must match a check named `budget_limiter`.
    let filter = Arc::new(ScriptedCheck::new("budget_limiter", vec![0]));
    let config = RouterConfig {
        optional_pre_call_checks: vec!["budget-limiter".to_string()],
        ..RouterConfig::default()
    };
    let mut router = Router::new(config);
    router.set_pre_call_checks(vec![filter]);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/a"),
        make_dep_with_model("gpt-4o", "openai/b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..10 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/b");
    }
}

// ---------------------------------------------------------------------------
// Complexity router (end-to-end through Router::pick_deployment)
// ---------------------------------------------------------------------------

fn complexity_test_config() -> RouterConfig {
    RouterConfig {
        routing_strategy: RoutingStrategyKind::ComplexityRouter,
        complexity_router_config: Some(ComplexityRouterConfig {
            tiers: HashMap::from([
                ("simple".to_string(), vec!["gpt-4o-mini".to_string()]),
                ("complex".to_string(), vec!["gpt-4o".to_string()]),
                ("reasoning".to_string(), vec!["o3".to_string()]),
            ]),
            default_model: None,
            tier_boundaries: None,
            escalation_keywords: None,
        }),
        ..RouterConfig::default()
    }
}

fn ctx_with_request_text(model: &str, text: &str) -> RequestContext {
    let mut ctx = RequestContext::new(model);
    ctx.request_text = Some(text.to_string());
    ctx
}

#[test]
fn complexity_router_routes_simple_to_simple_pool() {
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o"), make_dep("o3")];
    let ctx = ctx_with_request_text("auto", "hello");
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "gpt-4o-mini");
    }
}

#[test]
fn complexity_router_routes_complex_to_complex_pool() {
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o"), make_dep("o3")];
    let ctx = ctx_with_request_text(
        "auto",
        "analyze the tradeoffs of microservices vs monoliths",
    );
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "gpt-4o");
    }
}

#[test]
fn complexity_router_routes_reasoning_to_reasoning_pool() {
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o"), make_dep("o3")];
    let ctx = ctx_with_request_text("auto", "prove that P != NP");
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "o3");
    }
}

#[test]
fn complexity_router_falls_back_to_requested_model_group() {
    // No tier pools configured (and no default_model) → the requested
    // model's normal candidate group is used.
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::ComplexityRouter,
        complexity_router_config: Some(ComplexityRouterConfig::default()),
        ..RouterConfig::default()
    });
    let deps = [make_dep("gpt-4o")];
    let ctx = ctx_with_request_text("gpt-4o", "prove that P != NP");
    let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
    assert_eq!(picked.model_name, "gpt-4o");
}

#[test]
fn complexity_router_uses_default_model_when_tier_pool_missing() {
    // The classified tier has no configured pool, but default_model is
    // set → the default model's deployments are used.
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::ComplexityRouter,
        complexity_router_config: Some(ComplexityRouterConfig {
            tiers: HashMap::from([("complex".to_string(), vec!["gpt-4o".to_string()])]),
            default_model: Some("gpt-4o-mini".to_string()),
            tier_boundaries: None,
            escalation_keywords: None,
        }),
        ..RouterConfig::default()
    });
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o")];
    // "hello" → SIMPLE, which has no configured pool → default_model pool.
    let ctx = ctx_with_request_text("auto", "hello");
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "gpt-4o-mini");
    }
}

#[test]
fn complexity_router_cooled_tier_pool_does_not_pick_unrelated_model() {
    // The tier pool is fully cooled and the requested model's group resolves
    // nothing; a healthy unrelated deployment must NOT be served. The chain
    // stops at the requested model's group, yielding None →
    // 404 instead of silently billing a completely different model.
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o")];
    let ctx = ctx_with_request_text("auto", "hello");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "a cooled tier pool must not fall back to an unrelated deployment"
    );
}

#[test]
fn complexity_router_unknown_model_returns_none() {
    // A request for a model with no deployments at all — no tier pool, no
    // default_model, no requested-group match — must resolve to None (404),
    // not to the "all deployments" fallback that would serve an unrelated
    // model.
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o"), make_dep("o3")];
    let ctx = ctx_with_request_text("nope-9000", "hello");
    assert!(
        router.pick_deployment(&deps, &ctx).is_none(),
        "an unknown model must never be answered by an unrelated deployment"
    );
}

#[test]
fn complexity_router_cooled_tier_pool_uses_default_model() {
    // The classified tier's pool is fully cooled and `default_model` is
    // set → the default model's deployments are used.
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::ComplexityRouter,
        complexity_router_config: Some(ComplexityRouterConfig {
            tiers: HashMap::from([("simple".to_string(), vec!["gpt-4o".to_string()])]),
            default_model: Some("gpt-4o-mini".to_string()),
            tier_boundaries: None,
            escalation_keywords: None,
        }),
        ..RouterConfig::default()
    });
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o-mini")];
    let ctx = ctx_with_request_text("auto", "hello");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "gpt-4o-mini");
    }
}

#[test]
fn complexity_router_cooled_tier_pool_falls_back_to_requested_group() {
    // The classified tier's pool is fully cooled, no default_model → the
    // requested model's own group is used (degraded beats none WITHIN the
    // requested group, never outside it).
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o")];
    let ctx = ctx_with_request_text("gpt-4o", "hello");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "gpt-4o");
    }
}

#[test]
fn complexity_router_all_deployments_cooled_returns_none() {
    // The tier pool is cooling and the requested model's group resolves
    // nothing → no pick, same as any other strategy with a fully cooled pool.
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o")];
    let ctx = ctx_with_request_text("auto", "hello");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(router.pick_deployment(&deps, &ctx).is_none());
}

#[test]
fn complexity_router_without_config_uses_requested_model_group() {
    // Strategy selected but no complexity_router_config at all → behave
    // like a normal pick within the requested model's group.
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::ComplexityRouter,
        ..RouterConfig::default()
    });
    let deps = [make_dep("gpt-4o")];
    let ctx = ctx_with_request_text("gpt-4o", "hello");
    let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
    assert_eq!(picked.model_name, "gpt-4o");
}

#[test]
fn complexity_router_group_override_expands_only_for_that_model() {
    // A routing-group override selects complexity routing for one model;
    // other models keep the global strategy.
    let mut groups = HashMap::new();
    groups.insert(
        "auto".to_string(),
        RouterGroupOverride {
            routing_strategy: Some(RoutingStrategyKind::ComplexityRouter),
            ..Default::default()
        },
    );
    let router = Router::new(RouterConfig {
        routing_groups: groups,
        complexity_router_config: Some(ComplexityRouterConfig {
            tiers: HashMap::from([("reasoning".to_string(), vec!["o3".to_string()])]),
            ..Default::default()
        }),
        ..RouterConfig::default()
    });
    let deps = [make_dep("auto"), make_dep("o3")];
    // "auto" requests are complexity-routed into the o3 pool.
    let ctx = ctx_with_request_text("auto", "prove that P != NP");
    for _ in 0..10 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "o3");
    }
    // Direct "o3" requests use the global (simple_shuffle) strategy and
    // stay in their own group.
    let ctx = ctx_with_request_text("o3", "prove that P != NP");
    let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
    assert_eq!(picked.model_name, "o3");
}

#[test]
fn complexity_router_keeps_expanded_alias_and_wildcard_members() {
    // Literal re-filtering inside the strategy used to drop expanded
    // members whenever a direct model name also appeared in the tier. Give
    // the direct candidate zero weight to make that regression deterministic.
    for wildcard in [false, true] {
        let mut expanded = if wildcard {
            make_dep_with_model("*", "openai/*")
        } else {
            make_dep("alias-member")
        };
        expanded.oxllm_params.weight = Some(1);
        let mut direct = make_dep("direct");
        direct.oxllm_params.weight = Some(0);
        let tier_member = if wildcard {
            "openai/on-demand"
        } else {
            "pooled"
        };
        let router = Router::new(RouterConfig {
            routing_strategy: RoutingStrategyKind::ComplexityRouter,
            model_group_alias: HashMap::from([("pooled".into(), vec!["alias-member".into()])]),
            complexity_router_config: Some(ComplexityRouterConfig {
                tiers: HashMap::from([(
                    "reasoning".into(),
                    vec![tier_member.into(), "direct".into()],
                )]),
                ..Default::default()
            }),
            ..RouterConfig::default()
        });
        let deps = [expanded, direct];
        let ctx = ctx_with_request_text("auto", "prove the theorem");
        assert_eq!(router.pick_deployment_index(&deps, &ctx), Some(0));
        router.mark_cooldown(&router.cooldown_key(&deps[0]));
        assert_eq!(router.pick_deployment_index(&deps, &ctx), Some(1));
    }
}

#[test]
fn complexity_candidate_key_cache_does_not_cache_cooldown_state() {
    let router = Router::new(complexity_test_config());
    let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o")];
    let ctx = ctx_with_request_text("gpt-4o", "hello");
    let mut keys = CooldownKeyCache::new(&router, 0);
    let (pool, _) = router.complexity_candidates(&deps, &ctx, &mut keys);
    assert_eq!(pool.len(), 1);
    assert!(std::ptr::eq(pool[0], &deps[0]));

    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    // Reuse the same immutable-key cache, as the expansion and final
    // availability filter do within a pick. Current cooldown state wins.
    let (pool, _) = router.complexity_candidates(&deps, &ctx, &mut keys);
    assert_eq!(pool.len(), 1);
    assert!(std::ptr::eq(pool[0], &deps[1]));
    router.mark_cooldown(&router.cooldown_key(&deps[1]));
    assert!(
        router
            .complexity_candidates(&deps, &ctx, &mut keys)
            .0
            .is_empty()
    );
}

// ---------------------------------------------------------------------------
// Priority-order tiers (oxllm_params.order)
// ---------------------------------------------------------------------------

/// A "gpt-4o" deployment with an optional priority order, identified by a
/// distinct `oxllm_params.model` tag so tests can tell picks apart.
fn make_ordered_dep(tag: &str, order: Option<u32>) -> Deployment {
    let mut d = make_dep("gpt-4o");
    d.oxllm_params.model = format!("openai/{tag}");
    d.oxllm_params.order = order;
    d
}

fn picked_tag(picked: &Deployment) -> &str {
    picked.oxllm_params.model.strip_prefix("openai/").unwrap()
}

#[test]
fn order_tier_picks_lowest_order() {
    let router = default_router();
    let deps = [
        make_ordered_dep("low", Some(2)),
        make_ordered_dep("high", Some(1)),
    ];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..100 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked_tag(picked),
            "high",
            "lower order = higher priority: order 1 must always win over order 2"
        );
    }
}

#[test]
fn order_tier_escalates_when_top_tier_cools() {
    let router = default_router();
    let deps = [
        make_ordered_dep("high", Some(1)),
        make_ordered_dep("low", Some(2)),
    ];
    let ctx = RequestContext::new("gpt-4o");
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    for _ in 0..50 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked_tag(picked),
            "low",
            "cooled top tier must make the next tier eligible"
        );
    }
}

#[test]
fn order_unset_is_last_resort() {
    // Priority-order parity: a deployment WITHOUT `order` only becomes eligible
    // once every ordered deployment is cooling down.
    let router = default_router();
    let deps = [
        make_ordered_dep("first", Some(1)),
        make_ordered_dep("second", Some(2)),
        make_ordered_dep("unset", None),
    ];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..50 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked_tag(picked), "first");
    }
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    for _ in 0..50 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked_tag(picked), "second");
    }
    router.mark_cooldown(&router.cooldown_key(&deps[1]));
    for _ in 0..50 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked_tag(picked),
            "unset",
            "unset order = lowest-priority tier, eligible only after all ordered tiers cool"
        );
    }
}

#[test]
fn order_all_same_order_is_noop() {
    let router = default_router();
    let deps = [
        make_ordered_dep("a", Some(1)),
        make_ordered_dep("b", Some(1)),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        seen.insert(picked_tag(picked));
    }
    assert_eq!(
        seen.len(),
        2,
        "same-order deployments must all stay eligible, got: {seen:?}"
    );
}

#[test]
fn order_unset_everywhere_preserves_behavior() {
    let router = default_router();
    let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        let idx = router
            .pick_deployment_index(&deps, &ctx)
            .expect("should pick");
        seen.insert(idx);
    }
    assert_eq!(
        seen.len(),
        2,
        "no order set anywhere → filter is a no-op, both deployments picked"
    );
}

#[test]
fn order_filter_runs_before_weighted_shuffle() {
    // A heavy weight in a lower-priority tier must not outrank a light
    // weight in the top tier.
    let router = default_router();
    let mut light = make_ordered_dep("light", Some(1));
    light.oxllm_params.weight = Some(1);
    let mut heavy = make_ordered_dep("heavy", Some(2));
    heavy.oxllm_params.weight = Some(1000);
    let deps = [light, heavy];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..100 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked_tag(picked), "light");
    }
}

#[test]
fn order_weighted_shuffle_applies_within_a_tier() {
    // Two order-1 deployments (weights 9 and 1) plus an order-2 heavyweight:
    // the order-2 deployment must never be picked, and the order-1 pair
    // must still split ~90/10.
    let router = default_router();
    let mut big = make_ordered_dep("big", Some(1));
    big.oxllm_params.weight = Some(9);
    let mut small = make_ordered_dep("small", Some(1));
    small.oxllm_params.weight = Some(1);
    let mut heavy = make_ordered_dep("heavy", Some(2));
    heavy.oxllm_params.weight = Some(1000);
    let deps = [big, small, heavy];
    let ctx = RequestContext::new("gpt-4o");
    let mut big_hits = 0usize;
    let total = 10_000usize;
    for _ in 0..total {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_ne!(
            picked_tag(picked),
            "heavy",
            "lower-priority tier must never serve while the top tier is healthy"
        );
        if picked_tag(picked) == "big" {
            big_hits += 1;
        }
    }
    let ratio = big_hits as f64 / total as f64;
    assert!(
        (ratio - 0.9).abs() < 0.05,
        "weighted shuffle must still weight WITHIN the surviving tier (~0.9), got {ratio:.3}"
    );
}

// ---------------------------------------------------------------------------
// default_fallbacks (unknown model group)
// ---------------------------------------------------------------------------

#[test]
fn default_fallbacks_for_wildcard_rule_matches_any_model() {
    let router = Router::new(RouterConfig {
        default_fallbacks: vec![FallbackRule {
            primary: "*".to_string(),
            fallbacks: vec!["model-a".to_string(), "model-b".to_string()],
        }],
        ..RouterConfig::default()
    });
    assert_eq!(
        router.default_fallbacks_for("unknown-model"),
        vec!["model-a", "model-b"]
    );
}

#[test]
fn default_fallbacks_for_exact_rule_wins_over_wildcard() {
    let router = Router::new(RouterConfig {
        default_fallbacks: vec![
            FallbackRule {
                primary: "*".to_string(),
                fallbacks: vec!["global".to_string()],
            },
            FallbackRule {
                primary: "special".to_string(),
                fallbacks: vec!["specific".to_string()],
            },
        ],
        ..RouterConfig::default()
    });
    assert_eq!(router.default_fallbacks_for("special"), vec!["specific"]);
    assert_eq!(router.default_fallbacks_for("other"), vec!["global"]);
}

#[test]
fn default_fallbacks_for_empty_when_unconfigured() {
    let router = default_router();
    assert!(router.default_fallbacks_for("unknown-model").is_empty());
}

#[test]
fn default_fallbacks_empty_exact_rule_falls_through_to_wildcard() {
    // A half-written exact rule (`{special: []}`) must be treated as
    // absent: it falls through to the "*" wildcard chain instead of
    // silently disabling fallback for that model.
    let router = Router::new(RouterConfig {
        default_fallbacks: vec![
            FallbackRule {
                primary: "special".to_string(),
                fallbacks: vec![],
            },
            FallbackRule {
                primary: "*".to_string(),
                fallbacks: vec!["model-a".to_string()],
            },
        ],
        ..RouterConfig::default()
    });
    assert_eq!(router.default_fallbacks_for("special"), vec!["model-a"]);
    assert_eq!(router.default_fallbacks_for("other"), vec!["model-a"]);
}

#[test]
fn routing_group_empty_fallback_rule_falls_through_to_global() {
    // Same shape for the group-override accessors: an empty matching group
    // rule must fall through to the global chain rather than returning an
    // empty (fallback-disabled) list.
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "gpt-4o".to_string(),
        RouterGroupOverride {
            fallbacks: vec![FallbackRule {
                primary: "gpt-4o".to_string(),
                fallbacks: vec![],
            }],
            ..Default::default()
        },
    );
    let config = RouterConfig {
        fallbacks: vec![FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["claude".to_string()],
        }],
        routing_groups: groups,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(
        router.fallbacks_for("gpt-4o"),
        vec!["claude"],
        "an empty group rule must fall through to the global chain"
    );
}

// ---------------------------------------------------------------------------
// Adaptive router (router-level integration)
// ---------------------------------------------------------------------------

/// Adaptive config that engages the bandit immediately (no cold start, no
/// exploration) so assertions on the learned posterior are tight.
fn adaptive_test_config() -> AdaptiveRouterConfig {
    AdaptiveRouterConfig {
        cold_start_min_observations: 0,
        epsilon: 0.0,
        ..AdaptiveRouterConfig::default()
    }
}

fn adaptive_router() -> Router {
    Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::AdaptiveRouter,
        adaptive_router_config: Some(adaptive_test_config()),
        ..RouterConfig::default()
    })
}

/// Two deployments of the same model group with distinct provider models, so
/// their stable cooldown keys (`gpt-4o:openai/gpt-4o-a` / `gpt-4o:openai/gpt-4o-b`)
/// are distinct — mirroring two real backends (e.g. different accounts).
fn adaptive_dep_pair() -> [Deployment; 2] {
    [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ]
}

/// With ε = 0 and a Beta(31, 1) dominant arm vs the Beta(1, 1) prior, the
/// dominant arm wins ~30/31 of samples per pick; 100 picks must land well
/// above 80 for the dominant deployment (mean 96.8, std ≈ 1.8 — the bound
/// is ~9σ below the mean).
fn assert_dominant_share(router: &Router, deps: &[Deployment], ctx: &RequestContext, want: usize) {
    let mut hits = 0usize;
    for _ in 0..100 {
        let picked = router.pick_deployment(deps, ctx).expect("should pick");
        if std::ptr::eq(picked, &deps[want]) {
            hits += 1;
        }
    }
    assert!(
        hits > 80,
        "deployment {want} must dominate picks ({hits}/100)"
    );
}

#[test]
fn adaptive_router_prefers_deployment_with_recorded_successes() {
    let router = adaptive_router();
    let deps = adaptive_dep_pair();
    // Learning arrives through the shared store (the hook's handle), keyed
    // by the router's stable cooldown key (`stable_key`,
    // `{model_name}:{16-hex hash}`).
    let cell = (RequestType::Other, router.cooldown_key(&deps[1]));
    for _ in 0..30 {
        router.bandit_store().record_success(&cell);
    }
    // Empty request text classifies as Other, so the Other cells drive picks.
    let ctx = RequestContext::new("gpt-4o");
    assert_dominant_share(&router, &deps, &ctx, 1);
}

#[test]
fn adaptive_router_cells_are_keyed_by_cooldown_key() {
    // The strategy must key cells by the router's stable cooldown key
    // (`stable_key`, `{model_name}:{16-hex hash}`), so the recorded cell is
    // sampled on
    // pick. If it keyed by anything else, the dominant deployment would miss
    // its cell and picks would stay ~50/50 instead of favoring deployment 0.
    let router = adaptive_router();
    let deps = adaptive_dep_pair();
    let cell = (RequestType::Other, router.cooldown_key(&deps[0]));
    for _ in 0..30 {
        router.bandit_store().record_success(&cell);
    }
    let ctx = RequestContext::new("gpt-4o");
    assert_dominant_share(&router, &deps, &ctx, 0);
}

#[test]
fn adaptive_router_reuses_preset_request_type() {
    let router = adaptive_router();
    let deps = adaptive_dep_pair();
    // The dominant CODE cell sits on deployment 0, the dominant OTHER cell on
    // deployment 1. With the type pre-set to Code, picks must follow the CODE
    // cells — proving the router did not re-classify (empty text would
    // classify as Other and favor deployment 1).
    let code_cell = (RequestType::Code, router.cooldown_key(&deps[0]));
    let other_cell = (RequestType::Other, router.cooldown_key(&deps[1]));
    for _ in 0..30 {
        router.bandit_store().record_success(&code_cell);
        router.bandit_store().record_success(&other_cell);
    }
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.pre_call.adaptive_request_type = Some(RequestType::Code);
    assert_dominant_share(&router, &deps, &ctx, 0);
}

#[test]
fn adaptive_router_bandit_state_preserved_across_rebuild() {
    let router = adaptive_router();
    let deps0 = adaptive_dep_pair();
    let cell = (RequestType::QA, router.cooldown_key(&deps0[1]));
    for _ in 0..30 {
        router.bandit_store().record_success(&cell);
    }
    let rebuilt = Router::rebuild_from_stores(
        RouterConfig {
            routing_strategy: RoutingStrategyKind::AdaptiveRouter,
            adaptive_router_config: Some(adaptive_test_config()),
            ..RouterConfig::default()
        },
        router.cooldown_arc(),
        router.latency_store_arc(),
        router.usage_store_arc(),
        router.health_store_arc(),
        router.bandit_store_arc(),
        router.round_robin_counters_arc(),
    );
    assert_eq!(
        rebuilt.bandit_store().observations(&cell),
        30.0,
        "learned posteriors must survive a config hot-reload rebuild"
    );
    let deps = adaptive_dep_pair();
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.pre_call.adaptive_request_type = Some(RequestType::QA);
    assert_dominant_share(&rebuilt, &deps, &ctx, 1);
}

#[test]
fn prune_stale_store_keys_drops_state_for_removed_deployments() {
    // Simulate a config hot-reload that removes a deployment: the preserved
    // stores must not leak the removed deployment's state forever.
    let router = default_router();
    let keep = make_dep_with_model("gpt-4o", "openai/gpt-4o-a");
    let drop_dep = make_dep_with_model("claude", "anthropic/claude-3");

    // Populate every store with state for both deployments. Cooldown keys are
    // the stable `deployment_key`, so the two deployments never collide even
    // though they share the cooldown/health/bandit stores with latency/usage.
    router.latency_store().record(&stable_key(&keep), 100);
    router.latency_store().record(&stable_key(&drop_dep), 200);
    router.record_in_flight(&stable_key(&keep), 1);
    router.record_in_flight(&stable_key(&drop_dep), 2);
    router
        .cooldown_store()
        .mark(&router.cooldown_key(&keep), Duration::from_secs(60));
    router
        .cooldown_store()
        .mark(&router.cooldown_key(&drop_dep), Duration::from_secs(60));
    router
        .health_store()
        .mark_unhealthy(&router.cooldown_key(&keep), None);
    router
        .health_store()
        .mark_unhealthy(&router.cooldown_key(&drop_dep), None);
    let keep_cell = (RequestType::Other, router.cooldown_key(&keep));
    let drop_cell = (RequestType::Other, router.cooldown_key(&drop_dep));
    router.bandit_store().record_success(&keep_cell);
    router.bandit_store().record_success(&drop_cell);
    // Model-group sizes are tracked per model name (set on every pick).
    router.cooldown_store().set_group_size("gpt-4o", 1);
    router.cooldown_store().set_group_size("claude", 1);
    router.cooldown_store().set_group_size("retired-model", 3);
    assert_eq!(router.cooldown_store().group_size_key_count(), 3);

    // Reload keeping only `keep`.
    router.prune_stale_store_keys(std::slice::from_ref(&keep));

    // Removed deployment's state is gone everywhere; survivor's remains.
    assert!(
        router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&keep))
    );
    assert!(
        !router
            .cooldown_store()
            .is_cooling(&router.cooldown_key(&drop_dep))
    );
    let staleness = Duration::from_secs(60);
    assert!(
        !router
            .health_store()
            .is_healthy(&router.cooldown_key(&keep), staleness)
    );
    assert!(
        router
            .health_store()
            .is_healthy(&router.cooldown_key(&drop_dep), staleness)
    );
    assert_eq!(router.bandit_store().observations(&keep_cell), 1.0);
    assert_eq!(router.bandit_store().observations(&drop_cell), 0.0);
    assert_eq!(router.latency_store().p50(&stable_key(&keep)), Some(100));
    assert_eq!(router.latency_store().p50(&stable_key(&drop_dep)), None);
    assert_eq!(router.usage_store().in_flight(&stable_key(&keep)), 1);
    assert_eq!(router.usage_store().in_flight(&stable_key(&drop_dep)), 0);
    // Model-group sizes for removed/renamed models are swept too; a surviving
    // model's size stays (it re-populates on the next pick regardless).
    assert_eq!(
        router.cooldown_store().group_size_for("retired-model"),
        0,
        "a removed model's group-size entry is swept"
    );
    assert_eq!(
        router.cooldown_store().group_size_for("claude"),
        0,
        "the removed deployment's model group is swept"
    );
    assert_eq!(router.cooldown_store().group_size_for("gpt-4o"), 1);
    assert_eq!(router.cooldown_store().group_size_key_count(), 1);
}

#[test]
fn prune_stale_store_keys_sweeps_round_robin_counters() {
    // The per-model-group round-robin counters share the `model_name` key
    // space as `cooldown.group_sizes`, so a hot-reload that removes a model
    // group must sweep its counter too — otherwise it leaks monotonically
    // under canary churn / wildcard re-indexing.
    use std::sync::atomic::{AtomicU64, Ordering};

    let router = default_router();
    let counters = router.round_robin_counters_arc();
    counters
        .lock()
        .unwrap()
        .insert("gpt-4o".to_string(), Arc::new(AtomicU64::new(7)));
    counters
        .lock()
        .unwrap()
        .insert("retired-canary".to_string(), Arc::new(AtomicU64::new(3)));
    assert_eq!(counters.lock().unwrap().len(), 2);

    let keep = make_dep_with_model("gpt-4o", "openai/gpt-4o-a");
    router.prune_stale_store_keys(std::slice::from_ref(&keep));

    let guard = counters.lock().unwrap();
    assert!(
        guard.contains_key("gpt-4o"),
        "the surviving model group's counter must be retained"
    );
    assert!(
        !guard.contains_key("retired-canary"),
        "the retired model group's counter must be swept"
    );
    assert_eq!(
        guard.get("gpt-4o").unwrap().load(Ordering::Relaxed),
        7,
        "the surviving counter value must be preserved"
    );
    assert_eq!(guard.len(), 1);
}

#[test]
fn round_robin_reload_preserves_alias_and_wildcard_rotation() {
    // Cover a live alias, a normalized wildcard, and an alias whose empty
    // target list resolves through the wildcard fallback. Alias members are
    // deliberately repeated to also pin resolution's deduplicated count.
    let cases = [
        (
            "auto",
            "gpt-4o",
            "openai/gpt-4o",
            vec!["gpt-4o", "gpt-4o", "missing"],
            "auto",
        ),
        ("gpt-4o", "*", "openai/*", Vec::new(), "*"),
        ("auto", "*", "openai/*", vec!["missing"], "auto"),
    ];
    for (requested, model_name, provider_model, alias_members, group_key) in cases {
        let config = RouterConfig {
            routing_strategy: RoutingStrategyKind::RoundRobin,
            model_group_alias: if alias_members.is_empty() {
                HashMap::new()
            } else {
                HashMap::from([(
                    "auto".to_string(),
                    alias_members.into_iter().map(String::from).collect(),
                )])
            },
            ..RouterConfig::default()
        };
        let router = Router::new(config.clone());
        let deps: Vec<Deployment> = (0..2)
            .map(|i| {
                let mut dep = make_dep_with_model(model_name, provider_model);
                dep.oxllm_params.api_base =
                    Some(format!("https://endpoint-{i}.example.invalid/v1"));
                dep
            })
            .collect();
        let ctx = RequestContext::new(requested);
        assert_eq!(router.pick_deployment_index(&deps, &ctx), Some(0));
        let rebuilt = Router::rebuild_from_stores(
            config,
            router.cooldown_arc(),
            router.latency_store_arc(),
            router.usage_store_arc(),
            router.health_store_arc(),
            router.bandit_store_arc(),
            router.round_robin_counters_arc(),
        );
        rebuilt.prune_stale_store_keys(&deps);
        assert_eq!(rebuilt.cooldown_store().group_size_for(group_key), 2);
        assert_eq!(
            rebuilt.pick_deployment_index(&deps, &ctx),
            Some(1),
            "group {group_key}"
        );
        assert_eq!(rebuilt.round_robin_counters_arc().lock().unwrap().len(), 1);
    }
}

#[test]
fn prune_stale_store_keys_removes_retired_and_unresolvable_aliases() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let router = Router::new(RouterConfig {
        model_group_alias: HashMap::from([
            ("active-alias".into(), vec!["gpt-4o".into()]),
            ("orphaned-alias".into(), vec!["removed-model".into()]),
        ]),
        ..RouterConfig::default()
    });
    let counters = router.round_robin_counters_arc();
    for alias in ["active-alias", "retired-alias", "orphaned-alias"] {
        counters
            .lock()
            .unwrap()
            .insert(alias.into(), Arc::new(AtomicU64::new(7)));
        router.cooldown_store().set_group_size(alias, 3);
    }
    router.prune_stale_store_keys(&[make_dep("gpt-4o")]);
    {
        let guard = counters.lock().unwrap();
        assert_eq!(guard.len(), 1);
        assert_eq!(guard["active-alias"].load(Ordering::Relaxed), 7);
    }
    assert_eq!(router.cooldown_store().group_size_for("active-alias"), 1);
    assert_eq!(router.cooldown_store().group_size_for("retired-alias"), 0);
    assert_eq!(router.cooldown_store().group_size_for("orphaned-alias"), 0);

    router.prune_stale_store_keys(&[]);
    assert!(counters.lock().unwrap().is_empty());
    assert_eq!(router.cooldown_store().group_size_key_count(), 0);
}

#[test]
fn cooldown_state_follows_deployment_across_reorder_on_reload() {
    // Cooldown state is keyed by the stable deployment key, so a config
    // hot-reload that REORDERS the deployment list keeps each deployment's
    // cooldown attached to the deployment — the new index-0 occupant must not
    // inherit the old index-0 occupant's cooldown (the positional-key bug).
    let a = make_dep_with_model("gpt-4o", "openai/gpt-4o-a");
    let b = make_dep_with_model("gpt-4o", "openai/gpt-4o-b");
    let router = Router::new(RouterConfig::default());
    // Cool A while it sits at index 0 of the old list `[A, B]`.
    router.mark_cooldown(&router.cooldown_key(&a));
    assert!(router.cooldown_store().is_cooling(&router.cooldown_key(&a)));

    // Reload with the list reordered to `[B, A]`: stores are preserved and
    // swept against the new active set.
    let rebuilt = Router::rebuild_from_stores(
        RouterConfig::default(),
        router.cooldown_arc(),
        router.latency_store_arc(),
        router.usage_store_arc(),
        router.health_store_arc(),
        router.bandit_store_arc(),
        router.round_robin_counters_arc(),
    );
    rebuilt.prune_stale_store_keys(&[b.clone(), a.clone()]);

    // B (the new index-0 occupant) must NOT inherit A's cooldown...
    assert!(
        !rebuilt
            .cooldown_store()
            .is_cooling(&rebuilt.cooldown_key(&b)),
        "B must not inherit the old index-0 slot's cooldown"
    );
    // ...and A keeps its own cooldown: state follows the deployment.
    assert!(
        rebuilt
            .cooldown_store()
            .is_cooling(&rebuilt.cooldown_key(&a)),
        "A must retain its cooldown across the reorder"
    );
    // Routing reflects the same: B is pickable while A stays cooled.
    let reordered = [b.clone(), a.clone()];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        let picked = rebuilt
            .pick_deployment(&reordered, &ctx)
            .expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-b",
            "the reordered B must be served, not A"
        );
    }
}

#[test]
fn adaptive_router_routing_group_override_uses_bandit() {
    // Global strategy is simple_shuffle; the `gpt-4o` group overrides to
    // adaptive. Picks for `gpt-4o` must follow the bandit posterior.
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::SimpleShuffle,
        adaptive_router_config: Some(adaptive_test_config()),
        routing_groups: HashMap::from([(
            "gpt-4o".to_string(),
            RouterGroupOverride {
                routing_strategy: Some(RoutingStrategyKind::AdaptiveRouter),
                ..RouterGroupOverride::default()
            },
        )]),
        ..RouterConfig::default()
    });
    let deps = adaptive_dep_pair();
    let cell = (RequestType::Other, router.cooldown_key(&deps[1]));
    for _ in 0..30 {
        router.bandit_store().record_success(&cell);
    }
    let ctx = RequestContext::new("gpt-4o");
    assert_dominant_share(&router, &deps, &ctx, 1);
}

#[test]
fn adaptive_router_cold_start_picks_without_prior_data() {
    // Default cold-start threshold (20): with no recorded outcomes the
    // pick defers to least-busy — which must still return a candidate.
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::AdaptiveRouter,
        ..RouterConfig::default()
    });
    let deps = adaptive_dep_pair();
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        assert!(router.pick_deployment(&deps, &ctx).is_some());
    }
}

// ===========================================================================
// H4: Rich tag routing — default_tag, tag_regex, exclude patterns
// ===========================================================================

fn make_dep_with_tag_regex(name: &str, model: &str, patterns: Vec<&str>) -> Deployment {
    use crate::deployment::ModelInfo;
    let mut d = make_dep_with_model(name, model);
    let mut extra = serde_json::Map::new();
    extra.insert(
        "tag_regex".to_string(),
        serde_json::Value::Array(patterns.into_iter().map(|s| s.to_string().into()).collect()),
    );
    d.model_info = Some(ModelInfo {
        extra,
        ..Default::default()
    });
    d
}

#[test]
fn default_tag_applies_when_no_tag_in_context() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        default_tag: Some("canary".to_string()),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-canary", vec!["canary".to_string()]),
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-stable", vec!["stable".to_string()]),
    ];
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-canary",
            "default_tag must route untagged requests to the matching deployment"
        );
    }
}

#[test]
fn default_tag_overridden_by_explicit_tag() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        default_tag: Some("canary".to_string()),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-canary", vec!["canary".to_string()]),
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-stable", vec!["stable".to_string()]),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("stable".to_string());
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-stable",
            "explicit tag must override default_tag"
        );
    }
}

#[test]
fn tag_regex_matches_via_router() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tag_regex("gpt-4o", "openai/gpt-4o-prod", vec!["^prod-"]),
        make_dep_with_tag_regex("gpt-4o", "openai/gpt-4o-dev", vec!["^dev-"]),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("prod-blue".to_string());
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-prod",
            "tag_regex must match the request tag"
        );
    }
}

#[test]
fn tag_regex_escaped_dollar_is_literal() {
    // `\$` must match a literal '$' and must not act as an end anchor.
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tag_regex("gpt-4o", "openai/gpt-4o-dollar", vec!["^canary\\$"]),
        make_dep_with_tag_regex("gpt-4o", "openai/gpt-4o-plain", vec!["^canary-x$"]),
    ];
    // A tag ending in '$' matches only the escaped-dollar pattern.
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("canary$".to_string());
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-dollar",
            "\\$ must match a literal dollar, not anchor the end"
        );
    }
    // A tag *not* ending in '$' must not match `^canary\$`.
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("canary-x".to_string());
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-plain",
            "\\$ must not match a tag without a dollar sign"
        );
    }
}

#[test]
fn tag_regex_escaped_dot_is_literal() {
    // `\.` matches a literal dot instead of any char.
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tag_regex("gpt-4o", "openai/gpt-4o-dot", vec!["^v2\\.0$"]),
        make_dep_with_tag_regex("gpt-4o", "openai/gpt-4o-none", vec!["^v2x0$"]),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("v2.0".to_string());
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-dot");
    }
    // `v2x0` must not match `\.` (literal dot required).
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("v2x0".to_string());
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-none");
    }
}

#[test]
fn tag_exclude_drops_matching_deployment() {
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-a", vec!["canary".to_string()]),
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-b", vec!["stable".to_string()]),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("!canary".to_string());
    for _ in 0..20 {
        let picked = router
            .pick_deployment(&deps, &ctx)
            .expect("should pick survivor");
        assert_eq!(
            picked.oxllm_params.model, "openai/gpt-4o-b",
            "exclude tag must drop the matching deployment"
        );
    }
}

#[test]
fn tag_based_backward_compat_exact_match() {
    // Existing exact-match behavior unchanged.
    let config = RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-canary", vec!["canary".to_string()]),
        make_dep_with_tags("gpt-4o", "openai/gpt-4o-stable", vec!["stable".to_string()]),
    ];
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tag = Some("canary".to_string());
    for _ in 0..20 {
        let picked = router.pick_deployment(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-canary");
    }
}

#[test]
fn tag_constraints_reject_before_weighted_failover() {
    let deps = [make_dep_with_tags(
        "gpt-4o",
        "openai/default",
        vec!["default".into()],
    )];
    for weighted_failover in [false, true] {
        let router = Router::new(RouterConfig {
            routing_strategy: RoutingStrategyKind::TagBased,
            enable_weighted_failover: weighted_failover,
            ..RouterConfig::default()
        });
        let mut contexts = Vec::new();
        for tags in [
            vec!["&missing"],
            vec!["!default"],
            vec!["unmatched", "!default"],
        ] {
            let mut ctx = RequestContext::new("gpt-4o");
            ctx.tags = tags.into_iter().map(String::from).collect();
            contexts.push(ctx);
        }
        let mut legacy = RequestContext::new("gpt-4o");
        legacy.tag = Some("!default".into());
        contexts.push(legacy);
        for ctx in contexts {
            let err = router
                .try_pick_deployment(&deps, &ctx)
                .expect_err("hard selector rejection");
            assert!(matches!(err, crate::Error::BadRequest(_)));
            assert!(err.to_string().contains("required/excluded routing tags"));
            assert!(router.try_pick_deployment_index(&deps, &ctx).is_err());
            assert!(router.pick_deployment(&deps, &ctx).is_none());
            assert!(router.pick_deployment_index(&deps, &ctx).is_none());
        }
    }
}

#[test]
fn tag_constraints_apply_before_priority_and_default_fallback() {
    let mut forbidden =
        make_dep_with_tags("gpt-4o", "openai/us", vec!["default".into(), "us".into()]);
    forbidden.oxllm_params.order = Some(0);
    let mut allowed = make_dep_with_tags("gpt-4o", "openai/eu", vec!["eu".into()]);
    allowed.oxllm_params.order = Some(1);
    let deps = [forbidden, allowed];
    let router = Router::new(RouterConfig {
        // Group overrides must enforce the same constraints as the global strategy.
        routing_groups: HashMap::from([(
            "gpt-4o".into(),
            RouterGroupOverride {
                routing_strategy: Some(RoutingStrategyKind::TagBased),
                ..Default::default()
            },
        )]),
        enable_weighted_failover: true,
        ..RouterConfig::default()
    });
    let mut ctx = RequestContext::new("gpt-4o");
    ctx.tags = vec!["unmatched".into(), "&eu".into(), "!us".into()];
    assert_eq!(
        router.try_pick_deployment_index(&deps, &ctx).unwrap(),
        Some(1)
    );

    ctx.tags.clear();
    ctx.tag = Some("&eu".into());
    assert_eq!(
        router.try_pick_deployment_index(&deps, &ctx).unwrap(),
        Some(1)
    );
}

#[test]
fn default_tag_constraints_reject_instead_of_weighted_failover() {
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::TagBased,
        default_tag: Some("!blocked".into()),
        enable_weighted_failover: true,
        ..RouterConfig::default()
    });
    let deps = [make_dep_with_tags(
        "gpt-4o",
        "openai/blocked",
        vec!["blocked".into()],
    )];
    let ctx = RequestContext::new("gpt-4o");
    assert!(router.try_pick_deployment(&deps, &ctx).is_err());
    assert!(router.pick_deployment(&deps, &ctx).is_none());
}

#[test]
fn checked_router_selection_distinguishes_unavailability_from_rejection() {
    let router = default_router();
    let deps = [make_dep("gpt-4o")];
    let ctx = RequestContext::new("gpt-4o");
    assert_eq!(
        router.try_pick_deployment_index(&deps, &ctx).unwrap(),
        Some(0)
    );
    assert!(router.try_pick_deployment(&[], &ctx).unwrap().is_none());
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    assert!(router.try_pick_deployment(&deps, &ctx).unwrap().is_none());
    assert_eq!(router.try_pick_deployment_index(&deps, &ctx).unwrap(), None);
}

// ===========================================================================
// H5: Exception-type retry policies
// ===========================================================================

#[test]
fn exception_retry_policy_rate_limit_covers_429() {
    let config = RouterConfig {
        exception_retry_policy: Some(RetryPolicy {
            rate_limit_retries: Some(5),
            ..Default::default()
        }),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(router.allowed_fails_limit_for(429), Some(5));
    // Other statuses not covered by the policy fall through.
    assert_eq!(router.allowed_fails_limit_for(500), None);
}

#[test]
fn structured_retry_aliases_control_retry_counts_and_explicit_statuses_win() {
    let policy: RetryPolicy = serde_json::from_value(serde_json::json!({
        "RateLimitErrorRetries": 3, "TimeoutErrorRetries": 4,
        "InternalServerErrorRetries": 5, "ContentPolicyViolationErrorRetries": 0,
    }))
    .unwrap();
    let router = Router::new(RouterConfig {
        num_retries: 1,
        retry_policy: HashMap::from([("429".into(), 7)]),
        exception_retry_policy: Some(policy),
        ..RouterConfig::default()
    });
    assert_eq!(router.retries_for_status(429), 7);
    assert_eq!(router.retries_for_status(408), 4);
    assert_eq!(router.retries_for_status(503), 5);
    assert_eq!(router.retries_for_status(529), 5);
    assert_eq!(router.retries_for_status(401), 1);
    assert_eq!(router.retries_for_status(400), 0);
}

#[test]
fn exception_retry_policy_all_categories() {
    let config = RouterConfig {
        exception_retry_policy: Some(RetryPolicy {
            rate_limit_retries: Some(3),
            timeout_retries: Some(2),
            internal_server_error_retries: Some(4),
            content_policy_retries: Some(1),
        }),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(router.allowed_fails_limit_for(429), Some(3));
    assert_eq!(router.allowed_fails_limit_for(408), Some(2));
    assert_eq!(router.allowed_fails_limit_for(500), Some(4));
    assert_eq!(router.allowed_fails_limit_for(400), Some(1));
}

#[test]
fn exception_retry_policy_takes_precedence_over_allowed_fails() {
    let config = RouterConfig {
        exception_retry_policy: Some(RetryPolicy {
            internal_server_error_retries: Some(7),
            ..Default::default()
        }),
        allowed_fails: Some(2),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(
        router.allowed_fails_limit_for(500),
        Some(7),
        "exception policy must win over allowed_fails"
    );
    assert_eq!(
        router.allowed_fails_limit_for(429),
        Some(2),
        "uncovered status falls through to allowed_fails"
    );
}

#[test]
fn exception_retry_policy_falls_back_to_num_retries_when_configured() {
    // When exception_retry_policy is set, uncovered statuses use num_retries.
    let config = RouterConfig {
        num_retries: 4,
        exception_retry_policy: Some(RetryPolicy::default()),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let decision = router.record_failure_and_check("gpt-4o:0", 503, "gpt-4o");
    // 503 is not covered by exception_retry_policy categories, but since the
    // policy IS configured, num_retries acts as the fallback limit.
    match decision {
        AllowedFailDecision::Tolerated { count, limit } => {
            assert_eq!(count, 1);
            assert_eq!(limit, 4, "num_retries must be the fallback limit");
        }
        other => panic!("expected Tolerated, got {other:?}"),
    }
}

#[test]
fn exception_retry_policy_absent_preserves_no_policy_behavior() {
    // Without exception_retry_policy, uncovered statuses → NoPolicy.
    let router = default_router();
    let decision = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert_eq!(
        decision,
        AllowedFailDecision::NoPolicy,
        "no exception policy → uncovered status is not counted"
    );
}

#[test]
fn exception_retry_policy_rate_limit_retries_in_record_failure() {
    // Exclusive `>` semantics: `rate_limit_retries: 2` tolerates 2 failures,
    // cools on the 3rd.
    let config = RouterConfig {
        exception_retry_policy: Some(RetryPolicy {
            rate_limit_retries: Some(2),
            ..Default::default()
        }),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let d1 = router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o");
    assert_eq!(d1, AllowedFailDecision::Tolerated { count: 1, limit: 2 });
    let d2 = router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o");
    assert_eq!(d2, AllowedFailDecision::Tolerated { count: 2, limit: 2 });
    let d3 = router.record_failure_and_check("gpt-4o:0", 429, "gpt-4o");
    assert_eq!(d3, AllowedFailDecision::LimitReached { count: 3, limit: 2 });
}

#[test]
fn exception_retry_policy_internal_server_covers_gateway_failures_and_overloads() {
    // The reference allowed-fails vocabulary covers 500/502/503 via separate
    // `*AllowedFails` fields; oxllm's single 5xx category must cover
    // 502/503/504 so an operator who set `internal_server_error_retries: 5`
    // to tolerate 5xx flapping gets it for the most common flapping codes,
    // not just 500 (previously they silently fell to the flat num_retries
    // tier).
    let config = RouterConfig {
        exception_retry_policy: Some(RetryPolicy {
            internal_server_error_retries: Some(5),
            ..Default::default()
        }),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    for status in [500u16, 502, 503, 504, 529] {
        assert_eq!(
            router.allowed_fails_limit_for(status),
            Some(5),
            "{status} must resolve through internal_server_error_retries"
        );
    }
    // A category-less 4xx is unaffected by this policy.
    assert_eq!(router.allowed_fails_limit_for(403), None);
}

#[test]
fn exception_policy_num_retries_fallback_excludes_client_errors() {
    // The reference `_is_cooldown_required` (cooldown_handlers.py:233-247) skips
    // cooldown evaluation for 4xx other than 429/401/408/404: a generic
    // client error is the client's fault and must not cool a healthy
    // deployment (which would then cascade the abusive client across every
    // deployment in the group — the exact cascade the percentage path's
    // client-error exclusion exists to prevent). The `num_retries`
    // fallback tier must apply the same exclusion.
    let config = RouterConfig {
        num_retries: 2,
        exception_retry_policy: Some(RetryPolicy::default()),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    for status in [400u16, 403, 422] {
        let key = format!("gpt-4o:0:{status}");
        for _ in 0..10 {
            assert_eq!(
                router.record_failure_and_check(&key, status, "gpt-4o"),
                AllowedFailDecision::NoPolicy,
                "client-error {status} must never count toward cooldown via \
                 the num_retries fallback tier"
            );
        }
        assert_eq!(
            router.cooldown_store().failure_count(&key, status),
            0,
            "client-error {status} must not be recorded at all"
        );
    }
    // Cooldown-worthy statuses (server errors incl. 503, plus 429/408/401/
    // 404) still count under the num_retries fallback: tolerated
    // `num_retries` times, LimitReached on the next (exclusive `>`).
    for status in [429u16, 408, 401, 404, 500, 503] {
        let key = format!("gpt-4o:0:{status}");
        assert_eq!(
            router.record_failure_and_check(&key, status, "gpt-4o"),
            AllowedFailDecision::Tolerated { count: 1, limit: 2 }
        );
        assert_eq!(
            router.record_failure_and_check(&key, status, "gpt-4o"),
            AllowedFailDecision::Tolerated { count: 2, limit: 2 }
        );
        assert_eq!(
            router.record_failure_and_check(&key, status, "gpt-4o"),
            AllowedFailDecision::LimitReached { count: 3, limit: 2 }
        );
    }
}

#[test]
fn exception_policy_explicit_opt_in_still_counts_400() {
    // An explicit category naming the status (content_policy_retries → 400)
    // is an operator opt-in that overrides the client-error exclusion,
    // mirroring the reference explicit allowed-fails policy check.
    let config = RouterConfig {
        exception_retry_policy: Some(RetryPolicy {
            content_policy_retries: Some(1),
            ..Default::default()
        }),
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    assert_eq!(router.allowed_fails_limit_for(400), Some(1));
    let d = router.record_failure_and_check("gpt-4o:0", 400, "gpt-4o");
    assert_eq!(d, AllowedFailDecision::Tolerated { count: 1, limit: 1 });
    let d = router.record_failure_and_check("gpt-4o:0", 400, "gpt-4o");
    assert_eq!(d, AllowedFailDecision::LimitReached { count: 2, limit: 1 });
}

// ===========================================================================
// H6: Weighted intra-group failover
// ===========================================================================

#[test]
fn weighted_failover_re_picks_when_strategy_returns_none() {
    struct DeclinesPick;
    impl RoutingStrategy for DeclinesPick {
        fn pick<'a>(
            &self,
            _deps: &[&'a Deployment],
            _ctx: &RequestContext,
        ) -> Option<&'a Deployment> {
            None
        }
    }
    let mut deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    deps[0].oxllm_params.weight = Some(0);
    deps[1].oxllm_params.weight = Some(1);
    let ctx = RequestContext::new("gpt-4o");
    for enabled in [false, true] {
        let mut router = Router::new(RouterConfig {
            enable_weighted_failover: enabled,
            ..RouterConfig::default()
        });
        router.strategy = Box::new(DeclinesPick);
        assert_eq!(
            router.try_pick_deployment_index(&deps, &ctx).unwrap(),
            enabled.then_some(1)
        );
    }
}

#[test]
fn weighted_failover_disabled_by_default() {
    let config = RouterConfig::default();
    assert!(!config.enable_weighted_failover);
}

#[test]
fn weighted_failover_picks_from_remaining_after_cooldown() {
    // With weighted failover on, when one deployment is cooled, the router
    // still picks from the remaining same-group deployment.
    let config = RouterConfig {
        enable_weighted_failover: true,
        ..RouterConfig::default()
    };
    let router = Router::new(config);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    router.mark_cooldown(&router.cooldown_key(&deps[0]));
    let ctx = RequestContext::new("gpt-4o");
    for _ in 0..20 {
        let picked = router
            .pick_deployment(&deps, &ctx)
            .expect("should pick survivor");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-b");
    }
}

// ===========================================================================
// H8: Percentage-based cooldown
// ===========================================================================

fn router_with_percentage_threshold(threshold: f64, min_requests: u32) -> Router {
    Router::new(RouterConfig {
        failure_threshold_percent: Some(threshold),
        failure_threshold_min_requests: Some(min_requests),
        ..RouterConfig::default()
    })
}

#[test]
fn percentage_cooldown_cools_when_failure_ratio_exceeds_threshold() {
    let router = router_with_percentage_threshold(0.5, 4);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    // Prime the group size by picking once.
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    // 3 failures out of 3 → 100% > 50% threshold, but min_requests is 4.
    // Need at least 4 requests before the percentage triggers.
    for _ in 0..3 {
        router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    }
    // 3 failures: below min_requests → no cooldown yet.
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    // 4th failure: total >= 4 and ratio = 4/4 = 1.0 > 0.5 → LimitReached.
    assert!(
        matches!(d, AllowedFailDecision::LimitReached { .. }),
        "percentage threshold crossed → LimitReached, got {d:?}"
    );
}

#[test]
fn percentage_cooldown_respects_min_requests() {
    let router = router_with_percentage_threshold(0.5, 10);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    // 3 failures out of 3 → 100% ratio, but min_requests=10 not reached.
    for _ in 0..3 {
        let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
        assert!(
            !matches!(d, AllowedFailDecision::LimitReached { .. }),
            "below min_requests → must not cool on percentage"
        );
    }
}

#[test]
fn percentage_cooldown_single_deployment_safety() {
    // A model group with only ONE deployment must never cool on percentage.
    let router = router_with_percentage_threshold(0.1, 1);
    let deps = [make_dep_with_model("gpt-4o", "openai/gpt-4o-only")];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    // Even at 100% failure rate, a solo deployment must not be cooled.
    for _ in 0..10 {
        let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
        assert!(
            !matches!(d, AllowedFailDecision::LimitReached { .. }),
            "single-deployment safety: must not cool a solo deployment on percentage"
        );
    }
}

#[test]
fn complexity_tier_pool_does_not_inflate_solo_group_size() {
    // A complexity tier pool that spans model groups must not record the
    // POOL's size as the requested model's group size: `solo-model` is the
    // only deployment of its own group, so the single-deployment percentage
    // safety must engage even though its tier pool also contains
    // `other-group`'s two deployments. Before the fix the recorded size was
    // the pool length (3), disabling the exemption and letting the
    // percentage path cool the solo deployment.
    let router = Router::new(RouterConfig {
        routing_strategy: RoutingStrategyKind::ComplexityRouter,
        complexity_router_config: Some(ComplexityRouterConfig {
            tiers: HashMap::from([(
                "complex".to_string(),
                vec!["solo-model".to_string(), "other-group".to_string()],
            )]),
            ..Default::default()
        }),
        failure_threshold_percent: Some(0.8),
        failure_threshold_min_requests: Some(1),
        ..RouterConfig::default()
    });
    let deps = [
        make_dep_with_model("solo-model", "openai/solo"),
        make_dep_with_model("other-group", "openai/other-a"),
        make_dep_with_model("other-group", "openai/other-b"),
    ];
    let mut ctx = RequestContext::new("solo-model");
    // Scores ~0.46 by default → COMPLEX tier → pool spans both groups.
    ctx.request_text = Some("analyze the tradeoffs of microservices vs monoliths".to_string());
    let _ = router.pick_deployment(&deps, &ctx);

    assert_eq!(
        router.cooldown_store().group_size_for("solo-model"),
        1,
        "the complexity path must record the requested model's OWN group size, \
         not the cross-group tier pool length"
    );
    let solo_key = router.cooldown_key(&deps[0]);
    for _ in 0..5 {
        let d = router.record_failure_and_check(&solo_key, 500, "solo-model");
        assert!(
            !matches!(d, AllowedFailDecision::LimitReached { .. }),
            "solo deployment must stay protected by the single-deployment \
             percentage safety even when routed via a cross-group tier pool, \
             got {d:?}"
        );
    }
}

#[test]
fn percentage_cooldown_success_dilutes_failure_ratio() {
    let router = router_with_percentage_threshold(0.5, 4);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    // 2 failures + 2 successes → 50% ratio, which is NOT > 50% (strict).
    router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    router.clear_failure_counts("gpt-4o:0"); // records success outcome
    router.clear_failure_counts("gpt-4o:0"); // records another success

    let (failures, total) = router.cooldown_store().outcome_counts("gpt-4o:0");
    assert_eq!(failures, 2);
    assert_eq!(total, 4);

    // Next failure: total becomes 5, failures 3 → ratio 0.6 > 0.5 → cool.
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert!(
        matches!(d, AllowedFailDecision::LimitReached { .. }),
        "3/5 = 60% > 50% → should cool, got {d:?}"
    );
}

#[test]
fn percentage_cooldown_exact_boundary_is_tolerated_and_telemetry_is_consistent() {
    // `ratio > threshold` is strict, so a ratio exactly AT the threshold must
    // NOT cool. With threshold 0.8 and 8/10 failures the ratio equals 0.8 (not
    // greater) → Tolerated. This pins the boundary and that the reported
    // `(count, limit)` pair never disagrees with the float decision.
    let router = router_with_percentage_threshold(0.8, 1);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    // 7 failures (intermediate decisions ignored), then 2 successes dilute
    // the window to (7, 9). The next failure lands the window on (8, 10).
    for _ in 0..7 {
        let _ = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    }
    router.clear_failure_counts("gpt-4o:0");
    router.clear_failure_counts("gpt-4o:0");

    // 8th failure → window (8, 10): ratio = 0.8, NOT > 0.8 → Tolerated.
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert!(
        !matches!(d, AllowedFailDecision::LimitReached { .. }),
        "ratio exactly at threshold (8/10 = 0.8, not > 0.8) must be tolerated, got {d:?}"
    );

    // 9th failure → window (9, 11): ratio ≈ 0.818 > 0.8 → LimitReached, and
    // the reported pair must be self-consistent (`count >= limit`).
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    match d {
        AllowedFailDecision::LimitReached { count, limit } => {
            assert_eq!(count, 9);
            assert!(
                count >= limit,
                "LimitReached telemetry must satisfy count >= limit (got count={count}, limit={limit})"
            );
        }
        other => panic!("9/11 ≈ 0.818 > 0.8 → expected LimitReached, got {other:?}"),
    }
}

#[test]
fn percentage_cooldown_ignores_client_errors() {
    // Client-error statuses (4xx, other than 429/408) must NOT move the
    // sliding-window failure ratio: an abusive client spraying 401s at a
    // healthy deployment must not cool it via the percentage path (which
    // would then cascade the bad client across every deployment in the
    // group). Only server-reflecting failures count.
    let router = router_with_percentage_threshold(0.1, 2);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);
    let key = router.cooldown_key(&deps[0]);

    for _ in 0..10 {
        let d = router.record_failure_and_check(&key, 401, "gpt-4o");
        assert!(
            !matches!(d, AllowedFailDecision::LimitReached { .. }),
            "client errors must not trip percentage cooldown, got {d:?}"
        );
    }
    let (failures, total) = router.cooldown_store().outcome_counts(&key);
    assert_eq!(
        (failures, total),
        (0, 0),
        "client errors must not move the sliding-window ratio"
    );
}

#[test]
fn percentage_cooldown_mixed_only_server_side_failures_move_ratio() {
    // A mix of server failures and client errors: the ratio reflects only the
    // server-side failures (2/2), not a diluted 2/4 — so the percentage
    // check stays meaningful and a deployment under real load is still cooled.
    let router = router_with_percentage_threshold(0.5, 2);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);
    let key = router.cooldown_key(&deps[0]);

    router.record_failure_and_check(&key, 500, "gpt-4o");
    router.record_failure_and_check(&key, 500, "gpt-4o");
    router.record_failure_and_check(&key, 401, "gpt-4o");
    router.record_failure_and_check(&key, 401, "gpt-4o");

    let (failures, total) = router.cooldown_store().outcome_counts(&key);
    assert_eq!(
        (failures, total),
        (2, 2),
        "client errors must be excluded from the ratio"
    );
    let d = router.record_failure_and_check(&key, 503, "gpt-4o");
    assert!(
        matches!(d, AllowedFailDecision::LimitReached { .. }),
        "server failures at the threshold must still cool, got {d:?}"
    );
}

#[test]
fn percentage_cooldown_429_still_counts_as_server_reflecting() {
    // 429 (rate limit) is a 4xx but reflects a server-side condition: it
    // must still move the ratio.
    let router = router_with_percentage_threshold(0.5, 2);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);
    let key = router.cooldown_key(&deps[0]);

    let d = router.record_failure_and_check(&key, 429, "gpt-4o");
    assert!(
        !matches!(d, AllowedFailDecision::LimitReached { .. }),
        "1/1 below nothing... (no policy, count only)"
    );
    let (failures, total) = router.cooldown_store().outcome_counts(&key);
    assert_eq!((failures, total), (1, 1));
}

#[test]
fn percentage_cooldown_disabled_by_default() {
    let config = RouterConfig::default();
    assert!(config.failure_threshold_percent.is_none());
    assert!(config.failure_threshold_min_requests.is_none());
}

#[test]
fn percentage_cooldown_two_deployment_group_can_cool() {
    // Sanity: a two-deployment group IS subject to percentage cooldown.
    let router = router_with_percentage_threshold(0.5, 2);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert!(
        matches!(d, AllowedFailDecision::LimitReached { .. }),
        "2/2 = 100% > 50% in a 2-deployment group → cool, got {d:?}"
    );
}

#[test]
fn percentage_cooldown_no_policy_reports_meaningful_count_and_limit() {
    // With ONLY failure_threshold_percent set (no allowed_fails / policy /
    // exception policy), the per-status counter is never recorded, so the
    // percentage path must source count from the sliding window and a limit
    // derived from the threshold — not the misleading `count=0 / limit=1`.
    let router = router_with_percentage_threshold(0.5, 2);
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    match d {
        AllowedFailDecision::LimitReached { count, limit } => {
            assert!(
                count > 0,
                "count must reflect the windowed failures, not the unrecorded per-status counter"
            );
            assert!(
                count >= limit,
                "count ({count}) must reach the threshold-derived limit ({limit})"
            );
        }
        other => panic!("expected LimitReached via percentage, got {other:?}"),
    }
}

#[test]
fn percentage_cooldown_reports_same_domain_count_and_limit_with_policy() {
    // Even when a per-status allowed-fails policy exists (a DIFFERENT
    // counting domain), the percentage LimitReached must report a same-domain
    // (count, limit) pair: count = sliding-window failures (any status), so
    // limit = the threshold-derived sliding-window limit. Reporting the
    // per-status limit here would make `count >= limit` compare across
    // domains and mislead telemetry.
    let router = Router::new(RouterConfig {
        failure_threshold_percent: Some(0.5),
        failure_threshold_min_requests: Some(2),
        allowed_fails: Some(10),
        ..RouterConfig::default()
    });
    let deps = [
        make_dep_with_model("gpt-4o", "openai/gpt-4o-a"),
        make_dep_with_model("gpt-4o", "openai/gpt-4o-b"),
    ];
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&deps, &ctx);

    router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    match d {
        AllowedFailDecision::LimitReached { count, limit } => {
            // Window: 2 failures / 2 total. threshold 0.5 → floor(0.5*2)=1.
            assert_eq!(count, 2, "count is the sliding-window failure count");
            assert_eq!(
                limit, 1,
                "limit is the threshold-derived window limit, not the per-status policy limit (10)"
            );
        }
        other => panic!("expected LimitReached via percentage, got {other:?}"),
    }
}

// ===========================================================================
// H8: CooldownStore sliding-window unit tests
// ===========================================================================

#[test]
fn percentage_cooldown_group_size_refreshes_on_reload_shrink() {
    // After a hot-reload that shrinks a model group from 2 deployments to 1,
    // the single-deployment percentage safety must engage IMMEDIATELY —
    // without depending on a follow-up `pick_deployment` to refresh
    // `group_size`. The stale `group_size == 2` would otherwise let a
    // failing request cool the now-solo deployment via the percentage path
    // in the one-pick window before the next pick re-measures.
    let router = router_with_percentage_threshold(0.5, 5);
    let a = make_dep_with_model("gpt-4o", "openai/gpt-4o-a");
    let b = make_dep_with_model("gpt-4o", "openai/gpt-4o-b");

    // Prime group_size = 2 by picking once from the two-deployment group.
    let ctx = RequestContext::new("gpt-4o");
    let _ = router.pick_deployment(&[a.clone(), b.clone()], &ctx);
    assert_eq!(router.cooldown_store().group_size_for("gpt-4o"), 2);

    // Drive the failure window up but stay under `min_requests` so the
    // percentage path does not fire yet (the host would not have cooled).
    for _ in 0..4 {
        let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
        assert!(
            !matches!(d, AllowedFailDecision::LimitReached { .. }),
            "below min_requests → must not cool on percentage"
        );
    }

    // Hot-reload: B is removed, A survives as the solo deployment. This is
    // the call site that must refresh `group_size` from the live list.
    router.prune_stale_store_keys(std::slice::from_ref(&a));
    assert_eq!(
        router.cooldown_store().group_size_for("gpt-4o"),
        1,
        "prune_stale_store_keys must refresh group_size to the live count"
    );

    // Fire one more failure WITHOUT an intervening pick. Total is now 5
    // (>= min_requests) and the ratio is 5/5 = 1.0 > 0.5, so the percentage
    // path WOULD fire — except the solo-deployment safety engages.
    let d = router.record_failure_and_check("gpt-4o:0", 500, "gpt-4o");
    assert!(
        !matches!(d, AllowedFailDecision::LimitReached { .. }),
        "solo deployment must be protected by group_size != 1 immediately after reload"
    );
}

#[test]
fn cooldown_store_outcome_counts_track_sliding_window() {
    use std::time::Duration;
    let store = cooldown::CooldownStore::new();
    // Record 3 failures + 1 success.
    let _ = store.record_failure_outcome("dep1");
    let _ = store.record_failure_outcome("dep1");
    let _ = store.record_failure_outcome("dep1");
    store.record_success_outcome("dep1");
    let (failures, total) = store.outcome_counts("dep1");
    assert_eq!(failures, 3);
    assert_eq!(total, 4);
    // Unknown key → zero.
    let (f, t) = store.outcome_counts("unknown");
    assert_eq!(f, 0);
    assert_eq!(t, 0);
    let _ = Duration::from_secs(0); // suppress unused import in older toolchains
}

#[test]
fn cooldown_store_group_size_tracking() {
    let store = cooldown::CooldownStore::new();
    assert_eq!(store.group_size_for("gpt-4o"), 0, "unknown → 0");
    store.set_group_size("gpt-4o", 3);
    assert_eq!(store.group_size_for("gpt-4o"), 3);
}

#[test]
fn cooldown_store_clear_outcomes() {
    let store = cooldown::CooldownStore::new();
    let _ = store.record_failure_outcome("dep1");
    assert_eq!(store.outcome_counts("dep1").1, 1);
    store.clear_outcomes("dep1");
    assert_eq!(store.outcome_counts("dep1").1, 0);
}
