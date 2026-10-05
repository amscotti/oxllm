//! Pre-call routing checks.
//!
//! This module has two layers:
//!
//! - **Async request-flow logic** that runs in the orchestration loop:
//!   [`prompt_caching`] pin lookups/writes and
//!   `crate::orchestration::budget::check_router_budgets` spend enforcement.
//! - **Synchronous [`PreCallCheck`] filter implementations** for the
//!   router's pre-call check pipeline, built by [`build_pre_call_checks`]
//!   from `router_settings.optional_pre_call_checks`. The filters read
//!   host-pre-resolved data from
//!   [`oxllm_core::router::PreCallData`] so the router itself stays pure.
//!
//! A check is constructed when its name is listed AND its feature is
//! configured:
//!
//! - `"prompt_caching"` — requires `oxllm_settings.prompt_caching_enabled`.
//! - `"budget_limiter"` — requires `provider_budget_config` or at least one
//!   deployment with `max_budget`.
//! - `"health_check"` — router-managed: the router itself builds and
//!   positions the built-in health check when
//!   `enable_health_check_routing` is set (listed or not); nothing is
//!   constructed here for it.
//!
//! Exception: `prompt_caching` predates the optional-checks list, so
//! [`ensure_prompt_caching_listed`] appends it to the effective
//! `optional_pre_call_checks` whenever `prompt_caching_enabled` is true —
//! the router runs only *listed* checks, and the pre-call check pipeline
//! is the sole prompt-cache pin path.

pub mod budget_limiter;
pub(crate) mod image_inlining;
pub mod prompt_caching;

use std::sync::Arc;

use oxllm_core::Cache;
use oxllm_core::router::{PreCallCheck, RouterConfig, normalize_check_name};

use crate::config::ProxyConfig;

/// Append `prompt_caching` to the effective `optional_pre_call_checks`
/// when `oxllm_settings.prompt_caching_enabled` is true and the check
/// is not already listed (kebab-case spellings count as listed).
///
/// The feature predates the optional-checks list, so a config with only
/// `prompt_caching_enabled: true` must still get pinning. Registering the
/// check is not enough — the router runs only *listed* checks — so this
/// append is what makes the pre-call check pipeline the sole prompt-cache
/// pin path. State build calls this before constructing the router and
/// before [`build_pre_call_checks`] so the router's pick-time name lookup
/// and every other observer of `RouterConfig::optional_pre_call_checks`
/// (e.g. the orchestration loop) see the same effective list.
pub fn ensure_prompt_caching_listed(config: &ProxyConfig, router_config: &mut RouterConfig) {
    if config.oxllm_settings.prompt_caching_enabled == Some(true)
        && !router_config
            .optional_pre_call_checks
            .iter()
            .any(|n| prompt_caching::is_prompt_caching_name(n))
    {
        tracing::info!(
            check = prompt_caching::CHECK_NAME,
            "prompt_caching_enabled is set; added prompt_caching to the effective \
             optional_pre_call_checks — the pre-call check pipeline handles pinning"
        );
        router_config
            .optional_pre_call_checks
            .push(prompt_caching::CHECK_NAME.to_string());
    }
}

/// Build the host-side pre-call checks from
/// `router_settings.optional_pre_call_checks`.
///
/// Checks that are both listed and configured are constructed; unknown or
/// unconfigured names are skipped with a warning. Call
/// [`ensure_prompt_caching_listed`] first (state build does) so a
/// `prompt_caching` check that is enabled but unlisted is still built —
/// the router runs only listed checks, and the pipeline is the sole
/// prompt-cache pin path. The returned vector is injected into the router
/// via [`oxllm_core::router::Router::set_pre_call_checks`]; execution
/// order follows the config list, not this vector's order.
pub fn build_pre_call_checks(
    config: &ProxyConfig,
    router_config: &RouterConfig,
    cache: Arc<dyn Cache>,
) -> Vec<Arc<dyn PreCallCheck>> {
    let mut checks: Vec<Arc<dyn PreCallCheck>> = Vec::new();
    for name in &router_config.optional_pre_call_checks {
        match normalize_check_name(name).as_str() {
            prompt_caching::CHECK_NAME => {
                if config.oxllm_settings.prompt_caching_enabled == Some(true) {
                    checks.push(Arc::new(prompt_caching::PromptCachingCheck::new(
                        cache.clone(),
                        config.oxllm_settings.prompt_caching_min_tokens,
                    )));
                } else {
                    tracing::warn!(
                        check = %name,
                        "pre-call check listed but prompt_caching_enabled is not set; skipping"
                    );
                }
            }
            budget_limiter::CHECK_NAME => {
                if budget_limiter::BudgetLimiterCheck::is_configured(
                    config.model_list.iter().map(|m| m.oxllm_params.max_budget),
                    &router_config.provider_budget_config,
                ) {
                    checks.push(Arc::new(budget_limiter::BudgetLimiterCheck::new(
                        router_config.provider_budget_config.clone(),
                    )));
                } else {
                    tracing::warn!(
                        check = %name,
                        "pre-call check listed but no router budgets are configured; skipping"
                    );
                }
            }
            "health_check" => {
                // Router-managed: the router builds and positions its own
                // health check whenever enable_health_check_routing is set.
                if !router_config.enable_health_check_routing {
                    tracing::warn!(
                        check = %name,
                        "pre-call check listed but enable_health_check_routing is not set; skipping"
                    );
                }
            }
            other => {
                tracing::warn!(
                    check = %other,
                    "unknown pre-call check name in optional_pre_call_checks; skipping"
                );
            }
        }
    }
    checks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use crate::config::types::{ConfigModel, OxllmSettings, RouterSettings};
    use oxllm_core::deployment::OxllmParams;

    fn cache() -> Arc<dyn Cache> {
        Arc::new(InMemoryCache::new(1_000))
    }

    fn proxy(router_settings: RouterSettings, oxllm_settings: OxllmSettings) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings,
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings,
            extra: serde_json::Map::new(),
        }
    }

    fn router_config(names: &[&str]) -> RouterConfig {
        RouterConfig {
            optional_pre_call_checks: names.iter().map(|s| s.to_string()).collect(),
            ..RouterConfig::default()
        }
    }

    #[test]
    fn no_checks_listed_builds_nothing() {
        let config = proxy(RouterSettings::default(), OxllmSettings::default());
        let checks = build_pre_call_checks(&config, &router_config(&[]), cache());
        assert!(checks.is_empty());
    }

    #[test]
    fn prompt_caching_built_when_listed_and_enabled() {
        let config = proxy(
            RouterSettings::default(),
            OxllmSettings {
                prompt_caching_enabled: Some(true),
                ..Default::default()
            },
        );
        let checks = build_pre_call_checks(&config, &router_config(&["prompt_caching"]), cache());
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].name(), "prompt_caching");
    }

    #[test]
    fn prompt_caching_skipped_when_listed_but_disabled() {
        let config = proxy(RouterSettings::default(), OxllmSettings::default());
        let checks = build_pre_call_checks(&config, &router_config(&["prompt_caching"]), cache());
        assert!(
            checks.is_empty(),
            "prompt_caching requires prompt_caching_enabled"
        );
    }

    #[test]
    fn prompt_caching_not_built_when_enabled_but_unlisted() {
        // Without `ensure_prompt_caching_listed`, an enabled-but-unlisted
        // config builds nothing: the router runs only listed checks, and
        // registration without listing would be dead weight. State build
        // calls the ensure function first (tested below).
        let config = proxy(
            RouterSettings::default(),
            OxllmSettings {
                prompt_caching_enabled: Some(true),
                ..Default::default()
            },
        );
        let checks = build_pre_call_checks(&config, &router_config(&[]), cache());
        assert!(checks.is_empty());
    }

    #[test]
    fn ensure_prompt_caching_listed_appends_when_enabled_and_unlisted() {
        // Configs with `prompt_caching_enabled: true` and no
        // `optional_pre_call_checks` entry must still get pinning via the
        // pipeline: the effective list must name the check (the router
        // runs only listed checks), and build_pre_call_checks must then
        // construct it.
        let config = proxy(
            RouterSettings::default(),
            OxllmSettings {
                prompt_caching_enabled: Some(true),
                ..Default::default()
            },
        );
        let mut rc = router_config(&[]);
        ensure_prompt_caching_listed(&config, &mut rc);
        assert_eq!(
            rc.optional_pre_call_checks,
            vec!["prompt_caching".to_string()],
            "the effective list must contain prompt_caching"
        );
        let checks = build_pre_call_checks(&config, &rc, cache());
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].name(), "prompt_caching");
    }

    #[test]
    fn ensure_prompt_caching_listed_does_not_duplicate_when_listed() {
        let config = proxy(
            RouterSettings::default(),
            OxllmSettings {
                prompt_caching_enabled: Some(true),
                ..Default::default()
            },
        );
        for names in [&["prompt_caching"][..], &["prompt-caching"][..]] {
            let mut rc = router_config(names);
            ensure_prompt_caching_listed(&config, &mut rc);
            assert_eq!(
                rc.optional_pre_call_checks.len(),
                1,
                "already listed ({names:?}) → no duplicate append"
            );
            let checks = build_pre_call_checks(&config, &rc, cache());
            assert_eq!(
                checks.len(),
                1,
                "listed + enabled must build exactly one prompt_caching check"
            );
        }
    }

    #[test]
    fn ensure_prompt_caching_listed_noop_when_disabled() {
        let config = proxy(RouterSettings::default(), OxllmSettings::default());
        let mut rc = router_config(&[]);
        ensure_prompt_caching_listed(&config, &mut rc);
        assert!(rc.optional_pre_call_checks.is_empty());
        let checks = build_pre_call_checks(&config, &rc, cache());
        assert!(checks.is_empty());
    }

    #[test]
    fn budget_limiter_built_when_listed_and_provider_budget_configured() {
        let mut rc = router_config(&["budget_limiter"]);
        rc.provider_budget_config = Some(std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]));
        let config = proxy(RouterSettings::default(), OxllmSettings::default());
        let checks = build_pre_call_checks(&config, &rc, cache());
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].name(), "budget_limiter");
    }

    #[test]
    fn budget_limiter_built_when_listed_and_deployment_max_budget() {
        let mut config = proxy(RouterSettings::default(), OxllmSettings::default());
        config.model_list.push(ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                max_budget: Some(10.0),
                ..Default::default()
            },
            model_info: None,
            extra: serde_json::Map::new(),
        });
        let checks = build_pre_call_checks(&config, &router_config(&["budget_limiter"]), cache());
        assert_eq!(checks.len(), 1);
    }

    #[test]
    fn budget_limiter_skipped_when_listed_but_unconfigured() {
        let config = proxy(RouterSettings::default(), OxllmSettings::default());
        let checks = build_pre_call_checks(&config, &router_config(&["budget_limiter"]), cache());
        assert!(checks.is_empty());
    }

    #[test]
    fn health_check_is_router_managed_not_built_here() {
        let mut rc = router_config(&["health_check"]);
        rc.enable_health_check_routing = true;
        let config = proxy(RouterSettings::default(), OxllmSettings::default());
        let checks = build_pre_call_checks(&config, &rc, cache());
        assert!(
            checks.is_empty(),
            "the health check is constructed by the router itself"
        );
    }

    #[test]
    fn unknown_check_name_skipped() {
        let config = proxy(RouterSettings::default(), OxllmSettings::default());
        let checks = build_pre_call_checks(&config, &router_config(&["not_a_check"]), cache());
        assert!(checks.is_empty());
    }

    #[test]
    fn multiple_checks_built_in_one_pass() {
        let mut rc = router_config(&["prompt_caching", "budget_limiter", "health_check"]);
        rc.provider_budget_config = Some(std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]));
        rc.enable_health_check_routing = true;
        let config = proxy(
            RouterSettings::default(),
            OxllmSettings {
                prompt_caching_enabled: Some(true),
                ..Default::default()
            },
        );
        let checks = build_pre_call_checks(&config, &rc, cache());
        assert_eq!(checks.len(), 2, "prompt_caching + budget_limiter built");
        assert_eq!(checks[0].name(), "prompt_caching");
        assert_eq!(checks[1].name(), "budget_limiter");
    }

    #[test]
    fn kebab_case_names_are_accepted() {
        let config = proxy(
            RouterSettings::default(),
            OxllmSettings {
                prompt_caching_enabled: Some(true),
                ..Default::default()
            },
        );
        let checks = build_pre_call_checks(&config, &router_config(&["prompt-caching"]), cache());
        assert_eq!(checks.len(), 1);
    }
}
