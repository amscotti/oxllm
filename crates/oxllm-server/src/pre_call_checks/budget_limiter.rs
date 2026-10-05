//! Budget-limiter pre-call check.
//!
//! The pre-call-check form of router budget limiting
//! (`crate::orchestration::budget::check_router_budgets`): drops candidates whose
//! per-deployment spend (`oxllm_params.max_budget`, counter
//! `spend:dep:{deployment_key}`) or per-provider spend
//! (`router_settings.provider_budget_config`, counter
//! `spend:provider:{slug}`) is at or over its cap.
//!
//! The check itself is synchronous and pure: it reads spend values from
//! [`oxllm_core::router::PreCallData::spend_snapshot`], which the host
//! refreshes ahead of the pick via
//! `crate::orchestration::budget::snapshot_router_spends`. A missing snapshot (or
//! a missing counter) reads as zero spend, so every candidate passes —
//! fail-open, matching the fail-open cache-error behavior of the post-pick
//! check.
//!
//! # Staleness
//!
//! The spend snapshot is taken once per request, ahead of the pick, and can
//! be briefly stale: concurrent in-flight requests accrue (or reserve)
//! spend after the snapshot was read. The post-pick
//! `crate::orchestration::budget::reserve_router_budgets` is the second line of
//! defense — it atomically reserves against the live counters, so
//! enforcement stays bounded even when this check's snapshot misses
//! in-flight spend.

use std::collections::HashMap;

use oxllm_core::deployment::Deployment;
use oxllm_core::router::{PreCallCheck, normalize_check_name};
use oxllm_core::router::{ProviderBudgetConfig, RequestContext, deployment_key};

use crate::hooks::spend_keys::{deployment_spend_key, provider_spend_key};
use crate::orchestration::budget::{BudgetDecision, BudgetScope, evaluate_router_budget};

/// The config name of this check (`optional_pre_call_checks` entry).
pub const CHECK_NAME: &str = "budget_limiter";

/// Filters out deployments that are at/over their deployment or provider
/// spend cap.
#[derive(Debug, Clone, Default)]
pub struct BudgetLimiterCheck {
    provider_budget_config: Option<HashMap<String, ProviderBudgetConfig>>,
}

impl BudgetLimiterCheck {
    /// Create the check with the router's provider budget config.
    pub fn new(provider_budget_config: Option<HashMap<String, ProviderBudgetConfig>>) -> Self {
        Self {
            provider_budget_config,
        }
    }

    /// Whether router budget limiting is configured at all: a provider
    /// budget in the router config, or any deployment with `max_budget`
    /// (pass each deployment's `oxllm_params.max_budget`). Single source
    /// of truth gating both check construction
    /// ([`crate::pre_call_checks::build_pre_call_checks`]) and the
    /// spend-snapshot population in `orchestration::chat`, so proxies
    /// without budgets pay nothing.
    pub fn is_configured(
        deployment_max_budgets: impl IntoIterator<Item = Option<f64>>,
        provider_budget_config: &Option<HashMap<String, ProviderBudgetConfig>>,
    ) -> bool {
        provider_budget_config.is_some()
            || deployment_max_budgets.into_iter().any(|mb| mb.is_some())
    }

    /// Whether a single deployment is at/over its deployment or provider
    /// cap under the given spend snapshot.
    fn over_budget(&self, deployment: &Deployment, ctx: &RequestContext) -> bool {
        let snapshot = &ctx.pre_call.spend_snapshot;
        let dep_key = deployment_key(deployment);
        let dep_counter = deployment.oxllm_params.max_budget.map(|limit| {
            let spend = snapshot
                .get(&deployment_spend_key(&dep_key))
                .copied()
                .unwrap_or(0.0);
            (dep_key.as_str(), spend, limit)
        });
        let provider_slug = deployment.provider_slug();
        let provider_counter = self
            .provider_budget_config
            .as_ref()
            .and_then(|cfg| cfg.get(provider_slug))
            .map(|budget| {
                let spend = snapshot
                    .get(&provider_spend_key(provider_slug))
                    .copied()
                    .unwrap_or(0.0);
                (provider_slug, spend, budget.budget_limit)
            });
        match evaluate_router_budget(dep_counter, provider_counter) {
            BudgetDecision::Allowed => false,
            BudgetDecision::Exceeded {
                scope,
                id,
                spend,
                limit,
            } => {
                match scope {
                    BudgetScope::Deployment => tracing::debug!(
                        event = "budget_exceeded",
                        scope = scope.as_str(),
                        id = %id,
                        spend = spend,
                        limit = limit,
                        "deployment budget exceeded; filtered from candidates"
                    ),
                    BudgetScope::Provider => tracing::debug!(
                        event = "budget_exceeded",
                        scope = scope.as_str(),
                        id = %id,
                        spend = spend,
                        limit = limit,
                        "provider budget exceeded; filtered from candidates"
                    ),
                }
                true
            }
        }
    }
}

impl PreCallCheck for BudgetLimiterCheck {
    fn filter_deployments<'a>(
        &self,
        candidates: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Result<Vec<&'a Deployment>, oxllm_core::Error> {
        Ok(candidates
            .iter()
            .filter(|d| !self.over_budget(d, ctx))
            .copied()
            .collect())
    }

    fn name(&self) -> &'static str {
        CHECK_NAME
    }
}

/// Normalize a configured check name and compare it to [`CHECK_NAME`]
/// (accepts kebab-case spellings).
pub fn is_budget_limiter_name(name: &str) -> bool {
    normalize_check_name(name) == CHECK_NAME
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;
    use std::sync::Arc;

    fn make_dep(name: &str, model: &str) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn ctx_with_spends(spends: &[(&str, f64)]) -> RequestContext {
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.pre_call.spend_snapshot = Arc::new(
            spends
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect::<HashMap<_, _>>(),
        );
        ctx
    }

    #[test]
    fn no_budgets_configured_passes_all() {
        let check = BudgetLimiterCheck::new(None);
        let deps = [make_dep("gpt-4o", "openai/gpt-4o")];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        let ctx = RequestContext::new("gpt-4o");
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("infallible");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn deployment_over_budget_is_filtered() {
        let check = BudgetLimiterCheck::new(None);
        let mut capped = make_dep("gpt-4o", "openai/gpt-4o");
        capped.oxllm_params.max_budget = Some(10.0);
        let other = make_dep("gpt-4o", "openai/gpt-4o-b");
        let deps = [capped, other];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        // `deployment_key` is `{model_name}:{oxllm_params.model}`.
        let ctx = ctx_with_spends(&[("spend:dep:gpt-4o:openai/gpt-4o", 10.0)]);
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("infallible");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].oxllm_params.model, "openai/gpt-4o-b");
    }

    #[test]
    fn deployment_under_budget_is_kept() {
        let check = BudgetLimiterCheck::new(None);
        let mut capped = make_dep("gpt-4o", "openai/gpt-4o");
        capped.oxllm_params.max_budget = Some(10.0);
        let deps = [capped];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_spends(&[("spend:dep:gpt-4o:openai/gpt-4o", 9.99)]);
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("infallible");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn provider_over_budget_is_filtered() {
        let budgets = HashMap::from([(
            "openai".to_string(),
            ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let check = BudgetLimiterCheck::new(Some(budgets));
        let openai_dep = make_dep("gpt-4o", "openai/gpt-4o");
        let anthropic_dep = make_dep("claude", "anthropic/claude-3");
        let deps = [openai_dep, anthropic_dep];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_spends(&[("spend:provider:openai", 150.0)]);
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("infallible");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].oxllm_params.model, "anthropic/claude-3");
    }

    #[test]
    fn missing_snapshot_reads_as_zero_spend() {
        // No snapshot populated (e.g. a handler that does not resolve
        // spends): the check must fail open and keep every candidate.
        let budgets = HashMap::from([(
            "openai".to_string(),
            ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let check = BudgetLimiterCheck::new(Some(budgets));
        let mut capped = make_dep("gpt-4o", "openai/gpt-4o");
        capped.oxllm_params.max_budget = Some(10.0);
        let deps = [capped];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        let ctx = RequestContext::new("gpt-4o");
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("infallible");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn check_name_matching_accepts_kebab_case() {
        assert!(is_budget_limiter_name("budget_limiter"));
        assert!(is_budget_limiter_name("budget-limiter"));
        assert!(!is_budget_limiter_name("prompt_caching"));
        let check = BudgetLimiterCheck::new(None);
        assert_eq!(check.name(), "budget_limiter");
    }

    #[test]
    fn is_configured_gate() {
        assert!(!BudgetLimiterCheck::is_configured(
            std::iter::empty(),
            &None
        ));
        assert!(!BudgetLimiterCheck::is_configured([None, None], &None));
        assert!(BudgetLimiterCheck::is_configured([Some(1.0)], &None));
        assert!(BudgetLimiterCheck::is_configured([None, Some(2.0)], &None));
        let budgets = HashMap::from([(
            "openai".to_string(),
            ProviderBudgetConfig {
                budget_limit: 1.0,
                time_period: "1d".to_string(),
            },
        )]);
        assert!(BudgetLimiterCheck::is_configured(
            std::iter::empty(),
            &Some(budgets)
        ));
    }
}
