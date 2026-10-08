//! Routing knobs, router settings, and per-model routing-group overrides.

use serde::{Deserialize, Serialize};

/// Routing knobs shared by [`super::OxllmSettings`], [`RouterSettings`], and
/// [`RoutingGroupConfig`].
///
/// Routing accepts these keys in any of the three locations; embedding this
/// struct via `#[serde(flatten)]` keeps the YAML shape identical in all
/// three while the Rust side has exactly one definition. Precedence between
/// locations is resolved in `state::router_config::build_router_config`:
/// `router_settings` beats `oxllm_settings` field-by-field (see
/// [`RoutingKnobs::merge`]); per-model `routing_groups` entries are passed
/// through as overrides and win at request time inside `oxllm-core`'s
/// router.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingKnobs {
    /// Routing strategy name (e.g. `"simple_shuffle"`, `"lowest_cost"`).
    #[serde(default)]
    pub routing_strategy: Option<String>,
    /// Number of retries on 5xx/429/timeout.
    #[serde(default)]
    pub num_retries: Option<u32>,
    /// Cap on fallback-chain walks (default 5).
    #[serde(default)]
    pub max_fallbacks: Option<u32>,
    /// Cooldown duration (seconds) for failed deployments. Defaults to 60.
    #[serde(default)]
    pub cooldown_time: Option<u64>,
    /// Fallback model chains.
    #[serde(default)]
    pub fallbacks: Option<serde_json::Value>,
    /// Context-window fallback chains.
    #[serde(default)]
    pub context_window_fallbacks: Option<serde_json::Value>,
    /// Content-policy fallback chains. Same shape as `fallbacks`, but
    /// triggered when the upstream reports a content-policy / content-filter
    /// error.
    #[serde(default)]
    pub content_policy_fallbacks: Option<serde_json::Value>,
    /// Default fallback chain used when the REQUESTED model group is
    /// unknown (no deployments resolve) — distinct from `fallbacks`, which
    /// handles failures of a KNOWN model. Accepts a list of model names
    /// (`["model-a", "model-b"]`, applies to any unknown model) or the dict
    /// form (`[{"original-model": ["model-a"]}]`), same shape as
    /// `fallbacks`.
    #[serde(default)]
    pub default_fallbacks: Option<serde_json::Value>,
    /// Per-HTTP-status max retry counts. Keys are stringified status codes
    /// (e.g. `"429"`, `"500"`); values override the global `num_retries` for
    /// that status.
    #[serde(default)]
    pub retry_policy: Option<std::collections::HashMap<String, u32>>,
    /// Number of failures (any status) tolerated within the cooldown window
    /// before a deployment is cooled. Unset disables the allowed-fails
    /// policy (deployments cool after retries are exhausted, as before).
    #[serde(default)]
    pub allowed_fails: Option<u32>,
    /// Per-HTTP-status failure thresholds before cooling. Keys are
    /// stringified status codes (e.g. `"429"`, `"401"`); a deployment is
    /// cooled once its failure count for that status reaches the limit.
    /// Takes precedence over `allowed_fails` for listed statuses.
    #[serde(default)]
    pub allowed_fails_policy: Option<std::collections::HashMap<String, u32>>,
    /// Default tag used when a request carries no `tag` under the
    /// `tag_based` strategy. When set, untagged requests are treated as if
    /// they carried this tag; `None` (unset) preserves the fallback to
    /// `simple_shuffle` for untagged requests.
    #[serde(default)]
    pub default_tag: Option<String>,
    /// When `true`, a strategy pick that returns `None` (or a deployment
    /// that is no longer available) triggers a weighted (`simple_shuffle`)
    /// re-pick over the remaining same-group candidates before the router
    /// gives up — keeping traffic within the model group as long as any
    /// deployment is healthy. Defaults to `false`.
    #[serde(default)]
    pub enable_weighted_failover: Option<bool>,
    /// Structured per-exception-type retry policy. The named categories
    /// (`rate_limit_retries`, `timeout_retries`,
    /// `internal_server_error_retries`, `content_policy_retries`) provide
    /// per-category retry/failure limits that take precedence over
    /// `allowed_fails_policy` / `allowed_fails` for the matching HTTP status.
    #[serde(default)]
    pub exception_retry_policy: Option<oxllm_core::router::RetryPolicy>,
    /// Percentage-based cooldown threshold: when the ratio of failures to
    /// total requests for a deployment (over a sliding 60-second window)
    /// exceeds this value, the deployment is cooled. Expressed as a fraction
    /// in `[0.0, 1.0]` (e.g. `0.8` = 80% failures). `None` disables
    /// percentage-based cooldown.
    #[serde(default)]
    pub failure_threshold_percent: Option<f64>,
    /// Minimum total requests a deployment must have served within the
    /// sliding window before the percentage threshold is evaluated. Guards
    /// against cooling a deployment on a small sample (e.g. 1/1 = 100%
    /// failures). `None` defaults to `1`.
    #[serde(default)]
    pub failure_threshold_min_requests: Option<u32>,
    /// Budget-throttle percentage in `[0.0, 1.0]`. When set, a scope (key/
    /// team/org/project) whose spend is at or over its `max_budget` has its
    /// effective RPM/TPM limit multiplied by this fraction instead of being
    /// allowed the full rate. The throttle tightens the rate limit; it does
    /// not replace the budget hard-block (enforced separately by `BudgetHook`).
    /// `None` disables budget-based throttling.
    #[serde(default)]
    pub budget_throttle_pct: Option<f64>,
}

impl RoutingKnobs {
    /// Resolve two sets of knobs field-by-field with `group` taking
    /// precedence over `global`: each field falls back to `global` only when
    /// `group` leaves it unset. Used by `build_router_config` for the
    /// `router_settings` (group) over `oxllm_settings` (global) precedence
    /// rule.
    pub fn merge(global: &RoutingKnobs, group: &RoutingKnobs) -> RoutingKnobs {
        RoutingKnobs {
            routing_strategy: group
                .routing_strategy
                .clone()
                .or_else(|| global.routing_strategy.clone()),
            num_retries: group.num_retries.or(global.num_retries),
            max_fallbacks: group.max_fallbacks.or(global.max_fallbacks),
            cooldown_time: group.cooldown_time.or(global.cooldown_time),
            fallbacks: group.fallbacks.clone().or_else(|| global.fallbacks.clone()),
            context_window_fallbacks: group
                .context_window_fallbacks
                .clone()
                .or_else(|| global.context_window_fallbacks.clone()),
            content_policy_fallbacks: group
                .content_policy_fallbacks
                .clone()
                .or_else(|| global.content_policy_fallbacks.clone()),
            default_fallbacks: group
                .default_fallbacks
                .clone()
                .or_else(|| global.default_fallbacks.clone()),
            retry_policy: group
                .retry_policy
                .clone()
                .or_else(|| global.retry_policy.clone()),
            allowed_fails: group.allowed_fails.or(global.allowed_fails),
            allowed_fails_policy: group
                .allowed_fails_policy
                .clone()
                .or_else(|| global.allowed_fails_policy.clone()),
            default_tag: group
                .default_tag
                .clone()
                .or_else(|| global.default_tag.clone()),
            enable_weighted_failover: group
                .enable_weighted_failover
                .or(global.enable_weighted_failover),
            exception_retry_policy: group
                .exception_retry_policy
                .clone()
                .or_else(|| global.exception_retry_policy.clone()),
            failure_threshold_percent: group
                .failure_threshold_percent
                .or(global.failure_threshold_percent),
            failure_threshold_min_requests: group
                .failure_threshold_min_requests
                .or(global.failure_threshold_min_requests),
            budget_throttle_pct: group.budget_throttle_pct.or(global.budget_throttle_pct),
        }
    }

    /// Translate legacy exception names found in `retry_policy` onto the
    /// fields this proxy actually consults.
    ///
    /// Numeric status keys stay. Named keys are removed. The four categories
    /// this proxy models become `exception_retry_policy` fields and, when that
    /// status is not already present, a status-map retry count:
    /// `TimeoutErrorRetries` is 408 only (504 is not a timeout),
    /// `RateLimitErrorRetries` is 429, `ContentPolicyViolationErrorRetries` is
    /// 400, and `InternalServerErrorRetries` is 500/502/503/504.
    /// `BadRequestErrorRetries` and `AuthenticationErrorRetries` are accepted
    /// only at 0, which writes `allowed_fails_policy` for 400/422 and 401/403
    /// and a zero retry count when that status is still unset. Any other value
    /// fails: those names have no retry category of their own, and writing a
    /// non-zero 400 would clobber content-policy retries.
    ///
    /// Returns one note per translated name. The caller logs them.
    pub(crate) fn translate_named_retry_policy(&mut self) -> Result<Vec<String>, String> {
        let Some(mut policy) = self.retry_policy.take() else {
            return Ok(Vec::new());
        };
        for name in ["BadRequestErrorRetries", "AuthenticationErrorRetries"] {
            if let Some(count) = policy.get(name).copied() {
                if count != 0 {
                    self.retry_policy = Some(policy);
                    return Err(format!(
                        "{name}={count} has no retry translation; only 0 is accepted, \
                         and it maps to allowed_fails_policy (400 and 422 for bad \
                         request, 401 and 403 for authentication)"
                    ));
                }
            }
        }

        let mut found = Vec::new();
        for name in [
            "TimeoutErrorRetries",
            "RateLimitErrorRetries",
            "ContentPolicyViolationErrorRetries",
            "InternalServerErrorRetries",
            "BadRequestErrorRetries",
            "AuthenticationErrorRetries",
        ] {
            if let Some(count) = policy.remove(name) {
                found.push((name, count));
            }
        }
        if found.is_empty() {
            self.retry_policy = Some(policy);
            return Ok(Vec::new());
        }
        self.retry_policy = if policy.is_empty() {
            None
        } else {
            Some(policy)
        };

        let mut notes = Vec::new();
        let mut exception = self.exception_retry_policy.clone().unwrap_or_default();
        let mut touched_exception = false;
        let mut allowed = self.allowed_fails_policy.clone().unwrap_or_default();
        let mut touched_allowed = false;
        for (name, count) in found {
            match name {
                "TimeoutErrorRetries" => {
                    let resolved = self.insert_retry_status_if_absent("408", count);
                    exception.timeout_retries = Some(resolved);
                    touched_exception = true;
                    notes.push(format!(
                        "TimeoutErrorRetries={count} -> timeout_retries and retry_policy 408 = {resolved} (504 unchanged)"
                    ));
                }
                "RateLimitErrorRetries" => {
                    exception.rate_limit_retries =
                        Some(self.insert_retry_status_if_absent("429", count));
                    touched_exception = true;
                    notes.push(format!(
                        "RateLimitErrorRetries={count} -> rate_limit_retries and retry_policy 429"
                    ));
                }
                "ContentPolicyViolationErrorRetries" => {
                    exception.content_policy_retries =
                        Some(self.insert_retry_status_if_absent("400", count));
                    touched_exception = true;
                    notes.push(format!(
                        "ContentPolicyViolationErrorRetries={count} -> content_policy_retries and retry_policy 400"
                    ));
                }
                "InternalServerErrorRetries" => {
                    let resolved = ["500", "502", "503", "504"]
                        .into_iter()
                        .map(|code| self.insert_retry_status_if_absent(code, count))
                        .max()
                        .unwrap_or(count);
                    exception.internal_server_error_retries = Some(resolved);
                    touched_exception = true;
                    notes.push(format!(
                        "InternalServerErrorRetries={count} -> internal_server_error_retries and retry_policy 500, 502, 503, 504"
                    ));
                }
                "BadRequestErrorRetries" => {
                    allowed.insert("400".to_string(), 0);
                    allowed.insert("422".to_string(), 0);
                    touched_allowed = true;
                    self.insert_retry_status_if_absent("400", 0);
                    self.insert_retry_status_if_absent("422", 0);
                    notes.push(
                        "BadRequestErrorRetries=0 -> allowed_fails_policy 400 and 422".to_string(),
                    );
                }
                "AuthenticationErrorRetries" => {
                    allowed.insert("401".to_string(), 0);
                    allowed.insert("403".to_string(), 0);
                    touched_allowed = true;
                    self.insert_retry_status_if_absent("401", 0);
                    self.insert_retry_status_if_absent("403", 0);
                    notes.push(
                        "AuthenticationErrorRetries=0 -> allowed_fails_policy 401 and 403"
                            .to_string(),
                    );
                }
                _ => {}
            }
        }
        if touched_exception {
            self.exception_retry_policy = Some(exception);
        }
        if touched_allowed {
            self.allowed_fails_policy = Some(allowed);
        }
        Ok(notes)
    }

    fn insert_retry_status_if_absent(&mut self, status: &str, count: u32) -> u32 {
        let map = self
            .retry_policy
            .get_or_insert_with(std::collections::HashMap::new);
        *map.entry(status.to_string()).or_insert(count)
    }
}

/// The top-level `router_settings` block.
///
/// Most routing knobs are accepted either in `oxllm_settings` or in
/// `router_settings`; the latter is the canonical location in newer configs.
/// `state::router_config::build_router_config` consults both and prefers
/// `router_settings` when a field is set in both places. Unknown keys fall
/// through to `extra` for forward compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterSettings {
    /// Maps a client-requested model name to a list of concrete `model_name`
    /// entries in `model_list`. When a client sends `model: "gpt-4"`, the
    /// router expands the request to all deployments whose `model_name` is
    /// listed under the `gpt-4` alias.
    #[serde(default)]
    pub model_group_alias: Option<std::collections::HashMap<String, Vec<String>>>,
    /// Routing knobs (strategy, retries, cooldown, fallback chains, retry
    /// policy). Mirrors the flattened knobs on [`super::OxllmSettings`] and takes
    /// precedence field-by-field when both are set (see
    /// [`RoutingKnobs::merge`]).
    #[serde(flatten)]
    pub routing: RoutingKnobs,
    /// Per-model routing-group overrides. The key is a model name clients
    /// send; the value overrides the global routing strategy, retry count,
    /// cooldown, and fallback chains for that model only. Models without an
    /// entry here continue to use the global defaults.
    #[serde(default)]
    pub routing_groups: Option<std::collections::HashMap<String, RoutingGroupConfig>>,
    /// Per-provider spend caps, keyed by provider slug (e.g. `"openai"`).
    /// A deployment whose provider's spend counter (`spend:provider:{slug}`)
    /// reaches the cap is skipped in the routing retry loop until the
    /// `time_period` window resets. Unset disables provider-level budgets.
    #[serde(default)]
    pub provider_budget_config:
        Option<std::collections::HashMap<String, oxllm_core::router::ProviderBudgetConfig>>,
    /// Master switch for health-check routing: when `true`, deployments
    /// marked unhealthy (connection-level failures) are filtered out of
    /// deployment selection before the strategy pick. Defaults to `false`.
    #[serde(default)]
    pub enable_health_check_routing: Option<bool>,
    /// How long (seconds) a health entry stays authoritative before the
    /// deployment is treated as healthy again (fail-open). Defaults to 60.
    #[serde(default)]
    pub health_check_staleness_threshold_secs: Option<u64>,
    /// Consecutive connection failures before a deployment is marked
    /// unhealthy. Defaults to 3.
    #[serde(default)]
    pub health_check_failure_threshold: Option<u32>,
    /// Pre-call checks to run (in order) before the routing strategy pick,
    /// e.g. `["prompt_caching", "budget_limiter", "health_check"]`. Checks
    /// are constructed only when the named check is listed here AND its
    /// feature is configured (`prompt_caching_enabled`, router budgets,
    /// `enable_health_check_routing` respectively). Unset means no optional
    /// checks (health routing, when enabled, always runs).
    #[serde(default)]
    pub optional_pre_call_checks: Option<Vec<String>>,
    /// Configuration for the `complexity_router` strategy: per-tier model
    /// pools (`simple` / `medium` / `complex` / `reasoning` → model names),
    /// optional tier-boundary overrides, and escalation keywords.
    #[serde(default)]
    pub complexity_router_config: Option<oxllm_core::router::ComplexityRouterConfig>,
    /// Configuration for the `adaptive_router` strategy: quality/cost
    /// blend weights, epsilon-greedy exploration rate, and the cold-start
    /// observation threshold. Unset uses the documented defaults
    /// ([`oxllm_core::router::AdaptiveRouterConfig::default`]).
    #[serde(default)]
    pub adaptive_router_config: Option<oxllm_core::router::AdaptiveRouterConfig>,
    /// Unknown keys captured for forward compatibility.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Per-model routing-group override as parsed from the YAML config. Any knob
/// set here takes precedence over the corresponding global value when routing
/// the model named by the containing map key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingGroupConfig {
    /// The routing knobs overridden for this model group. Unset knobs fall
    /// back to the global values at request time.
    #[serde(flatten)]
    pub routing: RoutingKnobs,
}
