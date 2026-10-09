//! Routing configuration types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The routing strategy selector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategyKind {
    /// Simple random shuffle (default). Picks a random deployment from the
    /// non-cooled-down list.
    #[default]
    SimpleShuffle,
    /// Pick the deployment with the lowest per-token cost.
    LowestCost,
    /// Pick the deployment with the lowest rolling p50 latency.
    LowestLatency,
    /// Pick the deployment with the lowest current TPM/RPM usage.
    UsageBased,
    /// Pick the deployment with the lowest EWMA latency.
    LatencyBased,
    /// Pick the deployment with the highest configured quality score.
    QualityRouter,
    /// Pick deployments based on a `tag` field in the request context. When no
    /// tag is set, falls back to `SimpleShuffle`. The request tag is plumbed
    /// from the `x-oxllm-tag` header at the gateway edge; deployments
    /// advertise matching tags via `model_info.tags`.
    TagBased,
    /// Pick the deployment with the fewest in-flight requests. Falls back to
    /// `SimpleShuffle` when no in-flight data is present.
    LeastBusy,
    /// Stick a session to one deployment via a consistent hash of
    /// `RequestContext::session_id` (typically from the `x-oxllm-session-id`
    /// header). Keeps provider-side prompt caches warm across multi-turn
    /// conversations when multiple deployments share a model group. Falls
    /// back to `SimpleShuffle` when no session id is set.
    SessionAffinity,
    /// Classify request complexity (simple / medium / complex / reasoning)
    /// from keyword scoring over `RequestContext::request_text`, then route
    /// to the model pool configured for that tier in
    /// [`RouterConfig::complexity_router_config`]. Falls back to the
    /// requested model's candidate group when no tier pool resolves.
    ComplexityRouter,
    /// Thompson-sampling bandit: classify the request type (code /
    /// reasoning / creative / qa / summarization / translation / other),
    /// sample each candidate's per-type Beta posterior, and pick the argmax
    /// blended with a normalized cost term (see
    /// [`RouterConfig::adaptive_router_config`]). Post-call outcomes update
    /// the posteriors via the server's learning hook. Cold-start picks
    /// (any arm below `cold_start_min_observations`) defer to least-busy.
    AdaptiveRouter,
    /// oxllm extension: strict round-robin that cycles through the candidate
    /// list with a monotonically incrementing counter so consecutive
    /// requests hit different deployments. Weighted deployments occupy
    /// proportional slots. The reference does **not** ship a strict round-robin
    /// strategy (its `RoutingStrategy` enum lists only `simple_shuffle`,
    /// `least_busy`, `latency_based`, `cost_based`, `usage_based_routing*`,
    /// and `provider_budget_limiting`); this fills the gap for
    /// load-testing, deterministic debug traces, and fair request
    /// splitting. The default remains
    /// [`SimpleShuffle`](RoutingStrategyKind::SimpleShuffle), matching
    /// the reference `simple-shuffle` default.
    RoundRobin,
}

impl RoutingStrategyKind {
    /// The snake_case name of the strategy (`"simple_shuffle"`,
    /// `"adaptive_router"`, ...) — the same spelling the config parser
    /// accepts, for stamping headers (`x-oxllm-router-strategy`) and
    /// metric labels.
    pub fn as_str(&self) -> &'static str {
        match self {
            RoutingStrategyKind::SimpleShuffle => "simple_shuffle",
            RoutingStrategyKind::LowestCost => "lowest_cost",
            RoutingStrategyKind::LowestLatency => "lowest_latency",
            RoutingStrategyKind::UsageBased => "usage_based",
            RoutingStrategyKind::LatencyBased => "latency_based",
            RoutingStrategyKind::QualityRouter => "quality_router",
            RoutingStrategyKind::TagBased => "tag_based",
            RoutingStrategyKind::LeastBusy => "least_busy",
            RoutingStrategyKind::SessionAffinity => "session_affinity",
            RoutingStrategyKind::ComplexityRouter => "complexity_router",
            RoutingStrategyKind::AdaptiveRouter => "adaptive_router",
            RoutingStrategyKind::RoundRobin => "round_robin",
        }
    }
}

/// Configuration for the router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// The routing strategy to use.
    pub routing_strategy: RoutingStrategyKind,
    /// Number of retries on retryable errors (5xx, 429, timeout, network).
    pub num_retries: u32,
    /// Cap on how many fallback models are tried after the primary fails
    /// (default 5). Already-tried models are skipped so a looping fallback
    /// graph cannot retry the same target.
    pub max_fallbacks: u32,
    /// Regular fallback rules: primary model → list of fallback models.
    pub fallbacks: Vec<FallbackRule>,
    /// Context-window fallback rules: triggered when the upstream reports a
    /// context-length-exceeded error.
    pub context_window_fallbacks: Vec<FallbackRule>,
    /// Content-policy fallback rules: triggered when the upstream
    /// reports a content-policy / content-filter error before regular
    /// fallbacks are tried.
    pub content_policy_fallbacks: Vec<FallbackRule>,
    /// Default fallback chain consulted when the REQUESTED model group is
    /// unknown (no deployments resolve) — distinct from `fallbacks`, which
    /// handles failures of a KNOWN model. The list form
    /// (`default_fallbacks: ["a", "b"]`) is stored as a single rule with
    /// primary `"*"` applying to any unknown model; the dict form
    /// (`[{"original-model": ["a"]}]`) stores per-model rules that take
    /// precedence over the wildcard.
    pub default_fallbacks: Vec<FallbackRule>,
    /// Cooldown duration (seconds) for failed deployments.
    ///
    /// Defaults to 60 — an intentional divergence from the reference, which
    /// defaults `cooldown_time` to 5 seconds
    /// (`constants.py:33`, `DEFAULT_COOLDOWN_TIME_SECONDS`). The
    /// longer default keeps routing capacity stable after transient
    /// failures; set `cooldown_time` explicitly when porting an existing
    /// config so the behavior matches what upstream would have given you.
    pub cooldown_time_secs: u64,
    /// Quality scores per deployment model (used by `QualityRouter`).
    /// Keyed by the `oxllm_params.model` string (e.g. `"openai/gpt-4o"`).
    pub quality_scores: HashMap<String, f64>,
    /// Per-HTTP-status max retry counts. When set, a retryable
    /// error with status `N` retries up to `retry_policy[N]` times instead
    /// of the global `num_retries`. Statuses missing from the map fall back
    /// to `num_retries`.
    pub retry_policy: HashMap<String, u32>,
    /// Model group aliases. When a client requests a model that
    /// matches a key in this map, the router expands the request to all
    /// deployments whose `model_name` is listed in the alias group.
    pub model_group_alias: HashMap<String, Vec<String>>,
    /// Per-model routing-group overrides. The key is the model name
    /// (e.g. `"gpt-4o"`); the value overrides the global strategy, retry
    /// count, cooldown, and fallback chains for that model only.
    pub routing_groups: HashMap<String, RouterGroupOverride>,
    /// Number of failures (any status) tolerated within the cooldown window
    /// before a deployment is cooled. `None` disables the allowed-fails
    /// policy (existing behavior: cool after retries are exhausted).
    ///
    /// Failure-policy parity: the limit is the count of failures *tolerated* — a
    /// deployment with `allowed_fails: 3` survives 3 failures within the
    /// window and cools on the 4th (`updated_fails > allowed_fails`, see
    /// `should_cooldown_based_on_allowed_fails_policy` at
    /// `router_utils/cooldown_handlers.py:542`).
    pub allowed_fails: Option<u32>,
    /// Per-HTTP-status failure thresholds before cooling. Keys are
    /// stringified status codes (e.g. `"429"`, `"401"`); a deployment is
    /// cooled once its failure count for that status *exceeds* the limit
    /// (failure-policy parity: N tolerated, cool on N+1; see
    /// [`RouterConfig::allowed_fails`]). Takes precedence over
    /// `allowed_fails` for listed statuses.
    pub allowed_fails_policy: Option<HashMap<String, u32>>,
    /// Per-provider spend caps, keyed by provider slug (e.g. `"openai"`,
    /// `"anthropic"`). A deployment whose provider's spend counter
    /// (`spend:provider:{slug}`) is at or over the cap is skipped in the
    /// routing retry loop until the window resets. `None` disables
    /// provider-level budget enforcement.
    pub provider_budget_config: Option<HashMap<String, ProviderBudgetConfig>>,
    /// Master switch for health-check routing: when `true`, deployments
    /// marked unhealthy (connection-level failures) are filtered out of
    /// `pick_deployment` before the strategy pick. Defaults to `false`
    /// (existing behavior: only cooldowns filter candidates).
    pub enable_health_check_routing: bool,
    /// How long (seconds) a health entry stays authoritative. Entries older
    /// than this are treated as healthy again (fail-open), giving a
    /// deployment a chance to prove recovery. Defaults to 60.
    pub health_check_staleness_threshold_secs: u64,
    /// Consecutive connection failures before a deployment is marked
    /// unhealthy. Defaults to 3.
    pub health_check_failure_threshold: u32,
    /// Names of pre-call checks to run (in order) before the strategy pick,
    /// e.g. `["prompt_caching", "budget_limiter", "health_check"]`. Check
    /// instances are injected into the router by the host
    /// ([`crate::router::Router::set_pre_call_checks`]); the built-in
    /// `health_check` is router-managed and also runs (appended last)
    /// whenever `enable_health_check_routing` is set, listed or not.
    pub optional_pre_call_checks: Vec<String>,
    /// Configuration for the `complexity_router` strategy: per-tier model
    /// pools and optional scoring overrides. Required for
    /// [`RoutingStrategyKind::ComplexityRouter`] to do anything useful —
    /// without it, all requests fall back to the requested model's
    /// candidate group.
    pub complexity_router_config: Option<ComplexityRouterConfig>,
    /// Configuration for the `adaptive_router` strategy: quality/cost
    /// blend weights, epsilon-greedy exploration rate, and the cold-start
    /// threshold. `None` uses the documented defaults
    /// ([`AdaptiveRouterConfig::default`]).
    pub adaptive_router_config: Option<AdaptiveRouterConfig>,
    /// Default tag used when a request carries no `tag` under the
    /// `tag_based` strategy. When set, untagged requests are treated as if
    /// they carried this tag. `None` preserves the existing fallback to
    /// `SimpleShuffle` for untagged requests.
    pub default_tag: Option<String>,
    /// Structured per-exception-type retry policy. When set, the named
    /// categories (rate-limit, timeout, internal-server-error,
    /// content-policy) provide per-category retry/failure limits that take
    /// precedence over `allowed_fails_policy` / `allowed_fails` for the
    /// matching HTTP status.
    pub exception_retry_policy: Option<RetryPolicy>,
    /// When `true`, a strategy pick that returns `None` (or a deployment
    /// that is no longer available) triggers a weighted (`SimpleShuffle`)
    /// re-pick over the remaining same-group candidates before the router
    /// gives up — keeping traffic within the model group as long as any
    /// deployment is healthy. A hard selector error is propagated instead
    /// of triggering failover. Defaults to `false` (existing behavior).
    pub enable_weighted_failover: bool,
    /// Percentage-based cooldown threshold: when the ratio of failures to
    /// total requests for a deployment (over a sliding 60-second window)
    /// exceeds this value, the deployment is cooled. Expressed as a fraction
    /// in `[0.0, 1.0]` (e.g. `0.8` = 80% failures). `None` disables
    /// percentage-based cooldown (existing behavior).
    pub failure_threshold_percent: Option<f64>,
    /// Minimum total requests a deployment must have served within the
    /// sliding window before the percentage threshold is evaluated. Guards
    /// against cooling a deployment on a small sample (e.g. 1/1 = 100%
    /// failures). `None` defaults to `1`.
    pub failure_threshold_min_requests: Option<u32>,
    /// Budget throttle percentage: when a scope is over budget, reduce its
    /// effective RPM/TPM limits by this factor (e.g. 0.5 = 50%). `None`
    /// disables throttling (hard-block on budget exceeded).
    pub budget_throttle_pct: Option<f64>,
}

/// Structured retry policy keyed by exception category. Each field maps to
/// representative HTTP statuses: `rate_limit_retries` → 429,
/// `timeout_retries` → 408, `internal_server_error_retries` → 5xx
/// (500/502/503/504/529), `content_policy_retries` → 400. A `None` field
/// inherits from the per-status map / `allowed_fails` / `num_retries` chain
/// as usual.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Max retries/failures tolerated for rate-limit errors (HTTP 429).
    #[serde(alias = "RateLimitErrorRetries")]
    pub rate_limit_retries: Option<u32>,
    /// Max retries/failures tolerated for timeout errors (HTTP 408).
    #[serde(alias = "TimeoutErrorRetries")]
    pub timeout_retries: Option<u32>,
    /// Max retries/failures tolerated for internal server errors
    /// (HTTP 500/502/503/504/529). The reference splits these into separate
    /// `InternalServerErrorAllowedFails` / `BadGatewayErrorAllowedFails` /
    /// `ServiceUnavailableErrorAllowedFails` fields; oxllm's single 5xx
    /// category also covers gateway failures and provider overloads.
    #[serde(alias = "InternalServerErrorRetries")]
    pub internal_server_error_retries: Option<u32>,
    /// Max retries/failures tolerated for content-policy errors (HTTP 400).
    #[serde(alias = "ContentPolicyViolationErrorRetries")]
    pub content_policy_retries: Option<u32>,
}

impl RetryPolicy {
    /// Resolve the per-category limit for an HTTP status, if this policy
    /// covers it. Returns `None` for statuses not mapped to any category.
    ///
    /// 401/404 deliberately have no category here: the reference cools them, and
    /// oxllm counts them through the `num_retries` fallback tier of
    /// [`crate::router::Router::record_failure_and_check`] when an
    /// exception policy is configured; operators wanting an explicit
    /// per-status limit for them can name them in
    /// [`RouterConfig::allowed_fails_policy`].
    pub fn limit_for_status(&self, status: u16) -> Option<u32> {
        match status {
            429 => self.rate_limit_retries,
            408 => self.timeout_retries,
            500 | 502 | 503 | 504 | 529 => self.internal_server_error_retries,
            400 => self.content_policy_retries,
            _ => None,
        }
    }
}

/// Per-model routing-group override. Any field set here takes precedence over
/// the global [`RouterConfig`] value when routing the model named by the
/// containing map key. Fields left `None` fall through to the global default.
#[derive(Debug, Clone, Default)]
pub struct RouterGroupOverride {
    /// Override the routing strategy for this model group.
    pub routing_strategy: Option<RoutingStrategyKind>,
    /// Override the per-status retry count for this model group.
    pub num_retries: Option<u32>,
    /// Override the cooldown duration (seconds) for this model group.
    pub cooldown_time_secs: Option<u64>,
    /// Override the per-HTTP-status retry policy for this model group.
    /// Statuses missing from this map fall back to the global `retry_policy`,
    /// then to `num_retries` (group override first, then global).
    pub retry_policy: HashMap<String, u32>,
    /// Regular fallback rules for this model group.
    pub fallbacks: Vec<FallbackRule>,
    /// Context-window fallback rules for this model group.
    pub context_window_fallbacks: Vec<FallbackRule>,
    /// Content-policy fallback rules for this model group.
    pub content_policy_fallbacks: Vec<FallbackRule>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            routing_strategy: RoutingStrategyKind::default(),
            num_retries: 2,
            max_fallbacks: 5,
            fallbacks: Vec::new(),
            context_window_fallbacks: Vec::new(),
            content_policy_fallbacks: Vec::new(),
            default_fallbacks: Vec::new(),
            cooldown_time_secs: 60,
            quality_scores: HashMap::new(),
            retry_policy: HashMap::new(),
            model_group_alias: HashMap::new(),
            routing_groups: HashMap::new(),
            allowed_fails: None,
            allowed_fails_policy: None,
            provider_budget_config: None,
            enable_health_check_routing: false,
            health_check_staleness_threshold_secs: 60,
            health_check_failure_threshold: 3,
            optional_pre_call_checks: Vec::new(),
            complexity_router_config: None,
            adaptive_router_config: None,
            default_tag: None,
            exception_retry_policy: None,
            enable_weighted_failover: false,
            failure_threshold_percent: None,
            failure_threshold_min_requests: None,
            budget_throttle_pct: None,
        }
    }
}

/// Configuration for the `complexity_router` strategy.
///
/// ```yaml
/// router_settings:
///   routing_strategy: complexity_router
///   complexity_router_config:
///     tiers:
///       simple: [gpt-4o-mini]
///       medium: [gpt-4o-mini, claude-haiku]
///       complex: [gpt-4o]
///       reasoning: [o3]
///     default_model: gpt-4o-mini
///     tier_boundaries:
///       simple: 0.15
///       medium: 0.35
///       complex: 0.60
///     escalation_keywords: ["prove", "theorem"]
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ComplexityRouterConfig {
    /// Model pools per tier: tier name (`"simple"`, `"medium"`,
    /// `"complex"`, `"reasoning"`) → list of `model_name`s eligible for
    /// requests classified into that tier.
    #[serde(default)]
    pub tiers: HashMap<String, Vec<String>>,
    /// Model pool to use when the classified tier has no configured (or no
    /// resolvable) pool. When this also resolves to nothing, the router
    /// falls back to the requested model's normal candidate group.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Score ceiling per tier (`"simple"`, `"medium"`, `"complex"`), in
    /// `[0, 1]`. A request scoring below `"simple"`'s ceiling is SIMPLE,
    /// below `"medium"`'s is MEDIUM, below `"complex"`'s is COMPLEX, and
    /// anything at or above is REASONING. Defaults: 0.15 / 0.35 / 0.60.
    #[serde(default)]
    pub tier_boundaries: Option<HashMap<String, f64>>,
    /// Keywords that bump the classified tier one level up
    /// (simple→medium→complex→reasoning) when present in the request text
    /// (case-insensitive, word-boundary match).
    #[serde(default)]
    pub escalation_keywords: Option<Vec<String>>,
}

impl ComplexityRouterConfig {
    /// Validate `tier_boundaries`: every known-tier value must be a finite
    /// number in `[0, 1]`, and the effective ceilings (unset keys fall back
    /// to the classifier defaults) must be strictly increasing
    /// `simple < medium < complex` — inverted or out-of-range boundaries
    /// silently make tiers unreachable. Unknown tier keys are ignored here
    /// (the classifier ignores them too; hosts may warn on them
    /// separately).
    pub fn validate(&self) -> Result<(), crate::Error> {
        use crate::router::complexity::{
            DEFAULT_COMPLEX_MAX, DEFAULT_MEDIUM_MAX, DEFAULT_SIMPLE_MAX,
        };
        let Some(boundaries) = &self.tier_boundaries else {
            return Ok(());
        };
        for (tier, value) in boundaries {
            if !matches!(tier.as_str(), "simple" | "medium" | "complex") {
                continue;
            }
            if !(value.is_finite() && (0.0..=1.0).contains(value)) {
                return Err(crate::Error::Config(format!(
                    "complexity_router_config.tier_boundaries.{tier} must be a number in [0, 1], got {value}"
                )));
            }
        }
        let boundary = |key: &str, default: f64| boundaries.get(key).copied().unwrap_or(default);
        let simple = boundary("simple", DEFAULT_SIMPLE_MAX);
        let medium = boundary("medium", DEFAULT_MEDIUM_MAX);
        let complex = boundary("complex", DEFAULT_COMPLEX_MAX);
        if !(simple < medium && medium < complex) {
            return Err(crate::Error::Config(format!(
                "complexity_router_config.tier_boundaries must satisfy simple < medium < complex; \
                 effective ceilings are simple={simple}, medium={medium}, complex={complex} \
                 (unset keys use defaults {DEFAULT_SIMPLE_MAX}/{DEFAULT_MEDIUM_MAX}/{DEFAULT_COMPLEX_MAX})"
            )));
        }
        Ok(())
    }
}

/// Configuration for the `adaptive_router` strategy (Thompson-sampling
/// bandit with request-type classification).
///
/// ```yaml
/// router_settings:
///   routing_strategy: adaptive_router
///   adaptive_router_config:
///     quality_weight: 1.0
///     cost_weight: 0.0
///     epsilon: 0.1
///     cold_start_min_observations: 20
/// ```
///
/// Defaults: `quality_weight: 1.0` (the sampled posterior is the primary
/// signal), `cost_weight: 0.0` (cost blending is opt-in — it needs pricing
/// data and changes pick semantics, so quality-only is the safe default),
/// `epsilon: 0.1` (10% uniform exploration, the standard bandit rate),
/// `cold_start_min_observations: 20` (enough outcomes per arm for the
/// posterior to be meaningful before the bandit steers production
/// traffic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveRouterConfig {
    /// Weight on the Thompson sample (posterior quality estimate) in the
    /// pick score. Must be finite and non-negative.
    #[serde(default = "default_quality_weight")]
    pub quality_weight: f64,
    /// Weight on the normalized per-token cost in the pick score
    /// (`score = sample * quality_weight − normalized_cost * cost_weight`).
    /// `0.0` disables cost blending. Must be finite and non-negative.
    #[serde(default)]
    pub cost_weight: f64,
    /// Epsilon-greedy exploration rate in `[0, 1]`: the probability of
    /// picking uniformly at random instead of sampling Thompson scores.
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    /// Cold-start threshold: while ANY candidate cell has fewer recorded
    /// outcomes than this, picks defer to least-busy instead of sampling.
    /// `0` engages the bandit from the first request.
    #[serde(default = "default_cold_start_min_observations")]
    pub cold_start_min_observations: u32,
}

fn default_quality_weight() -> f64 {
    1.0
}

fn default_epsilon() -> f64 {
    0.1
}

fn default_cold_start_min_observations() -> u32 {
    20
}

impl Default for AdaptiveRouterConfig {
    fn default() -> Self {
        Self {
            quality_weight: default_quality_weight(),
            cost_weight: 0.0,
            epsilon: default_epsilon(),
            cold_start_min_observations: default_cold_start_min_observations(),
        }
    }
}

impl AdaptiveRouterConfig {
    /// Validate the blend weights and exploration rate: `epsilon` must lie
    /// in `[0, 1]` and both weights must be finite and non-negative — a
    /// NaN weight would poison every pick score, an out-of-range epsilon
    /// silently disables or maximizes exploration, and a negative weight
    /// inverts the optimization. (The strategy's defensive
    /// `sanitize_config` clamps negative weights to 0; validation rejects
    /// them so a config typo fails loudly at load instead of silently
    /// changing pick semantics.)
    pub fn validate(&self) -> Result<(), crate::Error> {
        if !(self.epsilon.is_finite() && (0.0..=1.0).contains(&self.epsilon)) {
            return Err(crate::Error::Config(format!(
                "adaptive_router_config.epsilon must be a number in [0, 1], got {}",
                self.epsilon
            )));
        }
        if !self.quality_weight.is_finite() {
            return Err(crate::Error::Config(format!(
                "adaptive_router_config.quality_weight must be finite, got {}",
                self.quality_weight
            )));
        }
        if self.quality_weight < 0.0 {
            return Err(crate::Error::Config(format!(
                "adaptive_router_config.quality_weight must be non-negative, got {}",
                self.quality_weight
            )));
        }
        if !self.cost_weight.is_finite() {
            return Err(crate::Error::Config(format!(
                "adaptive_router_config.cost_weight must be finite, got {}",
                self.cost_weight
            )));
        }
        if self.cost_weight < 0.0 {
            return Err(crate::Error::Config(format!(
                "adaptive_router_config.cost_weight must be non-negative, got {}",
                self.cost_weight
            )));
        }
        Ok(())
    }
}

/// Per-provider spend cap as parsed from `router_settings.provider_budget_config`.
///
/// ```yaml
/// router_settings:
///   provider_budget_config:
///     openai:
///       budget_limit: 100.0
///       time_period: 1d
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderBudgetConfig {
    /// Maximum total spend (USD) on the provider within `time_period`.
    pub budget_limit: f64,
    /// Window for the spend counter (e.g. `"1h"`, `"1d"`, `"1mo"`). The
    /// `spend:provider:{slug}` counter expires after this window.
    pub time_period: String,
}

/// A fallback rule: if `primary` fails, try each model in `fallbacks` in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackRule {
    /// The primary model name clients send.
    pub primary: String,
    /// The fallback model names to try if the primary fails with a retryable
    /// error (5xx, 429, timeout, network).
    pub fallbacks: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_boundaries(boundaries: &[(&str, f64)]) -> ComplexityRouterConfig {
        ComplexityRouterConfig {
            tier_boundaries: Some(
                boundaries
                    .iter()
                    .map(|(k, v)| (k.to_string(), *v))
                    .collect(),
            ),
            ..Default::default()
        }
    }

    #[test]
    fn validate_accepts_absent_boundaries() {
        assert!(ComplexityRouterConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_accepts_valid_partial_boundaries() {
        assert!(
            config_with_boundaries(&[("simple", 0.10)])
                .validate()
                .is_ok()
        );
        assert!(
            config_with_boundaries(&[("simple", 0.2), ("medium", 0.5), ("complex", 0.8)])
                .validate()
                .is_ok()
        );
        // Boundary values of exactly 0 and 1 are in range.
        assert!(
            config_with_boundaries(&[("simple", 0.0)])
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_out_of_range_boundaries() {
        assert!(
            config_with_boundaries(&[("simple", 1.5)])
                .validate()
                .is_err()
        );
        assert!(
            config_with_boundaries(&[("medium", -0.1)])
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_non_finite_boundaries() {
        assert!(
            config_with_boundaries(&[("complex", f64::NAN)])
                .validate()
                .is_err()
        );
        assert!(
            config_with_boundaries(&[("simple", f64::INFINITY)])
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_rejects_inverted_boundaries() {
        let err = config_with_boundaries(&[("simple", 0.9), ("medium", 0.2)])
            .validate()
            .expect_err("inverted boundaries must be rejected");
        assert!(
            err.to_string().contains("simple < medium < complex"),
            "error must explain the ordering requirement: {err}"
        );
    }

    #[test]
    fn validate_rejects_partial_boundaries_inverting_defaults() {
        // Only "simple" set, above the default medium ceiling (0.35) →
        // the medium tier would be unreachable → rejected.
        assert!(
            config_with_boundaries(&[("simple", 0.5)])
                .validate()
                .is_err()
        );
    }

    #[test]
    fn validate_ignores_unknown_tier_keys() {
        // Unknown keys carry no meaning for the classifier; they are
        // ignored (hosts may warn on them separately).
        assert!(
            config_with_boundaries(&[("trivial", 0.1)])
                .validate()
                .is_ok()
        );
    }

    // ------------------------------------------------------------------
    // AdaptiveRouterConfig
    // ------------------------------------------------------------------

    #[test]
    fn adaptive_defaults_are_sane() {
        let cfg = AdaptiveRouterConfig::default();
        assert_eq!(cfg.quality_weight, 1.0);
        assert_eq!(cfg.cost_weight, 0.0);
        assert_eq!(cfg.epsilon, 0.1);
        assert_eq!(cfg.cold_start_min_observations, 20);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn adaptive_validate_accepts_edge_epsilon() {
        for epsilon in [0.0, 1.0] {
            let cfg = AdaptiveRouterConfig {
                epsilon,
                ..Default::default()
            };
            assert!(cfg.validate().is_ok(), "epsilon {epsilon} must be valid");
        }
    }

    #[test]
    fn adaptive_validate_rejects_out_of_range_epsilon() {
        for epsilon in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            let cfg = AdaptiveRouterConfig {
                epsilon,
                ..Default::default()
            };
            assert!(
                cfg.validate().is_err(),
                "epsilon {epsilon} must be rejected"
            );
        }
    }

    #[test]
    fn adaptive_validate_rejects_non_finite_weights() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let quality = AdaptiveRouterConfig {
                quality_weight: bad,
                ..Default::default()
            };
            assert!(quality.validate().is_err());
            let cost = AdaptiveRouterConfig {
                cost_weight: bad,
                ..Default::default()
            };
            assert!(cost.validate().is_err());
        }
    }

    #[test]
    fn adaptive_validate_rejects_negative_weights() {
        // Matches the strategy's sanitize_config clamp: a negative weight
        // would invert the optimization, so it is a config error, not a
        // silently-coerced 0.
        let quality = AdaptiveRouterConfig {
            quality_weight: -0.5,
            ..Default::default()
        };
        let err = quality
            .validate()
            .expect_err("negative quality_weight must be rejected");
        assert!(
            err.to_string().contains("quality_weight"),
            "error must name the field: {err}"
        );
        let cost = AdaptiveRouterConfig {
            cost_weight: -1.0,
            ..Default::default()
        };
        let err = cost
            .validate()
            .expect_err("negative cost_weight must be rejected");
        assert!(
            err.to_string().contains("cost_weight"),
            "error must name the field: {err}"
        );
    }

    #[test]
    fn adaptive_validate_accepts_zero_and_positive_weights() {
        for (quality_weight, cost_weight) in [(0.0, 0.0), (1.0, 0.5), (2.5, 10.0)] {
            let cfg = AdaptiveRouterConfig {
                quality_weight,
                cost_weight,
                ..Default::default()
            };
            assert!(
                cfg.validate().is_ok(),
                "weights ({quality_weight}, {cost_weight}) must be valid"
            );
        }
    }

    #[test]
    fn adaptive_config_deserializes_with_defaults() {
        let cfg: AdaptiveRouterConfig = serde_json::from_str("{}").expect("empty object parses");
        assert_eq!(cfg, AdaptiveRouterConfig::default());
        let cfg: AdaptiveRouterConfig =
            serde_json::from_str(r#"{"epsilon": 0.25, "cost_weight": 0.5}"#)
                .expect("partial object parses");
        assert_eq!(cfg.epsilon, 0.25);
        assert_eq!(cfg.cost_weight, 0.5);
        assert_eq!(cfg.quality_weight, 1.0);
        assert_eq!(cfg.cold_start_min_observations, 20);
    }

    #[test]
    fn strategy_kind_as_str_is_snake_case() {
        let cases = [
            (RoutingStrategyKind::SimpleShuffle, "simple_shuffle"),
            (RoutingStrategyKind::LowestCost, "lowest_cost"),
            (RoutingStrategyKind::LowestLatency, "lowest_latency"),
            (RoutingStrategyKind::UsageBased, "usage_based"),
            (RoutingStrategyKind::LatencyBased, "latency_based"),
            (RoutingStrategyKind::QualityRouter, "quality_router"),
            (RoutingStrategyKind::TagBased, "tag_based"),
            (RoutingStrategyKind::LeastBusy, "least_busy"),
            (RoutingStrategyKind::SessionAffinity, "session_affinity"),
            (RoutingStrategyKind::ComplexityRouter, "complexity_router"),
            (RoutingStrategyKind::AdaptiveRouter, "adaptive_router"),
            (RoutingStrategyKind::RoundRobin, "round_robin"),
        ];
        for (kind, name) in cases {
            assert_eq!(kind.as_str(), name);
        }
    }

    #[test]
    fn router_config_defaults_budget_throttle_to_none() {
        let cfg = RouterConfig::default();
        assert_eq!(cfg.budget_throttle_pct, None);
    }

    #[test]
    fn router_config_default_num_retries_matches_litellm() {
        // The reference router defaults to `openai.DEFAULT_MAX_RETRIES` = 2;
        // oxllm must not make one extra attempt per retryable failure.
        assert_eq!(RouterConfig::default().num_retries, 2);
        assert_eq!(RouterConfig::default().max_fallbacks, 5);
    }

    #[test]
    fn router_config_default_cooldown_preserves_routing_capacity() {
        // A 60-second cooldown keeps routing capacity stable after transient failures.
        assert_eq!(RouterConfig::default().cooldown_time_secs, 60);
    }

    #[test]
    fn router_config_carries_budget_throttle_pct() {
        let cfg = RouterConfig {
            budget_throttle_pct: Some(0.5),
            ..RouterConfig::default()
        };
        assert_eq!(cfg.budget_throttle_pct, Some(0.5));
    }
}
