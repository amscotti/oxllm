//! The hook chain trait and supporting types.
//!
//! The [`Hook`] trait is the *only* place `async-trait` is allowed: the chain
//! holds `Vec<Box<dyn Hook>>` and native `async fn` in traits is not
//! dyn-compatible. Transform traits stay synchronous.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use serde_json::Value;

use crate::cost::DeploymentPricing;
use crate::error::Result;

/// The action a pre-call hook can request.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HookAction {
    /// Continue processing normally.
    Continue,
    /// Return a cached response immediately (skip the upstream call).
    ShortCircuit(Value),
    /// Reject the request (e.g. budget exceeded).
    Reject(crate::Error),
}

/// Rate-limit information stashed on the [`CallContext`] by the
/// `RateLimitHook` during `pre_call` so the response pipeline can stamp
/// `x-ratelimit-*` headers. All fields are `Option` because any given
/// dimension (RPM, TPM) may be unset for the request.
#[derive(Debug, Clone, Default)]
pub struct RateLimitInfo {
    /// Remaining requests in the current 60s window, when an RPM limit
    /// applied to this request.
    pub remaining_rpm: Option<u32>,
    /// Remaining tokens in the current 60s window, when a TPM limit applied
    /// to this request.
    pub remaining_tpm: Option<u32>,
    /// The effective RPM ceiling that bound this request, captured at
    /// `pre_call` time. Stashed alongside [`remaining_rpm`](Self::remaining_rpm)
    /// so the `x-ratelimit-limit-requests` header can be stamped from a
    /// consistent snapshot instead of re-deriving `remaining + live_counter`
    /// (which races under concurrency when a global/team limit applied
    /// without a per-key limit). `None` when no RPM limit applied.
    pub limit_rpm: Option<u32>,
    /// The effective TPM ceiling that bound this request, captured at
    /// `pre_call` time. See [`limit_rpm`](Self::limit_rpm).
    pub limit_tpm: Option<u32>,
    /// Seconds until the RPM window resets. Defaults to `60` when an RPM
    /// limit applied (the counter window is 60 seconds).
    pub reset_rpm_secs: Option<u64>,
    /// Seconds until the TPM window resets. Defaults to `60` when a TPM
    /// limit applied.
    pub reset_tpm_secs: Option<u64>,
}

/// A simple struct carrying request context through the hook chain.
#[derive(Debug, Clone, Default)]
pub struct CallContext {
    /// The model name requested by the client.
    pub model: String,
    /// A hash of the API key (never the raw key).
    pub api_key_hash: String,
    /// A unique request identifier.
    pub request_id: String,
    /// Opaque client session identity retained for routing and metering
    /// correlation. Independent of the gateway-generated request identifier.
    pub session_id: Option<String>,
    /// Shared terminal spend-record claim. Cloned request/stream contexts must
    /// not queue a second success, failure or partial row for the same request.
    pub spend_recorded: Arc<OnceLock<()>>,
    /// The serialized request body, for hooks that need to inspect it
    /// (e.g. cache key derivation). Wrapped in [`Arc`] so the body is
    /// deep-cloned at most once (on construction) and shared cheaply
    /// across the hook chain and any spawned task that needs a clone of
    /// the [`CallContext`] (e.g. the streaming task).
    pub request_body: Arc<Value>,
    /// The virtual key id that authenticated the request, if any (master key
    /// has none). Used for spend attribution.
    pub key_id: Option<String>,
    /// The team id attached to the authenticated key, if any.
    pub team_id: Option<String>,
    /// The user id attached to the authenticated key, if any.
    ///
    /// This is the **internal proxy user** (virtual-key `keys.user_id`), not
    /// the OpenAI request-body end-user. End-user / customer identity lives
    /// in [`CallContext::end_user_id`].
    pub user_id: Option<String>,
    /// The org id attached to the authenticated key, if any.
    pub org_id: Option<String>,
    /// The project id attached to the authenticated key, if any.
    pub project_id: Option<String>,
    /// End-user / customer id from the OpenAI request body `"user"` field.
    /// Populated by the pipeline when the body carries a non-empty `user`
    /// string. Distinct from [`CallContext::user_id`] (the virtual-key
    /// owner). Used for customer budget/block enforcement and spend
    /// attribution.
    pub end_user_id: Option<String>,
    /// Customer-level max budget (USD). Checked against
    /// `spend:customer:{end_user_id}` only when set on the customer record.
    pub customer_max_budget: Option<f64>,
    /// Customer-level budget window (e.g. `"7d"`). When set, the budget hook
    /// uses this for the customer spend counter TTL instead of the key's
    /// window.
    pub customer_budget_duration: Option<String>,
    /// Primary normalized request tag from body/metadata arrays and the
    /// native/migration header family. Used for routing compatibility,
    /// budget enforcement and durable attribution.
    pub request_tag: Option<String>,
    /// Every request tag (body `tags`, `metadata.tags` and normalized headers).
    /// The first element mirrors [`Self::request_tag`]; the full list drives
    /// multi-tag budget reservation (each carried tag with a budget row
    /// accrues this request) and the `spend_logs.tags` JSON column.
    pub request_tags: Vec<String>,
    /// Per-tag budget scopes resolved at tag-stamp time:
    /// `(tag name, that tag's own max_budget)` for every carried tag that
    /// HAS a `tags.max_budget` row. The budget hook reserves and settles
    /// against each entry's own `spend:tag:{name}` counter, so one request
    /// counts toward every application budget it is tagged with.
    /// [`Self::tag_max_budget`] stays the most-restrictive ceiling across
    /// these for single-value consumers.
    pub tag_budgets: Vec<(String, f64)>,
    /// Tag-level max budget (USD). Checked against `spend:tag:{name}` when
    /// the named tag row carries a ceiling.
    pub tag_max_budget: Option<f64>,
    /// True when the caller authenticated with the configured master key.
    /// Enforcement hooks (budget, rate-limit) short-circuit for master
    /// principals, matching the documented "master key bypasses all checks"
    /// contract.
    pub is_master: bool,
    /// Per-key budget override (USD). When set, hooks use this instead of the
    /// global `max_budget` for the key spend bucket.
    pub max_budget: Option<f64>,
    /// Per-key budget window (e.g. `"30d"`). When set, hooks use this instead
    /// of the global `budget_duration`.
    pub budget_duration: Option<String>,
    /// Per-key requests-per-minute limit. When set, hooks use this instead of
    /// the global `rpm` for the key RPM bucket.
    pub rpm: Option<u32>,
    /// Per-key tokens-per-minute limit. When set, hooks use this instead of
    /// the global `tpm` for the key TPM bucket.
    pub tpm: Option<u32>,
    /// Team-level max budget (USD). Checked against `spend:team:{team_id}`
    /// only when set — independent of the key ceiling.
    pub team_max_budget: Option<f64>,
    /// Team-level RPM. Checked against `rpm:team:{team_id}` only when set.
    pub team_rpm: Option<u32>,
    /// Team-level TPM. Checked against `tpm:team:{team_id}` only when set.
    pub team_tpm: Option<u32>,
    /// Team-level budget window (e.g. `"1d"`). When set, the budget hook uses
    /// this for the team spend counter TTL instead of the key's window.
    pub team_budget_duration: Option<String>,
    /// Org-level max budget (USD). Checked against `spend:org:{org_id}`
    /// only when set — independent of the key/team/user ceilings.
    pub org_max_budget: Option<f64>,
    /// Org-level RPM. Checked against `rpm:org:{org_id}` only when set.
    pub org_rpm: Option<u32>,
    /// Org-level TPM. Checked against `tpm:org:{org_id}` only when set.
    pub org_tpm: Option<u32>,
    /// Org-level budget window (e.g. `"1d"`). When set, the budget hook uses
    /// this for the org spend counter TTL instead of the key's window.
    pub org_budget_duration: Option<String>,
    /// Project-level max budget (USD). Checked against
    /// `spend:project:{project_id}` only when set.
    pub project_max_budget: Option<f64>,
    /// Project-level RPM. Checked against `rpm:project:{project_id}` only
    /// when set.
    pub project_rpm: Option<u32>,
    /// Project-level TPM. Checked against `tpm:project:{project_id}` only
    /// when set.
    pub project_tpm: Option<u32>,
    /// Project-level budget window (e.g. `"1d"`). When set, the budget hook
    /// uses this for the project spend counter TTL instead of the key's
    /// window.
    pub project_budget_duration: Option<String>,
    /// User-level max budget (USD). Checked against `spend:user:{user_id}`
    /// only when set.
    pub user_max_budget: Option<f64>,
    /// User-level budget window (e.g. `"7d"`). When set, the budget hook
    /// uses this for the user spend counter TTL instead of the key's window.
    pub user_budget_duration: Option<String>,
    /// Max concurrent in-flight requests for this principal (gauge, not a
    /// sliding window). Sourced from org/budget records. `None` = unlimited.
    pub max_parallel_requests: Option<u32>,
    /// Whether a parallel-request gauge slot was reserved in `pre_call` and
    /// must be released in `post_success` / `post_failure`.
    pub parallel_reserved: bool,
    /// Cache-control directives this principal is authorised to send
    /// (`["insert", "no-cache", ...]`). Populated from the virtual key during
    /// auth. When `Some` and non-empty, the cache hook rejects a request whose
    /// `cache:` object uses a key not on the allowlist (the reference
    /// `_PROXY_CacheControlCheck`). `None` or empty means unrestricted. Master
    /// key bypasses the check via [`CallContext::is_master`].
    pub allowed_cache_controls: Option<Vec<String>>,
    /// True when this request was served from the response cache. Metering
    /// hooks (budget, rate-limit, cost-tracking) skip their `post_success`
    /// work on cache hits so cached responses are not re-debited.
    pub cache_hit: bool,
    /// USD amount reserved by `BudgetHook` in `pre_call`. Reused in
    /// `post_success` / `post_failure` so token estimation runs once per
    /// request. `None` when no budget reservation was made.
    pub budget_estimate: Option<f64>,
    /// Admission hint: every configured primary-group deployment carries a
    /// provably zero-price override. Used only before concrete pricing is
    /// selected; a paid fallback still requires its normal budget top-up.
    pub pre_call_pricing_is_free: bool,
    /// Cached prompt-token estimate for the request body. Populated lazily
    /// by the first hook that needs it (e.g. `RateLimitHook` or
    /// `BudgetHook`) and reused by the other so the estimate runs at most
    /// once per request. `OnceLock` is fine here because the value is
    /// filled by a single hook before any other reader observes it (the
    /// hook chain is sequential in `pre_call`).
    pub token_estimate: OnceLock<u64>,
    /// Per-deployment pricing override, stashed after deployment selection
    /// so the cost hooks (`CostTrackingHook`, `BudgetHook`) can compute the
    /// actual spend using deployment-specific rates. `None` until the
    /// router picks a deployment (or when no override is configured).
    pub deployment_pricing: Option<DeploymentPricing>,
    /// Concrete serving model used when a provider reports a public alias or
    /// another model name without an embedded price entry.
    pub pricing_model: Option<String>,
    /// Azure Model Router infrastructure rate, retained when its response selects
    /// an underlying model and cleared when a different deployment serves a fallback.
    pub model_router_input_rate: Option<f64>,
    /// Start of the current upstream attempt. Duration pricing excludes local
    /// admission work, retries and backoff before this attempt.
    pub upstream_start_time: Option<std::time::Instant>,
    /// End of the serving upstream attempt; streams record first-content arrival.
    /// Response timing excludes post-call hooks and spend settlement.
    pub upstream_end_time: Option<std::time::Instant>,
    /// Sum of upstream attempt time, including failed attempts but excluding
    /// retry backoff and post-call hooks. Streams stop at first-content arrival.
    pub upstream_total_duration: std::time::Duration,
    /// Time spent looking up a response-cache hit.
    pub cache_lookup_duration: std::time::Duration,
    /// Selected deployment's timeout used to reserve duration-priced spend.
    /// Streaming reservations remain estimates because idle timeouts do not
    /// bound the total length of a healthy stream.
    pub upstream_duration_estimate: Option<std::time::Duration>,
    /// Whether any attempt received an upstream HTTP response. Later local
    /// failures must not refund rate-limit occupancy already used upstream.
    pub reached_upstream: bool,
    /// API surface of the cached representation. Native and translated
    /// responses must not collide with unrelated endpoint response shapes.
    pub cache_namespace: Option<String>,
    /// Cost of the original upstream response carried by a cache entry.
    /// Exposed for response metadata only; cache hits incur no new spend.
    pub cached_response_cost: Option<f64>,
    /// Public response model retained by a cache entry, including fallback groups.
    pub cached_response_model: Option<String>,
    /// Authoritative upstream cost, or a host-computed settlement snapshot.
    /// A supplied value wins over recomputation, including an explicit zero;
    /// this keeps every budget scope and spend exporter on the same amount.
    pub cost_override: Option<f64>,
    /// Service tier reported by the serving provider, retained across stream
    /// chunks that do not repeat it. It overrides the requested tier when
    /// billing, including an explicit standard/default response.
    pub served_service_tier: Option<String>,
    /// The adaptive-router bandit cell for this request:
    /// `(request_type, deployment_key)`, stamped by the orchestration after
    /// each pick whose leg resolves to the `adaptive_router` strategy (the
    /// deployment key is the router's cooldown key,
    /// [`Router::cooldown_key`](crate::router::Router::cooldown_key) =
    /// [`stable_key`](crate::router::stable_key): `{model_name}:{16-hex
    /// hash}` — *not* the old positional `{model_name}:{index}` scheme).
    /// Re-stamped on every adaptive pick and CLEARED by non-adaptive picks,
    /// so the post-call hooks only ever see a cell the bandit actually
    /// sampled — the serving one on success, the last tried on failure.
    /// `None` when adaptive routing is not in play (or the last pick
    /// belonged to a non-adaptive leg); the learning hook no-ops then.
    /// Intermediate failures cooled inside the retry loop are debited
    /// separately at the cooling point — they never reach this stamp.
    /// Follows the [`CallContext::deployment_pricing`] stash precedent.
    pub adaptive_cell: Option<(crate::router::RequestType, String)>,
    /// The routing strategy that made this request's current pick (the
    /// effective per-model strategy), stamped by the orchestration at pick
    /// time. Purely an observability stash: hooks read it for metrics and
    /// response headers (`RoutingStrategyKind::as_str` is the label
    /// spelling). Re-stamped on every pick, so post-call hooks see the
    /// strategy of the serving leg (or the last tried leg on failure).
    /// `None` until the first pick (e.g. a pre-routing rejection).
    /// Follows the [`CallContext::deployment_pricing`] stash precedent.
    pub router_strategy: Option<crate::router::RoutingStrategyKind>,
    /// A short tag describing how the serving deployment was chosen:
    /// `"picked"` (strategy pick), `"fallback"` (fallback-chain leg),
    /// `"default_fallback"` (unknown-model default chain), or
    /// `"prompt_cache_pin"` (pinned by the prompt-caching pre-call check).
    /// Stamped by the orchestration alongside
    /// [`CallContext::router_strategy`]; purely an observability stash for
    /// metrics/headers. `None` until the first pick. Follows the
    /// [`CallContext::deployment_pricing`] stash precedent.
    pub router_decision: Option<String>,
    /// Public identity of the selected deployment, distinct from credential-
    /// bearing internal routing fingerprints. Replaced on every serving pick.
    pub deployment_id: Option<String>,
    /// Public model group selected for the serving attempt.
    pub serving_model_group: Option<String>,
    /// Selected deployment's provider/model selector, independent of rate-card hints.
    pub deployment_model: Option<String>,
    /// Effective retry budget when the serving deployment needed a retry.
    pub max_retries: Option<u32>,
    /// Same-deployment retries actually attempted, including history repair.
    pub attempted_retries: u32,
    /// Configured cross-model fallback groups actually attempted.
    pub attempted_fallbacks: u32,
    /// Bounded, opt-in status summaries of failed fallback-chain attempts.
    pub fallback_errors: Vec<Value>,
    /// Vetted upstream throttle headers from the current serving attempt.
    pub upstream_response_headers: std::collections::HashMap<String, String>,
    /// Rate-limit quota snapshot for the request, stashed by
    /// `RateLimitHook::pre_call` so the response pipeline can stamp
    /// `x-ratelimit-*` headers. Empty (`Default`) when no limits applied
    /// (e.g. master key, or no RPM/TPM configured).
    pub ratelimit_info: RateLimitInfo,
    /// Opaque strong reference held while a semantic-cache embedding stashed
    /// in `crate::pre_call` is still in-flight. The matching
    /// [`std::sync::Weak`] inside the pending-embedding map lets cap-pressure
    /// eviction tell in-flight entries (guard alive) from abandoned ones
    /// (guard dead, handler future dropped without reaching post hooks) and
    /// prefer reclaiming the latter. `None` when no semantic stash was made.
    pub pending_cache_guard: Option<std::sync::Arc<()>>,
    /// Wall-clock time the request entered the pipeline, stamped at
    /// `CallContext` construction. Observability hooks (Langfuse, Langsmith,
    /// W&B) read this to derive `start_time` / `end_time` / `latency_ms` for
    /// their payloads — without it, every generation is reported with zero
    /// duration (`endTime == startTime`). `None` for contexts built without a
    /// known start time (tests, synthetic contexts).
    pub start_time: Option<std::time::Instant>,
}

impl CallContext {
    /// Claim terminal spend reporting once across cloned contexts.
    pub fn claim_spend_record(&self) -> bool {
        self.spend_recorded.set(()).is_ok()
    }

    /// Requested processing tier, or the native Messages speed setting.
    pub fn requested_service_tier(&self) -> Option<&str> {
        self.request_body
            .get("service_tier")
            .and_then(Value::as_str)
            .or_else(|| self.request_body.get("speed").and_then(Value::as_str))
    }

    /// Bill by the provider's served tier when available, falling back to
    /// the request only when the response does not report a known tier.
    pub fn billable_service_tier(&self, body: Option<&Value>) -> Option<&'static str> {
        crate::cost::resolve_service_tier(
            self.requested_service_tier(),
            body.and_then(crate::cost::response_service_tier)
                .or(self.served_service_tier.as_deref()),
        )
    }

    /// Test constructor with sensible defaults: `api_key_hash: "abc"`,
    /// `request_id: "req-1"`, a null request body, and no limits/identity.
    /// Override the fields a given test cares about with struct-update
    /// syntax:
    ///
    /// ```rust
    /// use oxllm_core::CallContext;
    ///
    /// let ctx = CallContext {
    ///     team_id: Some("team-1".to_string()),
    ///     ..CallContext::for_test("gpt-4o")
    /// };
    /// assert_eq!(ctx.request_id, "req-1");
    /// ```
    pub fn for_test(model: &str) -> Self {
        Self {
            model: model.to_string(),
            api_key_hash: "abc".to_string(),
            request_id: "req-1".to_string(),
            ..Default::default()
        }
    }

    /// Builder helper (primarily for tests): mark the context as
    /// master-key authenticated.
    ///
    /// ```rust
    /// use oxllm_core::CallContext;
    ///
    /// let ctx = CallContext::for_test("gpt-4o").with_master();
    /// assert!(ctx.is_master);
    /// ```
    pub fn with_master(mut self) -> Self {
        self.is_master = true;
        self
    }

    /// The per-key limits, grouped.
    pub fn key_limits(&self) -> crate::limits::ScopeLimits {
        crate::limits::ScopeLimits {
            max_budget: self.max_budget,
            budget_duration: self.budget_duration.clone(),
            rpm: self.rpm,
            tpm: self.tpm,
        }
    }

    /// The team-scope limits, grouped.
    pub fn team_limits(&self) -> crate::limits::ScopeLimits {
        crate::limits::ScopeLimits {
            max_budget: self.team_max_budget,
            budget_duration: self.team_budget_duration.clone(),
            rpm: self.team_rpm,
            tpm: self.team_tpm,
        }
    }

    /// The user-scope limits, grouped. Rate-limit fields are always `None`
    /// (the user table has no RPM/TPM columns); `max_budget` and
    /// `budget_duration` are populated when set on the user record.
    pub fn user_limits(&self) -> crate::limits::ScopeLimits {
        crate::limits::ScopeLimits {
            max_budget: self.user_max_budget,
            budget_duration: self.user_budget_duration.clone(),
            ..Default::default()
        }
    }

    /// The org-scope limits, grouped.
    pub fn org_limits(&self) -> crate::limits::ScopeLimits {
        crate::limits::ScopeLimits {
            max_budget: self.org_max_budget,
            budget_duration: self.org_budget_duration.clone(),
            rpm: self.org_rpm,
            tpm: self.org_tpm,
        }
    }

    /// The project-scope limits, grouped.
    pub fn project_limits(&self) -> crate::limits::ScopeLimits {
        crate::limits::ScopeLimits {
            max_budget: self.project_max_budget,
            budget_duration: self.project_budget_duration.clone(),
            rpm: self.project_rpm,
            tpm: self.project_tpm,
        }
    }

    /// All five scopes' limits, grouped.
    pub fn scope_limits(&self) -> crate::limits::AllScopeLimits {
        crate::limits::AllScopeLimits {
            key: self.key_limits(),
            team: self.team_limits(),
            user: self.user_limits(),
            org: self.org_limits(),
            project: self.project_limits(),
        }
    }

    /// Expand grouped limits onto the flat per-scope fields. Scopes without
    /// a matching column (user rate limits) are ignored.
    pub fn set_scope_limits(&mut self, limits: &crate::limits::AllScopeLimits) {
        self.max_budget = limits.key.max_budget;
        self.budget_duration = limits.key.budget_duration.clone();
        self.rpm = limits.key.rpm;
        self.tpm = limits.key.tpm;
        self.team_max_budget = limits.team.max_budget;
        self.team_budget_duration = limits.team.budget_duration.clone();
        self.team_rpm = limits.team.rpm;
        self.team_tpm = limits.team.tpm;
        self.user_max_budget = limits.user.max_budget;
        self.user_budget_duration = limits.user.budget_duration.clone();
        self.org_max_budget = limits.org.max_budget;
        self.org_budget_duration = limits.org.budget_duration.clone();
        self.org_rpm = limits.org.rpm;
        self.org_tpm = limits.org.tpm;
        self.project_max_budget = limits.project.max_budget;
        self.project_budget_duration = limits.project.budget_duration.clone();
        self.project_rpm = limits.project.rpm;
        self.project_tpm = limits.project.tpm;
    }
}

/// A summary of the response for post-call hooks.
#[derive(Debug, Clone)]
pub struct ResponseSummary {
    /// The model that served the request.
    pub model: String,
    /// The computed cost of the request, if available.
    pub cost: Option<f64>,
    /// Token usage from the upstream response, if available.
    pub usage: Option<UsageSummary>,
    /// The full response body, for hooks that need it (e.g. cache store).
    pub body: Option<serde_json::Value>,
    /// The provider slug that served the request (e.g. `"openai"`).
    pub provider: String,
}

/// Token usage breakdown.
#[derive(Debug, Clone, Default)]
pub struct UsageSummary {
    /// Tokens in the prompt.
    pub prompt_tokens: u64,
    /// Tokens in the completion.
    pub completion_tokens: u64,
    /// Total tokens (prompt + completion).
    pub total_tokens: u64,
    /// Prompt tokens served from a prompt cache (OpenAI
    /// `prompt_tokens_details.cached_tokens`). These are a subset of
    /// `prompt_tokens` and are priced at the cache-read rate.
    pub cache_read_input_tokens: Option<u64>,
    /// Tokens written to the prompt cache (Anthropic
    /// `cache_creation_input_tokens`). Priced at the cache-creation rate.
    pub cache_creation_input_tokens: Option<u64>,
    /// Cache-creation tokens with a 1h ephemeral TTL (Anthropic
    /// `cache_creation.ephemeral_1h_input_tokens`), a subset of
    /// `cache_creation_input_tokens`. Priced at the model's
    /// `cache_creation_input_token_cost_above_1hr`; the remainder of the
    /// cache-creation bucket bills at the 5m rate.
    pub cache_creation_1h_input_tokens: Option<u64>,
    /// Audio input tokens, when the model is multimodal. Priced at the
    /// audio-token rate.
    pub audio_input_tokens: Option<u64>,
    /// Reasoning (thinking) tokens consumed by the model, reported by
    /// Anthropic thinking models and DeepSeek/Grok via
    /// `completion_tokens_details.reasoning_tokens`. These are a subset of
    /// `completion_tokens` and are priced at the completion rate.
    pub reasoning_tokens: Option<u64>,
    /// Citation tokens reported by web-search-grounded providers (Perplexity
    /// Sonar/online). Priced at the model's `citation_cost_per_token` when
    /// declared. Derived from the response's `citations` array; absent on
    /// non-citing models.
    pub citation_tokens: Option<u64>,
    /// Number of web-search requests charged for the call (Perplexity
    /// `usage.num_search_queries`, or `prompt_tokens_details.web_search_requests`
    /// for OpenAI/xAI web-search models). Priced at the model's per-query
    /// search rate when declared.
    pub web_search_requests: Option<u64>,
}

impl UsageSummary {
    /// Saturating-accumulate another summary into this one.
    ///
    /// The three `u64` fields are added with [`u64::saturating_add`]. Each
    /// `Option<u64>` field follows the ladder `(Some(a), Some(b)) →
    /// Some(a.saturating_add(b))`, `(Some, None) → Some`, `(None, Some) →
    /// Some`, `(None, None) → None`, so a value reported by either side is
    /// preserved and a missing value does not zero out a present one.
    ///
    /// This is the single source of truth for folding two summaries (streaming
    /// chunk accumulation, realtime session totals). Consumer code should call
    /// this rather than hand-rolling the per-field ladder, which has
    /// historically dropped fields (e.g. `cache_creation_input_tokens`).
    ///
    /// `total_tokens` is folded independently, **not** recomputed as
    /// `prompt + completion`. Every in-tree `extract_usage` constructor
    /// populates `total_tokens` (with an `input + output` fallback), so this
    /// is correct for shipped providers. A third-party [`UsageSummary`]
    /// builder that leaves `total_tokens` at `0` while populating
    /// `prompt_tokens`/`completion_tokens` will produce a merged summary whose
    /// `total` understates the true count, silently zeroing any TPM/cost
    /// derivation that reads `total_tokens` directly. Such builders must set
    /// `total_tokens`.
    pub fn merge(&mut self, other: &UsageSummary) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
        self.cache_read_input_tokens =
            merge_opt(self.cache_read_input_tokens, other.cache_read_input_tokens);
        self.cache_creation_input_tokens = merge_opt(
            self.cache_creation_input_tokens,
            other.cache_creation_input_tokens,
        );
        self.cache_creation_1h_input_tokens = merge_opt(
            self.cache_creation_1h_input_tokens,
            other.cache_creation_1h_input_tokens,
        );
        self.audio_input_tokens = merge_opt(self.audio_input_tokens, other.audio_input_tokens);
        self.reasoning_tokens = merge_opt(self.reasoning_tokens, other.reasoning_tokens);
        self.citation_tokens = merge_opt(self.citation_tokens, other.citation_tokens);
        self.web_search_requests = merge_opt(self.web_search_requests, other.web_search_requests);
    }
}

#[inline]
fn merge_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// A dependency-health failure emitted independently of an LLM request.
#[derive(Debug, Clone)]
pub struct ServiceFailureEvent {
    /// Bounded dependency label, such as `db` or `redis`.
    pub service: String,
    /// Diagnostic description without credentials or request bodies.
    pub message: String,
}

/// The hook trait — invoked before and after each request.
///
/// Built-in impls ship early: `BudgetPreCheck`, `RateLimitPreCheck`,
/// `CacheLookup`. Third-party hooks (guardrails, Langfuse, Datadog) get a
/// stable home immediately.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Called before the upstream request. Can short-circuit or reject.
    ///
    /// # Rollback contract
    ///
    /// The host guarantees that if a hook's `pre_call` returns
    /// [`HookAction::Continue`] (or otherwise reserves a resource such as a
    /// budget/RPM counter) and a *later* hook in the same chain then aborts
    /// with `Err`, the reservations made by this hook are released. In
    /// practice the host drives this by invoking [`Hook::post_failure`] on
    /// every hook that returned `Continue` when any hook errors. Third-party
    /// hook authors can therefore reserve in `pre_call` and release in
    /// `post_failure` without worrying about leaks from an upstream hook
    /// failure. The exact ordering (whether `post_failure` runs on hooks
    /// before or after the failing hook) is host-defined; hooks must be
    /// idempotent on release.
    async fn pre_call(&self, ctx: &mut CallContext) -> Result<HookAction> {
        let _ = ctx;
        Ok(HookAction::Continue)
    }

    /// Called after a successful upstream response.
    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        let _ = (ctx, resp);
    }

    /// Called after a failed upstream response.
    async fn post_failure(&self, ctx: &CallContext, err: &crate::Error) {
        let _ = (ctx, err);
    }

    /// Observe a configured infrastructure dependency's health failure.
    /// These events have no caller, budget reservation, or billable usage.
    async fn post_service_failure(&self, event: &ServiceFailureEvent) {
        let _ = event;
    }

    /// Whether the hook accepts infrastructure dependency failure events.
    fn supports_service_events(&self) -> bool {
        false
    }

    /// Release no-TTL concurrency slots without settling or refunding
    /// budget/RPM. Used when the client disconnects mid-stream: spend is
    /// kept as the debit, but `max_parallel_requests` must not leak.
    async fn release_concurrency(&self, ctx: &CallContext) {
        let _ = ctx;
    }

    /// Whether this hook records useful information in `post_failure`.
    ///
    /// The default `post_failure` impl is a no-op; hooks that genuinely
    /// observe failure events (metrics, budget rollback, alerting) override
    /// this to `true`. The wrapper in
    /// `state::hook_builders::wrap_config_block_hook` consults this to warn
    /// at startup when an operator lists a slug under `failure_callback` /
    /// `callbacks` whose hook silently no-ops on failure.
    fn supports_failure_events(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::{AllScopeLimits, ScopeLimits};

    #[test]
    fn for_test_uses_conventional_test_defaults() {
        let ctx = CallContext::for_test("gpt-4o");
        assert_eq!(ctx.model, "gpt-4o");
        assert_eq!(ctx.api_key_hash, "abc");
        assert_eq!(ctx.request_id, "req-1");
        assert!(!ctx.is_master);
        assert!(!ctx.cache_hit);
        assert_eq!(ctx.max_budget, None);
    }

    #[test]
    fn with_master_sets_is_master() {
        let ctx = CallContext::for_test("gpt-4o").with_master();
        assert!(ctx.is_master);
    }

    #[test]
    fn scope_limits_roundtrip_through_flat_fields() {
        let limits = AllScopeLimits {
            key: ScopeLimits {
                max_budget: Some(5.0),
                budget_duration: Some("30d".to_string()),
                rpm: Some(100),
                tpm: Some(10_000),
            },
            team: ScopeLimits {
                max_budget: Some(50.0),
                budget_duration: Some("1d".to_string()),
                rpm: Some(500),
                tpm: None,
            },
            user: ScopeLimits {
                max_budget: Some(25.0),
                ..Default::default()
            },
            org: ScopeLimits {
                max_budget: Some(500.0),
                budget_duration: Some("1d".to_string()),
                rpm: Some(5_000),
                tpm: Some(100_000),
            },
            project: ScopeLimits {
                max_budget: Some(250.0),
                budget_duration: Some("7d".to_string()),
                rpm: Some(2_500),
                tpm: Some(50_000),
            },
        };
        let mut ctx = CallContext::for_test("gpt-4o");
        ctx.set_scope_limits(&limits);

        assert_eq!(ctx.max_budget, Some(5.0));
        assert_eq!(ctx.budget_duration.as_deref(), Some("30d"));
        assert_eq!(ctx.rpm, Some(100));
        assert_eq!(ctx.tpm, Some(10_000));
        assert_eq!(ctx.team_max_budget, Some(50.0));
        assert_eq!(ctx.team_budget_duration.as_deref(), Some("1d"));
        assert_eq!(ctx.team_rpm, Some(500));
        assert_eq!(ctx.team_tpm, None);
        assert_eq!(ctx.user_max_budget, Some(25.0));
        assert_eq!(ctx.org_max_budget, Some(500.0));
        assert_eq!(ctx.org_budget_duration.as_deref(), Some("1d"));
        assert_eq!(ctx.org_rpm, Some(5_000));
        assert_eq!(ctx.org_tpm, Some(100_000));
        assert_eq!(ctx.project_max_budget, Some(250.0));
        assert_eq!(ctx.project_budget_duration.as_deref(), Some("7d"));
        assert_eq!(ctx.project_rpm, Some(2_500));
        assert_eq!(ctx.project_tpm, Some(50_000));

        // Reading back collapses to the same grouped value (scopes without
        // columns keep their `None` fields).
        assert_eq!(ctx.scope_limits(), limits);
    }

    #[test]
    fn usage_summary_merge_adds_u64_fields() {
        let mut a = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 200,
            total_tokens: 300,
            ..Default::default()
        };
        let b = UsageSummary {
            prompt_tokens: 1_000,
            completion_tokens: 2_000,
            total_tokens: 3_000,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.prompt_tokens, 1_100);
        assert_eq!(a.completion_tokens, 2_200);
        assert_eq!(a.total_tokens, 3_300);
    }

    #[test]
    fn usage_summary_merge_saturates_u64_fields() {
        let mut a = UsageSummary {
            prompt_tokens: u64::MAX - 5,
            completion_tokens: u64::MAX - 5,
            total_tokens: u64::MAX,
            ..Default::default()
        };
        let b = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 100,
            total_tokens: 1,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.prompt_tokens, u64::MAX);
        assert_eq!(a.completion_tokens, u64::MAX);
        assert_eq!(a.total_tokens, u64::MAX);
    }

    #[test]
    fn usage_summary_merge_folds_every_option_field_for_some_some() {
        let mut a = UsageSummary {
            cache_read_input_tokens: Some(10),
            cache_creation_input_tokens: Some(20),
            cache_creation_1h_input_tokens: Some(30),
            audio_input_tokens: Some(40),
            reasoning_tokens: Some(50),
            citation_tokens: None,
            web_search_requests: None,
            ..Default::default()
        };
        let b = UsageSummary {
            cache_read_input_tokens: Some(1),
            cache_creation_input_tokens: Some(2),
            cache_creation_1h_input_tokens: Some(3),
            audio_input_tokens: Some(4),
            reasoning_tokens: Some(5),
            citation_tokens: None,
            web_search_requests: None,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.cache_read_input_tokens, Some(11));
        assert_eq!(a.cache_creation_input_tokens, Some(22));
        assert_eq!(a.cache_creation_1h_input_tokens, Some(33));
        assert_eq!(a.audio_input_tokens, Some(44));
        assert_eq!(a.reasoning_tokens, Some(55));
    }

    #[test]
    fn usage_summary_merge_folds_every_option_field_for_some_none() {
        // Existing `Some` is preserved when the incoming field is `None`.
        let mut a = UsageSummary {
            cache_read_input_tokens: Some(10),
            cache_creation_input_tokens: Some(20),
            cache_creation_1h_input_tokens: Some(30),
            audio_input_tokens: Some(40),
            reasoning_tokens: Some(50),
            citation_tokens: None,
            web_search_requests: None,
            ..Default::default()
        };
        a.merge(&UsageSummary::default());
        assert_eq!(a.cache_read_input_tokens, Some(10));
        assert_eq!(a.cache_creation_input_tokens, Some(20));
        assert_eq!(a.cache_creation_1h_input_tokens, Some(30));
        assert_eq!(a.audio_input_tokens, Some(40));
        assert_eq!(a.reasoning_tokens, Some(50));
    }

    #[test]
    fn usage_summary_merge_folds_every_option_field_for_none_some() {
        // Existing `None` is replaced when the incoming field is `Some`.
        let mut a = UsageSummary::default();
        let b = UsageSummary {
            cache_read_input_tokens: Some(10),
            cache_creation_input_tokens: Some(20),
            cache_creation_1h_input_tokens: Some(30),
            audio_input_tokens: Some(40),
            reasoning_tokens: Some(50),
            citation_tokens: None,
            web_search_requests: None,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.cache_read_input_tokens, Some(10));
        assert_eq!(a.cache_creation_input_tokens, Some(20));
        assert_eq!(a.cache_creation_1h_input_tokens, Some(30));
        assert_eq!(a.audio_input_tokens, Some(40));
        assert_eq!(a.reasoning_tokens, Some(50));
    }

    #[test]
    fn usage_summary_merge_folds_every_option_field_for_none_none() {
        let mut a = UsageSummary::default();
        a.merge(&UsageSummary::default());
        assert_eq!(a.cache_read_input_tokens, None);
        assert_eq!(a.cache_creation_input_tokens, None);
        assert_eq!(a.cache_creation_1h_input_tokens, None);
        assert_eq!(a.audio_input_tokens, None);
        assert_eq!(a.reasoning_tokens, None);
    }

    #[test]
    fn usage_summary_merge_option_field_saturates_on_some_some() {
        let mut a = UsageSummary {
            cache_read_input_tokens: Some(u64::MAX),
            cache_creation_input_tokens: Some(u64::MAX),
            cache_creation_1h_input_tokens: Some(u64::MAX),
            audio_input_tokens: Some(u64::MAX),
            reasoning_tokens: Some(u64::MAX),
            citation_tokens: None,
            web_search_requests: None,
            ..Default::default()
        };
        let b = UsageSummary {
            cache_read_input_tokens: Some(1),
            cache_creation_input_tokens: Some(1),
            cache_creation_1h_input_tokens: Some(1),
            audio_input_tokens: Some(1),
            reasoning_tokens: Some(1),
            citation_tokens: None,
            web_search_requests: None,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.cache_read_input_tokens, Some(u64::MAX));
        assert_eq!(a.cache_creation_input_tokens, Some(u64::MAX));
        assert_eq!(a.cache_creation_1h_input_tokens, Some(u64::MAX));
        assert_eq!(a.audio_input_tokens, Some(u64::MAX));
        assert_eq!(a.reasoning_tokens, Some(u64::MAX));
    }

    #[test]
    fn usage_summary_merge_preserves_cache_creation_buckets_from_both_sides() {
        // Regression: the realtime `meter_upstream_text` accumulator omitted
        // `cache_creation_input_tokens` and `cache_creation_1h_input_tokens`.
        // An Anthropic-style realtime provider reporting those across two
        // events must see the sum at settlement.
        let mut a = UsageSummary {
            prompt_tokens: 5,
            cache_creation_input_tokens: Some(100),
            cache_creation_1h_input_tokens: Some(40),
            ..Default::default()
        };
        let b = UsageSummary {
            prompt_tokens: 7,
            cache_creation_input_tokens: Some(200),
            cache_creation_1h_input_tokens: Some(60),
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.prompt_tokens, 12);
        assert_eq!(a.cache_creation_input_tokens, Some(300));
        assert_eq!(a.cache_creation_1h_input_tokens, Some(100));
    }

    #[test]
    fn usage_summary_merge_keeps_total_consistent_when_constructors_populate_it() {
        // Contract: `merge` folds `total_tokens` independently rather than
        // recomputing prompt + completion. For well-formed summaries (every
        // in-tree `extract_usage` sets total), the merged total must be at
        // least prompt + completion so a TPM/cost reader cannot undercount.
        let mut a = UsageSummary {
            prompt_tokens: 150,
            completion_tokens: 50,
            total_tokens: 200,
            ..Default::default()
        };
        let b = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 25,
            total_tokens: 125,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.total_tokens, 325);
        assert!(
            a.total_tokens >= a.prompt_tokens + a.completion_tokens,
            "merged total {} must cover prompt+completion {} for TPM/cost derivations",
            a.total_tokens,
            a.prompt_tokens + a.completion_tokens
        );
    }

    #[test]
    fn usage_summary_merge_does_not_recompute_total_from_prompt_completion() {
        // Pins the documented contract: a malformed summary with total_tokens
        // left at 0 is NOT healed on merge — downstream TPM/cost readers that
        // rely on total_tokens would undercount. This makes the dependency on
        // constructor-side population explicit so a future change that "fixes"
        // this silently is caught.
        let mut a = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 0,
            ..Default::default()
        };
        let b = UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 0,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.total_tokens, 0, "merge must not recompute total_tokens");
        assert_eq!(a.prompt_tokens, 110);
        assert_eq!(a.completion_tokens, 55);
    }
}
