//! The shared request-lifecycle pipeline.
//!
//! Every LLM handler (chat, embeddings, passthrough) funnels the common pieces
//! of the request lifecycle — [`CallContext`] construction, pre/post hook
//! orchestration, per-deployment RPM/TPM enforcement, cost calculation, and
//! response-header stamping — through this module. Handler-specific
//! orchestration (retries, fallbacks, streaming) stays in the handlers and
//! calls into the pipeline for the shared pieces.
//!
//! Two surfaces are provided:
//! - [`EndpointPipeline`]: a borrow-based struct used by the main request path.
//! - Free functions ([`post_success`], [`post_failure`]): for spawned tasks
//!   that own their state and call context (e.g. the streaming task in
//!   `handlers::chat`).

use std::time::Duration;

use axum::body::Bytes;
use axum::http::{HeaderMap, HeaderValue, Method};
use axum::response::Response;
use oxllm_core::deployment::Deployment;
use oxllm_core::{CallContext, IncrResult, RateLimitInfo, ResponseSummary, UsageSummary};
use serde_json::Value;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::hooks::{
    run_post_failure_hooks, run_post_success_hooks, run_pre_call_hooks, run_release_concurrency,
};
use crate::state::AppState;

pub(crate) mod guardrails;
pub(crate) mod multipart;
pub mod passthrough;
mod response_metadata;
pub(crate) mod routing;
pub use passthrough::{
    PassthroughMode, PassthroughRequest, catch_all_upstream_base_url, default_base_for_provider,
    forward_with_mode, upstream_base_url, vetted_client_headers,
};
pub(crate) use response_metadata::{
    capture_upstream_headers, normalize_client_response, public_response_model,
    record_fallback_error, record_upstream_timing, stamp_deployment_headers,
};

/// Response header carrying the computed cost of the request.
pub const COST_HEADER: &str = "x-oxllm-response-cost";

/// U3: header carrying total token count reported by an upstream pass-through
/// target (another oxllm instance). Passthrough routes read this to construct
/// a `UsageSummary` for cost/TPM tracking since they cannot reliably parse the
/// provider-native response body to extract usage.
pub const UPSTREAM_TOTAL_TOKENS_HEADER: &str = "x-oxllm-total-tokens";

/// U3: legacy alias for total tokens. Checked as a fallback so
/// oxllm works as a pass-through target behind existing proxies.
pub const UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM: &str = "x-litellm-total-tokens";

/// U3: header carrying the cost an upstream pass-through target computed for
/// the whole request. When present, the value overrides oxllm's local cost
/// recomputation so chained-gateway deployments bill the caller the same total
/// the upstream reported (response-cost contract parity — see
/// `proxy/pass_through_endpoints/upstream_usage_headers.py`).
pub const UPSTREAM_RESPONSE_COST_HEADER: &str = "x-oxllm-response-cost";

/// U3: legacy alias for the upstream-reported response cost.
/// Checked as a fallback so oxllm works as a pass-through target behind
/// existing proxies.
pub const UPSTREAM_RESPONSE_COST_HEADER_LITELLM: &str = "x-litellm-response-cost";

/// U3: extract token usage from upstream response headers.
///
/// Pass-through targets (another oxllm instance, or an existing proxy) can report
/// the total token count via the `x-oxllm-total-tokens` header or its legacy
/// alias. Since the passthrough handler
/// cannot reliably parse the provider-native response body to split prompt vs
/// completion tokens, all reported tokens are attributed to `prompt_tokens` so
/// that cost is at least tracked (approximate) rather than dropped entirely.
///
/// Cost bias: because `calculate_cost_with_override` prices completion tokens
/// at a higher rate than prompt tokens, attributing everything to `prompt_tokens`
/// systematically **under-counts** spend (it never over-counts). This is the
/// conservative direction for a gateway — under-counting may let slightly more
/// traffic through than the budget intends, while over-counting would wrongly
/// block legitimate requests — but passthrough spend is therefore lower than
/// the transformed chat path for an equivalent call.
///
/// Returns `None` when neither header is present or the value is not a valid
/// non-zero integer.
pub fn extract_upstream_usage(headers: &HeaderMap) -> Option<UsageSummary> {
    let total: u64 = headers
        .get(UPSTREAM_TOTAL_TOKENS_HEADER)
        .or_else(|| headers.get(UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok())?;
    if total == 0 {
        return None;
    }
    Some(UsageSummary {
        prompt_tokens: total,
        completion_tokens: 0,
        total_tokens: total,
        ..Default::default()
    })
}

/// U3: the outcome of reading an upstream-reported response-cost header.
///
/// The response-cost contract distinguishes three cases for a pass-through target
/// (another oxllm instance, or an existing proxy):
/// - the target did not send a cost header → it does not own the cost, so the
///   local price-table estimate runs as normal ([`UpstreamResponseCost::Absent`]
///   → `None` override);
/// - the target sent a valid non-negative cost → that value is authoritative
///   ([`UpstreamResponseCost::Valid`] → `Some(cost)` override);
/// - the target sent a cost header but the value was missing, unparseable,
///   NaN/±∞, or negative → the target *spoke* the cost contract, so it owns
///   the cost for the request and a value it could not honour bills `0`
///   rather than falling back to a local estimate it just contradicted
///   ([`UpstreamResponseCost::Invalid`] → `Some(0.0)` override).
///
/// The last case is the response-cost parity fix: the reference
/// `success_handler._set_cost_per_request` short-circuits whenever
/// `has_upstream_reported_usage` is true (the target sent any cost/tokens
/// header), so an unparseable reported cost records `0`, never the local
/// estimate. See [`extract_upstream_response_cost`] and
/// [`UpstreamResponseCost::into_override`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpstreamResponseCost {
    /// Neither the native nor the legacy cost header was present: the upstream
    /// did not speak the cost contract, so local price-table compute runs.
    Absent,
    /// The upstream reported a valid, finite, non-negative cost.
    Valid(f64),
    /// The upstream sent a cost header but the value was missing, unparseable,
    /// NaN/±∞, or negative. The upstream owns the cost (it spoke the contract),
    /// so the request bills `0` instead of falling back to a local estimate.
    Invalid,
}

impl UpstreamResponseCost {
    /// Convert the parsed outcome into the `cost_override` to stash on the
    /// call context. [`UpstreamResponseCost::Absent`] yields `None` so
    /// [`post_success`] runs local price-table compute; `Valid` and
    /// `Invalid` yield `Some` so `post_success` honours the upstream's
    /// ownership of the cost (contract parity — an invalid reported cost bills
    /// `0`, never the local estimate).
    pub fn into_override(self) -> Option<f64> {
        match self {
            UpstreamResponseCost::Absent => None,
            UpstreamResponseCost::Valid(c) => Some(c),
            UpstreamResponseCost::Invalid => Some(0.0),
        }
    }
}

/// U3: extract a pre-computed cost reported by an upstream pass-through target
/// (another oxllm instance, or an existing proxy) via the
/// [`UPSTREAM_RESPONSE_COST_HEADER`] (or the legacy alias).
///
/// The cost contract is that an upstream that reports its own cost *owns* the
/// cost for the request: the reported value wins over any local recomputation,
/// even when the value is `0`. Callers that receive [`UpstreamResponseCost::Valid`]
/// or [`UpstreamResponseCost::Invalid`] from this helper stamp the corresponding
/// override (via [`UpstreamResponseCost::into_override`]) on the call context
/// before invoking `post_success`, so the CostTrackingHook / spend log /
/// response header all agree with the upstream rather than re-deriving a
/// (different) number. A target that does not speak the contract returns
/// [`UpstreamResponseCost::Absent`], and the local
/// `calculate_cost_with_override_for_provider` path runs as before.
///
/// A header that is present but unparseable, NaN/±∞, or negative yields
/// [`UpstreamResponseCost::Invalid`] — distinct from [`UpstreamResponseCost::Absent`]
/// — so the caller can bill `0` (contract parity) instead of letting a local
/// estimate through. The reference applies the same
/// `math.isfinite && >= 0` guard to the *value*, but additionally tracks
/// whether the contract was spoken at all (`has_upstream_reported_usage`).
pub fn extract_upstream_response_cost(headers: &HeaderMap) -> UpstreamResponseCost {
    let raw = match headers
        .get(UPSTREAM_RESPONSE_COST_HEADER)
        .or_else(|| headers.get(UPSTREAM_RESPONSE_COST_HEADER_LITELLM))
        .and_then(|v| v.to_str().ok())
    {
        Some(r) => r,
        None => return UpstreamResponseCost::Absent,
    };
    match raw.trim().parse::<f64>() {
        Ok(cost) if cost.is_finite() && cost >= 0.0 => UpstreamResponseCost::Valid(cost),
        _ => UpstreamResponseCost::Invalid,
    }
}

/// Read a scalar `usage.cost` or Perplexity's `usage.cost.total_cost` receipt.
///
/// OpenRouter populates this when the request asked for it
/// (`usage: {include: true}`); the OpenRouter provider unconditionally injects
/// that flag. For models absent from the local price table — common with an
/// aggregator that routes hundreds of upstreams — this is the authoritative
/// spend figure. It is normally a fallback when the price table is missing;
/// Perplexity receipts also carry dynamic search/request fees that token-only
/// arithmetic cannot reproduce. Negative, NaN, or infinite values mirror
/// [`extract_upstream_response_cost`].
pub fn extract_body_usage_cost(body: &Value) -> Option<f64> {
    let cost = body.get("usage")?.get("cost")?;
    let cost = cost
        .as_f64()
        .or_else(|| cost.get("total_cost").and_then(Value::as_f64))?;
    if !cost.is_finite() || cost < 0.0 {
        return None;
    }
    Some(cost)
}

/// Header name carrying the unique request id.
pub const REQUEST_ID_HEADER: &str = "x-oxllm-request-id";

/// H8: standard OpenAI request-id header. Stamped alongside
/// [`REQUEST_ID_HEADER`] on every response so OpenAI-compatible clients can
/// read the request id without an oxllm-specific header.
pub const STANDARD_REQUEST_ID_HEADER: &str = "x-request-id";

tokio::task_local! {
    /// Gateway-generated correlation id scoped around the entire HTTP request,
    /// including middleware and extractor failures that precede a pipeline.
    pub(crate) static HTTP_REQUEST_ID: String;
    pub(crate) static HTTP_REQUEST_START: std::time::Instant;
}

/// Reuse the HTTP request's correlation id when running under the router, or
/// allocate one for direct/internal pipeline callers.
pub(crate) fn request_id() -> String {
    HTTP_REQUEST_ID
        .try_with(Clone::clone)
        .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string())
}

/// Stamp `REQUEST_ID_HEADER` AND `STANDARD_REQUEST_ID_HEADER` on the
/// response with the given request id. Both headers receive the same value:
/// oxllm's own clients can read `x-oxllm-request-id`, generic OpenAI clients
/// read `x-request-id`. Centralised here so every stamp site stays in sync
/// (the alternative — sprinkling two `set_header` calls at each site — is
/// the bug H8 was filed against: callers stamped only the oxllm variant).
pub fn stamp_request_id_header(response: &mut Response, request_id: &str) {
    set_header(response, REQUEST_ID_HEADER, request_id);
    set_header(response, STANDARD_REQUEST_ID_HEADER, request_id);
}

/// Header name flagging a cache hit.
pub const CACHE_HIT_HEADER: &str = "x-oxllm-cache-hit";

/// Collect the response headers that must survive a response-shape conversion
/// (legacy completions, Anthropic Messages conversion, etc.): every
/// `x-oxllm-*` transparency header (cost, cache-hit, router strategy/decision,
/// request id), every `x-ratelimit-*` throttle header, and the OpenAI-standard
/// `x-request-id`. Conversion handlers build a brand-new `Response` from the
/// converted body and would otherwise drop every header the orchestrator
/// stamped; calling this before consuming the body and re-inserting the result
/// keeps cost / routing / throttling / request-id observability consistent
/// across the converted and non-converted endpoints.
pub(crate) fn preserved_response_headers(
    headers: &HeaderMap,
) -> Vec<(axum::http::HeaderName, HeaderValue)> {
    headers
        .iter()
        .filter(|(name, _)| {
            let s = name.as_str();
            s.starts_with("x-oxllm-")
                || s.starts_with("x-litellm-")
                || s.starts_with("x-ratelimit-")
                || s == STANDARD_REQUEST_ID_HEADER
                || s == "retry-after"
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// `x-ratelimit-*` response headers exposed for client-side throttling.
/// Pattern follows the IETF draft (`draft-ietf-httpapi-ratelimit-headers`)
/// that OpenAI / Anthropic / GitHub all expose.
pub const RATELIMIT_LIMIT_REQUESTS: &str = "x-ratelimit-limit-requests";
pub const RATELIMIT_LIMIT_TOKENS: &str = "x-ratelimit-limit-tokens";
pub const RATELIMIT_REMAINING_REQUESTS: &str = "x-ratelimit-remaining-requests";
pub const RATELIMIT_REMAINING_TOKENS: &str = "x-ratelimit-remaining-tokens";
pub const RATELIMIT_RESET_REQUESTS: &str = "x-ratelimit-reset-requests";
pub const RATELIMIT_RESET_TOKENS: &str = "x-ratelimit-reset-tokens";

/// Request header carrying a client session id for sticky routing
/// (`routing_strategy: session_affinity`).
pub const SESSION_ID_HEADER: &str = "x-oxllm-session-id";

/// Alias accepted for clients that already send the legacy session header.
/// `x-oxllm-session-id` wins when both are present. An empty value falls
/// through to this header, then to the request body.
pub const LITELLM_SESSION_ID_HEADER: &str = "x-litellm-session-id";

/// Request header carrying a routing tag (`routing_strategy: tag_based`).
/// Clients send `x-oxllm-tag: canary` to select deployments whose
/// `model_info.tags` contains `"canary"`.
pub const TAG_HEADER: &str = "x-oxllm-tag";

/// Extract a non-empty session id from `x-oxllm-session-id`, if present.
pub fn extract_session_id(headers: &HeaderMap) -> Option<String> {
    extract_session_id_with_body(headers, None)
}

/// Session id for sticky routing. The explicit `x-oxllm-session-id` header
/// wins. An empty value falls through to the legacy session header, then to
/// Anthropic `/v1/messages` `metadata.user_id` (`…_session_<id>` or
/// `{ "session_id": "<id>" }`).
pub fn extract_session_id_with_body(headers: &HeaderMap, body: Option<&Value>) -> Option<String> {
    if let Some(id) = header_session_id(headers, SESSION_ID_HEADER) {
        return Some(id);
    }
    if let Some(id) = header_session_id(headers, LITELLM_SESSION_ID_HEADER) {
        tracing::debug!("session id taken from x-litellm-session-id");
        return Some(id);
    }
    body.and_then(anthropic_session_id_from_metadata)
}

fn header_session_id(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Characters allowed in an Anthropic-derived session id (UUIDs, CLI ids).
fn is_anthropic_session_id(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn anthropic_session_id_from_metadata(body: &Value) -> Option<String> {
    let user_id = body.get("metadata")?.get("user_id")?;
    if let Some(obj) = user_id.as_object() {
        let session = obj.get("session_id")?.as_str()?;
        return is_anthropic_session_id(session).then(|| session.to_string());
    }
    let user_id = user_id.as_str()?;
    let marker = "_session_";
    let idx = user_id.rfind(marker)?;
    let session = &user_id[idx + marker.len()..];
    is_anthropic_session_id(session).then(|| session.to_string())
}

/// Compatibility accessor for the primary header tag. New handlers should
/// use [`extract_tags`] so all tags participate in routing and budgets.
pub fn extract_tag(headers: &HeaderMap) -> Option<String> {
    extract_tags(headers, None).into_iter().next()
}

/// Upper bound on the number of tags a single request may carry. Tags are
/// client-controlled; every one of them costs a budget-cache lookup, a
/// routing pass, and a spend-log column, so a request carrying thousands
/// must not multiply that work. Real traffic carries a handful.
pub const MAX_REQUEST_TAGS: usize = 16;

/// Upper bound on a single tag's length (bytes, on a char boundary). Tags
/// are short selectors (`app-billing`, `env-prod`); the bound keeps the
/// spend-log column and cache keys bounded.
pub const MAX_TAG_BYTES: usize = 128;

/// Extract every request tag from the two supported carriers:
///
/// - the request body's `metadata.tags` array (the standard proxy
///   convention — the common case for per-application tagging on a shared
///   key), and
/// - the `x-oxllm-tag` header, which may itself carry a comma-separated
///   list.
///
/// Both sources are merged (order-preserving union, deduplicated, trimmed,
/// non-empty), then bounded to [`MAX_REQUEST_TAGS`] entries of at most
/// [`MAX_TAG_BYTES`] bytes each. The returned order is body-tags-first
/// (the first element is the request's primary tag).
pub fn extract_tags(headers: &HeaderMap, body: Option<&Value>) -> Vec<String> {
    merge_request_tags(
        body,
        headers
            .get_all(TAG_HEADER)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|raw| raw.split(',')),
    )
}

/// Merge body tags with already-split header/compatibility tags using the same
/// bounds as [`extract_tags`]. Shared by passthrough and legacy inner handlers.
pub(crate) fn merge_request_tags<'a>(
    body: Option<&Value>,
    extra_tags: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let push = |raw: &str, out: &mut Vec<String>| {
        let t = raw.trim();
        if t.is_empty() {
            return;
        }
        if out.len() >= MAX_REQUEST_TAGS {
            return;
        }
        if t.len() > MAX_TAG_BYTES {
            let mut end = MAX_TAG_BYTES;
            while end > 0 && !t.is_char_boundary(end) {
                end -= 1;
            }
            let t = &t[..end];
            if !out.iter().any(|e| e == t) {
                out.push(t.to_string());
            }
            return;
        }
        if !out.iter().any(|e| e == t) {
            out.push(t.to_string());
        }
    };
    if let Some(Value::Array(tags)) = body
        .and_then(|b| b.get("metadata"))
        .and_then(|m| m.get("tags"))
    {
        for t in tags {
            if out.len() == MAX_REQUEST_TAGS {
                break;
            }
            if let Some(s) = t.as_str() {
                push(s, &mut out);
            }
        }
    }
    for t in extra_tags {
        if out.len() == MAX_REQUEST_TAGS {
            break;
        }
        push(t, &mut out);
    }
    out
}

/// A shared request-lifecycle pipeline.
///
/// Construct one per request via [`EndpointPipeline::for_request`], then drive
/// the lifecycle: [`pre_call`](Self::pre_call) → route →
/// [`check_limits`](Self::check_limits) → upstream call →
/// [`post_success`](Self::post_success) or
/// [`post_failure`](Self::post_failure).
pub struct EndpointPipeline<'a> {
    /// The shared application state (borrowed for the request's duration).
    pub state: &'a AppState,
    /// The per-request call context (owned by the pipeline).
    pub call_ctx: CallContext,
}

impl<'a> EndpointPipeline<'a> {
    /// Build a pipeline for a new request.
    ///
    /// Uses the HTTP-scoped request id (or generates one for direct callers)
    /// and constructs the [`CallContext`] from
    /// the authenticated principal. This replaces the duplicated context
    /// construction that previously lived in each handler.
    ///
    /// Returns `Err` when the requested model is not allowed by the key/team
    /// model ACLs (HTTP 403). Master-key callers bypass the key/team/user ACL.
    /// When the body carries a non-empty OpenAI `"user"` field and a customer
    /// row exists, also enforces that customer's `blocked` flag and
    /// `model_blocklist` (including for master keys — abuse control).
    pub async fn for_request(
        state: &'a AppState,
        auth: &AuthUser,
        model: &str,
        request_body: Value,
    ) -> Result<Self, ApiError> {
        check_model_acl(
            auth,
            model,
            auth.key_models.as_deref(),
            auth.team_models.as_deref(),
            auth.user_models.as_deref(),
        )?;
        let request_id = request_id();
        // Attribute master-key traffic as "master" so spend logs always show a key.
        let key_id = auth.key_id.clone().or_else(|| {
            if auth.is_master {
                Some("master".to_string())
            } else {
                None
            }
        });
        let mut call_ctx = CallContext {
            model: model.to_string(),
            api_key_hash: auth.key_hash_hex().to_string(),
            request_id,
            request_body: std::sync::Arc::new(request_body),
            key_id,
            team_id: auth.team_id.clone(),
            user_id: auth.user_id.clone(),
            org_id: auth.org_id.clone(),
            project_id: auth.project_id.clone(),
            is_master: auth.is_master,
            allowed_cache_controls: auth.allowed_cache_controls.clone(),
            start_time: Some(
                HTTP_REQUEST_START
                    .try_with(|start| *start)
                    .unwrap_or_else(|_| std::time::Instant::now()),
            ),
            ..Default::default()
        };
        call_ctx.set_scope_limits(&auth.scope_limits());
        call_ctx.max_parallel_requests = auth.max_parallel_requests;
        apply_customer_context(state, &mut call_ctx).await?;
        Ok(Self { state, call_ctx })
    }

    /// Stamp request-tag budget limits from the `x-oxllm-tag` header value.
    ///
    /// Call after [`Self::for_request`] and before [`Self::pre_call`] so
    /// `BudgetHook` sees `tag_max_budget`. Unknown tags pass through; a DB
    /// error fails closed when the header is present.
    pub async fn apply_request_tag(&mut self, tag: Option<&str>) -> Result<(), ApiError> {
        apply_tag_context(self.state, &mut self.call_ctx, tag).await
    }

    /// Stamp every request tag (body `metadata.tags` + header) and resolve
    /// per-tag budget scopes. See [`apply_request_tags`].
    pub async fn apply_request_tags(&mut self, tags: &[String]) -> Result<(), ApiError> {
        apply_request_tags(self.state, &mut self.call_ctx, tags).await
    }

    /// The unique request id assigned to this request.
    pub fn request_id(&self) -> &str {
        &self.call_ctx.request_id
    }

    /// Run the pre-call hook chain (cache lookup, budget, rate limit).
    ///
    /// Returns:
    /// - `Ok(Some(value))` when a hook short-circuited (e.g. cache hit) — the
    ///   caller builds a response from `value` and skips the upstream call.
    /// - `Ok(None)` when every hook returned `Continue`.
    /// - `Err` when a hook rejected the request. Rejections roll back the
    ///   reservations of every earlier hook in the chain (see
    ///   [`run_pre_call_hooks`]), so callers only need to handle the error —
    ///   no extra cleanup.
    pub async fn pre_call(&mut self) -> Result<Option<Value>, ApiError> {
        run_pre_call_hooks(&self.state.hooks, &mut self.call_ctx).await
    }

    /// Check a deployment's per-minute RPM/TPM limits. Returns `Err(429)` when
    /// either limit is exceeded.
    ///
    /// NOTE: unlike hook rejections, a `check_limits` failure does NOT roll
    /// back the reservations made in [`pre_call`](Self::pre_call). Callers
    /// that bail out on this error must first release them via
    /// [`post_failure`](Self::post_failure) — see
    /// [`core_error_from_api`] for converting the [`ApiError`] for the hook
    /// chain.
    pub async fn check_limits(
        &self,
        deployment: &Deployment,
        dep_key: &str,
    ) -> Result<(), ApiError> {
        check_deployment_limits(self.state, deployment, dep_key).await
    }

    /// Record a successful response: compute cost from usage, build the
    /// [`ResponseSummary`], and run the post-success hook chain (budget debit,
    /// spend log, cache store, metrics). Returns the computed cost so the
    /// caller can stamp the cost header.
    pub async fn post_success(
        &self,
        model: &str,
        usage: Option<&UsageSummary>,
        body: Option<&Value>,
        provider: &str,
    ) -> Option<f64> {
        post_success(
            self.state,
            &self.call_ctx,
            model,
            usage,
            body,
            None,
            provider,
        )
        .await
    }

    /// Record a failed response: run the post-failure hook chain.
    pub async fn post_failure(&self, err: &oxllm_core::Error) {
        run_post_failure_hooks(&self.state.hooks, &self.call_ctx, err).await;
    }
}

/// Free-function form of post-success handling for spawned tasks that own
/// their state and call context (e.g. the streaming task).
///
/// Computes cost from `usage`, builds a [`ResponseSummary`], runs the
/// post-success hooks, and returns the cost (if any).
///
/// `body_cost` is the cost-only fallback (aggregator `usage.cost`, or resolved
/// deployment pricing when a stream reports an unpriced public alias) when
/// the caller has **no cacheable response body** to pass as `body`. It feeds only the cost
/// fallback chain, never [`ResponseSummary::body`]: `CacheHook::post_success`
/// stores `body` verbatim, so a synthetic `{"usage":{"cost":…}}` fragment
/// passed as `body` would be served as the next identical request's cached
/// "response" instead of a chat completion. Streaming callers therefore pass
/// `body = None` and thread the captured cost through this field.
pub async fn post_success(
    state: &AppState,
    call_ctx: &CallContext,
    model: &str,
    usage: Option<&UsageSummary>,
    body: Option<&Value>,
    body_cost: Option<f64>,
    provider: &str,
) -> Option<f64> {
    let cost = if call_ctx.cache_hit {
        call_ctx.cached_response_cost.or_else(|| {
            crate::hooks::cost_fallback::resolve_cost(
                call_ctx, model, usage, body, body_cost, provider,
            )
        })
    } else {
        crate::hooks::cost_fallback::resolve_cost(call_ctx, model, usage, body, body_cost, provider)
    };
    let summary = ResponseSummary {
        model: model.to_string(),
        cost,
        usage: usage.cloned(),
        body: body.cloned(),
        provider: provider.to_string(),
    };
    run_post_success_hooks(&state.hooks, call_ctx, &summary).await;
    cost
}

/// Free-function form of post-failure handling for spawned tasks.
pub async fn post_failure(state: &AppState, call_ctx: &CallContext, err: &oxllm_core::Error) {
    run_post_failure_hooks(&state.hooks, call_ctx, err).await;
}

/// Release no-TTL concurrency slots without refunding budget/RPM.
pub async fn release_concurrency(state: &AppState, call_ctx: &CallContext) {
    run_release_concurrency(&state.hooks, call_ctx).await;
}

/// Retain the concrete rate card and reserve any positive price delta after a
/// deployment is selected. Shared by transformed and hooked-raw endpoints.
pub(crate) async fn prepare_deployment_pricing(
    state: &AppState,
    ctx: &mut CallContext,
    deployment: &Deployment,
    request: &Value,
) -> oxllm_core::Result<()> {
    ctx.deployment_id = deployment
        .model_info
        .as_ref()
        .and_then(|info| info.id.clone());
    ctx.upstream_response_headers.clear();
    ctx.serving_model_group = Some(
        request
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&ctx.model)
            .to_string(),
    );
    ctx.deployment_model = Some(if deployment.oxllm_params.model.contains('*') {
        let prefix = format!("{}/", deployment.provider_slug());
        let requested = request
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&ctx.model);
        format!(
            "{prefix}{}",
            requested.strip_prefix(&prefix).unwrap_or(requested)
        )
    } else {
        deployment.oxllm_params.model.clone()
    });
    ctx.max_retries = None;
    ctx.upstream_start_time = None;
    ctx.upstream_end_time = None;
    let pricing_model = deployment_pricing_model(
        deployment,
        request
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&ctx.model),
    );
    let mut pricing_params = deployment.oxllm_params.clone();
    pricing_params.model = pricing_model.clone();
    ctx.deployment_pricing = pricing_params.pricing_override();
    ctx.model_router_input_rate = (matches!(deployment.provider_slug(), "azure" | "azure_ai")
        && ctx.deployment_pricing.is_none()
        && [
            ctx.deployment_model.as_deref(),
            Some(pricing_model.as_str()),
        ]
        .into_iter()
        .flatten()
        .any(is_model_router_name))
    .then(|| {
        oxllm_core::cost::price_table()
            .get("azure_ai/model-router")
            .map(|pricing| pricing.input_cost_per_token)
    })
    .flatten();
    ctx.pricing_model = Some(pricing_model);
    ctx.upstream_duration_estimate = Some(sanitize_timeout_secs(
        crate::orchestration::request_controls::seconds(request, "timeout")
            .or(deployment.oxllm_params.timeout)
            .unwrap_or_else(|| state.config.oxllm_settings.request_timeout.unwrap_or(600) as f64),
    ));
    let settings = &state.config.oxllm_settings;
    let mut hook = crate::hooks::BudgetHook::new(
        state.cache.clone(),
        settings.max_budget,
        settings.budget_duration.as_deref().unwrap_or("30d"),
    );
    if let Some(budgets) = settings.model_max_budget.as_ref() {
        hook = hook.with_model_budget(budgets.clone());
    }
    hook.top_up_for_deployment_pricing(ctx).await
}

pub(crate) fn is_model_router_name(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    let model = model
        .strip_prefix("azure/")
        .or_else(|| model.strip_prefix("azure_ai/"))
        .unwrap_or(&model);
    matches!(
        model,
        "model-router" | "model_router" | "azure-model-router" | "azure_model_router"
    ) || ["model-router/", "model_router/"]
        .into_iter()
        .any(|prefix| {
            model
                .strip_prefix(prefix)
                .is_some_and(|deployment| !deployment.is_empty())
        })
}

pub(crate) fn is_model_router_fee_entry(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    matches!(
        model
            .strip_prefix("azure/")
            .or_else(|| model.strip_prefix("azure_ai/"))
            .unwrap_or(&model),
        "model-router" | "model_router"
    )
}

/// Published serving-model prices win. An explicit underlying model can price
/// opaque profiles/deployment aliases without changing their wire identity.
pub(crate) fn deployment_pricing_model(deployment: &Deployment, requested: &str) -> String {
    let provider = deployment.provider_slug();
    let prefix = format!("{provider}/");
    let configured = &deployment.oxllm_params.model;
    let model = if configured.contains('*') {
        format!(
            "{prefix}{}",
            requested.strip_prefix(&prefix).unwrap_or(requested)
        )
    } else {
        configured.clone()
    };
    if oxllm_core::lookup_pricing_with_provider(&model, provider).is_some() {
        return model;
    }
    let hint = deployment
        .model_info
        .as_ref()
        .and_then(|info| info.base_model.as_deref())
        .or_else(|| {
            deployment
                .oxllm_params
                .extra
                .get("base_model")
                .and_then(Value::as_str)
        });
    if let Some(hint) = hint.filter(|hint| !hint.trim().is_empty() && !hint.contains('*')) {
        let hint = hint
            .split_once('/')
            .filter(|(slug, _)| {
                matches!(
                    *slug,
                    "openai" | "azure" | "anthropic" | "bedrock" | "gemini" | "vertex_ai"
                )
            })
            .map_or(hint, |(_, native)| native);
        let hinted = format!("{prefix}{hint}");
        if oxllm_core::lookup_pricing_with_provider(&hinted, provider).is_some() {
            return hinted;
        }
    }
    model
}

/// Map an [`ApiError`] raised by pipeline limit checks or the orchestrators
/// into the closest core [`oxllm_core::Error`] variant so failure hooks see a
/// typed error. This is the **single** `ApiError → Error` mapping for every
/// failure-hook entry point (chat/stream/unary orchestrators, passthrough,
/// realtime); the rate-limit hook keys off it to decide whether the request
/// reached a provider:
///
/// - `budget_exceeded` code → [`oxllm_core::Error::BudgetExceeded`] (a budget
///   cap is its own bucket, regardless of status);
/// - the `upstream_unreachable` / `upstream_timeout` transport marker codes
///   → [`oxllm_core::Error::Network`] (gateway-synthesized transport failure:
///   the request never reached the provider, so RPM/TPM reservations are
///   released);
/// - a 429 carrying the `proxy_rate_limited` code (gateway-enforced limit:
///   pick-loop exhaustion, deployment RPM/TPM caps) →
///   [`oxllm_core::Error::RateLimited`] — likewise never reached a provider;
/// - anything else — including a provider-served 429 (`upstream_rate_limited`
///   code) — is [`oxllm_core::Error::Upstream`]: the provider processed the
///   request, so the rate-limit hook keeps the RPM/TPM reservation (an
///   immediate local retry could otherwise exceed the provider's ceiling).
pub fn core_error_from_api(err: &ApiError) -> oxllm_core::Error {
    if routing::is_routing_error(err) {
        // Checked selector rejection is local: refund pre-call RPM/TPM and
        // budget reservations instead of pretending a provider answered 400.
        oxllm_core::Error::PreCallCheckFailed(err.message.clone())
    } else if err.code.as_deref() == Some("budget_exceeded") {
        oxllm_core::Error::BudgetExceeded(err.message.clone())
    } else if err.code.as_deref() == Some(crate::error::UPSTREAM_UNREACHABLE_CODE)
        || err.code.as_deref() == Some(crate::error::UPSTREAM_TIMEOUT_CODE)
    {
        oxllm_core::Error::Network(err.message.clone())
    } else if err.status == axum::http::StatusCode::TOO_MANY_REQUESTS
        && err.code.as_deref() == Some("proxy_rate_limited")
    {
        oxllm_core::Error::RateLimited(err.message.clone())
    } else if err.code.as_deref() == Some("model_not_found") {
        // Round-trip symmetry with `From<Error> for ApiError`: an unknown
        // model rejected at pick time must reach the post-failure hooks as
        // `ModelNotFound`, not `Upstream`. Two behaviors depend on it: the
        // metrics hook counts unknown models under the fixed `unknown`
        // placeholder (instead of one series per arbitrary name), and the
        // rate-limit hook releases the pre-pick RPM slot (an unknown model
        // never reached any provider, so holding the slot until TTL would
        // let garbage model names drain rate limits). `ApiError::upstream`
        // never uses this code, so the match is collision-free.
        oxllm_core::Error::ModelNotFound(err.message.clone())
    } else {
        oxllm_core::Error::Upstream {
            status: err.status.as_u16(),
            message: err.message.clone(),
        }
    }
}

/// U49: derive a stable, low-cardinality `error.type` label for the OTel span
/// error attribute from a core [`oxllm_core::Error`]. Used by the post-failure
/// path to stamp `error.type` / `error.message` on the request span. Returns a
/// static string so the label never allocates.
pub fn error_type_label(err: &oxllm_core::Error) -> &'static str {
    use oxllm_core::Error;
    match err {
        Error::BadRequest(_) => "bad_request",
        Error::Unauthorized => "unauthorized",
        Error::ModelNotFound(_) => "model_not_found",
        Error::RateLimited(_) => "rate_limited",
        Error::BudgetExceeded(_) => "budget_exceeded",
        Error::Upstream { .. } => "upstream_error",
        Error::Network(_) => "network_error",
        Error::Serialization(_) => "serialization",
        Error::Config(_) => "config",
        Error::PreCallCheckFailed(_) => "pre_call_check_failed",
        Error::Transform(_) => "transform",
        Error::Internal => "internal",
        // `Error` is `#[non_exhaustive]`; a future variant added to
        // oxllm-core must not break this mapping. Fall back to a generic
        // label rather than panicking so a new error type is still
        // surfaced on the span (just without a dedicated bucket).
        _ => "other",
    }
}

/// Check a deployment's per-minute `rpm` / `tpm` limits (set via
/// `oxllm_params.rpm` / `oxllm_params.tpm` in config).
///
/// Both RPM and TPM use the atomic [`oxllm_core::Cache::incr_if_below`]
/// primitive so concurrent requests cannot race past the ceiling. TPM reserves
/// a 1-token occupancy slot at pick time (true token count is unknown until
/// the upstream responds); [`record_deployment_tpm`] then adds the full token
/// count (over-counting by at most 1 under a limit — safer than under-
/// enforcing). Cache errors fail closed. RPM is not released on failure
/// (the attempt consumed the slot).
///
/// Returns `Err(429)` when either limit is exceeded.
/// Keys: `dep_rpm:{dep_key}`, `dep_tpm:{dep_key}` (TTL 60s).
pub async fn check_deployment_limits(
    state: &AppState,
    deployment: &Deployment,
    dep_key: &str,
) -> Result<(), ApiError> {
    let rpm_limit = deployment.oxllm_params.rpm;
    let tpm_limit = deployment.oxllm_params.tpm;

    if let Some(rpm_limit) = rpm_limit {
        let key = format!("dep_rpm:{dep_key}");
        match state
            .cache
            .incr_if_below(&key, 1.0, rpm_limit as f64, Some(Duration::from_secs(60)))
            .await
        {
            Ok(IncrResult::Allowed(_)) => {}
            Ok(IncrResult::Rejected(_)) => {
                return Err(ApiError::rate_limited(format!(
                    "deployment rpm limit ({rpm_limit}) exceeded for {dep_key}"
                )));
            }
            Err(e) => {
                // Fail closed: match key-level RateLimitHook. A cache outage
                // must not bypass per-deployment RPM.
                tracing::warn!(
                    error = %e,
                    key = %dep_key,
                    "rate limit check failed (cache unavailable); failing closed"
                );
                return Err(ApiError::rate_limited(format!(
                    "deployment rate limit check unavailable for {dep_key}"
                )));
            }
        }
    }

    if let Some(tpm_limit) = tpm_limit {
        // Atomic reserve of a 1-token occupancy slot so concurrent inflight
        // requests cannot all pass a soft counter_value snapshot.
        let key = format!("dep_tpm:{dep_key}");
        match state
            .cache
            .incr_if_below(&key, 1.0, tpm_limit as f64, Some(Duration::from_secs(60)))
            .await
        {
            Ok(IncrResult::Allowed(_)) => {}
            Ok(IncrResult::Rejected(_)) => {
                if rpm_limit.is_some() {
                    release_deployment_rpm_slot(state, dep_key).await;
                }
                return Err(ApiError::rate_limited(format!(
                    "deployment tpm limit ({tpm_limit}) exceeded for {dep_key}"
                )));
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    key = %dep_key,
                    "deployment tpm check failed (cache unavailable); failing closed"
                );
                if rpm_limit.is_some() {
                    release_deployment_rpm_slot(state, dep_key).await;
                }
                return Err(ApiError::rate_limited(format!(
                    "deployment tpm check unavailable for {dep_key}"
                )));
            }
        }
    }

    if rpm_limit.is_none() {
        // Unlimited and TPM-only deployments still feed usage-based routing.
        // This observation has no admission ceiling, so a failed write is not
        // a reason to reject a request with no RPM limit.
        if let Err(error) = state
            .cache
            .incr(
                &format!("dep_rpm:{dep_key}"),
                1.0,
                Some(Duration::from_secs(60)),
            )
            .await
        {
            tracing::warn!(%error, "deployment usage observation failed");
        }
    }
    Ok(())
}

/// Roll back the single RPM slot reserved by [`check_deployment_limits`]
/// when the subsequent TPM check rejects or the cache is unavailable.
///
/// Uses [`oxllm_core::Cache::decr_if_exists`] instead of
/// `incr(key, -1.0, ttl)`: if the RPM key expired between reserve and
/// rollback, a plain `incr` would mint a fresh `-1` counter that lets one
/// extra request through in the next window. Skipping the decrement on an
/// expired key accepts at most one over-counted request per window.
async fn release_deployment_rpm_slot(state: &AppState, dep_key: &str) {
    let key = format!("dep_rpm:{dep_key}");
    if let Err(e) = state.cache.decr_if_exists(&key, 1.0).await {
        tracing::warn!(error = %e, key = %dep_key, "dep rpm rollback failed");
    }
}

/// Increment the per-deployment TPM counter (`dep_tpm:{dep_key}`) by `tokens`
/// for the rolling minute window. Called from the post-success paths.
///
/// When a TPM limit is configured, pick time already reserved 1 token via
/// [`check_deployment_limits`]; adding the full count over-counts by at most
/// one token for limited deployments (stricter than under-enforcing). When no
/// limit is set, this is the sole write to the counter.
pub async fn record_deployment_tpm(state: &AppState, dep_key: &str, tokens: u64) {
    if tokens == 0 {
        return;
    }
    let counter_key = format!("dep_tpm:{dep_key}");
    if let Err(e) = state
        .cache
        .incr(&counter_key, tokens as f64, Some(Duration::from_secs(60)))
        .await
    {
        tracing::warn!(error = %e, "dep tpm record failed");
    }
}

/// Fallback request timeout (seconds) used when a deployment's configured
/// `oxllm_params.timeout` is non-finite or non-positive. YAML/JSON special
/// floats (`.inf`, `-.inf`, `.nan`) and negative values parse fine into `f64`
/// but panic `Duration::from_secs_f64`; clamping here keeps the worker alive
/// instead of crashing on the first request to a misconfigured deployment.
const SAFE_TIMEOUT_SECS: f64 = 600.0;

/// Clamp a configured timeout (seconds) to a finite, positive value, falling
/// back to `SAFE_TIMEOUT_SECS` when it is NaN, ±∞, or ≤ 0. Guards
/// `Duration::from_secs_f64` against panicking on operator input.
pub fn sanitize_timeout_secs(timeout_secs: f64) -> Duration {
    if let Ok(duration) = Duration::try_from_secs_f64(timeout_secs) {
        if !duration.is_zero() && std::time::Instant::now().checked_add(duration).is_some() {
            return duration;
        }
    }
    {
        tracing::warn!(
            timeout_secs,
            fallback_secs = SAFE_TIMEOUT_SECS,
            "non-finite or non-positive upstream timeout configured; using fallback"
        );
        Duration::from_secs_f64(SAFE_TIMEOUT_SECS)
    }
}

/// Execute an upstream HTTP call via the shared client.
///
/// Centralizes the `reqwest` send, timeout application, and network-error
/// classification (timeout → 504, other transport error → 502) so call sites
/// don't each duplicate the boilerplate. Transport failures are rendered via
/// the marker-carrying [`ApiError::gateway_timeout`] /
/// [`ApiError::network_unreachable`] constructors so downstream
/// re-classification (`core_error_from_api`) can tell "request never reached
/// the provider" apart from "provider served an error response". Callers
/// remain responsible for credential resolution, header building, and
/// response handling (sanitize, status classification, etc.) — the goal is
/// one place for the transport call, not forcing every caller into a single
/// response shape.
///
/// `timeout_secs` is a reqwest **total** deadline (connect until the last
/// body byte). That is the right shape for unary calls, but it hard-kills
/// any stream whose total duration exceeds the budget even while chunks keep
/// flowing. Streaming callers pass `None` (no total deadline; the send is
/// bounded separately and the body is bounded per-chunk by the stream pump's
/// idle timeout, mirroring the reference per-read httpx timeout semantics).
pub async fn execute_upstream_json(
    state: &AppState,
    deployment: &Deployment,
    method: Method,
    url: &str,
    body: Bytes,
    headers: HeaderMap,
    timeout_secs: Option<f64>,
) -> Result<reqwest::Response, ApiError> {
    tracing::debug!(
     provider = %deployment.provider_slug(),
     model = %deployment.oxllm_params.model,
     method = %method,
     url = %crate::security::sanitize_url(url),
     timeout_secs,
     "sending upstream request");

    let req_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|_| ApiError::bad_request("invalid HTTP method"))?;

    let mut request = state.http_client.request(req_method, url).headers(headers);
    if let Some(secs) = timeout_secs {
        request = request.timeout(sanitize_timeout_secs(secs));
    }
    request.body(body).send().await.map_err(|e| {
        if e.is_timeout() {
            ApiError::gateway_timeout()
        } else {
            // A transport failure (DNS/TLS/connect) is an upstream-side
            // problem: render 502 with a generic body. The raw reqwest
            // detail (which embeds the resolved upstream URL — custom
            // `api_base` targets, private gateways) must never reach the
            // client, so it is logged server-side only.
            tracing::error!(
                error.kind = "network",
                detail = %e,
                "network error reaching upstream"
            );
            ApiError::network_unreachable()
        }
    })
}

/// Stamp the cost header on a response from a pre-computed cost value.
pub fn apply_cost_header(response: &mut Response, cost: Option<f64>) {
    if let Some(c) = cost {
        set_header(response, COST_HEADER, &format!("{c}"));
    }
}

/// Stamp the **full** set of `x-ratelimit-*` headers (`limit`, `remaining`,
/// and `reset` for both RPM and TPM) on a response by reading the current
/// per-key RPM/TPM counters from the cache and the per-key limits from the
/// call context.
///
/// This is called after `pre_call` succeeds (so the counters already reflect
/// the in-flight reservation) and stamps all six headers — including the
/// `x-ratelimit-limit-*` headers that [`apply_ratelimit_headers`] (which
/// reads the stashed [`RateLimitInfo`]) does not carry.
///
/// No-op (stamps nothing) for master-key requests (master bypasses rate
/// limits, so the headers would be misleading) and cache hits (the request
/// did not consume an upstream slot). Only the dimensions whose limit is
/// active are stamped.
///
/// Counter keys: `rpm:{api_key_hash}`, `tpm:{api_key_hash}` (TTL 60s).
///
/// Limit source: the per-key limit (`ctx.rpm` / `ctx.tpm`) is preferred —
/// when set, `remaining` is derived live from the per-key counter
/// (`limit - current`). When no per-key limit is set but the rate-limit hook
/// stashed a binding limit (a global/team/org/project limit applied), the
/// stashed `limit-*` and `remaining-*` snapshot from `pre_call` is used
/// verbatim. Reading a pre_call snapshot (instead of re-deriving
/// `remaining + live_counter`) keeps the headers consistent under
/// concurrency: a concurrent request that bumps the counter between
/// `pre_call` and the stamp can no longer inflate the derived `limit-*`
/// header.
pub async fn stamp_ratelimit_headers(
    response: &mut Response,
    state: &AppState,
    call_ctx: &CallContext,
) {
    if call_ctx.is_master || call_ctx.cache_hit {
        return;
    }

    // RPM dimension.
    let rpm_key = format!("rpm:{}", call_ctx.api_key_hash);
    let reset_rpm = reset_secs_label(call_ctx.ratelimit_info.reset_rpm_secs);
    if let Some(limit) = call_ctx.rpm.filter(|limit| {
        call_ctx
            .ratelimit_info
            .limit_rpm
            .is_none_or(|binding| binding >= *limit)
    }) {
        let limit = call_ctx
            .ratelimit_info
            .limit_rpm
            .map_or(limit, |binding| binding.min(limit));
        // Fail closed: on a cache error, omit the headers entirely instead of
        // stamping remaining=limit (which would mislead clients the same way
        // a bypassed limit check would).
        match state.cache.try_counter_value(&rpm_key).await {
            Err(e) => {
                tracing::warn!(error = %e, "rpm counter read failed; omitting ratelimit headers")
            }
            Ok(rpm_current) => {
                let remaining = ((limit as f64) - rpm_current.unwrap_or(0.0))
                    .max(0.0)
                    .round() as u32;
                let remaining = call_ctx
                    .ratelimit_info
                    .remaining_rpm
                    .map_or(remaining, |binding| binding.min(remaining));
                set_header(response, RATELIMIT_LIMIT_REQUESTS, &limit.to_string());
                set_header(
                    response,
                    RATELIMIT_REMAINING_REQUESTS,
                    &remaining.to_string(),
                );
                set_header(response, RATELIMIT_RESET_REQUESTS, &reset_rpm);
            }
        }
    } else if let (Some(limit), Some(remaining)) = (
        call_ctx.ratelimit_info.limit_rpm,
        call_ctx.ratelimit_info.remaining_rpm,
    ) {
        // No per-key RPM, but a global/team/org/project limit applied. Use
        // the pre_call snapshot so a concurrent counter bump can't inflate
        // the derived limit header.
        set_header(response, RATELIMIT_LIMIT_REQUESTS, &limit.to_string());
        set_header(
            response,
            RATELIMIT_REMAINING_REQUESTS,
            &remaining.to_string(),
        );
        set_header(response, RATELIMIT_RESET_REQUESTS, &reset_rpm);
    }

    // TPM dimension.
    let tpm_key = format!("tpm:{}", call_ctx.api_key_hash);
    let reset_tpm = reset_secs_label(call_ctx.ratelimit_info.reset_tpm_secs);
    if let Some(limit) = call_ctx.tpm.filter(|limit| {
        call_ctx
            .ratelimit_info
            .limit_tpm
            .is_none_or(|binding| binding >= *limit)
    }) {
        let limit = call_ctx
            .ratelimit_info
            .limit_tpm
            .map_or(limit, |binding| binding.min(limit));
        match state.cache.try_counter_value(&tpm_key).await {
            Err(e) => {
                tracing::warn!(error = %e, "tpm counter read failed; omitting ratelimit headers")
            }
            Ok(tpm_current) => {
                let remaining = ((limit as f64) - tpm_current.unwrap_or(0.0))
                    .max(0.0)
                    .round() as u32;
                let remaining = call_ctx
                    .ratelimit_info
                    .remaining_tpm
                    .map_or(remaining, |binding| binding.min(remaining));
                set_header(response, RATELIMIT_LIMIT_TOKENS, &limit.to_string());
                set_header(response, RATELIMIT_REMAINING_TOKENS, &remaining.to_string());
                set_header(response, RATELIMIT_RESET_TOKENS, &reset_tpm);
            }
        }
    } else if let (Some(limit), Some(remaining)) = (
        call_ctx.ratelimit_info.limit_tpm,
        call_ctx.ratelimit_info.remaining_tpm,
    ) {
        set_header(response, RATELIMIT_LIMIT_TOKENS, &limit.to_string());
        set_header(response, RATELIMIT_REMAINING_TOKENS, &remaining.to_string());
        set_header(response, RATELIMIT_RESET_TOKENS, &reset_tpm);
    }
}

/// Format the `x-ratelimit-reset-*` window label. Falls back to `60s`
/// (per-minute counter window) when the snapshot carries no reset value.
fn reset_secs_label(secs: Option<u64>) -> String {
    format!("{}s", secs.unwrap_or(60))
}

/// Stamp `x-ratelimit-*` headers on a response from a stashed
/// [`RateLimitInfo`]. No-op when `is_master` is true (master key bypasses
/// rate limits, so the headers would be misleading).
pub fn apply_ratelimit_headers(response: &mut Response, info: &RateLimitInfo, is_master: bool) {
    if is_master {
        return;
    }
    if let Some(limit) = info.limit_rpm {
        set_header(response, RATELIMIT_LIMIT_REQUESTS, &limit.to_string());
    }
    if let Some(limit) = info.limit_tpm {
        set_header(response, RATELIMIT_LIMIT_TOKENS, &limit.to_string());
    }
    if let Some(remaining) = info.remaining_rpm {
        set_header(
            response,
            RATELIMIT_REMAINING_REQUESTS,
            &remaining.to_string(),
        );
    }
    if let Some(remaining) = info.remaining_tpm {
        set_header(response, RATELIMIT_REMAINING_TOKENS, &remaining.to_string());
    }
    if let Some(reset) = info.reset_rpm_secs {
        set_header(response, RATELIMIT_RESET_REQUESTS, &format!("{reset}s"));
    }
    if let Some(reset) = info.reset_tpm_secs {
        set_header(response, RATELIMIT_RESET_TOKENS, &format!("{reset}s"));
    }
}

/// Set a static header on the response if the value parses.
pub fn set_header(response: &mut Response, name: &'static str, value: &str) {
    if let Ok(hv) = HeaderValue::from_str(value) {
        response.headers_mut().insert(name, hv);
    }
}

/// Check if the requested model is allowed by the key/team model ACLs.
///
/// - Master-key callers bypass all checks.
/// - When `key_models` / `team_models` is `None` or empty, no restriction is
///   applied at that level (backward compatible).
/// - Both key and team restrictions must pass (the team ceiling is an
///   intersection, not a union).
/// - Patterns support `*` wildcards via [`glob_match`] (e.g. `"gpt-4*"`
///   matches `"gpt-4o"`).
///
/// Extract the OpenAI end-user id from a request body (`"user"` field).
/// Returns `None` when missing, non-string, or blank after trim.
pub fn extract_end_user_id(body: &Value) -> Option<String> {
    body.get("user")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Resolve customer limits from the request body `"user"` field and stamp
/// them onto `call_ctx`. Enforces `blocked` and `model_blocklist` when a
/// customer row exists. Unknown end-user ids pass through without customer
/// scope. No-op when the `db` feature is off or no database is configured.
async fn apply_customer_context(
    state: &AppState,
    call_ctx: &mut CallContext,
) -> Result<(), ApiError> {
    let Some(end_user_id) = extract_end_user_id(call_ctx.request_body.as_ref()) else {
        return Ok(());
    };
    // Always stamp the end-user id so spend logs can attribute traffic even
    // when no customer row exists yet (and when no DB is configured).
    call_ctx.end_user_id = Some(end_user_id.clone());

    #[cfg(feature = "db")]
    {
        let Some(db) = state.db.as_ref() else {
            return Ok(());
        };
        match crate::auth::customer_limits_for(state, db, &end_user_id).await {
            Ok(None) => Ok(()),
            Ok(Some(limits)) => {
                if limits.blocked {
                    tracing::info!(
                        end_user_id = %end_user_id,
                        "request rejected: end-user is blocked"
                    );
                    return Err(ApiError::forbidden(format!(
                        "end-user '{end_user_id}' is blocked"
                    )));
                }
                if let Some(ref blocklist) = limits.model_blocklist {
                    if !blocklist.is_empty()
                        && blocklist.iter().any(|pat| glob_match(pat, &call_ctx.model))
                    {
                        return Err(ApiError::forbidden(format!(
                            "Model '{}' is blocked for end-user '{end_user_id}'",
                            call_ctx.model
                        )));
                    }
                }
                call_ctx.customer_max_budget = limits.max_budget;
                call_ctx.customer_budget_duration = limits.budget_duration;
                Ok(())
            }
            Err(()) => {
                // Fail closed: a DB outage must not let a blocked customer
                // through. 503 so load balancers can retry / alert.
                Err(ApiError::upstream(
                    503,
                    "unable to resolve end-user limits; try again",
                ))
            }
        }
    }
    #[cfg(not(feature = "db"))]
    {
        let _ = state;
        Ok(())
    }
}

/// Recheck customer abuse policy for a model-group fallback. Budget attribution
/// stays on the original request context; only admission uses the new model.
pub(crate) async fn check_customer_model(
    state: &AppState,
    ctx: &CallContext,
    model: &str,
) -> Result<(), ApiError> {
    if model == ctx.model || ctx.end_user_id.is_none() {
        return Ok(());
    }
    let mut fallback = ctx.clone();
    fallback.model = model.to_string();
    apply_customer_context(state, &mut fallback).await
}

/// Resolve tag limits from the `x-oxllm-tag` header value and stamp them
/// onto `call_ctx`. Unknown tag names pass through without tag budget
/// scope. No-op when the header is absent, the `db` feature is off, or no
/// database is configured.
async fn apply_tag_context(
    state: &AppState,
    call_ctx: &mut CallContext,
    tag: Option<&str>,
) -> Result<(), ApiError> {
    let tags = merge_request_tags(
        Some(&call_ctx.request_body),
        tag.into_iter().flat_map(|tag| tag.split(',')),
    );
    apply_request_tags(state, call_ctx, &tags).await
}

/// Stamp every request tag onto `call_ctx` and resolve per-tag budget
/// limits for each. The first tag becomes [`CallContext::request_tag`]
/// (the primary, for spend attribution compatibility); the full list lands
/// in [`CallContext::request_tags`] (routing + multi-tag spend logging).
/// Each tag that carries a `tags.max_budget` row contributes a budget
/// scope, and the MOST restrictive ceiling wins — the budget hook reserves
/// against every carried tag's own counter, so a request counts toward
/// each application's budget it is tagged with (the shared-key /
/// per-application cap pattern).
///
/// No-op for an empty list. A tag-limits lookup failure fails closed with
/// 503, mirroring `apply_tag_context`.
pub async fn apply_request_tags(
    state: &AppState,
    call_ctx: &mut CallContext,
    tags: &[String],
) -> Result<(), ApiError> {
    // `state` is only read through the DB-backed tag-limit lookup below.
    #[cfg(not(feature = "db"))]
    let _ = state;
    let tags = merge_request_tags(None, tags.iter().map(String::as_str));
    if tags.is_empty() {
        return Ok(());
    }
    // Always stamp so spend attribution records the tags even when no tag
    // row exists (and when no DB is configured).
    call_ctx.request_tags = tags.clone();
    call_ctx.request_tag = tags.first().cloned();
    call_ctx.tag_budgets.clear();
    call_ctx.tag_max_budget = None;

    #[cfg(feature = "db")]
    {
        let Some(db) = state.db.as_ref() else {
            return Ok(());
        };
        for tag in &tags {
            match crate::auth::tag_limits_for(state, db, tag).await {
                Ok(None) => {}
                Ok(Some(limits)) => {
                    if let Some(budget) = limits.max_budget {
                        call_ctx.tag_budgets.push((tag.clone(), budget));
                        call_ctx.tag_max_budget = Some(match call_ctx.tag_max_budget {
                            Some(current) => current.min(budget),
                            None => budget,
                        });
                    }
                }
                Err(()) => {
                    return Err(ApiError::upstream(
                        503,
                        "unable to resolve tag limits; try again",
                    ));
                }
            }
        }
    }
    Ok(())
}

pub fn check_model_acl(
    auth_user: &AuthUser,
    model: &str,
    key_models: Option<&[String]>,
    team_models: Option<&[String]>,
    user_models: Option<&[String]>,
) -> Result<(), ApiError> {
    if auth_user.is_master {
        return Ok(());
    }
    if let Some(models) = key_models {
        if !models.is_empty() && !models.iter().any(|m| glob_match(m, model)) {
            // L5: do NOT echo the configured allowed-model list back to the
            // client — that leaks the full model ACL to anyone who can issue
            // a request, including attackers probing for valid model names.
            return Err(ApiError::forbidden(format!(
                "Model '{model}' is not allowed for this key"
            )));
        }
    }
    if let Some(models) = team_models {
        if !models.is_empty() && !models.iter().any(|m| glob_match(m, model)) {
            return Err(ApiError::forbidden(format!(
                "Model '{model}' is not allowed for this key's team."
            )));
        }
    }
    if let Some(models) = user_models {
        if !models.is_empty() && !models.iter().any(|m| glob_match(m, model)) {
            return Err(ApiError::forbidden(format!(
                "Model '{model}' is not allowed for this key's user."
            )));
        }
    }
    Ok(())
}

/// Simple glob matcher supporting `*` wildcards. A `*` matches any sequence
/// (including the empty sequence). A pattern without `*` requires an exact
/// match. Multiple `*` are supported (e.g. `"gpt-*-mini"`).
pub fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == text {
        return true;
    }
    if !pattern.contains('*') {
        return false;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !text[pos..].starts_with(part) {
                return false;
            }
            pos += part.len();
        } else if i == parts.len() - 1 {
            return text[pos..].ends_with(part);
        } else {
            match text[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig};
    use crate::state::AppState;
    use async_trait::async_trait;
    use oxllm_core::error::Result;
    use oxllm_core::{Error, Hook, HookAction};
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// A hook that records the order and arguments of its invocations.
    struct RecordingHook {
        calls: Arc<Mutex<Vec<&'static str>>>,
    }

    impl RecordingHook {
        fn new() -> (Self, Arc<Mutex<Vec<&'static str>>>) {
            let calls = Arc::new(Mutex::new(Vec::new()));
            (
                RecordingHook {
                    calls: calls.clone(),
                },
                calls,
            )
        }
    }

    #[async_trait]
    impl Hook for RecordingHook {
        async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
            self.calls.lock().unwrap().push("pre_call");
            Ok(HookAction::Continue)
        }

        async fn post_success(&self, _ctx: &CallContext, resp: &ResponseSummary) {
            let mut guard = self.calls.lock().unwrap();
            guard.push("post_success");
            guard.push(if resp.cost.is_some() {
                "post_success_with_cost"
            } else {
                "post_success_no_cost"
            });
        }

        async fn post_failure(&self, _ctx: &CallContext, _err: &Error) {
            self.calls.lock().unwrap().push("post_failure");
        }
    }

    /// A hook capturing the [`ResponseSummary`] each `post_success` observed,
    /// so tests can assert exactly what downstream hooks (e.g. the cache
    /// hook, which stores `body` verbatim) would see.
    struct SummaryCaptureHook {
        summaries: Arc<Mutex<Vec<ResponseSummary>>>,
    }

    #[async_trait]
    impl Hook for SummaryCaptureHook {
        async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
            Ok(HookAction::Continue)
        }
        async fn post_success(&self, _ctx: &CallContext, resp: &ResponseSummary) {
            self.summaries.lock().unwrap().push(resp.clone());
        }
    }

    /// A hook that short-circuits with a cached body.
    struct CacheHitHook;
    #[async_trait]
    impl Hook for CacheHitHook {
        async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
            Ok(HookAction::ShortCircuit(serde_json::json!({
            "cached": true
            })))
        }
    }

    /// A hook that rejects every request.
    struct RejectHook;
    #[async_trait]
    impl Hook for RejectHook {
        async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
            Ok(HookAction::Reject(Error::RateLimited("nope".to_string())))
        }
    }

    fn base_config() -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    fn state_with_hooks(hooks: Vec<Arc<dyn Hook>>) -> AppState {
        let mut state = AppState::with_registry(base_config(), oxllm_providers::default_registry())
            .expect("state built");
        state.hooks = hooks;
        state
    }

    fn state_with_in_memory_cache() -> AppState {
        AppState::with_registry(base_config(), oxllm_providers::default_registry())
            .expect("state built")
    }

    fn dummy_ctx() -> CallContext {
        CallContext::for_test("m")
    }

    fn deployment_with_limits(rpm: Option<u32>, tpm: Option<u32>) -> Deployment {
        use oxllm_core::deployment::OxllmParams;
        Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                rpm,
                tpm,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn pre_call_returns_none_when_all_hooks_continue() {
        let (hook, _calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let mut pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        let result = pipeline.pre_call().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn pre_call_returns_cached_value_on_short_circuit() {
        let state = state_with_hooks(vec![Arc::new(CacheHitHook)]);
        let mut pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        let result = pipeline.pre_call().await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap()["cached"], true);
    }

    #[tokio::test]
    async fn pre_call_returns_err_on_reject() {
        let state = state_with_hooks(vec![Arc::new(RejectHook)]);
        let mut pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        assert!(pipeline.pre_call().await.is_err());
    }

    // Bug F3: when a later hook rejects, the reservations of hooks that
    // already succeeded must be rolled back via post_failure.
    #[tokio::test]
    async fn pre_call_reject_rolls_back_earlier_hooks() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook), Arc::new(RejectHook)]);
        let mut pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        let result = pipeline.pre_call().await;
        assert!(result.is_err(), "reject must propagate");
        let guard = calls.lock().unwrap();
        assert_eq!(
            &guard[..],
            &["pre_call", "post_failure"],
            "earlier hook must be rolled back when a later hook rejects"
        );
    }

    #[test]
    fn core_error_from_api_maps_rate_limited() {
        let err = ApiError::rate_limited("deployment rpm limit exceeded");
        let core = core_error_from_api(&err);
        match core {
            Error::RateLimited(msg) => assert!(msg.contains("rpm limit")),
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn core_error_from_api_preserves_model_not_found() {
        // Unknown-model rejections must reach post-failure hooks as
        // `ModelNotFound` (metrics placeholder + RPM slot release), not as
        // generic `Upstream`.
        let err = ApiError::model_not_found("nope");
        match core_error_from_api(&err) {
            Error::ModelNotFound(_) => {}
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn core_error_from_api_maps_other_status_to_upstream() {
        let err = ApiError::internal("cache backend down");
        let core = core_error_from_api(&err);
        match core {
            Error::Upstream { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "cache backend down");
            }
            other => panic!("expected Upstream, got {other:?}"),
        }
    }

    #[test]
    fn core_error_from_api_maps_transport_markers_to_network() {
        // Transport failures rendered by `execute_upstream_json` (502
        // connection refused / DNS / TLS, 504 local timeout) never reached
        // the provider: they must map back to `Error::Network` so the
        // rate-limit hook releases RPM/TPM reservations instead of keeping
        // them for the full 60s window like a provider-served error.
        for err in [ApiError::network_unreachable(), ApiError::gateway_timeout()] {
            let core = core_error_from_api(&err);
            match core {
                Error::Network(msg) => assert!(!msg.is_empty()),
                other => panic!("expected Network for code {:?}, got {other:?}", err.code),
            }
        }
    }

    #[test]
    fn core_error_from_api_round_trips_core_network() {
        // `ApiError::from(Error::Network)` loses the variant, but the
        // `upstream_unreachable` marker lets the inverse mapping recover it —
        // passthrough failure funnels rely on this round trip.
        let core_err = core_error_from_api(&ApiError::from(Error::Network(
            "connection refused".to_string(),
        )));
        assert!(matches!(core_err, Error::Network(_)));
    }

    #[test]
    fn core_error_from_api_keeps_provider_served_502_as_upstream() {
        // A genuine provider 504/502 (an HTTP response arrived) must stay
        // `Error::Upstream` — the provider processed the request and the
        // RPM/TPM reservations must be kept.
        let err = ApiError::upstream(502, "provider returned 502");
        let core = core_error_from_api(&err);
        match core {
            Error::Upstream { status, .. } => assert_eq!(status, 502),
            other => panic!("expected Upstream, got {other:?}"),
        }
        let err = ApiError::upstream(504, "provider returned 504");
        assert!(matches!(core_error_from_api(&err), Error::Upstream { .. }));
    }

    /// LOW (C11): a provider-served 429 (`upstream_rate_limited` code — an
    /// HTTP response arrived) must map to `Error::Upstream` so the rate-limit
    /// hook KEEPS the RPM/TPM reservations. Only gateway-enforced 429s (the
    /// `proxy_rate_limited` code: pick-loop exhaustion, deployment caps,
    /// proxy rate limits) map to `Error::RateLimited` and release — the
    /// request never reached a provider there. Previously this mapping
    /// released RPM for provider 429s on the passthrough/realtime paths
    /// (and its orchestrator twin kept them — two sources of truth).
    #[test]
    fn core_error_from_api_keeps_provider_served_429_as_upstream() {
        let err = ApiError::upstream(429, "provider throttled us");
        match core_error_from_api(&err) {
            Error::Upstream { status, .. } => assert_eq!(status, 429),
            other => panic!("expected Upstream for provider 429, got {other:?}"),
        }
        let proxy = ApiError::rate_limited("all deployments unavailable");
        assert!(matches!(core_error_from_api(&proxy), Error::RateLimited(_)));
    }

    // ---- U49: error_type_label -------------------------------------------

    #[test]
    fn error_type_label_maps_each_variant() {
        assert_eq!(
            error_type_label(&Error::BadRequest("x".into())),
            "bad_request"
        );
        assert_eq!(error_type_label(&Error::Unauthorized), "unauthorized");
        assert_eq!(
            error_type_label(&Error::ModelNotFound("m".into())),
            "model_not_found"
        );
        assert_eq!(
            error_type_label(&Error::RateLimited("rpm".into())),
            "rate_limited"
        );
        assert_eq!(
            error_type_label(&Error::BudgetExceeded("budget".into())),
            "budget_exceeded"
        );
        assert_eq!(
            error_type_label(&Error::Upstream {
                status: 503,
                message: "boom".into()
            }),
            "upstream_error"
        );
        assert_eq!(
            error_type_label(&Error::Network("timeout".into())),
            "network_error"
        );
        assert_eq!(
            error_type_label(&Error::Serialization("bad".into())),
            "serialization"
        );
        assert_eq!(error_type_label(&Error::Config("c".into())), "config");
        assert_eq!(
            error_type_label(&Error::PreCallCheckFailed("f".into())),
            "pre_call_check_failed"
        );
        assert_eq!(error_type_label(&Error::Transform("t".into())), "transform");
        assert_eq!(error_type_label(&Error::Internal), "internal");
    }

    #[tokio::test]
    async fn post_success_runs_hooks_and_returns_cost() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        let usage = UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let cost = pipeline
            .post_success("gpt-4o", Some(&usage), None, "openai")
            .await;
        assert!(cost.is_some(), "cost should be computed for gpt-4o");
        let guard = calls.lock().unwrap();
        assert!(guard.contains(&"post_success"));
        assert!(guard.contains(&"post_success_with_cost"));
    }

    #[tokio::test]
    async fn post_success_returns_none_cost_when_no_usage() {
        let state = state_with_hooks(vec![]);
        let pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        let cost = pipeline.post_success("gpt-4o", None, None, "openai").await;
        assert!(cost.is_none());
    }

    #[tokio::test]
    async fn post_failure_runs_hooks() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        pipeline
            .post_failure(&Error::Network("boom".to_string()))
            .await;
        let guard = calls.lock().unwrap();
        assert!(guard.contains(&"post_failure"));
    }

    #[tokio::test]
    async fn for_request_generates_unique_request_ids() {
        let state = state_with_in_memory_cache();
        let auth = crate::auth::AuthUser::open();
        let p1 = EndpointPipeline::for_request(&state, &auth, "m", Value::Null)
            .await
            .expect("ok");
        let p2 = EndpointPipeline::for_request(&state, &auth, "m", Value::Null)
            .await
            .expect("ok");
        assert_ne!(p1.request_id(), p2.request_id());
        assert!(!p1.request_id().is_empty());
    }

    #[tokio::test]
    async fn for_request_propagates_auth_fields() {
        let state = state_with_in_memory_cache();
        let auth = crate::auth::AuthUser::open();
        let pipeline = EndpointPipeline::for_request(&state, &auth, "gpt-4o", Value::Null)
            .await
            .expect("ok");
        assert_eq!(pipeline.call_ctx.model, "gpt-4o");
        assert!(!pipeline.call_ctx.request_id.is_empty());
        assert!(!pipeline.call_ctx.cache_hit);
    }

    #[tokio::test]
    async fn check_limits_no_limits_passes() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(None, None);
        let pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        assert!(
            pipeline
                .check_limits(&dep, "gpt-4o:openai/gpt-4o")
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn check_limits_rpm_at_limit_rejects() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(Some(10), None);
        let _ = state
            .cache
            .incr("dep_rpm:gpt-4o:openai/gpt-4o", 10.0, None)
            .await;
        let pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        let result = pipeline.check_limits(&dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn check_limits_tpm_at_limit_rejects() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(None, Some(1000));
        let _ = state
            .cache
            .incr("dep_tpm:gpt-4o:openai/gpt-4o", 1000.0, None)
            .await;
        let pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };
        let result = pipeline.check_limits(&dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn free_post_success_computes_cost_and_runs_hooks() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let ctx = dummy_ctx();
        let usage = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            ..Default::default()
        };
        let cost = post_success(&state, &ctx, "gpt-4o", Some(&usage), None, None, "openai").await;
        assert!(cost.is_some());
        assert!(calls.lock().unwrap().contains(&"post_success"));
    }

    #[tokio::test]
    async fn free_post_failure_runs_hooks() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let ctx = dummy_ctx();
        post_failure(&state, &ctx, &Error::Network("x".to_string())).await;
        assert!(calls.lock().unwrap().contains(&"post_failure"));
    }

    #[tokio::test]
    async fn check_deployment_limits_no_limits_passes() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(None, None);
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn check_deployment_limits_rpm_under_limit_passes() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(Some(10), None);
        let _ = state
            .cache
            .incr("dep_rpm:gpt-4o:openai/gpt-4o", 5.0, None)
            .await;
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn check_deployment_limits_rpm_at_limit_rejects() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(Some(10), None);
        let _ = state
            .cache
            .incr("dep_rpm:gpt-4o:openai/gpt-4o", 10.0, None)
            .await;
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn check_deployment_limits_tpm_at_limit_rejects() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(None, Some(1000));
        let _ = state
            .cache
            .incr("dep_tpm:gpt-4o:openai/gpt-4o", 1000.0, None)
            .await;
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn record_deployment_tpm_increments_counter() {
        let state = state_with_in_memory_cache();
        record_deployment_tpm(&state, "gpt-4o:openai/gpt-4o", 500).await;
        let val = state
            .cache
            .get("dep_tpm:gpt-4o:openai/gpt-4o")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!((val - 500.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn record_deployment_tpm_zero_is_noop() {
        let state = state_with_in_memory_cache();
        record_deployment_tpm(&state, "gpt-4o:openai/gpt-4o", 0).await;
        assert!(
            state
                .cache
                .get("dep_tpm:gpt-4o:openai/gpt-4o")
                .await
                .is_none()
        );
    }

    #[test]
    fn apply_cost_header_sets_header_when_some() {
        let mut resp = Response::new(axum::body::Body::from(""));
        apply_cost_header(&mut resp, Some(0.0042));
        assert_eq!(
            resp.headers()
                .get(COST_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("0.0042")
        );
    }

    #[test]
    fn apply_ratelimit_headers_stamps_all_four_when_set() {
        let mut resp = Response::new(axum::body::Body::from(""));
        let info = RateLimitInfo {
            remaining_rpm: Some(7),
            remaining_tpm: Some(900),
            reset_rpm_secs: Some(60),
            reset_tpm_secs: Some(60),
            ..Default::default()
        };
        apply_ratelimit_headers(&mut resp, &info, false);
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_REMAINING_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("7")
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_REMAINING_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("900")
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_RESET_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("60s")
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_RESET_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("60s")
        );
    }

    #[test]
    fn apply_ratelimit_headers_skips_unset_dimensions() {
        let mut resp = Response::new(axum::body::Body::from(""));
        let info = RateLimitInfo {
            remaining_rpm: Some(5),
            remaining_tpm: None,
            reset_rpm_secs: Some(60),
            reset_tpm_secs: None,
            ..Default::default()
        };
        apply_ratelimit_headers(&mut resp, &info, false);
        assert!(resp.headers().get(RATELIMIT_REMAINING_REQUESTS).is_some());
        assert!(resp.headers().get(RATELIMIT_REMAINING_TOKENS).is_none());
        assert!(resp.headers().get(RATELIMIT_RESET_REQUESTS).is_some());
        assert!(resp.headers().get(RATELIMIT_RESET_TOKENS).is_none());
    }

    #[test]
    fn apply_ratelimit_headers_noop_when_default_info() {
        let mut resp = Response::new(axum::body::Body::from(""));
        apply_ratelimit_headers(&mut resp, &RateLimitInfo::default(), false);
        assert!(resp.headers().get(RATELIMIT_REMAINING_REQUESTS).is_none());
        assert!(resp.headers().get(RATELIMIT_REMAINING_TOKENS).is_none());
        assert!(resp.headers().get(RATELIMIT_RESET_REQUESTS).is_none());
        assert!(resp.headers().get(RATELIMIT_RESET_TOKENS).is_none());
    }

    #[test]
    fn apply_ratelimit_headers_skipped_for_master_key() {
        let mut resp = Response::new(axum::body::Body::from(""));
        let info = RateLimitInfo {
            remaining_rpm: Some(7),
            remaining_tpm: Some(900),
            reset_rpm_secs: Some(60),
            reset_tpm_secs: Some(60),
            ..Default::default()
        };
        apply_ratelimit_headers(&mut resp, &info, true);
        assert!(resp.headers().get(RATELIMIT_REMAINING_REQUESTS).is_none());
        assert!(resp.headers().get(RATELIMIT_REMAINING_TOKENS).is_none());
        assert!(resp.headers().get(RATELIMIT_RESET_REQUESTS).is_none());
        assert!(resp.headers().get(RATELIMIT_RESET_TOKENS).is_none());
    }

    // ---- stamp_ratelimit_headers -----------------------------------------

    fn ctx_with_key(model: &str, hash: &str) -> CallContext {
        CallContext {
            model: model.to_string(),
            api_key_hash: hash.to_string(),
            request_id: "r".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn stamp_ratelimit_headers_stamps_all_six_when_limits_set() {
        let state = state_with_in_memory_cache();
        // Pre-charge the counters as if pre_call already reserved: rpm at 3,
        // tpm at 150 (so remaining = limit - current).
        let _ = state.cache.incr("rpm:key1", 3.0, None).await;
        let _ = state.cache.incr("tpm:key1", 150.0, None).await;
        let ctx = CallContext {
            rpm: Some(10),
            tpm: Some(1000),
            ..ctx_with_key("gpt-4o", "key1")
        };
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_LIMIT_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("10")
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_REMAINING_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("7"),
            "remaining = limit(10) - current(3)"
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_RESET_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("60s")
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_LIMIT_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("1000")
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_REMAINING_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("850"),
            "remaining = limit(1000) - current(150)"
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_RESET_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("60s")
        );
    }

    #[tokio::test]
    async fn stamp_ratelimit_headers_remaining_clamps_at_zero() {
        let state = state_with_in_memory_cache();
        // Counter already over the limit (current > limit).
        let _ = state.cache.incr("rpm:key2", 12.0, None).await;
        let ctx = CallContext {
            rpm: Some(10),
            ..ctx_with_key("gpt-4o", "key2")
        };
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_REMAINING_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("0"),
            "remaining must clamp at 0 when current >= limit"
        );
    }

    #[tokio::test]
    async fn stamp_ratelimit_headers_noop_for_master_key() {
        let state = state_with_in_memory_cache();
        let _ = state.cache.incr("rpm:key3", 3.0, None).await;
        let ctx = CallContext {
            rpm: Some(10),
            tpm: Some(1000),
            is_master: true,
            ..ctx_with_key("gpt-4o", "key3")
        };
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
        for h in [
            RATELIMIT_LIMIT_REQUESTS,
            RATELIMIT_LIMIT_TOKENS,
            RATELIMIT_REMAINING_REQUESTS,
            RATELIMIT_REMAINING_TOKENS,
            RATELIMIT_RESET_REQUESTS,
            RATELIMIT_RESET_TOKENS,
        ] {
            assert!(
                resp.headers().get(h).is_none(),
                "master key must not stamp {h}"
            );
        }
    }

    #[tokio::test]
    async fn stamp_ratelimit_headers_noop_for_cache_hit() {
        let state = state_with_in_memory_cache();
        let _ = state.cache.incr("rpm:key4", 3.0, None).await;
        let ctx = CallContext {
            rpm: Some(10),
            tpm: Some(1000),
            cache_hit: true,
            ..ctx_with_key("gpt-4o", "key4")
        };
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
        assert!(resp.headers().get(RATELIMIT_LIMIT_REQUESTS).is_none());
        assert!(resp.headers().get(RATELIMIT_REMAINING_REQUESTS).is_none());
        assert!(resp.headers().get(RATELIMIT_LIMIT_TOKENS).is_none());
    }

    #[tokio::test]
    async fn stamp_ratelimit_headers_only_rpm_stamps_request_headers() {
        let state = state_with_in_memory_cache();
        let _ = state.cache.incr("rpm:key5", 1.0, None).await;
        let ctx = CallContext {
            rpm: Some(5),
            ..ctx_with_key("gpt-4o", "key5")
        };
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_REMAINING_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("4")
        );
        // No TPM configured → token headers must be absent.
        assert!(resp.headers().get(RATELIMIT_LIMIT_TOKENS).is_none());
        assert!(resp.headers().get(RATELIMIT_REMAINING_TOKENS).is_none());
        assert!(resp.headers().get(RATELIMIT_RESET_TOKENS).is_none());
    }

    #[tokio::test]
    async fn stamp_ratelimit_headers_no_limits_stamps_nothing() {
        let state = state_with_in_memory_cache();
        let ctx = ctx_with_key("gpt-4o", "key6");
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
        assert!(resp.headers().get(RATELIMIT_LIMIT_REQUESTS).is_none());
        assert!(resp.headers().get(RATELIMIT_LIMIT_TOKENS).is_none());
        assert!(resp.headers().get(RATELIMIT_REMAINING_REQUESTS).is_none());
    }

    #[tokio::test]
    async fn stamp_ratelimit_headers_uses_stashed_snapshot_without_per_key_limit() {
        // No per-key ctx.rpm, but a global/other-scope limit applied and the
        // hook stashed both the binding limit and remaining at pre_call. The
        // limit-* and remaining-* headers must come from that snapshot, NOT be
        // re-derived from the live counter (which races under concurrency).
        let state = state_with_in_memory_cache();
        // A concurrent request bumped the per-key counter AFTER pre_call. The
        // old derivation (remaining + live_counter) would wrongly report
        // limit = 6 + 5 = 11; the snapshot reports the stashed limit = 10.
        let _ = state.cache.incr("rpm:key7", 5.0, None).await;
        let ctx = CallContext {
            // ctx.rpm intentionally None: only a global limit is in effect.
            ratelimit_info: RateLimitInfo {
                remaining_rpm: Some(6),
                limit_rpm: Some(10),
                reset_rpm_secs: Some(60),
                ..Default::default()
            },
            ..ctx_with_key("gpt-4o", "key7")
        };
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_LIMIT_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("10"),
            "limit must be the stashed snapshot (10), not remaining(6) + live_counter(5) = 11"
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_REMAINING_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("6"),
            "remaining must be the stashed snapshot, consistent with the limit"
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_RESET_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("60s")
        );
    }

    // LOW (C12): under concurrency the derived `limit-*` header used to race —
    // `remaining` was stashed at pre_call but the limit was derived as
    // `remaining + live_counter` read at stamp time, so a concurrent request
    // that bumped the counter inflated the derived limit. With the stashed
    // snapshot, every stamped limit is identical regardless of concurrent
    // counter movement.
    #[tokio::test]
    async fn stamp_ratelimit_headers_limit_stable_under_concurrent_counter_growth() {
        let state = state_with_in_memory_cache();
        // Simulate a global RPM of 10 with 4 already consumed at pre_call:
        // remaining = 6, limit = 10.
        let ctx = CallContext {
            ratelimit_info: RateLimitInfo {
                remaining_rpm: Some(6),
                limit_rpm: Some(10),
                reset_rpm_secs: Some(60),
                ..Default::default()
            },
            ..ctx_with_key("gpt-4o", "concurrent")
        };
        let mut limits = Vec::new();
        for counter in 4..=12 {
            // Each iteration reflects a snapshot where a concurrent request
            // bumped the per-key counter further. The stashed limit must NOT
            // change with the counter.
            let _ = state
                .cache
                .incr("rpm:concurrent", if counter == 4 { 4.0 } else { 1.0 }, None)
                .await;
            let mut resp = Response::new(axum::body::Body::from(""));
            stamp_ratelimit_headers(&mut resp, &state, &ctx).await;
            limits.push(
                resp.headers()
                    .get(RATELIMIT_LIMIT_REQUESTS)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
            );
        }
        for l in &limits {
            assert_eq!(
                l, "10",
                "stamped limit must be the stable snapshot (10) under concurrent counter growth"
            );
        }
    }

    #[test]
    fn apply_cost_header_noop_when_none() {
        let mut resp = Response::new(axum::body::Body::from(""));
        apply_cost_header(&mut resp, None);
        assert!(resp.headers().get(COST_HEADER).is_none());
    }

    #[test]
    fn set_header_inserts_when_valid() {
        let mut resp = Response::new(axum::body::Body::from(""));
        set_header(&mut resp, REQUEST_ID_HEADER, "abc-123");
        assert_eq!(
            resp.headers()
                .get(REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("abc-123")
        );
    }

    #[test]
    fn stamp_request_id_header_sets_both_headers() {
        // H8: every response must carry BOTH `x-oxllm-request-id` (for our
        // own tooling) AND the standard OpenAI `x-request-id` (for generic
        // OpenAI clients). The single helper keeps them in sync so future
        // stamp sites can't regress.
        let mut resp = Response::new(axum::body::Body::from(""));
        stamp_request_id_header(&mut resp, "req-xyz");
        assert_eq!(
            resp.headers()
                .get(REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("req-xyz")
        );
        assert_eq!(
            resp.headers()
                .get(STANDARD_REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("req-xyz"),
            "x-request-id (standard OpenAI header) must also be stamped"
        );
    }

    #[tokio::test]
    async fn pipeline_pre_then_post_runs_hooks_in_order() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let mut pipeline = EndpointPipeline {
            state: &state,
            call_ctx: dummy_ctx(),
        };

        let _ = pipeline.pre_call().await;
        let usage = UsageSummary {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            ..Default::default()
        };
        let _ = pipeline
            .post_success("gpt-4o", Some(&usage), None, "openai")
            .await;

        let guard = calls.lock().unwrap();
        assert_eq!(
            &guard[..],
            &["pre_call", "post_success", "post_success_with_cost"]
        );
    }

    #[tokio::test]
    async fn latency_smoke() {
        let state = state_with_in_memory_cache();
        let auth = crate::auth::AuthUser::open();
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = EndpointPipeline::for_request(&state, &auth, "m", Value::Null).await;
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 1000,
            "1000 pipeline constructions should be fast, took {elapsed:?}"
        );
    }

    #[test]
    fn extract_session_id_reads_header() {
        let mut headers = HeaderMap::new();
        headers.insert(SESSION_ID_HEADER, HeaderValue::from_static("thread-1"));
        assert_eq!(extract_session_id(&headers).as_deref(), Some("thread-1"));
    }

    #[test]
    fn extract_session_id_trims_and_rejects_empty() {
        let mut headers = HeaderMap::new();
        headers.insert(SESSION_ID_HEADER, HeaderValue::from_static(" "));
        assert!(extract_session_id(&headers).is_none());
        assert!(extract_session_id(&HeaderMap::new()).is_none());
    }

    #[test]
    fn extract_session_id_from_anthropic_user_id_suffix() {
        let body = json!({
            "metadata": {
                "user_id": "user_abc123_account__session_e96634a3-fa28-4083-b354-55542e2dca01"
            }
        });
        assert_eq!(
            extract_session_id_with_body(&HeaderMap::new(), Some(&body)).as_deref(),
            Some("e96634a3-fa28-4083-b354-55542e2dca01")
        );
    }

    #[test]
    fn extract_session_id_from_anthropic_user_id_object() {
        let body = json!({
            "metadata": {
                "user_id": {
                    "device_id": "device",
                    "session_id": "sess_4f8c1d2a-1234"
                }
            }
        });
        assert_eq!(
            extract_session_id_with_body(&HeaderMap::new(), Some(&body)).as_deref(),
            Some("sess_4f8c1d2a-1234")
        );
    }

    #[test]
    fn extract_session_id_header_wins_over_anthropic_metadata() {
        let mut headers = HeaderMap::new();
        headers.insert(SESSION_ID_HEADER, HeaderValue::from_static("header-sess"));
        let body = json!({"metadata": {"user_id": "user_x_session_from-body"}});
        assert_eq!(
            extract_session_id_with_body(&headers, Some(&body)).as_deref(),
            Some("header-sess")
        );
    }

    #[test]
    fn extract_session_id_accepts_litellm_alias() {
        let mut headers = HeaderMap::new();
        headers.insert(
            LITELLM_SESSION_ID_HEADER,
            HeaderValue::from_static("alias-sess"),
        );
        assert_eq!(extract_session_id(&headers).as_deref(), Some("alias-sess"));
    }

    #[test]
    fn extract_session_id_oxllm_header_wins_over_alias() {
        let mut headers = HeaderMap::new();
        headers.insert(SESSION_ID_HEADER, HeaderValue::from_static("ox"));
        headers.insert(LITELLM_SESSION_ID_HEADER, HeaderValue::from_static("alias"));
        assert_eq!(extract_session_id(&headers).as_deref(), Some("ox"));
    }

    #[test]
    fn extract_session_id_empty_oxllm_header_falls_through_to_alias() {
        let mut headers = HeaderMap::new();
        headers.insert(SESSION_ID_HEADER, HeaderValue::from_static("  "));
        headers.insert(
            LITELLM_SESSION_ID_HEADER,
            HeaderValue::from_static("alias-sess"),
        );
        let body = json!({"metadata": {"user_id": "user_x_session_from-body"}});
        assert_eq!(
            extract_session_id_with_body(&headers, Some(&body)).as_deref(),
            Some("alias-sess")
        );
    }

    #[test]
    fn extract_session_id_empty_alias_falls_through_to_body() {
        let mut headers = HeaderMap::new();
        headers.insert(LITELLM_SESSION_ID_HEADER, HeaderValue::from_static(""));
        let body = json!({"metadata": {"user_id": "user_x_session_from-body"}});
        assert_eq!(
            extract_session_id_with_body(&headers, Some(&body)).as_deref(),
            Some("from-body")
        );
    }

    #[test]
    fn extract_tag_reads_header() {
        let mut headers = HeaderMap::new();
        headers.insert(TAG_HEADER, HeaderValue::from_static("canary"));
        assert_eq!(extract_tag(&headers).as_deref(), Some("canary"));
    }

    #[test]
    fn extract_tags_merges_body_metadata_and_header_in_order() {
        let body = serde_json::json!({
            "model": "m",
            "metadata": {"tags": ["app-a", "env-prod"]},
        });
        let mut headers = HeaderMap::new();
        headers.insert(TAG_HEADER, HeaderValue::from_static("app-a, canary"));
        let tags = extract_tags(&headers, Some(&body));
        assert_eq!(tags, ["app-a", "env-prod", "canary"]);
        // Body tags first; duplicates removed; header list split on commas.
    }

    #[test]
    fn extract_tags_header_only_and_none() {
        let mut headers = HeaderMap::new();
        headers.insert(TAG_HEADER, HeaderValue::from_static("solo"));
        assert_eq!(extract_tags(&headers, None), ["solo"]);
        assert!(extract_tags(&HeaderMap::new(), None).is_empty());
        // Non-array / non-string metadata.tags is ignored.
        let bad = serde_json::json!({"metadata": {"tags": "not-a-list"}});
        assert!(extract_tags(&HeaderMap::new(), Some(&bad)).is_empty());
    }

    #[test]
    fn extract_tags_bounds_count_and_length() {
        let many: Vec<String> = (0..40).map(|i| format!("t{i}")).collect();
        let body = serde_json::json!({"metadata": {"tags": many}});
        let tags = extract_tags(&HeaderMap::new(), Some(&body));
        assert_eq!(tags.len(), MAX_REQUEST_TAGS, "count is capped");
        let long = "x".repeat(MAX_TAG_BYTES + 50);
        let body2 = serde_json::json!({"metadata": {"tags": [long]}});
        let tags2 = extract_tags(&HeaderMap::new(), Some(&body2));
        assert_eq!(tags2.len(), 1);
        assert_eq!(tags2[0].len(), MAX_TAG_BYTES, "length is capped");
    }

    #[test]
    fn extract_tag_returns_none_when_absent() {
        assert!(extract_tag(&HeaderMap::new()).is_none());
        let mut headers = HeaderMap::new();
        headers.insert(TAG_HEADER, HeaderValue::from_static("   "));
        assert!(extract_tag(&headers).is_none());
    }

    #[test]
    fn repeated_tag_headers_share_primary_and_bounded_union() {
        let mut headers = HeaderMap::new();
        headers.append(TAG_HEADER, HeaderValue::from_static(" first, second "));
        headers.append(TAG_HEADER, HeaderValue::from_static("second, third"));
        assert_eq!(extract_tag(&headers).as_deref(), Some("first"));
        assert_eq!(
            extract_tags(
                &headers,
                Some(&json!({"metadata":{"tags":["body", "first"]}}))
            ),
            ["body", "first", "second", "third"]
        );
        let tags = (0..MAX_REQUEST_TAGS + 1)
            .map(|i| format!("{i}-{}", "é".repeat(MAX_TAG_BYTES)))
            .collect::<Vec<_>>();
        let normalized = merge_request_tags(None, tags.iter().map(String::as_str));
        assert_eq!(normalized.len(), MAX_REQUEST_TAGS);
        assert!(normalized.iter().all(|tag| tag.len() <= MAX_TAG_BYTES));
    }

    // ---- glob_match -------------------------------------------------------

    #[test]
    fn glob_match_exact() {
        assert!(glob_match("gpt-4o", "gpt-4o"));
        assert!(!glob_match("gpt-4o", "gpt-4o-mini"));
    }

    #[test]
    fn glob_match_prefix_wildcard() {
        assert!(glob_match("gpt-4*", "gpt-4o"));
        assert!(glob_match("gpt-4*", "gpt-4-turbo"));
        assert!(glob_match("gpt-4*", "gpt-4"));
        assert!(!glob_match("gpt-4*", "claude-3"));
    }

    #[test]
    fn glob_match_suffix_wildcard() {
        assert!(glob_match("*-mini", "gpt-4o-mini"));
        assert!(glob_match("*-mini", "claude-mini"));
        assert!(!glob_match("*-mini", "gpt-4o"));
    }

    #[test]
    fn glob_match_middle_wildcard() {
        assert!(glob_match("gpt-*-mini", "gpt-4o-mini"));
        assert!(glob_match("gpt-*-mini", "gpt-4-turbo-mini"));
        assert!(!glob_match("gpt-*-mini", "claude-3-mini"));
        assert!(!glob_match("gpt-*-mini", "gpt-4o"));
    }

    #[test]
    fn glob_match_empty_pattern_part() {
        // Multiple consecutive stars collapse to a single "match anything"
        // semantically. A bare `*` should match anything.
        assert!(glob_match("*", "anything"));
        assert!(glob_match("gpt**", "gpt-4o"));
    }

    #[test]
    fn glob_match_no_match_when_pattern_longer() {
        assert!(!glob_match("gpt-4o-mini", "gpt-4o"));
    }

    // ---- end-user / customer helpers --------------------------------------

    #[test]
    fn extract_end_user_id_reads_body_user() {
        assert_eq!(
            extract_end_user_id(&serde_json::json!({"user": "eu-1"})).as_deref(),
            Some("eu-1")
        );
        assert_eq!(
            extract_end_user_id(&serde_json::json!({"user": "  eu-2  "})).as_deref(),
            Some("eu-2")
        );
        assert_eq!(extract_end_user_id(&serde_json::json!({"user": ""})), None);
        assert_eq!(
            extract_end_user_id(&serde_json::json!({"user": "  "})),
            None
        );
        assert_eq!(extract_end_user_id(&serde_json::json!({})), None);
        assert_eq!(
            extract_end_user_id(&serde_json::json!({"user": 42})),
            None,
            "non-string user is ignored"
        );
    }

    #[tokio::test]
    async fn for_request_stamps_end_user_id_without_db() {
        let state = state_with_in_memory_cache();
        let auth = crate::auth::AuthUser::open();
        let body = serde_json::json!({
            "model": "gpt-4o",
            "user": "eu-no-db",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let pipeline = EndpointPipeline::for_request(&state, &auth, "gpt-4o", body)
            .await
            .expect("ok");
        assert_eq!(pipeline.call_ctx.end_user_id.as_deref(), Some("eu-no-db"));
        assert!(pipeline.call_ctx.customer_max_budget.is_none());
    }

    // ---- check_model_acl --------------------------------------------------

    fn auth_open() -> crate::auth::AuthUser {
        crate::auth::AuthUser::open()
    }

    fn auth_master() -> crate::auth::AuthUser {
        crate::auth::AuthUser::master([0u8; 32])
    }

    #[test]
    fn check_model_acl_master_bypasses_all() {
        let allowed = vec!["claude-only".to_string()];
        let res = check_model_acl(
            &auth_master(),
            "gpt-4o",
            Some(&allowed),
            Some(&allowed),
            Some(&allowed),
        );
        assert!(res.is_ok(), "master must bypass ACLs");
    }

    #[test]
    fn check_model_acl_none_models_allows_all() {
        let res = check_model_acl(&auth_open(), "gpt-4o", None, None, None);
        assert!(res.is_ok(), "no models configured → no restriction");
    }

    #[test]
    fn check_model_acl_empty_models_allows_all() {
        // Empty list = "no restriction" (same as None). Important for
        // backward compatibility with keys that have models = [].
        let empty: Vec<String> = vec![];
        let res = check_model_acl(
            &auth_open(),
            "gpt-4o",
            Some(&empty),
            Some(&empty),
            Some(&empty),
        );
        assert!(res.is_ok(), "empty models list must not restrict");
    }

    #[test]
    fn check_model_acl_key_allows_matching_model() {
        let allowed = vec!["gpt-4o".to_string()];
        let res = check_model_acl(&auth_open(), "gpt-4o", Some(&allowed), None, None);
        assert!(res.is_ok());
    }

    #[test]
    fn check_model_acl_key_allows_via_wildcard() {
        let allowed = vec!["gpt-4*".to_string()];
        assert!(check_model_acl(&auth_open(), "gpt-4o", Some(&allowed), None, None).is_ok());
        assert!(check_model_acl(&auth_open(), "gpt-4-turbo", Some(&allowed), None, None).is_ok());
        assert!(
            check_model_acl(&auth_open(), "claude-3", Some(&allowed), None, None).is_err(),
            "non-matching model must be rejected"
        );
    }

    #[test]
    fn check_model_acl_key_blocks_non_matching_model() {
        let allowed = vec!["claude-*".to_string()];
        let res = check_model_acl(&auth_open(), "gpt-4o", Some(&allowed), None, None);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
        assert!(err.message.contains("gpt-4o"));
    }

    #[test]
    fn check_model_acl_team_blocks_when_key_allows() {
        // Key allows gpt-4o, but team only allows claude. The intersection
        // (both must pass) means claude is rejected by the key and gpt-4o is
        // rejected by the team.
        let key_models = vec!["gpt-4o".to_string()];
        let team_models = vec!["claude-*".to_string()];
        let res = check_model_acl(
            &auth_open(),
            "gpt-4o",
            Some(&key_models),
            Some(&team_models),
            None,
        );
        assert!(res.is_err(), "team must reject even when key allows");
        let err = res.unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
        assert!(err.message.contains("team"));
    }

    #[test]
    fn check_model_acl_intersection_key_and_team_both_allow() {
        // Both allow the model — request proceeds.
        let key_models = vec!["gpt-4*".to_string()];
        let team_models = vec!["gpt-4o".to_string(), "claude-*".to_string()];
        let res = check_model_acl(
            &auth_open(),
            "gpt-4o",
            Some(&key_models),
            Some(&team_models),
            None,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn check_model_acl_user_level_blocks_when_key_allows() {
        let key_models = vec!["gpt-4o".to_string()];
        let user_models = vec!["claude-*".to_string()];
        let res = check_model_acl(
            &auth_open(),
            "gpt-4o",
            Some(&key_models),
            None,
            Some(&user_models),
        );
        assert!(res.is_err(), "user ACL must reject even when key allows");
        assert!(res.unwrap_err().message.contains("gpt-4o"));
    }

    #[test]
    fn check_model_acl_all_three_levels_must_allow() {
        let key_models = vec!["gpt-4*".to_string()];
        let team_models = vec!["gpt-4o".to_string()];
        let user_models = vec!["gpt-4o".to_string()];
        let res = check_model_acl(
            &auth_open(),
            "gpt-4o",
            Some(&key_models),
            Some(&team_models),
            Some(&user_models),
        );
        assert!(res.is_ok(), "all three levels allow gpt-4o");
    }

    #[test]
    fn check_model_acl_wildcard_anywhere_matches() {
        let models = vec!["gpt*4o".to_string()];
        assert!(
            check_model_acl(&auth_open(), "gpt-4o", Some(&models), None, None).is_ok(),
            "wildcard in middle of pattern must match"
        );
        assert!(
            check_model_acl(&auth_open(), "claude-3", Some(&models), None, None).is_err(),
            "non-matching model must be rejected"
        );
    }

    #[test]
    fn check_model_acl_multi_wildcard_matches() {
        let models = vec!["*-*".to_string()];
        assert!(
            check_model_acl(&auth_open(), "gpt-4o", Some(&models), None, None).is_ok(),
            "*-* must match gpt-4o"
        );
        assert!(
            check_model_acl(&auth_open(), "gpt4o", Some(&models), None, None).is_err(),
            "*-* must not match gpt4o (no hyphen)"
        );
    }

    #[test]
    fn check_model_acl_only_key_none_team_user_allows() {
        let key_models = vec!["gpt-4o".to_string()];
        let res = check_model_acl(&auth_open(), "gpt-4o", Some(&key_models), None, None);
        assert!(res.is_ok());
    }

    // ---- U3: extract_upstream_usage ---------------------------------------

    #[test]
    fn extract_upstream_usage_reads_oxllm_header() {
        let mut headers = HeaderMap::new();
        headers.insert(UPSTREAM_TOTAL_TOKENS_HEADER, HeaderValue::from_static("42"));
        let usage = extract_upstream_usage(&headers).expect("usage from header");
        assert_eq!(usage.total_tokens, 42);
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 0);
    }

    #[test]
    fn extract_upstream_usage_reads_litellm_header_fallback() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM,
            HeaderValue::from_static("100"),
        );
        let usage = extract_upstream_usage(&headers).expect("usage from litellm header");
        assert_eq!(usage.total_tokens, 100);
    }

    #[test]
    fn extract_upstream_usage_oxllm_header_takes_priority_over_litellm() {
        let mut headers = HeaderMap::new();
        headers.insert(UPSTREAM_TOTAL_TOKENS_HEADER, HeaderValue::from_static("7"));
        headers.insert(
            UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM,
            HeaderValue::from_static("99"),
        );
        let usage = extract_upstream_usage(&headers).expect("usage");
        assert_eq!(
            usage.total_tokens, 7,
            "oxllm header must take priority over litellm alias"
        );
    }

    #[test]
    fn extract_upstream_usage_returns_none_when_absent() {
        let headers = HeaderMap::new();
        assert!(extract_upstream_usage(&headers).is_none());
    }

    #[test]
    fn extract_upstream_usage_returns_none_for_zero() {
        let mut headers = HeaderMap::new();
        headers.insert(UPSTREAM_TOTAL_TOKENS_HEADER, HeaderValue::from_static("0"));
        assert!(extract_upstream_usage(&headers).is_none());
    }

    #[test]
    fn extract_upstream_usage_returns_none_for_invalid() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_TOTAL_TOKENS_HEADER,
            HeaderValue::from_static("not-a-number"),
        );
        assert!(extract_upstream_usage(&headers).is_none());
    }

    #[test]
    fn extract_upstream_usage_trims_whitespace() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_TOTAL_TOKENS_HEADER,
            HeaderValue::from_static(" 256 "),
        );
        let usage = extract_upstream_usage(&headers).expect("trimmed");
        assert_eq!(usage.total_tokens, 256);
    }

    // LOW-3: header-only usage attributes all tokens to prompt_tokens. Because
    // completion tokens are priced higher than prompt tokens, the resulting
    // cost must be non-zero AND must not exceed the cost of the same token
    // count attributed entirely to completion tokens (the underestimation
    // bound). This documents the direction of the cost bias.
    #[test]
    fn extract_upstream_usage_cost_underestimates_relative_to_completion() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_TOTAL_TOKENS_HEADER,
            HeaderValue::from_static("1000"),
        );
        let usage = extract_upstream_usage(&headers).expect("usage from header");

        let prompt_cost = oxllm_core::calculate_cost_with_override("gpt-4o", &usage, None)
            .expect("gpt-4o has pricing");
        assert!(prompt_cost > 0.0, "cost from header usage must be non-zero");

        let all_completion = UsageSummary {
            prompt_tokens: 0,
            completion_tokens: 1000,
            total_tokens: 1000,
            ..Default::default()
        };
        let completion_cost =
            oxllm_core::calculate_cost_with_override("gpt-4o", &all_completion, None)
                .expect("gpt-4o has pricing");
        assert!(
            prompt_cost <= completion_cost,
            "all-prompt attribution (${prompt_cost}) must not exceed all-completion (${completion_cost})"
        );
    }

    // ---- sanitize_timeout_secs (MEDIUM C12: no panic on non-finite) -------

    #[test]
    fn sanitize_timeout_secs_finite_positive_passes_through() {
        assert_eq!(sanitize_timeout_secs(0.5), Duration::from_secs_f64(0.5));
        assert_eq!(sanitize_timeout_secs(120.0), Duration::from_secs(120));
    }

    #[test]
    fn sanitize_timeout_secs_clamps_nan_to_fallback() {
        // `Duration::from_secs_f64(f64::NAN)` panics; the clamp must instead
        // yield the safe fallback so a misconfigured `timeout: .nan` never
        // crashes the worker.
        assert_eq!(sanitize_timeout_secs(f64::NAN), Duration::from_secs(600));
    }

    #[test]
    fn sanitize_timeout_secs_clamps_infinity_to_fallback() {
        assert_eq!(
            sanitize_timeout_secs(f64::INFINITY),
            Duration::from_secs(600)
        );
        assert_eq!(
            sanitize_timeout_secs(f64::NEG_INFINITY),
            Duration::from_secs(600)
        );
    }

    #[test]
    fn sanitize_timeout_secs_clamps_non_positive_to_fallback() {
        assert_eq!(sanitize_timeout_secs(0.0), Duration::from_secs(600));
        assert_eq!(sanitize_timeout_secs(-5.0), Duration::from_secs(600));
    }

    // ---- U3: extract_upstream_response_cost -------------------------------

    #[test]
    fn extract_upstream_response_cost_reads_oxllm_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static("0.42"),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Valid(0.42)
        );
    }

    #[test]
    fn extract_upstream_response_cost_reads_litellm_header_fallback() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER_LITELLM,
            HeaderValue::from_static("1.5"),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Valid(1.5)
        );
    }

    #[test]
    fn extract_upstream_response_cost_oxllm_header_takes_priority() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static("0.7"),
        );
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER_LITELLM,
            HeaderValue::from_static("0.1"),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Valid(0.7),
            "oxllm header must win over the litellm alias"
        );
    }

    #[test]
    fn extract_upstream_response_cost_absent_when_no_header() {
        assert_eq!(
            extract_upstream_response_cost(&HeaderMap::new()),
            UpstreamResponseCost::Absent
        );
    }

    #[test]
    fn extract_upstream_response_cost_allows_zero() {
        // Cost-contract precedence: an upstream that reports cost = 0 owns the
        // cost for the request; 0 is a valid override, not "no value".
        let mut headers = HeaderMap::new();
        headers.insert(UPSTREAM_RESPONSE_COST_HEADER, HeaderValue::from_static("0"));
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Valid(0.0)
        );
    }

    #[test]
    fn extract_upstream_response_cost_negative_is_invalid_not_absent() {
        // MEDIUM (C12): a present-but-invalid cost header must be `Invalid`,
        // not `Absent`, so the caller bills 0 (contract parity) rather than
        // falling back to a local estimate the upstream just contradicted.
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static("-1"),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Invalid
        );
    }

    #[test]
    fn extract_upstream_response_cost_nan_and_infinity_are_invalid() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static("NaN"),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Invalid
        );
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static("Infinity"),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Invalid
        );
    }

    #[test]
    fn extract_upstream_response_cost_non_numeric_is_invalid() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static("not-a-number"),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Invalid
        );
    }

    #[test]
    fn extract_upstream_response_cost_trims_whitespace() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static(" 0.42 "),
        );
        assert_eq!(
            extract_upstream_response_cost(&headers),
            UpstreamResponseCost::Valid(0.42)
        );
    }

    #[test]
    fn upstream_response_cost_into_override_mapping() {
        // The three states map to the cost_override decision: absent → local
        // compute (None); valid → the reported value; invalid → 0 (contract
        // parity: a target that spoke the contract owns the cost, so an
        // unusable value bills 0 rather than a local estimate).
        assert_eq!(UpstreamResponseCost::Absent.into_override(), None);
        assert_eq!(UpstreamResponseCost::Valid(0.0).into_override(), Some(0.0));
        assert_eq!(
            UpstreamResponseCost::Valid(1.25).into_override(),
            Some(1.25)
        );
        assert_eq!(
            UpstreamResponseCost::Invalid.into_override(),
            Some(0.0),
            "an invalid reported cost must bill 0, not fall back to local compute"
        );
    }

    // MEDIUM (C12): cost-contract parity — when an upstream sends a present-but-
    // invalid cost header (NaN / unparseable / negative), `post_success` must
    // record cost 0.0, NOT the local price-table estimate. The reference
    // `success_handler._set_cost_per_request` short-circuits whenever
    // `has_upstream_reported_usage` is true, so an unusable reported cost
    // records 0 rather than billing an estimate the upstream contradicted.
    #[tokio::test]
    async fn post_success_bills_zero_for_invalid_upstream_cost_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            UPSTREAM_RESPONSE_COST_HEADER,
            HeaderValue::from_static("NaN"),
        );
        let override_cost = extract_upstream_response_cost(&headers).into_override();

        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let mut ctx = dummy_ctx();
        ctx.cost_override = override_cost;
        // Usage that would normally compute a non-zero local cost.
        let usage = UsageSummary {
            prompt_tokens: 1000,
            completion_tokens: 0,
            total_tokens: 1000,
            ..Default::default()
        };
        let cost = post_success(&state, &ctx, "gpt-4o", Some(&usage), None, None, "openai").await;
        assert_eq!(
            cost,
            Some(0.0),
            "invalid reported cost must bill 0, not the local price-table estimate"
        );
        let guard = calls.lock().unwrap();
        assert!(
            guard.contains(&"post_success_with_cost"),
            "spend hook must see the 0.0 cost override"
        );
    }

    // U3: when an upstream reported cost is stashed on the call context,
    // post_success must use it verbatim instead of computing from usage.
    #[tokio::test]
    async fn post_success_honours_cost_override_over_local_compute() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let mut ctx = dummy_ctx();
        ctx.cost_override = Some(0.99);
        // Usage that would normally compute a different cost.
        let usage = UsageSummary {
            prompt_tokens: 1000,
            completion_tokens: 0,
            total_tokens: 1000,
            ..Default::default()
        };
        let cost = post_success(&state, &ctx, "gpt-4o", Some(&usage), None, None, "openai").await;
        assert_eq!(cost, Some(0.99), "override must win over local compute");
        let guard = calls.lock().unwrap();
        assert!(guard.contains(&"post_success_with_cost"));
    }

    // ---- extract_body_usage_cost (OpenRouter body-level cost fallback) ----

    #[test]
    fn extract_body_usage_cost_reads_usage_cost() {
        let body = serde_json::json!({"usage": {"prompt_tokens": 10, "cost": 0.007}});
        let cost = extract_body_usage_cost(&body).expect("cost present");
        assert!((cost - 0.007).abs() < 1e-9);
    }

    #[test]
    fn extract_body_usage_cost_allows_zero() {
        let body = serde_json::json!({"usage": {"cost": 0}});
        let cost = extract_body_usage_cost(&body).expect("zero is valid");
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn extract_body_usage_cost_returns_none_when_absent() {
        assert!(
            extract_body_usage_cost(&serde_json::json!({"usage": {"prompt_tokens": 5}})).is_none()
        );
        assert!(extract_body_usage_cost(&serde_json::json!({})).is_none());
    }

    #[test]
    fn extract_body_usage_cost_rejects_negative() {
        let body = serde_json::json!({"usage": {"cost": -0.5}});
        assert!(extract_body_usage_cost(&body).is_none());
    }

    #[test]
    fn extract_body_usage_cost_rejects_non_numeric() {
        let body = serde_json::json!({"usage": {"cost": "free"}});
        assert!(extract_body_usage_cost(&body).is_none());
    }

    // MEDIUM (C52): for a model absent from the local price table (common with
    // OpenRouter's aggregator surface), post_success must fall back to the
    // body-level `usage.cost` OpenRouter reports, so spend is recorded instead
    // of silently `null`. The price-table lookup runs first; only when it
    // returns `None` does the body cost apply.
    #[tokio::test]
    async fn post_success_uses_body_usage_cost_for_unpriced_model() {
        let (hook, calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let ctx = dummy_ctx();
        let usage = UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let body = serde_json::json!({
            "model": "unpriced-vendor/mystery-model",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.00123,
            },
        });
        let cost = post_success(
            &state,
            &ctx,
            "unpriced-vendor/mystery-model",
            Some(&usage),
            Some(&body),
            None,
            "unpriced-vendor",
        )
        .await;
        assert_eq!(cost, Some(0.00123), "body-level cost must be the fallback");
        let guard = calls.lock().unwrap();
        assert!(
            guard.contains(&"post_success_with_cost"),
            "spend hook must see a non-null cost"
        );
    }

    // MEDIUM: the streaming path has no cacheable response body (SSE chunks
    // were already relayed), so it threads the aggregator-reported
    // `usage.cost` through the dedicated `body_cost` field with `body = None`.
    // The fallback cost must still bill — but the `ResponseSummary` must
    // carry NO body, because `CacheHook::post_success` stores `body` verbatim
    // and a synthetic `{"usage":{"cost":…}}` fragment would be served as the
    // next identical request's cached "response".
    #[tokio::test]
    async fn post_success_body_cost_bills_without_caching_a_synthetic_body() {
        let summaries: Arc<Mutex<Vec<ResponseSummary>>> = Arc::new(Mutex::new(Vec::new()));
        let hook = SummaryCaptureHook {
            summaries: summaries.clone(),
        };
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let ctx = dummy_ctx();
        let usage = UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let cost = post_success(
            &state,
            &ctx,
            "unpriced-vendor/mystery-model",
            Some(&usage),
            None,
            Some(0.00123),
            "unpriced-vendor",
        )
        .await;
        assert_eq!(
            cost,
            Some(0.00123),
            "the dedicated body_cost field must be the fallback for unpriced models"
        );
        let guard = summaries.lock().unwrap();
        let summary = guard.last().expect("post_success must run the hook chain");
        assert_eq!(summary.cost, Some(0.00123));
        assert!(
            summary.body.is_none(),
            "the stream path must not hand hooks a synthetic cost body to cache"
        );
    }

    // The dedicated `body_cost` field is validated like a body-extracted
    // cost: negative values are rejected rather than billing a credit.
    #[tokio::test]
    async fn post_success_body_cost_rejects_negative() {
        let state = state_with_hooks(vec![]);
        let ctx = dummy_ctx();
        let usage = UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let cost = post_success(
            &state,
            &ctx,
            "unpriced-vendor/mystery-model",
            Some(&usage),
            None,
            Some(-0.5),
            "unpriced-vendor",
        )
        .await;
        assert_eq!(cost, None, "negative body_cost must not bill");
    }

    #[tokio::test]
    async fn post_success_prefers_price_table_over_body_usage_cost() {
        // When the price table DOES have an entry, its value must win over the
        // body-level cost — the body cost is a fallback only.
        let (hook, _calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let ctx = dummy_ctx();
        let usage = UsageSummary {
            prompt_tokens: 1000,
            completion_tokens: 0,
            total_tokens: 1000,
            ..Default::default()
        };
        let body = serde_json::json!({
            "usage": {"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000, "cost": 99.0},
        });
        let cost = post_success(
            &state,
            &ctx,
            "gpt-4o",
            Some(&usage),
            Some(&body),
            None,
            "openai",
        )
        .await;
        let expected =
            oxllm_core::calculate_cost_with_override_for_provider("gpt-4o", &usage, None, "openai");
        assert_eq!(
            cost, expected,
            "price table must take precedence over body-level cost"
        );
        assert!(
            cost.unwrap_or(0.0) < 99.0,
            "body-level cost must not leak when the table has a price"
        );
    }

    // MEDIUM (C10): a per-second deployment pricing override
    // (`input_cost_per_second` in oxllm_params) must bill
    // `rate * response_time_ms / 1000` instead of token math — the reference
    // per-second custom-pricing semantics for duration-priced models.
    #[tokio::test]
    async fn post_success_bills_per_second_deployment_override_via_duration() {
        let (hook, _calls) = RecordingHook::new();
        let state = state_with_hooks(vec![Arc::new(hook)]);
        let mut ctx = dummy_ctx();
        ctx.deployment_pricing = Some(oxllm_core::cost::DeploymentPricing {
            input_cost_per_second: Some(0.001),
            ..Default::default()
        });
        ctx.start_time = Some(Instant::now() - std::time::Duration::from_secs(2));
        ctx.upstream_start_time = Some(Instant::now() - std::time::Duration::from_secs(2));
        // Token usage that would otherwise compute a table/override cost.
        let usage = UsageSummary {
            prompt_tokens: 1_000,
            completion_tokens: 500,
            total_tokens: 1_500,
            ..Default::default()
        };
        let cost = post_success(
            &state,
            &ctx,
            "sagemaker/duration-model",
            Some(&usage),
            None,
            None,
            "",
        )
        .await
        .expect("per-second override must accrue spend");
        assert!(
            cost >= 0.002,
            "at least 2s elapsed * 0.001 $/s (token math would be 0): got {cost}"
        );
        assert!(cost < 0.02, "sanity upper bound (~10x elapsed): got {cost}");
    }
}
