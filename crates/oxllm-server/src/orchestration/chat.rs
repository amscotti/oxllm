//! Retry + fallback orchestration for chat completions.
//!
//! [`execute_request_chain`] is the top-level entry: it runs the primary model
//! through [`try_model_with_retries`], classifies the resulting
//! [`ModelOutcome`], and — for fallback-eligible outcomes (context window,
//! content policy, generic retryable) — walks the configured fallback chains.
//!
//! Per-deployment retries (with exponential backoff and per-status retry
//! counts) live in [`try_model_with_retries`]; cooldowns and deployment
//! rotation are driven by the router.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::http::HeaderValue;
use axum::response::{IntoResponse, Response};
use oxllm_core::CallContext;
use oxllm_core::RequestContext;
use oxllm_core::deployment::Deployment;
use oxllm_core::router::{AllowedFailDecision, deployment_key};
use serde_json::Value;

use super::FALLBACK_HEADER;
use super::outcomes::{ModelOutcome, UpstreamError, is_deployment_fault_status, retry_after_secs};
use super::stream::handle_stream_success;
use super::upstream::resolve_provider;
use super::usage::extract_usage;
use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::pipeline::{
    apply_cost_header, check_deployment_limits, check_model_acl, post_success,
    record_deployment_tpm,
};
use crate::pre_call_checks::prompt_caching::PromptCachingCheck;
use crate::state::AppState;

use tokio_stream::StreamExt;

/// Deployments newly marked unhealthy by health-check routing (the
/// transition only — repeat failures while already unhealthy do not count).
pub(crate) const HEALTH_CHECK_UNHEALTHY_TOTAL: &str = "oxllm_health_check_unhealthy_total";

/// Complexity-router classifications by tier, counted once per request.
const COMPLEXITY_TIER_TOTAL: &str = "oxllm_complexity_tier_total";

/// Execute the full request chain: primary model (with retries) → context-window
/// fallbacks → regular fallbacks.
///
/// Before any of that, `default_fallbacks` is consulted when the requested
/// model group is UNKNOWN (no deployments resolve): the request is rerouted
/// to the first ACL-allowed, resolvable chain entry and the `model` field of
/// the body/context is rewritten. A KNOWN model whose deployments are all
/// cooling down never reroutes here — that is what `fallbacks` is for.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn execute_request_chain(
    state: &AppState,
    call_ctx: &mut CallContext,
    auth: &AuthUser,
    primary_model: &str,
    request_body: &Value,
    ctx: &RequestContext,
    is_stream: bool,
    traceparent: Option<&str>,
) -> Result<Response, ApiError> {
    let mut effective_ctx = ctx.clone();
    crate::pipeline::routing::apply_default_tag(state, &mut effective_ctx);
    let ctx = &effective_ctx;
    // `default_fallbacks` rerouting: resolve the effective model BEFORE the
    // retry loop so the whole chain (including the rerouted model's own
    // `fallbacks`) operates on it.
    let rerouted = resolve_default_fallback(state, auth, primary_model).map(str::to_string);
    let df_ctx;
    let mut df_body;
    let (primary_model, ctx, request_body) = match &rerouted {
        Some(default_model) => {
            tracing::info!(
                requested_model = %primary_model,
                default_model = %default_model,
                "unknown model group; rerouting via default_fallbacks"
            );
            // Router observability: the reroute outcome wins over per-pick
            // stamps (a later fallback-chain success re-stamps `fallback`).
            call_ctx.router_decision = Some("default_fallback".to_string());
            df_ctx = RequestContext {
                model: default_model.clone(),
                ..ctx.clone()
            };
            df_body = request_body.clone();
            if let Some(m) = df_body.as_object_mut() {
                m.insert("model".to_string(), Value::String(default_model.clone()));
            }
            (default_model.as_str(), &df_ctx, &df_body)
        }
        None => (primary_model, ctx, request_body),
    };

    // Complexity-tier observability: the classification was computed once
    // per request by `ChatOrchestrator::run` and stamped onto the request
    // context (fallback legs inherit it through the cloned contexts), so
    // the metric is a pure read — no re-classification on this path.
    if let Some((tier, _)) = ctx.pre_call.complexity_classification {
        ::metrics::counter!(COMPLEXITY_TIER_TOTAL, "tier" => tier.as_str()).increment(1);
    }

    let primary_outcome = try_model_with_retries(
        state,
        call_ctx,
        primary_model,
        request_body,
        ctx,
        is_stream,
        traceparent,
    )
    .await;

    let mut attempted_fallbacks: HashSet<String> = HashSet::new();
    attempted_fallbacks.insert(primary_model.to_string());
    let max_fallbacks = state.router.config().max_fallbacks;
    let result = match primary_outcome {
        ModelOutcome::Success(resp) => Ok(resp),
        ModelOutcome::ContextWindow(_) => {
            let mut cwf = state.router.context_window_fallbacks_for(primary_model);
            cwf.extend(state.router.fallbacks_for(primary_model));
            if let Some(resp) = try_fallback_chain(
                state,
                call_ctx,
                auth,
                request_body,
                ctx,
                is_stream,
                traceparent,
                &cwf,
                &mut attempted_fallbacks,
                max_fallbacks,
            )
            .await
            {
                Ok(resp)
            } else {
                Err(primary_outcome.into_api_error())
            }
        }
        ModelOutcome::ContentPolicy(_) => {
            let mut cpf = state.router.content_policy_fallbacks_for(primary_model);
            cpf.extend(state.router.fallbacks_for(primary_model));
            if let Some(resp) = try_fallback_chain(
                state,
                call_ctx,
                auth,
                request_body,
                ctx,
                is_stream,
                traceparent,
                &cpf,
                &mut attempted_fallbacks,
                max_fallbacks,
            )
            .await
            {
                Ok(resp)
            } else {
                Err(primary_outcome.into_api_error())
            }
        }
        ModelOutcome::Retryable(_) => {
            let fbs = state.router.fallbacks_for(primary_model);
            if let Some(resp) = try_fallback_chain(
                state,
                call_ctx,
                auth,
                request_body,
                ctx,
                is_stream,
                traceparent,
                &fbs,
                &mut attempted_fallbacks,
                max_fallbacks,
            )
            .await
            {
                Ok(resp)
            } else {
                Err(primary_outcome.into_api_error())
            }
        }
        // LiteLLM parity (`async_function_with_fallbacks`,
        // `litellm/router.py:6435`): a primary-model 4xx that indicates a
        // DEPLOYMENT fault (401 stale credential, 403 not entitled, 404
        // decommissioned model) walks the configured generic `fallbacks`
        // chain — the chain exists exactly to rescue a bad primary. Genuine
        // request-content 4xx (400/413/422/…) stay terminal: an abusive
        // client must not fan one malformed request across every deployment
        // (see `outcomes::is_deployment_fault_status` for the deliberate
        // divergence from LiteLLM, which falls back on any exception).
        ModelOutcome::Client(ref e) if is_deployment_fault_status(e.status.as_u16()) => {
            let fbs = state.router.fallbacks_for(primary_model);
            if let Some(resp) = try_fallback_chain(
                state,
                call_ctx,
                auth,
                request_body,
                ctx,
                is_stream,
                traceparent,
                &fbs,
                &mut attempted_fallbacks,
                max_fallbacks,
            )
            .await
            {
                Ok(resp)
            } else {
                Err(primary_outcome.into_api_error())
            }
        }
        ModelOutcome::Client(e) => Err(e),
    };

    // Stamp the fallback header when a `default_fallbacks` reroute served
    // the request (same observability as the failure-driven chains). If the
    // rerouted model's own fallback chain already stamped the actual serving
    // model, keep it — the serving model is more useful than the reroute hop.
    match (result, &rerouted) {
        (Ok(mut resp), Some(default_model)) => {
            if !resp.headers().contains_key(FALLBACK_HEADER) {
                if let Ok(hv) = HeaderValue::from_str(default_model) {
                    resp.headers_mut().insert(FALLBACK_HEADER, hv);
                }
            }
            Ok(resp)
        }
        (result, _) => result,
    }
}

/// Resolve a `default_fallbacks` target for an UNKNOWN model group: the
/// first chain entry that passes the caller's model ACL (same check as
/// [`try_fallback_chain`]) and resolves to at least one deployment.
///
/// Returns `None` when the requested group is known — even if every one of
/// its deployments is cooling down, since failure of a KNOWN model is what
/// `fallbacks` handles — and when the chain is absent or exhausted (the
/// caller then preserves the usual model-not-found error).
fn resolve_default_fallback<'s>(
    state: &'s AppState,
    auth: &AuthUser,
    requested: &str,
) -> Option<&'s str> {
    if !state
        .router
        .deployments_for_model(&state.deployments, requested)
        .is_empty()
    {
        return None;
    }
    state
        .router
        .default_fallbacks_for(requested)
        .into_iter()
        .find(|default_model| {
            let allowed = check_model_acl(
                auth,
                default_model,
                auth.key_models.as_deref(),
                auth.team_models.as_deref(),
                auth.user_models.as_deref(),
            )
            .is_ok();
            if !allowed {
                tracing::info!(
                    fallback_model = %default_model,
                    "skipping default fallback model not allowed by key/team/user ACL"
                );
            }
            allowed
                && !state
                    .router
                    .deployments_for_model(&state.deployments, default_model)
                    .is_empty()
        })
}

/// Try a list of fallback models. Returns `Some(response)` if one succeeds.
///
/// A 4xx (client) error from a fallback model does **not** abort the chain:
/// the next fallback is tried instead, matching LiteLLM semantics. Only when
/// every fallback has been exhausted is the last error surfaced. The
/// `x-oxllm-fallback-used` header on a returned error reflects the last
/// fallback model that was tried.
#[expect(clippy::too_many_arguments)]
async fn try_fallback_chain(
    state: &AppState,
    call_ctx: &mut CallContext,
    auth: &AuthUser,
    request_body: &Value,
    ctx: &RequestContext,
    is_stream: bool,
    traceparent: Option<&str>,
    fallbacks: &[&str],
    attempted: &mut HashSet<String>,
    max_fallbacks: u32,
) -> Option<Response> {
    let mut last_error: Option<ApiError> = None;
    let mut last_tried: Option<&str> = None;
    let mut fallbacks_tried: u32 = 0;
    for fb_model in fallbacks {
        if fallbacks_tried >= max_fallbacks {
            tracing::info!(max_fallbacks, "stopping fallback walk at max_fallbacks");
            break;
        }
        if !attempted.insert((*fb_model).to_string()) {
            tracing::info!(
                fallback_model = %fb_model,
                "skipping fallback model already attempted for this request"
            );
            continue;
        }
        if check_model_acl(
            auth,
            fb_model,
            auth.key_models.as_deref(),
            auth.team_models.as_deref(),
            auth.user_models.as_deref(),
        )
        .is_err()
        {
            tracing::info!(
                fallback_model = %fb_model,
                "skipping fallback model not allowed by key/team/user ACL"
            );
            continue;
        }
        last_tried = Some(fb_model);
        if crate::pipeline::check_customer_model(state, call_ctx, fb_model)
            .await
            .is_err()
        {
            continue;
        }
        let fb_ctx = RequestContext {
            model: fb_model.to_string(),
            ..ctx.clone()
        };
        let mut fb_body = request_body.clone();
        if let Some(m) = fb_body.as_object_mut() {
            m.insert("model".to_string(), Value::String(fb_model.to_string()));
        }

        fallbacks_tried += 1;
        match try_model_with_retries(
            state,
            call_ctx,
            fb_model,
            &fb_body,
            &fb_ctx,
            is_stream,
            traceparent,
        )
        .await
        {
            ModelOutcome::Success(mut resp) => {
                if let Ok(hv) = HeaderValue::from_str(fb_model) {
                    resp.headers_mut().insert(FALLBACK_HEADER, hv);
                }
                // Router observability: a fallback chain served the request
                // (overrides any per-pick or default_fallbacks stamp).
                call_ctx.router_decision = Some("fallback".to_string());
                return Some(resp);
            }
            ModelOutcome::Client(e) => {
                let selector_rejected = crate::pipeline::routing::is_routing_error(&e);
                last_error = Some(e);
                if selector_rejected {
                    break;
                }
                continue;
            }
            outcome @ (ModelOutcome::ContextWindow(_)
            | ModelOutcome::ContentPolicy(_)
            | ModelOutcome::Retryable(_)) => {
                last_error = Some(outcome.into_api_error());
            }
        }
    }

    if let Some(e) = last_error {
        super::pipeline_failure(state, call_ctx, &e).await;
        let mut resp = e.into_response();
        if let Some(fb) = last_tried {
            if let Ok(hv) = HeaderValue::from_str(fb) {
                resp.headers_mut().insert(FALLBACK_HEADER, hv);
            }
        }
        return Some(resp);
    }
    None
}

/// Try a model across deployments. For each deployment, retry the **same**
/// deployment up to `num_retries` times (or the per-status count from
/// `retry_policy`) with exponential backoff. Only after exhausting retries on
/// a deployment is it marked as cooling, allowing the outer loop to pick a
/// different deployment.
///
/// U18 (LiteLLM commit `48fdaaa5cd`): LiteLLM's global `num_retries` was
/// stamped onto exceptions, overwriting a request-level override. oxllm does
/// NOT have this bug because it has **no per-request `num_retries` override
/// mechanism** — the retry budget is resolved solely from the router config
/// via `retries_for_status_model(status, model)`. There is nothing to
/// overwrite. (A request-level override is a future feature; see TODO.) This
/// is a missing-feature gap, not a correctness bug.
///
/// Two routing features hook into this loop:
///
/// - **Prompt-caching deployment check**: when the prompt prefix is large
///   enough for provider-side caching and a previous call pinned it to a
///   deployment, the router's pre-call check pipeline narrows the
///   candidates to the pinned deployment before the pick (this function
///   resolves the pin onto the request context beforehand — the pipeline
///   is the sole pin path); after a success, (re)write the pin.
/// - **Allowed-fails policy**: each failed attempt is counted per status;
///   once the configured threshold is reached the deployment is cooled
///   immediately instead of retrying up to `max_retries`.
async fn try_model_with_retries(
    state: &AppState,
    call_ctx: &mut CallContext,
    model: &str,
    request_body: &Value,
    ctx: &RequestContext,
    is_stream: bool,
    traceparent: Option<&str>,
) -> ModelOutcome {
    if let Err(error) = crate::pipeline::check_customer_model(state, call_ctx, model).await {
        return ModelOutcome::Client(error);
    }
    let default_retries = state.router.retries_for_status_model(0, model);
    let mut last_outcome: Option<ModelOutcome> = None;

    // Bedrock image-URL inlining. Bedrock's Converse and InvokeModel APIs
    // cannot fetch remote image URLs — they require base64-encoded image
    // bytes inline. When this model group has any Bedrock deployment AND
    // the request carries an HTTP(S) `image_url` block, download + re-encode
    // those URLs into `data:` URLs once, before the retry loop. Other
    // providers (OpenAI / Gemini / Anthropic-native / Vertex) fetch HTTP
    // URLs natively and skip the work. Failure is a hard 4xx: never
    // silently drop the image (data loss).
    let inlined_body_owned: Value;
    let request_body: &Value =
        if crate::pre_call_checks::image_inlining::needs_inlining(state, model, request_body) {
            let mut owned = request_body.clone();
            if let Err(e) = crate::pre_call_checks::image_inlining::inline_image_urls(
                &mut owned,
                &state.image_http_client,
            )
            .await
            {
                return ModelOutcome::Client(e.into());
            }
            inlined_body_owned = owned;
            &inlined_body_owned
        } else {
            request_body
        };

    // Prompt-caching check state. The prefix hash is computed at most once
    // per call; the pin lookup hits the cache at most once. The check is
    // opt-in: when `oxllm_settings.prompt_caching_enabled` is unset or
    // false the whole feature is skipped and routing is unaffected.
    let pcache_enabled = state.config.oxllm_settings.prompt_caching_enabled == Some(true);
    let pcache = PromptCachingCheck::new(
        state.cache.clone(),
        state.config.oxllm_settings.prompt_caching_min_tokens,
    );
    // Resolve the prompt hash and pin eagerly (they were previously lazy on
    // the first loop iteration — same one-shot memoization, hoisted so the
    // pre-call check pipeline can see the pin before the first pick).
    let hash_tokens: Option<(u128, u64)> = if pcache_enabled {
        pcache.prompt_hash(request_body, model)
    } else {
        None
    };
    // The lookup uses the *group* minimum threshold (min resolved
    // `prompt_cache_min_tokens` across this model group's deployments),
    // mirroring LiteLLM's `_get_min_token_count_for_deployments`: a high-floor
    // member (or a high global default) must not suppress a valid lookup for a
    // low-floor member that may hold a warm cache. Per-deployment thresholds
    // apply only to the pin write on success. The candidate group is the
    // model-name slice of the deployment list (health filtering happens later
    // in the router; using the whole group only makes the gate MORE permissive,
    // which is the safe direction for a cheap cache lookup).
    let group = state
        .router
        .deployments_for_model(&state.deployments, model);
    let pin_lookup: Option<String> = match hash_tokens {
        Some((hash, est_tokens)) if pcache.engages_globally(est_tokens, group.iter().copied()) => {
            pcache
                .preferred_deployment(hash, &call_ctx.api_key_hash)
                .await
        }
        _ => None,
    };

    // Pre-call check pipeline inputs. When prompt caching is enabled, the
    // `prompt_caching` check is always in the router's effective
    // `optional_pre_call_checks` (state build appends it when unlisted),
    // so the pipeline — the sole pin path — needs the host-resolved pin on
    // the request context. The `budget_limiter` check (listed AND
    // configured) needs the spend snapshot.
    let check_names = &state.router.config().optional_pre_call_checks;
    let budget_check_active = check_names
        .iter()
        .any(|n| crate::pre_call_checks::budget_limiter::is_budget_limiter_name(n))
        && crate::pre_call_checks::budget_limiter::BudgetLimiterCheck::is_configured(
            state.deployments.iter().map(|d| d.oxllm_params.max_budget),
            &state.router.config().provider_budget_config,
        );
    // The adaptive router's request-type classification is computed once
    // per request by `ChatOrchestrator::run` (whenever the adaptive
    // strategy is configured anywhere) and inherited by every leg through
    // the cloned request contexts — this function never re-classifies.
    // What is decided PER LEG is whether the classification may be USED:
    // stamping and per-pick failure debits engage only when THIS leg's
    // resolved strategy is the adaptive router, so an adaptive primary
    // falling back into a non-adaptive group never attributes outcomes to
    // bandit cells the strategy did not sample.
    let adaptive_active =
        state.router.strategy_kind_for(model) == oxllm_core::RoutingStrategyKind::AdaptiveRouter;
    let pick_ctx_owned;
    let pick_ctx: &RequestContext = if pcache_enabled || budget_check_active {
        let mut c = ctx.clone();
        if pcache_enabled {
            c.pre_call.pinned_deployment_key = pin_lookup.clone();
        }
        if budget_check_active {
            c.pre_call.spend_snapshot = super::budget::snapshot_router_spends(state).await;
        }
        pick_ctx_owned = c;
        &pick_ctx_owned
    } else {
        ctx
    };
    // The bandit cell request type for this leg, or `None` when the leg is
    // not adaptive-routed (or the request was never classified — e.g. a
    // caller that bypasses `ChatOrchestrator::run`): no stamping, no
    // debits, and the learning hook no-ops.
    let adaptive_type = if adaptive_active {
        pick_ctx.pre_call.adaptive_request_type
    } else {
        None
    };
    let mut pick_ctx = pick_ctx.clone();

    // Bound the outer pick loop so a misconfigured group cannot spin forever
    // (unary parity): with `cooldown_time_secs: 0` a failed deployment is
    // never excluded from rotation — `mark_cooldown_for` expires instantly —
    // so this bound, not the cooldown store, is the loop's only termination
    // guarantee. Deployments × (retries+1) covers every legitimate pick
    // (each deployment retried up to its budget) before giving up.
    let max_picks = state
        .deployments
        .len()
        .max(1)
        .saturating_mul((default_retries as usize + 1).max(1));
    'deployments: for _ in 0..max_picks {
        let dep_idx = match crate::pipeline::routing::try_pick_deployment_index(state, &pick_ctx) {
            Ok(Some(index)) => index,
            Ok(None) => break,
            Err(error) => return ModelOutcome::Client(error),
        };
        let deployment = &state.deployments[dep_idx];
        let dep_key = deployment_key(deployment);
        let cooldown_key = state.router.cooldown_key(deployment);

        // Stash the deployment's pricing override (if any) so the cost hooks
        // (CostTrackingHook, BudgetHook) and the cost header compute spend
        // using the deployment-specific rates. Reset on each iteration so a
        // previously-tried deployment's override does not leak into the next
        // attempt (e.g. fallback to a deployment without overrides).
        if let Err(error) =
            crate::pipeline::prepare_deployment_pricing(state, call_ctx, deployment, request_body)
                .await
        {
            pick_ctx
                .pre_call
                .excluded_deployment_keys
                .push(cooldown_key.clone());
            last_outcome = Some(ModelOutcome::Retryable(ApiError::budget_exceeded(
                error.to_string(),
            )));
            continue;
        }

        // Adaptive router: stamp the bandit cell identity for the learning
        // hook (`post_success` → alpha += 1, `post_failure` → beta += 1),
        // but only when THIS leg's pick was actually adaptive-sampled
        // (`adaptive_type` is Some) — outcomes of a non-adaptive leg must
        // not land in cells the bandit never sampled. Re-stamped on every
        // adaptive pick, so the post-call hooks always see the LAST
        // adaptive-sampled deployment; a non-adaptive pick CLEARS the
        // stamp so a later post-success/failure cannot credit an earlier
        // leg's cell. The deployment key is the router's stable cooldown key
        // (`stable_key(deployment)`, see
        // `oxllm_core::router::Router::cooldown_key`), matching the cell
        // keys the strategy sampled from.
        match adaptive_type {
            Some(request_type) => {
                call_ctx.adaptive_cell = Some((request_type, cooldown_key.clone()));
            }
            None => call_ctx.adaptive_cell = None,
        }

        // Router observability: stamp the strategy that produced this pick
        // and how the deployment was reached (re-stamped per leg, so the
        // serving leg's stamps win). Chain-level outcomes stamped by the
        // callers — `default_fallback` (execute_request_chain reroute) and
        // `fallback` (try_fallback_chain success) — take precedence and are
        // never clobbered by a per-pick stamp.
        call_ctx.router_strategy = Some(state.router.strategy_kind_for(model));
        if !matches!(
            call_ctx.router_decision.as_deref(),
            Some("default_fallback") | Some("fallback")
        ) {
            call_ctx.router_decision = Some(
                if pin_lookup.as_deref() == Some(cooldown_key.as_str()) {
                    "prompt_cache_pin"
                } else {
                    "picked"
                }
                .to_string(),
            );
        }

        // Reserve the in-flight slot immediately after picking this deployment
        // — BEFORE limit checks, provider resolution, or any other await that
        // would yield to another request. The previous ordering (increment
        // inside the retry loop, after `check_deployment_limits`) left a window
        // in which N concurrent requests could all observe count=0 and pick the
        // same "idle" deployment before any increment landed (#24).
        //
        // This narrows the race but does not fully close it: a complete
        // reserve-at-pick fix would atomically increment inside
        // `pick_deployment_index` itself (a router-level change). Here we
        // ensure the counter is visible to the next `pick_deployment_index`
        // caller as soon as we commit to a deployment.
        //
        // Every early-return path below performs the matching decrement. Inside
        // the retry loop, the decrement runs after each attempt; a retry
        // re-reserves the slot before the next attempt so the count stays
        // balanced. The in-flight/latency/usage stores are keyed by the stable
        // per-deployment key (`cooldown_key`), so sibling deployments sharing a
        // model name + provider model get distinct counters (a fast primary and
        // a slow fallback no longer share a least-busy/latency/usage bucket).
        state.router.record_in_flight(&cooldown_key, 1);

        if let Err(e) = check_deployment_limits(state, deployment, &dep_key).await {
            state.router.record_in_flight(&cooldown_key, -1);
            tracing::info!(
            deployment = %dep_key,
            "deployment rate-limited; marking cooling and trying next"
            );
            state.router.mark_cooldown_for(&cooldown_key, model);
            last_outcome = Some(ModelOutcome::Retryable(e));
            continue;
        }

        // Router budget limiting: atomically reserve an estimated cost
        // against the deployment/provider spend caps (same reservation
        // pattern the key-level BudgetHook uses). The reservation is settled
        // to the actual cost on success and released on every failure path.
        let budget_reservation = match super::budget::reserve_router_budgets(
            state, deployment, &dep_key, call_ctx,
        )
        .await
        {
            Ok(reservation) => reservation,
            Err(budget_err) => {
                state.router.record_in_flight(&cooldown_key, -1);
                tracing::info!(
                deployment = %dep_key,
                "deployment over budget; marking cooling and trying next"
                );
                // Cool for the exceeded budget's window (its configured
                // `budget_duration` / provider `time_period`), not the
                // default failure cooldown: a 60s cooldown would let the
                // deployment re-enter rotation and immediately re-fail the
                // budget check until the spend counter's window resets. A
                // zero window (pathological config) falls back to the
                // default cooldown.
                if budget_err.window.is_zero() {
                    state.router.mark_cooldown_for(&cooldown_key, model);
                } else {
                    state
                        .router
                        .cooldown_store()
                        .mark(&cooldown_key, budget_err.window);
                }
                last_outcome = Some(ModelOutcome::Retryable(budget_err.api_error));
                continue;
            }
        };

        let provider = match resolve_provider(state, deployment) {
            Ok(p) => p,
            Err(e) => {
                // Registry miss / unknown provider is a deployment-level
                // fault, not a client error for the whole model. Cool this
                // deployment and try siblings (same as credential failure).
                state.router.record_in_flight(&cooldown_key, -1);
                budget_reservation.release(&state.cache).await;
                tracing::warn!(
                    deployment = %cooldown_key,
                    error = ?e,
                    "provider resolution failed; cooling deployment and trying next"
                );
                state.router.mark_cooldown_for(&cooldown_key, model);
                last_outcome = Some(ModelOutcome::Retryable(e));
                continue;
            }
        };

        let mut attempt: u32 = 0;
        let mut max_retries = default_retries;
        let mut exhausted = false;
        let mut thinking_stripped = false;
        let mut stripped_body: Option<Value> = None;
        while !exhausted {
            let start = Instant::now();
            let body = stripped_body.as_ref().unwrap_or(request_body);
            let upstream_result = super::upstream::attempt_upstream_with_context(
                state,
                deployment,
                body,
                is_stream,
                &provider,
                traceparent,
                Some(call_ctx),
            )
            .await;
            // For a stream that got its 2xx headers, the in-flight slot stays
            // RESERVED until the stream task drains: `LeastBusy` must count
            // active streams, not just connect-time. `handle_stream_success`
            // decrements on every exit path of the spawned task (mirroring the
            // budget-reservation keep/release pattern). Non-streams — and
            // streams that failed BEFORE any 2xx headers (the slot was never
            // handed to a stream task) — release here as before.
            if !is_stream || upstream_result.is_err() {
                state.router.record_in_flight(&cooldown_key, -1);
            }

            // Health-check routing: track connectivity per attempt. Only
            // transport-level failures (network/DNS/TLS) and
            // locally-synthesized timeouts (no response ever arrived)
            // degrade health — any HTTP response, even an error status
            // (including a genuine upstream 504), proves the deployment
            // is reachable and resets its failure count. Both calls are
            // no-ops unless `enable_health_check_routing` is set.
            match &upstream_result {
                Err(UpstreamError::Network(msg)) | Err(UpstreamError::Timeout(msg)) => {
                    // `record_connection_failure` reports true only on the
                    // healthy→unhealthy transition — the exact moment the
                    // metric should fire (repeat failures while already
                    // unhealthy are not new events). The label is the
                    // non-secret `deployment_key`
                    // (`{model_name}:{provider_model}`): the cooldown key
                    // embeds a digest of credential material and must not
                    // surface on `/metrics`, which ordinary virtual-key
                    // holders can scrape.
                    if state.router.record_connection_failure(&cooldown_key, msg) {
                        ::metrics::counter!(
                            HEALTH_CHECK_UNHEALTHY_TOTAL,
                            "deployment" => dep_key.clone()
                        )
                        .increment(1);
                    }
                }
                _ => state.router.record_connection_success(&cooldown_key),
            }

            match upstream_result {
                Ok(response) => {
                    // Only record latency for successful attempts. A fast 500
                    // would otherwise look "healthier" than a slow 200 and skew
                    // the latency-based routing strategy (#24). For a stream
                    // this is the TTFB (headers-arrived elapsed) — the p50/EWMA
                    // stores must see streaming traffic, not just unary.
                    let elapsed = start.elapsed().as_millis() as u64;
                    state.router.record_latency(&cooldown_key, elapsed);
                    record_deployment_usage(state, &dep_key, &cooldown_key).await;
                    let result = if is_stream {
                        handle_stream_success(
                            response,
                            deployment.clone(),
                            provider.clone(),
                            state.clone(),
                            call_ctx.clone(),
                            budget_reservation.clone(),
                            model.to_string(),
                        )
                        .await
                        .map_err(NonStreamError::Deterministic)
                    } else {
                        handle_non_stream_success(
                            response,
                            deployment,
                            &provider,
                            state,
                            call_ctx,
                            &budget_reservation,
                        )
                        .await
                    };
                    match result {
                        Ok(resp) => {
                            // A healthy response resets the deployment's
                            // allowed-fails history.
                            state.router.clear_failure_counts(&cooldown_key);
                            // Post-call prompt-caching write: pin this prompt
                            // to the deployment that served it so follow-up
                            // requests hit a warm provider-side cache. The
                            // pin value is the stable cooldown key
                            // (`stable_key(deployment)`, see
                            // `oxllm_core::router::Router::cooldown_key`),
                            // which distinguishes deployments and is stable
                            // across config hot-reloads; the entry is scoped
                            // to the caller's API-key hash. The per-deployment
                            // threshold gates the write.
                            if let Some((hash, est_tokens)) = hash_tokens {
                                if pcache.engages(deployment, est_tokens) {
                                    pcache
                                        .pin(hash, &call_ctx.api_key_hash, &cooldown_key)
                                        .await;
                                }
                            }
                            // H3: traffic mirroring — fork a fire-and-forget
                            // shadow call to the silent model. Never awaited,
                            // never affects the response; failures land in a
                            // warn log. Spend is not charged (the silent task
                            // bypasses the budget/cost hooks).
                            super::spawn_silent_model_call(
                                state,
                                request_body,
                                deployment,
                                traceparent,
                            );
                            return ModelOutcome::Success(resp);
                        }
                        Err(NonStreamError::Deterministic(e)) => {
                            // The handler failed before settling the router
                            // budget reservation — release it. Deterministic
                            // post-2xx failures (oversize body, parse,
                            // transform) are not retryable: a sibling would
                            // reproduce them.
                            budget_reservation.release(&state.cache).await;
                            return ModelOutcome::Client(e);
                        }
                        Err(NonStreamError::Transport(e)) => {
                            // A transport failure while draining a 2xx body
                            // (e.g. connection reset mid-body) is transient.
                            // Mirror the `UpstreamError::Network` arm: retry
                            // within this deployment with backoff, counting
                            // against the allowed-fails status bucket 0 before
                            // cooling. The budget reservation is released once
                            // retries are exhausted (the loop-level release
                            // below).
                            last_outcome = Some(ModelOutcome::Retryable(e));
                            let limit_reached =
                                allowed_fails_limit_reached(state, &cooldown_key, 0, model);
                            if !limit_reached && attempt < max_retries {
                                crate::metrics::record_upstream_retry(
                                    model,
                                    deployment.provider_slug(),
                                );
                                retry_with_backoff(attempt, None).await;
                                attempt += 1;
                                state.router.record_in_flight(&cooldown_key, 1);
                            } else {
                                exhausted = true;
                            }
                        }
                    }
                }
                Err(UpstreamError::ContextWindow {
                    status,
                    message,
                    headers,
                }) => {
                    budget_reservation.release(&state.cache).await;
                    // A deployment persistently reporting context-window
                    // errors (e.g. a mis-declared max context) must be
                    // cooled down like any other deployment fault; without
                    // this, no failure is ever recorded and the deployment
                    // keeps being re-picked.
                    let limit_reached =
                        allowed_fails_limit_reached(state, &cooldown_key, status, model);
                    let sibling_available = state
                        .router
                        .deployments_for_model(&state.deployments, model)
                        .len()
                        > 1;
                    if (limit_reached || super::outcomes::is_deployment_fault_status(status))
                        && sibling_available
                    {
                        state.router.mark_cooldown_for(&cooldown_key, model);
                        tracing::info!(
                            deployment = %dep_key,
                            status = status,
                            "deployment-fault context-window error; cooling and failing over to a sibling"
                        );
                        last_outcome = Some(ModelOutcome::ContextWindow(
                            ApiError::upstream(status, message).with_headers(headers),
                        ));
                        continue 'deployments;
                    }
                    if limit_reached {
                        state.router.mark_cooldown_for(&cooldown_key, model);
                    }
                    return ModelOutcome::ContextWindow(
                        ApiError::upstream(status, message).with_headers(headers),
                    );
                }
                Err(UpstreamError::ContentPolicy {
                    status,
                    message,
                    headers,
                }) => {
                    budget_reservation.release(&state.cache).await;
                    // Same rationale as ContextWindow: record the failure so
                    // a deployment persistently blocked by upstream policy
                    // is eventually cooled down.
                    let limit_reached =
                        allowed_fails_limit_reached(state, &cooldown_key, status, model);
                    let sibling_available = state
                        .router
                        .deployments_for_model(&state.deployments, model)
                        .len()
                        > 1;
                    if limit_reached && sibling_available {
                        state.router.mark_cooldown_for(&cooldown_key, model);
                        tracing::info!(
                            deployment = %dep_key,
                            status = status,
                            "deployment-fault content-policy error; cooling and failing over to a sibling"
                        );
                        last_outcome = Some(ModelOutcome::ContentPolicy(
                            ApiError::upstream(status, message).with_headers(headers),
                        ));
                        continue 'deployments;
                    }
                    if limit_reached {
                        state.router.mark_cooldown_for(&cooldown_key, model);
                    }
                    return ModelOutcome::ContentPolicy(
                        ApiError::upstream(status, message).with_headers(headers),
                    );
                }
                Err(UpstreamError::Client {
                    status,
                    message,
                    headers,
                }) => {
                    if status == 400
                        && !thinking_stripped
                        && oxllm_providers::anthropic::is_invalid_thinking_signature_error(&message)
                    {
                        let mut body = stripped_body.take().unwrap_or_else(|| request_body.clone());
                        oxllm_providers::anthropic::strip_thinking_blocks_from_request(&mut body);
                        stripped_body = Some(body);
                        thinking_stripped = true;
                        tracing::warn!(
                            deployment = %dep_key,
                            "invalid thinking signature; retrying without thinking blocks"
                        );
                        crate::metrics::record_upstream_retry(model, deployment.provider_slug());
                        state.router.record_in_flight(&cooldown_key, 1);
                        continue;
                    }
                    // The allowed-fails policy can cool a deployment on
                    // specific 4xx statuses (e.g. cool immediately on 401).
                    // Without a policy covering the status nothing is
                    // recorded and behavior is unchanged.
                    let limit_reached =
                        allowed_fails_limit_reached(state, &cooldown_key, status, model);
                    // LiteLLM parity (`litellm/router.py` `_should_retry`:
                    // AuthenticationError is retried "if other deployments
                    // available"): when the policy just declared this 4xx a
                    // deployment fault (limit crossed — stale credentials, a
                    // gone model) AND the model group has a sibling to
                    // serve the request, cool this deployment and fail over
                    // instead of failing the caller. Genuinely client-caused
                    // 4xx (400/403/413, or any status no policy covers)
                    // still return immediately: an abusive client must not
                    // fan one bad request across every deployment.
                    let sibling_available = state
                        .router
                        .deployments_for_model(&state.deployments, model)
                        .len()
                        > 1;
                    if (limit_reached || is_deployment_fault_status(status)) && sibling_available {
                        state.router.mark_cooldown_for(&cooldown_key, model);
                        tracing::info!(
                            deployment = %dep_key,
                            status = status,
                            "deployment-fault client error; cooling and failing over to a sibling"
                        );
                        budget_reservation.release(&state.cache).await;
                        last_outcome = Some(ModelOutcome::Client(
                            ApiError::upstream(status, message).with_headers(headers),
                        ));
                        continue 'deployments;
                    }
                    if limit_reached {
                        // Zero-cooldown edge case: when
                        // `cooldown_time_secs == 0`, the failure-counting
                        // window falls back to 60s but the resulting
                        // cooldown is a no-op (the deployment is immediately
                        // available again) — intentional, matching LiteLLM.
                        state.router.mark_cooldown_for(&cooldown_key, model);
                    }
                    budget_reservation.release(&state.cache).await;
                    return ModelOutcome::Client(
                        ApiError::upstream(status, message).with_headers(headers),
                    );
                }
                Err(UpstreamError::Retryable {
                    status,
                    message,
                    headers,
                }) => {
                    max_retries = state.router.retries_for_status_model(status, model);
                    last_outcome = Some(ModelOutcome::Retryable(
                        ApiError::upstream(status, message).with_headers(headers.clone()),
                    ));
                    let limit_reached =
                        allowed_fails_limit_reached(state, &cooldown_key, status, model);
                    if !limit_reached && attempt < max_retries {
                        let retry_after = retry_after_secs(headers.as_ref());
                        crate::metrics::record_upstream_retry(model, deployment.provider_slug());
                        retry_with_backoff(attempt, retry_after).await;
                        attempt += 1;
                        // Re-reserve the slot decremented above for the next attempt.
                        state.router.record_in_flight(&cooldown_key, 1);
                    } else {
                        exhausted = true;
                    }
                }
                Err(UpstreamError::Timeout(message)) => {
                    // A local timeout is retried and cooled exactly like a
                    // retryable upstream 504 (same per-status retry count,
                    // same allowed-fails status bucket); only the health
                    // classification above differs.
                    max_retries = state.router.retries_for_status_model(504, model);
                    last_outcome = Some(ModelOutcome::Retryable(ApiError::upstream(504, message)));
                    let limit_reached =
                        allowed_fails_limit_reached(state, &cooldown_key, 504, model);
                    if !limit_reached && attempt < max_retries {
                        crate::metrics::record_upstream_retry(model, deployment.provider_slug());
                        retry_with_backoff(attempt, None).await;
                        attempt += 1;
                        state.router.record_in_flight(&cooldown_key, 1);
                    } else {
                        exhausted = true;
                    }
                }
                Err(UpstreamError::Network(msg)) => {
                    // The message carries the raw reqwest/URL detail (custom
                    // `api_base` targets, private gateways); route it through
                    // the redaction contract — the detail is logged
                    // server-side and the client sees a generic 502.
                    last_outcome = Some(ModelOutcome::Retryable(ApiError::from(
                        oxllm_core::Error::Network(msg),
                    )));
                    // Network failures carry no HTTP status; they count under
                    // the global allowed-fails threshold (status bucket 0).
                    let limit_reached = allowed_fails_limit_reached(state, &cooldown_key, 0, model);
                    if !limit_reached && attempt < max_retries {
                        crate::metrics::record_upstream_retry(model, deployment.provider_slug());
                        retry_with_backoff(attempt, None).await;
                        attempt += 1;
                        state.router.record_in_flight(&cooldown_key, 1);
                    } else {
                        exhausted = true;
                    }
                }
            }
        }

        // Giving up on this deployment: release the router budget
        // reservation before cooling, so the next candidate picks up a
        // clean slate.
        budget_reservation.release(&state.cache).await;
        // U1 (LiteLLM `96f58fac53`): the cooldown is applied to
        // `cooldown_key` — the stable per-deployment key, unique per
        // deployment. This means a failed sub-call only cools
        // down the deployment that actually failed, never the parent.
        // (LiteLLM's bug was a shared error path that attributed sub-call
        // failures to the parent deployment; oxllm's per-deployment key
        // avoids this.)
        // Zero-cooldown edge case: when `cooldown_time_secs == 0`, the
        // failure-counting window falls back to 60s but the resulting
        // cooldown is a no-op (the deployment is immediately available
        // again) — intentional, matching LiteLLM.
        state.router.mark_cooldown_for(&cooldown_key, model);
        // Adaptive router: debit the failing deployment's cell at the point
        // its retries are exhausted. The once-per-request learning hook
        // only ever sees the LAST stamped cell, so without this debit an
        // intermediate failure rescued by failover would learn nothing (a
        // permanently-failing-but-rescued deployment would also never
        // gather the observations its cell needs to leave cold start).
        // Only upstream-failure exhaustion debits — 4xx client errors,
        // budget rejections, and deployment-limit cooldowns say nothing
        // about the deployment's ability to serve. After the debit the
        // stamped cell is CLEARED, so a terminal failure's `post_failure`
        // takes the unstamped-skip path: each failed deployment is debited
        // exactly once per request. When failover rescues the request the
        // next adaptive pick simply re-stamps. The hook is bypassed for
        // this debit, so the alpha/beta gauges are re-exported here too —
        // otherwise always-failing paths leave them permanently behind
        // the store.
        // (The cell key is the stable cooldown key `{model_name}:{provider_model}`,
        // matching the keys the adaptive strategy sampled from.)
        if let Some(request_type) = adaptive_type {
            let cell = (request_type, cooldown_key.clone());
            state.router.bandit_store().record_failure(&cell);
            crate::hooks::adaptive_hook::export_cell_gauges(state.router.bandit_store(), &cell);
            call_ctx.adaptive_cell = None;
        }
    }

    last_outcome.unwrap_or({
        // Use the same resolution path as routing (aliases + wildcards), not
        // exact `model_name == model`, so a cooled-out `*` / alias group is
        // reported as temporarily unavailable (429) rather than 404.
        let model_exists = !state
            .router
            .deployments_for_model(&state.deployments, model)
            .is_empty();
        if model_exists {
            ModelOutcome::Retryable(ApiError::rate_limited(format!(
                "all deployments for '{}' are rate-limited or unavailable",
                crate::error::truncate_model_for_echo(model)
            )))
        } else {
            ModelOutcome::Retryable(ApiError::model_not_found(model))
        }
    })
}

/// Record a failed attempt under the allowed-fails policy and log it.
/// Returns `true` when the failure count reached the configured limit and
/// the deployment should be cooled instead of retried. When no policy
/// covers the status, records nothing and returns `false` (existing
/// behavior: retry up to `max_retries`, then cool).
///
/// Shared by streaming and realtime sessions for terminal failures, where
/// there is no retry loop to own allowed-fails / percentage cooldown.
pub(crate) fn allowed_fails_limit_reached(
    state: &AppState,
    cooldown_key: &str,
    status: u16,
    model: &str,
) -> bool {
    match state
        .router
        .record_failure_and_check(cooldown_key, status, model)
    {
        AllowedFailDecision::Tolerated { count, limit } => {
            tracing::debug!(
                event = "allowed_fail",
                deployment = %cooldown_key,
                status = status,
                count = count,
                limit = limit,
                "failure tolerated by allowed-fails policy"
            );
            false
        }
        AllowedFailDecision::LimitReached { count, limit } => {
            tracing::debug!(
                event = "allowed_fail",
                deployment = %cooldown_key,
                status = status,
                count = count,
                limit = limit,
                "allowed-fails limit reached; cooling deployment"
            );
            true
        }
        AllowedFailDecision::NoPolicy => false,
    }
}

/// Exponential backoff with jitter: `base * 2^attempt + random_jitter`,
/// capped at 30 seconds.
///
/// When the upstream sent a `Retry-After` header (parsed via
/// [`super::outcomes::retry_after_secs`]), the sleep is at least
/// `retry_after + jitter` — LiteLLM's `_calculate_retry_after`
/// (`litellm/utils.py`) honors the header whenever `0 < retry_after <= 60`
/// instead of burning the retry budget inside the provider's cooldown
/// window. The exponential value wins when it is larger.
///
/// Shared by the chat and unary retry loops so embeddings/rerank traffic
/// backs off identically to chat (preventing thundering-herd retries and
/// over-aggressive burn-through of the retry budget).
pub(super) async fn retry_with_backoff(attempt: u32, retry_after: Option<u64>) {
    const BASE_DELAY_MS: u64 = 1_000;
    const MAX_DELAY_MS: u64 = 30_000;
    const JITTER_MS: u64 = 1_000;
    const MAX_RETRY_AFTER_SECS: u64 = 60;

    let exponent = attempt.min(5) as u64;
    let base = BASE_DELAY_MS.saturating_mul(1u64 << exponent);
    let jitter = rand::random_range(0..JITTER_MS);
    let mut total = base.saturating_add(jitter).min(MAX_DELAY_MS);
    if let Some(secs) = retry_after {
        if 0 < secs && secs <= MAX_RETRY_AFTER_SECS {
            let retry_after_ms = secs.saturating_mul(1_000).saturating_add(jitter);
            total = total.max(retry_after_ms);
        }
    }

    tokio::time::sleep(Duration::from_millis(total)).await;
}

/// Snapshot the deployment's RPM cache counter into the `UsageStore` feed.
///
/// The counter itself stays keyed by `dep_key` (matching
/// [`check_deployment_limits`](crate::pipeline::check_deployment_limits)'s
/// `dep_rpm:` keys), while the store write uses the stable per-deployment
/// `store_key` so sibling deployments get distinct usage buckets.
///
/// Shared with unary and realtime executors so `usage_based` routing sees
/// embeddings, rerank and active-session admission, not just chat.
pub(crate) async fn record_deployment_usage(state: &AppState, dep_key: &str, store_key: &str) {
    let counter_key = format!("dep_rpm:{dep_key}");
    let current = state.cache.counter_value(&counter_key).await.unwrap_or(0.0);
    state.router.record_usage(store_key, current);
}

/// Failure modes after an upstream 2xx was received but before the client
/// response could be produced.
enum NonStreamError {
    /// Transport-level failure while draining the upstream body (e.g. a
    /// connection reset mid-body). Transient — retried within the deployment
    /// like a network error.
    Transport(ApiError),
    /// Deterministic failure (oversize body, JSON parse, response transform).
    /// Retrying a sibling would reproduce it, so it is a non-retryable client
    /// outcome.
    Deterministic(ApiError),
}

impl From<ApiError> for NonStreamError {
    fn from(e: ApiError) -> Self {
        NonStreamError::Deterministic(e)
    }
}

/// Read a non-streaming success body into memory, bounded by
/// [`crate::security::MAX_RESPONSE_BODY_BYTES`].
///
/// A chunked / no-`Content-Length` upstream that streams more than the cap is
/// aborted mid-stream rather than buffered wholesale, so a misbehaving
/// upstream cannot pin an unbounded body in proxy heap before the post-read
/// limit fires (a `Content-Length` pre-check rejects an honestly-advertised
/// oversize body header-only). An oversize body is a deterministic failure —
/// a sibling deployment would reproduce it — while a transport error
/// mid-read is transient and retryable.
async fn read_success_body(response: reqwest::Response) -> Result<Vec<u8>, NonStreamError> {
    crate::security::check_content_length(
        response.headers(),
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )?;
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if buf.len().saturating_add(bytes.len()) > crate::security::MAX_RESPONSE_BODY_BYTES
                {
                    return Err(NonStreamError::Deterministic(ApiError::internal(
                        "upstream response too large: body exceeds limit",
                    )));
                }
                buf.extend_from_slice(&bytes);
            }
            Err(e) => {
                return Err(NonStreamError::Transport(ApiError::from(
                    oxllm_core::Error::Network(format!("failed to read upstream response: {e}")),
                )));
            }
        }
    }
    Ok(buf)
}

/// Handle a non-streaming success response (2xx from upstream).
///
/// Transforms the response, sets the cost header, and runs the post-success
/// hook chain (budget debit, rate-limit increment, cache store, cost log).
/// `budget_reservation` is the router-budget reservation made at pick time;
/// it is settled to the actual cost here.
async fn handle_non_stream_success(
    upstream_response: reqwest::Response,
    deployment: &Deployment,
    provider: &Arc<dyn oxllm_core::Provider>,
    state: &AppState,
    call_ctx: &CallContext,
    budget_reservation: &super::budget::RouterBudgetReservation,
) -> Result<Response, NonStreamError> {
    let bytes = read_success_body(upstream_response).await?;
    let response_body: Value = serde_json::from_slice(&bytes).map_err(|e| {
        tracing::error!(
            error.kind = "serialization",
            detail = %e,
            "failed to parse upstream response as JSON"
        );
        NonStreamError::Deterministic(ApiError::internal(
            "failed to parse upstream response as JSON",
        ))
    })?;

    let chat = provider.chat().ok_or_else(|| {
        ApiError::internal("provider lost chat capability between request and response")
    })?;
    let transformed = chat
        .transform_response_with_request(
            &response_body,
            &deployment.oxllm_params.model,
            &call_ctx.request_body,
        )
        .map_err(|e| NonStreamError::Deterministic(ApiError::from(e)))?;

    let response_model = transformed
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&deployment.model_name)
        .to_string();
    let usage = extract_usage(&transformed);

    if let Some(u) = usage.as_ref() {
        record_deployment_tpm(state, &deployment_key(deployment), u.total_tokens).await;
    }

    let cost = post_success(
        state,
        call_ctx,
        &response_model,
        usage.as_ref(),
        Some(&transformed),
        None,
        deployment.provider_slug(),
    )
    .await;

    // Router budget limiting: settle the deployment/provider spend counters
    // with the actual cost (reservation made at pick time; a configured
    // budget with unknown cost records a nominal increment and logs a
    // warning).
    super::budget::settle_router_spend(
        state,
        deployment,
        &deployment_key(deployment),
        cost,
        budget_reservation,
    )
    .await;

    let mut resp = Json(transformed).into_response();
    apply_cost_header(&mut resp, cost);

    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{
        ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings, RoutingKnobs,
    };

    /// AppState whose deployments are one per given model name, with the
    /// given `router_settings.default_fallbacks` knob.
    fn state_with(models: &[&str], default_fallbacks: Option<Value>) -> AppState {
        let config = ProxyConfig {
            model_list: models
                .iter()
                .map(|m| ConfigModel {
                    model_name: m.to_string(),
                    oxllm_params: oxllm_core::deployment::OxllmParams {
                        model: format!("openai/{m}"),
                        api_key: Some("k".to_string()),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .collect(),
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    default_fallbacks,
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    fn auth_with_key_models(models: Option<Vec<String>>) -> AuthUser {
        let auth = AuthUser::open();
        AuthUser {
            key_models: models,
            ..auth
        }
    }

    #[test]
    fn default_fallback_unknown_model_resolves_first_reachable() {
        let state = state_with(
            &["model-a", "model-b"],
            Some(serde_json::json!(["model-a", "model-b"])),
        );
        let resolved = resolve_default_fallback(&state, &AuthUser::open(), "unknown-model");
        assert_eq!(resolved, Some("model-a"));
    }

    #[test]
    fn default_fallback_known_model_is_not_rerouted() {
        // Guard: a known group must NOT consult default_fallbacks — even
        // with a chain configured. (Cooled-but-known is the `fallbacks`
        // feature's job.)
        let state = state_with(&["model-a"], Some(serde_json::json!(["model-a"])));
        assert!(resolve_default_fallback(&state, &AuthUser::open(), "model-a").is_none());
    }

    #[test]
    fn default_fallback_skips_acl_denied_entries() {
        let state = state_with(
            &["model-a", "model-b"],
            Some(serde_json::json!(["model-a", "model-b"])),
        );
        let auth = auth_with_key_models(Some(vec!["model-b".to_string()]));
        let resolved = resolve_default_fallback(&state, &auth, "unknown-model");
        assert_eq!(
            resolved,
            Some("model-b"),
            "ACL-denied chain entries must be skipped"
        );
    }

    #[test]
    fn default_fallback_all_acl_denied_returns_none() {
        let state = state_with(&["model-a"], Some(serde_json::json!(["model-a"])));
        let auth = auth_with_key_models(Some(vec!["other".to_string()]));
        assert!(resolve_default_fallback(&state, &auth, "unknown-model").is_none());
    }

    #[test]
    fn default_fallback_skips_entries_without_deployments() {
        let state = state_with(&["model-a"], Some(serde_json::json!(["ghost", "model-a"])));
        let resolved = resolve_default_fallback(&state, &AuthUser::open(), "unknown-model");
        assert_eq!(resolved, Some("model-a"));
    }

    #[test]
    fn default_fallback_unconfigured_returns_none() {
        let state = state_with(&["model-a"], None);
        assert!(resolve_default_fallback(&state, &AuthUser::open(), "unknown-model").is_none());
    }

    #[test]
    fn default_fallback_dict_form_matches_original_model() {
        let state = state_with(
            &["model-a", "model-b"],
            Some(serde_json::json!([
                {"original-model": ["model-a"]},
                {"*": ["model-b"]}
            ])),
        );
        assert_eq!(
            resolve_default_fallback(&state, &AuthUser::open(), "original-model"),
            Some("model-a"),
            "exact rule wins over the wildcard"
        );
        assert_eq!(
            resolve_default_fallback(&state, &AuthUser::open(), "anything-else"),
            Some("model-b"),
            "unmatched models fall through to the wildcard rule"
        );
    }

    /// LOW: `execute_request_chain` must not panic when the request body is
    /// not a JSON object. The reroute (`default_fallbacks`) and fallback-chain
    /// paths rewrite `body["model"]` via `Value::IndexMut`, which panics on
    /// non-object values. The orchestrator guards those writes with
    /// `as_object_mut` so a malformed body surfaces as a typed `ApiError`
    /// rather than a panic. In practice the handler validates the body is an
    /// object before reaching the orchestrator, so this is defence-in-depth.
    #[tokio::test]
    async fn execute_request_chain_non_object_body_does_not_panic() {
        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "real-model".to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: "openai/real-model".to_string(),
                    api_key: Some("k".to_string()),
                    // Closed port: upstream fails instantly without real I/O.
                    api_base: Some("http://127.0.0.1:1".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    default_fallbacks: Some(serde_json::json!(["real-model"])),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let state =
            AppState::with_registry(config, oxllm_providers::default_registry()).expect("state");

        let mut call_ctx = CallContext::for_test("unknown-model");
        let ctx = RequestContext::new("unknown-model");
        // Non-object body: previously triggered a panic in the reroute path
        // via `df_body["model"] = ...`.
        let body = Value::String("not-an-object".to_string());

        let result = execute_request_chain(
            &state,
            &mut call_ctx,
            &AuthUser::open(),
            "unknown-model",
            &body,
            &ctx,
            false,
            None,
        )
        .await;

        // The body is invalid; the orchestrator must return a typed ApiError
        // (never panic). The exact status depends on where the upstream call
        // fails, so we only assert it is an error.
        assert!(
            result.is_err(),
            "non-object body must surface as an ApiError, not a panic"
        );
    }

    /// Build a single-deployment state whose upstream points at a closed port
    /// and whose router never retries, so a connection-refused request fails
    /// immediately without backoff sleeps.
    fn closed_port_state(allow_private: bool) -> AppState {
        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("k".to_string()),
                    api_base: Some("http://127.0.0.1:1".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: allow_private.then_some(true),
                ..Default::default()
            },
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    num_retries: Some(0),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    /// A connection-refused upstream must render a generic 502 whose body
    /// contains neither the resolved upstream URL nor the raw reqwest detail:
    /// `UpstreamError::Network` carries `execute_upstream_json`'s message,
    /// which embeds the target URL, and the retry loop previously forwarded it
    /// verbatim via `ApiError::internal`. The redaction contract
    /// (error.rs:245-252) says the detail is logged server-side only.
    #[tokio::test]
    async fn network_error_response_body_is_redacted() {
        let state = closed_port_state(true);

        let mut call_ctx = CallContext::for_test("gpt-4o");
        let ctx = RequestContext::new("gpt-4o");
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let result = execute_request_chain(
            &state,
            &mut call_ctx,
            &AuthUser::open(),
            "gpt-4o",
            &body,
            &ctx,
            false,
            None,
        )
        .await;

        let err = result.expect_err("connection refused must fail");
        assert_eq!(
            err.status,
            axum::http::StatusCode::BAD_GATEWAY,
            "reachability failure must be a 502, not a 500"
        );
        let resp = err.into_response();
        let bytes = axum::body::to_bytes(resp.into_body(), 8 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            !text.contains("127.0.0.1"),
            "resolved upstream URL must not leak into the body: {text}"
        );
        assert!(
            !text.contains("network error"),
            "raw reqwest detail must not leak into the body: {text}"
        );
        assert!(
            text.contains("upstream error"),
            "client must receive the generic message, got: {text}"
        );
    }

    /// MEDIUM (C11): the retry backoff must honor an upstream `Retry-After`
    /// of 2s (LiteLLM `_calculate_retry_after`): the sleep is at least
    /// `retry_after + jitter` rather than the raw 1s exponential base, so a
    /// provider 429 with a cooldown window is not hammered at ~1s intervals.
    /// Paused-clock: tokio auto-advances the mock clock through the sleep.
    #[tokio::test(start_paused = true)]
    async fn retry_backoff_honors_retry_after_header_value() {
        let start = tokio::time::Instant::now();
        retry_with_backoff(0, Some(2)).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_secs(2),
            "retry-after: 2 must delay ≥2s (got {elapsed:?})"
        );
        assert!(
            elapsed < Duration::from_secs(4),
            "delay must stay near retry-after + jitter (got {elapsed:?})"
        );
    }

    /// A `Retry-After` larger than LiteLLM's 60s ceiling is ignored: the
    /// exponential backoff applies instead.
    #[tokio::test(start_paused = true)]
    async fn retry_backoff_ignores_oversized_retry_after() {
        let start = tokio::time::Instant::now();
        retry_with_backoff(0, Some(600)).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "oversized retry-after must not be honored (got {elapsed:?})"
        );
    }

    /// Spawn a raw TCP server that simulates a mid-body transport failure.
    /// The first connection receives a `200 OK` head advertising a
    /// `content-length` larger than the body actually transmitted, then the
    /// socket closes — the client observes a body-read error. Subsequent
    /// connections serve a well-formed chat-completion body so a retry can
    /// succeed.
    ///
    /// Returns the server base URL and an atomic count of accepted
    /// connections.
    fn spawn_partial_then_ok_server() -> (String, Arc<std::sync::atomic::AtomicU32>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::atomic::{AtomicU32, Ordering};

        let hits = Arc::new(AtomicU32::new(0));
        let hits_for_server = hits.clone();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let hits_for_conn = hits_for_server.clone();
                hits_for_conn.fetch_add(1, Ordering::SeqCst);
                std::thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
                    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
                    let mut buf = [0u8; 8192];
                    let _ = stream.read(&mut buf);
                    let head = if hits_for_conn.load(Ordering::SeqCst) == 1 {
                        // 9 bytes of body against a declared 100 → truncated.
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\n{\"partial\""
                            .as_bytes()
                            .to_vec()
                    } else {
                        let body = r#"{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"model":"gpt-4o"}"#;
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                            body.len()
                        )
                        .into_bytes()
                    };
                    let _ = stream.write_all(&head);
                    // Dropping the socket after the truncated body simulates a
                    // connection reset/close mid-response.
                });
            }
        });
        (format!("http://{addr}"), hits)
    }

    /// A connection reset while draining a 2xx body is a transient transport
    /// failure: it must be retried within the deployment (LiteLLM treats this
    /// as `APIConnectionError` and retries) rather than surfacing as a hard
    /// non-retryable client error. The first attempt gets a truncated body and
    /// a closed connection; the retry hits the healthy second response and the
    /// request succeeds.
    #[tokio::test]
    async fn post_2xx_read_failure_is_retried() {
        let (url, hits) = spawn_partial_then_ok_server();
        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("k".to_string()),
                    api_base: Some(url),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    num_retries: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built");

        let mut call_ctx = CallContext::for_test("gpt-4o");
        let ctx = RequestContext::new("gpt-4o");
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let result = execute_request_chain(
            &state,
            &mut call_ctx,
            &AuthUser::open(),
            "gpt-4o",
            &body,
            &ctx,
            false,
            None,
        )
        .await;

        let resp = result.unwrap_or_else(|_| {
            panic!(
                "mid-body reset must be retried and succeed (hits={})",
                hits.load(std::sync::atomic::Ordering::SeqCst)
            );
        });
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "the retried request must succeed"
        );
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the failed read must be retried once (first truncated, then healthy)"
        );
    }

    /// `record_upstream_retry` must fire for the mid-body transport retry
    /// site (`NonStreamError::Transport`) as well as retryable HTTP statuses.
    /// A truncated-then-healthy upstream (same harness as
    /// `post_2xx_read_failure_is_retried`) must start exactly one additional
    /// same-deployment attempt and increment
    /// `oxllm_upstream_retries_total` once under the requested model +
    /// serving provider. The unique model label isolates this series from
    /// other tests sharing the process-wide recorder. The retryable-status
    /// (500) site is pinned end to end by
    /// `tests/router_e2e.rs::test_retry_succeeds_on_second_deployment`.
    #[tokio::test]
    #[serial_test::serial]
    async fn upstream_retry_metric_increments_on_transport_retry() {
        let _ = crate::metrics::install();
        let (url, hits) = spawn_partial_then_ok_server();
        let model = "retry-metric-transport-probe";
        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: model.to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: format!("openai/{model}"),
                    api_key: Some("k".to_string()),
                    api_base: Some(url),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    num_retries: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built");

        let mut call_ctx = CallContext::for_test(model);
        let ctx = RequestContext::new(model);
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let before = upstream_retry_count(model, "openai");
        let resp = execute_request_chain(
            &state,
            &mut call_ctx,
            &AuthUser::open(),
            model,
            &body,
            &ctx,
            false,
            None,
        )
        .await
        .expect("a mid-body reset must be retried and succeed");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the first attempt was truncated; the retry must be the second hit"
        );
        assert_eq!(
            upstream_retry_count(model, "openai").saturating_sub(before),
            1,
            "exactly one started retry must increment oxllm_upstream_retries_total"
        );
    }

    /// Parse the current value of `oxllm_upstream_retries_total` for the
    /// given `model` / `provider` labels out of the rendered Prometheus
    /// exposition text (0 when the series is absent).
    fn upstream_retry_count(model: &str, provider: &str) -> u64 {
        let model_needle = format!("model=\"{model}\"");
        let provider_needle = format!("provider=\"{provider}\"");
        for line in crate::metrics::render().lines() {
            if line.starts_with("oxllm_upstream_retries_total{")
                && line.contains(&model_needle)
                && line.contains(&provider_needle)
            {
                if let Some(v) = line.rsplit(' ').next() {
                    if let Ok(n) = v.parse::<u64>() {
                        return n;
                    }
                }
            }
        }
        0
    }

    // ---- MEDIUM (C11): primary deployment-fault 4xx vs configured fallbacks --

    /// State with a `primary-model` group (upstream at `primary_url`) and a
    /// `secondary-model` group (upstream at `secondary_url`), plus
    /// `fallbacks: [{primary-model: [secondary-model]}]` and zero retries so
    /// failures surface without backoff sleeps.
    fn fallback_chain_state(primary_url: String, secondary_url: String) -> AppState {
        let config = ProxyConfig {
            model_list: vec![
                ConfigModel {
                    model_name: "primary-model".to_string(),
                    oxllm_params: oxllm_core::deployment::OxllmParams {
                        model: "openai/primary-model".to_string(),
                        api_key: Some("k".to_string()),
                        api_base: Some(primary_url),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ConfigModel {
                    model_name: "secondary-model".to_string(),
                    oxllm_params: oxllm_core::deployment::OxllmParams {
                        model: "openai/secondary-model".to_string(),
                        api_key: Some("k".to_string()),
                        api_base: Some(secondary_url),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    num_retries: Some(0),
                    fallbacks: Some(serde_json::json!([
                        {"primary-model": ["secondary-model"]}
                    ])),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    /// MEDIUM (C11): LiteLLM's `async_function_with_fallbacks`
    /// (`litellm/router.py:6435`) walks the configured `fallbacks` chain on
    /// any primary exception after retries — including a 401 (stale vendor
    /// credential). A 401 from the primary with a healthy secondary
    /// configured must therefore serve the request from the secondary;
    /// previously the 401 aborted the whole chain and the caller saw the
    /// primary's error even though the operator configured a rescue.
    #[tokio::test]
    async fn primary_401_falls_back_to_configured_secondary() {
        use httpmock::MockServer;
        let primary = MockServer::start();
        let secondary = MockServer::start();
        primary.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(401)
                .header("content-type", "application/json")
                .body(serde_json::json!({"error": {"message": "invalid api key"}}).to_string());
        });
        let secondary_mock = secondary.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200).header("content-type", "application/json").body(
                serde_json::json!({
                    "id": "c", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "model": "secondary-model"
                })
                .to_string(),
            );
        });

        let state = fallback_chain_state(primary.base_url(), secondary.base_url());
        let mut call_ctx = CallContext::for_test("primary-model");
        let ctx = RequestContext::new("primary-model");
        let body = serde_json::json!({
            "model": "primary-model",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let resp = execute_request_chain(
            &state,
            &mut call_ctx,
            &AuthUser::open(),
            "primary-model",
            &body,
            &ctx,
            false,
            None,
        )
        .await
        .expect("the secondary must rescue the primary's 401");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            secondary_mock.hits(),
            1,
            "the secondary must have served the request"
        );
        assert!(
            resp.headers().contains_key(super::FALLBACK_HEADER),
            "the fallback header must be stamped"
        );
    }

    /// Companion guard: a genuine request-content 400 from the primary must
    /// NOT fan out to the fallback chain — an abusive client must not be
    /// able to push one malformed request to every configured deployment
    /// (deliberate divergence from LiteLLM, which falls back on any
    /// exception; see `outcomes::is_deployment_fault_status`).
    #[tokio::test]
    async fn primary_400_does_not_fan_out_to_fallbacks() {
        use httpmock::MockServer;
        let primary = MockServer::start();
        let secondary = MockServer::start();
        primary.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(400)
                .header("content-type", "application/json")
                .body(
                    serde_json::json!({"error": {"message": "max_tokens is too large"}})
                        .to_string(),
                );
        });
        let secondary_mock = secondary.mock(|_when, _then| {});

        let state = fallback_chain_state(primary.base_url(), secondary.base_url());
        let mut call_ctx = CallContext::for_test("primary-model");
        let ctx = RequestContext::new("primary-model");
        let body = serde_json::json!({
            "model": "primary-model",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let err = execute_request_chain(
            &state,
            &mut call_ctx,
            &AuthUser::open(),
            "primary-model",
            &body,
            &ctx,
            false,
            None,
        )
        .await
        .expect_err("a request-content 400 must surface to the caller");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            secondary_mock.hits(),
            0,
            "a client-caused 4xx must not reach the fallback chain"
        );
    }

    /// MEDIUM (C11): a provider `transform_request` failure is a
    /// deployment-level fault: in a heterogeneous model group (one OpenAI +
    /// one stub deployment whose transform cannot map the request), the
    /// request must still be served by the healthy sibling. Previously the
    /// transform error was terminal `Client { 500 }` and the whole chain
    /// aborted whenever the untransformable member was picked.
    /// `round_robin` (deterministic first pick) puts the stub first, so the
    /// failover is exercised on every run.
    #[tokio::test]
    async fn heterogeneous_group_transform_failure_fails_over_to_sibling() {
        use httpmock::MockServer;

        struct StubFailingTransformProvider;
        impl oxllm_core::Provider for StubFailingTransformProvider {
            fn id(&self) -> oxllm_core::ProviderId {
                "stub-transform"
            }
            fn supports(&self, _: oxllm_core::Endpoint) -> bool {
                false
            }
            fn chat(&self) -> Option<&dyn oxllm_core::provider::ChatProvider> {
                Some(self)
            }
        }
        impl oxllm_core::provider::ChatProvider for StubFailingTransformProvider {
            fn endpoint_url(&self, _: &oxllm_core::Deployment) -> String {
                String::new()
            }
            fn apply_auth(
                &self,
                _: &oxllm_core::Deployment,
                _: &oxllm_core::provider::ResolvedCredentials,
                _: &mut std::collections::HashMap<String, String>,
            ) {
            }
            fn transform_request(
                &self,
                _: &Value,
                _: &oxllm_core::Deployment,
            ) -> oxllm_core::Result<Value> {
                Err(oxllm_core::Error::Transform(
                    "cannot map request for this provider".to_string(),
                ))
            }
            fn transform_response(&self, resp: &Value, _: &str) -> oxllm_core::Result<Value> {
                Ok(resp.clone())
            }
            fn transform_stream_chunk(&self, _: &str) -> oxllm_core::Result<Option<Value>> {
                Ok(None)
            }
        }

        let server = MockServer::start();
        let server_mock = server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200).header("content-type", "application/json").body(
                serde_json::json!({
                    "id": "c", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "model": "llm"
                })
                .to_string(),
            );
        });

        let config = ProxyConfig {
            model_list: vec![
                // Stub member: registered provider whose transform fails.
                ConfigModel {
                    model_name: "llm".to_string(),
                    oxllm_params: oxllm_core::deployment::OxllmParams {
                        model: "stub-transform/some-model".to_string(),
                        api_key: Some("k".to_string()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                // Healthy OpenAI sibling.
                ConfigModel {
                    model_name: "llm".to_string(),
                    oxllm_params: oxllm_core::deployment::OxllmParams {
                        model: "openai/llm".to_string(),
                        api_key: Some("k".to_string()),
                        api_base: Some(server.base_url()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    num_retries: Some(0),
                    routing_strategy: Some("round_robin".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let mut registry = oxllm_providers::default_registry();
        registry.register(
            "stub-transform",
            std::sync::Arc::new(StubFailingTransformProvider),
        );
        let state = crate::state::AppState::with_registry(config, registry).expect("state built");

        let mut call_ctx = CallContext::for_test("llm");
        let ctx = RequestContext::new("llm");
        let body = serde_json::json!({
            "model": "llm",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let resp = execute_request_chain(
            &state,
            &mut call_ctx,
            &AuthUser::open(),
            "llm",
            &body,
            &ctx,
            false,
            None,
        )
        .await
        .expect("the healthy sibling must serve after the stub's transform failure");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(server_mock.hits(), 1, "the openai sibling must have served");
    }
}
