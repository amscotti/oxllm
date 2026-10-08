//! Shared unary (non-streaming) retry/failover executor.
//!
//! [`execute_unary`] provides **chat-parity** deployment selection, per-
//! deployment retries (via [`Router::retries_for_status_model`]), allowed-
//! fails cooldowns, model ACL enforcement on every fallback leg, and
//! model-level `fallbacks` / `default_fallbacks` chains. Embeddings and other
//! unary handlers inject only the upstream call.

use std::time::Instant;

use oxllm_core::CallContext;
use oxllm_core::UsageSummary;
use oxllm_core::deployment::Deployment;
use oxllm_core::router::AllowedFailDecision;
use serde_json::Value;

use super::chat::{record_deployment_usage, retry_with_backoff};
use super::outcomes::is_deployment_fault_status;
use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::pipeline::{
    check_deployment_limits, check_model_acl, post_failure, post_success, record_deployment_tpm,
};
use crate::state::AppState;

/// Result of a successful unary execution.
#[derive(Debug)]
pub(crate) struct UnarySuccess {
    /// The transformed response body ready to return to the client.
    pub body: Value,
    /// The cost computed by the post-success hooks (for the cost header).
    pub cost: Option<f64>,
}

/// Execute a unary (non-streaming) LLM request with chat-parity retry/failover.
///
/// 1. Try the primary model with an outer pick loop and inner per-deployment
///    retries (status-aware `num_retries` / `retry_policy`).
/// 2. On retryable exhaustion, walk `fallbacks` then (for unknown models)
///    `default_fallbacks`.
/// 3. Client errors (4xx except 408/409/429) from the PRIMARY model abort
///    immediately — except deployment-fault statuses (401/403/404), which
///    walk the fallback chains like a retryable failure (reference
///    `async_function_with_fallbacks` parity, chat parity). From a fallback
///    leg the chain continues to the next leg and the last error surfaces
///    at exhaustion (chat parity: `try_fallback_chain`).
///
/// Every fallback / default-fallback leg is gated by the caller's model ACL
/// ([`check_model_acl`]), exactly like [`crate::orchestration::chat::try_fallback_chain`]:
/// a key restricted to a subset of models must not be served by a fallback
/// model it is not authorised to access.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_unary<F, Fut>(
    state: &AppState,
    call_ctx: &mut CallContext,
    auth: &AuthUser,
    model: &str,
    request_body: &Value,
    session_id: Option<String>,
    tag: Option<String>,
    make_upstream_call: F,
) -> Result<UnarySuccess, ApiError>
where
    F: Fn(&Deployment, &Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<(Value, Option<UsageSummary>), ApiError>> + Send,
{
    let mut last_err: Option<ApiError>;
    let mut primary_ctx = oxllm_core::RequestContext::new(model);
    primary_ctx.tag = tag;
    primary_ctx.tags = call_ctx.request_tags.clone();
    crate::pipeline::routing::apply_default_tag(state, &mut primary_ctx);
    let tag = primary_ctx.tag;

    match try_unary_model(
        state,
        call_ctx,
        auth,
        model,
        request_body,
        session_id.clone(),
        tag.clone(),
        &make_upstream_call,
    )
    .await
    {
        Ok(ok) => return Ok(ok),
        Err(e) if crate::pipeline::routing::is_routing_error(&e) => {
            post_failure(state, call_ctx, &crate::pipeline::core_error_from_api(&e)).await;
            return Err(e);
        }
        // Fallback parity (`async_function_with_fallbacks`,
        // `router.py:6435`): a DEPLOYMENT-fault 4xx from the
        // primary (401 stale credential, 403 not entitled, 404 decommissioned
        // model) is fallback-eligible — fall through to the chain walk below
        // instead of aborting, so the operator's `fallbacks` config rescues
        // a bad primary (chat parity:
        // `outcomes::is_deployment_fault_status`). Request-content 4xx stay
        // terminal.
        Err(e)
            if is_client_error(&e)
                && !is_deployment_fault_status(e.status.as_u16())
                && !super::outcomes::explicitly_allows_client_fallback(
                    state,
                    e.status.as_u16(),
                ) =>
        {
            post_failure(state, call_ctx, &crate::pipeline::core_error_from_api(&e)).await;
            return Err(e);
        }
        Err(e) => last_err = Some(e),
    }

    // Failure-driven model fallback chain (chat parity). Each leg is ACL-
    // checked before it is attempted: a denied model is skipped, never
    // served, matching `try_fallback_chain`. A 4xx from a FALLBACK leg
    // does not abort the chain — the next fallback is tried and the last
    // error surfaces only when every leg failed (chat semantics:
    // The reference tries the next fallback on any exception from a fallback
    // deployment).
    let primary_exists = !state
        .router
        .deployments_for_model(&state.deployments, model)
        .is_empty();
    let mut fbs = state.router.fallbacks_for(model);
    if !primary_exists {
        fbs.extend(state.router.default_fallbacks_for(model));
    }
    let mut attempted = std::collections::HashSet::from([model.to_string()]);
    let mut fallbacks_tried = 0;
    for fb in fbs {
        if fallbacks_tried >= state.router.config().max_fallbacks {
            break;
        }
        if !attempted.insert(fb.to_string()) {
            continue;
        }
        if !model_allowed(auth, fb) {
            continue;
        }
        if crate::pipeline::check_customer_model(state, call_ctx, fb)
            .await
            .is_err()
        {
            continue;
        }
        fallbacks_tried += 1;
        match try_unary_model(
            state,
            call_ctx,
            auth,
            fb,
            request_body,
            session_id.clone(),
            tag.clone(),
            &make_upstream_call,
        )
        .await
        {
            Ok(ok) => return Ok(ok),
            Err(e) if crate::pipeline::routing::is_routing_error(&e) => {
                post_failure(state, call_ctx, &crate::pipeline::core_error_from_api(&e)).await;
                return Err(e);
            }
            Err(e) if is_client_error(&e) => {
                last_err = Some(e);
                continue;
            }
            Err(e) => last_err = Some(e),
        }
    }

    let model_exists = !state
        .router
        .deployments_for_model(&state.deployments, model)
        .is_empty()
        || !state.router.fallbacks_for(model).is_empty();
    let err = last_err.unwrap_or_else(|| {
        if model_exists {
            ApiError::rate_limited(format!(
                "all deployments for '{}' are rate-limited or unavailable",
                crate::error::truncate_model_for_echo(model)
            ))
        } else {
            ApiError::model_not_found(model)
        }
    });
    post_failure(state, call_ctx, &crate::pipeline::core_error_from_api(&err)).await;
    Err(err)
}

/// Try one model group: outer pick loop over deployments, inner per-deployment
/// retries with status-aware `num_retries` and allowed-fails cooldowns.
#[allow(clippy::too_many_arguments)]
async fn try_unary_model<F, Fut>(
    state: &AppState,
    call_ctx: &mut CallContext,
    _auth: &AuthUser,
    model: &str,
    request_body: &Value,
    session_id: Option<String>,
    tag: Option<String>,
    make_upstream_call: &F,
) -> Result<UnarySuccess, ApiError>
where
    F: Fn(&Deployment, &Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<(Value, Option<UsageSummary>), ApiError>> + Send,
{
    let mut ctx = oxllm_core::RequestContext::new(model);
    ctx.session_id = session_id;
    ctx.tag = tag.or_else(|| call_ctx.request_tags.first().cloned());
    ctx.tags = call_ctx.request_tags.clone();

    let default_retries = state.router.retries_for_status_model(0, model);
    let mut last_err: Option<ApiError> = None;
    // Bound the outer pick loop so a misconfigured group cannot spin forever.
    let max_picks = state
        .deployments
        .len()
        .max(1)
        .saturating_mul((default_retries as usize).saturating_add(1));
    let mut routed_body = request_body.clone();
    if let Some(body) = routed_body.as_object_mut() {
        body.insert("model".into(), Value::String(model.to_string()));
    }
    let request_body = &routed_body;

    for _ in 0..max_picks {
        let Some(dep_idx) = crate::pipeline::routing::try_pick_deployment_index(state, &ctx)?
        else {
            break;
        };
        let deployment = &state.deployments[dep_idx];
        let dep_key = oxllm_core::router::deployment_key(deployment);
        let cooldown_key = state.router.cooldown_key(deployment);

        if let Err(error) =
            crate::pipeline::prepare_deployment_pricing(state, call_ctx, deployment, request_body)
                .await
        {
            ctx.pre_call
                .excluded_deployment_keys
                .push(cooldown_key.clone());
            last_err = Some(ApiError::budget_exceeded(error.to_string()));
            continue;
        }
        state.router.record_in_flight(&cooldown_key, 1);

        if let Err(e) = check_deployment_limits(state, deployment, &dep_key).await {
            state.router.record_in_flight(&cooldown_key, -1);
            tracing::info!(
                deployment = %dep_key,
                "deployment rate-limited; marking cooling and trying next"
            );
            state.router.mark_cooldown_for(&cooldown_key, model);
            ctx.pre_call
                .excluded_deployment_keys
                .push(cooldown_key.clone());
            last_err = Some(e);
            continue;
        }

        let reservation = match crate::orchestration::budget::reserve_router_budgets(
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
                if budget_err.window.is_zero() {
                    state.router.mark_cooldown_for(&cooldown_key, model);
                } else {
                    state
                        .router
                        .cooldown_store()
                        .mark(&cooldown_key, budget_err.window);
                }
                last_err = Some(budget_err.api_error);
                ctx.pre_call
                    .excluded_deployment_keys
                    .push(cooldown_key.clone());
                continue;
            }
        };

        // Inner per-deployment retry loop (chat parity).
        let mut attempt: u32 = 0;
        loop {
            let call_start = Instant::now();
            call_ctx.upstream_start_time = Some(call_start);
            match make_upstream_call(deployment, request_body).await {
                Ok((body, usage)) => {
                    call_ctx.reached_upstream = true;
                    state.router.record_in_flight(&cooldown_key, -1);
                    // Router observability: feed the latency and usage stores
                    // so `lowest_latency` / `latency_based` and `usage_based`
                    // strategies see embeddings/rerank traffic, not just chat
                    // (chat parity: `try_model_with_retries` records both on
                    // success). Only successful attempts are recorded — a fast
                    // 500 must not look healthier than a slow 200.
                    state
                        .router
                        .record_latency(&cooldown_key, call_start.elapsed().as_millis() as u64);
                    record_deployment_usage(state, &dep_key, &cooldown_key).await;
                    if let Some(u) = usage.as_ref() {
                        record_deployment_tpm(state, &dep_key, u.total_tokens).await;
                    }
                    let response_model = body
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or(model)
                        .to_string();
                    let cost = post_success(
                        state,
                        call_ctx,
                        &response_model,
                        usage.as_ref(),
                        Some(&body),
                        None,
                        deployment.provider_slug(),
                    )
                    .await;
                    crate::orchestration::budget::settle_router_spend(
                        state,
                        deployment,
                        &dep_key,
                        cost,
                        &reservation,
                    )
                    .await;
                    // A healthy response resets the deployment's allowed-fails
                    // history (chat parity: `clear_failure_counts` on success).
                    state.router.clear_failure_counts(&cooldown_key);
                    return Ok(UnarySuccess { body, cost });
                }
                Err(err) => {
                    call_ctx.reached_upstream |= err.headers.is_some();
                    if is_client_error(&err) {
                        // Chat parity: when the allowed-fails policy declares
                        // this 4xx a deployment fault (limit crossed — e.g.
                        // a 401 from stale credentials) AND the model group
                        // has a sibling, cool and fail over instead of
                        // failing the caller (the reference retries
                        // AuthenticationError when other deployments are
                        // available). Client-caused 4xx no policy covers
                        // still return immediately.
                        let status = err.status.as_u16();
                        let limit_reached = matches!(
                            state
                                .router
                                .record_failure_and_check(&cooldown_key, status, model),
                            AllowedFailDecision::LimitReached { .. }
                        );
                        let sibling_available = state
                            .router
                            .deployments_for_model(&state.deployments, model)
                            .len()
                            > 1;
                        if (limit_reached || is_deployment_fault_status(status))
                            && sibling_available
                        {
                            state.router.record_in_flight(&cooldown_key, -1);
                            reservation.release(&state.cache).await;
                            tracing::info!(
                                deployment = %dep_key,
                                status = status,
                                "deployment-fault client error; cooling and failing over to a sibling"
                            );
                            state.router.mark_cooldown_for(&cooldown_key, model);
                            ctx.pre_call
                                .excluded_deployment_keys
                                .push(cooldown_key.clone());
                            last_err = Some(err);
                            break;
                        }
                        if limit_reached {
                            state.router.mark_cooldown_for(&cooldown_key, model);
                        }
                        state.router.record_in_flight(&cooldown_key, -1);
                        reservation.release(&state.cache).await;
                        return Err(err);
                    }
                    let status = err.status.as_u16();
                    let max_retries = state.router.retries_for_status_model(status, model);
                    let cool_now =
                        match state
                            .router
                            .record_failure_and_check(&cooldown_key, status, model)
                        {
                            AllowedFailDecision::LimitReached { .. } => true,
                            AllowedFailDecision::Tolerated { .. } => false,
                            AllowedFailDecision::NoPolicy => attempt >= max_retries,
                        };
                    attempt = attempt.saturating_add(1);
                    if cool_now || attempt > max_retries {
                        state.router.record_in_flight(&cooldown_key, -1);
                        reservation.release(&state.cache).await;
                        tracing::info!(
                            deployment = %dep_key,
                            attempt,
                            max_retries,
                            "deployment failed; marking cooling and trying next"
                        );
                        state.router.mark_cooldown_for(&cooldown_key, model);
                        ctx.pre_call
                            .excluded_deployment_keys
                            .push(cooldown_key.clone());
                        last_err = Some(err);
                        break;
                    }
                    // Same-deployment retry with backoff. Shares the chat
                    // backoff (1s base + jitter, 30s cap) so unary traffic
                    // does not hammer a transiently-failing provider 20x
                    // harder than chat would, and honors an upstream
                    // `Retry-After` header exactly like the chat loop.
                    // (`last_err` is left untouched here: it is only read
                    // after the loop ends, and the next attempt — success,
                    // client-error, or exhaustion — sets or discards it
                    // before any read.)
                    let retry_after =
                        crate::orchestration::outcomes::retry_after_secs(err.headers.as_deref());
                    retry_with_backoff(attempt.saturating_sub(1), retry_after).await;
                }
            }
        }
    }

    let model_exists = !state
        .router
        .deployments_for_model(&state.deployments, model)
        .is_empty();
    Err(last_err.unwrap_or_else(|| {
        if model_exists {
            ApiError::rate_limited(format!(
                "all deployments for '{}' are rate-limited or unavailable",
                crate::error::truncate_model_for_echo(model)
            ))
        } else {
            ApiError::model_not_found(model)
        }
    }))
}

/// Whether an error should abort the retry loop (4xx other than 429) or
/// trigger failover to the next deployment.
///
/// 408 (request timeout) and 409 (lock timeout) are excluded alongside
/// 429: the chat path retries them (retry-classifier parity, see
/// `outcomes::is_retryable_status`), so the unary executor must not treat
/// them as terminal client errors — an upstream 408 on embeddings
/// previously aborted with no retry and no failover.
fn is_client_error(err: &ApiError) -> bool {
    let status = err.status.as_u16();
    err.status.is_client_error() && !matches!(status, 408 | 409 | 429)
}

/// Gate a fallback / default-fallback model against the caller's key/team/user
/// ACL. Mirrors [`crate::orchestration::chat::try_fallback_chain`]: a denied
/// model is skipped (returns `false`), never served. Logged server-side; the
/// configured allow-list is never echoed to the client.
fn model_allowed(auth: &AuthUser, model: &str) -> bool {
    let allowed = check_model_acl(
        auth,
        model,
        auth.key_models.as_deref(),
        auth.team_models.as_deref(),
        auth.user_models.as_deref(),
    )
    .is_ok();
    if !allowed {
        tracing::info!(
            fallback_model = %model,
            "skipping unary fallback model not allowed by key/team/user ACL"
        );
    }
    allowed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{
        ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings,
    };
    use oxllm_core::deployment::OxllmParams;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    type UpstreamOutcome = Result<(Value, Option<UsageSummary>), ApiError>;

    fn open_auth() -> AuthUser {
        AuthUser::open()
    }

    fn auth_with_key_models(models: Vec<&str>) -> AuthUser {
        AuthUser {
            key_models: Some(models.iter().map(|m| m.to_string()).collect()),
            ..AuthUser::open()
        }
    }

    fn state_with_model(model: &str) -> AppState {
        state_with_model_and_budget(model, None, None)
    }

    fn state_with_model_and_budget(
        model: &str,
        max_budget: Option<f64>,
        budget_duration: Option<&str>,
    ) -> AppState {
        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: model.to_string(),
                oxllm_params: OxllmParams {
                    model: format!("openai/{model}"),
                    api_base: Some("http://127.0.0.1:1".to_string()),
                    api_key: Some("sk-test".to_string()),
                    max_budget,
                    budget_duration: budget_duration.map(str::to_string),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings {
                routing: crate::config::types::RoutingKnobs {
                    // Default to 0 retries so budget/limit tests do not pay
                    // the (chat-parity, ~1s+) backoff sleep on every failure.
                    num_retries: Some(0),
                    ..Default::default()
                },
                ..Default::default()
            },
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    fn ctx_for(model: &str) -> CallContext {
        let mut ctx = CallContext::for_test(model);
        ctx.request_body = Arc::new(json!({"model": model, "input": "hi"}));
        ctx
    }

    /// Build a state with one deployment per given model, plus optional
    /// `fallbacks` / `default_fallbacks` and a configurable `num_retries`.
    /// All routing knobs live in `router_settings.routing` (the canonical
    /// location) so a single field set is authoritative.
    fn state_with_chain(
        models: &[&str],
        fallbacks: Option<Value>,
        default_fallbacks: Option<Value>,
        num_retries: u32,
    ) -> AppState {
        use crate::config::types::RoutingKnobs;
        let config = ProxyConfig {
            model_list: models
                .iter()
                .map(|m| ConfigModel {
                    model_name: m.to_string(),
                    oxllm_params: OxllmParams {
                        model: format!("openai/{m}"),
                        api_base: Some("http://127.0.0.1:1".to_string()),
                        api_key: Some("sk-test".to_string()),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .collect(),
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    num_retries: Some(num_retries),
                    fallbacks,
                    default_fallbacks,
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    fn mock_closure(
        calls: Arc<AtomicU32>,
        result: ApiError,
    ) -> impl Fn(&Deployment, &Value) -> std::future::Ready<UpstreamOutcome> {
        move |_deployment, _body| {
            calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Err(result.clone()))
        }
    }

    fn ok_closure(
        calls: Arc<AtomicU32>,
    ) -> impl Fn(&Deployment, &Value) -> std::future::Ready<UpstreamOutcome> {
        move |_deployment, _body| {
            calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok((
                json!({"data": [], "model": "test-model"}),
                Some(UsageSummary {
                    prompt_tokens: 10,
                    completion_tokens: 0,
                    total_tokens: 10,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    audio_input_tokens: None,
                    reasoning_tokens: None,
                    citation_tokens: None,
                    web_search_requests: None,
                    cache_creation_1h_input_tokens: None,
                }),
            )))
        }
    }

    /// Closure that FAILS (500) for the primary model and SUCCEEDS for the
    /// fallback model. Used to prove the fallback model is reached (or, under
    /// an ACL restriction, skipped). `fb_calls` counts how many times the
    /// fallback model's deployment was hit.
    fn selective_closure(
        primary: &str,
        fallback: &str,
        fb_calls: Arc<AtomicU32>,
    ) -> impl Fn(&Deployment, &Value) -> std::future::Ready<UpstreamOutcome> {
        let primary = primary.to_string();
        let fallback = fallback.to_string();
        move |deployment, _body| {
            if deployment.model_name == fallback {
                fb_calls.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Ok((
                    json!({"data": [], "model": fallback.clone()}),
                    Some(UsageSummary {
                        prompt_tokens: 1,
                        completion_tokens: 0,
                        total_tokens: 1,
                        cache_read_input_tokens: None,
                        cache_creation_input_tokens: None,
                        audio_input_tokens: None,
                        reasoning_tokens: None,
                        citation_tokens: None,
                        web_search_requests: None,
                        cache_creation_1h_input_tokens: None,
                    }),
                )))
            } else {
                let _ = &primary;
                std::future::ready(Err(ApiError::upstream(500, "primary down")))
            }
        }
    }

    #[test]
    fn is_client_error_true_for_4xx_except_429() {
        assert!(is_client_error(&ApiError::bad_request("bad")));
        assert!(is_client_error(&ApiError::unauthorized()));
        assert!(is_client_error(&ApiError::not_found("x")));
    }

    #[test]
    fn is_client_error_false_for_429_and_5xx() {
        assert!(!is_client_error(&ApiError::rate_limited("slow")));
        assert!(!is_client_error(&ApiError::budget_exceeded("over")));
        assert!(!is_client_error(&ApiError::upstream(500, "err")));
        assert!(!is_client_error(&ApiError::upstream(503, "err")));
        assert!(!is_client_error(&ApiError::internal("boom")));
    }

    /// LOW (C11): 408/409 are retryable on the chat path (reference
    /// `_should_retry`); the unary executor must not treat them as
    /// terminal client errors.
    #[test]
    fn is_client_error_false_for_408_and_409() {
        assert!(
            !is_client_error(&ApiError::upstream(408, "request timeout")),
            "408 must be retried (chat parity)"
        );
        assert!(
            !is_client_error(&ApiError::upstream(409, "lock timeout")),
            "409 must be retried (chat parity)"
        );
    }

    /// LOW (C11): with `num_retries: 1` an upstream 408 must be re-attempted
    /// (two closure calls), mirroring the chat retry loop.
    #[tokio::test]
    async fn upstream_408_is_retried() {
        let state = state_with_chain(&["test-model"], None, None, 1);
        let mut ctx = ctx_for("test-model");
        let calls = Arc::new(AtomicU32::new(0));
        let _err = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "test-model",
            &json!({"input": "hi"}),
            None,
            None,
            mock_closure(calls.clone(), ApiError::upstream(408, "request timeout")),
        )
        .await
        .expect_err("408 exhausts retries and fails");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "408 must be retried once under num_retries=1 (initial + 1 retry)"
        );
    }

    /// LOW (C11): a 4xx from a FALLBACK leg must not abort the chain — the
    /// next fallback serves (chat parity: the reference tries the next fallback
    /// on any exception from a fallback deployment). Previously the unary
    /// executor returned the fallback's 4xx immediately.
    #[tokio::test]
    async fn fallback_chain_continues_past_4xx_from_fallback_leg() {
        // primary 500 → fb1 400 → fb2 healthy: fb2 must serve.
        let state = state_with_chain(
            &["primary", "fb1", "fb2"],
            Some(json!([{"primary": ["fb1", "fb2"]}])),
            None,
            0,
        );
        let fb1_calls = Arc::new(AtomicU32::new(0));
        let fb2_calls = Arc::new(AtomicU32::new(0));
        let f1 = fb1_calls.clone();
        let f2 = fb2_calls.clone();
        let closure =
            move |deployment: &Deployment, _body: &Value| match deployment.model_name.as_str() {
                "fb1" => {
                    f1.fetch_add(1, Ordering::SeqCst);
                    std::future::ready(Err(ApiError::upstream(400, "bad from fb1")))
                }
                "fb2" => {
                    f2.fetch_add(1, Ordering::SeqCst);
                    std::future::ready(Ok((json!({"data": [], "model": "fb2"}), None)))
                }
                _ => std::future::ready(Err(ApiError::upstream(500, "primary down"))),
            };
        let mut ctx = ctx_for("primary");
        let result = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "primary",
            &json!({"input": "hi"}),
            None,
            None,
            closure,
        )
        .await
        .expect("second fallback must serve after fb1's 4xx");
        assert_eq!(result.body["model"], "fb2");
        assert_eq!(fb1_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fb2_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn success_returns_body_and_runs_post_success() {
        let state = state_with_model("test-model");
        let mut ctx = ctx_for("test-model");
        let calls = Arc::new(AtomicU32::new(0));
        let result = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "test-model",
            &json!({"input": "hi"}),
            None,
            None,
            ok_closure(calls.clone()),
        )
        .await;
        assert!(result.is_ok());
        let success = result.unwrap();
        assert_eq!(success.body["model"], "test-model");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one upstream call");
    }

    #[tokio::test]
    async fn client_error_aborts_immediately() {
        let state = state_with_model("test-model");
        let mut ctx = ctx_for("test-model");
        let calls = Arc::new(AtomicU32::new(0));
        let err = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "test-model",
            &json!({"input": "hi"}),
            None,
            None,
            mock_closure(calls.clone(), ApiError::bad_request("bad input")),
        )
        .await
        .expect_err("4xx must abort");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "must NOT retry on 4xx");
    }

    /// MEDIUM (C11): a DEPLOYMENT-fault 4xx from the primary (401 — stale
    /// vendor credential) is fallback-eligible (reference
    /// `async_function_with_fallbacks` walks the chain on any primary
    /// exception; chat parity): the configured fallback must serve.
    #[tokio::test]
    async fn primary_401_walks_fallback_chain() {
        let state = state_with_chain(
            &["primary", "rescue"],
            Some(json!([{"primary": ["rescue"]}])),
            None,
            0,
        );
        let rescue_calls = Arc::new(AtomicU32::new(0));
        let r = rescue_calls.clone();
        let closure = move |deployment: &Deployment, _body: &Value| {
            if deployment.model_name == "rescue" {
                r.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Ok((json!({"data": [], "model": "rescue"}), None)))
            } else {
                std::future::ready(Err(ApiError::upstream(401, "invalid api key")))
            }
        };
        let mut ctx = ctx_for("primary");
        let result = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "primary",
            &json!({"input": "hi"}),
            None,
            None,
            closure,
        )
        .await
        .expect("the 401 primary must be rescued by the fallback chain");
        assert_eq!(result.body["model"], "rescue");
        assert_eq!(rescue_calls.load(Ordering::SeqCst), 1);
    }

    /// Companion guard: a request-content 400 from the primary stays
    /// terminal — no fallback fan-out (deliberate divergence from the reference;
    /// see `outcomes::is_deployment_fault_status`).
    #[tokio::test]
    async fn primary_400_does_not_walk_fallback_chain() {
        let state = state_with_chain(
            &["primary", "rescue"],
            Some(json!([{"primary": ["rescue"]}])),
            None,
            0,
        );
        let rescue_calls = Arc::new(AtomicU32::new(0));
        let r = rescue_calls.clone();
        let closure = move |deployment: &Deployment, _body: &Value| {
            if deployment.model_name == "rescue" {
                r.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Ok((json!({"data": [], "model": "rescue"}), None)))
            } else {
                std::future::ready(Err(ApiError::upstream(400, "bad input")))
            }
        };
        let mut ctx = ctx_for("primary");
        let err = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "primary",
            &json!({"input": "hi"}),
            None,
            None,
            closure,
        )
        .await
        .expect_err("a client-caused 400 must surface");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            rescue_calls.load(Ordering::SeqCst),
            0,
            "the fallback must not be consulted for a client-caused 4xx"
        );
    }

    #[tokio::test]
    async fn retryable_error_retries_same_deployment() {
        // num_retries: 1 → initial + 1 retry = 2 attempts on the one dep
        // before cooling. Uses num_retries=1 (not higher) to bound the
        // chat-parity backoff to a single ~1-2s sleep.
        let state = state_with_chain(&["test-model"], None, None, 1);
        let mut ctx = ctx_for("test-model");
        let calls = Arc::new(AtomicU32::new(0));
        let _err = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "test-model",
            &json!({"input": "hi"}),
            None,
            None,
            mock_closure(calls.clone(), ApiError::upstream(500, "server error")),
        )
        .await
        .expect_err("5xx exhausts retries");
        let n = calls.load(Ordering::SeqCst);
        assert!(
            n >= 2,
            "chat-parity retries must re-attempt the same deployment at least twice, got {n}"
        );
    }

    const DEP_KEY: &str = "budget-model:openai/budget-model";

    fn budget_state() -> AppState {
        state_with_model_and_budget("budget-model", Some(10.0), Some("1h"))
    }

    #[tokio::test]
    async fn budget_rejection_releases_no_spend() {
        let state = budget_state();
        let _ = state
            .cache
            .incr(&format!("spend:dep:{DEP_KEY}"), 10.0, None)
            .await;
        let mut ctx = ctx_for("budget-model");
        let calls = Arc::new(AtomicU32::new(0));
        let _err = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "budget-model",
            &json!({"input": "hi"}),
            None,
            None,
            ok_closure(calls.clone()),
        )
        .await
        .expect_err("at-cap budget must reject");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "closure must not be called when budget rejects"
        );
        let spend = state
            .cache
            .counter_value(&format!("spend:dep:{DEP_KEY}"))
            .await
            .unwrap_or(0.0);
        assert!(
            (spend - 10.0).abs() < 1e-9,
            "rejected reservation must not add spend, got {spend}"
        );
    }

    #[tokio::test]
    async fn budget_released_on_upstream_failure() {
        let state = state_with_model_and_budget("budget-model", Some(10.0), None);
        let mut ctx = ctx_for("budget-model");
        ctx.budget_estimate = Some(0.5);
        let calls = Arc::new(AtomicU32::new(0));
        let _err = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "budget-model",
            &json!({"input": "hi"}),
            None,
            None,
            mock_closure(calls.clone(), ApiError::upstream(500, "boom")),
        )
        .await;
        assert!(calls.load(Ordering::SeqCst) >= 1);
        let spend = state
            .cache
            .counter_value(&format!("spend:dep:{DEP_KEY}"))
            .await
            .unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "failed attempt must release the reservation, got {spend}"
        );
    }

    #[test]
    fn unary_success_holds_body_and_cost() {
        let s = UnarySuccess {
            body: json!({"ok": true}),
            cost: Some(0.001),
        };
        assert_eq!(s.body["ok"], true);
        assert_eq!(s.cost, Some(0.001));
    }

    // ---------------------------------------------------------------
    // HIGH (C11): model ACL enforcement on fallback / default-fallback
    // legs. A key restricted to a subset of models must not be served by
    // a fallback model it is not authorised to access.
    // ---------------------------------------------------------------

    /// `fallbacks: [{"model-a": ["model-b"]}]`, key restricted to
    /// `["model-a"]`. model-a's deployment 500s; model-b's deployment would
    /// succeed. With the ACL fix the fallback model-b is skipped (denied),
    /// the request fails, and model-b is NEVER called.
    #[tokio::test]
    async fn fallback_skips_acl_denied_model() {
        let state = state_with_chain(
            &["model-a", "model-b"],
            Some(json!([{"model-a": ["model-b"]}])),
            None,
            0,
        );
        let mut ctx = ctx_for("model-a");
        let auth = auth_with_key_models(vec!["model-a"]);
        let fb_calls = Arc::new(AtomicU32::new(0));
        let err = execute_unary(
            &state,
            &mut ctx,
            &auth,
            "model-a",
            &json!({"input": "hi"}),
            None,
            None,
            selective_closure("model-a", "model-b", fb_calls.clone()),
        )
        .await
        .expect_err("ACL-denied fallback must not serve model-b");
        assert_eq!(
            fb_calls.load(Ordering::SeqCst),
            0,
            "model-b must never be called when ACL denies it"
        );
        assert!(
            err.status == axum::http::StatusCode::TOO_MANY_REQUESTS || err.status.is_server_error(),
            "exhausted primary with no allowed fallback surfaces a retryable error"
        );
    }

    /// Same setup as above but an OPEN key (no model restriction): the
    /// fallback model-b IS authorised and serves the request. This guards
    /// against the ACL check accidentally denying allowed models.
    #[tokio::test]
    async fn fallback_serves_allowed_model_for_open_key() {
        let state = state_with_chain(
            &["model-a", "model-b"],
            Some(json!([{"model-a": ["model-b"]}])),
            None,
            0,
        );
        let mut ctx = ctx_for("model-a");
        let fb_calls = Arc::new(AtomicU32::new(0));
        let success = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "model-a",
            &json!({"input": "hi"}),
            None,
            None,
            selective_closure("model-a", "model-b", fb_calls.clone()),
        )
        .await
        .expect("open key is authorised for model-b fallback");
        assert_eq!(success.body["model"], "model-b");
        assert_eq!(
            fb_calls.load(Ordering::SeqCst),
            1,
            "model-b fallback must be called once for an open key"
        );
    }

    /// `default_fallbacks: ["model-b"]`, requested model is UNKNOWN (no
    /// deployments), key restricted to `["model-a"]`. model-b would succeed.
    /// With the ACL fix model-b is skipped (denied) and the request fails
    /// with model-not-found, never calling model-b.
    #[tokio::test]
    async fn default_fallback_skips_acl_denied_model() {
        let state = state_with_chain(&["model-b"], None, Some(json!(["model-b"])), 0);
        let mut ctx = ctx_for("unknown-model");
        let auth = auth_with_key_models(vec!["model-a"]);
        let fb_calls = Arc::new(AtomicU32::new(0));
        let err = execute_unary(
            &state,
            &mut ctx,
            &auth,
            "unknown-model",
            &json!({"input": "hi"}),
            None,
            None,
            selective_closure("unknown-model", "model-b", fb_calls.clone()),
        )
        .await
        .expect_err("ACL-denied default_fallback must not serve model-b");
        assert_eq!(
            fb_calls.load(Ordering::SeqCst),
            0,
            "model-b must never be called when ACL denies it"
        );
        assert_eq!(err.status, axum::http::StatusCode::NOT_FOUND);
    }

    /// Guard (chat parity: `default_fallback_known_model_is_not_rerouted`):
    /// a KNOWN model group whose single deployment is fully cooled must NOT
    /// consult `default_fallbacks` — failure of a known model is what
    /// `fallbacks` handles. The cooled deployment is cooled via the router's
    /// cooldown store with a long window; the request must fail with the
    /// all-deployments-unavailable 429 and never call the default-fallback
    /// model.
    #[tokio::test]
    async fn default_fallback_known_cooled_model_is_not_rerouted() {
        let state = state_with_chain(
            &["known-model", "df-model"],
            None,
            Some(json!(["df-model"])),
            0,
        );
        let deployment = state.deployments[0].clone();
        let cooldown_key = state.router.cooldown_key(&deployment);
        state
            .router
            .cooldown_store()
            .mark(&cooldown_key, std::time::Duration::from_secs(300));

        let df_calls = Arc::new(AtomicU32::new(0));
        let d = df_calls.clone();
        let closure = move |deployment: &Deployment, _body: &Value| {
            if deployment.model_name == "df-model" {
                d.fetch_add(1, Ordering::SeqCst);
            }
            std::future::ready(Err(ApiError::upstream(500, "unreachable")))
        };
        let mut ctx = ctx_for("known-model");
        let err = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "known-model",
            &json!({"input": "hi"}),
            None,
            None,
            closure,
        )
        .await
        .expect_err("cooled known group must fail, not reroute");
        assert_eq!(
            err.status,
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            "cooled-out known group reports all-deployments-unavailable"
        );
        assert_eq!(
            df_calls.load(Ordering::SeqCst),
            0,
            "default_fallbacks must not be consulted for a known-but-cooled group"
        );
    }

    // ---------------------------------------------------------------
    // MEDIUM (C11): clear_failure_counts on success (chat parity).
    // ---------------------------------------------------------------

    /// A successful unary call must reset the deployment's allowed-fails
    /// failure history (chat parity: `clear_failure_counts`). We record two
    /// failures against the cooldown key, run a successful unary call, then
    /// assert the router's failure count for that key is gone.
    #[tokio::test]
    async fn success_clears_failure_counts() {
        let state = state_with_model("test-model");
        let deployment = state.deployments[0].clone();
        let cooldown_key = state.router.cooldown_key(&deployment);
        // Seed two failures (status 500) directly via the router so the
        // allowed-fails bookkeeping has something that success must clear.
        let _ = state
            .router
            .record_failure_and_check(&cooldown_key, 500, "test-model");
        let _ = state
            .router
            .record_failure_and_check(&cooldown_key, 500, "test-model");

        let mut ctx = ctx_for("test-model");
        let calls = Arc::new(AtomicU32::new(0));
        let result = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "test-model",
            &json!({"input": "hi"}),
            None,
            None,
            ok_closure(calls.clone()),
        )
        .await;
        assert!(result.is_ok());

        // After a successful call the failure count for the cooldown key must
        // be reset: a single subsequent failure should NOT reach a limit of 3
        // (it would if the prior count of 2 had persisted).
        let decision = state
            .router
            .record_failure_and_check(&cooldown_key, 500, "test-model");
        assert!(
            !matches!(decision, AllowedFailDecision::LimitReached { .. }),
            "failure history must be cleared on success; a single failure must not trip the limit"
        );
    }

    // ---------------------------------------------------------------
    // MEDIUM (C11): router observability — latency store fed on success.
    // ---------------------------------------------------------------

    /// A successful unary call must record the deployment's latency so the
    /// `lowest_latency` / `latency_based` strategies see embeddings/rerank
    /// traffic (chat parity).
    #[tokio::test]
    async fn success_records_latency() {
        let state = state_with_model("test-model");
        let deployment = state.deployments[0].clone();
        let cooldown_key = state.router.cooldown_key(&deployment);

        let mut ctx = ctx_for("test-model");
        let calls = Arc::new(AtomicU32::new(0));
        let result = execute_unary(
            &state,
            &mut ctx,
            &open_auth(),
            "test-model",
            &json!({"input": "hi"}),
            None,
            None,
            ok_closure(calls.clone()),
        )
        .await;
        assert!(result.is_ok());

        let p50 = state.router.latency_store().p50(&cooldown_key);
        assert!(
            p50.is_some(),
            "latency store must have a recorded value for the serving deployment after a successful unary call"
        );
    }

    // ---------------------------------------------------------------
    // MEDIUM (C11): unary backoff parity with chat. The shared
    // `retry_with_backoff` must sleep at least the chat base delay (1s),
    // not the legacy 50ms unary base.
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn unary_backoff_matches_chat_magnitude() {
        let start = Instant::now();
        retry_with_backoff(0, None).await;
        let elapsed_ms = start.elapsed().as_millis();
        // attempt=0 → base 1000ms + jitter [0,1000)ms, so at least 1000ms.
        // The legacy unary backoff was 50ms; this guards against a revert.
        assert!(
            elapsed_ms >= 1000,
            "unary backoff must match chat magnitude (>= 1000ms for attempt 0), got {elapsed_ms}ms"
        );
    }

    #[allow(dead_code)]
    const _: Duration = Duration::from_secs(0);
}
