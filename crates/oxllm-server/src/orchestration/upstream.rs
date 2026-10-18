//! Upstream HTTP call for the chat orchestrator.
//!
//! A single function, [`attempt_upstream`], owns the entire outbound request:
//! credential resolution, request transform (with `drop_params` semantics),
//! header application, timeout, send, and error classification into the
//! [`UpstreamError`] taxonomy used by the retry loop.

use std::collections::HashMap;
use std::sync::Arc;

use oxllm_core::Provider;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::ChatProvider;
use oxllm_core::provider::ResolvedCredentials;
use serde_json::Value;
use tokio_stream::StreamExt;

use super::DEFAULT_TIMEOUT_SECS;
use super::outcomes::{
    UpstreamError, is_content_policy_error, is_context_window_error, is_retryable_status,
};
use crate::bedrock_sigv4::sign_request as sign_bedrock_request;
use crate::error::ApiError;
use crate::pre_call_checks::prompt_caching::{PIN_TTL, pin_ttl_for_native_body};
use crate::security::MAX_ERROR_BODY_BYTES;
use crate::state::AppState;

tokio::task_local! {
    /// Client beta flags scoped around a chat request's awaited orchestration.
    /// Keeping transport metadata out of the JSON body prevents it from being
    /// forwarded as an unknown model parameter or retained by body hooks.
    pub(crate) static CLIENT_ANTHROPIC_BETAS: Vec<String>;
    pub(crate) static CLIENT_TRACE_HEADERS: axum::http::HeaderMap;
}

pub(crate) fn client_trace_headers(headers: &axum::http::HeaderMap) -> axum::http::HeaderMap {
    let mut trace = axum::http::HeaderMap::new();
    if let Some(value) = headers.get("traceparent") {
        trace.insert("traceparent", value.clone());
    }
    let baggage = headers
        .get_all("baggage")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect::<Vec<_>>()
        .join(",");
    if !baggage.is_empty()
        && let Ok(value) = axum::http::HeaderValue::from_str(&baggage)
    {
        trace.insert("baggage", value);
    }
    trace
}

pub(crate) fn client_anthropic_betas(headers: &axum::http::HeaderMap) -> Vec<String> {
    let mut betas = Vec::new();
    for value in headers
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|v| v.to_str().ok())
    {
        for beta in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if !betas.iter().any(|existing| existing == beta) {
                betas.push(beta.to_string());
            }
        }
    }
    betas
}

/// Send the upstream request and classify the result.
pub(crate) async fn attempt_upstream(
    state: &AppState,
    deployment: &Deployment,
    request_body: &Value,
    is_stream: bool,
    provider: &Arc<dyn Provider>,
    traceparent: Option<&str>,
) -> Result<reqwest::Response, UpstreamError> {
    attempt_upstream_with_context(
        state,
        deployment,
        request_body,
        is_stream,
        provider,
        traceparent,
        None,
    )
    .await
}

pub(crate) async fn attempt_upstream_with_context(
    state: &AppState,
    deployment: &Deployment,
    request_body: &Value,
    is_stream: bool,
    provider: &Arc<dyn Provider>,
    traceparent: Option<&str>,
    mut call_ctx: Option<&mut oxllm_core::CallContext>,
) -> Result<reqwest::Response, UpstreamError> {
    if let Some(ctx) = call_ctx.as_deref_mut() {
        ctx.upstream_prompt_cache_pin_ttl = None;
    }
    super::request_controls::validate(request_body).map_err(|error| UpstreamError::Client {
        status: error.status.as_u16(),
        message: error.message,
        headers: None,
    })?;
    let effective_deployment = deployment_with_transform_settings(deployment, &state.config);
    let deployment = effective_deployment.as_ref();
    let chat = provider.chat().ok_or_else(|| UpstreamError::Client {
        status: 400,
        message: format!(
            "provider '{}' does not implement chat completions",
            deployment.provider_slug()
        ),
        headers: None,
    })?;

    let document_body = normalize_document_parts(request_body, deployment).map_err(|err| {
        UpstreamError::Client {
            status: 400,
            message: err.message,
            headers: None,
        }
    })?;
    let request_body = document_body.as_ref().unwrap_or(request_body);

    let creds = state
        .credential_resolver
        .resolve_for(deployment)
        .resolve(deployment)
        .await
        // Credential 401/403 failures retain their status and can cool/fail
        // over to authorized siblings or configured fallback groups. Other
        // credential failures use the retry policy before deployment rotation.
        .map_err(|error| {
            let error = ApiError::from(error);
            if error.status.is_client_error() {
                UpstreamError::Client {
                    status: error.status.as_u16(),
                    message: error.message,
                    headers: None,
                }
            } else {
                UpstreamError::Retryable {
                    status: error.status.as_u16(),
                    message: error.message,
                    headers: None,
                }
            }
        })?;

    // A request-transform failure is provider-specific, not request-
    // specific: in a mixed model group (e.g. one OpenAI + one Gemini
    // deployment under the same `model_name`), a body shape this provider's
    // transform rejects may map fine on a sibling. Classifying it `Client`
    // would abort the whole chain and let one untransformable member fail
    // every request routed to it; `Retryable` (same treatment as a
    // credential failure) cools this deployment and fails over. An explicit
    // `BadRequest` from a provider adapter (unsupported parameter shape) is
    // the exception: it is a deterministic client error, so it keeps its 400
    // status instead of degrading into a retryable 500.
    let transformed =
        apply_drop_params(state, chat, request_body, deployment).map_err(transform_failure)?;

    let mut transformed = match transformed {
        Some(v) => v,
        // No stripping was needed; transform the original body directly. M31:
        // when `drop_params=false` and there's nothing to warn-strip, we
        // return `None` here so the caller passes the original `request`
        // through to `transform_request` without an extra deep `Value::clone`.
        None => chat
            .transform_request(request_body, deployment)
            .map_err(transform_failure)?,
    };
    super::request_controls::strip(&mut transformed);
    let body_model = transformed
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let auth_deployment = if provider.id() == "azure_ai"
        && deployment.oxllm_params.model.contains('*')
        && let Some(model) = body_model.as_deref()
    {
        let mut resolved = deployment.clone();
        resolved.oxllm_params.model = format!("azure_ai/{model}");
        std::borrow::Cow::Owned(resolved)
    } else {
        std::borrow::Cow::Borrowed(deployment)
    };
    let azure_anthropic = provider.id() == "azure_ai"
        && oxllm_providers::azure_ai::AzureAiProvider::is_anthropic(&auth_deployment);
    if !azure_anthropic {
        inject_stream_usage(&mut transformed, is_stream, deployment.provider_slug());
    }
    let endpoint_deployment = deployment_with_resolved_region(deployment, &creds);
    let url = chat.endpoint_url_for_request(&endpoint_deployment, request_body);
    let url = build_upstream_url(
        url,
        body_model.as_deref(),
        is_stream,
        deployment.provider_slug(),
        &transformed,
    );

    // C2/H1: SSRF / DNS-rebinding defense on the request hot path. The
    // config-load-time `validate_api_base` check runs once at startup; DNS
    // may change between then and this actual HTTP request, so we re-check
    // the resolved host immediately before going on the wire. The cached
    // best-effort variant keeps the per-request cost at one `getaddrinfo`
    // per 10s per host and deliberately allows unresolvable hostnames
    // through (the upstream call will fail on its own). Operators that
    // legitimately need to hit private ranges (local Ollama, etc.) opt out
    // via `general_settings.allow_private_api_base`.
    //
    // H1: the DNS lookup is offloaded via `tokio::task::spawn_blocking` so a
    // slow resolver cannot stall the async executor thread (and every other
    // request parked on it). `is_private_host_async` checks the cache first
    // and only spawns a blocking task on a cache miss.
    if !state
        .config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false)
    {
        if let Some(host) = url::Url::parse(&url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h.to_string()))
        {
            if crate::security::is_private_host_async(&host).await {
                tracing::warn!(
                    upstream_host = %host,
                    "rejecting upstream api_base: host resolves to a private IP address"
                );
                // Deployment-level fault, not a request-level one: a mixed
                // group (one private-base deployment + healthy siblings)
                // must not fail hard for every request routed to the bad
                // member. `Retryable` cools this deployment and fails over;
                // on terminal exhaustion the client still sees a 502 (same
                // status the hard abort produced).
                return Err(UpstreamError::Retryable {
                    status: 502,
                    message: "upstream host resolves to a private IP address".to_string(),
                    headers: None,
                });
            }
        }
    }

    let mut headers = HashMap::new();
    chat.apply_auth(&auth_deployment, &creds, &mut headers);
    oxllm_core::provider::apply_auth_overrides(deployment, &creds, &mut headers);
    merge_extra_headers(&mut headers, deployment);
    if let Some(traceparent) = traceparent {
        headers.insert("traceparent".into(), traceparent.to_string());
    }
    if let Ok(client_headers) = CLIENT_TRACE_HEADERS.try_with(Clone::clone) {
        for (name, value) in &client_headers {
            if let Ok(value) = value.to_str() {
                headers.insert(name.to_string(), value.to_string());
            }
        }
    }
    let bedrock_converse = provider.id() == "bedrock"
        && url::Url::parse(&url).is_ok_and(|url| {
            matches!(
                url.path().rsplit('/').next(),
                Some("converse" | "converse-stream")
            )
        });
    if bedrock_converse {
        oxllm_providers::bedrock::converse::filter_converse_beta_headers(&mut headers);
    }

    // Anthropic gates several features behind comma-separated `anthropic-beta`
    // header tags; the provider's request transform emits the body fields for
    // those features (speed, web_search, structured outputs, context
    // management, advanced tool use, advisor, pre-4.6 reasoning_effort) but
    // `apply_auth` cannot see the request body to know which tags apply. Compute
    // them from the OpenAI-shape request and merge with any deployment-supplied
    // `anthropic-beta` so the upstream accepts the gated fields instead of 400.
    if provider.id() == "anthropic" || azure_anthropic {
        // For wildcard deployments (`anthropic/*`) the deployment model is the
        // literal glob, not the request's actual model. `collect_beta_headers`
        // makes per-model decisions (e.g. `effort` beta only for pre-4.6), so
        // derive the model from the transformed body — which the provider's
        // `transform_request` has already resolved to a concrete id — and fall
        // back to the deployment model only when the body is missing one.
        let beta_model = body_model
            .as_deref()
            .unwrap_or(&deployment.oxllm_params.model);
        let betas = oxllm_providers::anthropic::collect_beta_headers(request_body, beta_model);
        merge_anthropic_beta_headers(&mut headers, betas);
        if let Ok(client_betas) = CLIENT_ANTHROPIC_BETAS.try_with(Clone::clone) {
            merge_anthropic_beta_headers(&mut headers, client_betas);
        }
    }
    if provider.id() == "vertex_ai" && transformed.get("anthropic_version").is_some() {
        let mut betas = transformed
            .get("anthropic_beta")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut supplied = headers
            .remove("anthropic-beta")
            .into_iter()
            .flat_map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if let Ok(client_betas) = CLIENT_ANTHROPIC_BETAS.try_with(Clone::clone) {
            supplied.extend(client_betas);
        }
        for beta in supplied {
            if !betas.contains(&beta) {
                betas.push(beta);
            }
        }
        if !betas.is_empty() {
            transformed["anthropic_beta"] = serde_json::json!(betas);
        }
    }

    // The reference `_get_timeout` (`router.py`): a STREAMING request
    // resolves the idle budget via `_get_stream_timeout`
    // (`stream_timeout ?? request_timeout`) and httpx applies it per read
    // operation — a stream that keeps producing chunks is never killed by a
    // total deadline. reqwest's per-request `.timeout()` is a TOTAL deadline
    // (connect until the last body byte), so applying it to a stream
    // hard-kills any stream whose total duration exceeds `timeout` (with the
    // 600s default this truncates long agentic/reasoning streams, and an
    // operator setting `request_timeout: 60` for fast unary failure would
    // silently kill every stream longer than a minute even with
    // `stream_timeout: 600` configured). For streams we therefore apply NO
    // total deadline: the header wait is bounded here by the same idle
    // budget the pump enforces per-chunk (`resolve_stream_idle_timeout`).
    let stream_idle_budget =
        resolve_stream_idle_timeout_for_request(deployment, &state.config, request_body);
    let timeout_secs = if is_stream {
        None
    } else {
        Some(
            super::request_controls::seconds(request_body, "timeout")
                .or(deployment.oxllm_params.timeout)
                .unwrap_or_else(|| {
                    state
                        .config
                        .oxllm_settings
                        .request_timeout
                        .map(|t| t as f64)
                        .unwrap_or(DEFAULT_TIMEOUT_SECS as f64)
                }),
        )
    };

    let body_bytes = serde_json::to_vec(&transformed).map_err(|e| UpstreamError::Client {
        status: 500,
        message: format!("failed to serialize request body: {e}"),
        headers: None,
    })?;

    // AWS SigV4 signing must run after the body is serialized (it signs the
    // body hash) but before the request goes on the wire. Bedrock's provider
    // transform leaves `apply_auth` empty precisely so the host can sign the
    // fully-materialized request here. Signing operates on the header vec so
    // it can add the `host`, `x-amz-date`, `x-amz-content-sha256`,
    // `x-amz-security-token`, and `Authorization` entries in one pass.
    if let ResolvedCredentials::AwsSigV4 {
        access_key,
        secret_key,
        session_token,
        region,
    } = &creds
    {
        let mut header_pairs: Vec<(String, String)> = headers
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
            .collect();
        sign_bedrock_request(
            "POST",
            &url,
            &mut header_pairs,
            &body_bytes,
            access_key,
            secret_key,
            region,
            "bedrock",
            session_token.as_deref(),
        )
        .map_err(|e| {
            // The sign error embeds the full target URL; it must never reach
            // the client body. Log it server-side and forward a generic
            // message (same redaction contract as `Error::Network`).
            tracing::error!(
                error.kind = "config",
                detail = %e,
                "SigV4 request signing failed"
            );
            UpstreamError::Client {
                status: 500,
                message: ApiError::internal("upstream request signing failed").message,
                headers: None,
            }
        })?;
        headers = header_pairs.into_iter().collect();
    }

    let mut header_map = reqwest::header::HeaderMap::new();
    if let Ok(ct) = reqwest::header::HeaderValue::from_str("application/json") {
        header_map.insert(reqwest::header::CONTENT_TYPE, ct);
    }
    for (key, value) in &headers {
        let invalid_header = || UpstreamError::Retryable {
            status: 500,
            message: "invalid configured upstream header".to_string(),
            headers: None,
        };
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|_| invalid_header())?;
        let value = reqwest::header::HeaderValue::from_str(value).map_err(|_| invalid_header())?;
        header_map.insert(name, value);
    }

    if let Some(ctx) = call_ctx.as_deref_mut() {
        // Observe after provider transformation, configured drops/extra_body
        // and all final shaping. Prices and raw request hints do not prove
        // that a cache breakpoint was actually sent on this attempt. Select
        // the envelope from adapter-owned routing, never editable version
        // fields in this body (including a forged anthropic_version: null).
        let prompt_cache_wire = chat.prompt_cache_wire(request_body, deployment);
        ctx.upstream_prompt_cache_pin_ttl = Some(
            prompt_cache_wire.map_or(PIN_TTL, |wire| pin_ttl_for_native_body(&transformed, wire)),
        );
        ctx.upstream_start_time = Some(std::time::Instant::now());
        if let Some(model) = body_model.as_deref() {
            let prefix = format!("{}/", deployment.provider_slug());
            let served_model = format!("{prefix}{}", model.strip_prefix(&prefix).unwrap_or(model));
            if oxllm_core::lookup_pricing_with_provider(&served_model, deployment.provider_slug())
                .is_some()
                || ctx.pricing_model.as_deref().is_none_or(|model| {
                    oxllm_core::lookup_pricing_with_provider(model, deployment.provider_slug())
                        .is_none()
                })
            {
                ctx.pricing_model = Some(served_model);
            }
        }
    }
    let response = if is_stream {
        // No total reqwest deadline for streams (see `timeout_secs` above):
        // bound only the header wait with the stream idle budget. On
        // elapsed, synthesize the same marker-carrying 504
        // `execute_upstream_json` produces for a locally expired timeout so
        // the classification below (and the retry loop's `Timeout` arm, the
        // health tracker) treat it identically.
        match tokio::time::timeout(
            stream_idle_budget,
            crate::pipeline::execute_upstream_json(
                state,
                deployment,
                axum::http::Method::POST,
                &url,
                axum::body::Bytes::from(body_bytes),
                header_map,
                None,
            ),
        )
        .await
        {
            Ok(result) => result,
            Err(_elapsed) => Err(ApiError::gateway_timeout()),
        }
    } else {
        crate::pipeline::execute_upstream_json(
            state,
            deployment,
            axum::http::Method::POST,
            &url,
            axum::body::Bytes::from(body_bytes),
            header_map,
            timeout_secs,
        )
        .await
    }
    .map_err(|e| {
        if e.status == axum::http::StatusCode::GATEWAY_TIMEOUT {
            // `execute_upstream_json` synthesizes a 504 only when the
            // reqwest future timed out locally — no response ever
            // arrived, so this proves nothing about reachability. A
            // genuine upstream 504 arrives as an HTTP response and is
            // classified by status below (as `Retryable`).
            UpstreamError::Timeout(e.message)
        } else {
            UpstreamError::Network(e.message)
        }
    })?;

    if let Some(ctx) = call_ctx {
        ctx.reached_upstream = true;
        crate::pipeline::capture_upstream_headers(ctx, response.headers());
    }
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    // Capture the upstream response headers before `bytes().await` consumes
    // the response. Error responses carry debugging metadata (`x-request-id`,
    // critical for filing provider support tickets) and rate-limit signals
    // (`retry-after`, `x-ratelimit-*`) that the client needs. Without this
    // the proxy would replace the vendor's `retry-after` with a hardcoded
    // 60s and drop every `x-request-id`, making provider-side incidents
    // uncorrelatable. Only an allowlist is forwarded downstream (see
    // `ApiError::into_response`), preserving the strip_leaking_headers
    // security posture — arbitrary upstream headers never reach the client.
    let headers = response.headers().clone();
    let status_u16 = status.as_u16();
    // Pre-check Content-Length BEFORE buffering the body so an error
    // response that advertises more than the general response cap is rejected
    // without reading a single byte (same guard the success path applies at
    // `chat.rs` / `security::sanitize_upstream_response`). A malicious or
    // compromised upstream could otherwise stream a multi-GB error body that
    // the post-read check below would only reject after pinning it all in
    // heap. Mirrors `handle_non_stream_success` (`chat.rs:923`).
    crate::security::check_content_length(&headers, crate::security::MAX_RESPONSE_BODY_BYTES)
        .map_err(|e| UpstreamError::Client {
            status: 502,
            message: e.message,
            headers: Some(headers.clone()),
        })?;
    // Bound the buffered body at the tighter error-body cap even when the
    // upstream omits Content-Length (chunked transfer) or lies about it: stop
    // reading as soon as the cap is exceeded rather than calling
    // `response.bytes().await` (which would buffer the entire body first).
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    let mut oversized = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if buf.len().saturating_add(bytes.len()) > MAX_ERROR_BODY_BYTES {
                    oversized = true;
                    break;
                }
                buf.extend_from_slice(&bytes);
            }
            Err(e) => {
                tracing::warn!(error = %e.without_url(), "failed to read upstream error body");
                return Err(UpstreamError::Retryable {
                    status: 502,
                    message: "failed to read upstream error body".into(),
                    headers: Some(headers.clone()),
                });
            }
        }
    }
    if oversized {
        return Err(UpstreamError::Client {
            status: 502,
            message: format!("upstream error body exceeds {MAX_ERROR_BODY_BYTES} byte limit"),
            headers: Some(headers),
        });
    }
    let bytes = buf;
    // Defence-in-depth: the read loop above already `break`s the moment the
    // running buffer would exceed the cap, so `buf.len()` here is always
    // `<= MAX_ERROR_BODY_BYTES`. This redundant check is intentionally kept as
    // a guard against a future refactor that changes the loop's accumulation
    // strategy (e.g. accepting one final oversized chunk before checking).
    if bytes.len() > MAX_ERROR_BODY_BYTES {
        return Err(UpstreamError::Client {
            status: 502,
            message: format!(
                "upstream error body {} bytes exceeds {MAX_ERROR_BODY_BYTES} byte limit",
                bytes.len()
            ),
            headers: Some(headers),
        });
    }
    let body_text = String::from_utf8_lossy(&bytes).to_string();
    let body: Value = serde_json::from_str(&body_text).unwrap_or(Value::Null);
    // OpenAI: {"error":{"message":"..."}}, xAI/some gateways: {"error":"..."}.
    // When no message can be extracted (non-JSON HTML/text error pages from
    // gateways and reverse proxies), include a truncated sanitized excerpt
    // of the raw body so operators can diagnose the incident — the reference
    // surfaces the raw text in its exception message too.
    let message = body
        .get("error")
        .and_then(|e| {
            e.get("message")
                .and_then(|m| m.as_str())
                .or_else(|| e.as_str())
        })
        .map(str::to_string)
        .unwrap_or_else(|| {
            let excerpt = sanitize_error_body_excerpt(&body_text);
            match excerpt {
                Some(text) => format!("upstream error: {text}"),
                None => "upstream error".to_string(),
            }
        });

    if is_context_window_error(&body) {
        Err(UpstreamError::ContextWindow {
            status: status_u16,
            message,
            headers: Some(headers),
        })
    } else if is_content_policy_error(&body) {
        Err(UpstreamError::ContentPolicy {
            status: status_u16,
            message,
            headers: Some(headers),
        })
    } else if is_retryable_status(status) {
        Err(UpstreamError::Retryable {
            status: status_u16,
            message,
            headers: Some(headers),
        })
    } else {
        Err(UpstreamError::Client {
            status: status_u16,
            message,
            headers: Some(headers),
        })
    }
}

/// Preserve Anthropic document blocks on native targets. On targets with an
/// OpenAI file-part contract, translate inline PDFs to `file_data` (never
/// `image_url`). Unsupported sources fail explicitly rather than disappearing
/// from the prompt or being mislabeled as images.
fn normalize_document_parts(
    request: &Value,
    deployment: &Deployment,
) -> Result<Option<Value>, ApiError> {
    let messages = request.get("messages").and_then(Value::as_array);
    let has_documents = messages.is_some_and(|messages| {
        messages.iter().any(|message| {
            message
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|parts| {
                    parts
                        .iter()
                        .any(|part| part.get("type").and_then(Value::as_str) == Some("document"))
                })
        })
    });
    if !has_documents {
        return Ok(None);
    }
    let provider = deployment.provider_slug();
    if provider == "anthropic"
        || (provider == "vertex_ai" && deployment.oxllm_params.model.contains("claude"))
    {
        return Ok(None);
    }
    // Bedrock only speaks the OpenAI `file` shape through Converse. Claude on
    // InvokeModel keeps Anthropic `document` blocks (the native transform
    // consumes them), so rewriting them would produce an invalid Messages
    // body. Non-Claude InvokeModel models have no document support at all —
    // leave their blocks for the flattening adapter to reject explicitly.
    if provider == "bedrock" {
        let model = &deployment.oxllm_params.model;
        let is_claude = model.to_ascii_lowercase().contains("claude")
            || model.to_ascii_lowercase().contains("anthropic");
        if is_claude {
            let explicit = deployment
                .oxllm_params
                .extra
                .get("use_converse_api")
                .or_else(|| deployment.oxllm_params.extra.get("use_converse"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !oxllm_providers::bedrock::should_use_converse(model, explicit) {
                return Ok(None);
            }
        }
    }
    if !matches!(provider, "openai" | "azure" | "bedrock") {
        return Err(ApiError::bad_request(
            "document translation is not supported for this provider; use a native document-capable endpoint",
        ));
    }
    let mut normalized = request.clone();
    for message in normalized["messages"].as_array_mut().unwrap() {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("document") {
                continue;
            }
            let source = part
                .get("source")
                .ok_or_else(|| ApiError::bad_request("document is missing its source"))?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty());
            *part = match (source.get("type").and_then(Value::as_str), data) {
                (Some("base64"), Some(data))
                    if source.get("media_type").and_then(Value::as_str)
                        == Some("application/pdf") =>
                {
                    serde_json::json!({"type": "file", "file": {
                        "filename": "document.pdf", "file_data": format!("data:application/pdf;base64,{data}")
                    }})
                }
                (Some("text"), Some(data)) => serde_json::json!({"type": "text", "text": data}),
                _ => {
                    return Err(ApiError::bad_request(
                        "document source cannot be translated to this provider; use an inline PDF or a native Anthropic endpoint",
                    ));
                }
            };
        }
    }
    Ok(Some(normalized))
}

/// Transform-time failures: an explicit `BadRequest` from a provider adapter
/// is a deterministic client error (an unsupported parameter shape) and keeps
/// its 400 status; every other failure retains the historical retry/fail-over
/// classification so one untransformable member of a mixed model group cannot
/// fail requests a sibling could serve.
fn transform_failure(e: oxllm_core::Error) -> UpstreamError {
    match e {
        oxllm_core::Error::BadRequest(message) => UpstreamError::Client {
            status: 400,
            message,
            headers: None,
        },
        other => UpstreamError::Retryable {
            status: 500,
            message: other.to_string(),
            headers: None,
        },
    }
}

/// Build a diagnostic excerpt of a non-JSON upstream error body: control
/// characters replaced with spaces, truncated to 200 characters on a char
/// boundary. Returns `None` for empty/whitespace-only bodies (the generic
/// "upstream error" message stands).
fn sanitize_error_body_excerpt(body_text: &str) -> Option<String> {
    let sanitized: String = body_text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let trimmed = sanitized.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut excerpt: String = trimmed.chars().take(200).collect();
    if excerpt.len() < trimmed.len() {
        excerpt.push('…');
    }
    Some(excerpt)
}

/// Characters that must be percent-encoded inside a single URL path
/// segment. Mirrors the Bedrock provider's `MODEL_PATH_SEGMENT` so the
/// wildcard-substituted model lands in the path exactly as a
/// provider-encoded model would.
const WILDCARD_PATH_SEGMENT: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'<')
    .add(b'>')
    .add(b'`')
    .add(b'#')
    .add(b'?')
    .add(b'{')
    .add(b'}')
    .add(b'/')
    .add(b'\\')
    .add(b'%');

/// Whether a wildcard-substituted model string is safe to splice into the
/// URL path. Rejects `.` / `..` segments outright (defense-in-depth on top
/// of percent-encoding, which already neutralises `/`).
fn wildcard_model_substitution(model: &str) -> Option<String> {
    if model == "*" || model.is_empty() {
        return None;
    }
    if model
        .split(['/', '\\'])
        .any(|seg| seg == "." || seg == "..")
    {
        return None;
    }
    Some(percent_encoding::utf8_percent_encode(model, WILDCARD_PATH_SEGMENT).to_string())
}

#[cfg(test)]
mod url_security_regressions {
    use super::*;

    #[test]
    fn wildcard_backslashes_cannot_escape_the_model_path() {
        for model in [r"..\..\admin", r"safe\..\admin", "../admin"] {
            let url = build_upstream_url(
                "https://example.test/api/models/*/invoke".into(),
                Some(model),
                false,
                "bedrock",
                &Value::Null,
            );
            assert_eq!(
                url::Url::parse(&url).expect("url").path(),
                "/api/models/*/invoke"
            );
        }
        assert_eq!(
            wildcard_model_substitution(r"folder\model"),
            Some("folder%5Cmodel".into())
        );
    }

    #[test]
    fn bedrock_streaming_preserves_guardrail_queries() {
        let url = build_upstream_url(
            "https://example.test/model/m/invoke?guardrailIdentifier=g&guardrailVersion=1".into(),
            None,
            true,
            "bedrock",
            &Value::Null,
        );
        assert_eq!(
            url,
            "https://example.test/model/m/invoke-with-response-stream?guardrailIdentifier=g&guardrailVersion=1"
        );
    }
}

/// Build the upstream URL after provider transform has run.
///
/// Applies three normalisations to the URL returned by [`ChatProvider::endpoint_url`]:
///
/// 1. **Wildcard substitution.** When the deployment was a wildcard
///    (`bedrock/*`, `vertex_ai/*`), `endpoint_url` bakes the literal `*`
///    into the path. The resolved body `model` (already normalised by the
///    provider's `transform_request`) replaces every `*` in the URL,
///    percent-encoded as a single path segment — the body model is
///    client-controlled, so splicing it in raw would allow path traversal
///    and query/fragment injection against custom `api_base` gateways.
///    Model strings containing `.` / `..` segments leave the literal `*`
///    in place (surfacing the rejection) rather than being substituted.
/// 2. **Bedrock streaming suffix.** Bedrock encodes streaming in the URL
///    path; only the trailing `/invoke` is rewritten to
///    `/invoke-with-response-stream` (and, for the Converse API, `/converse`
///    is rewritten to `/converse-stream`), so a custom `api_base` containing
///    `/invoke` somewhere earlier is left intact.
/// 3. **Vertex AI streaming suffix.** Claude on Vertex uses
///    `:streamRawPredict` for streaming and `:rawPredict` otherwise; the
///    rewrite fires only when the body still carries `stream: true` (the
///    Anthropic transform keeps it for Vertex).
/// 4. **WatsonX streaming endpoint.** WatsonX exposes streaming under a
///    distinct `/text/chat_stream` route (and
///    `/deployments/{id}/text/chat_stream`) rather than the non-streaming
///    `/text/chat`. The provider trait's `endpoint_url` cannot see the stream
///    flag, so the non-streaming path is baked in and rewritten here. Mirrors
///    the reference `IBMWatsonXChatConfig.get_complete_url`.
pub(crate) fn build_upstream_url(
    url: String,
    body_model: Option<&str>,
    is_stream: bool,
    provider_slug: &str,
    body: &Value,
) -> String {
    let url = match body_model {
        Some(m) if url.contains('*') => match wildcard_model_substitution(m) {
            Some(encoded) => url.replace('*', &encoded),
            // A hostile/missing model string leaves the literal `*` in the
            // URL: the upstream 404s and the misconfiguration (or attack)
            // surfaces instead of being spliced unencoded into the path.
            None => url,
        },
        _ => url,
    };
    let url = if is_stream && provider_slug == "bedrock" {
        if let Some(rewritten) =
            replace_url_path_suffix(&url, "/invoke", "/invoke-with-response-stream")
        {
            rewritten
        } else if let Some(rewritten) =
            replace_url_path_suffix(&url, "/converse", "/converse-stream")
        {
            rewritten
        } else {
            url
        }
    } else {
        url
    };
    let url = if is_stream && provider_slug == "vertex_ai" && url.ends_with(":rawPredict") {
        let needs_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        if needs_stream {
            let prefix_len = url.len() - ":rawPredict".len();
            format!("{}:streamRawPredict", &url[..prefix_len])
        } else {
            url
        }
    } else {
        url
    };
    if is_stream && provider_slug == "watsonx" {
        watsonx_stream_url(url)
    } else {
        url
    }
}

fn replace_url_path_suffix(url: &str, suffix: &str, replacement: &str) -> Option<String> {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    let prefix = url[..end].strip_suffix(suffix)?;
    Some(format!("{prefix}{replacement}{}", &url[end..]))
}

/// Rewrite a WatsonX `/text/chat` URL to its streaming `/text/chat_stream`
/// variant. Handles both the bare and deployments shapes, with or without a
/// trailing query string. Leaves an already-streaming URL untouched.
fn watsonx_stream_url(url: String) -> String {
    if url.contains("/text/chat_stream") {
        return url;
    }
    if let Some(rest) = url.strip_suffix("/text/chat") {
        return format!("{rest}/text/chat_stream");
    }
    if let Some(idx) = url.find("/text/chat?") {
        let (head, tail) = url.split_at(idx + "/text/chat".len());
        return format!("{head}_stream{tail}");
    }
    url
}

/// When streaming, ask the upstream to include usage info in the final
/// SSE chunk. Most OpenAI-compatible providers omit usage otherwise, which
/// would leave `accumulated_usage` null and cause budgets/TPM to miss the
/// request entirely.
///
/// Injected after the provider's `transform_request` has run (so provider
/// transforms see the original body) and before serialization to bytes.
/// Preserves any other fields the client may have set under `stream_options`
/// and **never overwrites an explicit client `include_usage`** (the reference
/// `get_stream_options` also injects only when the field is absent): a
/// client that deliberately opted out keeps its wire contract, and
/// metering still works because the stream pump estimates usage when no
/// usage chunk arrives (`stream_chunk_builder` parity).
///
/// `stream_options` is an OpenAI wire-format field. Providers whose
/// `transform_request` rebuilds the body into a non-OpenAI shape must not
/// receive it: Anthropic's Messages API rejects unknown fields with a 400,
/// and Cohere's v2 Chat format has no such field. Both report usage natively
/// in their streams (Anthropic via `message_start.usage` /
/// `message_delta.usage`, Cohere via `message_end.usage`), so the injection
/// is skipped for them. Bedrock (Anthropic on Bedrock) and Vertex AI's
/// Claude branch likewise take the native Anthropic body shape and report
/// usage natively.
pub(crate) fn inject_stream_usage(body: &mut Value, is_stream: bool, provider_slug: &str) {
    if !is_stream {
        return;
    }
    // Providers with a non-OpenAI wire format report usage natively; splicing
    // OpenAI's `stream_options` into their transformed body is rejected
    // upstream (Anthropic 400s on unknown fields).
    if matches!(provider_slug, "anthropic" | "cohere" | "bedrock") {
        return;
    }
    // Vertex AI serves two wire shapes behind one slug. The Claude branch
    // rewrites the body into the native Anthropic Messages shape (carrying
    // `anthropic_version`) and reports usage natively; the Gemini branch keeps
    // the OpenAI Chat Completions shape and needs the injection. Distinguish
    // by the presence of `anthropic_version`, which only the Claude transform
    // adds.
    if provider_slug == "vertex_ai" && body.get("anthropic_version").is_some() {
        return;
    }
    if let Some(obj) = body.as_object_mut() {
        let stream_options = obj
            .entry("stream_options".to_string())
            .or_insert(serde_json::json!({}));
        if let Some(so) = stream_options.as_object_mut() {
            so.entry("include_usage".to_string())
                .or_insert(serde_json::json!(true));
        }
    }
}

/// Resolve the per-read budget: request `stream_timeout` or `timeout`, then
/// deployment `stream_timeout`, global `request_timeout`, and the pump default.
/// Deployment unary timeouts do not shorten streams. The header wait and pump
/// use the same idle allowance, without a total streaming deadline.
#[cfg(test)]
fn resolve_stream_idle_timeout(
    deployment: &Deployment,
    config: &crate::config::types::ProxyConfig,
) -> std::time::Duration {
    resolve_stream_idle_timeout_for_request(deployment, config, &Value::Null)
}

pub(crate) fn resolve_stream_idle_timeout_for_request(
    deployment: &Deployment,
    config: &crate::config::types::ProxyConfig,
    request: &Value,
) -> std::time::Duration {
    let secs = super::request_controls::seconds(request, "stream_timeout")
        .or_else(|| super::request_controls::seconds(request, "timeout"))
        .or(deployment
            .oxllm_params
            .stream_timeout
            .map(|seconds| seconds as f64))
        .or(config
            .oxllm_settings
            .request_timeout
            .map(|seconds| seconds as f64))
        .unwrap_or(super::stream::DEFAULT_STREAM_TIMEOUT_SECS as f64);
    crate::pipeline::sanitize_timeout_secs(secs)
}

/// Merge deployment-level `extra_headers` (e.g. `anthropic-beta` for prompt
/// caching betas, PDF, computer-use, or custom tracking headers) on top of the
/// provider-applied auth headers so they take precedence. Applied after
/// `apply_auth` and before any SigV4 signing pass.
pub(crate) fn merge_extra_headers(headers: &mut HashMap<String, String>, deployment: &Deployment) {
    if let Some(extra) = &deployment.oxllm_params.extra_headers {
        for (k, v) in extra {
            let key = k.to_ascii_lowercase();
            if key == "anthropic-beta" {
                merge_anthropic_beta_headers(headers, vec![v.clone()]);
            } else {
                headers.insert(key, v.clone());
            }
        }
    }
}

fn deployment_with_transform_settings<'a>(
    deployment: &'a Deployment,
    config: &crate::config::types::ProxyConfig,
) -> std::borrow::Cow<'a, Deployment> {
    if deployment.provider_slug() == "bedrock"
        && deployment.oxllm_params.modify_params.is_none()
        && let Some(modify) = config.oxllm_settings.modify_params
    {
        let mut effective = deployment.clone();
        effective.oxllm_params.modify_params = Some(modify);
        std::borrow::Cow::Owned(effective)
    } else {
        std::borrow::Cow::Borrowed(deployment)
    }
}

/// Provider endpoint defaults use the same resolved credential snapshot as
/// authentication/signing. Explicit operator base URLs always take precedence.
pub(crate) fn deployment_with_resolved_region<'a>(
    deployment: &'a Deployment,
    creds: &ResolvedCredentials,
) -> std::borrow::Cow<'a, Deployment> {
    if deployment.provider_slug() == "voyage"
        && deployment.oxllm_params.api_base.is_none()
        && creds
            .best_effort_bearer()
            .is_some_and(|key| key.starts_with("al-"))
    {
        let mut resolved = deployment.clone();
        resolved.oxllm_params.api_base = Some(oxllm_providers::voyage::MONGODB_VOYAGE_BASE.into());
        return std::borrow::Cow::Owned(resolved);
    }
    if deployment.provider_slug() == "bedrock" && deployment.oxllm_params.api_base.is_none() {
        if let ResolvedCredentials::AwsSigV4 { region, .. } = creds {
            let mut resolved = deployment.clone();
            resolved
                .oxllm_params
                .extra
                .insert("aws_region_name".into(), Value::String(region.clone()));
            return std::borrow::Cow::Owned(resolved);
        }
    }
    std::borrow::Cow::Borrowed(deployment)
}

/// Native relays construct their path before credentials are resolved. Adjust
/// only the default endpoint's origin, retaining the already-confined path and
/// query and using the same credential snapshot as authentication.
pub(crate) fn upstream_url_with_resolved_endpoint<'a>(
    deployment: &Deployment,
    creds: &ResolvedCredentials,
    upstream_url: &'a str,
) -> Result<std::borrow::Cow<'a, str>, ApiError> {
    let effective = deployment_with_resolved_region(deployment, creds);
    if effective.oxllm_params.api_base == deployment.oxllm_params.api_base {
        return Ok(std::borrow::Cow::Borrowed(upstream_url));
    }
    let default_base =
        crate::pipeline::passthrough::default_base_for_provider(deployment.provider_slug())
            .ok_or_else(|| ApiError::internal("provider has no default endpoint"))?;
    let default = url::Url::parse(default_base)
        .map_err(|_| ApiError::internal("invalid default upstream endpoint"))?;
    let resolved_base = effective
        .oxllm_params
        .api_base
        .as_deref()
        .ok_or_else(|| ApiError::internal("resolved upstream endpoint is unavailable"))?;
    let target = url::Url::parse(resolved_base)
        .map_err(|_| ApiError::internal("invalid resolved upstream endpoint"))?;
    let mut resolved =
        url::Url::parse(upstream_url).map_err(|_| ApiError::internal("invalid upstream URL"))?;
    if resolved.origin() != default.origin() {
        return Err(ApiError::internal(
            "upstream URL does not match its default endpoint",
        ));
    }
    resolved
        .set_scheme(target.scheme())
        .map_err(|_| ApiError::internal("invalid resolved endpoint scheme"))?;
    resolved
        .set_host(target.host_str())
        .map_err(|_| ApiError::internal("invalid resolved endpoint host"))?;
    resolved
        .set_port(target.port())
        .map_err(|_| ApiError::internal("invalid resolved endpoint port"))?;
    Ok(std::borrow::Cow::Owned(resolved.into()))
}

/// Merge request-derived `anthropic-beta` tags into the header set, joining
/// them with any deployment-supplied `anthropic-beta` value (set by
/// [`merge_extra_headers`]).
///
/// Tags already present on the existing header are not repeated, so a
/// deployment that pre-configures a tag the request also triggers (e.g.
/// `structured-outputs-2025-11-13` for a request that uses structured
/// outputs) does not produce `tag, tag`. Anthropic tolerates duplicates, but
/// the resulting header is messy and harder to debug. New tags are prepended
/// to preserve the request-derived ordering while keeping operator-supplied
/// tags visible last.
///
/// No-op when `new_betas` is empty (the header is left untouched, including
/// any pre-existing deployment value).
pub(crate) fn merge_anthropic_beta_headers(
    headers: &mut HashMap<String, String>,
    new_betas: Vec<String>,
) {
    if new_betas.is_empty() {
        return;
    }
    let mut merged = Vec::new();
    for beta in headers
        .get("anthropic-beta")
        .into_iter()
        .flat_map(|h| h.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()))
    {
        if !merged.contains(&beta) {
            merged.push(beta);
        }
    }
    for beta in new_betas
        .iter()
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|beta| !beta.is_empty())
    {
        if !merged.contains(&beta) {
            merged.insert(0, beta);
        }
    }
    if merged.is_empty() {
        headers.remove("anthropic-beta");
    } else {
        headers.insert("anthropic-beta".to_string(), merged.join(", "));
    }
}

/// Apply `drop_params` semantics and run the provider's
/// [`ChatProvider::transform_request`].
///
/// Returns:
/// - `Ok(Some(value))` when either (a) `drop_params=true` and unsupported
///   keys were stripped, or (b) the unsupported-param warning path ran but
///   the provider still wants a fresh body. The caller uses the returned
///   value as the transformed request body.
/// - `Ok(None)` when `drop_params=false` AND the provider does not publish a
///   `supported_params` list (no stripping, no warning, no body mutation).
///   The caller then passes the original `request` straight to
///   `transform_request`, avoiding a deep `Value::clone` (M31) of bodies
///   that would otherwise be untouched on the hot path.
/// - `Err` when the provider's `transform_request` failed.
///
/// (Previously this always returned a `Value`, cloning `request` even when
/// nothing about it changed. Under load — most production traffic has
/// `drop_params=false` and most providers don't publish `supported_params` —
/// that's a free deep clone of every request body.)
pub(crate) fn apply_drop_params(
    state: &AppState,
    chat: &dyn ChatProvider,
    request: &Value,
    deployment: &Deployment,
) -> Result<Option<Value>, oxllm_core::Error> {
    // An explicit per-deployment flag beats the global one. Unset inherits.
    let drop_params = deployment
        .oxllm_params
        .drop_params
        .or(state.config.oxllm_settings.drop_params)
        .unwrap_or(false);

    // Gateway control objects must never reach a provider. The `cache`
    // opt-in object (read by the CacheHook from `ctx.request_body`) is an
    // oxllm control, not a model param: OpenAI rejects it with
    // `Unrecognized request argument supplied: cache`, so every
    // cache-opted-in chat request against the native OpenAI provider 400'd.
    // Strip it for ALL providers — including those that publish no
    // `supported_params` allowlist (the native OpenAI provider deliberately
    // forwards unknown keys for forward-compat, which is exactly how the
    // control object leaked). The same applies to `metadata.tags` (the
    // routing/metering tag list): real upstreams validate `metadata` as a
    // string map, so an array-valued `tags` fails validation (Groq:
    // `'metadata.tags': value must be a string`). Only the `tags` key is
    // removed — the rest of `metadata` (e.g. `user_id`) is provider-visible
    // by design. The strip only clones when a control is present,
    // preserving the no-clone fast path below for the common case.
    let control_stripped = super::request_controls::contains(request).then(|| {
        let mut body = request.clone();
        super::request_controls::strip_before_transform(&mut body);
        body
    });
    let request: &Value = control_stripped.as_ref().unwrap_or(request);
    crate::native_output_policy::validate_formats(request)?;
    let requested_output = crate::native_output_policy::contract(request);

    let needs_outbound_shape = deployment
        .oxllm_params
        .additional_drop_params
        .as_ref()
        .is_some_and(|keys| !keys.is_empty())
        || deployment
            .oxllm_params
            .extra_body
            .as_ref()
            .is_some_and(|body| !body.is_empty())
        || deployment
            .oxllm_params
            .allowed_openai_params
            .as_ref()
            .is_some_and(|keys| !keys.is_empty());

    // Fast path: no stripping and no post-transform shaping. Returning
    // `None` tells the caller to forward the original body to
    // `transform_request` unchanged. Skipped when gateway controls were
    // stripped above, or when the deployment asks for extra drops or
    // `extra_body`.
    if !drop_params
        && chat.supported_params().is_none()
        && control_stripped.is_none()
        && !needs_outbound_shape
        && requested_output.is_none()
    {
        warn_unsupported_params(chat, request);
        return Ok(None);
    }

    let allowed = deployment.oxllm_params.allowed_openai_params.as_deref();
    let mut request = body_after_drop_policy(chat, request, drop_params, allowed);
    remove_additional_params(
        &mut request,
        deployment.oxllm_params.additional_drop_params.as_deref(),
    );
    crate::native_output_policy::ensure_preserved(&requested_output, &request)?;

    // Providers may apply value-aware dropping after the top-level key filter.
    // Resolve the global policy onto the deployment without exposing a gateway
    // control in the request body. Explicit deployment false still wins.
    let mut transform_deployment = std::borrow::Cow::Borrowed(deployment);
    if drop_params && deployment.oxllm_params.drop_params.is_none() {
        transform_deployment.to_mut().oxllm_params.drop_params = Some(true);
    }
    let mut transformed = chat.transform_request(&request, &transform_deployment)?;
    if crate::native_output_policy::has_message_formats(&request)
        && !uses_anthropic_messages_wire(deployment, &transformed)
    {
        return Err(oxllm_core::Error::BadRequest("message-level output_config.format has no native representation in this adapter; use a native Anthropic Messages deployment".into()));
    }
    let native_output =
        crate::native_output_policy::native_contract(&requested_output, &transformed)?;
    // Bedrock intentionally publishes no exhaustive allow-list, but consumes
    // service_tier by renaming it (Converse) or omitting it (Invoke). Explicit
    // allow-list restoration must not put the raw spelling back on the wire.
    let mapped = chat
        .supported_params()
        .or_else(|| (deployment.provider_slug() == "bedrock").then(|| vec!["service_tier"]));
    restore_allowed_params(
        &mut transformed,
        &request,
        allowed,
        mapped.as_deref(),
        requested_output.as_ref(),
    );
    shape_outbound_body(
        &mut transformed,
        deployment.oxllm_params.additional_drop_params.as_deref(),
        deployment.oxllm_params.extra_body.as_ref(),
    );
    crate::native_output_policy::ensure_preserved(&native_output, &transformed)?;
    Ok(Some(transformed))
}

/// This identifies the API envelope actually chosen by an adapter, never a
/// model's feature availability. Model, schema and beta acceptance stay remote.
fn uses_anthropic_messages_wire(deployment: &Deployment, body: &Value) -> bool {
    match deployment.provider_slug() {
        "anthropic" => true,
        "bedrock" | "vertex_ai" => body.get("anthropic_version").is_some(),
        "azure_ai" => {
            let mut resolved = deployment.clone();
            if resolved.oxllm_params.model.contains('*')
                && let Some(model) = body.get("model").and_then(Value::as_str)
            {
                resolved.oxllm_params.model = format!("azure_ai/{model}");
            }
            oxllm_providers::azure_ai::AzureAiProvider::is_anthropic(&resolved)
        }
        _ => false,
    }
}

/// Apply the pre-transform drop/allow policy.
///
/// `drop_params` strips keys the provider does not list, except keys in
/// `allowed`. When drop is off the body is cloned unchanged (after the
/// unsupported-param warning).
pub(crate) fn body_after_drop_policy(
    chat: &dyn ChatProvider,
    request: &Value,
    drop_params: bool,
    allowed: Option<&[String]>,
) -> Value {
    if drop_params {
        strip_unsupported_params(chat, request, allowed)
    } else {
        warn_unsupported_params(chat, request);
        request.clone()
    }
}

/// Restore explicitly allowed request keys that an adapter's allow-list
/// omitted. Mapped provider fields win, and routing always owns `model`.
pub(crate) fn restore_allowed_params(
    body: &mut Value,
    request: &Value,
    allowed: Option<&[String]>,
    mapped: Option<&[&str]>,
    protected: Option<&Value>,
) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    for key in allowed
        .into_iter()
        .flatten()
        .filter(|key| key.as_str() != "model")
        .filter(|key| !mapped.is_some_and(|params| params.contains(&key.as_str())))
        .filter(|key| !crate::native_output_policy::controls_param(protected, key))
    {
        if let Some(value) = request.get(key).filter(|value| !value.is_null()) {
            obj.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
}

/// Remove request keys before adapters can rename or nest them. The final
/// outbound pass repeats this after configured body overrides.
pub(crate) fn remove_additional_params(body: &mut Value, additional_drop: Option<&[String]>) {
    if let Some(obj) = body.as_object_mut() {
        for key in additional_drop.into_iter().flatten() {
            obj.remove(key);
        }
    }
}

/// Shallow-merge configured body overrides, then drop named keys. The
/// deployment overrides client/provider values, including privacy controls.
/// Routing owns `model`; body overrides cannot replace the resolved model.
pub(crate) fn shape_outbound_body(
    body: &mut Value,
    additional_drop: Option<&[String]>,
    extra_body: Option<&serde_json::Map<String, Value>>,
) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    if let Some(extra) = extra_body {
        for (key, value) in extra {
            if key != "model" {
                obj.insert(key.clone(), value.clone());
            }
        }
    }
    if let Some(drops) = additional_drop {
        for key in drops {
            obj.remove(key);
        }
    }
    super::request_controls::strip(body);
}

/// Strip top-level request keys not present in the provider's
/// `supported_params` list. When no list is published, returns the input
/// unchanged (the provider treats unknown keys as forward-compat).
pub(crate) fn strip_unsupported_params(
    chat: &dyn ChatProvider,
    request: &Value,
    allowed: Option<&[String]>,
) -> Value {
    let Some(supported) = chat.supported_params() else {
        return request.clone();
    };
    let Some(obj) = request.as_object() else {
        return request.clone();
    };
    let mut out = serde_json::Map::new();
    for (key, value) in obj {
        let kept = supported.contains(&key.as_str())
            || key == "cache_control_injection_points"
            || allowed.is_some_and(|names| names.iter().any(|name| name == key));
        if kept {
            out.insert(key.clone(), value.clone());
        }
    }
    Value::Object(out)
}

/// Log a warning for any top-level request key the provider doesn't list in
/// `supported_params`. No-op when the provider does not publish a list.
pub(crate) fn warn_unsupported_params(chat: &dyn ChatProvider, request: &Value) {
    let Some(supported) = chat.supported_params() else {
        return;
    };
    let Some(obj) = request.as_object() else {
        return;
    };
    for key in obj.keys() {
        if !supported.contains(&key.as_str()) {
            tracing::warn!(
            param = %key,
            "unsupported param for provider (drop_params=false); passing through"
            );
        }
    }
}

/// Resolve the provider Arc for a deployment via the registry.
pub(crate) fn resolve_provider(
    state: &AppState,
    deployment: &Deployment,
) -> Result<Arc<dyn Provider>, ApiError> {
    state
        .registry
        .resolve_for_deployment(deployment)
        .cloned()
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "unsupported provider: '{}'. Register it in the registry or use a known provider.",
                deployment.provider_slug()
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voyage_default_endpoint_uses_resolved_credentials_and_respects_explicit_bases() {
        let mut deployment: Deployment =
            serde_json::from_value(json!({"model_name":"embedded", "oxllm_params":{
                "model":"voyage/voyage-3","api_key":"env/UNRESOLVED_TEST_KEY"
            }}))
            .unwrap();
        let ordinary = ResolvedCredentials::from_key("ordinary-key");
        let database = ResolvedCredentials::from_key("al-issued-key");
        for (creds, base) in [
            (&ordinary, oxllm_providers::voyage::DEFAULT_VOYAGE_BASE),
            (&database, oxllm_providers::voyage::MONGODB_VOYAGE_BASE),
        ] {
            let resolved = deployment_with_resolved_region(&deployment, creds);
            assert_eq!(
                oxllm_core::EmbeddingProvider::embedding_url(
                    &oxllm_providers::voyage::VoyageProvider,
                    &resolved
                ),
                format!("{base}/embeddings")
            );
            assert_eq!(
                oxllm_core::RerankProvider::rerank_url(
                    &oxllm_providers::voyage::VoyageProvider,
                    &resolved
                ),
                format!("{base}/rerank")
            );
        }
        deployment.oxllm_params.api_base = Some("https://operator.example.test/v1".into());
        let resolved = deployment_with_resolved_region(&deployment, &database);
        assert_eq!(
            resolved.oxllm_params.api_base.as_deref(),
            Some("https://operator.example.test/v1")
        );
        assert_eq!(
            deployment.oxllm_params.api_key.as_deref(),
            Some("env/UNRESOLVED_TEST_KEY")
        );
    }

    #[test]
    fn voyage_native_endpoints_keep_paths_queries_and_explicit_operator_bases() {
        let mut deployment: Deployment = serde_json::from_value(json!({
            "model_name":"embedded", "oxllm_params":{
                "model":"voyage/voyage-3", "api_key":"env/UNRESOLVED_TEST_KEY"
            }
        }))
        .unwrap();
        let issued = ResolvedCredentials::from_key("al-issued-key");
        let ordinary = ResolvedCredentials::from_key("ordinary-key");
        for path in ["/v1/embeddings?value=a%2Fb", "/v1/rerank", "/embeddings"] {
            let original = format!("https://api.voyageai.com{path}");
            assert_eq!(
                upstream_url_with_resolved_endpoint(&deployment, &issued, &original).unwrap(),
                format!("https://ai.mongodb.com{path}")
            );
            assert!(matches!(
                upstream_url_with_resolved_endpoint(&deployment, &ordinary, &original).unwrap(),
                std::borrow::Cow::Borrowed(_)
            ));
        }
        assert!(
            upstream_url_with_resolved_endpoint(
                &deployment,
                &issued,
                "https://unrelated.example.test/v1/embeddings"
            )
            .is_err()
        );
        deployment.oxllm_params.api_base = Some("https://operator.example.test/custom".into());
        let explicit = "https://operator.example.test/custom/embeddings?value=a%2Fb";
        assert_eq!(
            upstream_url_with_resolved_endpoint(&deployment, &issued, explicit).unwrap(),
            explicit
        );
    }
    use oxllm_core::deployment::OxllmParams;
    use oxllm_providers::anthropic::AnthropicProvider;
    use oxllm_providers::openai::OpenAiProvider;
    use serde_json::json;

    #[test]
    fn document_normalization_uses_file_parts_and_rejects_unmappable_sources() {
        let mut deployment = Deployment {
            model_name: "m".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut request = json!({"messages":[{"role":"user","content":[
            {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}
        ]}]});
        let normalized = normalize_document_parts(&request, &deployment)
            .unwrap()
            .unwrap();
        let part = &normalized["messages"][0]["content"][0];
        assert_eq!(part["type"], "file");
        assert_eq!(
            part["file"]["file_data"],
            "data:application/pdf;base64,JVBERi0="
        );
        assert!(part.get("image_url").is_none());
        deployment.oxllm_params.model = "anthropic/claude-3-5-sonnet".to_string();
        assert!(
            normalize_document_parts(&request, &deployment)
                .unwrap()
                .is_none()
        );
        deployment.oxllm_params.model = "openai/gpt-4o".to_string();
        request["messages"][0]["content"][0]["source"] =
            json!({"type":"url","url":"https://example.com/report.pdf"});
        assert_eq!(
            normalize_document_parts(&request, &deployment)
                .unwrap_err()
                .status,
            axum::http::StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn strip_unsupported_params_drops_unknown_keys() {
        let chat = AnthropicProvider;
        let request = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 100,
        "frequency_penalty": 0.5,
        "logprobs": true,
        });
        let out = strip_unsupported_params(&chat, &request, None);
        assert!(out.get("frequency_penalty").is_none());
        assert!(out.get("logprobs").is_none());
        assert_eq!(out["max_tokens"], 100);
        assert_eq!(out["model"], "claude");
    }

    #[tokio::test]
    async fn service_tier_drop_policy_preserves_valid_values_and_drops_only_invalid_ones() {
        for model in ["anthropic/claude-sonnet-4-5", "azure_ai/claude-sonnet-4-5"] {
            for global in [false, true] {
                for local in [None, Some(false), Some(true)] {
                    let mut state = state_for_openai_test();
                    std::sync::Arc::make_mut(&mut state.config)
                        .oxllm_settings
                        .drop_params = Some(global);
                    let deployment = Deployment {
                        oxllm_params: OxllmParams {
                            model: model.into(),
                            api_base: Some("https://example.test/anthropic/v1".into()),
                            drop_params: local,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    let provider = state.registry.resolve_for_deployment(&deployment).unwrap();
                    let chat = provider.chat().unwrap();
                    for (tier, expected) in [
                        ("auto", "auto"),
                        ("default", "standard_only"),
                        ("standard_only", "standard_only"),
                    ] {
                        let request = json!({"messages":[{"role":"user","content":"hello"}],"service_tier":tier});
                        let body = apply_drop_params(&state, chat, &request, &deployment)
                            .unwrap()
                            .unwrap();
                        assert_eq!(
                            body["service_tier"], expected,
                            "{model} global={global} local={local:?}"
                        );
                    }
                    let request = json!({"messages":[{"role":"user","content":"hello"}],"service_tier":"flex"});
                    let result = apply_drop_params(&state, chat, &request, &deployment);
                    if local.unwrap_or(global) {
                        assert!(result.unwrap().unwrap().get("service_tier").is_none());
                        let mut allowed = deployment.clone();
                        allowed.oxllm_params.allowed_openai_params =
                            Some(vec!["service_tier".into()]);
                        assert!(
                            apply_drop_params(&state, chat, &request, &allowed)
                                .unwrap()
                                .unwrap()
                                .get("service_tier")
                                .is_none(),
                            "invalid tier must not be restored by the allow list"
                        );
                    } else {
                        assert!(
                            matches!(result, Err(oxllm_core::Error::BadRequest(message)) if message.contains("service_tier"))
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn bedrock_allow_list_cannot_restore_a_translated_service_tier() {
        let state = state_for_openai_test();
        for converse in [false, true] {
            let deployment = Deployment {
                oxllm_params: OxllmParams {
                    model: "bedrock/anthropic.claude-sonnet-4-5".into(),
                    allowed_openai_params: Some(vec!["service_tier".into()]),
                    extra: serde_json::from_value(json!({"use_converse_api":converse})).unwrap(),
                    ..Default::default()
                },
                ..Default::default()
            };
            let request =
                json!({"messages":[{"role":"user","content":"hello"}],"service_tier":"auto"});
            let body = apply_drop_params(
                &state,
                &oxllm_providers::bedrock::BedrockProvider,
                &request,
                &deployment,
            )
            .unwrap()
            .unwrap();
            assert!(
                body.get("service_tier").is_none(),
                "raw tier must not be reinserted: {body}"
            );
            if converse {
                assert_eq!(body["serviceTier"], json!({"type":"default"}));
            }
        }
    }

    /// `metadata.tags` never reaches a provider: real upstreams validate
    /// `metadata` as a string map and reject the gateway's tag array
    /// (real-API failure: `'metadata.tags': value must be a string`). The
    /// rest of `metadata` passes through for providers that accept it.
    #[tokio::test]
    async fn apply_drop_params_strips_metadata_tags_for_all_providers() {
        let state = state_for_openai_test();
        let chat = OpenAiProvider;
        let deployment = Deployment {
            model_name: "gpt".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o-mini".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let request = json!({
            "model": "gpt",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"tags": ["app-a"], "user_id": "u-1"},
        });
        let out = apply_drop_params(&state, &chat, &request, &deployment)
            .expect("transform ok")
            .expect("stripped body must be returned");
        // The native OpenAI transform drops the whole `metadata` object
        // unless `store: true` (the API rejects it otherwise), so either
        // outcome must be tag-free downstream.
        assert!(
            out.pointer("/metadata/tags").is_none(),
            "metadata.tags must be stripped: {out}"
        );
    }

    /// The `cache` opt-in object is a GATEWAY control (read by the CacheHook
    /// from `ctx.request_body`), not a model param: it must be stripped for
    /// every provider before the transform. The native OpenAI provider
    /// publishes no `supported_params` allowlist (unknown keys forward for
    /// forward-compat), so without this strip every cache-opted-in request
    /// against real OpenAI failed with `Unrecognized request argument
    /// supplied: cache` — a leak the mock upstreams in the e2e suite cannot
    /// catch because they ignore unknown fields.
    /// State with `drop_params` deliberately OFF: control strips must
    /// happen regardless, because the openai provider has no allowlist for
    /// `drop_params` to act on.
    fn state_for_openai_test() -> crate::state::AppState {
        crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state")
    }

    #[tokio::test]
    async fn apply_drop_params_strips_gateway_cache_control_for_openai() {
        let state = state_for_openai_test();
        let chat = OpenAiProvider;
        let deployment = Deployment {
            model_name: "gpt".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o-mini".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let request = json!({
            "model": "gpt",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "cache": {"insert": true, "use_cache": true},
        });
        let out = apply_drop_params(&state, &chat, &request, &deployment)
            .expect("transform ok")
            .expect("the stripped body must be returned, not Ok(None)");
        assert!(
            out.get("cache").is_none(),
            "gateway cache control must not reach the provider transform: {out}"
        );
        assert_eq!(out["model"], "gpt-4o-mini");

        let with_chunk_size = json!({
            "model": "gpt",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "stream_chunk_size": 64,
            "log_client_error_tracebacks": true,
        });
        let out = apply_drop_params(&state, &chat, &with_chunk_size, &deployment)
            .expect("transform ok")
            .expect("stream_chunk_size must force a stripped body");
        assert!(
            out.get("stream_chunk_size").is_none(),
            "stream_chunk_size is a proxy read size, not a model parameter: {out}"
        );
        assert!(out.get("log_client_error_tracebacks").is_none());
        assert_eq!(with_chunk_size["log_client_error_tracebacks"], true);

        // Without the control object the fast path is preserved: the openai
        // provider publishes no allowlist and drop_params is off, so the
        // original body is forwarded untouched (Ok(None)).
        let plain = json!({
            "model": "gpt",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        });
        assert!(
            apply_drop_params(&state, &chat, &plain, &deployment)
                .expect("transform ok")
                .is_none(),
            "no control object + no allowlist + drop_params=false must keep the no-clone fast path"
        );
    }

    #[test]
    fn strip_unsupported_params_keeps_anthropic_thinking() {
        // Issue #12: `thinking` must be in supported_params so it survives
        // `drop_params: true` instead of being stripped before transform.
        let chat = AnthropicProvider;
        let request = json!({
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        });
        let out = strip_unsupported_params(&chat, &request, None);
        assert_eq!(out["thinking"]["type"], "enabled");
        assert_eq!(out["thinking"]["budget_tokens"], 1024);
    }

    #[test]
    fn strip_unsupported_params_keeps_anthropic_top_k_and_metadata() {
        let chat = AnthropicProvider;
        let request = json!({
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "top_k": 40,
        "metadata": {"user_id": "u1"},
        });
        let out = strip_unsupported_params(&chat, &request, None);
        assert_eq!(out["top_k"], 40);
        assert_eq!(out["metadata"]["user_id"], "u1");
    }

    #[test]
    fn thinking_survives_drop_params_through_transform() {
        // Issue #12 regression: with `drop_params: true`, `thinking` survives
        // the strip+transform pipeline and reaches the Anthropic body.
        let chat = AnthropicProvider;
        let request = json!({
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        });
        let stripped = strip_unsupported_params(&chat, &request, None);
        let deployment = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let transformed = chat
            .transform_request(&stripped, &deployment)
            .expect("transform ok");
        assert_eq!(transformed["thinking"]["type"], "enabled");
        assert_eq!(transformed["thinking"]["budget_tokens"], 1024);
    }

    #[test]
    fn extra_headers_merged_on_top_of_auth() {
        // Issue #9 regression: deployment `extra_headers` (e.g.
        // `anthropic-beta`) must be forwarded on the upstream request and take
        // precedence over provider-applied headers.
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-api-key".to_string(), "sk-test".to_string());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        let deployment = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                extra_headers: Some(std::collections::HashMap::from([
                    (
                        "anthropic-beta".to_string(),
                        "prompt-caching-2024-07-31".to_string(),
                    ),
                    ("x-custom".to_string(), "track-me".to_string()),
                ])),
                ..Default::default()
            },
            ..Default::default()
        };
        merge_extra_headers(&mut headers, &deployment);
        assert_eq!(
            headers.get("anthropic-beta").map(String::as_str),
            Some("prompt-caching-2024-07-31")
        );
        assert_eq!(
            headers.get("x-custom").map(String::as_str),
            Some("track-me")
        );
        // Pre-existing auth headers are preserved.
        assert_eq!(
            headers.get("x-api-key").map(String::as_str),
            Some("sk-test")
        );
    }

    #[test]
    fn extra_headers_override_provider_header() {
        // extra_headers take precedence over an existing provider-applied
        // header of the same name.
        let mut headers = std::collections::HashMap::new();
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
        let deployment = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                extra_headers: Some(std::collections::HashMap::from([(
                    "anthropic-version".to_string(),
                    "custom-version".to_string(),
                )])),
                ..Default::default()
            },
            ..Default::default()
        };
        merge_extra_headers(&mut headers, &deployment);
        assert_eq!(
            headers.get("anthropic-version").map(String::as_str),
            Some("custom-version")
        );
    }

    #[test]
    fn extra_headers_noop_when_absent() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-api-key".to_string(), "sk-test".to_string());
        let deployment = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        merge_extra_headers(&mut headers, &deployment);
        assert_eq!(headers.len(), 1);
        assert_eq!(
            headers.get("x-api-key").map(String::as_str),
            Some("sk-test")
        );
    }

    #[test]
    fn strip_unsupported_params_passthrough_when_no_list() {
        let chat = OpenAiProvider;
        let request = json!({"a": 1, "b": 2});
        let out = strip_unsupported_params(&chat, &request, None);
        assert_eq!(out, request);
    }

    #[test]
    fn warn_unsupported_params_is_safe_no_op_when_no_object() {
        let chat = AnthropicProvider;
        warn_unsupported_params(&chat, &json!("not an object"));
        warn_unsupported_params(&chat, &json!(42));
        warn_unsupported_params(&chat, &json!([1, 2, 3]));
    }

    #[test]
    fn resolve_provider_for_unsupported_slug_errors() {
        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        let dep = Deployment {
            model_name: "x".to_string(),
            oxllm_params: OxllmParams {
                model: "madeup/model".to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(resolve_provider(&state, &dep).is_err());
    }

    #[test]
    fn inject_stream_usage_adds_include_usage_when_streaming() {
        let mut body = json!({"model": "gpt-4o", "messages": [], "stream": true});
        inject_stream_usage(&mut body, true, "openai");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn inject_stream_usage_noop_when_not_streaming() {
        let mut body = json!({"model": "gpt-4o", "messages": []});
        inject_stream_usage(&mut body, false, "openai");
        assert!(
            body.get("stream_options").is_none(),
            "non-streaming requests must not get stream_options"
        );
    }

    #[test]
    fn inject_stream_usage_noop_when_body_is_not_object() {
        let mut body = json!("not an object");
        inject_stream_usage(&mut body, true, "openai");
        assert_eq!(body, json!("not an object"));
    }

    #[test]
    fn inject_stream_usage_preserves_explicit_client_false() {
        // MEDIUM (C11): an explicit client `include_usage` must never be
        // overwritten (the reference `get_stream_options` injects only when the
        // field is absent). Metering for opted-out streams still works via
        // the stream pump's estimated-usage fallback.
        let mut body = json!({"stream": true, "stream_options": {"include_usage": false}});
        inject_stream_usage(&mut body, true, "openai");
        assert_eq!(
            body["stream_options"]["include_usage"], false,
            "explicit client opt-out must be preserved"
        );
        // An explicit true stays true.
        let mut body = json!({"stream": true, "stream_options": {"include_usage": true}});
        inject_stream_usage(&mut body, true, "openai");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn inject_stream_usage_preserves_other_stream_options_fields() {
        let mut body =
            json!({"stream": true, "stream_options": {"include_usage": false, "other": "keep"}});
        inject_stream_usage(&mut body, true, "openai");
        assert_eq!(body["stream_options"]["include_usage"], false);
        assert_eq!(body["stream_options"]["other"], "keep");
    }

    #[test]
    fn inject_stream_usage_skipped_for_anthropic() {
        // Anthropic's transform rebuilds the body into Messages API format,
        // which 400s on unknown fields; usage arrives natively via
        // `message_start.usage` / `message_delta.usage`.
        let mut body = json!({"model": "claude", "messages": [], "stream": true});
        inject_stream_usage(&mut body, true, "anthropic");
        assert!(
            body.get("stream_options").is_none(),
            "anthropic requests must not get OpenAI's stream_options"
        );
    }

    #[test]
    fn inject_stream_usage_skipped_for_cohere() {
        // Cohere's transform rebuilds the body into v2 Chat format; usage
        // arrives natively via `message_end.usage`.
        let mut body = json!({"model": "command-r", "messages": [], "stream": true});
        inject_stream_usage(&mut body, true, "cohere");
        assert!(
            body.get("stream_options").is_none(),
            "cohere requests must not get OpenAI's stream_options"
        );
    }

    #[test]
    fn inject_stream_usage_skipped_for_bedrock() {
        // Bedrock Anthropic takes the native Messages body shape, which 400s
        // on unknown fields. Usage arrives natively in the eventstream.
        let mut body =
            json!({"model": "anthropic.claude-3-5-sonnet", "messages": [], "stream": true});
        inject_stream_usage(&mut body, true, "bedrock");
        assert!(
            body.get("stream_options").is_none(),
            "bedrock requests must not get OpenAI's stream_options"
        );
    }

    #[test]
    fn inject_stream_usage_skipped_for_vertex_ai() {
        // Vertex AI's Claude branch rewrites the body into the native Messages
        // shape (carrying `anthropic_version`); usage arrives natively in the
        // SSE stream. The body model name alone is no longer the discriminator.
        let mut body = json!({
            "model": "claude-3-5-sonnet",
            "messages": [],
            "stream": true,
            "anthropic_version": "vertex-2023-10-16"
        });
        inject_stream_usage(&mut body, true, "vertex_ai");
        assert!(
            body.get("stream_options").is_none(),
            "vertex_ai Claude requests must not get OpenAI's stream_options"
        );
    }

    #[test]
    fn inject_stream_usage_added_for_vertex_ai_gemini() {
        // Vertex AI's Gemini branch keeps the OpenAI Chat Completions shape
        // (no `anthropic_version`); usage must be injected so metering sees
        // the final chunk.
        let mut body = json!({
            "model": "gemini-2.0-flash",
            "messages": [],
            "stream": true,
        });
        inject_stream_usage(&mut body, true, "vertex_ai");
        assert_eq!(
            body["stream_options"]["include_usage"], true,
            "vertex_ai Gemini requests must get stream_options.include_usage"
        );
    }

    #[test]
    fn inject_stream_usage_added_for_openai() {
        // OpenAI-compat providers keep the injection: their wire format is
        // OpenAI's, and usage is otherwise omitted from the stream.
        // `bedrock_mantle` is OpenAI-compatible (it forwards to Bedrock
        // Mantle's `/v1/chat/completions` endpoint), so it belongs HERE, not
        // in the native-shape skip list.
        for slug in [
            "openai",
            "azure",
            "groq",
            "xai",
            "together_ai",
            "openrouter",
            "gemini",
            "ollama",
            "mistral",
            "bedrock_mantle",
        ] {
            let mut body = json!({"model": "m", "messages": [], "stream": true});
            inject_stream_usage(&mut body, true, slug);
            assert_eq!(
                body["stream_options"]["include_usage"], true,
                "provider '{slug}' must get stream_options.include_usage"
            );
        }
    }

    #[tokio::test]
    async fn credential_resolution_failure_maps_to_retryable() {
        // F13: an unresolvable credential (missing/expired key) is a
        // deployment-level fault, not a request-level client error. Mapping
        // it to `Retryable` lets the retry loop cool this deployment and
        // fail over to a healthy twin instead of aborting the whole chain.
        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        // No api_key: StaticKeyResolver fails before any network I/O.
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let err = attempt_upstream(
            &state,
            &dep,
            &json!({"messages": []}),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("missing api_key must fail credential resolution");

        match err {
            UpstreamError::Retryable {
                status,
                message,
                headers,
            } => {
                assert_eq!(status, 500);
                // Credential-resolution detail (env-var/field names) is an
                // operational internal and must be redacted from the
                // client-facing message — only a generic "internal error"
                // reaches the Retryable payload. The real detail is logged
                // server-side via the `From<oxllm_core::Error>` conversion.
                assert_eq!(
                    message, "internal error",
                    "credential detail must not leak into the message: {message}"
                );
                assert!(
                    !message.contains("missing api_key"),
                    "credential detail leaked: {message}"
                );
                // Credential resolution fails before any HTTP response
                // arrives, so there are no upstream headers to preserve.
                assert!(headers.is_none());
            }
            other => panic!("expected UpstreamError::Retryable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn upstream_429_captures_response_headers() {
        // A provider 429's headers (x-request-id, retry-after, x-ratelimit-*)
        // must be preserved on the UpstreamError so the client-facing
        // response can forward the allowlist. Without this capture the
        // vendor's retry-after would be replaced with a hardcoded 60s and
        // every x-request-id lost — making provider-side incidents
        // uncorrelatable.
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(429)
                .header("content-type", "application/json")
                .header("retry-after", "5")
                .header("x-request-id", "req_xyz")
                .header("x-ratelimit-remaining-requests", "0")
                .header("server", "vendor/1.0")
                .body(r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#);
        });

        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings {
                    // Mock server runs on loopback; allow it through the
                    // SSRF defense.
                    allow_private_api_base: Some(true),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let err = attempt_upstream(
            &state,
            &dep,
            &json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("429 must classify as UpstreamError");

        match err {
            UpstreamError::Retryable {
                status,
                message,
                headers,
            } => {
                assert_eq!(status, 429);
                assert!(
                    message.contains("rate limited"),
                    "unexpected message: {message}"
                );
                let headers = headers.expect("429 response must carry captured headers");
                assert_eq!(
                    headers.get("retry-after").and_then(|v| v.to_str().ok()),
                    Some("5")
                );
                assert_eq!(
                    headers.get("x-request-id").and_then(|v| v.to_str().ok()),
                    Some("req_xyz")
                );
                assert_eq!(
                    headers
                        .get("x-ratelimit-remaining-requests")
                        .and_then(|v| v.to_str().ok()),
                    Some("0")
                );
                // The full HeaderMap is captured (forwarding is gated by the
                // allowlist in ApiError::into_response); even non-allowlisted
                // headers are present here so they remain available for
                // logging/diagnostics.
                assert_eq!(
                    headers.get("server").and_then(|v| v.to_str().ok()),
                    Some("vendor/1.0")
                );
            }
            other => panic!("expected UpstreamError::Retryable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn upstream_4xx_captures_response_headers_as_client() {
        // A non-retryable 4xx (e.g. 400) must still capture headers —
        // x-request-id is just as valuable for debugging client errors.
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(400)
                .header("content-type", "application/json")
                .header("x-request-id", "req_400")
                .body(r#"{"error":{"message":"bad model"}}"#);
        });

        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings {
                    allow_private_api_base: Some(true),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let err = attempt_upstream(
            &state,
            &dep,
            &json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("400 must classify as UpstreamError");

        match err {
            UpstreamError::Client {
                status,
                message,
                headers,
            } => {
                assert_eq!(status, 400);
                assert!(
                    message.contains("bad model"),
                    "unexpected message: {message}"
                );
                let headers = headers.expect("400 response must carry captured headers");
                assert_eq!(
                    headers.get("x-request-id").and_then(|v| v.to_str().ok()),
                    Some("req_400")
                );
            }
            other => panic!("expected UpstreamError::Client, got {other:?}"),
        }
    }

    #[test]
    fn max_error_body_bytes_is_tighter_than_response_limit() {
        // The error-body cap must be far smaller than the 50 MB general
        // response cap: upstream error bodies are buffered, parsed, and
        // surfaced into our own error message, so an oversized one is a
        // memory / log DoS vector.
        assert_eq!(MAX_ERROR_BODY_BYTES, 10 * 1024);
        // Keep this comparison non-const from clippy's POV by binding first.
        let error_cap = MAX_ERROR_BODY_BYTES;
        let response_cap = crate::security::MAX_RESPONSE_BODY_BYTES;
        assert!(
            error_cap < response_cap,
            "error-body cap must be tighter than the general response cap"
        );
    }

    // ---- LOW (C11): non-JSON error bodies keep a diagnostic excerpt -----

    #[test]
    fn sanitize_error_body_excerpt_truncates_and_strips_controls() {
        let html = "<html>\r\n<head><title>502 Bad Gateway</title></head></html>";
        let out = sanitize_error_body_excerpt(html).expect("excerpt");
        assert!(
            !out.chars().any(char::is_control),
            "control chars must be sanitized: {out:?}"
        );
        assert!(
            out.contains("<html>"),
            "excerpt keeps the body start: {out}"
        );
        assert!(out.len() <= 201, "excerpt must be truncated: {out}");

        // Empty/whitespace-only bodies produce no excerpt.
        assert!(sanitize_error_body_excerpt("").is_none());
        assert!(sanitize_error_body_excerpt("  \n\t ").is_none());
    }

    /// LOW (C11): a non-JSON upstream error body (HTML from a gateway /
    /// reverse proxy) must surface a sanitized excerpt in the error message
    /// instead of collapsing to the generic "upstream error".
    #[tokio::test]
    async fn upstream_html_error_body_surfaces_excerpt() {
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(502)
                .header("content-type", "text/html")
                .body("<html><body><h1>502 Bad Gateway</h1><p>nginx</p></body></html>");
        });

        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings {
                    allow_private_api_base: Some(true),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let err = attempt_upstream(
            &state,
            &dep,
            &json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("502 must classify as UpstreamError");

        match err {
            UpstreamError::Retryable {
                status, message, ..
            } => {
                assert_eq!(status, 502);
                assert!(
                    message.contains("502 Bad Gateway"),
                    "non-JSON error body must surface an excerpt, got: {message}"
                );
            }
            other => panic!("expected UpstreamError::Retryable, got {other:?}"),
        }
    }

    /// LOW (C11): the SSRF private-IP rejection is a deployment-level fault
    /// (`Retryable`), not a request-level 502 — one misconfigured private
    /// `api_base` in a load-balanced group must cool that deployment and
    /// fail over to healthy siblings instead of poisoning every routed
    /// request. (Test asserts the classification; the Retryable failover
    /// machinery itself is covered by the retry-loop tests.)
    #[tokio::test]
    async fn ssrf_private_host_rejection_is_retryable() {
        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings {
                    // SSRF defense ON: private api_base hosts are rejected.
                    allow_private_api_base: Some(false),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some("http://127.0.0.1:9".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let err = attempt_upstream(
            &state,
            &dep,
            &json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("private api_base must be rejected");

        match err {
            UpstreamError::Retryable {
                status, message, ..
            } => {
                assert_eq!(status, 502, "client-facing status stays 502 on exhaustion");
                assert!(message.contains("private IP address"), "got: {message}");
            }
            other => panic!("expected UpstreamError::Retryable, got {other:?}"),
        }
    }

    // ---- MEDIUM: upstream error body must be size-bounded -----------------

    /// An error response (5xx) whose body exceeds the 10 KB error-body cap
    /// must be rejected without buffering the entire body. The capped reader
    /// stops reading as soon as the cap is exceeded (this also bounds
    /// chunked responses that omit Content-Length). We assert the rejection
    /// message rather than the exact byte count, and that the upstream
    /// message text is NOT pulled from a fully-buffered oversized body.
    #[tokio::test]
    async fn upstream_oversized_error_body_rejected_before_full_read() {
        let server = httpmock::MockServer::start();
        // Body comfortably larger than MAX_ERROR_BODY_BYTES (10 KB).
        let big_body = format!(
            "{{\"error\":{{\"message\":\"{}\"}}}}",
            "x".repeat(MAX_ERROR_BODY_BYTES * 2)
        );
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(500)
                .header("content-type", "application/json")
                .body(big_body);
        });

        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings {
                    allow_private_api_base: Some(true),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let err = attempt_upstream(
            &state,
            &dep,
            &json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("oversized error body must be rejected");

        match err {
            UpstreamError::Client {
                status, message, ..
            } => {
                assert_eq!(
                    status, 502,
                    "oversized error body must surface as a 502, not the upstream 500"
                );
                assert!(
                    message.contains("exceeds") && message.contains("limit"),
                    "rejection must cite the size limit, got: {message}"
                );
            }
            other => panic!("expected UpstreamError::Client, got {other:?}"),
        }
    }

    #[test]
    fn build_upstream_url_substitutes_wildcard_model() {
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/*/invoke".to_string();
        let out = build_upstream_url(
            url,
            Some("anthropic.claude-3-5-sonnet"),
            false,
            "bedrock",
            &json!({}),
        );
        assert_eq!(
            out,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke"
        );
    }

    // ---- HIGH (C11): wildcard substitution must be encoded / traversal-free ----

    #[test]
    fn build_upstream_url_wildcard_percent_encodes_hostile_model() {
        // `?`, `#`, spaces, `/`, and `%` must never reach the upstream URL
        // raw: against a custom `api_base` they would inject query
        // parameters, fragments, or extra path segments.
        let url = "https://gateway.example.test/model/*/invoke".to_string();
        let out = build_upstream_url(url, Some("m?x=1#f"), false, "bedrock", &json!({}));
        assert_eq!(
            out, "https://gateway.example.test/model/m%3Fx=1%23f/invoke",
            "hostile chars must be percent-encoded into one segment"
        );

        let url = "https://gateway.example.test/model/*/invoke".to_string();
        let out = build_upstream_url(url, Some("a b%2Fc"), false, "bedrock", &json!({}));
        assert_eq!(
            out, "https://gateway.example.test/model/a%20b%252Fc/invoke",
            "space and percent must be encoded (pre-encoded `%2F` must not decode to a separator)"
        );
    }

    #[test]
    fn build_upstream_url_wildcard_rejects_dot_dot_traversal() {
        // `..` segments are never substituted: the literal `*` stays in the
        // URL so the misconfiguration/attack surfaces as an upstream 404.
        let url = "https://gateway.example.test/model/*/invoke".to_string();
        for hostile in ["../../escape", "..", "a/../b", "./x"] {
            let out = build_upstream_url(url.clone(), Some(hostile), false, "bedrock", &json!({}));
            assert_eq!(
                out, url,
                "model {hostile:?} must not be substituted into the path"
            );
        }
    }

    #[test]
    fn build_upstream_url_wildcard_encodes_slashes_so_no_extra_segments() {
        let url = "https://gateway.example.test/model/*/invoke".to_string();
        let out = build_upstream_url(url, Some("a/b"), false, "bedrock", &json!({}));
        assert_eq!(
            out, "https://gateway.example.test/model/a%2Fb/invoke",
            "`/` inside the model must stay one path segment"
        );
    }

    /// e2e: a wildcard `bedrock/*` deployment must forward the substituted
    /// model as a single percent-encoded path segment — never raw
    /// client-controlled path text.
    #[tokio::test]
    async fn wildcard_url_substitution_is_encoded_on_the_wire() {
        use crate::credentials::{CredentialResolver, CredentialResolverRegistry};
        use async_trait::async_trait;
        use oxllm_core::provider::ResolvedCredentials;

        struct SigV4Resolver;
        #[async_trait]
        impl CredentialResolver for SigV4Resolver {
            async fn resolve(
                &self,
                _deployment: &Deployment,
            ) -> Result<ResolvedCredentials, oxllm_core::Error> {
                Ok(ResolvedCredentials::AwsSigV4 {
                    access_key: "AK".to_string(),
                    secret_key: "SK".to_string(),
                    session_token: None,
                    region: "us-east-1".to_string(),
                })
            }
        }

        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST").path("/model/m%3Fx=1%23f/invoke");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"model":"m?x=1#f"}"#);
        });

        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .expect("state built");
        let mut cred_registry = CredentialResolverRegistry::new();
        cred_registry.register("bedrock", std::sync::Arc::new(SigV4Resolver));
        state.credential_resolver = std::sync::Arc::new(cred_registry);

        let dep = Deployment {
            model_name: "wild".to_string(),
            oxllm_params: OxllmParams {
                model: "bedrock/*".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("bedrock provider registered");

        let response = attempt_upstream(
            &state,
            &dep,
            &json!({
                "model": "m?x=1#f",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16
            }),
            false,
            &provider,
            None,
        )
        .await
        .expect("wildcard request must reach the encoded path");

        assert_eq!(response.status(), 200);
        assert_eq!(mock.hits(), 1);
    }

    #[test]
    fn build_upstream_url_wildcard_substitution_skips_star_model() {
        // If the body still has a `*` (e.g. unconfigured wildcard), do not
        // substitute — leaving the literal `*` surfaces the misconfiguration.
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/*/invoke".to_string();
        let out = build_upstream_url(url.clone(), Some("*"), false, "bedrock", &json!({}));
        assert_eq!(out, url);
    }

    #[test]
    fn build_upstream_url_rewrites_bedrock_streaming_suffix_only_once() {
        // M2: only the trailing `/invoke` must be rewritten; an `api_base`
        // that contains `/invoke` elsewhere is left intact.
        let url = "https://gateway.example.test/invoke-proxy/model/m/invoke".to_string();
        let out = build_upstream_url(url, Some("m"), true, "bedrock", &json!({}));
        assert_eq!(
            out,
            "https://gateway.example.test/invoke-proxy/model/m/invoke-with-response-stream"
        );
    }

    #[test]
    fn build_upstream_url_rewrites_vertex_raw_predict_for_streaming() {
        // C4: Anthropic on Vertex must use :streamRawPredict when streaming.
        let url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet:rawPredict".to_string();
        let out = build_upstream_url(
            url,
            Some("claude-3-5-sonnet"),
            true,
            "vertex_ai",
            &json!({"stream": true}),
        );
        assert!(out.ends_with(":streamRawPredict"));
        assert!(!out.contains(":rawPredict:"));
    }

    #[test]
    fn build_upstream_url_keeps_vertex_raw_predict_when_not_streaming() {
        let url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet:rawPredict".to_string();
        let out = build_upstream_url(
            url.clone(),
            Some("claude-3-5-sonnet"),
            false,
            "vertex_ai",
            &json!({"stream": false}),
        );
        assert_eq!(out, url);
    }

    #[test]
    fn build_upstream_url_keeps_vertex_raw_predict_when_body_lacks_stream_flag() {
        // The Anthropic-on-Vertex transform strips `stream` from the body, so
        // streaming Claude calls route via `:rawPredict` (no rewrite). This
        // mirrors Bedrock, where `stream` is encoded in the URL instead.
        let url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/anthropic/models/claude:rawPredict".to_string();
        let out = build_upstream_url(url.clone(), Some("claude"), true, "vertex_ai", &json!({}));
        assert_eq!(out, url);
    }

    #[test]
    fn build_upstream_url_combined_wildcard_and_streaming() {
        // bedrock/* streaming: substitute wildcard, then rewrite suffix.
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/*/invoke".to_string();
        let out = build_upstream_url(
            url,
            Some("anthropic.claude-3-opus"),
            true,
            "bedrock",
            &json!({}),
        );
        assert_eq!(
            out,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-opus/invoke-with-response-stream"
        );
    }

    #[test]
    fn build_upstream_url_rewrites_converse_to_converse_stream() {
        // Converse streaming: `/converse` → `/converse-stream`.
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/converse".to_string();
        let out = build_upstream_url(
            url,
            Some("anthropic.claude-3-5-sonnet"),
            true,
            "bedrock",
            &json!({}),
        );
        assert_eq!(
            out,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/converse-stream"
        );
    }

    #[test]
    fn build_upstream_url_keeps_converse_when_not_streaming() {
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/converse".to_string();
        let out = build_upstream_url(
            url.clone(),
            Some("anthropic.claude-3-5-sonnet"),
            false,
            "bedrock",
            &json!({}),
        );
        assert_eq!(out, url);
    }

    #[test]
    fn build_upstream_url_converse_wildcard_then_streaming() {
        // bedrock/* converse streaming: substitute wildcard, then rewrite
        // `/converse` → `/converse-stream`.
        let url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/*/converse".to_string();
        let out = build_upstream_url(
            url,
            Some("meta.llama3-1-70b-instruct-v1:0"),
            true,
            "bedrock",
            &json!({}),
        );
        assert_eq!(
            out,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/meta.llama3-1-70b-instruct-v1:0/converse-stream"
        );
    }

    #[test]
    fn build_upstream_url_rewrites_watsonx_chat_to_chat_stream() {
        // WatsonX streaming: `/text/chat?version=...` → `/text/chat_stream?version=...`.
        let url =
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-03-13".to_string();
        let out = build_upstream_url(url, None, true, "watsonx", &json!({}));
        assert_eq!(
            out,
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat_stream?version=2024-03-13"
        );
    }

    #[test]
    fn build_upstream_url_keeps_watsonx_chat_when_not_streaming() {
        let url =
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-03-13".to_string();
        let out = build_upstream_url(url.clone(), None, false, "watsonx", &json!({}));
        assert_eq!(out, url);
    }

    #[test]
    fn build_upstream_url_rewrites_watsonx_deployment_chat_to_stream() {
        // Deployment-variant URL `/ml/v1/deployments/<id>/text/chat?...` is
        // also rewritten for streaming.
        let url = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/dep-1/text/chat?version=2024-03-13".to_string();
        let out = build_upstream_url(url, None, true, "watsonx", &json!({}));
        assert_eq!(
            out,
            "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/dep-1/text/chat_stream?version=2024-03-13"
        );
    }

    #[test]
    fn build_upstream_url_watsonx_idempotent_when_already_stream() {
        // Already-streaming URL is left untouched (no double `_stream_stream`).
        let url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat_stream?version=2024-03-13"
            .to_string();
        let out = build_upstream_url(url.clone(), None, true, "watsonx", &json!({}));
        assert_eq!(out, url);
    }

    // ---- U19: HTTP client must not double-count retries ---------------------
    //
    // The reference deployment's `num_retries` reached the
    // provider client, which applied its own retries on top of the router
    // retry loop. oxllm's `build_http_client` sets no retry policy, so a
    // single `execute_upstream_json` call must produce exactly ONE upstream
    // HTTP hit — never a transport-level retry fan-out.

    #[tokio::test]
    async fn u19_single_execute_upstream_json_hits_exactly_once() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(500)
                .header("content-type", "application/json")
                .body(r#"{"error":{"message":"fail"}}"#);
        });

        let state = crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings::default(),
                general_settings: crate::config::types::GeneralSettings {
                    allow_private_api_base: Some(true),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        let url = server.base_url() + "/chat/completions";
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        let result = crate::pipeline::execute_upstream_json(
            &state,
            &dep,
            axum::http::Method::POST,
            &url,
            axum::body::Bytes::from_static(b"{}"),
            reqwest::header::HeaderMap::new(),
            Some(5.0),
        )
        .await;

        assert!(result.is_ok(), "transport should return the 500 response");
        assert_eq!(
            mock.hits(),
            1,
            "U19: a single execute_upstream_json call must hit exactly once, got {}",
            mock.hits()
        );
    }

    /// A SigV4 signing failure (e.g. an unparseable `api_base`) embeds the
    /// full target URL in its message; it must be logged server-side and
    /// replaced with a generic message so no internal topology reaches the
    /// client body.
    #[tokio::test]
    async fn sigv4_signing_failure_message_is_redacted() {
        use crate::credentials::{CredentialResolver, CredentialResolverRegistry};
        use async_trait::async_trait;
        use oxllm_core::Deployment;
        use oxllm_core::provider::ResolvedCredentials;

        struct SigV4Resolver;
        #[async_trait]
        impl CredentialResolver for SigV4Resolver {
            async fn resolve(
                &self,
                _deployment: &Deployment,
            ) -> Result<ResolvedCredentials, oxllm_core::Error> {
                Ok(ResolvedCredentials::AwsSigV4 {
                    access_key: "AK".to_string(),
                    secret_key: "SK".to_string(),
                    session_token: None,
                    region: "us-east-1".to_string(),
                })
            }
        }

        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .expect("state built");
        let mut cred_registry = CredentialResolverRegistry::new();
        cred_registry.register("bedrock", std::sync::Arc::new(SigV4Resolver));
        state.credential_resolver = std::sync::Arc::new(cred_registry);

        // Unparseable api_base → `sign_request` fails on URL parsing with the
        // URL embedded in the message.
        let dep = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "bedrock/anthropic.claude-3-5-sonnet".to_string(),
                api_base: Some("not-a-url".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("bedrock provider registered");

        let err = attempt_upstream(
            &state,
            &dep,
            &json!({
                "model": "anthropic.claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16
            }),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("unparseable api_base must fail signing");

        match err {
            UpstreamError::Client {
                status, message, ..
            } => {
                assert_eq!(status, 500);
                assert_eq!(
                    message, "upstream request signing failed",
                    "sign detail must be replaced by a generic message, got: {message}"
                );
                assert!(
                    !message.contains("not-a-url"),
                    "sign detail must not leak the URL, got: {message}"
                );
            }
            other => panic!("expected UpstreamError::Client, got {other:?}"),
        }
    }

    #[test]
    fn merge_anthropic_beta_headers_deduplicates_against_existing() {
        // A deployment-supplied `anthropic-beta` (set via `extra_headers`)
        // that overlaps with a request-derived tag must not be repeated.
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "structured-outputs-2025-11-13".to_string(),
        );
        merge_anthropic_beta_headers(
            &mut headers,
            vec![
                "structured-outputs-2025-11-13".to_string(),
                "interleaved-thinking-2025-05-14".to_string(),
            ],
        );
        let h = headers["anthropic-beta"]
            .split(',')
            .map(|s| s.trim())
            .collect::<Vec<_>>();
        assert_eq!(
            h.iter()
                .filter(|t| **t == "structured-outputs-2025-11-13")
                .count(),
            1
        );
        assert!(h.contains(&"interleaved-thinking-2025-05-14"));
        assert!(h.contains(&"structured-outputs-2025-11-13"));
    }

    #[test]
    fn merge_anthropic_beta_headers_inserts_when_absent() {
        let mut headers = std::collections::HashMap::new();
        merge_anthropic_beta_headers(&mut headers, vec!["fast-mode-2025-04-18".to_string()]);
        assert_eq!(headers["anthropic-beta"], "fast-mode-2025-04-18");
    }

    #[test]
    fn merge_anthropic_beta_headers_noop_when_empty() {
        // No request-derived betas → any pre-existing deployment header is
        // preserved untouched (not overwritten, not re-joined).
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "prompt-caching-2024-07-31".to_string(),
        );
        merge_anthropic_beta_headers(&mut headers, vec![]);
        assert_eq!(headers["anthropic-beta"], "prompt-caching-2024-07-31");

        // And when there's no pre-existing header either, none is created.
        let mut headers = std::collections::HashMap::new();
        merge_anthropic_beta_headers(&mut headers, vec![]);
        assert!(!headers.contains_key("anthropic-beta"));
    }

    #[test]
    fn merge_anthropic_beta_headers_preserves_multiple_existing() {
        // Multiple deployment-supplied tags, some duplicated by the request,
        // some not — all unique tags survive, no duplicates.
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "prompt-caching-2024-07-31, structured-outputs-2025-11-13".to_string(),
        );
        merge_anthropic_beta_headers(
            &mut headers,
            vec![
                "structured-outputs-2025-11-13".to_string(),
                "web-fetch-2025-05-20".to_string(),
            ],
        );
        let tags: Vec<&str> = headers["anthropic-beta"]
            .split(',')
            .map(|s| s.trim())
            .collect();
        let mut sorted = tags.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            [
                "prompt-caching-2024-07-31",
                "structured-outputs-2025-11-13",
                "web-fetch-2025-05-20"
            ]
        );
        // No tag appears twice.
        for t in &tags {
            assert_eq!(tags.iter().filter(|x| *x == t).count(), 1, "duplicate: {t}");
        }
    }

    // ---- MEDIUM (C11): stream total-timeout vs idle-timeout ---------------

    /// A state with `allow_private_api_base` (raw-TCP test servers live on
    /// 127.0.0.1) and a helper to resolve the OpenAI provider for a
    /// deployment pointing at `api_base`.
    fn stream_timeout_state() -> crate::state::AppState {
        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built")
    }

    /// Raw TCP server that answers with SSE headers and then drips one SSE
    /// chunk every `chunk_delay_ms` for `chunks` chunks before closing —
    /// a live stream whose TOTAL duration is far longer than the configured
    /// unary `timeout`.
    fn spawn_sse_drip_server(chunk_delay_ms: u64, chunks: usize) -> String {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                std::thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
                    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
                    let mut buf = [0u8; 8192];
                    let _ = stream.read(&mut buf);
                    let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
                    if stream.write_all(head.as_bytes()).is_err() {
                        return;
                    }
                    for i in 0..chunks {
                        std::thread::sleep(std::time::Duration::from_millis(chunk_delay_ms));
                        let chunk = format!(
                            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{i}\"}}}}]}}\n\n"
                        );
                        if stream.write_all(chunk.as_bytes()).is_err() {
                            return;
                        }
                    }
                });
            }
        });
        format!("http://{addr}")
    }

    /// Raw TCP server that accepts the connection and never sends anything —
    /// an upstream that hangs before response headers.
    fn spawn_silent_server() -> String {
        use std::io::Read;
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                std::thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
                    let mut buf = [0u8; 8192];
                    // Hold the connection open, never respond.
                    let _ = stream.read(&mut buf);
                    let _ = stream.read(&mut buf);
                });
            }
        });
        format!("http://{addr}")
    }

    /// MEDIUM (C11): reqwest's per-request `.timeout()` is a TOTAL deadline
    /// that follows the body stream — applied to a streaming request it
    /// hard-kills any stream whose total duration exceeds `timeout`, even
    /// while chunks keep flowing and `stream_timeout` would allow it
    /// (the reference applies the stream timeout as a per-read idle limit only,
    /// `_get_stream_timeout`). The upstream send for `is_stream` must carry
    /// no total deadline: a drip stream (20 chunks × 100ms ≈ 2s) must drain
    /// to completion despite `timeout: 0.5`.
    #[tokio::test]
    async fn stream_body_outlives_total_request_timeout() {
        let url = spawn_sse_drip_server(100, 20);
        let state = stream_timeout_state();
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(url),
                timeout: Some(0.5),
                stream_timeout: Some(5),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let response = attempt_upstream(
            &state,
            &dep,
            &json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
            true,
            &provider,
            None,
        )
        .await
        .expect("stream headers must arrive (header wait < stream_timeout)");

        let mut stream = response.bytes_stream();
        let mut events = 0usize;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.expect("no mid-body total-timeout kill");
            events += String::from_utf8_lossy(&bytes).matches("data: ").count();
        }
        assert_eq!(
            events, 20,
            "the whole drip stream must arrive: total duration (~2s) exceeds timeout (0.5s)"
        );
    }

    /// MEDIUM (C11): removing the total deadline for streams must not leave
    /// the header wait unbounded — the send is wrapped in the same idle
    /// budget the pump uses (`stream_timeout ?? request_timeout ?? 60s`), so
    /// an upstream that never sends headers fails as a local
    /// `UpstreamError::Timeout` (retried/cooled like a 504, health-degraded)
    /// instead of hanging forever.
    #[tokio::test]
    async fn stream_header_wait_bounded_by_idle_budget() {
        let url = spawn_silent_server();
        let state = stream_timeout_state();
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(url),
                stream_timeout: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider = resolve_provider(&state, &dep).expect("openai provider registered");

        let start = std::time::Instant::now();
        let err = attempt_upstream(
            &state,
            &dep,
            &json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
            true,
            &provider,
            None,
        )
        .await
        .expect_err("silent upstream must fail the header wait");
        assert!(
            matches!(err, UpstreamError::Timeout(_)),
            "expected Timeout, got {err:?}"
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "header wait must be bounded by the ~1s idle budget, took {:?}",
            start.elapsed()
        );
    }

    /// The stream idle-budget precedence matches the reference
    /// `_get_stream_timeout`: deployment `stream_timeout` wins over
    /// `request_timeout`; `request_timeout` is used when `stream_timeout` is
    /// unset (so an operator raising it for long streams also extends the
    /// idle allowance); the 60s pump default applies only when neither is
    /// configured.
    #[test]
    fn resolve_stream_idle_timeout_precedence() {
        let mut config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let dep = |stream_timeout: Option<u64>| Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                stream_timeout,
                ..Default::default()
            },
            ..Default::default()
        };

        config.oxllm_settings.request_timeout = None;
        assert_eq!(
            resolve_stream_idle_timeout(&dep(None), &config),
            std::time::Duration::from_secs(super::super::stream::DEFAULT_STREAM_TIMEOUT_SECS),
            "neither configured → 60s default"
        );

        config.oxllm_settings.request_timeout = Some(600);
        assert_eq!(
            resolve_stream_idle_timeout(&dep(None), &config),
            std::time::Duration::from_secs(600),
            "request_timeout must extend the stream idle budget"
        );
        assert_eq!(
            resolve_stream_idle_timeout(&dep(Some(120)), &config),
            std::time::Duration::from_secs(120),
            "deployment stream_timeout wins over request_timeout"
        );
    }

    #[test]
    fn stream_idle_timeout_preserves_unary_deployment_timeouts_and_request_overrides() {
        let mut config = stream_timeout_state().config.as_ref().clone();
        config.oxllm_settings.request_timeout = Some(600);
        let mut deployment = Deployment::default();
        deployment.oxllm_params.timeout = Some(0.025);
        assert_eq!(
            resolve_stream_idle_timeout(&deployment, &config),
            std::time::Duration::from_secs(600),
            "a unary deployment timeout must not shorten streams"
        );
        config.oxllm_settings.request_timeout = None;
        assert_eq!(
            resolve_stream_idle_timeout(&deployment, &config),
            std::time::Duration::from_secs(60)
        );
        deployment.oxllm_params.stream_timeout = Some(120);
        for (request, expected) in [
            (
                json!({"timeout":0.025}),
                std::time::Duration::from_millis(25),
            ),
            (
                json!({"timeout":0.025, "stream_timeout":0.05}),
                std::time::Duration::from_millis(50),
            ),
            (
                json!({"timeout":null, "stream_timeout":null}),
                std::time::Duration::from_secs(120),
            ),
        ] {
            assert_eq!(
                resolve_stream_idle_timeout_for_request(&deployment, &config, &request),
                expected
            );
        }
    }

    // ---- MEDIUM (C11): transform failure is a deployment fault -----------

    /// A provider whose `transform_request` deterministically fails —
    /// stands in for a heterogeneous-group member that cannot map the
    /// request shape.
    struct FailingTransformProvider;

    impl oxllm_core::Provider for FailingTransformProvider {
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
    impl oxllm_core::provider::ChatProvider for FailingTransformProvider {
        fn endpoint_url(&self, _: &Deployment) -> String {
            String::new()
        }
        fn apply_auth(
            &self,
            _: &Deployment,
            _: &oxllm_core::provider::ResolvedCredentials,
            _: &mut std::collections::HashMap<String, String>,
        ) {
        }
        fn transform_request(
            &self,
            _: &serde_json::Value,
            _: &Deployment,
        ) -> oxllm_core::Result<serde_json::Value> {
            Err(oxllm_core::Error::Transform(
                "cannot map request for this provider".to_string(),
            ))
        }
        fn transform_response(
            &self,
            resp: &serde_json::Value,
            _: &str,
        ) -> oxllm_core::Result<serde_json::Value> {
            Ok(resp.clone())
        }
        fn transform_stream_chunk(&self, _: &str) -> oxllm_core::Result<Option<serde_json::Value>> {
            Ok(None)
        }
    }

    /// MEDIUM (C11): a provider `transform_request` failure is a
    /// DEPLOYMENT-level fault, not a request-level 500 — in a mixed model
    /// group a sibling with a different transform may map the same body
    /// fine. `Retryable` lets the retry loop cool this deployment and fail
    /// over (same treatment as a credential failure); previously the error
    /// was terminal `Client { 500 }` and one untransformable member failed
    /// every request routed to it.
    #[tokio::test]
    async fn transform_request_failure_is_retryable() {
        let state = stream_timeout_state();
        let dep = Deployment {
            model_name: "llm".to_string(),
            oxllm_params: OxllmParams {
                model: "stub-transform/some-model".to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider: std::sync::Arc<dyn oxllm_core::Provider> =
            std::sync::Arc::new(FailingTransformProvider);
        let err = attempt_upstream(
            &state,
            &dep,
            &json!({"model": "llm", "messages": [{"role": "user", "content": "hi"}]}),
            false,
            &provider,
            None,
        )
        .await
        .expect_err("transform must fail");
        match err {
            UpstreamError::Retryable {
                status, message, ..
            } => {
                assert_eq!(status, 500);
                assert!(
                    message.contains("cannot map request"),
                    "the transform error must surface, got: {message}"
                );
            }
            other => panic!("expected UpstreamError::Retryable, got {other:?}"),
        }
    }

    // ---- post-audit: provider BadRequest stays a non-retryable 400 --------

    /// An explicit `BadRequest` from a provider adapter is deterministic
    /// (an unsupported parameter shape), so `transform_failure` must keep it
    /// as a non-retryable client 400 rather than degrading it to a retryable
    /// 500.
    #[test]
    fn transform_failure_maps_bad_request_to_client_400() {
        let err = transform_failure(oxllm_core::Error::BadRequest(
            "unsupported parameter shape".to_string(),
        ));
        match err {
            UpstreamError::Client {
                status,
                message,
                headers,
            } => {
                assert_eq!(status, 400);
                assert_eq!(message, "unsupported parameter shape");
                assert!(headers.is_none());
            }
            other => panic!("expected UpstreamError::Client, got {other:?}"),
        }
    }

    /// Every other transform failure keeps the historical retryable
    /// classification, so one untransformable member of a mixed model group
    /// cannot fail requests a sibling could serve.
    #[test]
    fn transform_failure_maps_other_errors_to_retryable_500() {
        for error in [
            oxllm_core::Error::Transform("cannot map request".to_string()),
            oxllm_core::Error::Serialization("bad body".to_string()),
            oxllm_core::Error::Internal,
        ] {
            let expected = error.to_string();
            match transform_failure(error) {
                UpstreamError::Retryable {
                    status,
                    message,
                    headers,
                } => {
                    assert_eq!(status, 500);
                    assert_eq!(message, expected);
                    assert!(headers.is_none());
                }
                other => panic!("expected UpstreamError::Retryable, got {other:?}"),
            }
        }
    }

    // ---- post-audit: bedrock document scoping ----------------------------

    fn pdf_document_request() -> Value {
        json!({"messages":[{"role":"user","content":[
            {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}
        ]}]})
    }

    fn assert_rewritten_to_file(part: &Value) {
        assert_eq!(part["type"], "file");
        assert_eq!(
            part["file"]["file_data"],
            "data:application/pdf;base64,JVBERi0="
        );
        assert!(part.get("source").is_none());
    }

    /// Bedrock scoping: Claude on InvokeModel keeps Anthropic `document`
    /// blocks (`None` means "no rewrite"), while Claude actually routed to
    /// Converse is rewritten to the OpenAI `file` shape — the native
    /// InvokeModel transform would reject a `file` part. The explicit
    /// `use_converse_api` opt-in and the ARN auto-route must both rewrite.
    #[test]
    fn document_normalization_scopes_bedrock_rewrite_to_converse() {
        let request = pdf_document_request();
        let mut deployment = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        // Default Claude routing is InvokeModel: document blocks pass through.
        assert!(
            normalize_document_parts(&request, &deployment)
                .unwrap()
                .is_none(),
            "Claude on InvokeModel must keep native document blocks"
        );

        // Explicit Converse opt-in: rewrite to the file shape.
        deployment.oxllm_params.extra =
            serde_json::from_value(json!({"use_converse_api": true})).expect("extra map");
        let normalized = normalize_document_parts(&request, &deployment)
            .unwrap()
            .expect("Converse must rewrite the document");
        assert_rewritten_to_file(&normalized["messages"][0]["content"][0]);

        // An application-inference-profile ARN auto-routes to Converse even
        // without the explicit flag, so it must also rewrite.
        deployment.oxllm_params.extra = serde_json::Map::new();
        deployment.oxllm_params.model =
            "bedrock/anthropic.claude-3-5-sonnet:application-inference-profile/abc123".to_string();
        let normalized = normalize_document_parts(&request, &deployment)
            .unwrap()
            .expect("Converse auto-route must rewrite the document");
        assert_rewritten_to_file(&normalized["messages"][0]["content"][0]);
    }

    /// Azure follows OpenAI's file-part contract, and Claude on Vertex keeps
    /// native `document` blocks — the two provider families the scoping must
    /// not confuse.
    #[test]
    fn document_normalization_rewrites_azure_and_keeps_vertex_claude() {
        let request = pdf_document_request();
        let mut deployment = Deployment {
            model_name: "m".to_string(),
            oxllm_params: OxllmParams {
                model: "azure/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let normalized = normalize_document_parts(&request, &deployment)
            .unwrap()
            .expect("Azure must rewrite the document");
        assert_rewritten_to_file(&normalized["messages"][0]["content"][0]);

        deployment.oxllm_params.model = "vertex_ai/claude-3-5-sonnet".to_string();
        assert!(
            normalize_document_parts(&request, &deployment)
                .unwrap()
                .is_none(),
            "Claude on Vertex must keep native document blocks"
        );
    }

    /// A provider with no document contract must fail explicitly instead of
    /// silently dropping the block from the prompt or mislabelling it as an
    /// image.
    #[test]
    fn document_normalization_rejects_providers_without_document_support() {
        let request = pdf_document_request();
        for model in ["google/gemini-2.0-flash", "cohere/command-r"] {
            let deployment = Deployment {
                model_name: "m".to_string(),
                oxllm_params: OxllmParams {
                    model: model.to_string(),
                    ..Default::default()
                },
                ..Default::default()
            };
            let error = normalize_document_parts(&request, &deployment)
                .expect_err("unmappable provider must error");
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST, "{model}");
            assert!(
                error.message.contains("not supported for this provider"),
                "{model}: {}",
                error.message
            );
        }
    }

    fn state_with_global_drop(drop_params: bool) -> crate::state::AppState {
        crate::state::AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: crate::config::types::OxllmSettings {
                    drop_params: Some(drop_params),
                    ..Default::default()
                },
                general_settings: crate::config::types::GeneralSettings::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state")
    }

    fn anthropic_deployment() -> Deployment {
        Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn deployment_drop_false_keeps_param_global_drop_would_strip() {
        let chat = AnthropicProvider;
        let request = json!({
            "model": "claude",
            "messages": [],
            "max_tokens": 16,
            "frequency_penalty": 0.4,
        });
        let stripped = body_after_drop_policy(&chat, &request, true, None);
        assert!(stripped.get("frequency_penalty").is_none());
        let kept = body_after_drop_policy(&chat, &request, false, None);
        assert_eq!(kept["frequency_penalty"], 0.4);
    }

    #[test]
    fn allowed_param_survives_strip_until_additional_drop_removes_it() {
        let chat = AnthropicProvider;
        let request = json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "frequency_penalty": 0.4,
        });
        let allowed = vec!["frequency_penalty".to_string()];
        let kept = strip_unsupported_params(&chat, &request, Some(&allowed));
        assert_eq!(kept["frequency_penalty"], 0.4);
        assert_eq!(kept["temperature"], 0.2);

        let state = state_with_global_drop(true);
        let mut deployment = anthropic_deployment();
        deployment.oxllm_params.allowed_openai_params = Some(allowed);
        deployment.oxllm_params.additional_drop_params = Some(vec![
            "temperature".to_string(),
            "frequency_penalty".to_string(),
        ]);
        let out = apply_drop_params(&state, &chat, &request, &deployment)
            .expect("transform")
            .expect("shaped body");
        assert!(out.get("temperature").is_none(), "{out}");
        assert!(out.get("frequency_penalty").is_none(), "{out}");
    }

    #[test]
    fn extra_body_overrides_provider_keys_and_cannot_replace_routed_model() {
        let state = state_with_global_drop(false);
        let chat = AnthropicProvider;
        let mut deployment = anthropic_deployment();
        deployment.oxllm_params.extra_body = Some(serde_json::Map::from_iter([
            ("store".to_string(), json!(false)),
            ("temperature".to_string(), json!(0.0)),
            ("model".to_string(), json!("unrouted-model")),
        ]));
        let request = json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        });
        let out = apply_drop_params(&state, &chat, &request, &deployment)
            .expect("transform")
            .expect("shaped body");
        assert_eq!(out["store"], false);
        assert_eq!(out["temperature"], 0.0, "deployment override wins: {out}");
        assert_eq!(out["model"], "claude-3-5-sonnet");
    }

    #[test]
    fn additional_drop_removes_temperature_anthropic_would_copy() {
        let state = state_with_global_drop(false);
        let chat = AnthropicProvider;
        let mut deployment = anthropic_deployment();
        deployment.oxllm_params.additional_drop_params = Some(vec!["temperature".to_string()]);
        let request = json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
        });
        let out = apply_drop_params(&state, &chat, &request, &deployment)
            .expect("transform")
            .expect("shaped body");
        assert!(out.get("temperature").is_none(), "{out}");
    }

    #[test]
    fn allowed_params_survive_an_adapter_that_rebuilds_the_body() {
        let state = state_with_global_drop(true);
        let mut deployment = anthropic_deployment();
        deployment.oxllm_params.allowed_openai_params = Some(vec!["frequency_penalty".into()]);
        let request =
            json!({"messages": [{"role": "user", "content": "hi"}], "frequency_penalty": 0.4});
        let out = apply_drop_params(&state, &AnthropicProvider, &request, &deployment)
            .unwrap()
            .unwrap();
        assert_eq!(out["frequency_penalty"], 0.4);
    }

    #[test]
    fn allowed_known_params_keep_their_provider_native_mapping() {
        let state = state_with_global_drop(true);
        let mut deployment = anthropic_deployment();
        deployment.oxllm_params.model = "anthropic/claude-sonnet-4-20250514".into();
        deployment.oxllm_params.allowed_openai_params = Some(vec!["reasoning_effort".into()]);
        let request = json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "medium", "max_tokens": 16000});
        let out = apply_drop_params(&state, &AnthropicProvider, &request, &deployment)
            .unwrap()
            .unwrap();
        assert!(out.get("reasoning_effort").is_none());
        assert_eq!(out["thinking"]["type"], "enabled");
    }

    #[test]
    fn drops_happen_before_openai_renames_max_tokens() {
        let state = state_with_global_drop(false);
        let mut deployment = anthropic_deployment();
        deployment.oxllm_params.model = "openai/gpt-6-sol".into();
        deployment.oxllm_params.additional_drop_params = Some(vec!["max_tokens".into()]);
        let request = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 40});
        let out = apply_drop_params(&state, &OpenAiProvider, &request, &deployment)
            .unwrap()
            .unwrap();
        assert!(out.get("max_tokens").is_none());
        assert!(out.get("max_completion_tokens").is_none());
    }

    #[test]
    fn configured_privacy_override_wins_before_additional_drop() {
        let mut body = json!({"model": "routed", "store": true, "temperature": 0.5});
        let extra = serde_json::json!({"model": "other", "store": false, "temperature": 0.1});
        shape_outbound_body(&mut body, Some(&["temperature".into()]), extra.as_object());
        assert_eq!(body["model"], "routed");
        assert_eq!(body["store"], false);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn resolved_bedrock_region_is_shared_by_chat_and_embedding_urls() {
        let deployment = Deployment {
            oxllm_params: OxllmParams {
                model: "bedrock/us.cohere.embed-v4:0".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let creds = ResolvedCredentials::AwsSigV4 {
            access_key: "test".into(),
            secret_key: "secret".into(),
            session_token: None,
            region: "us-west-2".into(),
        };
        let resolved = deployment_with_resolved_region(&deployment, &creds);
        let provider = oxllm_providers::bedrock::BedrockProvider;
        assert!(
            provider
                .endpoint_url(&resolved)
                .starts_with("https://bedrock-runtime.us-west-2.amazonaws.com/")
        );
        assert!(
            provider
                .embeddings()
                .unwrap()
                .embedding_url(&resolved)
                .starts_with("https://bedrock-runtime.us-west-2.amazonaws.com/")
        );
    }
}
