//! Passthrough handlers for non-transform endpoints.
//!
//! These forward requests to the upstream provider without transformation.
//! They share the auth middleware, the shared HTTP client, and — for LLM model
//! paths routed through [`forward_with_mode`] with [`PassthroughMode::HookedRaw`]
//! — the full pre/post hook chain (rate-limit, budget, cache, cost-tracking)
//! plus per-deployment limits.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::{OriginalUri, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::pipeline::passthrough::{DEFAULT_TIMEOUT_SECS, path_within_prefix};
use crate::pipeline::{
    PassthroughMode, PassthroughRequest, extract_session_id_with_body, extract_tags,
    forward_with_mode,
};
use crate::state::SharedState;

/// Shared query-param extractor for the Tier 1 passthrough handlers. The
/// optional `model` selects the deployment when the body doesn't carry one.
#[derive(Debug, Deserialize, Default)]
pub struct ModelQuery {
    pub model: Option<String>,
}

/// `POST /v1/rerank` — passthrough with full hook chain.
///
/// Build the body for the rerank raw relay: a copy of the parsed request
/// with the `model` field rewritten from the client-facing alias to the
/// deployment's own (provider-prefix-stripped) model id. Routing already
/// resolved the alias; forwarding it verbatim sends the upstream an id it
/// does not know (e.g. Fireworks answers a generic 500 for
/// `model: "fw-rerank"` where the real id is
/// `accounts/fireworks/models/qwen3-reranker-8b`). Returns the original
/// bytes untouched when the body is not a JSON object or the model already
/// equals the deployment id.
fn rerank_relay_body(
    request_body: &serde_json::Value,
    deployment: &oxllm_core::deployment::Deployment,
    original: &Bytes,
) -> Result<Bytes, ApiError> {
    let slug = deployment.provider_slug();
    let provider_model = deployment
        .oxllm_params
        .model
        .strip_prefix(&format!("{slug}/"))
        .unwrap_or(deployment.oxllm_params.model.as_str());
    let Some(obj) = request_body.as_object() else {
        return Ok(original.clone());
    };
    if obj.get("model").and_then(|v| v.as_str()) == Some(provider_model) {
        return Ok(original.clone());
    }
    let mut relayed = request_body.clone();
    if let Some(map) = relayed.as_object_mut() {
        map.insert(
            "model".to_string(),
            serde_json::Value::String(provider_model.to_string()),
        );
    }
    serde_json::to_vec(&relayed)
        .map(Bytes::from)
        .map_err(|e| ApiError::internal(format!("failed to serialize rerank relay body: {e}")))
}

/// Known behaviour difference (L14): the two error paths return different
/// error *shapes*. When the provider does not implement the rerank transform
/// (the raw relay path), a non-success upstream status is relayed verbatim as
/// the upstream's own response body + status. When the provider DOES implement
/// the transform, a non-success status is re-shaped into a normalised
/// `ApiError::upstream(status, message)` with the message extracted from
/// `error.message`. This is intentional — the transform path exists precisely
/// to normalise vendor quirks — but clients should not assume byte-identical
/// error bodies across the two code paths.
pub async fn rerank(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let app_state = state.load_full();
    let mut request_body: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;
    crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
    crate::orchestration::request_controls::validate(&request_body)?;
    let model = request_body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'model'"))?
        .to_string();

    let tags = extract_tags(&headers, Some(&request_body));
    let mut pipeline = crate::pipeline::EndpointPipeline::for_request_with_tags(
        &app_state,
        &auth,
        &model,
        request_body.clone(),
        &tags,
    )
    .await?;
    pipeline.call_ctx.cache_namespace = Some("rerank".into());
    pipeline.apply_request_tags(&tags).await?;
    let guarded = crate::pipeline::guardrails::apply_raw_request_guardrails(
        &app_state,
        &axum::http::Method::POST,
        "rerank",
        &mut request_body,
    )
    .await?;
    let body = if guarded {
        pipeline.call_ctx.request_body = std::sync::Arc::new(request_body.clone());
        Bytes::from(request_body.to_string())
    } else {
        body
    };
    if let Some(cached) = pipeline.pre_call().await? {
        pipeline.call_ctx.cache_hit = true;
        let cost = pipeline
            .post_success(&model, None, Some(&cached), "cache")
            .await;
        let mut resp = axum::Json(cached).into_response();
        crate::pipeline::apply_cost_header(&mut resp, cost);
        crate::pipeline::set_header(&mut resp, crate::pipeline::CACHE_HIT_HEADER, "true");
        crate::pipeline::stamp_request_id_header(&mut resp, pipeline.request_id());
        return Ok(resp);
    }

    let deployment = match crate::pipeline::passthrough::resolve_deployment_with_tags(
        &app_state,
        &model,
        extract_session_id_with_body(&headers, Some(&request_body)),
        &tags,
    ) {
        Ok(d) => d,
        Err(err) => {
            // Release the budget reservation pinned in pre_call — otherwise
            // it leaks for the entire budget window (router returned no
            // deployment, e.g. all are cooling down or the model is unknown).
            // Mirrors `hooked_raw`'s release on resolve_deployment error.
            pipeline
                .post_failure(&crate::pipeline::core_error_from_api(&err))
                .await;
            return Err(err);
        }
    };
    let dep_key = oxllm_core::router::deployment_key(&deployment);
    if let Err(error) = crate::pipeline::prepare_deployment_pricing(
        &app_state,
        &mut pipeline.call_ctx,
        &deployment,
        &request_body,
    )
    .await
    {
        pipeline.post_failure(&error).await;
        return Err(ApiError::from(error));
    }
    if let Err(err) = pipeline.check_limits(&deployment, &dep_key).await {
        // Release the budget reservation pinned in pre_call — otherwise it
        // leaks for the entire budget window.
        pipeline
            .post_failure(&crate::pipeline::core_error_from_api(&err))
            .await;
        return Err(err);
    }
    let reservation = match crate::orchestration::budget::reserve_router_budgets(
        &app_state,
        &deployment,
        &dep_key,
        &pipeline.call_ctx,
    )
    .await
    {
        Ok(reservation) => reservation,
        Err(error) => {
            pipeline
                .post_failure(&crate::pipeline::core_error_from_api(&error.api_error))
                .await;
            return Err(error.api_error);
        }
    };

    let provider = app_state.registry.resolve_for_deployment(&deployment);
    let Some(rerank) = provider.and_then(|p| p.rerank()) else {
        // Raw relay: the provider exposes no rerank transform, so forward the
        // request verbatim and relay the upstream response — except the model
        // id: routing resolved the client's alias onto this deployment, and
        // the upstream rejects the alias (it expects its own model id).
        // Rewrite it to the provider-prefix-stripped deployment model,
        // mirroring what the transform branch's `transform_request` does.
        let prepared: Result<_, ApiError> = (|| {
            let body = rerank_relay_body(&request_body, &deployment, &body)?;
            let url = crate::pipeline::passthrough::merge_base_path_query(
                &crate::pipeline::upstream_base_url(&deployment)?,
                "rerank",
                None,
            )?;
            Ok((body, url))
        })();
        let (body, url) = match prepared {
            Ok(prepared) => prepared,
            Err(err) => {
                reservation.release(&app_state.cache).await;
                pipeline
                    .post_failure(&crate::pipeline::core_error_from_api(&err))
                    .await;
                return Err(err);
            }
        };
        let send_result = crate::pipeline::passthrough::send_upstream_tracked(
            &app_state,
            &deployment,
            axum::http::Method::POST,
            &url,
            body,
            Some("application/json"),
            &[],
            Some(&mut pipeline.call_ctx),
        )
        .await;
        let (status, mut resp, usage, upstream_cost) = match send_result {
            Ok(ok) => ok,
            Err(e) => {
                reservation.release(&app_state.cache).await;
                // Route the failure through the shared ApiError→core-Error
                // mapping so gateway-synthesized transport failures
                // (`upstream_unreachable` / `upstream_timeout`) surface as
                // `Error::Network` and the rate-limit hook releases the key
                // RPM/TPM reservations — the request never reached a
                // provider. Mapping everything to `Error::Upstream` here
                // diverged from `hooked_raw` and the transform branch below.
                pipeline
                    .post_failure(&crate::pipeline::core_error_from_api(&e))
                    .await;
                return Err(e);
            }
        };
        if status.is_success() {
            // U3: honour the gateway response-cost contract on the rerank
            // passthrough surface too — when the upstream reported its own
            // cost, that value is authoritative.
            pipeline.call_ctx.cost_override = upstream_cost;
            let response_body =
                crate::pipeline::passthrough::recover_json_response(&mut resp).await;
            let usage = usage.or_else(|| {
                response_body
                    .as_ref()
                    .and_then(crate::orchestration::usage::extract_usage)
            });
            let cost = pipeline
                .post_success(
                    &model,
                    usage.as_ref(),
                    response_body.as_ref(),
                    deployment.provider_slug(),
                )
                .await;
            crate::orchestration::budget::settle_router_spend(
                &app_state,
                &deployment,
                &dep_key,
                cost,
                &reservation,
            )
            .await;
            crate::pipeline::apply_cost_header(&mut resp, cost);
            if let Some(u) = usage.as_ref() {
                crate::pipeline::record_deployment_tpm(&app_state, &dep_key, u.total_tokens).await;
            }
        } else {
            // Surface the upstream's own error message (best-effort) so the
            // failure hook can classify the rejection instead of seeing a
            // generic label. The relayed response body is restored verbatim.
            reservation.release(&app_state.cache).await;
            pipeline
                .post_failure(&oxllm_core::Error::Upstream {
                    status: status.as_u16(),
                    message: extract_relayed_error_message(&mut resp).await,
                })
                .await;
        }
        crate::pipeline::stamp_request_id_header(&mut resp, pipeline.request_id());
        crate::pipeline::stamp_ratelimit_headers(&mut resp, &app_state, &pipeline.call_ctx).await;
        return Ok(resp);
    };

    // Transform branch: the provider owns a rerank transform. Every failure
    // after `pre_call` is funneled through the single release point below so
    // the budget reservation pinned in `pre_call` cannot leak for the budget
    // window (transform_request, credential resolution, serialization,
    // transport, body limits, response parse — all release on `Err`).
    let outcome: Result<Response, ApiError> = async {
        let mut provider_request = request_body.clone();
        crate::orchestration::request_controls::strip_before_transform(&mut provider_request);
        let mut transformed = rerank
            .transform_request(&provider_request, &deployment)
            .map_err(ApiError::from)?;
        crate::orchestration::request_controls::strip(&mut transformed);
        let creds =
            crate::pipeline::passthrough::resolve_credentials(&app_state, &deployment).await?;
        let endpoint_deployment =
            crate::orchestration::upstream::deployment_with_resolved_region(&deployment, &creds);
        let auth_headers =
            crate::pipeline::passthrough::provider_auth_headers(provider, &deployment, &creds)?;
        let mut req_headers = reqwest::header::HeaderMap::new();
        req_headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        for (key, value) in &auth_headers {
            let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|_| ApiError::internal("invalid configured upstream header"))?;
            let value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|_| ApiError::internal("invalid configured upstream header"))?;
            req_headers.insert(name, value);
        }
        let timeout_secs = deployment
            .oxllm_params
            .timeout
            .unwrap_or(DEFAULT_TIMEOUT_SECS as f64);
        pipeline.call_ctx.upstream_start_time = Some(std::time::Instant::now());
        let response = crate::pipeline::execute_upstream_json(
            &app_state,
            &deployment,
            axum::http::Method::POST,
            &rerank.rerank_url(&endpoint_deployment),
            axum::body::Bytes::from(serde_json::to_vec(&transformed).map_err(|e| {
                ApiError::internal(format!("failed to serialize rerank request: {e}"))
            })?),
            req_headers,
            Some(timeout_secs),
        )
        .await?;
        pipeline.call_ctx.reached_upstream = true;
        let status = response.status();
        // Capture the upstream headers BEFORE reading the body so a non-success
        // status can forward the debugging allowlist (`x-request-id`,
        // `retry-after`, `x-ratelimit-*`) — mirroring the chat error path.
        let headers = response.headers().clone();
        // Stream the upstream body through a mid-stream cap so a malicious or
        // buggy rerank upstream that streams a multi-GB body via chunked
        // transfer (no Content-Length) is rejected after roughly the cap
        // instead of being fully buffered in proxy heap (OOM vector). The
        // raw-relay branch gets this via `sanitize_upstream_response`; the
        // transform branch reads the body explicitly and must bound it too.
        // `read_bounded_bytes` performs the Content-Length pre-check and the
        // post-read limit internally, so the explicit calls are subsumed.
        let bytes =
            crate::security::read_bounded_bytes(response, crate::security::MAX_RESPONSE_BODY_BYTES)
                .await?;
        let body_text = String::from_utf8_lossy(&bytes).to_string();
        if !status.is_success() {
            let body: serde_json::Value =
                serde_json::from_str(&body_text).unwrap_or(serde_json::Value::Null);
            let message = body
                .get("error")
                .and_then(|e| {
                    e.get("message")
                        .and_then(|m| m.as_str())
                        .or_else(|| e.as_str())
                })
                .unwrap_or("rerank upstream error");
            return Err(ApiError::upstream(status.as_u16(), message).with_headers(Some(headers)));
        }
        let body: serde_json::Value = serde_json::from_str(&body_text).map_err(|e| {
            tracing::error!(
                error.kind = "serialization",
                detail = %e,
                "failed to parse rerank response as JSON"
            );
            ApiError::internal("failed to parse rerank response as JSON")
        })?;
        let transformed = rerank
            .transform_response(&body, &deployment.oxllm_params.model)
            .map_err(ApiError::from)?;
        // U3: extract cost/usage from the captured upstream headers (the
        // reqwest::Response was already consumed by `response.bytes()`) so
        // rerank providers that own a transform track spend and charge the
        // per-deployment TPM window — matching the raw-relay branch. Without
        // this, transform-mode rerank tracked zero spend and bypassed the
        // `tpm` cap.
        let usage = crate::pipeline::extract_upstream_usage(&headers)
            .or_else(|| crate::orchestration::usage::extract_usage(&transformed));
        let upstream_cost =
            crate::pipeline::extract_upstream_response_cost(&headers).into_override();
        pipeline.call_ctx.cost_override = upstream_cost;
        let cost = pipeline
            .post_success(
                &model,
                usage.as_ref(),
                Some(&transformed),
                deployment.provider_slug(),
            )
            .await;
        crate::orchestration::budget::settle_router_spend(
            &app_state,
            &deployment,
            &dep_key,
            cost,
            &reservation,
        )
        .await;
        if let Some(u) = usage.as_ref() {
            crate::pipeline::record_deployment_tpm(&app_state, &dep_key, u.total_tokens).await;
        }
        let mut resp = axum::Json(transformed).into_response();
        crate::pipeline::apply_cost_header(&mut resp, cost);
        crate::pipeline::stamp_request_id_header(&mut resp, pipeline.request_id());
        crate::pipeline::stamp_ratelimit_headers(&mut resp, &app_state, &pipeline.call_ctx).await;
        Ok(resp)
    }
    .await;

    match outcome {
        Ok(resp) => Ok(resp),
        Err(err) => {
            reservation.release(&app_state.cache).await;
            pipeline
                .post_failure(&crate::pipeline::core_error_from_api(&err))
                .await;
            Err(err)
        }
    }
}

/// `POST /v1/images/generations` — passthrough with full hook chain.
pub async fn images_generations(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let model = extract_model(&body);
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "images/generations".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            model.as_deref(),
            None,
            body,
            Some("application/json"),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/images/edits` — JSON raw relay. Multipart inference is rejected
/// until a parser can validate and rewrite its model field safely.
pub async fn images_edits(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("multipart/form-data");
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "images/edits".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            None,
            None,
            body,
            Some(content_type),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/images/variations` — raw relay with explicit multipart rejection.
pub async fn images_variations(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("multipart/form-data");
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "images/variations".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            None,
            None,
            body,
            Some(content_type),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/audio/transcriptions` — raw relay with explicit multipart rejection.
pub async fn audio_transcriptions(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("multipart/form-data");
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "audio/transcriptions".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            None,
            None,
            body,
            Some(content_type),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/audio/translations` — same routing validation as transcriptions.
pub async fn audio_translations(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("multipart/form-data");
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "audio/translations".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            None,
            None,
            body,
            Some(content_type),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/audio/speech` — forward JSON for TTS.
pub async fn audio_speech(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let model = extract_model(&body);
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "audio/speech".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            model.as_deref(),
            None,
            body,
            Some("application/json"),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/moderations` — passthrough with full hook chain.
pub async fn moderations(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let model = extract_model(&body);
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "moderations".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            model.as_deref(),
            None,
            body,
            Some("application/json"),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// Generic provider passthrough: `/anthropic/v1/messages` → Anthropic.
///
/// Runs the full [`crate::pipeline::EndpointPipeline`] (rate-limit, budget, spend, deployment
/// RPM/TPM) so catch-all paths cannot bypass metering that `/v1/*` enforces.
pub async fn provider_passthrough(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original_uri): OriginalUri,
    axum::extract::Path(path): axum::extract::Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: Bytes,
) -> Result<Response, ApiError> {
    let app_state = state.load_full();

    let (provider_slug, remaining_path) = parse_passthrough_path(&path);
    if provider_slug == "v1" || app_state.registry.get(&provider_slug).is_none() {
        return Err(ApiError::not_found(
            "unknown API route; provider passthrough requires a registered provider prefix",
        ));
    }

    // Parse the body as JSON only when the content-type is JSON-ish (and the
    // body is non-empty). A request that claims `application/json` but fails
    // to parse is rejected with a clear 400 rather than silently degrading to
    // `Value::Null` (which would then drive routing with no body context).
    // Non-JSON bodies (multipart, empty GET) stay `Null`.
    let content_type_hdr = headers.get("content-type").and_then(|v| v.to_str().ok());
    let treat_as_json = content_type_hdr.map(content_type_is_json).unwrap_or(true);
    let multipart_model = if treat_as_json {
        None
    } else {
        crate::pipeline::multipart::extract_field(&body, content_type_hdr, "model")
            .filter(|model| !model.trim().is_empty())
    };
    crate::pipeline::passthrough::validate_raw_model_carrier(
        &remaining_path,
        content_type_hdr,
        multipart_model.as_deref(),
    )?;
    let mut request_body: serde_json::Value = if treat_as_json && !body.is_empty() {
        serde_json::from_slice(&body)
            .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?
    } else {
        serde_json::Value::Null
    };
    if !request_body.is_null() {
        crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
        crate::orchestration::request_controls::validate(&request_body)?;
    }

    let mut query_models =
        url::form_urlencoded::parse(original_uri.query().unwrap_or("").as_bytes())
            .filter(|(key, _)| key == "model")
            .map(|(_, model)| model.into_owned());
    let query_model = query_models.next();
    if query_models.next().is_some() {
        return Err(ApiError::bad_request("ambiguous model query parameter"));
    }
    let body_model = request_body
        .get("model")
        .map(|model| {
            model
                .as_str()
                .filter(|model| !model.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| ApiError::bad_request("field 'model' must be a non-empty string"))
        })
        .transpose()?;
    if body_model
        .as_ref()
        .zip(query_model.as_ref())
        .is_some_and(|(body, query)| body != query)
    {
        return Err(ApiError::bad_request(
            "conflicting body and query model selectors",
        ));
    }
    let requested_model = body_model
        .or(query_model)
        .or_else(|| multipart_model.clone());
    crate::pipeline::passthrough::require_inference_model(
        &remaining_path,
        requested_model.as_deref(),
        &request_body,
    )?;
    let inserted_selector = if treat_as_json
        && crate::pipeline::passthrough::inference_path(&remaining_path)
        && request_body.get("model").is_none()
    {
        if let Some(selector) = requested_model.as_deref() {
            request_body
                .as_object_mut()
                .ok_or_else(|| ApiError::bad_request("inference request body must be an object"))?
                .insert("model".into(), serde_json::json!(selector));
            true
        } else {
            false
        }
    } else {
        false
    };
    if let Some(model) = requested_model.as_deref() {
        crate::pipeline::check_request_model_acl(&app_state, &auth, model, &request_body, &headers)
            .await?;
    }
    let model = crate::pipeline::passthrough::resolve_provider_model(
        &app_state,
        &provider_slug,
        requested_model.as_deref(),
    )?;

    let tags = extract_tags(&headers, Some(&request_body));
    let mut pipeline = crate::pipeline::EndpointPipeline::for_request_with_tags(
        &app_state,
        &auth,
        &model,
        request_body.clone(),
        &tags,
    )
    .await?;
    if let Err(error) =
        crate::pipeline::passthrough::validate_nested_model_acl(&auth, &request_body)
    {
        crate::pipeline::record_request_rejection(&app_state, &pipeline.call_ctx, &error).await;
        return Err(error);
    }
    pipeline.call_ctx.cache_namespace = Some(format!("raw:{method}:{remaining_path}"));
    let session_id = extract_session_id_with_body(&headers, Some(&request_body));
    pipeline.apply_request_tags(&tags).await?;
    let guarded = crate::pipeline::guardrails::apply_raw_request_guardrails(
        &app_state,
        &method,
        &remaining_path,
        &mut request_body,
    )
    .await?;
    let body = if guarded || inserted_selector {
        pipeline.call_ctx.request_body = std::sync::Arc::new(request_body.clone());
        Bytes::from(request_body.to_string())
    } else {
        body
    };
    crate::pipeline::passthrough::validate_file_namespace(
        &app_state,
        &remaining_path,
        &model,
        Some(&provider_slug),
    )?;

    if let Some(cached) = pipeline.pre_call().await? {
        pipeline.call_ctx.cache_hit = true;
        let cost = pipeline
            .post_success(&model, None, Some(&cached), "cache")
            .await;
        let mut resp = axum::Json(cached).into_response();
        crate::pipeline::apply_cost_header(&mut resp, cost);
        crate::pipeline::set_header(&mut resp, crate::pipeline::CACHE_HIT_HEADER, "true");
        crate::pipeline::stamp_request_id_header(&mut resp, pipeline.request_id());
        return Ok(resp);
    }

    // Every post-`pre_call` failure is funneled through the single release
    // point below so the budget reservation pinned in `pre_call` cannot leak
    // for the budget window. This mirrors `hooked_raw`: deployment
    // resolution, limit checks, URL build/confinement, credential
    // resolution, transport, and response sanitisation all release on `Err`.
    let mut router_reservation = None;
    let outcome: Result<Response, ApiError> = async {
        // Native IDs were resolved to a configured group above. A failed
        // router pick (including cooldown/tag filtering) must stay a failure;
        // falling back to the first provider credential bypasses that policy.
        let deployment = crate::pipeline::passthrough::resolve_deployment_with_tags(
            &app_state, &model, session_id, &tags,
        )?;
        if deployment.provider_slug() != provider_slug {
            return Err(ApiError::bad_request(
                "routing selected a different provider for this provider-prefixed endpoint",
            ));
        }
        crate::pipeline::passthrough::validate_raw_endpoint(
            &app_state,
            &deployment,
            &remaining_path,
            Some(&provider_slug),
        )?;
        crate::pipeline::passthrough::validate_selected_file_deployment(
            &app_state,
            &remaining_path,
            &model,
            &deployment,
        )?;
        let dep_key = oxllm_core::router::deployment_key(&deployment);
        crate::pipeline::prepare_deployment_pricing(
            &app_state,
            &mut pipeline.call_ctx,
            &deployment,
            &request_body,
        )
        .await
        .map_err(ApiError::from)?;
        pipeline.check_limits(&deployment, &dep_key).await?;

        let base = crate::pipeline::catch_all_upstream_base_url(&deployment)?;
        let azure_native = deployment.provider_slug() == "azure"
            && matches!(
                crate::pipeline::passthrough::endpoint_path(&remaining_path),
                "images/generations"
                    | "images/edits"
                    | "images/variations"
                    | "audio/speech"
                    | "audio/transcriptions"
                    | "audio/translations"
                    | "moderations"
            );
        let merged = if azure_native {
            crate::pipeline::passthrough::build_native_upstream_url(
                &deployment,
                &remaining_path,
                original_uri.query(),
                requested_model.as_deref(),
            )?
        } else {
            crate::pipeline::passthrough::merge_base_path_query(
                &base,
                &remaining_path,
                original_uri.query(),
            )?
        };
        // Path confinement: normalise dot-segments and reject paths whose
        // normalised form escapes the deployment's configured base path, so
        // an authenticated client cannot reach arbitrary endpoints on the
        // upstream (e.g. account-management routes) via the catch-all using
        // the gateway's pooled credential.
        let url = if azure_native {
            merged
        } else {
            confine_upstream_url(&base, &merged)?
        };

        let content_type = content_type_hdr.unwrap_or("application/json");
        router_reservation = Some(
            crate::orchestration::budget::reserve_router_budgets(
                &app_state,
                &deployment,
                &dep_key,
                &pipeline.call_ctx,
            )
            .await
            .map_err(|error| error.api_error)?,
        );

        let (status, mut sanitized, header_usage, upstream_cost) =
            crate::pipeline::passthrough::send_upstream_tracked(
                &app_state,
                &deployment,
                method,
                &url,
                if treat_as_json {
                    crate::pipeline::passthrough::rewrite_model_in_body(
                        body,
                        &deployment,
                        crate::pipeline::passthrough::inference_path(&remaining_path),
                    )?
                } else if multipart_model.is_some() {
                    // Multipart inference (images/edits, audio/*): replace the
                    // alias in the form field with the deployment's upstream
                    // model id, preserving every other byte of the body.
                    let upstream_model = crate::pipeline::passthrough::resolve_raw_model(
                        &deployment,
                        multipart_model.as_deref(),
                    )?;
                    crate::pipeline::multipart::rewrite_field(
                        &body,
                        content_type_hdr,
                        "model",
                        &upstream_model,
                    )
                    .ok_or_else(|| {
                        ApiError::bad_request(
                            "multipart model field could not be located for rewrite; \
                         resend with a model field or use the JSON transport",
                        )
                    })?
                } else {
                    body
                },
                Some(content_type),
                &crate::pipeline::passthrough::vetted_client_headers(&headers),
                Some(&mut pipeline.call_ctx),
            )
            .await?;
        if status.is_success() {
            pipeline.call_ctx.cost_override = upstream_cost;
            if crate::pipeline::passthrough::response_is_sse(&sanitized) {
                crate::pipeline::stamp_request_id_header(&mut sanitized, pipeline.request_id());
                crate::pipeline::stamp_ratelimit_headers(
                    &mut sanitized,
                    &app_state,
                    &pipeline.call_ctx,
                )
                .await;
                return Ok(crate::pipeline::passthrough::meter_native_stream(
                    &app_state,
                    deployment,
                    pipeline.call_ctx.clone(),
                    sanitized,
                    router_reservation.take().unwrap_or_default(),
                ));
            }
            // Prefer header-reported totals; otherwise recover native JSON
            // usage. Native SSE is metered by the byte-preserving relay above.
            let response_body =
                crate::pipeline::passthrough::recover_json_response(&mut sanitized).await;
            let body_usage = response_body
                .as_ref()
                .and_then(crate::orchestration::usage::extract_usage);
            let usage = header_usage.or(body_usage);
            let cost = pipeline
                .post_success(
                    response_body
                        .as_ref()
                        .and_then(|body| body.get("model"))
                        .and_then(serde_json::Value::as_str)
                        .filter(|model| !model.trim().is_empty())
                        .unwrap_or(&model),
                    usage.as_ref(),
                    response_body.as_ref(),
                    deployment.provider_slug(),
                )
                .await;
            crate::orchestration::budget::settle_router_spend(
                &app_state,
                &deployment,
                &dep_key,
                cost,
                router_reservation.as_ref().unwrap_or(&Default::default()),
            )
            .await;
            crate::pipeline::apply_cost_header(&mut sanitized, cost);
            if let Some(u) = usage.as_ref() {
                crate::pipeline::record_deployment_tpm(&app_state, &dep_key, u.total_tokens).await;
            }
        } else {
            if let Some(reservation) = router_reservation.take() {
                reservation.release(&app_state.cache).await;
            }
            pipeline
                .post_failure(&oxllm_core::Error::Upstream {
                    status: status.as_u16(),
                    message: "passthrough upstream error".to_string(),
                })
                .await;
        }
        // Stamp the standard observability headers on every non-cache-hit
        // catch-all response (success and relayed-error alike) so clients can
        // correlate the response with the request id (H8) and read the live
        // RPM/TPM counters for client-side throttling — matching
        // `hooked_raw`. `sanitize_upstream_response` strips the upstream's own
        // `x-request-id`, so without this the catch-all would return a
        // response with no correlation header at all.
        crate::pipeline::stamp_request_id_header(&mut sanitized, pipeline.request_id());
        crate::pipeline::stamp_ratelimit_headers(&mut sanitized, &app_state, &pipeline.call_ctx)
            .await;
        Ok(sanitized)
    }
    .await;

    match outcome {
        Ok(resp) => Ok(resp),
        Err(err) => {
            if let Some(reservation) = router_reservation {
                reservation.release(&app_state.cache).await;
            }
            pipeline
                .post_failure(&crate::pipeline::core_error_from_api(&err))
                .await;
            Err(err)
        }
    }
}

fn parse_passthrough_path(path: &str) -> (String, String) {
    let trimmed = path.trim_start_matches('/');
    if let Some((provider, rest)) = trimmed.split_once('/') {
        (provider.to_string(), rest.to_string())
    } else {
        (trimmed.to_string(), String::new())
    }
}

fn extract_model(body: &Bytes) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    value
        .get("model")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Whether a content-type value should be treated as a single JSON document
/// for body parsing. Mirrors the gating in
/// `pipeline::passthrough::is_json_content_type` — including the deliberate
/// exclusion of `application/x-ndjson`: JSONL cannot be parsed by
/// `serde_json::from_slice`, so an NDJSON body must be forwarded verbatim
/// rather than rejected with a spurious 400.
fn content_type_is_json(ct: &str) -> bool {
    crate::pipeline::passthrough::is_json_content_type(Some(ct))
}

/// Normalise dot-segments in the merged upstream URL and confine the result to
/// the deployment's configured base path. `url::Url::parse` collapses `.` and
/// `..` segments per RFC 3986; the prefix check then rejects paths that escape
/// the base (e.g. a client sending `files/../../users` against an OpenAI base
/// of `https://api.openai.com/v1`). Returns the normalised URL on success.
fn confine_upstream_url(base: &str, merged: &str) -> Result<String, ApiError> {
    let parsed_target = url::Url::parse(base)
        .map_err(|e| ApiError::internal(format!("invalid upstream base URL '{base}': {e}")))?;
    let parsed_final = url::Url::parse(merged)
        .map_err(|e| ApiError::bad_request(format!("invalid inbound path for catch-all: {e}")))?;
    if !path_within_prefix(parsed_final.path(), parsed_target.path()) {
        return Err(ApiError::forbidden(
            "catch-all path escapes the configured upstream prefix",
        ));
    }
    Ok(parsed_final.to_string())
}

/// Best-effort extract an upstream error message from an already-sanitised
/// relayed response (errors are buffered and ≤500 chars). The body is read,
/// inspected, then restored verbatim so the relayed response is unchanged.
/// Falls back to a generic label when the body is not JSON or has no
/// `error.message` field. When the body cannot be read back (e.g. it exceeds
/// the bounded read limit — near-impossible for an already-sanitised body but
/// defended against for correctness), the relayed body is unrecoverable and
/// left empty rather than silently keeping a half-read state.
async fn extract_relayed_error_message(resp: &mut Response) -> String {
    const FALLBACK: &str = "rerank upstream error";
    let body = std::mem::replace(resp.body_mut(), axum::body::Body::empty());
    let bytes = match axum::body::to_bytes(body, crate::security::MAX_RESPONSE_BODY_BYTES).await {
        Ok(b) => b,
        Err(_) => {
            return FALLBACK.to_string();
        }
    };
    let message = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| v.get("error").cloned())
        .and_then(|e| {
            e.get("message")
                .and_then(|m| m.as_str().map(str::to_string))
                .or_else(|| e.as_str().map(str::to_string))
        });
    *resp.body_mut() = axum::body::Body::from(bytes);
    message.unwrap_or_else(|| FALLBACK.to_string())
}

/// `GET /metrics` — Prometheus metrics endpoint.
pub async fn metrics_handler() -> impl IntoResponse {
    let body = crate::metrics::render();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_passthrough_path_splits_provider_and_rest() {
        let (provider, rest) = parse_passthrough_path("anthropic/v1/messages");
        assert_eq!(provider, "anthropic");
        assert_eq!(rest, "v1/messages");
    }

    #[test]
    fn parse_passthrough_path_handles_leading_slash() {
        let (provider, rest) = parse_passthrough_path("/openai/v1/chat/completions");
        assert_eq!(provider, "openai");
        assert_eq!(rest, "v1/chat/completions");
    }

    #[test]
    fn parse_passthrough_path_no_slash() {
        let (provider, rest) = parse_passthrough_path("openai");
        assert_eq!(provider, "openai");
        assert_eq!(rest, "");
    }

    // ---- shared test builders -------------------------------------------------

    use crate::auth::AuthUser;
    use crate::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
    use crate::hooks::BudgetHook;
    use crate::hooks::spend_keys::spend_key;
    use crate::pipeline::UPSTREAM_TOTAL_TOKENS_HEADER;
    use crate::state::AppState;
    use oxllm_core::deployment::OxllmParams;
    use oxllm_core::provider::{Provider, ProviderId, RerankProvider, ResolvedCredentials};
    use oxllm_core::{Deployment, Endpoint};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    fn shared_from(state: AppState) -> SharedState {
        Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    fn auth_with_budget() -> AuthUser {
        AuthUser {
            max_budget: Some(1.0),
            ..AuthUser::open()
        }
    }

    /// Build a single-deployment state. When `with_budget` is set a small
    /// `BudgetHook` is installed so `pre_call` reserves a spend estimate that
    /// must be released on failure paths.
    fn build_state(
        model_name: &str,
        litellm_model: &str,
        api_base: Option<String>,
        with_budget: bool,
    ) -> AppState {
        let model_list = vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: litellm_model.to_string(),
                api_key: Some("sk-test".to_string()),
                api_base,
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut state =
            AppState::with_registry(config, oxllm_providers::default_registry()).expect("state");
        if with_budget {
            let budget = BudgetHook::with_duration(
                state.cache.clone(),
                Some(1.0),
                Duration::from_secs(3600),
            );
            state.hooks = vec![Arc::new(budget)];
        }
        state
    }

    // ---- MEDIUM 7: catch-all path confinement --------------------------------

    #[test]
    fn confine_upstream_url_accepts_path_within_prefix() {
        let url = confine_upstream_url(
            "https://api.example.com/v1",
            "https://api.example.com/v1/files",
        )
        .expect("within prefix");
        assert_eq!(url, "https://api.example.com/v1/files");
    }

    #[test]
    fn confine_upstream_url_rejects_dotdot_escape() {
        // `/v1/files/../../users` normalises to `/users`, which escapes the
        // `/v1` base prefix — must be rejected.
        let err = confine_upstream_url(
            "https://api.example.com/v1",
            "https://api.example.com/v1/files/../../users",
        )
        .expect_err("escaped path must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
    }

    #[test]
    fn confine_upstream_url_accepts_anything_when_base_has_no_path() {
        // A bare host has no prefix to escape, so any suffix is allowed.
        let url = confine_upstream_url(
            "https://api.example.com",
            "https://api.example.com/anything/here",
        )
        .expect("no prefix to escape");
        assert_eq!(url, "https://api.example.com/anything/here");
    }

    // ---- HIGH: budget reservation release on post-pre_call failures ---------

    /// A rerank provider whose `transform_request` always fails, so the rerank
    /// transform branch errors after `pre_call` has reserved the budget.
    struct FailingRerank;

    impl RerankProvider for FailingRerank {
        fn rerank_url(&self, _deployment: &Deployment) -> String {
            String::new()
        }
        fn apply_auth(
            &self,
            _deployment: &Deployment,
            _creds: &ResolvedCredentials,
            _headers: &mut HashMap<String, String>,
        ) {
        }
        fn transform_request(
            &self,
            _request: &serde_json::Value,
            _deployment: &Deployment,
        ) -> oxllm_core::error::Result<serde_json::Value> {
            Err(oxllm_core::Error::Transform(
                "intentional transform_request failure".to_string(),
            ))
        }
        fn transform_response(
            &self,
            body: &serde_json::Value,
            _model: &str,
        ) -> oxllm_core::error::Result<serde_json::Value> {
            Ok(body.clone())
        }
    }

    impl Provider for FailingRerank {
        fn id(&self) -> ProviderId {
            "myrerank"
        }
        fn supports(&self, endpoint: Endpoint) -> bool {
            matches!(endpoint, Endpoint::Rerank)
        }
        fn rerank(&self) -> Option<&dyn RerankProvider> {
            Some(self)
        }
    }

    fn state_with_failing_rerank() -> AppState {
        let model_list = vec![ConfigModel {
            model_name: "rerank-model".to_string(),
            oxllm_params: OxllmParams {
                model: "myrerank/rerank-1".to_string(),
                api_key: Some("sk-test".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut registry = oxllm_providers::default_registry();
        registry.register("myrerank", Arc::new(FailingRerank) as Arc<dyn Provider>);
        let mut state = AppState::with_registry(config, registry).expect("state built");
        let budget =
            BudgetHook::with_duration(state.cache.clone(), Some(1.0), Duration::from_secs(3600));
        state.hooks = vec![Arc::new(budget)];
        state
    }

    #[tokio::test]
    async fn rerank_transform_request_failure_releases_budget() {
        let state = state_with_failing_rerank();
        let shared = shared_from(state.clone());
        let auth = auth_with_budget();
        let spend_key = spend_key(auth.key_hash_hex());

        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let result = rerank(
            State(shared),
            Extension(auth),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await;
        assert!(
            result.is_err(),
            "transform_request failure must surface an error"
        );

        let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "budget reservation must be released after transform_request failure, got {spend}"
        );
    }

    /// `resolve_deployment` failure (no deployment serves the requested model,
    /// or all candidates are cooling down) happens AFTER `pre_call` reserved
    /// the budget estimate. The reservation must be released through
    /// `post_failure` rather than leaking for the whole budget window — the
    /// same discipline `hooked_raw` applies and that the transform branch's
    /// `outcome` block already enforces.
    #[tokio::test]
    async fn rerank_resolve_deployment_failure_releases_budget() {
        let state = state_with_failing_rerank();
        let shared = shared_from(state.clone());
        let auth = auth_with_budget();
        let spend_key = spend_key(auth.key_hash_hex());

        // A model that matches no deployment so the router returns no index.
        let body = serde_json::json!({
            "model": "no-such-rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let result = rerank(
            State(shared),
            Extension(auth),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await;
        assert!(
            result.is_err(),
            "resolve_deployment failure must surface an error"
        );

        let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "budget reservation must be released after resolve_deployment failure, got {spend}"
        );
    }

    /// The rerank transform branch must reject an oversized upstream response
    /// through the bounded read and release the budget reservation, mirroring
    /// the catch-all oversize path. The mid-stream chunked-body case is
    /// exercised by `read_bounded_bytes`' own unit tests; this integration
    /// test pins the handler to the bounded-read path so a future change
    /// cannot silently regress to `response.bytes()`.
    #[tokio::test]
    async fn rerank_transform_oversize_upstream_releases_budget() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/rerank");
            then.status(200)
                .header("content-type", "application/json")
                .header(
                    "content-length",
                    format!("{}", crate::security::MAX_RESPONSE_BODY_BYTES + 1),
                )
                .body("{}");
        });

        let model_list = vec![ConfigModel {
            model_name: "rerank-model".to_string(),
            oxllm_params: OxllmParams {
                model: "mockrerank/rerank-1".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut registry = oxllm_providers::default_registry();
        registry.register("mockrerank", Arc::new(MockRerank) as Arc<dyn Provider>);
        let mut state = AppState::with_registry(config, registry).expect("state built");
        let budget =
            BudgetHook::with_duration(state.cache.clone(), Some(1.0), Duration::from_secs(3600));
        state.hooks = vec![Arc::new(budget)];
        let shared = shared_from(state.clone());
        let auth = auth_with_budget();
        let spend_key = spend_key(auth.key_hash_hex());

        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let result = rerank(
            State(shared),
            Extension(auth),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await;
        assert!(
            result.is_err(),
            "oversized rerank upstream must be rejected by the bounded read"
        );

        let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "budget reservation must be released after oversized rerank upstream, got {spend}"
        );
    }

    /// A rerank deployment whose per-deployment spend counter is already at
    /// its `max_budget` cap must be rejected before the upstream call — the
    /// rerank surface honours the same read-only router-budget gate as
    /// `hooked_raw`.
    #[tokio::test]
    async fn rerank_router_budget_gate_rejects_at_cap() {
        let model_list = vec![ConfigModel {
            model_name: "rerank-model".to_string(),
            oxllm_params: OxllmParams {
                model: "mockrerank/rerank-1".to_string(),
                api_key: Some("sk-test".to_string()),
                max_budget: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut registry = oxllm_providers::default_registry();
        registry.register("mockrerank", Arc::new(MockRerank) as Arc<dyn Provider>);
        let state = AppState::with_registry(config, registry).expect("state built");
        // Pre-fill the deployment spend counter to exactly the cap so the
        // read-only gate rejects (spend >= limit).
        let dep_spend =
            crate::hooks::spend_keys::deployment_spend_key("rerank-model:mockrerank/rerank-1");
        let _ = state.cache.incr(&dep_spend, 1.0, None).await;

        let shared = shared_from(state);
        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let err = rerank(
            State(shared),
            Extension(AuthUser::open()),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect_err("deployment at max_budget cap must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn provider_passthrough_upstream_oversize_releases_budget() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v1/files");
            then.status(200)
                .header("content-type", "application/json")
                .header(
                    "content-length",
                    format!("{}", crate::security::MAX_RESPONSE_BODY_BYTES + 1),
                )
                .body("{}");
        });

        let state = build_state("gpt-4o", "openai/gpt-4o", Some(server.base_url()), true);
        let shared = shared_from(state.clone());
        let auth = auth_with_budget();
        let spend_key = spend_key(auth.key_hash_hex());

        let body = serde_json::json!({"model": "gpt-4o"});
        let result = provider_passthrough(
            State(shared),
            Extension(auth),
            axum::extract::OriginalUri("/openai/v1/files".parse().unwrap()),
            axum::extract::Path("openai/v1/files".to_string()),
            axum::http::HeaderMap::new(),
            axum::http::Method::POST,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await;
        assert!(result.is_err(), "oversize upstream must error");

        let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "budget reservation must be released after oversize upstream, got {spend}"
        );
    }

    // ---- MEDIUM 3: catch-all charges per-deployment TPM ----------------------

    #[tokio::test]
    async fn provider_passthrough_charges_total_tokens_to_tpm_window() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v1/files");
            then.status(200)
                .header("content-type", "application/json")
                .header(UPSTREAM_TOTAL_TOKENS_HEADER, "1234")
                .body("{}");
        });

        let state = build_state("gpt-4o", "openai/gpt-4o", Some(server.base_url()), false);
        let shared = shared_from(state.clone());
        let body = serde_json::json!({"model": "gpt-4o"});
        let resp = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/openai/v1/files".parse().unwrap()),
            axum::extract::Path("openai/v1/files".to_string()),
            axum::http::HeaderMap::new(),
            axum::http::Method::POST,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        let tpm = state
            .cache
            .get("dep_tpm:gpt-4o:openai/gpt-4o")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!(
            (tpm - 1234.0).abs() < 1e-9,
            "aggregate total_tokens must be charged to the TPM window, got {tpm}"
        );
    }

    #[tokio::test]
    async fn provider_passthrough_recovers_usage_from_json_body() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#);
        });

        let state = build_state("gpt-4o", "openai/gpt-4o", Some(server.base_url()), false);
        let shared = shared_from(state.clone());
        let body = serde_json::json!({"model": "gpt-4o", "messages": []});
        let resp = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/openai/v1/chat/completions".parse().unwrap()),
            axum::extract::Path("openai/v1/chat/completions".to_string()),
            axum::http::HeaderMap::new(),
            axum::http::Method::POST,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        let tpm = state
            .cache
            .get("dep_tpm:gpt-4o:openai/gpt-4o")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!(
            (tpm - 15.0).abs() < 1e-9,
            "JSON-body usage must be charged to the TPM window, got {tpm}"
        );
    }

    // ---- MEDIUM 5: malformed JSON body rejected on the catch-all ------------

    /// An `application/x-ndjson` body (a stream of newline-delimited JSON
    /// documents) must be forwarded VERBATIM — not fed to
    /// `serde_json::from_slice`, which cannot parse JSONL and used to turn a
    /// valid batch-relay request into a spurious `400 invalid JSON body`.
    #[tokio::test]
    async fn provider_passthrough_forwards_ndjson_body_verbatim() {
        use httpmock::MockServer;
        let server = MockServer::start();
        let jsonl = "{\"model\":\"gpt-4o\"}\n{\"model\":\"gpt-4o\"}\n";
        server.mock(|when, then| {
            when.method("POST")
                .path("/v1/files")
                .body(jsonl.to_string());
            then.status(200)
                .header("content-type", "application/x-ndjson")
                .body("{}");
        });

        let state = build_state("gpt-4o", "openai/gpt-4o", Some(server.base_url()), false);
        let shared = shared_from(state);
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("content-type", "application/x-ndjson".parse().unwrap());
        let resp = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/openai/v1/files".parse().unwrap()),
            axum::extract::Path("openai/v1/files".to_string()),
            headers,
            axum::http::Method::POST,
            Bytes::from_static(b"{\"model\":\"gpt-4o\"}\n{\"model\":\"gpt-4o\"}\n"),
        )
        .await
        .expect("NDJSON must be forwarded verbatim, not rejected as invalid JSON");
        assert!(resp.status().is_success());
    }

    /// The rerank raw relay must rewrite the client alias to the
    /// deployment's own model id: forwarding the alias verbatim sends the
    /// upstream an id it does not know (Fireworks 500s on
    /// `model: "fw-rerank"`). Found by live-testing rerank against
    /// fireworks qwen3-reranker-8b.
    #[test]
    fn rerank_relay_body_rewrites_alias_to_provider_model() {
        let deployment = oxllm_core::deployment::Deployment {
            model_name: "fw-rerank".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "fireworks_ai/accounts/fireworks/models/qwen3-reranker-8b".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let request = serde_json::json!({
            "model": "fw-rerank",
            "query": "q",
            "documents": ["a", "b"],
        });
        let out = rerank_relay_body(
            &request,
            &deployment,
            &Bytes::from_static(b"{\"model\":\"fw-rerank\"}"),
        )
        .expect("serialize");
        let v: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            v["model"], "accounts/fireworks/models/qwen3-reranker-8b",
            "the alias must be rewritten to the provider model id"
        );
        assert_eq!(v["query"], "q", "other fields pass through verbatim");

        // Already the provider model -> original bytes untouched.
        let same = serde_json::json!({"model": "accounts/fireworks/models/qwen3-reranker-8b"});
        let out2 = rerank_relay_body(
            &same,
            &deployment,
            &Bytes::from_static(b"{\"model\":\"x\"}"),
        )
        .expect("serialize");
        assert_eq!(out2, Bytes::from_static(b"{\"model\":\"x\"}"));
    }

    #[tokio::test]
    async fn provider_passthrough_rejects_malformed_json_body() {
        let state = build_state(
            "gpt-4o",
            "openai/gpt-4o",
            Some("https://example.test".to_string()),
            false,
        );
        let shared = shared_from(state);
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        let err = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/openai/v1/files".parse().unwrap()),
            axum::extract::Path("openai/v1/files".to_string()),
            headers,
            axum::http::Method::POST,
            Bytes::from_static(b"{not json"),
        )
        .await
        .expect_err("malformed JSON body must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    }

    // ---- MEDIUM 6: vetted client headers forwarded on the catch-all ---------

    #[tokio::test]
    async fn provider_passthrough_forwards_anthropic_beta_header() {
        use httpmock::MockServer;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST")
                .path("/v1/messages")
                .header("anthropic-beta", "prompt-caching-2024-07-31");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = build_state(
            "claude",
            "anthropic/claude-3-5-sonnet",
            Some(server.base_url()),
            false,
        );
        let shared = shared_from(state);
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            "prompt-caching-2024-07-31".parse().unwrap(),
        );
        headers.insert("content-type", "application/json".parse().unwrap());
        let body = serde_json::json!({
            "model": "claude",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let resp = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/anthropic/v1/messages".parse().unwrap()),
            axum::extract::Path("anthropic/v1/messages".to_string()),
            headers,
            axum::http::Method::POST,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());
        mock.assert();
    }

    // ---- MEDIUM 7 (integration): dot-segment confinement on the catch-all ---

    #[tokio::test]
    async fn provider_passthrough_rejects_path_that_escapes_base_prefix() {
        // Base configured with a `/v1` prefix; a client path whose normalised
        // form escapes `/v1` must be rejected before any upstream call.
        let state = build_state(
            "gpt-4o",
            "openai/gpt-4o",
            Some("https://example.test/v1".to_string()),
            false,
        );
        let shared = shared_from(state);
        let err = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/openai/files/../../users".parse().unwrap()),
            axum::extract::Path("openai/files/../../users".to_string()),
            axum::http::HeaderMap::new(),
            axum::http::Method::GET,
            Bytes::new(),
        )
        .await
        .expect_err("escaped path must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
    }

    // ---- MEDIUM 4: image edits resolve non-default model names ------------

    /// Multipart inference bodies cannot be safely routed (no multipart
    /// parser: the model field cannot be read or rewritten), so they must be
    /// rejected explicitly instead of silently falling back to
    /// `deployments.first()` — which could forward the request on the wrong
    /// credential to the wrong upstream.
    #[tokio::test]
    async fn images_edits_multipart_without_model_is_rejected_explicitly() {
        use httpmock::MockServer;
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method("POST").path("/images/edits");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = build_state(
            "my-dalle",
            "openai/dall-e-3",
            Some(server.base_url()),
            false,
        );
        let shared = shared_from(state);
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            "multipart/form-data; boundary=xyz".parse().unwrap(),
        );
        let err = images_edits(
            State(shared),
            Extension(AuthUser::open()),
            headers,
            Bytes::from_static(b"--xyz--\r\n"),
        )
        .await
        .expect_err("multipart inference must be rejected, not silently routed");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("multipart"),
            "explicit rejection reason expected, got: {}",
            err.message
        );
        // Nothing was forwarded upstream on the silent first-deployment path.
        upstream.assert_hits(0);
    }

    /// A multipart `images/edits` request carrying the OpenAI SDK's `model`
    /// form field routes to the named deployment, and the field is rewritten
    /// to the deployment's upstream model id before forwarding.
    #[tokio::test]
    async fn images_edits_multipart_with_model_routes_and_rewrites() {
        use httpmock::MockServer;
        let server = MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method("POST")
                .path("/images/edits")
                .body_contains("dall-e-3");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = build_state(
            "my-dalle",
            "openai/dall-e-3",
            Some(server.base_url()),
            false,
        );
        let shared = shared_from(state);
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            "multipart/form-data; boundary=xyz".parse().unwrap(),
        );
        let body = "--xyz\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nmy-dalle\r\n\
                    --xyz\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhi\r\n\
                    --xyz--\r\n";
        let resp = images_edits(
            State(shared),
            Extension(AuthUser::open()),
            headers,
            Bytes::from(body),
        )
        .await
        .expect("multipart with a model field must route");
        assert!(
            resp.status().is_success(),
            "multipart with model must resolve (200), got {}",
            resp.status()
        );
        upstream.assert_hits(1);
    }

    /// The original intent of this suite: a non-default model name routes to
    /// its configured deployment (not a hardcoded "dall-e-2"). JSON bodies
    /// carry the model field and route exactly.
    #[tokio::test]
    async fn images_edits_json_body_routes_non_default_model() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/images/edits");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = build_state(
            "my-dalle",
            "openai/dall-e-3",
            Some(server.base_url()),
            false,
        );
        let shared = shared_from(state);
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        let resp = images_edits(
            State(shared),
            Extension(AuthUser::open()),
            headers,
            Bytes::from_static(b"{\"model\": \"my-dalle\"}"),
        )
        .await
        .expect("JSON body with explicit model must route");
        assert!(
            resp.status().is_success(),
            "non-default model name must resolve (200), got {}",
            resp.status()
        );
    }

    // ---- LOW 9: rerank raw-relay stamps both request-id headers -------------

    /// An identity-shaped rerank provider with input/output control probes so
    /// the transform branch's gateway boundary is exercised end-to-end.
    struct MockRerank;

    impl RerankProvider for MockRerank {
        fn rerank_url(&self, deployment: &Deployment) -> String {
            let base = deployment
                .oxllm_params
                .api_base
                .as_deref()
                .unwrap_or("")
                .trim_end_matches('/');
            format!("{base}/rerank")
        }
        fn apply_auth(
            &self,
            _deployment: &Deployment,
            _creds: &ResolvedCredentials,
            _headers: &mut HashMap<String, String>,
        ) {
        }
        fn transform_request(
            &self,
            request: &serde_json::Value,
            _deployment: &Deployment,
        ) -> oxllm_core::error::Result<serde_json::Value> {
            if crate::orchestration::request_controls::contains(request) {
                return Err(oxllm_core::Error::Transform(
                    "gateway controls reached the provider transform".into(),
                ));
            }
            let mut transformed = request.clone();
            transformed["tags"] = serde_json::json!(["provider-output-probe"]);
            Ok(transformed)
        }
        fn transform_response(
            &self,
            body: &serde_json::Value,
            _model: &str,
        ) -> oxllm_core::error::Result<serde_json::Value> {
            Ok(body.clone())
        }
    }

    impl Provider for MockRerank {
        fn id(&self) -> ProviderId {
            "mockrerank"
        }
        fn supports(&self, endpoint: Endpoint) -> bool {
            matches!(endpoint, Endpoint::Rerank)
        }
        fn rerank(&self) -> Option<&dyn RerankProvider> {
            Some(self)
        }
    }

    #[tokio::test]
    async fn rerank_transform_strips_gateway_controls_before_and_after_provider_shaping() {
        let server = httpmock::MockServer::start();
        let wire = server.mock(|when, then| {
            when.method("POST").path("/rerank").matches(|request| {
                request
                    .body
                    .as_ref()
                    .and_then(|body| serde_json::from_slice::<serde_json::Value>(body).ok())
                    .is_some_and(|body| {
                        !crate::orchestration::request_controls::contains(&body)
                            && body["metadata"]["user_id"] == "kept"
                    })
            });
            then.status(200)
                .json_body(serde_json::json!({"results":[]}));
        });
        let config: ProxyConfig = serde_json::from_value(serde_json::json!({
            "model_list":[{"model_name":"ranked", "oxllm_params":{
                "model":"mockrerank/native-rank", "api_key":"synthetic-key", "api_base":server.base_url()
            }}], "general_settings":{"allow_private_api_base":true}
        })).unwrap();
        let mut registry = oxllm_providers::default_registry();
        registry.register("mockrerank", Arc::new(MockRerank));
        let shared = shared_from(AppState::with_registry(config, registry).unwrap());
        let response = rerank(
            State(shared),
            Extension(AuthUser::open()),
            HeaderMap::new(),
            Bytes::from(
                serde_json::json!({"model":"ranked", "query":"hello", "documents":["one"],
                "tags":["body"], "metadata":{"tags":["meta"], "user_id":"kept"},
                "num_retries":0, "cache":{"use_cache":false}})
                .to_string(),
            ),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        wire.assert_hits(1);
    }

    /// The rerank transform branch must forward the upstream debugging
    /// allowlist (`x-request-id`, `retry-after`) on a provider 429 and must
    /// not leak non-allowlisted headers — same parity as the chat error path.
    #[tokio::test]
    async fn rerank_upstream_error_forwards_debug_headers() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/rerank");
            then.status(429)
                .header("content-type", "application/json")
                .header("x-request-id", "req_rerank_xyz")
                .header("retry-after", "11")
                .header("server", "vendor/1.0")
                .body(r#"{"error":{"message":"rate limited"}}"#);
        });

        let model_list = vec![ConfigModel {
            model_name: "rerank-model".to_string(),
            oxllm_params: OxllmParams {
                model: "mockrerank/rerank-1".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut registry = oxllm_providers::default_registry();
        registry.register("mockrerank", Arc::new(MockRerank) as Arc<dyn Provider>);
        let state = AppState::with_registry(config, registry).expect("state built");
        let shared = shared_from(state);

        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let result = rerank(
            State(shared),
            Extension(AuthUser::open()),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).expect("serialize")),
        )
        .await;
        let err = result.expect_err("provider 429 must surface an error");
        assert_eq!(err.status, axum::http::StatusCode::TOO_MANY_REQUESTS);
        let resp = err.into_response();
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok()),
            Some("req_rerank_xyz"),
            "provider x-request-id must be forwarded"
        );
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("11"),
            "provider retry-after must be forwarded"
        );
        assert!(
            resp.headers().get("server").is_none(),
            "non-allowlisted upstream headers must not leak"
        );
    }

    #[tokio::test]
    async fn rerank_relay_response_carries_request_id_headers() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/rerank");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        // OpenAI exposes no rerank transform → raw relay branch.
        let state = build_state("gpt-4o", "openai/gpt-4o", Some(server.base_url()), false);
        let shared = shared_from(state);
        let body = serde_json::json!({"model": "gpt-4o", "query": "x", "documents": ["a", "b"]});
        let resp = rerank(
            State(shared),
            Extension(AuthUser::open()),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("relay ok");
        let oxllm = resp
            .headers()
            .get("x-oxllm-request-id")
            .expect("oxllm request id stamped")
            .to_str()
            .unwrap()
            .to_string();
        let standard = resp
            .headers()
            .get("x-request-id")
            .expect("standard request id stamped")
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(oxllm, standard);
        assert!(!oxllm.is_empty());
    }

    // ---- HIGH: catch-all default base must not double the /v1 prefix -------

    #[test]
    fn catch_all_default_base_is_host_only_so_no_double_version() {
        // A deployment with no explicit api_base falls back to the provider
        // default table. The catch-all forwards the full client path, which
        // carries the version prefix (e.g. `v1/messages`). The default base
        // must therefore be reduced to host-only so the merged URL is not
        // `https://api.anthropic.com/v1/v1/messages`.
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let base = crate::pipeline::catch_all_upstream_base_url(&deployment)
            .expect("catch-all base resolves");
        // Host-only: no trailing version path.
        assert_eq!(base, "https://api.anthropic.com");
        let merged =
            crate::pipeline::passthrough::merge_base_path_query(&base, "v1/messages", None)
                .expect("merged url");
        assert_eq!(
            merged, "https://api.anthropic.com/v1/messages",
            "version prefix must not be doubled"
        );
    }

    #[test]
    fn catch_all_default_base_host_only_for_each_known_provider() {
        use oxllm_core::deployment::OxllmParams;
        fn base_for(litellm_model: &str) -> String {
            let deployment = Deployment {
                model_name: "m".to_string(),
                oxllm_params: OxllmParams {
                    model: litellm_model.to_string(),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            };
            crate::pipeline::catch_all_upstream_base_url(&deployment).unwrap()
        }
        assert_eq!(base_for("openai/gpt-4o"), "https://api.openai.com");
        assert_eq!(base_for("cohere/rerank-v3"), "https://api.cohere.com");
        // Gemini's default carries a two-segment version path; both must be
        // stripped so the client-supplied path is the sole version source.
        assert_eq!(
            base_for("gemini/gemini-1.5-pro"),
            "https://generativelanguage.googleapis.com"
        );
    }

    #[tokio::test]
    async fn provider_passthrough_no_api_base_routes_to_single_version_path() {
        // End-to-end: with api_base unset the catch-all must forward to the
        // upstream at exactly `/v1/messages`, NOT `/v1/v1/messages`. Because
        // the default base points at the real Anthropic host (not a mock),
        // we stub the network by asserting the merged URL shape via a
        // deployment whose default base is exercised — the mock cannot
        // intercept api.anthropic.com, so this test validates the URL
        // construction directly through the public helpers the handler uses.
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let base = crate::pipeline::catch_all_upstream_base_url(&deployment).unwrap();
        let merged =
            crate::pipeline::passthrough::merge_base_path_query(&base, "v1/messages", None)
                .unwrap();
        assert_eq!(merged, "https://api.anthropic.com/v1/messages");
        // Path confinement against a host-only base accepts any path (no
        // prefix to escape), so the messages route is allowed.
        let confined = confine_upstream_url(&base, &merged).expect("host-only base allows path");
        assert_eq!(confined, "https://api.anthropic.com/v1/messages");
    }

    // ---- HIGH: catch-all honours upstream-reported response cost ------------

    /// A post_success hook that records the ResponseSummary cost so the test
    /// can assert what was logged to the spend pipeline.
    struct CostCaptureHook {
        captured: std::sync::Arc<std::sync::Mutex<Option<f64>>>,
    }
    #[async_trait::async_trait]
    impl oxllm_core::Hook for CostCaptureHook {
        async fn pre_call(
            &self,
            _ctx: &mut oxllm_core::CallContext,
        ) -> oxllm_core::error::Result<oxllm_core::HookAction> {
            Ok(oxllm_core::HookAction::Continue)
        }
        async fn post_success(
            &self,
            _ctx: &oxllm_core::CallContext,
            resp: &oxllm_core::ResponseSummary,
        ) {
            *self.captured.lock().unwrap() = resp.cost;
        }
    }

    fn state_with_cost_capture(
        api_base: String,
        recorded: std::sync::Arc<std::sync::Mutex<Option<f64>>>,
    ) -> AppState {
        let mut state = build_state("gpt-4o", "openai/gpt-4o", Some(api_base), false);
        state.hooks = vec![std::sync::Arc::new(CostCaptureHook { captured: recorded })];
        state
    }

    #[tokio::test]
    async fn provider_passthrough_honours_upstream_response_cost() {
        use crate::pipeline::{
            COST_HEADER, UPSTREAM_RESPONSE_COST_HEADER_LITELLM,
            UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM,
        };
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v1/files");
            then.status(200)
                .header("content-type", "application/json")
                .header(UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM, "1000")
                .header(UPSTREAM_RESPONSE_COST_HEADER_LITELLM, "0.42")
                .body("{}");
        });

        let recorded = std::sync::Arc::new(std::sync::Mutex::new(None));
        let state = state_with_cost_capture(server.base_url(), recorded.clone());
        let shared = shared_from(state);
        let body = serde_json::json!({"model": "gpt-4o"});
        let resp = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/openai/v1/files".parse().unwrap()),
            axum::extract::Path("openai/v1/files".to_string()),
            axum::http::HeaderMap::new(),
            axum::http::Method::POST,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        let captured = *recorded.lock().unwrap();
        assert_eq!(
            captured,
            Some(0.42),
            "spend log must carry the upstream-reported cost, got {captured:?}"
        );
        let header = resp
            .headers()
            .get(COST_HEADER)
            .and_then(|v| v.to_str().ok())
            .expect("cost header must be stamped on catch-all success");
        let parsed: f64 = header.parse().expect("numeric cost header");
        assert!(
            (parsed - 0.42).abs() < 1e-9,
            "x-oxllm-response-cost must be 0.42, got {parsed}"
        );
    }

    // ---- MEDIUM: catch-all stamps request-id + ratelimit headers -----------

    fn auth_with_rpm(rpm: u32) -> AuthUser {
        AuthUser {
            rpm: Some(rpm),
            ..AuthUser::open()
        }
    }

    #[tokio::test]
    async fn provider_passthrough_stamps_request_id_and_ratelimit_headers() {
        use crate::pipeline::{REQUEST_ID_HEADER, STANDARD_REQUEST_ID_HEADER};
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v1/files");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = build_state("gpt-4o", "openai/gpt-4o", Some(server.base_url()), false);
        let shared = shared_from(state);
        let body = serde_json::json!({"model": "gpt-4o"});
        let resp = provider_passthrough(
            State(shared),
            Extension(auth_with_rpm(100)),
            axum::extract::OriginalUri("/openai/v1/files".parse().unwrap()),
            axum::extract::Path("openai/v1/files".to_string()),
            axum::http::HeaderMap::new(),
            axum::http::Method::POST,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("passthrough ok");
        let headers = resp.headers();
        // Both request-id headers (H8 contract).
        assert!(
            headers
                .get(REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok())
                .map(|s| !s.is_empty())
                .unwrap_or(false),
            "x-oxllm-request-id must be present and non-empty"
        );
        assert!(
            headers.get(STANDARD_REQUEST_ID_HEADER).is_some(),
            "x-request-id must be present"
        );
        // Rate-limit throttle headers are stamped because the key carries an
        // RPM limit.
        assert!(
            headers.get("x-ratelimit-remaining-requests").is_some(),
            "x-ratelimit-remaining-requests must be stamped on catch-all success"
        );
        assert!(
            headers.get("x-ratelimit-limit-requests").is_some(),
            "x-ratelimit-limit-requests must be stamped on catch-all success"
        );
    }

    #[tokio::test]
    async fn provider_passthrough_stamps_request_id_on_relayed_error() {
        use crate::pipeline::REQUEST_ID_HEADER;
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v1/files");
            then.status(500).body("upstream boom");
        });

        let state = build_state("gpt-4o", "openai/gpt-4o", Some(server.base_url()), false);
        let shared = shared_from(state);
        let body = serde_json::json!({"model": "gpt-4o"});
        let resp = provider_passthrough(
            State(shared),
            Extension(AuthUser::open()),
            axum::extract::OriginalUri("/openai/v1/files".parse().unwrap()),
            axum::extract::Path("openai/v1/files".to_string()),
            axum::http::HeaderMap::new(),
            axum::http::Method::POST,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("relayed 5xx is Ok(..), not Err");
        assert_eq!(resp.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            resp.headers()
                .get(REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok())
                .map(|s| !s.is_empty())
                .unwrap_or(false),
            "x-oxllm-request-id must be stamped on a relayed error so clients can correlate"
        );
    }

    // ---- MEDIUM: rerank transform tracks cost/usage + TPM + headers ---------

    fn state_with_mock_rerank(api_base: String) -> AppState {
        let model_list = vec![ConfigModel {
            model_name: "rerank-model".to_string(),
            oxllm_params: OxllmParams {
                model: "mockrerank/rerank-1".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(api_base),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut registry = oxllm_providers::default_registry();
        registry.register("mockrerank", Arc::new(MockRerank) as Arc<dyn Provider>);
        AppState::with_registry(config, registry).expect("state built")
    }

    /// C02 MEDIUM: a transport failure on a passthrough route (rerank
    /// transform branch → `execute_upstream_json`) must reach the failure
    /// hooks as `Error::Network`, not `Error::Upstream`, so the rate-limit
    /// hook RELEASES the per-key RPM reservation — 60 connection-refused
    /// attempts against an `rpm: 60` key must not lock the key out for the
    /// full 60s window when no provider capacity was consumed. Pins the
    /// `upstream_unreachable` marker round trip through
    /// `core_error_from_api`.
    #[tokio::test]
    async fn rerank_transport_failure_releases_rpm_reservation() {
        // Grab a free port then drop the listener: nothing accepts, so the
        // upstream call fails fast with ECONNREFUSED.
        let refused_base = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            let port = listener.local_addr().expect("addr").port();
            drop(listener);
            format!("http://127.0.0.1:{port}")
        };

        let mut state = state_with_mock_rerank(refused_base);
        state.hooks = vec![Arc::new(crate::hooks::RateLimitHook::new(
            state.cache.clone(),
            Some(60),
            None,
        ))];
        let shared = shared_from(state.clone());
        let auth = AuthUser::open();
        let rpm_key = format!("rpm:{}", auth.key_hash_hex());

        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let err = rerank(
            State(shared),
            Extension(auth),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect_err("connection-refused upstream must surface an error");
        assert_eq!(err.status, axum::http::StatusCode::BAD_GATEWAY);
        assert_eq!(err.code.as_deref(), Some("upstream_unreachable"));

        assert_eq!(
            state.cache.counter_value(&rpm_key).await,
            Some(0.0),
            "transport failure must release the RPM reservation (request never reached the provider)"
        );
    }

    /// C02 MEDIUM (raw-relay counterpart): the same transport-failure
    /// contract must hold on the rerank RAW-RELAY branch (provider without a
    /// rerank transform). That branch previously mapped every
    /// `send_upstream` error to `Error::Upstream`, so gateway-synthesized
    /// 502/504 transport failures kept the key RPM/TPM reservations for the
    /// full 60s window even though no provider was ever reached — diverging
    /// from `hooked_raw` and the transform branch. Route through
    /// `core_error_from_api` so the failure hooks see `Error::Network`.
    #[tokio::test]
    async fn rerank_raw_relay_transport_failure_releases_rpm_reservation() {
        // Nothing accepts on this port: the raw-relay `send_upstream` fails
        // fast with ECONNREFUSED (gateway-synthesized 502).
        let refused_base = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
            let port = listener.local_addr().expect("addr").port();
            drop(listener);
            format!("http://127.0.0.1:{port}")
        };

        // OpenAI exposes no rerank transform, so the handler takes the raw
        // relay branch and forwards the body verbatim to `{base}/rerank`.
        let mut state = build_state("rerank-model", "openai/rerank-1", Some(refused_base), false);
        state.hooks = vec![Arc::new(crate::hooks::RateLimitHook::new(
            state.cache.clone(),
            Some(60),
            None,
        ))];
        let shared = shared_from(state.clone());
        let auth = AuthUser::open();
        let rpm_key = format!("rpm:{}", auth.key_hash_hex());

        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let err = rerank(
            State(shared),
            Extension(auth),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect_err("connection-refused upstream must surface an error");
        assert_eq!(err.status, axum::http::StatusCode::BAD_GATEWAY);
        assert_eq!(err.code.as_deref(), Some("upstream_unreachable"));

        assert_eq!(
            state.cache.counter_value(&rpm_key).await,
            Some(0.0),
            "raw-relay transport failure must release the RPM reservation (request never reached the provider)"
        );
    }

    /// C02 MEDIUM (counterpart): a provider-SERVED error on the same rerank
    /// funnel must KEEP the RPM reservation — the provider processed the
    /// request and it counted against the provider's rate limit.
    #[tokio::test]
    async fn rerank_provider_served_error_keeps_rpm_reservation() {
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/rerank");
            then.status(500)
                .header("content-type", "application/json")
                .body(r#"{"error":{"message":"boom"}}"#);
        });

        let mut state = state_with_mock_rerank(server.base_url());
        state.hooks = vec![Arc::new(crate::hooks::RateLimitHook::new(
            state.cache.clone(),
            Some(60),
            None,
        ))];
        let shared = shared_from(state.clone());
        let auth = AuthUser::open();
        let rpm_key = format!("rpm:{}", auth.key_hash_hex());

        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let err = rerank(
            State(shared),
            Extension(auth),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect_err("provider 500 must surface an error");
        assert_eq!(err.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.code.as_deref(), Some("upstream_error"));

        assert_eq!(
            state.cache.counter_value(&rpm_key).await,
            Some(1.0),
            "provider-served failure must KEEP the RPM reservation"
        );
    }

    #[tokio::test]
    async fn rerank_transform_tracks_usage_and_charges_tpm() {
        use crate::pipeline::UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM;
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/rerank");
            then.status(200)
                .header("content-type", "application/json")
                .header(UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM, "500")
                .body("{}");
        });

        let state = state_with_mock_rerank(server.base_url());
        let shared = shared_from(state.clone());
        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let resp = rerank(
            State(shared),
            Extension(AuthUser::open()),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("rerank ok");
        assert!(resp.status().is_success());

        let tpm = state
            .cache
            .get("dep_tpm:rerank-model:mockrerank/rerank-1")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!(
            (tpm - 500.0).abs() < 1e-9,
            "transform branch must charge TPM window from header usage, got {tpm}"
        );
    }

    #[tokio::test]
    async fn rerank_transform_surfaces_cost_and_ratelimit_headers() {
        use crate::pipeline::{
            COST_HEADER, UPSTREAM_RESPONSE_COST_HEADER_LITELLM,
            UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM,
        };
        use httpmock::MockServer;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/rerank");
            then.status(200)
                .header("content-type", "application/json")
                .header(UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM, "100")
                .header(UPSTREAM_RESPONSE_COST_HEADER_LITELLM, "0.05")
                .body("{}");
        });

        let state = state_with_mock_rerank(server.base_url());
        let shared = shared_from(state);
        let body = serde_json::json!({
            "model": "rerank-model",
            "query": "hello",
            "documents": ["a", "b"],
        });
        let resp = rerank(
            State(shared),
            Extension(auth_with_rpm(50)),
            axum::http::HeaderMap::new(),
            Bytes::from(serde_json::to_vec(&body).unwrap()),
        )
        .await
        .expect("rerank ok");
        let headers = resp.headers();
        let cost_hdr = headers
            .get(COST_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok())
            .expect("x-oxllm-response-cost must be stamped on rerank transform success");
        assert!(
            (cost_hdr - 0.05).abs() < 1e-9,
            "rerank cost header must reflect upstream-reported cost, got {cost_hdr}"
        );
        assert!(
            headers.get("x-ratelimit-remaining-requests").is_some(),
            "x-ratelimit-remaining-requests must be stamped on rerank transform success"
        );
    }
}
