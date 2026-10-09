use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::Response;
use oxllm_core::CallContext;
use serde_json::{Value, json};

use super::set_header;

pub(crate) fn record_upstream_timing(ctx: &mut CallContext) {
    if let Some(start) = ctx.upstream_start_time {
        let end = std::time::Instant::now();
        let previous = ctx.upstream_end_time.unwrap_or(start);
        ctx.upstream_total_duration = ctx
            .upstream_total_duration
            .saturating_add(end.saturating_duration_since(previous));
        ctx.upstream_end_time = Some(end);
    }
}

/// Direct calls echo the request; fallback calls expose the public serving group.
pub(crate) fn public_response_model(ctx: &CallContext) -> &str {
    if let Some(model) = ctx.cached_response_model.as_deref() {
        return model;
    }
    if ctx.attempted_fallbacks > 0 {
        return ctx.serving_model_group.as_deref().unwrap_or(&ctx.model);
    }
    &ctx.model
}

pub(crate) fn record_fallback_error(ctx: &mut CallContext, error: &crate::error::ApiError) {
    if ctx
        .request_body
        .get("include_fallback_errors")
        .and_then(Value::as_bool)
        == Some(true)
        && ctx.fallback_errors.len() < 16
    {
        ctx.fallback_errors.push(json!({
            "message": "upstream attempt failed", "type": error.error_type,
            "param": null, "code": error.status.as_u16()
        }));
    }
}

pub(crate) fn capture_upstream_headers(ctx: &mut CallContext, headers: &HeaderMap) {
    ctx.upstream_response_headers = headers
        .iter()
        .filter_map(|(name, value)| {
            (name.as_str().starts_with("x-ratelimit-"))
                .then(|| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (name.to_string(), value.to_string()))
                })
                .flatten()
        })
        .collect();
}

pub(crate) fn stamp_deployment_headers(response: &mut Response, ctx: &CallContext) {
    for (name, value) in &ctx.upstream_response_headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            response.headers_mut().entry(name).or_insert(value);
        }
    }
    if let Some(id) = ctx.deployment_id.as_deref() {
        for header in [
            "x-oxllm-deployment-id",
            "x-oxllm-model-id",
            "x-litellm-model-id",
        ] {
            set_header(response, header, id);
        }
    }
    if let Some(model) = ctx.deployment_model.as_deref() {
        for header in ["x-oxllm-model-name", "x-litellm-model-name"] {
            set_header(response, header, model);
        }
    }
    set_header(response, "x-oxllm-requested-model", &ctx.model);
    for header in ["x-oxllm-model-group", "x-litellm-model-group"] {
        set_header(
            response,
            header,
            ctx.serving_model_group.as_deref().unwrap_or(&ctx.model),
        );
    }
    for header in ["x-oxllm-attempted-retries", "x-litellm-attempted-retries"] {
        set_header(response, header, &ctx.attempted_retries.to_string());
    }
    for header in [
        "x-oxllm-attempted-fallbacks",
        "x-litellm-attempted-fallbacks",
    ] {
        set_header(response, header, &ctx.attempted_fallbacks.to_string());
    }
    if let Some(budget) = ctx.max_retries {
        for header in ["x-oxllm-max-retries", "x-litellm-max-retries"] {
            set_header(response, header, &budget.to_string());
        }
    }
    if ctx.attempted_fallbacks > 0 && !ctx.fallback_errors.is_empty() {
        let errors = serde_json::to_string(&ctx.fallback_errors).unwrap_or_default();
        for header in ["x-oxllm-fallback-errors", "x-litellm-fallback-errors"] {
            set_header(response, header, &errors);
        }
    }
    if let Some(start) = ctx.start_time {
        let elapsed = start.elapsed();
        let upstream = ctx.upstream_total_duration;
        for header in [
            "x-oxllm-response-duration-ms",
            "x-litellm-response-duration-ms",
        ] {
            set_header(
                response,
                header,
                &(elapsed.as_secs_f64() * 1000.0).to_string(),
            );
        }
        for header in [
            "x-oxllm-overhead-duration-ms",
            "x-litellm-overhead-duration-ms",
        ] {
            set_header(
                response,
                header,
                &(elapsed
                    .saturating_sub(if ctx.cache_hit {
                        ctx.cache_lookup_duration
                    } else {
                        upstream
                    })
                    .as_secs_f64()
                    * 1000.0)
                    .to_string(),
            );
        }
        if ctx.upstream_start_time.is_some() {
            set_header(
                response,
                "x-oxllm-upstream-duration-ms",
                &(upstream.as_secs_f64() * 1000.0).to_string(),
            );
        }
    }
}

/// Apply public response identities after accounting has consumed the serving
/// identity. Cached bodies and provider usage remain available for billing.
pub(crate) fn normalize_client_response(body: &mut Value, requested: &str) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    if object.contains_key("model") || object.contains_key("choices") {
        object.insert("model".into(), json!(requested));
    }
    if let Some(usage) = object.get_mut("usage").and_then(Value::as_object_mut)
        && usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .is_some()
    {
        let details = usage
            .entry("completion_tokens_details")
            .or_insert_with(|| json!({}));
        if details.is_null() {
            *details = json!({});
        }
        if let Some(details) = details.as_object_mut()
            && details.get("reasoning_tokens").is_none_or(Value::is_null)
        {
            details.insert("reasoning_tokens".into(), json!(0));
        }
    }
}
