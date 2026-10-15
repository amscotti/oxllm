//! Request-scoped router controls, shared by every retry/fallback attempt.

use std::time::Duration;

use oxllm_core::Deployment;
use serde_json::Value;

use crate::error::ApiError;
use crate::state::AppState;

pub(crate) const FIELDS: &[&str] = &[
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "disable_fallbacks",
    "max_fallbacks",
    "num_retries",
    "timeout",
    "stream_timeout",
    "include_fallback_errors",
    "log_client_error_tracebacks",
    "stream_chunk_size",
    "cache",
    "cache_control_injection_points",
    "tags",
];

/// Client retry overrides are bounded independently of trusted server policies.
const MAX_REQUEST_RETRIES: u32 = 10;

pub(crate) fn validate(request: &Value) -> Result<(), ApiError> {
    if let Some(id) = request.get("user").and_then(Value::as_str) {
        crate::pipeline::validate_end_user_id(id)?;
    }
    if request
        .as_object()
        .is_some_and(|object| object.keys().any(|field| deployment_credential(field)))
    {
        return Err(ApiError::bad_request(
            "federation credentials must be configured on a deployment",
        ));
    }
    for field in ["timeout", "stream_timeout"] {
        if let Some(value) = request.get(field).filter(|value| !value.is_null()) {
            let valid = value
                .as_f64()
                .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                .is_some_and(|duration| {
                    !duration.is_zero() && std::time::Instant::now().checked_add(duration).is_some()
                });
            if !valid {
                return Err(ApiError::bad_request(format!(
                    "{field} must be a finite positive number of seconds"
                )));
            }
        }
    }
    for field in ["num_retries", "max_fallbacks"] {
        if request
            .get(field)
            .filter(|value| !value.is_null())
            .is_some_and(|value| {
                value
                    .as_u64()
                    .and_then(|count| u32::try_from(count).ok())
                    .is_none()
            })
        {
            return Err(ApiError::bad_request(format!(
                "{field} must be a non-negative integer"
            )));
        }
    }
    if request
        .get("num_retries")
        .and_then(Value::as_u64)
        .is_some_and(|count| count > u64::from(MAX_REQUEST_RETRIES))
    {
        return Err(ApiError::bad_request(format!(
            "num_retries must not exceed {MAX_REQUEST_RETRIES}"
        )));
    }
    for field in [
        "disable_fallbacks",
        "include_fallback_errors",
        "log_client_error_tracebacks",
    ] {
        if request
            .get(field)
            .filter(|value| !value.is_null())
            .is_some_and(|value| !value.is_boolean())
        {
            return Err(ApiError::bad_request(format!("{field} must be a boolean")));
        }
    }
    for field in [
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
    ] {
        let Some(value) = request.get(field).filter(|value| !value.is_null()) else {
            continue;
        };
        let valid = value.as_array().is_some_and(|entries| {
            entries.iter().all(valid_name)
                || entries.iter().all(|entry| {
                    entry.as_object().is_some_and(|rules| {
                        rules.iter().all(|(model, targets)| {
                            !model.trim().is_empty()
                                && (targets.is_null()
                                    || valid_name(targets)
                                    || targets
                                        .as_array()
                                        .is_some_and(|names| names.iter().all(valid_name)))
                        })
                    })
                })
        });
        if !valid {
            return Err(ApiError::bad_request(format!(
                "{field} must be a list of model names or fallback rules"
            )));
        }
    }
    if request
        .get("tags")
        .filter(|value| !value.is_null())
        .is_some_and(|value| {
            value
                .as_array()
                .is_none_or(|tags| tags.iter().any(|tag| !tag.is_string()))
        })
    {
        return Err(ApiError::bad_request("tags must be an array of strings"));
    }
    Ok(())
}

fn valid_name(value: &Value) -> bool {
    value.as_str().is_some_and(|name| !name.trim().is_empty())
}

fn deployment_credential(field: &str) -> bool {
    field.starts_with("anthropic_identity_")
        || field.starts_with("anthropic_federation_")
        || matches!(
            field,
            "anthropic_organization_id"
                | "anthropic_service_account_id"
                | "anthropic_disable_workload_identity_federation"
        )
}

pub(crate) fn contains(request: &Value) -> bool {
    request.as_object().is_some_and(|object| {
        object
            .keys()
            .any(|field| FIELDS.contains(&field.as_str()) || deployment_credential(field))
            || object
                .get("metadata")
                .and_then(Value::as_object)
                .is_some_and(|metadata| metadata.contains_key("tags"))
    })
}

/// Provider transforms still need cache selectors; all other gateway controls
/// are removed before an adapter can rename or nest them.
pub(crate) fn strip_before_transform(body: &mut Value) {
    strip_fields(body, true);
}

pub(crate) fn strip(body: &mut Value) {
    strip_fields(body, false);
}

fn strip_fields(body: &mut Value, keep_cache_selectors: bool) {
    if let Some(object) = body.as_object_mut() {
        object.retain(|field, _| {
            !deployment_credential(field)
                && (!FIELDS.contains(&field.as_str())
                    || (keep_cache_selectors && field == "cache_control_injection_points"))
        });
        if let Some(metadata) = object.get_mut("metadata").and_then(Value::as_object_mut) {
            metadata.remove("tags");
        }
    }
}

pub(crate) fn copy(request: &Value, out: &mut serde_json::Map<String, Value>) {
    for field in FIELDS {
        if let Some(value) = request.get(*field).filter(|value| {
            !value.is_null() || !matches!(*field, "cache" | "cache_control_injection_points")
        }) {
            out.insert((*field).into(), value.clone());
        }
    }
}

pub(crate) fn seconds(request: &Value, field: &str) -> Option<f64> {
    request
        .get(field)
        .and_then(Value::as_f64)
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
}

pub(crate) fn retries(
    state: &AppState,
    model: &str,
    deployment: Option<&Deployment>,
    request: &Value,
    status: u16,
) -> u32 {
    state
        .router
        .retry_policy_for_status_model(status, model)
        .or_else(|| {
            request
                .get("num_retries")
                .and_then(Value::as_u64)
                .and_then(|count| u32::try_from(count).ok())
                .map(|count| count.min(MAX_REQUEST_RETRIES))
        })
        .or_else(|| deployment.and_then(|dep| dep.oxllm_params.num_retries))
        .unwrap_or_else(|| state.router.retries_for_status_model(status, model))
}

pub(crate) fn fallbacks<'a>(
    request: &'a Value,
    field: &str,
    model: &str,
    defaults: Vec<&'a str>,
) -> Vec<&'a str> {
    if request.get("disable_fallbacks").and_then(Value::as_bool) == Some(true) {
        return Vec::new();
    }
    let Some(entries) = request
        .get(field)
        .filter(|value| !value.is_null())
        .and_then(Value::as_array)
    else {
        return defaults;
    };
    if entries.iter().all(Value::is_string) {
        return entries.iter().filter_map(Value::as_str).collect();
    }
    let targets = [model, "*"]
        .iter()
        .find_map(|name| entries.iter().find_map(|entry| entry.get(*name)));
    match targets {
        Some(Value::String(name)) => vec![name.as_str()],
        Some(Value::Array(names)) => names.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn max_fallbacks(state: &AppState, request: &Value) -> u32 {
    request
        .get("max_fallbacks")
        .and_then(Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
        .map_or(state.router.config().max_fallbacks, |count| {
            count.min(state.router.config().max_fallbacks)
        })
}
