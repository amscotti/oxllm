//! `GET /v1/models` and `GET /v1/models/{model}` — list and retrieve
//! configured models in the OpenAI shape.

use std::collections::HashSet;

use axum::Json;
use axum::extract::{Extension, Path, State};
use axum::response::{IntoResponse, Response};
use oxllm_core::Deployment;
use serde_json::json;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::pipeline::check_model_acl;
use crate::state::SharedState;

/// Single `model` record as exposed by `GET /v1/models` and
/// `GET /v1/models/{model}`. Matches the OpenAI shape so existing clients
/// parse it without per-provider branches.
///
/// `max_input_tokens` / `max_output_tokens` are included only when resolved
/// (see [`resolve_token_limits`]); absent otherwise. They are never the cause
/// of a 500 — malformed config values are treated as absent (U54).
fn model_object(id: &str, max_input: Option<u64>, max_output: Option<u64>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("id".to_string(), json!(id));
    map.insert("object".to_string(), json!("model"));
    map.insert("created".to_string(), json!(0));
    map.insert("owned_by".to_string(), json!("oxllm"));
    if let Some(v) = max_input {
        map.insert("max_input_tokens".to_string(), json!(v));
    }
    if let Some(v) = max_output {
        map.insert("max_output_tokens".to_string(), json!(v));
    }
    serde_json::Value::Object(map)
}

/// Resolve the token context limits for a model name.
///
/// Preference order (U55):
/// 1. The static, compile-time-embedded price table
///    ([`oxllm_core::cost::lookup_pricing`]) — always current, never stale by
///    a hot-reload race. When the model is priced and carries either limit,
///    the price table's values win outright.
/// 2. The first matching deployment's config (`model_info` then
///    `oxllm_params` extra map) — a per-deployment override for models that
///    are absent from the price table.
///
/// Malformed config values (non-numeric strings, empty strings, null) are
/// skipped rather than fatal (U54) — Rust's TOML parser already types known
/// fields, but the deployment `extra` catch-all can hold arbitrary JSON, so
/// parsing is defensive.
fn resolve_token_limits(
    deployments: &[Deployment],
    model_name: &str,
) -> (Option<u64>, Option<u64>) {
    // 1. The static, compile-time-embedded price table
    //    ([`oxllm_core::cost::lookup_pricing`]) — always current, never stale
    //    by a hot-reload race. When the model is priced and carries either
    //    limit, the price table's values win outright.
    if let Some(pricing) = oxllm_core::cost::lookup_pricing(model_name) {
        if pricing.max_input_tokens.is_some() || pricing.max_output_tokens.is_some() {
            return (
                pricing.max_input_tokens.map(|v| v as u64),
                pricing.max_output_tokens.map(|v| v as u64),
            );
        }
    }
    let Some(dep) = deployments.iter().find(|d| d.model_name == model_name) else {
        return (None, None);
    };
    // 1b. Alias case: `model_name` is a client-facing alias that is not itself
    //     priced, but the deployment's upstream model id is.
    if let Some(pricing) = oxllm_core::cost::lookup_pricing(&dep.oxllm_params.model) {
        if pricing.max_input_tokens.is_some() || pricing.max_output_tokens.is_some() {
            return (
                pricing.max_input_tokens.map(|v| v as u64),
                pricing.max_output_tokens.map(|v| v as u64),
            );
        }
    }
    // 2. Per-deployment config fallback (defensive parse — malformed → None).
    config_token_limits(dep)
}

/// Read `max_input_tokens` / `max_output_tokens` from a deployment's config
/// maps (`model_info.extra` then `oxllm_params.extra`). A value is accepted
/// when it is a positive integer or a numeric string; everything else
/// (booleans, objects, non-numeric or empty strings) yields `None`.
fn config_token_limits(deployment: &Deployment) -> (Option<u64>, Option<u64>) {
    let read = |field: &str| -> Option<u64> {
        deployment
            .model_info
            .as_ref()
            .and_then(|mi| mi.extra.get(field))
            .or_else(|| deployment.oxllm_params.extra.get(field))
            .and_then(token_value)
    };
    (read("max_input_tokens"), read("max_output_tokens"))
}

/// Coerce a JSON value into a positive token count, accepting both numbers
/// and numeric strings. Malformed values resolve to `None` (skipped).
fn token_value(v: &serde_json::Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
        .filter(|&n| n > 0)
}

/// List all configured model names in the OpenAI format.
///
/// With multiple deployments per model_name, duplicate names are
/// deduplicated. Wildcard (`*`) deployments are excluded. The returned list
/// is filtered by the caller's model ACL (key/team/user, via
/// [`check_model_acl`]); the master key sees every model. A virtual key
/// restricted to `gpt-4o` must not learn the full catalogue through
/// `GET /v1/models`.
pub async fn list_models(
    State(shared): State<SharedState>,
    auth: Extension<AuthUser>,
) -> Result<Response, ApiError> {
    let state = shared.load_full();
    let mut seen = HashSet::new();
    let data: Vec<serde_json::Value> = state
        .deployments
        .iter()
        .filter(|d| d.model_name != "*")
        .filter(|d| seen.insert(d.model_name.clone()))
        .filter(|d| {
            check_model_acl(
                &auth,
                &d.model_name,
                auth.key_models.as_deref(),
                auth.team_models.as_deref(),
                auth.user_models.as_deref(),
            )
            .is_ok()
        })
        .map(|d| {
            let (mi, mo) = resolve_token_limits(&state.deployments, &d.model_name);
            model_object(&d.model_name, mi, mo)
        })
        .collect();

    let body = json!({
    "object": "list",
    "data": data,
    });

    Ok(Json(body).into_response())
}

/// Retrieve a single configured model by name. H7: OpenAI clients (and
/// the reference `/v1/models/{model}` endpoint) expect this route to exist.
///
/// Looks up the model in the configured deployments. Wildcard (`*`)
/// deployments are excluded — they are not addressable by name. Returns
/// `404` when no deployment matches. Honors the caller's model ACL
/// ([`check_model_acl`]) like [`list_models`]; the master key bypasses.
///
/// The response shape is a single `model` object (not wrapped in a list),
/// matching OpenAI: `{ "id": "...", "object": "model", ... }`.
pub async fn retrieve_model(
    State(shared): State<SharedState>,
    Path(model): Path<String>,
    auth: Extension<AuthUser>,
) -> Result<Response, ApiError> {
    check_model_acl(
        &auth,
        &model,
        auth.key_models.as_deref(),
        auth.team_models.as_deref(),
        auth.user_models.as_deref(),
    )?;
    let state = shared.load_full();
    let found = state
        .deployments
        .iter()
        .filter(|d| d.model_name != "*")
        .any(|d| d.model_name == model);

    if !found {
        return Err(ApiError::not_found(format!("model '{model}' not found")));
    }

    let (mi, mo) = resolve_token_limits(&state.deployments, &model);
    let body = model_object(&model, mi, mo);
    Ok(Json(body).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::http::StatusCode;
    use oxllm_core::deployment::OxllmParams;

    fn state_with_models(names: &[&str]) -> AppState {
        let model_list = names
            .iter()
            .map(|n| crate::config::types::ConfigModel {
                model_name: (*n).to_string(),
                oxllm_params: OxllmParams {
                    model: format!("openai/{n}"),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect();
        let config = crate::config::types::ProxyConfig {
            model_list,
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    fn shared_from(state: AppState) -> SharedState {
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    /// An open (no-auth) principal with no model restrictions.
    fn ext_open() -> Extension<AuthUser> {
        Extension(AuthUser::open())
    }

    #[tokio::test]
    async fn retrieve_model_returns_model_object_when_present() {
        let state = state_with_models(&["gpt-4o", "claude-3"]);
        let shared = shared_from(state);
        let resp = retrieve_model(State(shared), Path("gpt-4o".to_string()), ext_open())
            .await
            .expect("present");
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["id"], "gpt-4o");
        assert_eq!(v["object"], "model");
    }

    #[tokio::test]
    async fn retrieve_model_returns_404_when_absent() {
        let state = state_with_models(&["gpt-4o"]);
        let shared = shared_from(state);
        let err = retrieve_model(State(shared), Path("made-up".to_string()), ext_open())
            .await
            .expect_err("not found");
        assert_eq!(err.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn retrieve_model_ignores_wildcard_deployments() {
        let state = state_with_models(&["*"]);
        let shared = shared_from(state);
        let err = retrieve_model(State(shared), Path("*".to_string()), ext_open())
            .await
            .expect_err("wildcard not addressable");
        assert_eq!(err.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_models_returns_deduplicated_set() {
        let state = state_with_models(&["gpt-4o", "gpt-4o", "claude-3"]);
        let shared = shared_from(state);
        let resp = list_models(State(shared), ext_open()).await.expect("ok");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let ids: Vec<&str> = v["data"]
            .as_array()
            .expect("array")
            .iter()
            .map(|d| d["id"].as_str().expect("id"))
            .collect();
        assert_eq!(ids, vec!["gpt-4o", "claude-3"]);
    }

    // ---- model ACL filtering (reference parity) -----------------------------

    fn restricted_auth(models: &[&str]) -> AuthUser {
        let mut auth = AuthUser::open();
        auth.key_models = Some(models.iter().map(|m| m.to_string()).collect());
        auth
    }

    #[tokio::test]
    async fn list_models_filters_by_key_model_acl() {
        let state = state_with_models(&["gpt-4o", "gpt-4o-mini", "claude-3"]);
        let shared = shared_from(state);
        let resp = list_models(State(shared), Extension(restricted_auth(&["gpt-4o"])))
            .await
            .expect("ok");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let ids: Vec<&str> = v["data"]
            .as_array()
            .expect("array")
            .iter()
            .map(|d| d["id"].as_str().expect("id"))
            .collect();
        assert_eq!(
            ids,
            vec!["gpt-4o"],
            "a key restricted to gpt-4o must not see non-matching model ids"
        );
    }

    #[tokio::test]
    async fn list_models_acl_supports_wildcards() {
        let state = state_with_models(&["gpt-4o", "gpt-4-turbo", "claude-3"]);
        let shared = shared_from(state);
        let resp = list_models(State(shared), Extension(restricted_auth(&["gpt-4*"])))
            .await
            .expect("ok");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let ids: Vec<&str> = v["data"]
            .as_array()
            .expect("array")
            .iter()
            .map(|d| d["id"].as_str().expect("id"))
            .collect();
        assert_eq!(ids, vec!["gpt-4o", "gpt-4-turbo"]);
    }

    #[tokio::test]
    async fn list_models_master_bypasses_acl() {
        let state = state_with_models(&["gpt-4o", "claude-3"]);
        let shared = shared_from(state);
        let mut auth = restricted_auth(&["gpt-4o"]);
        auth.is_master = true;
        let resp = list_models(State(shared), Extension(auth))
            .await
            .expect("ok");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let ids: Vec<&str> = v["data"]
            .as_array()
            .expect("array")
            .iter()
            .map(|d| d["id"].as_str().expect("id"))
            .collect();
        assert_eq!(
            ids,
            vec!["gpt-4o", "claude-3"],
            "master-key callers must see every model"
        );
    }

    #[tokio::test]
    async fn retrieve_model_denies_non_matching_acl() {
        let state = state_with_models(&["gpt-4o", "claude-3"]);
        let shared = shared_from(state);
        let err = retrieve_model(
            State(shared),
            Path("claude-3".to_string()),
            Extension(restricted_auth(&["gpt-4o"])),
        )
        .await
        .expect_err("not allowed");
        assert_eq!(
            err.status,
            axum::http::StatusCode::FORBIDDEN,
            "retrieving a model outside the key's ACL must be forbidden"
        );
    }

    // ---- token limits (U54 / U55) ----------------------------------------

    fn state_with_config_model(cm: crate::config::types::ConfigModel) -> AppState {
        let config = crate::config::types::ProxyConfig {
            model_list: vec![cm],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    fn config_model(name: &str, upstream: &str) -> crate::config::types::ConfigModel {
        crate::config::types::ConfigModel {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: upstream.to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn model_info_with(extra: &[(&str, serde_json::Value)]) -> oxllm_core::deployment::ModelInfo {
        let mut info = oxllm_core::deployment::ModelInfo::default();
        for (k, v) in extra {
            info.extra.insert((*k).to_string(), v.clone());
        }
        info
    }

    #[tokio::test]
    async fn retrieve_model_sources_token_limits_from_price_table() {
        // gpt-4o is in the vendored price table with both token limits.
        let state = state_with_config_model(config_model("gpt-4o", "openai/gpt-4o"));
        let shared = shared_from(state);
        let resp = retrieve_model(State(shared), Path("gpt-4o".to_string()), ext_open())
            .await
            .expect("present");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let mi = v["max_input_tokens"].as_u64().expect("max_input_tokens");
        let mo = v["max_output_tokens"].as_u64().expect("max_output_tokens");
        assert!(mi > 0, "price-table max_input_tokens must be positive");
        assert!(mo > 0, "price-table max_output_tokens must be positive");
    }

    #[tokio::test]
    async fn retrieve_model_resolves_alias_via_upstream_model_id() {
        // Client alias "fast" is not priced, but its upstream id is.
        let state = state_with_config_model(config_model("fast", "openai/gpt-4o"));
        let shared = shared_from(state);
        let resp = retrieve_model(State(shared), Path("fast".to_string()), ext_open())
            .await
            .expect("present");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(
            v.get("max_input_tokens")
                .and_then(|x| x.as_u64())
                .unwrap_or(0)
                > 0,
            "alias should resolve token limits via the upstream model id"
        );
    }

    #[tokio::test]
    async fn retrieve_model_falls_back_to_config_token_limits() {
        // A model absent from the price table but configured with limits.
        let mut cm = config_model("custom-internal", "madeup/custom-internal");
        cm.model_info = Some(model_info_with(&[
            ("max_input_tokens", json!(8192)),
            ("max_output_tokens", json!(1024)),
        ]));
        let state = state_with_config_model(cm);
        let shared = shared_from(state);
        let resp = retrieve_model(
            State(shared),
            Path("custom-internal".to_string()),
            ext_open(),
        )
        .await
        .expect("present");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["max_input_tokens"], 8192);
        assert_eq!(v["max_output_tokens"], 1024);
    }

    #[tokio::test]
    async fn retrieve_model_skips_malformed_config_token_limits() {
        // Malformed values (string, empty string, null) must not appear and
        // must not crash (U54). Only the well-formed output limit survives.
        let mut cm = config_model("custom-bad", "madeup/custom-bad");
        cm.model_info = Some(model_info_with(&[
            ("max_input_tokens", json!("not-a-number")),
            ("max_output_tokens", json!(" 2048 ")),
        ]));
        let state = state_with_config_model(cm);
        let shared = shared_from(state);
        let resp = retrieve_model(State(shared), Path("custom-bad".to_string()), ext_open())
            .await
            .expect("present");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(
            v.get("max_input_tokens").is_none(),
            "malformed max_input_tokens must be skipped"
        );
        assert_eq!(
            v["max_output_tokens"].as_u64(),
            Some(2048),
            "numeric-string max_output_tokens should parse"
        );
    }

    #[tokio::test]
    async fn retrieve_model_omits_token_limits_when_unknown() {
        // No price-table entry and no config → fields simply absent.
        let state = state_with_config_model(config_model("lonely", "madeup/lonely"));
        let shared = shared_from(state);
        let resp = retrieve_model(State(shared), Path("lonely".to_string()), ext_open())
            .await
            .expect("present");
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(v.get("max_input_tokens").is_none());
        assert!(v.get("max_output_tokens").is_none());
    }

    #[test]
    fn token_value_accepts_numbers_and_numeric_strings() {
        assert_eq!(token_value(&json!(0)), None, "zero is not a usable limit");
        assert_eq!(token_value(&json!(4096)), Some(4096));
        assert_eq!(token_value(&json!("8192")), Some(8192));
        assert_eq!(token_value(&json!("  100  ")), Some(100));
        assert_eq!(token_value(&json!("")), None);
        assert_eq!(token_value(&json!("abc")), None);
        assert_eq!(token_value(&json!(null)), None);
        assert_eq!(token_value(&json!(true)), None);
        assert_eq!(token_value(&json!([1])), None);
    }

    #[test]
    fn config_token_limits_reads_model_info_then_oxllm_params() {
        let mut cm = config_model("x", "madeup/x");
        cm.model_info = Some(model_info_with(&[("max_input_tokens", json!(1000))]));
        cm.oxllm_params
            .extra
            .insert("max_output_tokens".to_string(), json!(2000));
        let dep: Deployment = cm.into();
        let (mi, mo) = config_token_limits(&dep);
        assert_eq!(mi, Some(1000));
        assert_eq!(mo, Some(2000));
    }
}
