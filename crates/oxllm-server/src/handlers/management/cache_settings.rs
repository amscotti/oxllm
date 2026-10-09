//! Cache settings endpoints: `GET/POST /cache/settings`.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::AuthUser;
use crate::config::types::CacheConfig;
use crate::error::ApiError;
use crate::state::SharedState;

use super::{redact_secrets, require_master};

/// Map a cache-reload error into an internal `ApiError`. The full error chain
/// (driver text, malformed URLs with embedded credentials, backend constructor
/// messages) is logged server-side; only a generic message reaches the client
/// so a malformed `redis_url` like `redis://:hunter2@host:port` cannot leak its
/// password through the response body. Mirrors the discipline of
/// [`super::db_err`].
fn cache_apply_err(e: anyhow::Error) -> ApiError {
    tracing::warn!(error = %e, "failed to apply cache settings");
    ApiError::internal("failed to apply cache settings")
}

/// `GET /cache/settings` — return current cache config with secrets redacted.
pub async fn get_cache_settings(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let state = shared.load_full();
    let cache = state.config.oxllm_settings.cache.clone();
    let ttl = state.config.oxllm_settings.cache_ttl_secs;
    let mut value = serde_json::to_value(&cache)
        .map_err(|e| ApiError::internal(format!("failed to serialize cache config: {e}")))?;
    redact_secrets(&mut value);
    Ok(Json(json!({
        "cache": value,
        "cache_ttl_secs": ttl,
        "enabled": cache.is_some(),
    }))
    .into_response())
}

/// Request body for `POST /cache/settings`.
#[derive(Debug, Deserialize)]
pub struct UpdateCacheSettingsRequest {
    /// Replacement cache block. `null` clears cache config (falls back to
    /// in-memory defaults at next process start / when hooks re-read config).
    ///
    /// Mutually exclusive with [`clear_cache`](Self::clear_cache): sending both
    /// is rejected with 400, since the precedence between "install this cache"
    /// and "clear the cache" is ambiguous and silently picking one would
    /// surprise an operator PATCHing both fields.
    #[serde(default)]
    pub cache: Option<CacheConfig>,
    /// When true, set `cache` to `None`. Mutually exclusive with
    /// [`cache`](Self::cache) (see its doc-comment).
    #[serde(default)]
    pub clear_cache: bool,
    /// Replacement TTL in seconds. Mutually exclusive with
    /// [`clear_cache_ttl_secs`](Self::clear_cache_ttl_secs).
    #[serde(default)]
    pub cache_ttl_secs: Option<u64>,
    /// When true, set `cache_ttl_secs` to `None`. Mutually exclusive with
    /// [`cache_ttl_secs`](Self::cache_ttl_secs).
    #[serde(default)]
    pub clear_cache_ttl_secs: bool,
}

/// `POST /cache/settings` — update in-memory cache config via hot-reload.
///
/// Updates `oxllm_settings.cache` / `cache_ttl_secs` on the shared config
/// and swaps state through [`crate::state::AppState::reload_from_config`]. The live
/// `Arc<dyn Cache>` handle is preserved across reload (same as file watcher
/// reloads); switching Redis host/type takes full effect on process restart.
/// Semantic-cache config changes that pass the equality check keep the
/// existing embedding runtime; other changes rebuild hooks.
pub async fn update_cache_settings(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateCacheSettingsRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    // Reject contradictory field combinations up front. Previously the
    // `clear_*` flags silently won over an inline value, so an operator
    // PATCHing both `clear_cache: true` and a populated `cache` ended up with
    // caching disabled and no error.
    if req.clear_cache && req.cache.is_some() {
        return Err(ApiError::bad_request(
            "cannot set both `clear_cache` and `cache`",
        ));
    }
    if req.clear_cache_ttl_secs && req.cache_ttl_secs.is_some() {
        return Err(ApiError::bad_request(
            "cannot set both `clear_cache_ttl_secs` and `cache_ttl_secs`",
        ));
    }
    let current = shared.load_full();
    let mut new_config = (*current.config).clone();

    if req.clear_cache {
        new_config.oxllm_settings.cache = None;
    } else if let Some(cache) = req.cache {
        new_config.oxllm_settings.cache = Some(cache);
    }
    if req.clear_cache_ttl_secs {
        new_config.oxllm_settings.cache_ttl_secs = None;
    } else if req.cache_ttl_secs.is_some() {
        new_config.oxllm_settings.cache_ttl_secs = req.cache_ttl_secs;
    }

    let reloaded = current
        .reload_from_config(new_config)
        .map_err(cache_apply_err)?;
    shared.store(Arc::new(reloaded));

    let state = shared.load_full();
    let mut value = serde_json::to_value(&state.config.oxllm_settings.cache)
        .map_err(|e| ApiError::internal(format!("failed to serialize cache config: {e}")))?;
    redact_secrets(&mut value);
    Ok(Json(json!({
        "status": "ok",
        "cache": value,
        "cache_ttl_secs": state.config.oxllm_settings.cache_ttl_secs,
        "note": "cache backend Arc is preserved across reload; Redis host/type changes apply on restart",
    }))
    .into_response())
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::*;
    use crate::handlers::management::REDACTED;
    use serde_json::json;

    #[test]
    fn redact_secrets_hides_api_key_and_redis_url() {
        let mut v = json!({
            "type": "redis",
            "host": "localhost",
            "semantic_cache": {
                "api_key": "sk-secret",
                "redis_url": "redis://:pass@localhost:6379",
                "enabled": true
            }
        });
        redact_secrets(&mut v);
        assert_eq!(v["semantic_cache"]["api_key"], REDACTED);
        assert_eq!(v["semantic_cache"]["redis_url"], REDACTED);
        assert_eq!(v["host"], "localhost");
        assert_eq!(v["semantic_cache"]["enabled"], true);
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    // ---- C30 MEDIUM: cache reload error must not leak url/password ---------
    //
    // `reload_from_config` returns an `anyhow::Error` whose chain can include
    // driver/URL parser text (e.g. a malformed `redis_url` with an embedded
    // password). The client-facing message must be generic; the chain is logged
    // server-side only. Mirrors the `db_err` discipline.

    #[test]
    fn cache_apply_err_does_not_leak_error_chain() {
        let raw = anyhow::anyhow!("invalid redis url: redis://:pa$$w0rd@bad-host:99999");
        let err = cache_apply_err(raw);
        assert_eq!(err.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.message, "failed to apply cache settings");
        assert!(
            !err.message.contains("pa$$w0rd"),
            "password must not leak into client-facing message"
        );
        assert!(
            !err.message.contains("99999"),
            "raw url fragment must not leak into client-facing message"
        );
        assert!(
            !err.message.contains("redis"),
            "driver name must not leak into client-facing message"
        );
    }
}
