//! Pass-through routes: configurable catch-all prefixes that forward to
//! external targets, plus the global provider catch-all `/{*path}`.
//!
//! Each pass-through endpoint inherits the standard proxy auth stack (auth,
//! allowed-IPs, auth-brake). The global catch-all is registered last so it
//! only matches paths not handled by any explicit route.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::handler::Handler;
use axum::response::{IntoResponse, Response};
use axum::routing::any;

use crate::config::types::PassThroughEndpoint;
use crate::error::ApiError;
use crate::handlers::passthrough;
use crate::pipeline::passthrough::{DEFAULT_TIMEOUT_SECS, path_within_prefix};
use crate::state::SharedState;

/// Register each `general_settings.pass_through_endpoints` entry as a
/// catch-all route that forwards requests to the configured target URL. Each
/// route inherits the same auth + allowed-ips + auth-brake layers as the
/// rest of the proxy.
pub(crate) fn attach_endpoints(
    router: Router<SharedState>,
    shared: SharedState,
) -> Router<SharedState> {
    let endpoints: Vec<PassThroughEndpoint> = shared
        .load_full()
        .config
        .general_settings
        .pass_through_endpoints
        .clone()
        .unwrap_or_default();
    if endpoints.is_empty() {
        return router;
    }
    let mut seen: HashSet<String> = HashSet::new();
    for ep in endpoints {
        let path = match pass_through_route(&ep.path) {
            Ok(p) => p,
            Err(reason) => {
                tracing::warn!(
                    path = %ep.path,
                    target = %crate::security::sanitize_url(&ep.target),
                    reason,
                    "skipping misconfigured pass_through_endpoints entry"
                );
                continue;
            }
        };
        if !seen.insert(path.clone()) {
            tracing::warn!(
                path = %path,
                target = %crate::security::sanitize_url(&ep.target),
                "skipping duplicate pass_through_endpoints entry (normalised path already registered)"
            );
            continue;
        }
    }
    // Prefix dispatch lives inside the already-authenticated global catch-all.
    // Registering another wildcard beside a parameter route can panic during
    // merge; explicit API routes retain precedence over configured prefixes.
    router
}

/// Register the global provider catch-all `/{*path}`. The returned router
/// carries no auth layers — `build_router` applies the standard proxy auth
/// stack.
pub(crate) fn catch_all_routes() -> Router<SharedState> {
    Router::new().route("/{*path}", any(dispatch_passthrough))
}

async fn dispatch_passthrough(
    State(shared): State<SharedState>,
    req: axum::extract::Request,
) -> Response {
    let state = shared.load_full();
    let target = state
        .config
        .general_settings
        .pass_through_endpoints
        .as_ref()
        .into_iter()
        .flatten()
        .filter(|ep| pass_through_route(&ep.path).is_ok())
        .filter(|ep| path_within_prefix(req.uri().path(), ep.path.trim_end_matches('/')))
        .min_by_key(|ep| std::cmp::Reverse(ep.path.trim_end_matches('/').len()))
        .map(|ep| PassThroughTarget {
            target_url: ep.target.clone(),
            headers: ep.headers.clone().unwrap_or_default(),
        });
    if let Some(target) = target {
        forward_to_target(req, shared, target).await
    } else {
        passthrough::provider_passthrough.call(req, shared).await
    }
}

/// Computed pass-through target: the base URL and merged headers, captured by
/// the route handler.
#[derive(Clone, Debug)]
struct PassThroughTarget {
    target_url: String,
    headers: HashMap<String, String>,
}

/// Compute the axum catch-all route pattern for a pass-through prefix, after
/// validating the prefix. The matched suffix is forwarded to the target.
///
/// Returns `Err(reason)` for inputs that would either collide with the global
/// catch-all `/{*path}` (registered separately in [`catch_all_routes`]) or
/// produce a malformed axum pattern. Concretely, an empty string and `/` both
/// normalise to `/{*path}` and would panic at `Router::merge` time, and a
/// prefix without a leading `/` is not a valid axum route. The caller is
/// expected to log a warning and skip the entry rather than abort startup —
/// the YAML config is caller input and must never cause a panic.
fn pass_through_route(prefix: &str) -> Result<String, &'static str> {
    let trimmed = prefix.trim_end_matches('/');
    if trimmed.is_empty() {
        return Err("path must not be empty or '/'");
    }
    if !trimmed.starts_with('/') {
        return Err("path must start with '/'");
    }
    if trimmed.contains(['{', '}', '*']) {
        return Err("path must be a literal prefix, without route parameters");
    }
    Ok(format!("{trimmed}/{{*path}}"))
}

/// Safely combine the configured `target_url` with the inbound path suffix,
/// rejecting inputs whose normalized form escapes the target's path prefix
/// (`..` traversal — #28).
///
/// Both sides are parsed with `url::Url`, which normalizes dot-segments per
/// RFC 3986. After normalization we verify the final path still begins with
/// the target's configured path (boundary-aware: `/api` matches `/api/...`
/// but not `/api2`). When the target has no path component (the common case
/// — a bare `https://host`), any suffix is allowed since there is no prefix
/// to escape.
fn build_safe_target_url(target_url: &str, suffix: &str) -> Result<String, ApiError> {
    let parsed_target = url::Url::parse(target_url)
        .map_err(|e| ApiError::internal(format!("invalid pass-through target URL: {e}")))?;
    let target_path = parsed_target.path();

    let (suffix_path, suffix_query) = suffix
        .split_once('?')
        .map_or((suffix, None), |(path, query)| (path, Some(query)));
    let mut parsed_final = parsed_target.clone();
    parsed_final.set_path(&format!(
        "{}{suffix_path}",
        target_path.trim_end_matches('/')
    ));
    if let Some(query) = suffix_query {
        let combined = match parsed_target.query() {
            Some(configured) => format!("{configured}&{query}"),
            None => query.to_string(),
        };
        parsed_final.set_query(Some(&combined));
    }
    let final_path = parsed_final.path();

    if !path_within_prefix(final_path, target_path) {
        return Err(ApiError::forbidden(
            "pass-through path escapes the configured target prefix",
        ));
    }
    Ok(parsed_final.to_string())
}

/// Forward a request to the configured pass-through target. The path suffix
/// (everything after the matched prefix) is appended to the target URL, and
/// the configured `headers` are merged on top of the inbound headers.
///
/// # Security
///
/// The inbound suffix is normalized via [`build_safe_target_url`] before
/// forwarding so `..` segments cannot escape the target's path prefix and
/// hit other endpoints with the configured secret headers attached (#28).
/// The resolved target host is re-checked via
/// [`crate::security::is_private_host_async`] immediately before the upstream
/// send to catch DNS rebinding (unless `allow_private_api_base` is set).
async fn forward_to_target(
    req: axum::extract::Request,
    shared: SharedState,
    target: PassThroughTarget,
) -> Response {
    let app_state = shared.load_full();
    let (parts, body) = req.into_parts();
    // Use the configured body limit (default 10 MB) instead of a hardcoded
    // constant — `general_settings.max_request_body_size_mb` already drives
    // the global body-limit middleware, so pass-through honors it too.
    let body_limit = crate::security::body_limit_bytes(
        app_state.config.general_settings.max_request_body_size_mb,
    );
    let mut body_bytes = match axum::body::to_bytes(body, body_limit).await {
        Ok(b) => b,
        Err(e) => {
            return ApiError::internal(format!("failed to read request body: {e}")).into_response();
        }
    };

    if crate::pipeline::guardrails::request_guardrails_enabled(&app_state) {
        let content_type = parts
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok());
        let mut request_body = if crate::pipeline::passthrough::is_json_content_type(content_type)
            && !body_bytes.is_empty()
        {
            match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                Ok(body) => body,
                Err(_) => return ApiError::bad_request("invalid JSON body").into_response(),
            }
        } else {
            serde_json::Value::Null
        };
        if let Err(err) =
            crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)
        {
            return err.into_response();
        }
        match crate::pipeline::guardrails::apply_raw_request_guardrails(
            &app_state,
            &parts.method,
            parts.uri.path(),
            &mut request_body,
        )
        .await
        {
            Ok(true) => body_bytes = axum::body::Bytes::from(request_body.to_string()),
            Ok(false) => {}
            Err(err) => return err.into_response(),
        }
    }

    let prefix = uri_prefix(&target.target_url);
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    let suffix = path_and_query
        .strip_prefix(&prefix)
        .unwrap_or(path_and_query);
    let _ = prefix;

    let target_url = match build_safe_target_url(&target.target_url, suffix) {
        Ok(u) => u,
        Err(e) => return e.into_response(),
    };

    // Defense-in-depth against DNS rebinding of the configured target: even
    // though the target URL is operator-configured (the client-controlled
    // suffix cannot change the host), a hostname that resolved to a public IP
    // at config-load time may resolve to a private IP at request time. Re-check
    // the resolved host immediately before going on the wire, mirroring the
    // orchestration upstream path (`orchestration/upstream.rs:122`). Operators
    // that legitimately target private ranges opt out via
    // `general_settings.allow_private_api_base`.
    if !app_state
        .config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false)
    {
        if let Some(host) = url::Url::parse(&target_url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h.to_string()))
        {
            if crate::security::is_private_host_async(&host).await {
                tracing::warn!(
                    target_host = %host,
                    "rejecting pass-through target: host resolves to a private IP address"
                );
                return ApiError::forbidden(
                    "pass-through target host resolves to a private IP address",
                )
                .into_response();
            }
        }
    }

    let mut headers = parts.headers;
    strip_forwarded_headers(&mut headers);
    for (k, v) in &target.headers {
        if let (Ok(name), Ok(value)) = (
            axum::http::HeaderName::from_bytes(k.as_bytes()),
            axum::http::HeaderValue::from_str(v),
        ) {
            headers.insert(name, value);
        }
    }

    let timeout_secs = app_state
        .config
        .oxllm_settings
        .request_timeout
        .unwrap_or(DEFAULT_TIMEOUT_SECS);
    let upstream = app_state
        .http_client
        .request(parts.method, &target_url)
        .headers(headers)
        .timeout(Duration::from_secs(timeout_secs))
        .body(reqwest::Body::from(body_bytes))
        .send()
        .await;
    let upstream = match upstream {
        Ok(r) => r,
        Err(e) => {
            if e.is_timeout() {
                tracing::warn!(
                    error = %e,
                    url = %crate::security::sanitize_url(&target_url),
                    timeout_secs,
                    "pass-through request timed out"
                );
                return ApiError::internal(format!(
                    "pass-through upstream timeout after {timeout_secs}s"
                ))
                .into_response();
            }
            tracing::warn!(
            error = %e,
            url = %crate::security::sanitize_url(&target_url),
            "pass-through request failed"
            );
            return ApiError::bad_request(format!("pass-through upstream error: {e}"))
                .into_response();
        }
    };
    match crate::security::sanitize_upstream_response(upstream).await {
        Ok(resp) => resp,
        Err(e) => e.into_response(),
    }
}

/// Returns the path prefix to strip from the inbound URI before forwarding.
///
/// The pass-through route's matched `path` config field is not available here
/// (axum doesn't surface it), so we treat the entire inbound path as the
/// suffix and forward it verbatim. This is a deliberate divergence from
/// LiteLLM's `construct_target_url_with_subpath`
/// (`pass_through_endpoints.py:589-624`), which strips the matched route
/// prefix and appends only the captured subpath (when `include_subpath=True`).
/// oxllm forwards the full inbound path instead — e.g. a request to
/// `/v1/ep/foo` with target `https://api.example.com` reaches
/// `https://api.example.com/v1/ep/foo`, not `https://api.example.com/foo`.
/// The oxllm behaviour is exercised by the integration test
/// `pass_through_endpoint_forwards_to_target` (`callbacks_ops_e2e.rs`).
fn uri_prefix(_target_url: &str) -> String {
    String::new()
}

/// Drop hop-by-hop and secret-bearing headers so the client's proxy credentials
/// (Authorization, cookies, API keys) are never forwarded to the configured
/// target. Explicit `target.headers` are re-applied after this strip.
fn strip_forwarded_headers(headers: &mut axum::http::HeaderMap) {
    const STRIP: &[&str] = &[
        "host",
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "te",
        "trailer",
        "upgrade",
        "content-length",
    ];
    for name in STRIP {
        headers.remove(*name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- pass_through_route -----------------------------------------------

    #[test]
    fn pass_through_route_adds_catch_all_with_trailing_slash() {
        assert_eq!(pass_through_route("/v1/ep/").unwrap(), "/v1/ep/{*path}");
    }

    #[test]
    fn pass_through_route_adds_catch_all_without_trailing_slash() {
        assert_eq!(pass_through_route("/v1/ep").unwrap(), "/v1/ep/{*path}");
    }

    #[test]
    fn pass_through_route_normalises_root_trailing_slash_to_same_pattern() {
        // `/foo` and `/foo/` must produce the SAME normalised pattern so the
        // de-duplication in `attach_endpoints` can detect the collision.
        assert_eq!(
            pass_through_route("/foo").unwrap(),
            pass_through_route("/foo/").unwrap()
        );
    }

    #[test]
    fn pass_through_route_rejects_empty_path() {
        assert_eq!(
            pass_through_route("").unwrap_err(),
            "path must not be empty or '/'"
        );
    }

    #[test]
    fn pass_through_route_rejects_root_path() {
        // `/` normalises to `/{*path}` and would collide with the global
        // catch-all registered by `catch_all_routes` — must be rejected.
        assert_eq!(
            pass_through_route("/").unwrap_err(),
            "path must not be empty or '/'"
        );
    }

    #[test]
    fn pass_through_route_rejects_path_without_leading_slash() {
        assert_eq!(
            pass_through_route("foo").unwrap_err(),
            "path must start with '/'"
        );
    }

    // ---- path_within_prefix ------------------------------------------------

    #[test]
    fn path_within_prefix_empty_target_allows_anything() {
        assert!(path_within_prefix("/anything", ""));
        assert!(path_within_prefix("/", ""));
    }

    #[test]
    fn path_within_prefix_root_target_allows_anything() {
        assert!(path_within_prefix("/anything", "/"));
    }

    #[test]
    fn path_within_prefix_exact_match() {
        assert!(path_within_prefix("/api", "/api"));
    }

    #[test]
    fn path_within_prefix_child_path_matches() {
        assert!(path_within_prefix("/api/foo", "/api"));
        assert!(path_within_prefix("/api/foo/bar", "/api"));
    }

    #[test]
    fn path_within_prefix_rejects_sibling_prefix() {
        // /api2 shares the literal prefix `/api` but is a different path
        // segment — must not be treated as inside /api.
        assert!(!path_within_prefix("/api2", "/api"));
        assert!(!path_within_prefix("/api2/secret", "/api"));
    }

    #[test]
    fn path_within_prefix_handles_trailing_slash_target() {
        assert!(path_within_prefix("/api/foo", "/api/"));
        assert!(path_within_prefix("/api", "/api/"));
    }

    // ---- build_safe_target_url: happy paths --------------------------------

    #[test]
    fn build_safe_url_appends_simple_suffix() {
        let url = build_safe_target_url("https://target.example.com", "/v1/foo").unwrap();
        assert_eq!(url, "https://target.example.com/v1/foo");
    }

    #[test]
    fn build_safe_url_preserves_query_string() {
        let url = build_safe_target_url(
            "https://target.example.com",
            "/v1/foo?api-version=2024-01-01&beta=true",
        )
        .unwrap();
        assert_eq!(
            url,
            "https://target.example.com/v1/foo?api-version=2024-01-01&beta=true"
        );
    }

    #[test]
    fn build_safe_url_appends_path_before_configured_query() {
        assert_eq!(
            build_safe_target_url(
                "https://target.example.com/api?key=secret",
                "/v1/foo?beta=true"
            )
            .unwrap(),
            "https://target.example.com/api/v1/foo?key=secret&beta=true"
        );
    }

    #[test]
    fn configured_prefix_does_not_conflict_with_parameter_routes() {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};
        let config = ProxyConfig {
            model_list: vec![],
            general_settings: GeneralSettings {
                pass_through_endpoints: Some(vec![PassThroughEndpoint {
                    path: "/anthropic/v1/files".into(),
                    target: "https://example.test".into(),
                    headers: None,
                }]),
                ..Default::default()
            },
            oxllm_settings: OxllmSettings::default(),
            router_settings: RouterSettings::default(),
            extra: Default::default(),
        };
        let shared = crate::app::build_shared_state(config).expect("state");
        let _ = crate::routes::build_router(shared);
    }

    #[test]
    fn build_safe_url_appends_to_target_with_path() {
        let url =
            build_safe_target_url("https://target.example.com/expected", "/v1/ep/foo").unwrap();
        assert_eq!(url, "https://target.example.com/expected/v1/ep/foo");
    }

    #[test]
    fn build_safe_url_normalizes_redundant_slashes_within_prefix() {
        // `..` that stays under the target prefix is fine — it just collapses.
        let url =
            build_safe_target_url("https://target.example.com/expected", "/a/b/../c").unwrap();
        assert_eq!(url, "https://target.example.com/expected/a/c");
    }

    // ---- build_safe_target_url: traversal attacks (#28) --------------------

    #[test]
    fn build_safe_url_rejects_traversal_escape_from_path_target() {
        // suffix `/a/../../../forbidden` joined with target `/expected`:
        //   /expected/a/../../../forbidden  →  /forbidden
        // The normalized result escapes `/expected` and must be rejected.
        let err = build_safe_target_url(
            "https://target.example.com/expected",
            "/a/../../../forbidden",
        )
        .expect_err("escape must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
    }

    #[test]
    fn build_safe_url_rejects_literal_dotdot_escape() {
        // `/v1/ep/../../../admin` joined with target `/secret-prefix`:
        //   /secret-prefix/v1/ep/../../../admin  →  /admin
        // Three `..` segments are needed to climb out of `/secret-prefix/v1/ep`
        // and then `/secret-prefix` itself.
        let err = build_safe_target_url(
            "https://target.example.com/secret-prefix",
            "/v1/ep/../../../admin",
        )
        .expect_err("escape must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
    }

    #[test]
    fn build_safe_url_rejects_root_escape_attempt() {
        // `/..` resolves to `/` which is outside any non-root target prefix.
        let err = build_safe_target_url("https://target.example.com/secret", "/..")
            .expect_err("escape must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
    }

    #[test]
    fn build_safe_url_allows_dotdot_that_stays_inside_prefix() {
        // /expected/v1/ep/../foo  →  /expected/v1/foo   (still under /expected)
        let url =
            build_safe_target_url("https://target.example.com/expected", "/v1/ep/../foo").unwrap();
        assert_eq!(url, "https://target.example.com/expected/v1/foo");
    }

    #[test]
    fn build_safe_url_allows_any_path_when_target_has_no_path_component() {
        // Bare host target — there is no prefix to escape, so any suffix
        // (including ones that normalize to `/`) is accepted.
        let url = build_safe_target_url("https://target.example.com", "/v1/ep/../../../x").unwrap();
        assert_eq!(url, "https://target.example.com/x");
    }

    #[test]
    fn build_safe_url_rejects_invalid_target_url() {
        let err = build_safe_target_url("not-a-url", "/foo").expect_err("invalid target");
        assert_eq!(err.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    }
}
