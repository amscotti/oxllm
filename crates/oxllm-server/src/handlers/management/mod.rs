//! Management REST endpoints (keys, teams, users, spend, models).
//!
//! These require the `db` feature and a configured `database_url`. All routes
//! require master-key authentication (the `management_auth_middleware`).

pub mod audit;
pub mod budgets;
pub mod cache_settings;
pub mod config;
pub mod customers;
pub mod keys;
pub mod models_mgmt;
pub mod orgs;
pub mod projects;
pub mod routing;
pub mod skills;
pub mod spend;
pub mod tags;
pub mod teams;
pub mod users;

use std::sync::Arc;

use axum::Extension;
use rand::Rng;
use serde_json::Value;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::state::{AppState, SharedState};

/// Reject the request unless the caller is the master key.
pub(crate) fn require_master(auth: &AuthUser) -> Result<(), ApiError> {
    if auth.is_master {
        Ok(())
    } else {
        Err(ApiError::forbidden("master key required"))
    }
}

/// Load the current state snapshot and its database handle.
pub(crate) async fn load_db(
    shared: &SharedState,
    _auth: &AuthUser,
) -> Result<(Arc<AppState>, Arc<oxllm_db::Database>), ApiError> {
    let state = shared.load_full();
    let db = state
        .db
        .clone()
        .ok_or_else(|| ApiError::internal("database not configured"))?;
    Ok((state, db))
}

/// Map a database error into an internal `ApiError`. The full error chain
/// is logged server-side; only a generic message reaches the client so
/// driver-specific text (constraint names, table/column names, query
/// fragments) never leaks through the response body.
pub(crate) fn db_err(e: anyhow::Error) -> ApiError {
    tracing::warn!(
        error = %oxllm_core::redaction::redact_connection_text(&format!("{e:#}")),
        "database error"
    );
    ApiError::internal("database error")
}

/// Detect a unique-constraint (PRIMARY KEY / UNIQUE) violation across the
/// supported backends. Used by the create handlers to surface a
/// TOCTOU-shaped concurrent same-id insert as a friendly 400 "already
/// exists" rather than the generic 500 from [`db_err`].
///
/// Prefer [`sqlx::error::DatabaseError::is_unique_violation`] (which both
/// sqlx 0.8 backends implement via `kind()`), with a code/message fallback
/// in case a future driver does not populate the structured kind.
pub(crate) fn is_unique_violation(e: &anyhow::Error) -> bool {
    let Some(sqlx_err) = e.downcast_ref::<sqlx::Error>() else {
        return false;
    };
    let sqlx::Error::Database(db) = sqlx_err else {
        return false;
    };
    if db.is_unique_violation() {
        return true;
    }
    matches!(
        db.code().as_deref(),
        Some("23505") | Some("1555") | Some("2067")
    ) || db.message().contains("UNIQUE constraint")
}

/// Map a database error from a create handler: a unique-constraint failure
/// becomes `on_exists`, every other failure goes through [`db_err`]. Used to
/// turn the TOCTOU window between the existence pre-check and the `INSERT`
/// (two concurrent creates with the same caller-chosen id) into the same
/// 400 the pre-check would have produced.
pub(crate) fn map_create_err<F>(e: anyhow::Error, on_exists: F) -> ApiError
where
    F: FnOnce() -> ApiError,
{
    if is_unique_violation(&e) {
        on_exists()
    } else {
        db_err(e)
    }
}

/// Keys whose values must never be exposed through the management API. Their
/// values are replaced with [`REDACTED`] before any config-shaped document is
/// serialized into a response.
///
/// Matching is a case-insensitive **substring** match against the lower-cased
/// key name, so vendor-prefixed keys like `aws_secret_access_key`,
/// `openai_api_key`, and `qdrant_api_key` are all covered by the bare entries
/// below. Substring matching is only applied to **string** values (see
/// [`should_redact`]); a numeric leaf such as `input_cost_per_token` is never
/// a credential, so the bare `token` entry cannot corrupt the pricing fields
/// that happen to contain the substring "token".
///
/// **Known over-match (fail-safe):** the bare `token` entry (meant to catch
/// `access_token`, `aws_session_token`, `refresh_token`, …) also redacts
/// benign string-valued fields whose name merely contains "token" — e.g. a
/// custom `tokenizer_model: "bpe"`. This is fail-safe (it hides data, never
/// leaks it) and intentional: numeric token fields are already protected by
/// the string-only restriction, and the cost of a false positive (a harmless
/// string disappears from a management read) is lower than the cost of a false
/// negative (a credential leaks). Operators who see a field redacted that
/// should not be should rename it to avoid the substring.
///
/// This is the single shared denylist for every management read path
/// (`GET /config`, `GET /cache/settings`, and `POST /model/update`'s
/// response). Previously `config.rs` and `cache_settings.rs` each kept their
/// own narrower list, which let cache-specific secrets (`redis_url`) leak via
/// `/config` and config-specific secrets (`secret_access_key`,
/// `private_key`, …) leak via `/cache/settings`.
pub(crate) const REDACTED_KEYS: &[&str] = &[
    "api_key",
    "master_key",
    "secret_key",        // Langfuse secret key
    "secret_access_key", // AWS Bedrock
    "access_key_id",     // AWS access IDs are credential material too
    "session_token",     // AWS Bedrock
    "private_key",
    "password",
    "token",
    "authorization",
    "redis_url",            // cache backend connection string (embeds credentials)
    "jwt_secret",           // JWT signing key (token forgery if leaked)
    "database_url",         // Postgres/MySQL URLs embed `user:password@host`
    "connection_string",    // Azure Blob strings embed `AccountKey=…`
    "service_account_json", // GCS service-account blob (embeds a private_key)
    "secret",               // bare: client_secret, signing_secret, …
    "credential",           // azure_credential, google_credentials, …
    "apikey",               // kebab/camel-free spelling: x-apikey, apikey
    "cert",                 // client_cert, ssl_cert, certificate bundles
    "routing_key",          // PagerDuty Events API credential
    "webhook_url",          // Slack and other path-bearing capability URLs
];

/// Placeholder shown in place of a secret value.
pub(crate) const REDACTED: &str = "***";

/// Value heuristic: any string value beginning with this prefix is treated as
/// a leaked OpenAI-style API key and redacted regardless of its key name.
const SECRET_VALUE_PREFIX: &str = "sk-";

/// Returns `true` when `key` names a container whose **every** string value
/// is secret regardless of the nested field names. A header map sent to a
/// provider carries auth material under arbitrary header names
/// (`X-Backend-Key`, `X-Api-Token`, …) that the [`REDACTED_KEYS`] substring
/// denylist can never enumerate, so the whole container is treated as
/// secret. `extra_headers` qualifies anywhere; `headers` also qualifies
/// under an `extra_body` or guardrail `config` object. This predicate is shared
/// with model encryption/decryption: do not classify every `headers` object
/// as an all-secret container. Management redaction separately hides opaque
/// header values while allowing well-formed public content negotiation.
pub(crate) fn is_secret_container(key: &str, parent: Option<&str>) -> bool {
    let lower = key.to_ascii_lowercase();
    lower == "extra_headers"
        || (lower == "headers"
            && parent.is_some_and(|parent| {
                parent.eq_ignore_ascii_case("extra_body") || parent.eq_ignore_ascii_case("config")
            }))
}

/// Recursively replace the value of any sensitive key with [`REDACTED`].
///
/// A value is redacted when either:
/// - it is a **string** whose key name (lower-cased) is in [`REDACTED_KEYS`]
///   (substring match), or
/// - it is a string whose content starts with [`SECRET_VALUE_PREFIX`] (`sk-`),
///   or
/// - it is a string nested anywhere inside a secret container
///   ([`is_secret_container`]), regardless of its field name.
///
/// Header maps outside those containers use a conservative public-value
/// allowlist; URL fields keep endpoint context but lose embedded credentials.
///
/// `null` values are left untouched (they carry no secret material and help
/// callers distinguish "absent" from "redacted"). Non-string leaves (numbers,
/// booleans) are never redacted by the denylist: credentials are always
/// strings, and restricting to strings stops the bare `token` entry from
/// clobbering numeric pricing fields like `input_cost_per_token`.
///
/// Recursion is capped at [`REDACT_MAX_DEPTH`] levels so an adversarial,
/// deeply-nested `oxllm_params.extra` value (within the 10 MiB body limit)
/// cannot exhaust the stack.
pub(crate) fn redact_secrets(value: &mut Value) {
    redact_secrets_inner(value, 0, None, false);
}

/// Maximum nesting depth [`redact_secrets`] descends into before stopping.
/// The config types serialise to shallow shapes (a handful of levels); this
/// only exists as a backstop against adversarial nesting in caller-provided
/// `extra` maps. Set to match `serde_json`'s own default recursion limit (128)
/// so a secret nested just under the cap cannot survive redaction — the stack
/// cost at these depths is negligible.
pub(crate) const REDACT_MAX_DEPTH: u32 = 128;

fn redact_secrets_inner(
    value: &mut Value,
    depth: u32,
    parent: Option<&str>,
    secret_container: bool,
) {
    if depth >= REDACT_MAX_DEPTH {
        return;
    }
    match value {
        Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                let secret =
                    secret_container || is_secret_container(k, parent) || should_redact(k, v);
                if secret {
                    match v {
                        Value::String(_) => *v = Value::String(REDACTED.to_string()),
                        other => redact_secrets_inner(other, depth + 1, Some(k), true),
                    }
                } else if k.eq_ignore_ascii_case("headers") {
                    redact_headers(v);
                } else if let Value::String(url) = v
                    && is_url_field(k)
                    && !url.is_empty()
                {
                    // Safe values retain their exact serialized spelling; URL
                    // normalization is not itself a config mutation/redaction.
                    if url::Url::parse(url).is_err()
                        || oxllm_core::redaction::redact_connection_text(url) != *url
                    {
                        *url = crate::security::sanitize_url(url);
                    }
                } else {
                    redact_secrets_inner(v, depth + 1, Some(k), false);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_secrets_inner(item, depth + 1, parent, secret_container);
            }
        }
        Value::String(s) if secret_container => *s = REDACTED.to_string(),
        _ => {}
    }
}

fn is_url_field(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key == "url"
        || key.ends_with("_url")
        || matches!(
            key.as_str(),
            "api_base" | "target" | "endpoint" | "tracking_uri"
        )
}

fn redact_headers(value: &mut Value) {
    match value {
        Value::Null => {}
        Value::Object(headers) => {
            for (name, value) in headers {
                if !value.is_null() && !value.as_str().is_some_and(|s| is_public_header(name, s)) {
                    *value = Value::String(REDACTED.to_string());
                }
            }
        }
        other => *other = Value::String(REDACTED.to_string()),
    }
}

// Header names alone cannot establish that a value is public. In particular,
// do not expose an opaque key placed in `Accept` or an arbitrary MIME parameter.
// This policy is intentionally only for management reads, not model encryption.
fn is_public_header(name: &str, value: &str) -> bool {
    if value.trim().is_empty() || !value.is_ascii() || value.bytes().any(|b| b.is_ascii_control()) {
        return false;
    }
    match name.to_ascii_lowercase().as_str() {
        "content-type" => public_media_type(value, false),
        "accept" => value.split(',').all(|item| public_media_type(item, true)),
        "accept-encoding" => value.split(',').all(|item| {
            let mut parts = item.split(';');
            let encoding = parts.next().unwrap_or("").trim().to_ascii_lowercase();
            matches!(
                encoding.as_str(),
                "gzip" | "br" | "deflate" | "zstd" | "identity" | "*"
            ) && parts.all(|part| {
                part.trim().split_once('=').is_some_and(|(name, value)| {
                    name.trim().eq_ignore_ascii_case("q") && public_quality(value.trim())
                })
            })
        }),
        _ => false,
    }
}

fn public_media_type(value: &str, accept: bool) -> bool {
    let mut parts = value.split(';');
    let media_type = parts.next().unwrap_or("").trim().to_ascii_lowercase();
    let public = matches!(
        media_type.as_str(),
        "application/json"
            | "application/problem+json"
            | "application/octet-stream"
            | "application/x-www-form-urlencoded"
            | "application/xml"
            | "text/xml"
            | "text/plain"
            | "text/html"
            | "text/event-stream"
    ) || (accept && matches!(media_type.as_str(), "*/*" | "application/*" | "text/*"));
    public
        && parts.all(|part| {
            part.trim().split_once('=').is_some_and(|(name, value)| {
                let value = value.trim();
                if name.trim().eq_ignore_ascii_case("charset") {
                    let charset = value
                        .strip_prefix('"')
                        .and_then(|s| s.strip_suffix('"'))
                        .unwrap_or(value);
                    ["utf-8", "us-ascii", "iso-8859-1"]
                        .iter()
                        .any(|public| charset.eq_ignore_ascii_case(public))
                } else {
                    accept && name.trim().eq_ignore_ascii_case("q") && public_quality(value)
                }
            })
        })
}

fn public_quality(value: &str) -> bool {
    if matches!(value, "0" | "1") {
        return true;
    }
    value.split_once('.').is_some_and(|(whole, fraction)| {
        fraction.len() <= 3
            && fraction.bytes().all(|b| b.is_ascii_digit())
            && (whole == "0" || (whole == "1" && fraction.bytes().all(|b| b == b'0')))
    })
}

/// Returns `true` if the entry identified by `key` holding `value` must be
/// redacted. See [`redact_secrets`] for the full policy.
pub(crate) fn should_redact(key: &str, value: &Value) -> bool {
    if value.is_null() {
        return false;
    }
    let lower = key.to_ascii_lowercase();
    // Credentials are always strings (API keys, passwords, tokens, connection
    // strings, service-account blobs). Restricting the denylist to string
    // values keeps the bare `token` entry from over-matching numeric fields
    // like `input_cost_per_token` / `prompt_caching_min_tokens`.
    if let Value::String(s) = value {
        if REDACTED_KEYS.iter().any(|deny| lower.contains(deny)) {
            return true;
        }
        if s.starts_with(SECRET_VALUE_PREFIX) {
            return true;
        }
    }
    false
}

/// Reject negative `rpm` / `tpm` / `max_budget` values up front. Negative
/// values would otherwise be silently coerced to `None` by `u32::try_from(v)`
/// / `f64` comparisons at enforcement time, making the limit disappear
/// without warning (e.g. a team created with `rpm = -1` bypasses the team
/// RPM ceiling entirely). `max_budget == 0.0` is valid (means "block all
/// spend"). Applied uniformly across keys, teams, users, orgs, and projects.
///
/// `rpm` / `tpm` are stored as `i64` but coerced to `u32` at enforcement
/// time (`u32::try_from(v).ok()` in `VirtualKeyInfo::scope_limits`). Values
/// above `u32::MAX` would collapse to `None` → silently unlimited, so they
/// are rejected here with a 400 rather than letting a master-key operator
/// intending "a very high ceiling" accidentally get "no ceiling".
pub(crate) fn validate_limits(
    rpm: &Option<i64>,
    tpm: &Option<i64>,
    max_budget: Option<f64>,
) -> Result<(), ApiError> {
    if let Some(v) = rpm {
        if *v < 0 {
            return Err(ApiError::bad_request("rpm must be non-negative"));
        }
        if *v > u32::MAX as i64 {
            return Err(ApiError::bad_request("rpm must not exceed u32::MAX"));
        }
    }
    if let Some(v) = tpm {
        if *v < 0 {
            return Err(ApiError::bad_request("tpm must be non-negative"));
        }
        if *v > u32::MAX as i64 {
            return Err(ApiError::bad_request("tpm must not exceed u32::MAX"));
        }
    }
    if let Some(b) = max_budget {
        if !b.is_finite() {
            return Err(ApiError::bad_request("max_budget must be a finite number"));
        }
        if b < 0.0 {
            return Err(ApiError::bad_request("max_budget must be non-negative"));
        }
    }
    Ok(())
}

/// Concurrency ceilings use the same representable range as rate ceilings.
/// Zero deliberately blocks admission; removing the limit requires `None`.
pub(crate) fn validate_parallel_limit(limit: Option<i64>) -> Result<(), ApiError> {
    if limit.is_some_and(|value| !(0..=i64::from(u32::MAX)).contains(&value)) {
        return Err(ApiError::bad_request(
            "max_parallel_requests must be between 0 and u32::MAX",
        ));
    }
    Ok(())
}

/// Reject blank / whitespace-only required name fields (e.g. `team_name`,
/// `user_name`) before they are persisted. Applied on create and update so a
/// malformed name is reported at the API boundary rather than silently stored.
pub(crate) fn validate_non_empty_name(value: &str, field: &'static str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(ApiError::bad_request(format!("{field} must not be empty")));
    }
    Ok(())
}

/// Validate a key `expires_at` string at create time. Accepts any string that
/// parses as a `jiff::Timestamp` and lies in the future; rejects unparseable
/// and already-past timestamps with a 400 so the operator gets immediate
/// feedback instead of a key that silently fails on first use (the auth path
/// treats unparseable expiry as expired — fail closed). `None` (no expiry)
/// is allowed.
pub(crate) fn validate_expires(expires: &Option<String>) -> Result<(), ApiError> {
    let Some(exp) = expires.as_deref() else {
        return Ok(());
    };
    match exp.parse::<jiff::Timestamp>() {
        Ok(ts) => {
            if ts <= jiff::Timestamp::now() {
                return Err(ApiError::bad_request("expires must be a future timestamp"));
            }
            Ok(())
        }
        Err(_) => Err(ApiError::bad_request(
            "expires must be an ISO-8601 timestamp",
        )),
    }
}

/// Validate a `budget_duration` string at the API boundary. Rejects any
/// non-empty string that [`crate::scheduler::parse_budget_duration`] cannot
/// parse, returning a 400 so the operator gets immediate feedback instead of
/// a silent 30d default at enforcement time (billing surprise). `None` and
/// empty strings are valid (= use the default window).
pub(crate) fn validate_budget_duration(duration: &Option<String>) -> Result<(), ApiError> {
    let Some(d) = duration.as_deref() else {
        return Ok(());
    };
    if crate::scheduler::budget_duration_is_valid(d) {
        Ok(())
    } else {
        Err(ApiError::bad_request(
            "budget_duration must be a duration string like '30d', '1mo', '5m', '1h', '30s', or '1w'",
        ))
    }
}

/// Generate a short random id suffix (16 hex chars).
pub(crate) fn random_id() -> String {
    let mut rng = rand::rng();
    let bytes: [u8; 8] = rng.random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Convenience so handlers share the extraction boilerplate.
#[allow(dead_code)]
pub(crate) fn extract_auth(ext: &Extension<AuthUser>) -> Result<&AuthUser, ApiError> {
    Ok(ext)
}

/// Maximum number of rows returned by the `list_users` / `list_teams` /
/// `list_organizations` endpoints before the result is truncated. Mirrors the
/// `KEY_LIST_LIMIT` backstop in `keys::list_keys` against unbounded scans.
pub(crate) const SCOPE_LIST_LIMIT: usize = 1000;

/// Truncate `rows` to [`SCOPE_LIST_LIMIT`] and report whether truncation
/// happened. When the cap kicks in a warning is logged so operators notice.
/// Used by the scope `list_*` handlers to surface a `"truncated": true` flag
/// consistently with `keys::list_keys`.
pub(crate) fn truncate_scope_list<T>(rows: &mut Vec<T>) -> bool {
    let truncated = rows.len() > SCOPE_LIST_LIMIT;
    if truncated {
        tracing::warn!(
            total = rows.len(),
            limit = SCOPE_LIST_LIMIT,
            "scope list returned more than SCOPE_LIST_LIMIT rows; truncating. \
             Add explicit pagination to view all rows."
        );
        rows.truncate(SCOPE_LIST_LIMIT);
    }
    truncated
}

/// Serialize a JSON value into the TEXT column used to store JSON. `None` and
/// explicit JSON `null` both map to `None` (NULL column). Used by the org and
/// tag handlers.
///
/// Serialization of a `serde_json::Value` is effectively infallible in
/// practice (the type only holds serializable leaves), so a failure here is
/// swallowed into `None` rather than propagated as a 500. Logging isn't
/// added because there is no meaningful recovery action; if this ever fires
/// it indicates memory corruption, not user input.
pub(crate) fn serialize_json(v: Option<serde_json::Value>) -> Option<String> {
    match v {
        Some(serde_json::Value::Null) => None,
        Some(val) => serde_json::to_string(&val).ok(),
        None => None,
    }
}

/// Encode a string list as a JSON string for the TEXT columns that store
/// arrays. Trims each entry, drops blanks, and returns `None` for an empty
/// result (no restriction). Used by the customer and tag handlers.
pub(crate) fn serialize_string_list(items: Option<Vec<String>>) -> Option<String> {
    let filtered: Vec<String> = items
        .unwrap_or_default()
        .into_iter()
        .filter_map(|s| {
            let trimmed = s.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        })
        .collect();
    if filtered.is_empty() {
        None
    } else {
        serde_json::to_string(&filtered).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn r26_header_values_are_public_only_with_allowlisted_names_and_values() {
        use crate::config::types::PassThroughEndpoint;

        let endpoint = PassThroughEndpoint {
            path: "/vendor".into(),
            target: "https://vendor.example/v1".into(),
            headers: Some(std::collections::HashMap::from([
                ("X-Opaque-Handle".into(), "opaque-auth-value".into()),
                ("a".into(), "another-opaque-auth-value".into()),
                ("Authorization".into(), "Bearer credential".into()),
                (
                    "Content-Type".into(),
                    "application/json; charset=\"utf-8\"".into(),
                ),
                (
                    "aCcEpT".into(),
                    "application/json, text/event-stream;q=0.8".into(),
                ),
            ])),
        };
        let mut value = serde_json::to_value(endpoint).expect("typed endpoint");
        redact_secrets(&mut value);
        for name in ["X-Opaque-Handle", "a", "Authorization"] {
            assert_eq!(value["headers"][name], REDACTED);
        }
        assert_eq!(
            value["headers"]["Content-Type"],
            "application/json; charset=\"utf-8\""
        );
        assert_eq!(
            value["headers"]["aCcEpT"],
            "application/json, text/event-stream;q=0.8"
        );
        assert_eq!(value["target"], "https://vendor.example/v1");

        // The model-at-rest predicate must retain its existing semantics:
        // ordinary headers are not an all-secret encryption container.
        assert!(!is_secret_container("headers", None));
        assert!(!is_secret_container("headers", Some("oxllm_params")));
        assert!(is_secret_container("extra_headers", None));
        assert!(is_secret_container("headers", Some("extra_body")));
    }

    #[test]
    fn r26_public_header_names_cannot_hide_opaque_auth_values() {
        use crate::config::types::WebhookAlertConfig;

        for (name, value) in [
            ("Content-Type", "opaque-auth-value"),
            ("Content-Type", "application/json; custom=opaque-auth-value"),
            ("Accept", "Bearer opaque-auth-value"),
            ("Accept", "application/json; q=opaque-auth-value"),
            ("Accept", "application/json; q=1.0001"),
            ("Accept-Encoding", "gzip; lease=opaque-auth-value"),
            (
                "Content-Type",
                "application/json\r\nX-Credential: opaque-auth-value",
            ),
        ] {
            let typed = WebhookAlertConfig {
                url: "https://vendor.example/alerts".into(),
                headers: Some(std::collections::HashMap::from([(
                    name.into(),
                    value.into(),
                )])),
                ..Default::default()
            };
            let mut value = serde_json::to_value(typed).expect("typed webhook");
            redact_secrets(&mut value);
            assert_eq!(value["headers"][name], REDACTED);
        }
    }

    #[test]
    fn r20_database_error_logs_redact_the_entire_chain() {
        let error = anyhow::anyhow!(
            "driver rejected postgres://alice:url-secret@db/app?PaSS%77ord=query-secret"
        )
        .context("fallback mysql://bob:backup-secret@backup/app?api%5Fkey=key-secret")
        .context("host=third user=carol password='quoted credential\\' with spaces' dbname=app");
        let (response, log) = crate::security::capture_test_logs(|| db_err(error));
        assert_eq!(response.message, "database error");
        assert!(log.contains("database error"));
        assert!(log.contains("db/app"));
        assert!(log.contains("backup/app"));
        assert!(log.contains("host=third"));
        for secret in [
            "alice",
            "url-secret",
            "query-secret",
            "bob",
            "backup-secret",
            "key-secret",
            "carol",
            "quoted credential",
            "with spaces",
        ] {
            assert!(
                !log.contains(secret),
                "credential leaked in database log: {secret}: {log}"
            );
        }
    }

    #[tokio::test]
    async fn r26_model_encryption_and_public_headers_roundtrip_through_real_handlers() {
        use axum::extract::State;
        use axum::{Extension, Json};
        use models_mgmt::{AddModelRequest, UpdateModelRequest, add_model, update_model};

        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("database");
        db.migrate().await.expect("schema");
        let shared = crate::app::build_shared_state(crate::config::ProxyConfig {
            model_list: vec![],
            general_settings: crate::config::GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            oxllm_settings: Default::default(),
            router_settings: Default::default(),
            extra: Default::default(),
        })
        .expect("state");
        let mut next = (*shared.load_full()).clone();
        next.db = Some(db.clone());
        next.credential_cipher = Some(crate::crypto::CredentialCipher::from_raw_key([7; 32]));
        shared.store(Arc::new(next));
        let typed = oxllm_core::OxllmParams {
            model: "bedrock/anthropic.claude-sonnet-4-20250514".into(),
            aws_access_key_id: Some("AKIA-ROUNDTRIP-SECRET".into()),
            aws_secret_access_key: Some("aws-roundtrip-secret".into()),
            extra_headers: Some(std::collections::HashMap::from([
                ("X-Opaque-Handle".into(), "opaque-roundtrip-secret".into()),
                ("Content-Type".into(), "application/json".into()),
            ])),
            input_cost_per_token: Some(0.000002),
            output_cost_per_token: Some(0.000006),
            extra: serde_json::Map::from_iter([(
                "headers".into(),
                json!({
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }),
            )]),
            ..Default::default()
        };
        let mut expected = serde_json::to_value(typed).expect("typed model params");
        let auth = AuthUser::master([0; 32]);
        let response = add_model(
            State(shared.clone()),
            Extension(auth.clone()),
            Json(AddModelRequest {
                model_name: "r26-roundtrip".into(),
                oxllm_params: expected.clone(),
                model_info: None,
            }),
        )
        .await
        .expect("model create");
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("create response");
        let body: Value = serde_json::from_slice(&bytes).expect("create JSON");
        let model_id = body["model_id"].as_str().expect("model id").to_string();
        let before = db
            .get_model(&model_id)
            .await
            .expect("read stored model")
            .expect("stored model");
        for value in [
            &before.oxllm_params["aws_access_key_id"],
            &before.oxllm_params["aws_secret_access_key"],
            &before.oxllm_params["extra_headers"]["X-Opaque-Handle"],
            &before.oxllm_params["extra_headers"]["Content-Type"],
        ] {
            assert!(crate::crypto::CredentialCipher::looks_encrypted(
                value.as_str().expect("encrypted string")
            ));
        }
        assert_eq!(
            before.oxllm_params["headers"]["Content-Type"],
            "application/json"
        );
        assert_eq!(before.oxllm_params["headers"]["Accept"], "application/json");
        assert_eq!(before.oxllm_params["input_cost_per_token"], 0.000002);

        let response = update_model(
            State(shared.clone()),
            Extension(auth),
            Json(UpdateModelRequest {
                model_id: model_id.clone(),
                model_name: None,
                oxllm_params: Some(json!({"weight": 3})),
                model_info: None,
                clear_model_info: false,
            }),
        )
        .await
        .expect("model update");
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("update response");
        let body: Value = serde_json::from_slice(&bytes).expect("update JSON");
        assert_eq!(body["oxllm_params"]["aws_access_key_id"], REDACTED);
        assert_eq!(
            body["oxllm_params"]["extra_headers"]["X-Opaque-Handle"],
            REDACTED
        );
        assert_eq!(
            body["oxllm_params"]["extra_headers"]["Content-Type"],
            REDACTED
        );
        assert_eq!(
            body["oxllm_params"]["headers"]["Content-Type"],
            "application/json"
        );
        assert_eq!(body["oxllm_params"]["input_cost_per_token"], 0.000002);
        assert_eq!(body["oxllm_params"]["weight"], 3);
        let after = db
            .get_model(&model_id)
            .await
            .expect("read updated model")
            .expect("updated model");
        assert_eq!(
            after.oxllm_params["aws_access_key_id"],
            before.oxllm_params["aws_access_key_id"]
        );
        assert_eq!(
            after.oxllm_params["extra_headers"],
            before.oxllm_params["extra_headers"]
        );
        expected["weight"] = json!(3);
        let state = shared.load_full();
        let loaded = state
            .deployments
            .iter()
            .find(|d| d.model_name == "r26-roundtrip")
            .expect("live model");
        assert_eq!(
            serde_json::to_value(&loaded.oxllm_params).expect("loaded params"),
            expected
        );
    }

    // ---- C06 MEDIUM: secret containers + bare denylist substrings ----------

    #[test]
    fn redact_secrets_redacts_all_values_inside_extra_headers() {
        // Header names like `X-Backend-Key` match no denylist entry; without
        // container awareness the secret would be echoed back in management
        // reads. Every string inside `extra_headers` is now redacted.
        let mut v = json!({
            "oxllm_params": {
                "model": "openai/gpt-4o",
                "extra_headers": {
                    "X-Backend-Key": "hunter2",
                    "X-Benign-Header": "also-redacted"
                }
            }
        });
        redact_secrets(&mut v);
        assert_eq!(
            v["oxllm_params"]["extra_headers"]["X-Backend-Key"],
            REDACTED
        );
        assert_eq!(
            v["oxllm_params"]["extra_headers"]["X-Benign-Header"],
            REDACTED
        );
        assert_eq!(v["oxllm_params"]["model"], "openai/gpt-4o");
    }

    #[test]
    fn redact_secrets_redacts_headers_nested_under_extra_body() {
        let mut v = json!({
            "extra_body": {
                "headers": { "X-Signature": "sig-secret", "X-Other": "v" },
                "conversation_id": "conv-keep"
            }
        });
        redact_secrets(&mut v);
        assert_eq!(v["extra_body"]["headers"]["X-Signature"], REDACTED);
        assert_eq!(v["extra_body"]["headers"]["X-Other"], REDACTED);
        assert_eq!(v["extra_body"]["conversation_id"], "conv-keep");
    }

    #[test]
    fn guardrail_headers_are_redacted_even_with_arbitrary_names() {
        let mut config = serde_json::json!({"guardrails":[{"provider":"custom","config":{
            "headers":{"X-Backend-Key":"private-value","X-Signature":"signed-value"},
            "timeout_secs":10
        }}]});
        redact_secrets(&mut config);
        assert_eq!(
            config["guardrails"][0]["config"]["headers"]["X-Backend-Key"],
            REDACTED
        );
        assert_eq!(
            config["guardrails"][0]["config"]["headers"]["X-Signature"],
            REDACTED
        );
        assert_eq!(config["guardrails"][0]["config"]["timeout_secs"], 10);
    }

    #[test]
    fn redact_secrets_covers_bare_secret_credential_apikey_cert_substrings() {
        let mut v = json!({
            "client_secret": "cs",
            "google_credential": "gc",
            "x-apikey": "ak",
            "client_cert": "cert-data",
            "region": "us-east-1"
        });
        redact_secrets(&mut v);
        assert_eq!(v["client_secret"], REDACTED);
        assert_eq!(v["google_credential"], REDACTED);
        assert_eq!(v["x-apikey"], REDACTED);
        assert_eq!(v["client_cert"], REDACTED);
        assert_eq!(v["region"], "us-east-1");
    }

    // ---- validate_limits (C27: negative rpm/tpm/max_budget) ----------------

    #[test]
    fn validate_limits_accepts_none() {
        assert!(validate_limits(&None, &None, None).is_ok());
    }

    #[test]
    fn validate_limits_accepts_zero() {
        // 0 is valid: rpm=0 means "no requests", max_budget=0 means "no spend".
        assert!(validate_limits(&Some(0), &Some(0), Some(0.0)).is_ok());
    }

    #[test]
    fn validate_limits_accepts_positive() {
        assert!(validate_limits(&Some(100), &Some(10_000), Some(50.0)).is_ok());
    }

    #[test]
    fn validate_limits_rejects_negative_rpm() {
        let err = validate_limits(&Some(-1), &None, None).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("rpm"));
    }

    #[test]
    fn validate_limits_rejects_negative_tpm() {
        let err = validate_limits(&None, &Some(-5), None).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("tpm"));
    }

    #[test]
    fn validate_limits_rejects_negative_max_budget() {
        let err = validate_limits(&None, &None, Some(-0.01)).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("max_budget"));
    }

    // ---- validate_non_empty_name (C27: blank team_name/user_name) ----------

    #[test]
    fn validate_non_empty_name_accepts_real_name() {
        assert!(validate_non_empty_name("Engineering", "team_name").is_ok());
        assert!(validate_non_empty_name(" Alice ", "user_name").is_ok());
    }

    #[test]
    fn validate_non_empty_name_rejects_blank() {
        for bad in ["", "   ", "\t"] {
            let err = validate_non_empty_name(bad, "team_name").expect_err("blank must reject");
            assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
            assert!(err.message.contains("team_name"));
        }
    }

    // ---- validate_expires (C27: malformed / past expires at create time) ----

    #[test]
    fn validate_expires_none_is_ok() {
        assert!(validate_expires(&None).is_ok());
    }

    #[test]
    fn validate_expires_future_is_ok() {
        assert!(validate_expires(&Some("2999-01-01T00:00:00Z".to_string())).is_ok());
    }

    #[test]
    fn validate_expires_unparseable_rejected() {
        let err = validate_expires(&Some("oops".to_string())).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_expires_past_rejected() {
        let err = validate_expires(&Some("2000-01-01T00:00:00Z".to_string()))
            .expect_err("must reject past");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("future"));
    }

    // ---- truncate_scope_list (C27: list_users/list_teams row cap) ----------

    #[test]
    fn truncate_scope_list_keeps_small_lists() {
        let mut rows = vec![1, 2, 3];
        assert!(!truncate_scope_list(&mut rows));
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn truncate_scope_list_caps_and_flags_overflow() {
        let mut rows: Vec<usize> = (0..(SCOPE_LIST_LIMIT + 5)).collect();
        let truncated = truncate_scope_list(&mut rows);
        assert!(truncated);
        assert_eq!(rows.len(), SCOPE_LIST_LIMIT);
    }

    // ---- db_err (C27: must not leak driver text) ---------------------------

    #[test]
    fn db_err_returns_generic_message() {
        // The raw error chain (constraint names, table names, query fragments)
        // must NOT appear in the client-facing message.
        let raw = anyhow::anyhow!("UNIQUE constraint failed: keys.key_id");
        let err = db_err(raw);
        assert_eq!(err.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.message, "database error");
        assert!(!err.message.contains("UNIQUE"));
        assert!(!err.message.contains("keys"));
    }

    // ---- validate_budget_duration (C27: malformed at API boundary) ---------

    #[test]
    fn validate_budget_duration_none_is_ok() {
        assert!(validate_budget_duration(&None).is_ok());
    }

    #[test]
    fn validate_budget_duration_empty_is_ok() {
        assert!(validate_budget_duration(&Some(String::new())).is_ok());
    }

    #[test]
    fn validate_budget_duration_valid_suffixes() {
        for ok in ["30s", "5m", "1h", "30d", "1mo", "2mo", "1w", "500ms"] {
            assert!(
                validate_budget_duration(&Some(ok.to_string())).is_ok(),
                "'{ok}' should be accepted"
            );
        }
    }

    #[test]
    fn validate_budget_duration_rejects_typo() {
        let err = validate_budget_duration(&Some("1x".to_string())).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("budget_duration"));
    }

    #[test]
    fn validate_budget_duration_rejects_garbage() {
        for bad in ["oops", "1.5h", "xxd", "abc"] {
            assert!(
                validate_budget_duration(&Some(bad.to_string())).is_err(),
                "'{bad}' should be rejected"
            );
        }
    }

    // ---- validate_limits: non-finite max_budget (C27: NaN / Infinity) ------

    #[test]
    fn validate_limits_rejects_nan_max_budget() {
        let err = validate_limits(&None, &None, Some(f64::NAN)).expect_err("must reject NaN");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("finite"));
    }

    #[test]
    fn validate_limits_rejects_infinity_max_budget() {
        let err = validate_limits(&None, &None, Some(f64::INFINITY)).expect_err("must reject Inf");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("finite"));
    }

    #[test]
    fn validate_limits_rejects_neg_infinity_max_budget() {
        let err =
            validate_limits(&None, &None, Some(f64::NEG_INFINITY)).expect_err("must reject -Inf");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("finite"));
    }

    // ---- validate_limits: rpm/tpm above u32::MAX (C27 LOW: silent unlimited) -
    //
    // Limits are stored as i64 but coerced to u32 at enforcement time
    // (`u32::try_from(v).ok()`). A value > u32::MAX would collapse to `None`
    // and silently mean "unlimited"; reject it at the API boundary instead.

    #[test]
    fn validate_limits_rejects_rpm_above_u32_max() {
        let err = validate_limits(&Some(u32::MAX as i64 + 1), &None, None)
            .expect_err("must reject rpm > u32::MAX");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("rpm"));
    }

    #[test]
    fn validate_limits_rejects_tpm_above_u32_max() {
        let err = validate_limits(&None, &Some(u32::MAX as i64 + 1), None)
            .expect_err("must reject tpm > u32::MAX");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("tpm"));
    }

    #[test]
    fn validate_limits_accepts_rpm_at_u32_max() {
        // u32::MAX is the highest value that survives the u32 coercion; it
        // means "effectively unlimited but finite" and must be accepted.
        assert!(validate_limits(&Some(u32::MAX as i64), &Some(u32::MAX as i64), None).is_ok());
    }

    // ---- is_unique_violation / map_create_err (C28 LOW: TOCTOU 500 → 400) --

    #[test]
    fn is_unique_violation_recognizes_structured_kind() {
        // Drivers set `kind()` to `UniqueViolation`; that path is preferred
        // over the code/message fallback.
        let db_err = sqlx::Error::Database(Box::new(StubDbError {
            code: None,
            message: "duplicate key".to_string(),
            unique: true,
        }));
        let err: anyhow::Error = db_err.into();
        assert!(is_unique_violation(&err));
    }

    #[test]
    fn is_unique_violation_recognizes_sqlite_constraint_text() {
        // SQLite generic SQLITE_CONSTRAINT shape: driver may surface the
        // human-readable text without an extended code.
        let db_err = sqlx::Error::Database(Box::new(StubDbError {
            code: Some("1".to_string()),
            message: "UNIQUE constraint failed: organizations.org_id".to_string(),
            unique: false,
        }));
        let err: anyhow::Error = db_err.into();
        assert!(is_unique_violation(&err));
    }

    #[test]
    fn is_unique_violation_recognizes_sqlite_extended_codes() {
        for code in ["1555", "2067"] {
            let db_err = sqlx::Error::Database(Box::new(StubDbError {
                code: Some(code.to_string()),
                message: "constraint failed".to_string(),
                unique: false,
            }));
            let err: anyhow::Error = db_err.into();
            assert!(is_unique_violation(&err), "code {code} should match");
        }
    }

    #[test]
    fn is_unique_violation_recognizes_postgres_sqlstate_23505() {
        let db_err = sqlx::Error::Database(Box::new(StubDbError {
            code: Some("23505".to_string()),
            message: "duplicate key value violates unique constraint".to_string(),
            unique: false,
        }));
        let err: anyhow::Error = db_err.into();
        assert!(is_unique_violation(&err));
    }

    #[test]
    fn is_unique_violation_ignores_other_database_errors() {
        let db_err = sqlx::Error::Database(Box::new(StubDbError {
            code: Some("23000".to_string()),
            message: "some other integrity violation".to_string(),
            unique: false,
        }));
        let err: anyhow::Error = db_err.into();
        assert!(!is_unique_violation(&err));
    }

    #[test]
    fn is_unique_violation_ignores_non_database_errors() {
        let err: anyhow::Error = sqlx::Error::PoolClosed.into();
        assert!(!is_unique_violation(&err));
    }

    #[test]
    fn map_create_err_translates_unique_violation_to_400() {
        let db_err = sqlx::Error::Database(Box::new(StubDbError {
            code: None,
            message: "duplicate key".to_string(),
            unique: true,
        }));
        let err: anyhow::Error = db_err.into();
        let api = map_create_err(err, || ApiError::bad_request("org 'org-x' already exists"));
        assert_eq!(api.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(api.message, "org 'org-x' already exists");
    }

    #[test]
    fn map_create_err_passes_through_other_errors_as_500() {
        let err: anyhow::Error = sqlx::Error::PoolClosed.into();
        let api = map_create_err(err, || ApiError::bad_request("should not be used"));
        assert_eq!(api.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(api.message, "database error");
    }

    /// Minimal `DatabaseError` stand-in so unit tests can build a
    /// `sqlx::Error::Database` without a live driver. Only `code`, `kind`,
    /// and `message` are exercised by [`is_unique_violation`].
    #[derive(Debug)]
    struct StubDbError {
        code: Option<String>,
        message: String,
        unique: bool,
    }

    impl std::fmt::Display for StubDbError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.message)
        }
    }

    impl std::error::Error for StubDbError {}

    impl sqlx::error::DatabaseError for StubDbError {
        fn message(&self) -> &str {
            &self.message
        }
        fn code(&self) -> Option<std::borrow::Cow<'_, str>> {
            self.code.as_deref().map(Into::into)
        }
        fn constraint(&self) -> Option<&str> {
            None
        }
        fn as_error(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
            self
        }
        fn as_error_mut(&mut self) -> &mut (dyn std::error::Error + Send + Sync + 'static) {
            self
        }
        fn into_error(self: Box<Self>) -> Box<dyn std::error::Error + Send + Sync + 'static> {
            self
        }
        fn kind(&self) -> sqlx::error::ErrorKind {
            if self.unique {
                sqlx::error::ErrorKind::UniqueViolation
            } else {
                sqlx::error::ErrorKind::Other
            }
        }
    }
}
