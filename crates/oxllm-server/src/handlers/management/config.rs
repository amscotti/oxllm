//! Config introspection endpoints: `GET /config`, `GET /routes`, and
//! `POST /config/update`.

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde_json::{Value, json};

use crate::auth::AuthUser;
use crate::config;
use crate::error::ApiError;
use crate::state::SharedState;

use super::{redact_secrets, require_master};

/// `GET /config` — return the current proxy config with every secret value
/// redacted (see [`super::redact_secrets`] and the `sk-` value heuristic).
pub async fn get_config(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let state = shared.load_full();
    let mut value = serde_json::to_value(&*state.config)
        .map_err(|e| ApiError::internal(format!("failed to serialize config: {e}")))?;
    redact_secrets(&mut value);
    Ok(Json(value).into_response())
}

/// `GET /routes` — return the static list of routes the proxy registers.
pub async fn get_routes(
    State(_shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    Ok(Json(json!({ "routes": known_routes() })).into_response())
}

/// `POST /config/update` — write a new (or partial) config body to the YAML
/// file the proxy loaded from and let the hot-reload watcher apply it.
///
/// The request body is a JSON document matching the LiteLLM config shape. It
/// is re-emitted as YAML and written verbatim to [`config::current_config_path`],
/// overwriting the previous file. The reload watcher then rebuilds the
/// [`crate::state::AppState`] from the new file — this handler deliberately
/// does not duplicate the reload logic.
///
/// Requires the master key (enforced by the management route group).
///
/// **Concurrency:** two concurrent calls each read-validate-write-rename
/// independently with no cross-request lock. The atomic temp+rename guarantees
/// no corruption (the on-disk file is always one complete body or the other,
/// never a hybrid), but the second rename silently clobbers the first
/// writer's entire config — last-writer-wins. An operator who needs
/// transactional consistency should serialize config updates externally
/// (e.g. via an external lock or a single orchestrator).
///
/// Returns the config as it stood just before the write (redacted). Callers
/// that need the reloaded view should re-`GET /config` after the watcher has
/// fired (typically well under a second).
///
/// Returns 500 when the config was not loaded from a file (e.g. built
/// programmatically in tests) and there is therefore no path to write back
/// to.
///
/// U17 (LiteLLM commit `e9ac84dc8b`): LiteLLM's `save_config` serialized the
/// in-memory config (with environment-variable-expanded secrets baked in) into
/// the DB, leaking secrets the operator never intended to persist. oxllm is
/// safe: this handler writes only the **caller-provided** `value` to disk — it
/// never snapshots the resolved in-memory `state.config` or process env vars.
/// The in-memory config is serialized only for the **response**, and that
/// serialization always passes through [`redact_secrets`]. Env-var expansion
/// happens at *load* time (the reload watcher re-resolves from the file), not
/// at write time, so no expanded secret is ever persisted.
pub async fn update_config(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(value): Json<Value>,
) -> Result<Response, ApiError> {
    let require_auth = matches!(
        std::env::var("OXLLM_REQUIRE_AUTH").as_deref(),
        Ok("true") | Ok("1")
    );
    update_config_with_auth_requirement(shared, auth, value, require_auth).await
}

// Pass the process policy into the implementation so tests can exercise the
// complete write/rejection path without mutating other tests' environment.
async fn update_config_with_auth_requirement(
    shared: SharedState,
    auth: AuthUser,
    value: Value,
    require_auth: bool,
) -> Result<Response, ApiError> {
    require_master(&auth)?;

    let path = config::loader::current_config_path().ok_or_else(|| {
        ApiError::internal(
            "config file path is unknown; cannot update a config that was \
             not loaded from a file",
        )
    })?;

    let yaml = serde_yml::to_string(&value)
        .map_err(|e| ApiError::internal(format!("failed to serialize config body to YAML: {e}")))?;

    // Validate BEFORE overwriting the file so a bad body never bricks the
    // next process start. `parse_config_str` is not used here: it applies
    // the `environment_variables` block to the process env. We still run
    // `os.environ/` expansion + the same post-deserialize checks (SSRF,
    // thresholds) and, when `OXLLM_REQUIRE_AUTH` is set, refuse a body
    // that would drop every auth path.
    let mut to_validate = value.clone();
    if let Err(e) = config::loader::resolve_env_vars(&mut to_validate) {
        tracing::warn!(error = %e, "POST /config/update rejected: env resolution failed");
        return Err(ApiError::bad_request(format!(
            "config body failed env resolution: {e}"
        )));
    }
    let parsed: config::ProxyConfig = match serde_json::from_value(to_validate) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "POST /config/update rejected: body does not deserialize as ProxyConfig"
            );
            return Err(ApiError::bad_request(format!(
                "config body is not a valid ProxyConfig: {e}"
            )));
        }
    };
    if let Err(e) = config::loader::validate_proxy_config(&parsed) {
        tracing::warn!(error = %e, "POST /config/update rejected: config validation failed");
        return Err(ApiError::bad_request(format!(
            "config body failed validation: {e}"
        )));
    }
    if require_auth
        && !crate::app::auth_is_configured(
            &parsed.general_settings.master_key,
            &parsed.general_settings.jwt_secret,
            &parsed.general_settings.database_url,
        )
    {
        return Err(ApiError::bad_request(
            "OXLLM_REQUIRE_AUTH is set but the new config has no master_key, \
             jwt_secret, or database_url",
        ));
    }

    // Atomic write: serialize to a sibling temp file, then rename into place.
    // A plain `std::fs::write` truncates then streams bytes; a crash, OOM, or
    // disk-full between truncate and completion would leave a partial YAML
    // file on disk that no subsequent process start can parse — bricking the
    // proxy until an operator manually restores the file. The temp-then-rename
    // dance is atomic on POSIX (same filesystem), so the on-disk config is
    // either the previous version or the complete new version, never a hybrid.
    atomic_write_config(&path, yaml.as_bytes()).map_err(|e| {
        ApiError::internal(format!(
            "failed to write config file {}: {e}",
            path.display()
        ))
    })?;

    tracing::info!(
        path = %path.display(),
        "config overwritten via POST /config/update; hot-reload watcher will apply it",
    );

    // Return the redacted config so the caller sees what is live. Note: the
    // reload is asynchronous, so this reflects the PRE-write snapshot until
    // the watcher swaps in the new state. The file on disk is already
    // updated, which is the durable change.
    let state = shared.load_full();
    let mut current = serde_json::to_value(&*state.config)
        .map_err(|e| ApiError::internal(format!("failed to serialize config: {e}")))?;
    redact_secrets(&mut current);
    Ok(Json(current).into_response())
}

/// Atomically replace the config file at `path` with `bytes`.
///
/// Writes to a sibling temp file (`<path>.<pid>.<nanos>.tmp`), then renames it
/// over the destination. On POSIX the rename is atomic, so the file on disk is
/// always either the previous or the complete new version — never a partial
/// write. The temp file is cleaned up on write failure (the rename itself
/// failing after a successful write leaves the temp behind and is removed in a
/// best-effort way; the error is still surfaced). Mirrors the pattern in
/// `cache/disk.rs::write_entry`.
///
/// On Unix the temp file's permissions are tightened to `0o600` (or, when the
/// destination already exists, copied from the destination's current mode)
/// before the rename. `std::fs::write` would otherwise create the temp with the
/// process umask (typically `0o644`), and the rename replaces the inode — so
/// an operator who `chmod 600`'d the config file would silently lose that
/// restriction on the first update, exposing the `master_key` / `api_key` /
/// `database_url` / `redis_url` the caller writes verbatim (U17) to every
/// local user.
fn atomic_write_config(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = tmp_config_path(path);
    if let Err(e) = std::fs::write(&tmp, bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // Tighten the temp file's permissions before the rename so the destination
    // does not relax to the umask default (typically world/other-readable).
    apply_restricted_permissions(&tmp, path);
    // fsync the temp file before rename so the rename is durable, not just
    // atomic. We open by the path we just wrote; a missing file maps to the
    // write error above, which already returned.
    if let Ok(file) = std::fs::File::open(&tmp) {
        let _ = file.sync_all();
        drop(file);
    }
    let rename_result = std::fs::rename(&tmp, path);
    if rename_result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    rename_result
}

/// Set the mode of the temp file at `tmp` to match the destination's current
/// mode when the destination exists, or `0o600` for a first-time create. On
/// non-Unix targets this is a no-op. Failures are swallowed: a permission
/// tightening failure must not abort an otherwise-valid atomic write, and the
/// destination's previous mode is preserved by the rename only when the temp
/// was tightened successfully — but the alternatives (skip the write, or
/// surface the chmod error to the operator) are both worse for availability.
#[cfg(unix)]
fn apply_restricted_permissions(tmp: &std::path::Path, dest: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(dest)
        .ok()
        .and_then(|m| {
            let mode = m.permissions().mode();
            (mode != 0).then_some(mode)
        })
        .unwrap_or(0o600);
    let perms = std::fs::Permissions::from_mode(mode);
    let _ = std::fs::set_permissions(tmp, perms);
}

#[cfg(not(unix))]
fn apply_restricted_permissions(_tmp: &std::path::Path, _dest: &std::path::Path) {}

/// Build a unique temp-file path next to the config file. Including the PID
/// and current time guards against collisions between concurrent writers and
/// across processes.
fn tmp_config_path(path: &std::path::Path) -> std::path::PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".{pid}.{nanos}.tmp"));
    std::path::PathBuf::from(s)
}

/// The fixed set of LLM / observability routes the proxy serves. This excludes
/// the management group (mounted only with the `db` feature), the optional
/// platform group, and the dynamic pass-through catch-all.
fn known_routes() -> Vec<&'static str> {
    vec![
        "/health",
        "/health/readiness",
        "/metrics",
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/v1/embeddings",
        "/embeddings",
        "/v1/messages",
        "/v1/responses",
        "/v1/rerank",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/images/variations",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/audio/speech",
        "/v1/moderations",
        "/v1/models",
        "/v1/realtime",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::management::REDACTED;

    /// Exercise the real serialized field layout, including the slug-keyed
    /// callback map and singleton alerting webhook (neither is a `webhooks` array).
    fn credential_config() -> config::ProxyConfig {
        use std::collections::HashMap;

        use crate::config::types::{
            AlertingConfig, ConfigModel, GeneralSettings, GuardrailsConfig, McpGuardrailConfig,
            OxllmSettings, PagerDutyConfig, PassThroughEndpoint, S3LogConfig, SlackAlertConfig,
            WebhookAlertConfig,
        };

        config::ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "r26-model".into(),
                oxllm_params: oxllm_core::OxllmParams {
                    model: "bedrock/anthropic.claude-sonnet-4-20250514".into(),
                    api_key: Some("provider-api-secret".into()),
                    aws_access_key_id: Some("AKIA-MODEL-SECRET".into()),
                    aws_secret_access_key: Some("provider-aws-secret".into()),
                    extra_headers: Some(HashMap::from([
                        ("X-Opaque-Provider".into(), "provider-header-secret".into()),
                        ("Content-Type".into(), "application/json".into()),
                    ])),
                    input_cost_per_token: Some(0.000002),
                    output_cost_per_token: Some(0.000006),
                    cache_read_input_token_cost: Some(0.000001),
                    cost_per_second: Some(0.5),
                    prompt_caching_min_tokens: Some(1024),
                    rpm: Some(30),
                    tpm: Some(4000),
                    max_budget: Some(20.0),
                    ..Default::default()
                },
                ..Default::default()
            }],
            general_settings: GeneralSettings {
                master_key: Some("sk-config-redaction-master-0123456789abcdef0123456789abcdef".into()),
                allow_private_api_base: Some(true),
                max_request_body_size_mb: Some(50),
                pass_through_endpoints: Some(vec![PassThroughEndpoint {
                    path: "/vendor".into(),
                    target: "https://203.0.113.20/vendor?version=2".into(),
                    headers: Some(HashMap::from([
                        ("X-Internal-Lease".into(), "pass-opaque-secret".into()),
                        ("cOnTeNt-TyPe".into(), "application/json; charset=utf-8".into()),
                    ])),
                }]),
                guardrails: Some(GuardrailsConfig {
                    mcp_guardrails: Some(vec![McpGuardrailConfig {
                        name: "mcp-scan".into(),
                        server_url: "https://mcp-user:mcp-password@203.0.113.10:8443/rpc?tenant=public&API%5FKEY=mcp-query-secret".into(),
                        tool_name: Some("scan_prompt".into()),
                        timeout_secs: Some(3),
                        fail_open: Some(false),
                    }]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            oxllm_settings: OxllmSettings {
                callback_settings: Some(HashMap::from([
                    ("generic".into(), json!({
                        "url": "https://203.0.113.30/events?version=1",
                        "headers": {
                            "X-Hook-Handle": "callback-opaque-secret",
                            "Accept": "application/json, text/event-stream;q=0.8"
                        }
                    })),
                    ("custom-collector".into(), json!({
                        "url": "https://203.0.113.31/events",
                        "headers": { "X-Collector-Lease": "custom-callback-secret" }
                    })),
                ])),
                alerting_config: Some(AlertingConfig {
                    webhook: Some(WebhookAlertConfig {
                        url: "https://203.0.113.40/alerts".into(),
                        headers: Some(HashMap::from([
                            ("X-Event-Capability".into(), "webhook-opaque-secret".into()),
                            ("Accept-Encoding".into(), "gzip, br".into()),
                        ])),
                        thresholds: vec![0.8, 1.0],
                        soft_budget_pct: Some(0.7),
                        ..Default::default()
                    }),
                }),
                slack_alerting_config: Some(SlackAlertConfig {
                    webhook_url: "https://hooks.slack.com/services/TEAM/CHANNEL/slack-live-capability".into(),
                    channel: Some("#ops".into()),
                    thresholds: vec![0.8, 1.0],
                    ..Default::default()
                }),
                pagerduty_config: Some(PagerDutyConfig {
                    routing_key: "pagerduty-routing-secret".into(),
                    thresholds: vec![0.8, 1.0],
                    ..Default::default()
                }),
                s3_log_config: Some(S3LogConfig {
                    bucket: "public-bucket-name".into(),
                    region: "us-east-1".into(),
                    access_key_id: Some("AKIA-LOG-SECRET".into()),
                    secret_access_key: Some("s3-secret".into()),
                    ..Default::default()
                }),
                max_budget: Some(100.0),
                rpm: Some(300),
                tpm: Some(10000),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: Default::default(),
        }
    }

    fn assert_credential_config_redacted(body: &Value) {
        let general = &body["general_settings"];
        let settings = &body["oxllm_settings"];
        let params = &body["model_list"][0]["oxllm_params"];
        assert_eq!(
            general["pass_through_endpoints"][0]["headers"]["X-Internal-Lease"],
            REDACTED
        );
        assert_eq!(
            general["pass_through_endpoints"][0]["headers"]["cOnTeNt-TyPe"],
            "application/json; charset=utf-8"
        );
        assert_eq!(
            settings["callback_settings"]["generic"]["headers"]["X-Hook-Handle"],
            REDACTED
        );
        assert_eq!(
            settings["callback_settings"]["custom-collector"]["headers"]["X-Collector-Lease"],
            REDACTED
        );
        assert_eq!(
            settings["callback_settings"]["generic"]["headers"]["Accept"],
            "application/json, text/event-stream;q=0.8"
        );
        assert_eq!(
            settings["alerting_config"]["webhook"]["headers"]["X-Event-Capability"],
            REDACTED
        );
        assert_eq!(
            settings["alerting_config"]["webhook"]["headers"]["Accept-Encoding"],
            "gzip, br"
        );
        assert_eq!(settings["slack_alerting_config"]["webhook_url"], REDACTED);
        assert_eq!(settings["slack_alerting_config"]["channel"], "#ops");
        assert_eq!(settings["pagerduty_config"]["routing_key"], REDACTED);
        assert_eq!(settings["s3_log_config"]["access_key_id"], REDACTED);
        assert_eq!(params["aws_access_key_id"], REDACTED);
        assert_eq!(params["extra_headers"]["X-Opaque-Provider"], REDACTED);
        assert_eq!(params["extra_headers"]["Content-Type"], REDACTED);
        assert_eq!(
            general["guardrails"]["mcp_guardrails"][0]["server_url"],
            "https://203.0.113.10:8443/rpc?tenant=public"
        );
        assert_eq!(
            general["guardrails"]["mcp_guardrails"][0]["timeout_secs"],
            3
        );
        assert_eq!(
            general["guardrails"]["mcp_guardrails"][0]["fail_open"],
            false
        );
        assert_eq!(general["max_request_body_size_mb"], 50);
        assert_eq!(
            settings["alerting_config"]["webhook"]["thresholds"],
            json!([0.8, 1.0])
        );
        assert_eq!(
            settings["alerting_config"]["webhook"]["soft_budget_pct"],
            0.7
        );
        assert_eq!(settings["s3_log_config"]["region"], "us-east-1");
        assert_eq!(settings["max_budget"], 100.0);
        assert_eq!(settings["rpm"], 300);
        assert_eq!(settings["tpm"], 10000);
        for (key, expected) in [
            ("input_cost_per_token", json!(0.000002)),
            ("output_cost_per_token", json!(0.000006)),
            ("cache_read_input_token_cost", json!(0.000001)),
            ("cost_per_second", json!(0.5)),
            ("prompt_caching_min_tokens", json!(1024)),
            ("rpm", json!(30)),
            ("tpm", json!(4000)),
            ("max_budget", json!(20.0)),
        ] {
            assert_eq!(params[key], expected);
            assert!(params[key].is_number());
        }
        let serialized = body.to_string();
        for secret in [
            "pass-opaque-secret",
            "callback-opaque-secret",
            "custom-callback-secret",
            "webhook-opaque-secret",
            "slack-live-capability",
            "pagerduty-routing-secret",
            "AKIA-LOG-SECRET",
            "AKIA-MODEL-SECRET",
            "mcp-user",
            "mcp-password",
            "mcp-query-secret",
            "provider-header-secret",
            "provider-api-secret",
            "provider-aws-secret",
        ] {
            assert!(
                !serialized.contains(secret),
                "secret leaked in config response: {secret}"
            );
        }
    }

    fn install_config_snapshot(shared: &SharedState, config: config::ProxyConfig) {
        // These tests exercise config serialization, so no outbound telemetry
        // or guardrail resources need to be constructed for the loaded snapshot.
        let mut next = (*shared.load_full()).clone();
        next.config = std::sync::Arc::new(config);
        shared.store(std::sync::Arc::new(next));
    }

    #[test]
    fn r26_redacts_real_typed_config_without_corrupting_pricing_or_limits() {
        let typed = credential_config();
        let original = serde_json::to_value(&typed).expect("serialize typed config");
        let mut redacted = original.clone();
        redact_secrets(&mut redacted);
        assert_credential_config_redacted(&redacted);
        assert_eq!(
            serde_json::to_value(&typed).expect("serialize again"),
            original
        );
    }

    #[tokio::test]
    async fn r26_get_config_redacts_real_loaded_config() {
        let shared = crate::app::build_shared_state(config::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: Default::default(),
            router_settings: Default::default(),
            extra: Default::default(),
        })
        .expect("state");
        install_config_snapshot(&shared, credential_config());
        let response = get_config(
            State(shared),
            Extension(crate::auth::AuthUser::master([0; 32])),
        )
        .await
        .expect("GET /config");
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("response");
        let body: Value = serde_json::from_slice(&bytes).expect("JSON config");
        assert_credential_config_redacted(&body);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn r26_config_update_redacts_live_snapshot_and_persists_original_credentials() {
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK
            .lock()
            .expect("config path lock");
        let (shared, path, auth) = state_from_temp_file();
        install_config_snapshot(&shared, credential_config());
        let caller = serde_json::to_value(credential_config()).expect("typed caller config");
        let response = update_config(State(shared), Extension(auth), Json(caller.clone()))
            .await
            .expect("POST /config/update");
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("response");
        let body: Value = serde_json::from_slice(&bytes).expect("JSON config");
        assert_credential_config_redacted(&body);
        let yaml = std::fs::read_to_string(&path).expect("written config");
        let written: config::ProxyConfig =
            serde_yml::from_str(&yaml).expect("typed YAML roundtrip");
        assert_eq!(
            serde_json::to_value(written).expect("serialized YAML"),
            caller
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn r26_mcp_validation_error_chains_hide_url_credentials() {
        let mut typed = credential_config();
        let mcp = &mut typed
            .general_settings
            .guardrails
            .as_mut()
            .expect("guardrails")
            .mcp_guardrails
            .as_mut()
            .expect("MCP config")[0];
        mcp.server_url =
            "https://mcp-user:mcp-password@[broken/rpc?API%5FKEY=mcp-query-secret".into();
        let error = config::loader::validate_proxy_config(&typed).expect_err("invalid MCP URL");
        for rendered in [format!("{error:#}"), format!("{error:?}")] {
            for secret in ["mcp-user", "mcp-password", "mcp-query-secret"] {
                assert!(
                    !rendered.contains(secret),
                    "validation leaked {secret}: {rendered}"
                );
            }
            assert!(rendered.contains("mcp-scan"));
        }
    }

    #[test]
    fn redact_secrets_replaces_api_key_and_master_key() {
        let mut v = json!({
            "model_list": [
                {
                    "oxllm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "sk-secret"
                    }
                }
            ],
            "general_settings": {
                "master_key": "sk-master-test-key"
            }
        });
        redact_secrets(&mut v);
        assert_eq!(v["model_list"][0]["oxllm_params"]["api_key"], REDACTED);
        assert_eq!(v["general_settings"]["master_key"], REDACTED);
        // Non-secret fields are untouched.
        assert_eq!(v["model_list"][0]["oxllm_params"]["model"], "openai/gpt-4o");
    }

    #[test]
    fn redact_secrets_leaves_null_values_alone() {
        let mut v = json!({ "api_key": null });
        redact_secrets(&mut v);
        assert!(v["api_key"].is_null());
    }

    #[test]
    fn redact_secrets_descends_into_arrays_and_nested_objects() {
        let mut v = json!({
            "arr": [{ "api_key": "deep" }],
            "nested": { "api_key": "nested" }
        });
        redact_secrets(&mut v);
        assert_eq!(v["arr"][0]["api_key"], REDACTED);
        assert_eq!(v["nested"]["api_key"], REDACTED);
    }

    #[test]
    fn redact_secrets_redacts_langfuse_secret_key() {
        let mut v = json!({
            "langfuse_config": { "secret_key": "lf-secret", "public_key": "pk-public" }
        });
        redact_secrets(&mut v);
        assert_eq!(v["langfuse_config"]["secret_key"], REDACTED);
        // The non-secret public key is preserved.
        assert_eq!(v["langfuse_config"]["public_key"], "pk-public");
    }

    #[test]
    fn redact_secrets_redacts_aws_bedrock_credentials() {
        let mut v = json!({
            "extra": {
                "aws_secret_access_key": "AWSSECRET123",
                "aws_session_token": "AQoEXAMPLEH4aoAH0eNSPE",
                "aws_region": "us-east-1"
            }
        });
        redact_secrets(&mut v);
        assert_eq!(v["extra"]["aws_secret_access_key"], REDACTED);
        assert_eq!(v["extra"]["aws_session_token"], REDACTED);
        // The region is config, not a secret.
        assert_eq!(v["extra"]["aws_region"], "us-east-1");
    }

    #[test]
    fn redact_secrets_redacts_webhook_authorization_header_case_insensitively() {
        let mut v = json!({
            "webhooks": [
                {
                    "headers": {
                        "Authorization": "Bearer abc123",
                        "Content-Type": "application/json"
                    }
                }
            ]
        });
        redact_secrets(&mut v);
        assert_eq!(v["webhooks"][0]["headers"]["Authorization"], REDACTED);
        // A benign header survives.
        assert_eq!(
            v["webhooks"][0]["headers"]["Content-Type"],
            "application/json"
        );
    }

    #[test]
    fn redact_secrets_redacts_by_sk_value_prefix() {
        // A key name that is not on the denylist but whose value looks like a
        // leaked OpenAI-style key should still be redacted.
        let mut v = json!({ "custom_field": "sk-leaked-key-12345" });
        redact_secrets(&mut v);
        assert_eq!(v["custom_field"], REDACTED);
    }

    #[test]
    fn redact_secrets_preserves_non_secret_values() {
        let mut v = json!({
            "model_list": [{ "oxllm_params": { "model": "openai/gpt-4o" } }],
            "base_url": "https://api.openai.com/v1",
            "enabled": true,
            "timeout_ms": 30_000,
            "count": 42,
            "nullable": null
        });
        redact_secrets(&mut v);
        assert_eq!(v["model_list"][0]["oxllm_params"]["model"], "openai/gpt-4o");
        assert_eq!(v["base_url"], "https://api.openai.com/v1");
        assert_eq!(v["enabled"], true);
        assert_eq!(v["timeout_ms"], 30_000);
        assert_eq!(v["count"], 42);
        assert!(v["nullable"].is_null());
    }

    #[test]
    fn known_routes_includes_core_endpoints() {
        let routes = known_routes();
        assert!(routes.contains(&"/v1/chat/completions"));
        assert!(routes.contains(&"/health"));
    }

    // ---- M12: POST /config/update ----------------------------------------

    /// Build a `SharedState` whose config was loaded from a real temp file,
    /// so `current_config_path()` points somewhere writable. Returns the
    /// shared state and the temp file path.
    fn state_from_temp_file() -> (
        crate::state::SharedState,
        std::path::PathBuf,
        crate::auth::AuthUser,
    ) {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxllm_test_config_update_{}.yaml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, "model_list: []\n").expect("write temp config");

        // load_config records the path that /config/update writes back to.
        let cfg = crate::config::load_config(&path.to_string_lossy()).expect("load temp config");
        let shared = crate::app::build_shared_state(cfg).expect("build state");

        let auth = crate::auth::AuthUser::master([0u8; 32]);
        (shared, path, auth)
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn update_config_writes_new_yaml_to_config_path() {
        // `current_config_path()` is a process-global, so serialize against
        // the loader's config-path tests (and any other test that calls
        // load_config) for the whole critical section. The sync guard is
        // held across the handler's `.await` on purpose — this is a test
        // serializing a process-global, not production async code.
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let (shared, path, auth) = state_from_temp_file();

        let new_body = json!({
            "model_list": [
                {
                    "model_name": "gpt-4o-mini",
                    "oxllm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-updated"
                    }
                }
            ]
        });

        let state = axum::extract::State(shared.clone());
        let ext = axum::Extension(auth);
        let response = update_config(state, ext, Json(new_body))
            .await
            .expect("handler succeeds");

        // The file on disk must now contain the new YAML (the reload trigger).
        let on_disk = std::fs::read_to_string(&path).expect("read back temp config");
        assert!(
            on_disk.contains("gpt-4o-mini"),
            "updated file must contain the new model, got:\n{on_disk}"
        );
        assert!(
            on_disk.contains("model_name"),
            "updated file must be serialized YAML, got:\n{on_disk}"
        );

        // The handler returns a redacted config response.
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("read body");
        let body: Value = serde_json::from_slice(&bytes).expect("json body");
        // The written-back api_key must be redacted in the response.
        let serialized = body.to_string();
        assert!(
            !serialized.contains("sk-updated"),
            "response must not leak the secret api_key, got: {serialized}"
        );

        // Atomic write must leave no temp file behind on success.
        assert!(
            !any_tmp_sibling(&path),
            "atomic write should not leave a .tmp file behind"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn update_config_rejects_non_master() {
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let (shared, path, _master_auth) = state_from_temp_file();
        let non_master = crate::auth::AuthUser::default();

        let state = axum::extract::State(shared.clone());
        let ext = axum::Extension(non_master);
        let result = update_config(state, ext, Json(json!({"model_list": []}))).await;
        assert!(
            result.is_err(),
            "non-master caller must be rejected (master-key route group)"
        );
        let err = result.unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);

        let _ = std::fs::remove_file(&path);
    }

    // ---- U17: config update must never snapshot env vars / resolved config --
    //
    // LiteLLM commit e9ac84dc8b: `save_config` serialized the in-memory config
    // (with env-var-expanded secrets) into the DB. oxllm's `update_config`
    // writes only the caller-provided body. This test verifies the written
    // file contains exactly the caller's JSON, not a snapshot of the resolved
    // in-memory config (which would include redacted `***` markers or
    // expanded env vars).

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn u17_update_config_writes_caller_body_not_resolved_config() {
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let (shared, path, auth) = state_from_temp_file();

        let caller_body = json!({
            "model_list": [
                {
                    "model_name": "gpt-4o",
                    "oxllm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "sk-caller-provided"
                    }
                }
            ]
        });

        let state = axum::extract::State(shared.clone());
        let ext = axum::Extension(auth);
        let response = update_config(state, ext, Json(caller_body))
            .await
            .expect("handler succeeds");

        // The file on disk must contain the caller's raw body verbatim —
        // including the secret they explicitly provided. It must NOT contain
        // the redacted "***" marker (which would mean the resolved in-memory
        // config was serialized instead of the caller's body).
        let on_disk = std::fs::read_to_string(&path).expect("read back temp config");
        assert!(
            on_disk.contains("sk-caller-provided"),
            "U17: written file must contain the caller's exact body"
        );
        assert!(
            !on_disk.contains("***"),
            "U17: written file must not be a redacted snapshot of the in-memory config"
        );

        // The *response*, however, must be redacted (it serializes the
        // in-memory config).
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("read body");
        let body: Value = serde_json::from_slice(&bytes).expect("json body");
        let serialized = body.to_string();
        assert!(
            !serialized.contains("sk-caller-provided"),
            "U17: response must redact secrets"
        );

        let _ = std::fs::remove_file(&path);
    }

    // ---- C30 MEDIUM: POST /config/update must reject malformed bodies --------
    //
    // A body that does not round-trip as a ProxyConfig must be rejected with
    // 400 BEFORE the file is overwritten, so the on-disk config is never
    // bricked for the next process restart.

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn update_config_rejects_body_that_is_not_a_proxyconfig() {
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let (shared, path, auth) = state_from_temp_file();

        // Seed the file with a known-good body so we can assert it is preserved.
        let seed = "model_list: []\n";
        std::fs::write(&path, seed).expect("seed good config");

        // `model_list` must be a sequence; a string is structurally invalid.
        let bad_body = json!({ "model_list": "not-a-list" });

        let state = axum::extract::State(shared.clone());
        let ext = axum::Extension(auth);
        let result = update_config(state, ext, Json(bad_body)).await;
        let err = result.expect_err("malformed body must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("ProxyConfig"),
            "error should name the validation failure: {}",
            err.message
        );

        // The on-disk file must be unchanged (the pre-existing good config).
        let on_disk = std::fs::read_to_string(&path).expect("read back temp config");
        assert_eq!(
            on_disk, seed,
            "malformed body must not overwrite the on-disk config"
        );
        // No temp scratch file left behind (we never reached the write).
        assert!(!any_tmp_sibling(&path));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn update_config_accepts_well_formed_body() {
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let (shared, path, auth) = state_from_temp_file();

        let good_body = json!({
            "model_list": [
                {
                    "model_name": "gpt-4o",
                    "oxllm_params": { "model": "openai/gpt-4o" }
                }
            ]
        });

        let state = axum::extract::State(shared.clone());
        let ext = axum::Extension(auth);
        let _response = update_config(state, ext, Json(good_body))
            .await
            .expect("well-formed body is accepted");

        let on_disk = std::fs::read_to_string(&path).expect("read back temp config");
        assert!(
            on_disk.contains("gpt-4o"),
            "well-formed body should be written"
        );
        assert!(!any_tmp_sibling(&path));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn update_config_rejects_private_api_base() {
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let (shared, path, auth) = state_from_temp_file();
        let seed = "model_list: []\n";
        std::fs::write(&path, seed).expect("seed good config");

        let bad_body = json!({
            "model_list": [
                {
                    "model_name": "local",
                    "oxllm_params": {
                        "model": "openai/gpt-4o",
                        "api_base": "http://127.0.0.1:8080/v1"
                    }
                }
            ]
        });

        let result = update_config(
            axum::extract::State(shared),
            axum::Extension(auth),
            Json(bad_body),
        )
        .await;
        let err = result.expect_err("loopback api_base must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        let on_disk = std::fs::read_to_string(&path).expect("read back");
        assert_eq!(on_disk, seed, "rejected body must not overwrite the file");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn update_config_rejects_open_mode_when_require_auth() {
        let _lock = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let (shared, path, auth) = state_from_temp_file();
        let seed = "model_list: []\n";
        std::fs::write(&path, seed).expect("seed");

        let open_body = json!({
            "model_list": [
                {
                    "model_name": "gpt-4o",
                    "oxllm_params": { "model": "openai/gpt-4o" }
                }
            ]
        });
        let result = update_config_with_auth_requirement(shared, auth, open_body, true).await;
        let err = result.expect_err("secret-less body must be rejected when OXLLM_REQUIRE_AUTH");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("OXLLM_REQUIRE_AUTH"),
            "error should mention the env guard: {}",
            err.message
        );
        let on_disk = std::fs::read_to_string(&path).expect("read back");
        assert_eq!(on_disk, seed);
        let _ = std::fs::remove_file(&path);
    }

    // ---- C30 MEDIUM: atomic config write -----------------------------------

    /// True if any sibling of `path` is one of *this* path's atomic-write
    /// scratch files (named `<filename>.<pid>.<nanos>.tmp`). Scoped to the
    /// path's own prefix rather than any `.tmp` in the directory, because the
    /// shared system temp dir may hold unrelated `.tmp` files from other
    /// parallel tests.
    fn any_tmp_sibling(path: &std::path::Path) -> bool {
        let Some(parent) = path.parent() else {
            return false;
        };
        let prefix = format!(
            "{}.",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("")
        );
        std::fs::read_dir(parent)
            .map(|entries| {
                entries.filter_map(Result::ok).any(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with(&prefix) && name.ends_with(".tmp")
                })
            })
            .unwrap_or(false)
    }

    #[test]
    fn atomic_write_config_replaces_file_and_leaves_no_tmp() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxllm_test_atomic_write_{}.yaml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, "before\n").expect("seed file");

        atomic_write_config(&path, b"after\n").expect("atomic write succeeds");

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after\n");
        assert!(
            !any_tmp_sibling(&path),
            "no .tmp scratch file should remain after a successful write"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn atomic_write_config_cleans_up_tmp_on_write_failure() {
        // Writing into a missing directory fails before rename; the temp file
        // path is never created, but the cleanup branch (`remove_file(&tmp)`)
        // runs and the error is surfaced rather than swallowed.
        let dir = std::env::temp_dir();
        let missing = dir.join(format!("oxllm_test_missing_{}", uuid::Uuid::new_v4()));
        let target = missing.join("config.yaml");

        let err = atomic_write_config(&target, b"body").expect_err("write must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    // ---- C30 MEDIUM: atomic write must preserve a tightened file mode -------
    //
    // `std::fs::write` creates the temp with the umask default (typically
    // 0o644); the rename replaces the inode, so an operator's `chmod 600
    // config.yaml` would silently relax on the first `POST /config/update`,
    // exposing the verbatim secrets (U17) to every local user. The temp is
    // now chmod'd to the destination's existing mode (or 0o600 on first
    // create) before the rename. Unix-only: file modes do not exist on
    // non-Unix targets.
    #[cfg(unix)]
    #[test]
    fn atomic_write_config_preserves_tightened_file_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxllm_test_atomic_mode_{}.yaml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, "before\n").expect("seed file");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod 600");

        atomic_write_config(&path, b"after\n").expect("atomic write succeeds");

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after\n");
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "tightened mode (0o600) must survive the atomic write, got {mode:o}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_config_preserves_operator_chosen_mode() {
        // An operator-chosen mode other than 0o600 (e.g. group-readable
        // 0o640) is preserved rather than reset to 0o600.
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxllm_test_atomic_mode_640_{}.yaml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, "before\n").expect("seed file");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).expect("chmod 640");

        atomic_write_config(&path, b"after\n").expect("atomic write succeeds");

        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o640,
            "operator-chosen mode must survive, got {mode:o}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_config_first_time_create_uses_secure_mode() {
        // Creating a brand-new config (destination does not exist) must
        // default to 0o600, never the umask's 0o644.
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxllm_test_atomic_mode_new_{}.yaml",
            uuid::Uuid::new_v4()
        ));
        assert!(!path.exists(), "precondition: destination absent");

        atomic_write_config(&path, b"first\n").expect("atomic create succeeds");

        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "first-time create must default to 0o600, got {mode:o}"
        );
        let _ = std::fs::remove_file(&path);
    }

    // ---- C30 LOW: redaction parity (config vs cache_settings) --------------
    //
    // Both GET /config and GET /cache/settings now route through the single
    // shared `redact_secrets`. This test feeds the same synthetic secret
    // shapes — covering both config-side (AWS Bedrock, master_key) and
    // cache-side (redis_url) fields — and asserts they all redact identically.
    // A drift between two private denylists was the original bug.

    #[test]
    fn redact_secrets_covers_config_and_cache_secret_shapes_uniformly() {
        let mut v = json!({
            "general_settings": {
                "master_key": "sk-master-test-key",
                "secret_key": "lf-secret"
            },
            "model_list": [{
                "oxllm_params": {
                    "api_key": "sk-openai",
                    "aws_secret_access_key": "AWSSECRET",
                    "aws_session_token": "AQoSESSION"
                }
            }],
            "oxllm_settings": {
                "cache": {
                    "type": "redis",
                    "redis_url": "redis://:hunter2@localhost:6379",
                    "qdrant_api_key": "qdrant-secret"
                }
            }
        });
        redact_secrets(&mut v);
        assert_eq!(v["general_settings"]["master_key"], REDACTED);
        assert_eq!(v["general_settings"]["secret_key"], REDACTED);
        assert_eq!(v["model_list"][0]["oxllm_params"]["api_key"], REDACTED);
        assert_eq!(
            v["model_list"][0]["oxllm_params"]["aws_secret_access_key"],
            REDACTED
        );
        assert_eq!(
            v["model_list"][0]["oxllm_params"]["aws_session_token"],
            REDACTED
        );
        assert_eq!(v["oxllm_settings"]["cache"]["redis_url"], REDACTED);
        // `qdrant_api_key` is covered by the `api_key` substring rule.
        assert_eq!(v["oxllm_settings"]["cache"]["qdrant_api_key"], REDACTED);
        // Benign config survives.
        assert_eq!(v["oxllm_settings"]["cache"]["type"], "redis");
    }

    // ---- C30 HIGH: numeric pricing/token fields must not be over-redacted ----
    //
    // The denylist contains the bare `token` entry to catch `access_token` /
    // `refresh_token` / `aws_session_token`. Because matching is now restricted
    // to string values, numeric fields whose NAME contains "token" (the six
    // `OxllmParams` pricing/caching fields) survive with their original value
    // AND type. Previously they were corrupted to the string "***".

    #[test]
    fn redact_secrets_preserves_numeric_pricing_and_token_fields() {
        let mut v = json!({
            "oxllm_params": {
                "model": "openai/gpt-4o",
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000015,
                "output_cost_per_reasoning_token": 0.0000276,
                "cache_read_input_token_cost": 0.00000125,
                "cache_creation_input_token_cost": 0.0000069,
                "prompt_caching_min_tokens": 1024,
                "max_tokens": 4096,
                "max_input_tokens": 128000,
                "api_key": "sk-secret"
            }
        });
        redact_secrets(&mut v);
        let p = &v["oxllm_params"];
        // The real secret is still redacted.
        assert_eq!(p["api_key"], REDACTED);
        // Every numeric pricing/token field survives with value AND type.
        assert_eq!(p["input_cost_per_token"], 0.000005);
        assert!(p["input_cost_per_token"].is_number());
        assert_eq!(p["output_cost_per_token"], 0.000015);
        assert!(p["output_cost_per_token"].is_number());
        assert_eq!(p["output_cost_per_reasoning_token"], 0.0000276);
        assert!(p["output_cost_per_reasoning_token"].is_number());
        assert_eq!(p["cache_read_input_token_cost"], 0.00000125);
        assert!(p["cache_read_input_token_cost"].is_number());
        assert_eq!(p["cache_creation_input_token_cost"], 0.0000069);
        assert!(p["cache_creation_input_token_cost"].is_number());
        assert_eq!(p["prompt_caching_min_tokens"], 1024);
        assert!(p["prompt_caching_min_tokens"].is_number());
        assert_eq!(p["max_tokens"], 4096);
        assert!(p["max_tokens"].is_number());
        assert_eq!(p["max_input_tokens"], 128000);
        assert!(p["max_input_tokens"].is_number());
    }

    // ---- C30 HIGH: denylist must cover jwt/database/azure/gcs secrets -------
    //
    // These four fields were missing from the denylist and leaked verbatim via
    // GET /config and GET /cache/settings (both master-key-only). Their values
    // are strings, so they are now caught by the (string-restricted) denylist.

    #[test]
    fn redact_secrets_redacts_jwt_database_azure_gcs_secret_fields() {
        let mut v = json!({
            "general_settings": {
                "jwt_secret": "super-secret-jwt-signing-key",
                "database_url": "postgres://proxy:hunter2@10.0.0.5:5432/oxllm",
                "host": "0.0.0.0",
                "port": 4000
            },
            "oxllm_settings": {
                "cache": {
                    "type": "azure_blob",
                    "connection_string": "DefaultEndpointsProtocol=https;AccountName=ox;AccountKey=base64key==;EndpointSuffix=core.windows.net"
                }
            },
            "model_list": [{
                "oxllm_params": {
                    "model": "vertex_ai/text-embedding-004",
                    "service_account_json": "{\"private_key\":\"-----BEGIN PRIVATE KEY-----\\nMIIEvQIB\\n-----END PRIVATE KEY-----\\n\",\"client_email\":\"svc@example.iam.gserviceaccount.com\"}",
                    "regions": ["us-central1"]
                }
            }]
        });
        redact_secrets(&mut v);
        assert_eq!(v["general_settings"]["jwt_secret"], REDACTED);
        assert_eq!(v["general_settings"]["database_url"], REDACTED);
        assert_eq!(v["oxllm_settings"]["cache"]["connection_string"], REDACTED);
        assert_eq!(
            v["model_list"][0]["oxllm_params"]["service_account_json"],
            REDACTED
        );
        // Benign config survives.
        assert_eq!(v["general_settings"]["host"], "0.0.0.0");
        assert_eq!(v["general_settings"]["port"], 4000);
        assert_eq!(
            v["model_list"][0]["oxllm_params"]["regions"][0],
            "us-central1"
        );
    }

    // ---- C30 LOW: redact_secrets recursion is depth-capped -----------------
    //
    // The recursion stops at REDACT_MAX_DEPTH, so a secret buried deeper than
    // the cap is never reached and survives. (A truly stack-overflowing depth
    // cannot even be constructed as a `serde_json::Value` — its own `Drop` is
    // recursive — so the cap is `redact_secrets`'s defense against the nesting
    // that does fit in the 10 MiB body limit.)

    #[test]
    fn redact_secrets_depth_cap_stops_recursion_at_max_depth() {
        use crate::handlers::management::REDACT_MAX_DEPTH;

        // Nest an api_key just past the cap. The recursion stops before
        // reaching it, so the secret survives unredacted.
        let depth = REDACT_MAX_DEPTH + 5;
        let mut deep = json!({ "api_key": "sk-buried" });
        for _ in 0..depth {
            deep = json!({ "inner": deep });
        }
        redact_secrets(&mut deep);
        // Walk back down to the buried secret and assert it survived.
        let mut cursor = &deep;
        for _ in 0..depth {
            cursor = &cursor["inner"];
        }
        assert_eq!(cursor["api_key"], "sk-buried");
    }

    // ---- C30 LOW: redact_secrets "token" substring over-match (fail-safe) ---
    //
    // The bare "token" denylist entry is substring-matched against key names
    // to catch access_token / aws_session_token / refresh_token. As a
    // documented fail-safe trade-off, it also redacts benign string-valued
    // fields whose name merely contains "token" (e.g. tokenizer_model). This
    // is intentional: over-redaction (hiding a harmless value) is safer than
    // under-redaction (leaking a credential). This test pins the current
    // behavior so a future tightening of the "token" rule surfaces as a
    // deliberate change rather than a silent drift.

    #[test]
    fn redact_secrets_token_substring_over_matches_benign_string_fields() {
        let mut v = json!({
            "tokenizer_model": "bpe",
            "tokenizer": "tiktoken",
            "max_tokens": 4096,
            "api_key": "sk-secret"
        });
        redact_secrets(&mut v);
        // Benign string fields whose names contain "token" are over-redacted
        // (fail-safe).
        assert_eq!(v["tokenizer_model"], REDACTED);
        assert_eq!(v["tokenizer"], REDACTED);
        // Numeric token fields survive (string-only restriction).
        assert_eq!(v["max_tokens"], 4096);
        assert!(v["max_tokens"].is_number());
        // Real secrets are still redacted.
        assert_eq!(v["api_key"], REDACTED);
    }
}
