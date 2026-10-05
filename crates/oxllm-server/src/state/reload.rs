//! Config hot-reload: rebuild config-derived state while preserving
//! infrastructure connections.
//!
//! [`AppState::reload_from_config`] is a thin wrapper over the shared
//! [`assemble`](super::build::assemble) construction path: it clones the
//! previous state's infrastructure handles into [`BuildParts`] and asks for
//! [`ReusePolicy::Preserve`]. The `*_config_unchanged` helpers below are the
//! preservation rules: when the config section a component was built from is
//! byte-identical between old and new config, the existing `Arc` (and its
//! accumulated runtime state) is reused instead of rebuilt.

use crate::config::ProxyConfig;
use crate::state::AppState;
use crate::state::build::{self, BuildParts, ReusePolicy};

impl AppState {
    /// Rebuild config-derived fields while preserving infrastructure connections.
    ///
    /// Used by the config hot-reload watcher to swap in a new config without
    /// dropping the database handle, Redis-backed cache, spend-writer channel,
    /// shared HTTP client, JWT validator, or auth brake. Only fields derived
    /// from [`ProxyConfig`] (deployments, router, hooks, master-key hash,
    /// config itself) are rebuilt.
    pub fn reload_from_config(&self, new_config: ProxyConfig) -> anyhow::Result<Self> {
        // Mirror the startup `OXLLM_REQUIRE_AUTH` guard (`app::enforce_require_auth_guard`):
        // a config edit that removes the last auth secret must not silently
        // drop the proxy into open mode. Returning an error here makes the
        // watcher keep the previous state (see `config::watcher::do_reload`).
        enforce_reload_require_auth(&new_config)?;

        let parts = BuildParts {
            registry: self.registry.clone(),
            cache: self.cache.clone(),
            cache_backend: self.cache_backend,
            shutdown: self.shutdown.clone(),
            spend_tx: self.spend_tx.clone(),
            #[cfg(feature = "db")]
            db: self.db.clone(),
            #[cfg(not(feature = "db"))]
            db: self.db,
            coordinator: self.coordinator.clone(),
        };
        build::assemble(new_config, parts, ReusePolicy::Preserve(self))
    }
}

/// Reload-path counterpart to `app::enforce_require_auth_guard`: reads
/// `OXLLM_REQUIRE_AUTH` and rejects a reload that would remove every auth
/// secret, dropping the proxy into open mode. Split from the pure guard
/// below so tests can exercise the logic without mutating process-wide env
/// (`std::env::set_var` is `unsafe` in edition 2024 and this crate is
/// `#![deny(unsafe_code)]`).
fn enforce_reload_require_auth(config: &ProxyConfig) -> anyhow::Result<()> {
    let env_value = std::env::var("OXLLM_REQUIRE_AUTH").ok();
    reload_require_auth_guard(
        &config.general_settings.master_key,
        &config.general_settings.jwt_secret,
        &config.general_settings.database_url,
        env_value.as_deref(),
    )
}

/// Pure implementation of the reload-path `OXLLM_REQUIRE_AUTH` guard. When
/// the env var is `"true"`/`"1"`, the new config must keep at least one auth
/// path (`master_key`, `jwt_secret`, or `database_url`); otherwise the reload
/// would silently re-open every endpoint — the exact regression the startup
/// guard in `app.rs` prevents for initial boot.
fn reload_require_auth_guard(
    master_key: &Option<String>,
    jwt_secret: &Option<String>,
    database_url: &Option<String>,
    require_auth_env: Option<&str>,
) -> anyhow::Result<()> {
    let require_auth = require_auth_env
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    if require_auth && !crate::app::auth_is_configured(master_key, jwt_secret, database_url) {
        anyhow::bail!(
            "OXLLM_REQUIRE_AUTH is set but the reloaded config has no master_key, \
 jwt_secret, or database_url; refusing to apply reload in open mode. Keep an \
 auth path configured or unset OXLLM_REQUIRE_AUTH."
        );
    }
    Ok(())
}

/// Returns true when the JWT-relevant fields of `general_settings` are
/// identical between the old and new configs, so the `JwtValidator` `Arc`
/// can be reused on hot-reload. This is deliberately a field-subset check
/// (not `PartialEq` on the whole `GeneralSettings`): unrelated edits such as
/// `log_format` or `max_request_body_size_mb` must not churn the validator.
pub(crate) fn jwt_secret_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    let g_old = &old.general_settings;
    let g_new = &new.general_settings;
    g_old.jwt_secret == g_new.jwt_secret
        && g_old.jwt_algorithm == g_new.jwt_algorithm
        && g_old.jwt_audience == g_new.jwt_audience
        && g_old.jwt_issuer == g_new.jwt_issuer
        && g_old.jwt_replay_protection == g_new.jwt_replay_protection
}

/// Returns true when the alerting config (backends + webhook block) is
/// identical between the old and new configs, so the `AlertState` `Arc`
/// can be reused on hot-reload. Uses the derived `PartialEq` on the alerting
/// config sections, so any field change (including `webhook_type`)
/// conservatively triggers a rebuild.
pub(crate) fn alerting_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    old.oxllm_settings.alerting == new.oxllm_settings.alerting
        && old.oxllm_settings.alerting_config == new.oxllm_settings.alerting_config
        && old.oxllm_settings.budget_duration == new.oxllm_settings.budget_duration
        && old.general_settings.allow_private_api_base
            == new.general_settings.allow_private_api_base
}

/// Returns true when the Slack alerting config (plus `budget_duration`, which
/// sets the dedup window) is identical between the old and new configs, so the
/// Slack hook's [`AlertState`] Arc — and its in-memory fired markers — can be
/// reused on hot-reload instead of being discarded (which would cause
/// duplicate alerts on the next threshold crossing).
///
/// [`AlertState`]: crate::hooks::AlertState
pub(crate) fn slack_alerting_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    old.oxllm_settings.slack_alerting_config == new.oxllm_settings.slack_alerting_config
        && old.oxllm_settings.budget_duration == new.oxllm_settings.budget_duration
        && old.general_settings.allow_private_api_base
            == new.general_settings.allow_private_api_base
}

/// Like [`slack_alerting_config_unchanged`] but for the PagerDuty hook.
pub(crate) fn pagerduty_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    old.oxllm_settings.pagerduty_config == new.oxllm_settings.pagerduty_config
        && old.oxllm_settings.budget_duration == new.oxllm_settings.budget_duration
        && old.general_settings.allow_private_api_base
            == new.general_settings.allow_private_api_base
}

/// Like [`slack_alerting_config_unchanged`] but for the email hook.
pub(crate) fn email_alert_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    old.oxllm_settings.email_alert_config == new.oxllm_settings.email_alert_config
        && old.oxllm_settings.budget_duration == new.oxllm_settings.budget_duration
        && old.general_settings.allow_private_api_base
            == new.general_settings.allow_private_api_base
}

/// Returns true when the semantic-cache config is identical between the old
/// and new configs (including the resolved embedding endpoint), so the
/// `SemanticCacheRuntime` Arc — and its accumulated embeddings — can be
/// reused on hot-reload.
///
/// `allow_private_api_base` is included because the runtime owns its own
/// `reqwest::Client` (see [`SemanticCacheRuntime::http_client`]), built with
/// the SSRF-aware DNS resolver gated on that flag. Toggling the flag must
/// rebuild the runtime so the embedding-fetch client carries the new resolver
/// policy — mirroring [`langfuse_config_unchanged`] and
/// [`http_client_config_unchanged`].
///
/// [`SemanticCacheRuntime::http_client`]: crate::hooks::cache_hook::SemanticCacheRuntime::http_client
pub(crate) fn semantic_cache_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    let old_sc = old
        .oxllm_settings
        .cache
        .as_ref()
        .and_then(|c| c.semantic_cache.as_ref());
    let new_sc = new
        .oxllm_settings
        .cache
        .as_ref()
        .and_then(|c| c.semantic_cache.as_ref());
    old_sc == new_sc
        && old.general_settings.allow_private_api_base
            == new.general_settings.allow_private_api_base
        && crate::cache::semantic::resolve_embedding_endpoint(
            old,
            old_sc.unwrap_or(&Default::default()),
        ) == crate::cache::semantic::resolve_embedding_endpoint(
            new,
            new_sc.unwrap_or(&Default::default()),
        )
}

/// Returns true when the langfuse config is identical between the old and
/// new configs, so the [`LangfuseHook`] Arc — and its background flush task
/// — can be reused on hot-reload.
///
/// [`LangfuseHook`]: crate::hooks::LangfuseHook
pub(crate) fn langfuse_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    old.oxllm_settings.langfuse_config == new.oxllm_settings.langfuse_config
        && old.general_settings.allow_private_api_base
            == new.general_settings.allow_private_api_base
}

/// Returns true when the mlflow config is identical between the old and new
/// configs, so the [`MlflowHook`] Arc — and its lazily-resolved `run_id` —
/// can be reused on hot-reload. `allow_private_api_base` is included because
/// the hook's reqwest client carries the SSRF-aware DNS resolver gated on
/// that flag, so toggling it must rebuild the hook.
///
/// [`MlflowHook`]: crate::hooks::MlflowHook
pub(crate) fn mlflow_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    old.oxllm_settings.mlflow_config == new.oxllm_settings.mlflow_config
        && old.general_settings.allow_private_api_base
            == new.general_settings.allow_private_api_base
}

/// Returns true when the HTTP connection-pool settings read by
/// [`build_http_client`](super::build::build_http_client) are identical
/// between the old and new configs, so the shared `reqwest::Client` Arc can
/// be reused on hot-reload. Any change to the pool sizing or timeouts
/// triggers a rebuild so operators can tune the pool without restarting.
///
/// This is deliberately a field-subset check rather than whole-config
/// `PartialEq`: unrelated edits must not churn a client whose idle pool has
/// accumulated warm connections.
pub(crate) fn http_client_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    let g_old = &old.general_settings;
    let g_new = &new.general_settings;
    g_old.http_pool_max_idle_per_host == g_new.http_pool_max_idle_per_host
        && g_old.http_pool_idle_timeout_secs == g_new.http_pool_idle_timeout_secs
        && g_old.http_connect_timeout_secs == g_new.http_connect_timeout_secs
        && g_old.allow_private_api_base == g_new.allow_private_api_base
}

/// Like [`http_client_config_unchanged`] but ignores
/// `allow_private_api_base`. Used for the image-download client
/// ([`build_image_http_client`](super::build::build_image_http_client)),
/// which always installs the SSRF resolver regardless of that opt-in, so
/// flipping it must not churn a client whose idle pool has warm connections.
pub(crate) fn http_pool_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    let g_old = &old.general_settings;
    let g_new = &new.general_settings;
    g_old.http_pool_max_idle_per_host == g_new.http_pool_max_idle_per_host
        && g_old.http_pool_idle_timeout_secs == g_new.http_pool_idle_timeout_secs
        && g_old.http_connect_timeout_secs == g_new.http_connect_timeout_secs
}

/// Returns true when the guardrails config section (plus the
/// `allow_private_api_base` flag that gates SSRF checks during guardrail
/// construction) is identical between the old and new configs, so the
/// pre-built [`CustomGuardrailChain`] and [`ModelArmorGuardrail`] can be
/// reused on hot-reload instead of recompiling regexes and rebuilding HTTP
/// clients.
///
/// `GuardrailsConfig` does not derive `PartialEq`, so the section is compared
/// via its `serde_json::Value` representation — a byte-level equality check
/// that triggers a conservative rebuild on any field change.
///
/// [`CustomGuardrailChain`]: crate::guardrails::CustomGuardrailChain
/// [`ModelArmorGuardrail`]: crate::guardrails::ModelArmorGuardrail
pub(crate) fn guardrails_config_unchanged(old: &ProxyConfig, new: &ProxyConfig) -> bool {
    old.general_settings.allow_private_api_base == new.general_settings.allow_private_api_base
        && match (
            serde_json::to_value(&old.general_settings.guardrails).ok(),
            serde_json::to_value(&new.general_settings.guardrails).ok(),
        ) {
            (Some(a), Some(b)) => a == b,
            // A serialization failure on either side is conservatively
            // treated as "changed": returning `true` when both sides fail
            // (None == None) would preserve a potentially stale chain
            // instead of rebuilding it.
            _ => false,
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{OxllmSettings, RouterSettings};
    use std::sync::Arc;

    /// Helper for building a `ProxyConfig` with explicit `general_settings`.
    fn proxy_with_general(
        general: crate::config::types::GeneralSettings,
        settings: OxllmSettings,
    ) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings,
            general_settings: general,
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        }
    }

    /// Issue #13: a JWT-only deployment (`jwt_secret` set, `master_key` unset)
    /// must keep its auth brake after a hot-reload. Previously the reload path
    /// only checked `master_key`, silently disabling brute-force protection.
    #[test]
    fn reload_preserves_auth_brake_for_jwt_only_deployment() {
        let jwt_only = crate::config::types::GeneralSettings {
            jwt_secret: Some("s3cr3t-long-enough!".to_string()),
            master_key: None,
            ..Default::default()
        };

        let initial = proxy_with_general(jwt_only.clone(), OxllmSettings::default());
        let state = AppState::new(initial).expect("build initial state");

        // Sanity check: startup path installs the auth brake for JWT-only.
        assert!(
            state.auth_brake.is_some(),
            "auth_brake must be present at startup for a JWT-only deployment"
        );

        // Reload with the same JWT-only config (simulates any config edit
        // that leaves JWT auth in place).
        let reloaded_config = proxy_with_general(jwt_only.clone(), OxllmSettings::default());
        let reloaded = state
            .reload_from_config(reloaded_config)
            .expect("reload state");

        assert!(
            reloaded.auth_brake.is_some(),
            "auth_brake must be preserved after reload for a JWT-only deployment (#13)"
        );
    }

    /// Reloading from a config with no auth secrets and no DB must still
    /// produce `auth_brake: None` (the fix should not over-correct).
    #[test]
    fn reload_disables_auth_brake_when_no_auth_secrets() {
        let jwt_only = crate::config::types::GeneralSettings {
            jwt_secret: Some("s3cr3t-long-enough!".to_string()),
            master_key: None,
            ..Default::default()
        };
        let initial = proxy_with_general(jwt_only, OxllmSettings::default());
        let state = AppState::new(initial).expect("build initial state");
        assert!(state.auth_brake.is_some());

        // Reload with a config that drops master_key, jwt_secret, and database.
        let no_auth = crate::config::types::GeneralSettings {
            jwt_secret: None,
            master_key: None,
            database_url: None,
            ..Default::default()
        };
        let reloaded_config = proxy_with_general(no_auth, OxllmSettings::default());
        let reloaded = state
            .reload_from_config(reloaded_config)
            .expect("reload state");

        assert!(
            reloaded.auth_brake.is_none(),
            "auth_brake must be None when no master_key, jwt_secret, or db is set"
        );
    }

    /// Issue #13: reloading must preserve the *same* auth brake `Arc` when the
    /// secret material is unchanged (avoids dropping in-flight brake state).
    #[test]
    fn reload_reuses_auth_brake_arc_when_unchanged() {
        let jwt_only = crate::config::types::GeneralSettings {
            jwt_secret: Some("s3cr3t-long-enough!".to_string()),
            master_key: None,
            ..Default::default()
        };
        let initial = proxy_with_general(jwt_only.clone(), OxllmSettings::default());
        let state = AppState::new(initial).expect("build initial state");
        let original = state.auth_brake.clone().expect("auth_brake present");

        let reloaded_config = proxy_with_general(jwt_only, OxllmSettings::default());
        let reloaded = state
            .reload_from_config(reloaded_config)
            .expect("reload state");

        let reused = reloaded.auth_brake.expect("auth_brake preserved");
        assert!(
            Arc::as_ptr(&reused) == Arc::as_ptr(&original),
            "reload should reuse the existing AuthBrake Arc when JWT secret is unchanged"
        );
    }

    /// F8: with `OXLLM_REQUIRE_AUTH=true`, a reload whose config drops every
    /// auth path must be rejected — applying it would silently re-open every
    /// endpoint.
    #[test]
    fn reload_require_auth_guard_rejects_open_mode_when_env_true() {
        let none: Option<String> = None;
        let result = reload_require_auth_guard(&none, &none, &none, Some("true"));
        assert!(
            result.is_err(),
            "reload to a secret-less config must fail when OXLLM_REQUIRE_AUTH=true"
        );
        let result = reload_require_auth_guard(&none, &none, &none, Some("1"));
        assert!(
            result.is_err(),
            "reload to a secret-less config must fail when OXLLM_REQUIRE_AUTH=1"
        );
    }

    /// F8: the guard error must name the env var and the missing secrets so
    /// operators can act on the watcher error log.
    #[test]
    fn reload_require_auth_guard_error_message_is_actionable() {
        let none: Option<String> = None;
        let err = reload_require_auth_guard(&none, &none, &none, Some("true"))
            .expect_err("must reject open-mode reload");
        let msg = format!("{err}");
        assert!(msg.contains("OXLLM_REQUIRE_AUTH"), "msg: {msg}");
        assert!(msg.contains("master_key"), "msg: {msg}");
        assert!(msg.contains("jwt_secret"), "msg: {msg}");
        assert!(msg.contains("database_url"), "msg: {msg}");
    }

    /// F8: keeping any auth path satisfies the guard — only a reload to
    /// *zero* auth is rejected.
    #[test]
    fn reload_require_auth_guard_allows_when_any_secret_kept() {
        let key = Some("sk-master-test-key".to_string());
        let jwt = Some("s3cr3t-long-enough!".to_string());
        let db = Some("sqlite::memory:".to_string());
        let none: Option<String> = None;
        assert!(reload_require_auth_guard(&key, &none, &none, Some("true")).is_ok());
        assert!(reload_require_auth_guard(&none, &jwt, &none, Some("true")).is_ok());
        assert!(reload_require_auth_guard(&none, &none, &db, Some("true")).is_ok());
        assert!(reload_require_auth_guard(&key, &jwt, &none, Some("1")).is_ok());
    }

    /// F8: when `OXLLM_REQUIRE_AUTH` is unset/false, open-mode reloads stay
    /// allowed (the guard must not change default behavior).
    #[test]
    fn reload_require_auth_guard_allows_open_mode_when_env_unset() {
        let none: Option<String> = None;
        assert!(reload_require_auth_guard(&none, &none, &none, None).is_ok());
        assert!(reload_require_auth_guard(&none, &none, &none, Some("false")).is_ok());
        assert!(reload_require_auth_guard(&none, &none, &none, Some("")).is_ok());
        assert!(reload_require_auth_guard(&none, &none, &none, Some("no")).is_ok());
    }

    /// F8: end-to-end through `reload_from_config` — with the env var unset
    /// in this test process, dropping all secrets still succeeds (guard is
    /// inert), proving the guard is wired into the reload path without
    /// changing default reload behavior.
    #[test]
    fn reload_from_config_applies_guard_and_succeeds_when_env_unset() {
        let with_key = crate::config::types::GeneralSettings {
            master_key: Some("sk-master-test-key".to_string()),
            ..Default::default()
        };
        let initial = proxy_with_general(with_key, OxllmSettings::default());
        let state = AppState::new(initial).expect("build initial state");
        assert!(state.master_key_hash.is_some());

        let no_auth = crate::config::types::GeneralSettings::default();
        let reloaded = state
            .reload_from_config(proxy_with_general(no_auth, OxllmSettings::default()))
            .expect("reload with OXLLM_REQUIRE_AUTH unset must succeed");
        assert!(reloaded.master_key_hash.is_none());
    }

    /// H6: `http_client_config_unchanged` must return true when all three
    /// pool-related fields match between old and new config, so the shared
    /// `reqwest::Client` Arc (with its warm connection pool) is reused.
    #[test]
    fn http_client_config_unchanged_when_pool_fields_match() {
        let gs = crate::config::types::GeneralSettings {
            http_pool_max_idle_per_host: Some(5),
            http_pool_idle_timeout_secs: Some(30),
            http_connect_timeout_secs: Some(10),
            ..Default::default()
        };
        let old = proxy_with_general(gs.clone(), OxllmSettings::default());
        let new = proxy_with_general(gs, OxllmSettings::default());
        assert!(
            http_client_config_unchanged(&old, &new),
            "identical pool config must preserve the http client"
        );
    }

    /// H6: a change to ANY of the three pool fields must report the config as
    /// changed so the client is rebuilt with the new pool settings.
    #[test]
    fn http_client_config_unchanged_detects_each_pool_field_change() {
        let base = crate::config::types::GeneralSettings {
            http_pool_max_idle_per_host: Some(5),
            http_pool_idle_timeout_secs: Some(30),
            http_connect_timeout_secs: Some(10),
            ..Default::default()
        };
        let old = proxy_with_general(base.clone(), OxllmSettings::default());

        // max_idle_per_host changed
        let mut changed = base.clone();
        changed.http_pool_max_idle_per_host = Some(10);
        assert!(
            !http_client_config_unchanged(
                &old,
                &proxy_with_general(changed, OxllmSettings::default())
            ),
            "max_idle_per_host change must trigger rebuild"
        );

        // idle_timeout_secs changed
        let mut changed = base.clone();
        changed.http_pool_idle_timeout_secs = Some(60);
        assert!(
            !http_client_config_unchanged(
                &old,
                &proxy_with_general(changed, OxllmSettings::default())
            ),
            "idle_timeout_secs change must trigger rebuild"
        );

        // connect_timeout_secs changed
        let mut changed = base.clone();
        changed.http_connect_timeout_secs = Some(20);
        assert!(
            !http_client_config_unchanged(
                &old,
                &proxy_with_general(changed, OxllmSettings::default())
            ),
            "connect_timeout_secs change must trigger rebuild"
        );

        // clearing a previously-set field also counts as a change
        let mut changed = base.clone();
        changed.http_pool_max_idle_per_host = None;
        assert!(
            !http_client_config_unchanged(
                &old,
                &proxy_with_general(changed, OxllmSettings::default())
            ),
            "transitioning Some->None must trigger rebuild"
        );
    }

    /// C05/HIGH: `allow_private_api_base` gates whether the SSRF-aware DNS
    /// resolver is installed on the shared client, so toggling it must rebuild
    /// the client (a stale client would carry the wrong resolver policy).
    #[test]
    fn http_client_config_unchanged_detects_allow_private_api_base_change() {
        let base = crate::config::types::GeneralSettings {
            allow_private_api_base: Some(false),
            ..Default::default()
        };
        let old = proxy_with_general(base.clone(), OxllmSettings::default());

        let mut changed = base.clone();
        changed.allow_private_api_base = Some(true);
        assert!(
            !http_client_config_unchanged(
                &old,
                &proxy_with_general(changed, OxllmSettings::default())
            ),
            "toggling allow_private_api_base must rebuild the client (it changes the resolver)"
        );
    }

    /// H6: unrelated config edits (e.g. master_key) must NOT churn the http
    /// client — the check is a deliberate field-subset, not whole-config.
    #[test]
    fn http_client_config_unchanged_ignores_unrelated_fields() {
        let gs = crate::config::types::GeneralSettings::default();
        let old = proxy_with_general(gs.clone(), OxllmSettings::default());
        let new_gs = crate::config::types::GeneralSettings {
            master_key: Some("sk-different-key-test-1".to_string()),
            ..gs
        };
        let new = proxy_with_general(new_gs, OxllmSettings::default());
        assert!(
            http_client_config_unchanged(&old, &new),
            "master_key change must not churn the http client pool"
        );
    }

    // ---- MEDIUM: semantic_cache_config_unchanged + allow_private_api_base -----

    /// Helper: a config with semantic cache enabled + the given
    /// `allow_private_api_base` flag.
    fn proxy_with_semantic_cache(allow_private: bool) -> ProxyConfig {
        use crate::config::types::{CacheConfig, GeneralSettings, SemanticCacheConfig};
        let settings = OxllmSettings {
            cache: Some(CacheConfig {
                semantic_cache: Some(SemanticCacheConfig {
                    enabled: true,
                    embedding_model: "text-embedding-3-small".to_string(),
                    api_base: Some("https://api.openai.com/v1".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let general = GeneralSettings {
            allow_private_api_base: Some(allow_private),
            ..Default::default()
        };
        proxy_with_general(general, settings)
    }

    /// The semantic-cache runtime owns its own `reqwest::Client`, so toggling
    /// `allow_private_api_base` must report the config as changed — the
    /// runtime must be rebuilt so its embedding-fetch client carries the new
    /// SSRF resolver policy. Mirrors `langfuse_config_unchanged`.
    #[test]
    fn semantic_cache_config_unchanged_detects_allow_private_api_base_toggle() {
        let old = proxy_with_semantic_cache(true);
        let new_true = proxy_with_semantic_cache(true);
        let new_false = proxy_with_semantic_cache(false);

        assert!(
            semantic_cache_config_unchanged(&old, &new_true),
            "identical config (same allow_private_api_base) must preserve runtime"
        );
        assert!(
            !semantic_cache_config_unchanged(&old, &new_false),
            "toggling allow_private_api_base must rebuild the semantic-cache runtime"
        );
    }

    /// End-to-end: toggling `allow_private_api_base` must rebuild the
    /// `semantic_cache_runtime` Arc (it embeds its own HTTP client). The old
    /// client carried the relaxed resolver; the new one must carry the
    /// SSRF-aware resolver, so the Arc must NOT be reused.
    #[test]
    fn semantic_cache_runtime_rebuilt_when_allow_private_api_base_toggled() {
        let initial = proxy_with_semantic_cache(true);
        let state = AppState::new(initial).expect("build initial state");
        let original = state
            .semantic_cache_runtime
            .clone()
            .expect("semantic cache runtime built");

        let reloaded = state
            .reload_from_config(proxy_with_semantic_cache(false))
            .expect("reload state");
        let rebuilt = reloaded
            .semantic_cache_runtime
            .as_ref()
            .expect("semantic cache runtime still built after reload");

        assert!(
            !std::sync::Arc::ptr_eq(rebuilt, &original),
            "toggling allow_private_api_base must rebuild the semantic-cache runtime \
             (its embedded HTTP client carries the wrong SSRF resolver policy)"
        );
    }

    /// The semantic-cache runtime must be reused when `allow_private_api_base`
    /// is unchanged (the common case: an unrelated config edit should not
    /// discard accumulated embeddings).
    #[test]
    fn semantic_cache_runtime_reused_when_allow_private_api_base_unchanged() {
        let initial = proxy_with_semantic_cache(true);
        let state = AppState::new(initial).expect("build initial state");
        let original = state
            .semantic_cache_runtime
            .clone()
            .expect("semantic cache runtime built");

        let reloaded = state
            .reload_from_config(proxy_with_semantic_cache(true))
            .expect("reload state");
        let reused = reloaded
            .semantic_cache_runtime
            .as_ref()
            .expect("semantic cache runtime still built after reload");

        assert!(
            std::sync::Arc::ptr_eq(reused, &original),
            "identical semantic-cache + SSRF config must reuse the runtime"
        );
    }

    /// `guardrails_config_unchanged` must return `true` when both configs
    /// serialize identically (the common no-op reload case).
    #[test]
    fn guardrails_config_unchanged_when_identical() {
        let old = proxy_with_general(
            crate::config::types::GeneralSettings::default(),
            OxllmSettings::default(),
        );
        let new = proxy_with_general(
            crate::config::types::GeneralSettings::default(),
            OxllmSettings::default(),
        );
        assert!(
            guardrails_config_unchanged(&old, &new),
            "identical guardrails config must be preserved"
        );
    }

    /// `guardrails_config_unchanged` must return `false` when
    /// `allow_private_api_base` differs, even if the guardrails section
    /// serializes identically.
    #[test]
    fn guardrails_config_unchanged_detects_allow_private_api_base_toggle() {
        let mk = |allow: bool| {
            proxy_with_general(
                crate::config::types::GeneralSettings {
                    allow_private_api_base: Some(allow),
                    ..Default::default()
                },
                OxllmSettings::default(),
            )
        };
        assert!(
            !guardrails_config_unchanged(&mk(true), &mk(false)),
            "allow_private_api_base toggle must force a guardrails rebuild"
        );
    }

    // ---- MEDIUM: Slack/PagerDuty/Email alert-state preservation on reload ----

    /// Helper: a config with a Slack alerting hook (allow_private so the
    /// `.invalid` webhook URL builds without DNS in the test env).
    fn proxy_with_slack_alert() -> ProxyConfig {
        use crate::config::types::{GeneralSettings, SlackAlertConfig};
        let settings = OxllmSettings {
            slack_alerting_config: Some(SlackAlertConfig {
                webhook_url: "https://hooks.example.invalid/x".to_string(),
                thresholds: vec![0.8],
                ..Default::default()
            }),
            ..Default::default()
        };
        let general = GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        };
        proxy_with_general(general, settings)
    }

    /// MEDIUM: a hot-reload that leaves the Slack config unchanged must
    /// preserve the Slack hook's `AlertState` Arc so in-memory fired markers
    /// survive — otherwise every crossed threshold re-fires after a reload.
    /// This is the regression test for the bug where `build_slack_hook_opt`
    /// always built a fresh `AlertState`, dropping the old `HashSet`.
    #[test]
    fn reload_preserves_slack_alert_state_arc_when_unchanged() {
        let initial = proxy_with_slack_alert();
        let state = AppState::new(initial).expect("build initial state");
        let original = state
            .slack_alert_state
            .clone()
            .expect("slack alert state built");

        let reloaded = state
            .reload_from_config(proxy_with_slack_alert())
            .expect("reload state");
        let reused = reloaded
            .slack_alert_state
            .as_ref()
            .expect("slack alert state preserved");

        assert!(
            std::sync::Arc::ptr_eq(reused, &original),
            "reload must reuse the Slack AlertState Arc when config is unchanged \
             (markers would otherwise be lost → duplicate alerts)"
        );
    }

    /// A Slack config change must rebuild the `AlertState` Arc (new thresholds
    /// require a fresh marker set). Confirms the preservation is gated on the
    /// config comparison rather than always reusing.
    #[test]
    fn reload_rebuilds_slack_alert_state_arc_when_thresholds_change() {
        use crate::config::types::SlackAlertConfig;
        let initial = proxy_with_slack_alert();
        let state = AppState::new(initial).expect("build initial state");
        let original = state
            .slack_alert_state
            .clone()
            .expect("slack alert state built");

        let mut changed = proxy_with_slack_alert();
        changed.oxllm_settings.slack_alerting_config = Some(SlackAlertConfig {
            webhook_url: "https://hooks.example.invalid/x".to_string(),
            thresholds: vec![0.9],
            ..Default::default()
        });
        let reloaded = state.reload_from_config(changed).expect("reload state");
        let rebuilt = reloaded
            .slack_alert_state
            .as_ref()
            .expect("slack alert state rebuilt");

        assert!(
            !std::sync::Arc::ptr_eq(rebuilt, &original),
            "threshold change must rebuild the Slack AlertState Arc"
        );
    }
}
