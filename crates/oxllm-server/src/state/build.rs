//! Canonical construction path for [`AppState`].
//!
//! [`StateBuilder`] collects the injectable infrastructure dependencies
//! (provider registry, cache, spend-writer sender, database handle,
//! coordination layer) via optional setters; unset dependencies fall back to
//! the production defaults (the global provider registry and a fresh
//! in-memory cache). The six historical `AppState::with_*` constructors are
//! kept as thin shims over the builder so existing call sites compile
//! unchanged.
//!
//! Both startup ([`StateBuilder::build`]) and hot-reload
//! ([`AppState::reload_from_config`](super::AppState::reload_from_config))
//! funnel through the single [`assemble`] function, parameterized by
//! [`ReusePolicy`]. This is what keeps the two paths from drifting apart:
//! the Bug #13 class of bug (reload dropped the auth brake for JWT-only
//! deployments because the condition was duplicated and diverged) cannot
//! recur while there is exactly one implementation of each decision.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use oxllm_core::Cache;
use oxllm_core::Deployment;
use oxllm_core::ProviderRegistry;
use oxllm_core::router::Router;

use crate::auth::hash_key;
use crate::cache::InMemoryCache;
use crate::config::ProxyConfig;
use crate::coordination::Coordinator;
use crate::credentials::CredentialResolverRegistry;
use crate::spend_writer::SpendEntry;
use crate::state::reload;
use crate::state::{AppState, CacheBackendKind, DbOpt};

/// The injectable infrastructure dependencies [`assemble`] wires into the
/// state. On the startup path these come from [`StateBuilder`]; on the
/// reload path they are cloned out of the previous state so connections
/// (cache, DB, spend writer, coordinator) survive the swap.
pub(crate) struct BuildParts {
    pub registry: Arc<ProviderRegistry>,
    pub cache: Arc<dyn Cache>,
    pub cache_backend: CacheBackendKind,
    pub shutdown: Arc<AtomicBool>,
    pub spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
    pub db: DbOpt,
    pub coordinator: Option<Coordinator>,
}

/// Controls whether [`assemble`] builds every derived field from scratch
/// (startup) or reuses the previous state's infrastructure `Arc`s where the
/// config that produced them is unchanged (hot-reload).
pub(crate) enum ReusePolicy<'a> {
    /// Startup: everything is built fresh from the config and [`BuildParts`].
    Fresh,
    /// Hot-reload: preserve the previous state's HTTP client, credential
    /// cipher, router runtime stores, JWT validator (when the JWT config is
    /// unchanged), auth brake, alert state, semantic-cache runtime, langfuse
    /// hook, and credential-resolver registry.
    Preserve(&'a AppState),
}

/// Builder for [`AppState`] with optional setters for each injectable
/// dependency. This is the single canonical construction path; prefer it
/// over adding new `AppState::with_*` constructor wrappers.
///
/// Construction is synchronous: every derived component (router, hooks,
/// guardrails, JWT validator, ...) is built from in-memory config without
/// I/O, so `build` does not need an async context.
pub struct StateBuilder {
    config: ProxyConfig,
    registry: Option<ProviderRegistry>,
    cache: Option<Arc<dyn Cache>>,
    cache_backend: Option<CacheBackendKind>,
    shutdown: Option<Arc<AtomicBool>>,
    spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
    db: DbOpt,
    coordinator: Option<Coordinator>,
    /// Test-only override for the config-derived JWT validator: lets tests
    /// wire a validator whose jti replay cache fails — a setup the
    /// config-driven path cannot express without a live Redis.
    #[cfg(test)]
    jwt_validator: Option<std::sync::Arc<crate::auth::jwt::JwtValidator>>,
}

impl StateBuilder {
    /// Start building state from a loaded config. Unset dependencies default
    /// to [`oxllm_providers::default_registry`] and a fresh [`InMemoryCache`].
    pub fn new(config: ProxyConfig) -> Self {
        Self {
            config,
            registry: None,
            cache: None,
            cache_backend: None,
            shutdown: None,
            spend_tx: None,
            db: None,
            coordinator: None,
            #[cfg(test)]
            jwt_validator: None,
        }
    }

    /// Use an explicit provider registry (for testing).
    pub fn with_registry(mut self, registry: ProviderRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Use an explicit cache (for testing, or the Redis-backed `DualCache`
    /// from the production bootstrap).
    pub fn with_cache(mut self, cache: Arc<dyn Cache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the cache backend kind explicitly. Used by the production
    /// bootstrap (`crate::app::build_cache`) so the readiness probe sees the
    /// backend that was *actually constructed*, not the config string.
    pub fn with_cache_backend(mut self, kind: CacheBackendKind) -> Self {
        self.cache_backend = Some(kind);
        self
    }

    /// Set the process-wide graceful-shutdown flag explicitly. Used by the
    /// production bootstrap ([`crate::app::run`]) so the health/readiness
    /// probes observe drain state. When unset, a fresh (always-`false`) flag
    /// is constructed — correct for tests and paths that never serve traffic.
    pub fn with_shutdown(mut self, shutdown: Arc<AtomicBool>) -> Self {
        self.shutdown = Some(shutdown);
        self
    }

    /// Wire in the optional database handle (virtual-key persistence +
    /// spend logs).
    pub fn with_db(mut self, db: DbOpt) -> Self {
        self.db = db;
        self
    }

    /// Wire in the batched spend-writer sender. `None` means no spend
    /// writer (no database configured).
    pub fn with_spend_tx(
        mut self,
        spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
    ) -> Self {
        self.spend_tx = spend_tx;
        self
    }

    /// Wire in the optional distributed coordination layer.
    pub fn with_coordinator(mut self, coordinator: Option<Coordinator>) -> Self {
        self.coordinator = coordinator;
        self
    }

    /// Override the config-derived JWT validator (test-only): used to wire a
    /// validator whose jti replay cache fails so outage handling can be
    /// exercised without a live Redis.
    #[cfg(test)]
    pub fn with_jwt_validator(
        mut self,
        validator: std::sync::Arc<crate::auth::jwt::JwtValidator>,
    ) -> Self {
        self.jwt_validator = Some(validator);
        self
    }

    /// Build the state. All config-derived fields are constructed fresh.
    pub fn build(self) -> anyhow::Result<AppState> {
        let registry: Arc<ProviderRegistry> = match self.registry {
            Some(r) => Arc::new(r),
            None => Arc::new(oxllm_providers::default_registry()),
        };
        let cache: Arc<dyn Cache> = match self.cache {
            Some(c) => c,
            None => Arc::new(InMemoryCache::default()),
        };
        let cache_backend = self
            .cache_backend
            .unwrap_or_else(|| CacheBackendKind::from_config(&self.config));
        let shutdown = self.shutdown.unwrap_or_default();
        let parts = BuildParts {
            registry,
            cache,
            cache_backend,
            shutdown,
            spend_tx: self.spend_tx,
            db: self.db,
            coordinator: self.coordinator,
        };
        let state = assemble(self.config, parts, ReusePolicy::Fresh)?;
        // Test-only override applied after assembly so the injected validator
        // replaces whatever `build_jwt_validator` derived from the config.
        // Shadowed (not mutated in place) so non-test builds see no unused
        // `mut`.
        #[cfg(test)]
        let state = if let Some(validator) = self.jwt_validator {
            let mut s = state;
            s.jwt_validator = Some(validator);
            s
        } else {
            state
        };
        Ok(state)
    }
}

impl AppState {
    /// Build application state from a loaded config.
    pub fn new(config: ProxyConfig) -> anyhow::Result<Self> {
        StateBuilder::new(config).build()
    }

    /// Build application state with an explicit provider registry (for testing).
    pub fn with_registry(config: ProxyConfig, registry: ProviderRegistry) -> anyhow::Result<Self> {
        StateBuilder::new(config).with_registry(registry).build()
    }

    /// Build application state with an explicit registry and cache (for testing).
    pub fn with_registry_and_cache(
        config: ProxyConfig,
        registry: ProviderRegistry,
        cache: Arc<dyn Cache>,
    ) -> anyhow::Result<Self> {
        StateBuilder::new(config)
            .with_registry(registry)
            .with_cache(cache)
            .build()
    }

    /// Build application state with an explicit registry, cache, and an optional
    /// batched-spend-writer sender.
    pub fn with_registry_cache_and_spend(
        config: ProxyConfig,
        registry: ProviderRegistry,
        cache: Arc<dyn Cache>,
        spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
    ) -> anyhow::Result<Self> {
        StateBuilder::new(config)
            .with_registry(registry)
            .with_cache(cache)
            .with_spend_tx(spend_tx)
            .build()
    }

    /// Build application state with an explicit registry, cache, spend-writer
    /// sender, and an optional database handle.
    pub fn with_registry_cache_db(
        config: ProxyConfig,
        registry: ProviderRegistry,
        cache: Arc<dyn Cache>,
        spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
        db: DbOpt,
    ) -> anyhow::Result<Self> {
        StateBuilder::new(config)
            .with_registry(registry)
            .with_cache(cache)
            .with_spend_tx(spend_tx)
            .with_db(db)
            .build()
    }

    /// Build application state with an explicit registry, cache, spend-writer
    /// sender, and an optional distributed coordination layer. Used by the
    /// production bootstrap path when no database is configured.
    pub fn with_registry_cache_and_coordinator(
        config: ProxyConfig,
        registry: ProviderRegistry,
        cache: Arc<dyn Cache>,
        spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
        coordinator: Option<Coordinator>,
    ) -> anyhow::Result<Self> {
        StateBuilder::new(config)
            .with_registry(registry)
            .with_cache(cache)
            .with_spend_tx(spend_tx)
            .with_coordinator(coordinator)
            .build()
    }

    /// Build application state with an explicit registry, cache, spend-writer
    /// sender, database handle, and an optional distributed coordination
    /// layer. Used by the production bootstrap path.
    pub fn with_registry_cache_db_and_coordinator(
        config: ProxyConfig,
        registry: ProviderRegistry,
        cache: Arc<dyn Cache>,
        spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
        db: DbOpt,
        coordinator: Option<Coordinator>,
    ) -> anyhow::Result<Self> {
        StateBuilder::new(config)
            .with_registry(registry)
            .with_cache(cache)
            .with_spend_tx(spend_tx)
            .with_db(db)
            .with_coordinator(coordinator)
            .build()
    }
}

/// The single construction path shared by startup and hot-reload.
///
/// Always rebuilt from `config`: deployments, master-key hash, router config,
/// trusted proxies / allowed IPs, hooks, and guardrails. Everything else is
/// either built fresh ([`ReusePolicy::Fresh`]) or preserved from the previous
/// state when the config section that produced it is unchanged
/// ([`ReusePolicy::Preserve`]), mirroring the documented preservation rules
/// on each [`AppState`] field.
pub(crate) fn assemble(
    config: ProxyConfig,
    parts: BuildParts,
    policy: ReusePolicy<'_>,
) -> anyhow::Result<AppState> {
    crate::auth::validate_auth_config(&config.general_settings)?;
    let old = match policy {
        ReusePolicy::Fresh => None,
        ReusePolicy::Preserve(prev) => Some(prev),
    };

    // Pre-warm the vendored price table here (a no-op on reloads: the table
    // is a `OnceLock`), so the first user-facing request after boot never
    // absorbs the ~12k-line TOML parse on the hot path.
    let _ = oxllm_core::cost::price_table();

    let deployments: Vec<Deployment> = config
        .model_list
        .iter()
        .map(|cm| cm.clone().into())
        .collect();

    // Pre-register every operator-controlled model label with the metrics
    // layer so runtime counters can only use known labels (aliases accepted
    // as request names included). Anything else falls back to the bounded
    // placeholder instead of growing the registry per request.
    crate::metrics::register_model_labels(deployments.iter().map(|d| d.model_name.as_str()));
    if let Some(aliases) = config
        .router_settings
        .model_group_alias
        .as_ref()
        .map(|aliases| aliases.keys())
    {
        crate::metrics::register_model_labels(aliases.map(String::as_str));
    }

    // Config-compat observability: oxllm honors ANY non-empty `model`
    // prefix as a provider slug (deliberate divergence from LiteLLM, which
    // only honors known providers and falls back to OpenAI), so a
    // HuggingFace-style id like `meta-llama/Llama-3.1-8B` produces slug
    // `meta-llama` and a request-time 400 "unsupported provider". Warn at
    // load time so the misconfiguration is visible before the first
    // request trips over it.
    for dep in &deployments {
        let slug = dep.provider_slug();
        if parts.registry.get(slug).is_none() {
            tracing::warn!(
                provider = %slug,
                model = %dep.oxllm_params.model,
                "model prefix is not a registered provider slug; requests for it will \
                 fail with 'unsupported provider' (LiteLLM would fall back to OpenAI)"
            );
        }
        if let Some(warning) = dep.credential_mismatch_warning() {
            tracing::warn!(
                model_name = %dep.model_name,
                model = %dep.oxllm_params.model,
                provider = %slug,
                "{warning}"
            );
        }
    }

    let master_key_hash = config
        .general_settings
        .master_key
        .as_ref()
        .map(|key| hash_key(key.as_bytes()));

    // Env-derived. Re-read on every build, but the process environment is
    // fixed at exec time (there is no dotenv re-load in the watcher), so
    // `ENCRYPTION_KEY` / `ENCRYPTION_KEY_PREVIOUS` changes (key rotation)
    // only take effect after a full process restart.
    let credential_cipher = crate::crypto::CredentialCipher::from_env();
    if old.is_none() && credential_cipher.is_some() {
        tracing::info!(
            max_plaintext_bytes = crate::crypto::MAX_PLAINTEXT_LEN,
            "credential encryption enabled (ENCRYPTION_KEY set)"
        );
    }

    let mut router_config = super::router_config::build_router_config(&config)?;
    // Sole-pin-path wiring: when prompt caching is enabled, the
    // `prompt_caching` check must be named in the effective
    // `optional_pre_call_checks` — the router runs only listed checks — so
    // the pre-call check pipeline handles pinning whether or not the
    // config lists the check. Runs before the router and the checks are
    // built so both observe the same effective list.
    crate::pre_call_checks::ensure_prompt_caching_listed(&config, &mut router_config);
    let mut router = match old {
        // Preserve runtime state (cooldowns, latency samples, in-flight
        // counters, usage tracking, health entries, bandit cells) across
        // config reloads by reusing the existing stores.
        Some(prev) => Router::rebuild_from_stores(
            router_config.clone(),
            prev.router.cooldown_arc(),
            prev.router.latency_store_arc(),
            prev.router.usage_store_arc(),
            prev.router.health_store_arc(),
            prev.router.bandit_store_arc(),
            prev.router.round_robin_counters_arc(),
        ),
        None => Router::new(router_config.clone()),
    };
    // Hot-reload sweep: drop runtime state (cooldowns, latency samples,
    // in-flight counts, health entries, learned bandit cells) for deployments
    // removed from the model list so the preserved stores do not leak across
    // config churn. Fresh builds start empty so the sweep is a no-op there.
    //
    // When a DB is wired (`parts.db` is `Some`), this sweep is deferred to
    // the caller: both the config-watcher's post-merge step
    // (`config::watcher::do_reload`) and `refresh_db_models` run their own
    // prune against the full merged deployment list (YAML + DB). Running it
    // here with the YAML-only `deployments` would drop every DB-stored
    // model's learned state (cooldowns, latency, usage, bandit cells) on
    // every reload, causing a temporary routing regression for
    // latency/usage/adaptive strategies after each config edit.
    if parts.db.is_none() {
        router.prune_stale_store_keys(&deployments);
    }
    // Pre-call checks are config-derived, so they are rebuilt on every
    // reload (like the router config itself).
    router.set_pre_call_checks(crate::pre_call_checks::build_pre_call_checks(
        &config,
        &router_config,
        parts.cache.clone(),
    ));
    let router = Arc::new(router);

    // The adaptive-router learning hook is registered only when the
    // adaptive strategy is configured — globally or for any routing group.
    // The hook gets a clone of the router's bandit store Arc, so hook
    // updates and strategy picks share the same cells (the strategies got
    // their clones at Router construction via StrategyDeps).
    let adaptive_bandit = router.adaptive_active().then(|| router.bandit_store_arc());

    let http_client = match old {
        Some(prev) if reload::http_client_config_unchanged(&prev.config, &config) => {
            prev.http_client.clone()
        }
        _ => build_http_client(&config)?,
    };

    // The image-download client always carries the SSRF resolver, so its
    // configuration is independent of `allow_private_api_base`. It only
    // depends on the pool/timeout settings covered by
    // `http_client_config_unchanged` (minus the `allow_private_api_base`
    // term, which is irrelevant here) — reuse the previous client when those
    // pool fields are unchanged so warm connections survive a reload, and
    // reuse it unconditionally when only `allow_private_api_base` flipped
    // (the resolver is always installed either way).
    let image_http_client = match old {
        Some(prev) if reload::http_pool_config_unchanged(&prev.config, &config) => {
            prev.image_http_client.clone()
        }
        _ => build_image_http_client(&config)?,
    };

    let jwt_validator = match old {
        Some(prev) if reload::jwt_secret_unchanged(&prev.config, &config) => {
            prev.jwt_validator.clone()
        }
        _ => build_jwt_validator(&config, &parts.cache, parts.cache_backend),
    };
    if config.general_settings.jwt_secret.is_some() && jwt_validator.is_none() {
        anyhow::bail!(
            "failed to construct configured JWT authentication; refusing to enter open mode"
        );
    }

    // Auth brake is enabled whenever any auth path is active: master key,
    // JWT, or a database (virtual keys). A DB-only deployment must still
    // rate-limit credential stuffing against virtual-key hashes. When
    // enabled, the existing brake `Arc` is preserved so in-flight brake
    // state is not dropped across hot-reloads.
    let auth_brake = if config.general_settings.master_key.is_some()
        || config.general_settings.jwt_secret.is_some()
        || parts.db.is_some()
    {
        match old.and_then(|prev| prev.auth_brake.clone()) {
            Some(existing) => Some(existing),
            None => Some(Arc::new(
                crate::security::AuthBrake::default_policy()
                    .with_coordinator(parts.coordinator.clone()),
            )),
        }
    } else {
        None
    };

    let trusted_proxies = parse_trusted_proxies(&config);

    let allowed_ips = parse_allowed_ips(&config)?;

    // C05/MEDIUM: `allowed_ips` without `trusted_proxy_cidrs` checks the peer
    // address, not the client (forwarded headers are only honored for trusted
    // peers). Behind a reverse proxy / LB that silently weakens the allowlist
    // to "whatever the peer is" — and if the operator adds the proxy's own
    // private range to `allowed_ips`, every client behind it passes. Warn once
    // at startup so the misconfiguration is visible.
    if allowed_ips.is_some() && trusted_proxies.is_empty() {
        tracing::warn!(
            "allowed_ips is configured without trusted_proxy_cidrs: behind a proxy/LB \
             the allowlist checks the peer address, not the client IP. Configure \
             trusted_proxy_cidrs so forwarded headers are honored for the real client."
        );
    }

    let alert_state = match old {
        Some(prev) if reload::alerting_config_unchanged(&prev.config, &config) => {
            prev.alert_state.clone()
        }
        _ => crate::hooks::alerting::build_alert_state(&config, parts.cache.clone()),
    };

    let slack_alert_state = match old {
        Some(prev) if reload::slack_alerting_config_unchanged(&prev.config, &config) => {
            prev.slack_alert_state.clone()
        }
        _ => crate::hooks::slack_alert::build_slack_alert_state(&config, parts.cache.clone()),
    };

    let pagerduty_alert_state = match old {
        Some(prev) if reload::pagerduty_config_unchanged(&prev.config, &config) => {
            prev.pagerduty_alert_state.clone()
        }
        _ => {
            crate::hooks::pagerduty_alert::build_pagerduty_alert_state(&config, parts.cache.clone())
        }
    };

    let email_alert_state = match old {
        Some(prev) if reload::email_alert_config_unchanged(&prev.config, &config) => {
            prev.email_alert_state.clone()
        }
        _ => crate::hooks::email_alert::build_email_alert_state(&config, parts.cache.clone()),
    };

    let alert_states = super::hook_builders::AlertStates {
        webhook: alert_state.clone(),
        slack: slack_alert_state.clone(),
        pagerduty: pagerduty_alert_state.clone(),
        email: email_alert_state.clone(),
    };

    let semantic_runtime = match old {
        Some(prev) if reload::semantic_cache_config_unchanged(&prev.config, &config) => {
            prev.semantic_cache_runtime.clone()
        }
        _ => crate::cache::semantic::build_semantic_runtime(&config, &http_client),
    };

    let langfuse_hook = match old {
        Some(prev) if reload::langfuse_config_unchanged(&prev.config, &config) => {
            prev.langfuse_hook.clone()
        }
        _ => crate::hooks::langfuse::build_langfuse_hook_opt(&config),
    };

    let mlflow_hook = match old {
        Some(prev) if reload::mlflow_config_unchanged(&prev.config, &config) => {
            prev.mlflow_hook.clone()
        }
        _ => crate::hooks::mlflow::build_mlflow_hook_opt(&config),
    };

    let hooks = super::hook_builders::build_hooks(
        &config,
        parts.cache.clone(),
        parts.spend_tx.clone(),
        alert_states,
        semantic_runtime.clone(),
        langfuse_hook.clone(),
        mlflow_hook.clone(),
        adaptive_bandit,
    );

    let credential_resolver = match old {
        Some(prev) => prev.credential_resolver.clone(),
        None => build_credential_resolver(),
    };

    let (
        pii_guardrail,
        injection_guardrail,
        presidio_guardrail,
        aim_guardrail,
        lakera_guardrail,
        bedrock_guardrail,
    ) = crate::guardrails::build_guardrails(&config)?;

    // The user-defined guardrail chain (custom regex/keyword/json_path/http +
    // semantic + MCP) and Model Armor are built once and cached on AppState
    // so chat requests do not recompile regexes or rebuild reqwest::Client per
    // call. Preserved across reloads when the guardrails config section is
    // unchanged (same pattern as the JWT validator / HTTP client preservation).
    let (custom_guardrail_chain, model_armor) = match old {
        Some(prev) if reload::guardrails_config_unchanged(&prev.config, &config) => (
            prev.custom_guardrail_chain.clone(),
            prev.model_armor.clone(),
        ),
        _ => crate::guardrails::build_user_guardrails(&config)?,
    };

    Ok(AppState {
        config: Arc::new(config),
        deployments,
        http_client,
        image_http_client,
        master_key_hash,
        credential_cipher,
        router,
        registry: parts.registry,
        cache: parts.cache,
        cache_backend: parts.cache_backend,
        shutdown: parts.shutdown,
        hooks,
        alert_state,
        slack_alert_state,
        pagerduty_alert_state,
        email_alert_state,
        semantic_cache_runtime: semantic_runtime,
        langfuse_hook,
        mlflow_hook,
        spend_tx: parts.spend_tx,
        db: parts.db,
        jwt_validator,
        auth_brake,
        trusted_proxies,
        allowed_ips,
        credential_resolver,
        pii_guardrail,
        injection_guardrail,
        presidio_guardrail,
        aim_guardrail,
        lakera_guardrail,
        bedrock_guardrail,
        custom_guardrail_chain,
        model_armor,
        coordinator: parts.coordinator,
    })
}

/// Build the JWT validator from `general_settings.jwt_*`, wiring the `jti`
/// replay cache to the shared cache when replay protection is enabled AND a
/// Redis backend is *actually active* (per-key `jti` tracking requires a
/// shared store; in-memory per-replica tracking would not stop cross-replica
/// replays).
///
/// Gating on [`CacheBackendKind`] (the backend that was *constructed*) rather
/// than [`redis_configured`] (which reads the config *string*) closes a
/// silent-degradation gap: when Redis is configured but unreachable at boot,
/// [`crate::app::build_cache`] falls back to `InMemoryCache` and reports
/// `cache_backend = Memory`. Using `redis_configured` here would wire the
/// per-process memory cache as the jti store and skip the warning, leaving
/// the operator to believe cross-replica replay protection was active when it
/// was not. By checking the actual backend, the warning fires and the jti
/// cache is left `None` until Redis is genuinely live (next boot).
fn build_jwt_validator(
    config: &ProxyConfig,
    cache: &Arc<dyn Cache>,
    cache_backend: CacheBackendKind,
) -> Option<Arc<crate::auth::jwt::JwtValidator>> {
    let jwt_replay = config
        .general_settings
        .jwt_replay_protection
        .unwrap_or(false);
    let jwt_jti_cache = if jwt_replay && cache_backend == CacheBackendKind::Redis {
        Some(cache.clone())
    } else {
        None
    };
    if jwt_replay && jwt_jti_cache.is_none() {
        tracing::warn!(
            "jwt_replay_protection is enabled but no Redis cache is active; \
             replay protection is inactive"
        );
    }
    crate::auth::jwt::build_validator(
        &config.general_settings.jwt_secret,
        &config.general_settings.jwt_algorithm,
        &config.general_settings.jwt_audience,
        &config.general_settings.jwt_issuer,
        jwt_jti_cache,
    )
}

/// Build the credential-resolver registry with the cloud resolvers
/// (Bedrock, Vertex AI) registered by slug; every other slug falls through
/// to the default static-key resolver.
fn build_credential_resolver() -> Arc<CredentialResolverRegistry> {
    let mut registry = CredentialResolverRegistry::new();
    registry.register(
        "bedrock",
        Arc::new(crate::credentials::BedrockCredentialResolver::new()),
    );
    registry.register(
        "bedrock_mantle",
        Arc::new(crate::credentials::BedrockMantleCredentialResolver::new()),
    );
    registry.register(
        "vertex_ai",
        Arc::new(crate::credentials::VertexResolver::new()),
    );
    Arc::new(registry)
}

/// Default maximum idle connections per host when
/// `http_pool_max_idle_per_host` is unset.
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 20;
/// Default idle-connection timeout (seconds) when `http_pool_idle_timeout_secs`
/// is unset.
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 90;
/// Default end-to-end timeout (seconds) for a single image download performed
/// by [`build_image_http_client`]. Bounds the headers fetch **and** the full
/// body stream so a slow-loris image host cannot hold the orchestration task
/// (and the caller's HTTP connection) open indefinitely; every other egress
/// builder in the workspace sets an equivalent total deadline.
const DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECS: u64 = 30;

/// Build the shared HTTP client with tuned connection-pool settings.
///
/// Pool sizing and timeouts are read from `general_settings`
/// (`http_pool_max_idle_per_host`, `http_pool_idle_timeout_secs`,
/// `http_connect_timeout_secs`), falling back to documented defaults when
/// unset.
///
/// Redirects are disabled (`Policy::none`) so the proxy cannot be coerced into
/// amplifying SSRF probes via 3xx chains. If a provider genuinely needs to
/// follow a redirect, handle it explicitly at the call site.
///
/// U19 (LiteLLM commit `fcaf673418`): the deployment's `num_retries` reached
/// the provider client, which applied its own retries on top of the router's
/// retry loop — double-counting. oxllm is safe: this builder sets **no retry
/// policy** on the `reqwest` client (no `.retry()` / retry middleware). The
/// only retry loop is the orchestration-level `try_model_with_retries` in
/// `orchestration/chat.rs`, which calls `execute_upstream_json` exactly once
/// per attempt. There is no transport-level retry to double-count.
pub(crate) fn build_http_client(config: &ProxyConfig) -> anyhow::Result<reqwest::Client> {
    let gs = &config.general_settings;
    let max_idle = gs
        .http_pool_max_idle_per_host
        .unwrap_or(DEFAULT_POOL_MAX_IDLE_PER_HOST);
    let idle_timeout = gs
        .http_pool_idle_timeout_secs
        .unwrap_or(DEFAULT_POOL_IDLE_TIMEOUT_SECS);
    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(max_idle)
        .pool_idle_timeout(Duration::from_secs(idle_timeout))
        .tcp_nodelay(true)
        .tcp_keepalive(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::none());
    // C05/HIGH: install an SSRF-aware DNS resolver on the shared client so the
    // private-range verdict and the actual connect share one DNS lookup,
    // closing the DNS-rebinding TOCTOU between `is_private_host_async` and the
    // fetch (the image-URL path takes user-supplied hosts). Gated on
    // `allow_private_api_base` to preserve the local-Ollama-on-loopback opt-in
    // (and to keep the e2e suite, which targets httpmock on 127.0.0.1, working).
    if !gs.allow_private_api_base.unwrap_or(false) {
        builder = builder.dns_resolver(std::sync::Arc::new(crate::security::SsrfResolver::new()));
    }
    if let Some(connect_secs) = gs.http_connect_timeout_secs {
        builder = builder.connect_timeout(Duration::from_secs(connect_secs));
    }
    builder.build().map_err(Into::into)
}

/// Build the dedicated HTTP client used to download user-supplied image
/// URLs during Bedrock image inlining
/// ([`crate::pre_call_checks::image_inlining::inline_image_urls`]).
///
/// Identical pool/timeout/redirect configuration to [`build_http_client`],
/// but the SSRF-aware DNS resolver
/// ([`crate::security::SsrfResolver`]) is installed **unconditionally** —
/// independent of `allow_private_api_base`. That opt-in governs
/// operator-chosen provider endpoints (e.g. a local Ollama instance on
/// loopback), which have a different trust level than user-supplied image
/// URLs that any authenticated caller can place in a chat request body.
/// Without this dedicated client, an `allow_private_api_base=true` deploy
/// would leave image fetches with only the best-effort, fail-open pre-check
/// ([`crate::security::is_private_host_async`]), reopening the
/// DNS-rebinding TOCTOU the connect-time resolver is meant to close.
///
/// In addition to the connect deadline, a **total request deadline**
/// ([`DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECS`]) is applied. Unlike the shared
/// upstream client (whose latency is bounded by the outer request timeout),
/// image downloads stream an arbitrary third-party body with no per-chunk
/// deadline; the total timeout ensures a drip-feed host cannot pin a task
/// for the whole request lifetime.
pub(crate) fn build_image_http_client(config: &ProxyConfig) -> anyhow::Result<reqwest::Client> {
    let gs = &config.general_settings;
    let max_idle = gs
        .http_pool_max_idle_per_host
        .unwrap_or(DEFAULT_POOL_MAX_IDLE_PER_HOST);
    let idle_timeout = gs
        .http_pool_idle_timeout_secs
        .unwrap_or(DEFAULT_POOL_IDLE_TIMEOUT_SECS);
    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(max_idle)
        .pool_idle_timeout(Duration::from_secs(idle_timeout))
        .tcp_nodelay(true)
        .tcp_keepalive(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECS))
        .dns_resolver(std::sync::Arc::new(crate::security::SsrfResolver::new()));
    if let Some(connect_secs) = gs.http_connect_timeout_secs {
        builder = builder.connect_timeout(Duration::from_secs(connect_secs));
    }
    builder.build().map_err(Into::into)
}

/// Parse `general_settings.trusted_proxy_cidrs` into a sorted list of
/// `IpNet`s. Invalid entries are skipped with a warning (never a hard error,
/// following the "unknown config keys are ignored with a warning" rule).
fn parse_trusted_proxies(config: &ProxyConfig) -> Arc<[ipnet::IpNet]> {
    let Some(raw) = &config.general_settings.trusted_proxy_cidrs else {
        return Arc::from([]);
    };
    let parsed: Vec<ipnet::IpNet> = raw
        .iter()
        .filter_map(|s| match s.parse::<ipnet::IpNet>() {
            Ok(net) => Some(net),
            Err(e) => {
                tracing::warn!(cidr = %s, error = %e, "ignoring unparseable trusted_proxy_cidr");
                None
            }
        })
        .collect();
    if parsed.len() != raw.len() {
        tracing::warn!(
            parsed = parsed.len(),
            total = raw.len(),
            "some trusted_proxy_cidrs were unparseable and ignored"
        );
    }
    Arc::from(parsed)
}

/// Parse `general_settings.allowed_ips` into an `Option<Arc<[IpNet]>>`.
///
/// Returns `Ok(None)` when no allowlist is configured so the middleware can
/// short-circuit and allow all IPs. Returns `Ok(Some(ips))` when one or more
/// entries parse. Invalid entries are skipped with a warning, per the
/// "unknown config keys are ignored with a warning" rule.
///
/// An explicitly **empty** list yields `Ok(Some(empty))` — deny all requests,
/// matching LiteLLM's `_check_valid_ip`, which compares against the list
/// directly so an empty allowlist matches no one. Collapsing `Some([])` into
/// `None` (allow-all) would turn an operator typo into an open relay.
///
/// Returns `Err` (fail-closed) when the allowlist is configured with entries
/// but every one of them is unparseable. The IP-allowlist middleware treats
/// `None` as "allow all IPs", so returning `None` here would make a
/// syntactically broken allowlist indistinguishable from an absent one —
/// turning the proxy into an open relay on a typo. Surfacing a hard error at
/// config load (startup and reload) forces the operator to fix the syntax
/// before traffic flows.
fn parse_allowed_ips(config: &ProxyConfig) -> anyhow::Result<Option<Arc<[ipnet::IpNet]>>> {
    let Some(raw) = config.general_settings.allowed_ips.as_ref() else {
        return Ok(None);
    };
    if raw.is_empty() {
        return Ok(Some(Arc::from(Vec::new())));
    }
    let parsed = crate::security::parse_cidr_list(raw);
    if parsed.len() != raw.len() {
        tracing::warn!(
            parsed = parsed.len(),
            total = raw.len(),
            "some allowed_ips entries were unparseable and ignored"
        );
    }
    if parsed.is_empty() {
        anyhow::bail!(
            "allowed_ips is configured but has ZERO valid CIDR entries; \
             refusing to start with an inert IP allowlist (all IPs would be \
             allowed). Fix the syntax or remove the setting."
        );
    }
    Ok(Some(Arc::from(parsed)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{OxllmSettings, RouterSettings};

    fn proxy_with_general(general: crate::config::types::GeneralSettings) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: general,
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn parse_trusted_proxies_none_yields_empty() {
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            trusted_proxy_cidrs: None,
            ..Default::default()
        });
        assert!(parse_trusted_proxies(&config).is_empty());
    }

    #[test]
    fn parse_trusted_proxies_valid_cidrs() {
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            trusted_proxy_cidrs: Some(vec!["10.0.0.0/8".to_string(), "192.168.0.0/16".to_string()]),
            ..Default::default()
        });
        let parsed = parse_trusted_proxies(&config);
        assert_eq!(parsed.len(), 2);
        assert!(parsed.iter().any(|c| c.to_string() == "10.0.0.0/8"));
        assert!(parsed.iter().any(|c| c.to_string() == "192.168.0.0/16"));
    }

    #[test]
    fn parse_trusted_proxies_skips_invalid() {
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            trusted_proxy_cidrs: Some(vec!["10.0.0.0/8".to_string(), "not-a-cidr".to_string()]),
            ..Default::default()
        });
        let parsed = parse_trusted_proxies(&config);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].to_string(), "10.0.0.0/8");
    }

    #[test]
    fn http_pool_defaults_when_unset() {
        // Issue #17: unset pool config falls back to the documented defaults
        // and still produces a usable client.
        assert_eq!(DEFAULT_POOL_MAX_IDLE_PER_HOST, 20);
        assert_eq!(DEFAULT_POOL_IDLE_TIMEOUT_SECS, 90);
        let config = proxy_with_general(crate::config::types::GeneralSettings::default());
        build_http_client(&config).expect("client builds with defaults");
    }

    #[test]
    fn http_pool_reads_custom_idle_per_host() {
        // Issue #17: `http_pool_max_idle_per_host` overrides the default and
        // builds a client without error.
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            http_pool_max_idle_per_host: Some(5),
            ..Default::default()
        });
        build_http_client(&config).expect("client builds with custom idle");
    }

    #[test]
    fn http_pool_reads_custom_idle_timeout() {
        // Issue #17: `http_pool_idle_timeout_secs` is accepted without error.
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            http_pool_idle_timeout_secs: Some(30),
            ..Default::default()
        });
        build_http_client(&config).expect("client builds with custom idle timeout");
    }

    #[test]
    fn http_pool_reads_connect_timeout() {
        // Issue #17: `http_connect_timeout_secs` is applied without error.
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            http_connect_timeout_secs: Some(10),
            ..Default::default()
        });
        build_http_client(&config).expect("client builds with connect timeout");
    }

    #[test]
    fn parse_allowed_ips_none_when_unset() {
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            allowed_ips: None,
            ..Default::default()
        });
        assert!(parse_allowed_ips(&config).expect("ok").is_none());
    }

    #[test]
    fn unusable_authentication_rejects_validation_startup_and_reload() {
        use crate::config::types::GeneralSettings;
        let state = AppState::new(proxy_with_general(GeneralSettings {
            master_key: Some("sk-existing-valid-master".into()),
            ..Default::default()
        }))
        .expect("valid state");
        for settings in [
            GeneralSettings {
                master_key: Some("   ".into()),
                ..Default::default()
            },
            GeneralSettings {
                jwt_secret: Some(" ".repeat(32)),
                ..Default::default()
            },
            GeneralSettings {
                jwt_secret: Some("short".into()),
                ..Default::default()
            },
            GeneralSettings {
                jwt_secret: Some("valid-secret-at-least-16".into()),
                jwt_algorithm: Some("RS256".into()),
                ..Default::default()
            },
            GeneralSettings {
                jwt_secret: Some("valid-secret-at-least-16".into()),
                jwt_algorithm: Some("typo".into()),
                ..Default::default()
            },
        ] {
            let config = proxy_with_general(settings);
            assert!(crate::config::loader::validate_proxy_config(&config).is_err());
            assert!(crate::app::enforce_require_auth_guard(&config).is_err());
            assert!(AppState::new(config.clone()).is_err());
            assert!(state.reload_from_config(config).is_err());
            assert!(state.master_key_hash.is_some());
        }
    }

    #[test]
    fn parse_allowed_ips_empty_list_yields_deny_all() {
        // An explicitly empty allowlist must NOT collapse to `None`
        // (allow-all): LiteLLM's `_check_valid_ip` compares against the list
        // directly, so an empty list denies everyone — fail-closed.
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            allowed_ips: Some(vec![]),
            ..Default::default()
        });
        let parsed = parse_allowed_ips(&config).expect("empty list must not fail config load");
        let ips = parsed.expect("empty list must stay Some (deny-all), not None (allow-all)");
        assert!(ips.is_empty());
    }

    #[test]
    fn parse_allowed_ips_valid_entries() {
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            allowed_ips: Some(vec!["10.0.0.0/8".to_string(), "192.168.0.0/16".to_string()]),
            ..Default::default()
        });
        let parsed = parse_allowed_ips(&config)
            .expect("ok")
            .expect("some entries");
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn parse_allowed_ips_partial_invalid_keeps_valid() {
        // Some good + some bad: keep the good ones (do not fail), matching the
        // "skip invalid entries with a warning" rule.
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            allowed_ips: Some(vec!["10.0.0.0/8".to_string(), "not-a-cidr".to_string()]),
            ..Default::default()
        });
        let parsed = parse_allowed_ips(&config)
            .expect("ok")
            .expect("some entries");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].to_string(), "10.0.0.0/8");
    }

    #[test]
    fn parse_allowed_ips_all_invalid_fails_closed() {
        // Fail-closed: a configured allowlist with ZERO valid CIDRs must NOT
        // return None (which the middleware treats as "allow all IPs"). It
        // must surface a hard error so a typo does not silently open the proxy.
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            allowed_ips: Some(vec!["not-a-cidr".to_string()]),
            ..Default::default()
        });
        let err = parse_allowed_ips(&config).expect_err("all-invalid must fail closed");
        assert!(
            err.to_string().contains("allowed_ips"),
            "error must name allowed_ips: {err}"
        );
    }

    #[test]
    fn assemble_rejects_all_invalid_allowed_ips() {
        // The fail-closed error must propagate through `assemble` so neither
        // startup nor a hot-reload can go live with an inert IP allowlist.
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings {
                allowed_ips: Some(vec!["totally-bogus".to_string()]),
                ..Default::default()
            },
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        let err = AppState::new(config)
            .err()
            .expect("must reject inert allowlist");
        assert!(err.to_string().contains("allowed_ips"));
    }

    /// C05/HIGH: a distributed coordinator passed through the production
    /// bootstrap path must be attached to the state AND to the auth brake, and
    /// a block recorded on one replica's brake must be visible on a second
    /// replica's brake (the sharing behavior that was previously only
    /// exercised at unit level).
    #[tokio::test]
    async fn assemble_attaches_distributed_coordinator_to_auth_brake() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let coord = Coordinator::new(cache.clone()).with_distributed(true);
        let config = proxy_with_general(crate::config::types::GeneralSettings {
            master_key: Some("sk-master-key-ok!!".to_string()),
            ..Default::default()
        });
        let registry = oxllm_providers::default_registry();
        let state_a = AppState::with_registry_cache_and_coordinator(
            config.clone(),
            registry,
            cache.clone(),
            None,
            Some(coord.clone()),
        )
        .expect("state a");
        let state_b = AppState::with_registry_cache_and_coordinator(
            config,
            oxllm_providers::default_registry(),
            cache.clone(),
            None,
            Some(coord),
        )
        .expect("state b");

        assert!(
            state_a
                .coordinator
                .as_ref()
                .expect("coordinator")
                .is_distributed(),
            "production bootstrap must carry a distributed coordinator"
        );
        assert!(
            state_b
                .coordinator
                .as_ref()
                .expect("coordinator")
                .is_distributed()
        );

        let brake_a = state_a.auth_brake.as_ref().expect("auth brake");
        let brake_b = state_b.auth_brake.as_ref().expect("auth brake");
        assert!(
            brake_a.is_blocked("5.5.5.5").await == brake_b.is_blocked("5.5.5.5").await,
            "both replicas must observe the same block state"
        );
        // default_policy: 10 fails / 60s → block 300s.
        for _ in 0..10 {
            brake_a.record_failure("5.5.5.5").await;
        }
        assert!(
            brake_a.is_blocked("5.5.5.5").await,
            "replica A must block after the threshold"
        );
        assert!(
            brake_b.is_blocked("5.5.5.5").await,
            "a block recorded on replica A must be visible on replica B (fleet-wide)"
        );
    }

    // ---- MEDIUM: build_jwt_validator gates on actual cache backend -----------

    /// Helper: a config with `jwt_secret` + `jwt_replay_protection` and a
    /// `cache.type: redis` string (config-intent says Redis).
    fn config_with_jwt_replay_and_redis_config() -> ProxyConfig {
        use crate::config::types::{CacheConfig, GeneralSettings};
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings {
                cache: Some(CacheConfig {
                    cache_type: Some("redis".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            general_settings: GeneralSettings {
                jwt_secret: Some("s3cr3t-long-enough!".to_string()),
                jwt_replay_protection: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        }
    }

    /// When Redis is *configured* but the actual backend is `Memory`
    /// (Redis was unreachable at boot → silent fallback), the jti replay
    /// cache must NOT be wired: per-process memory tracking would not stop
    /// cross-replica replays. Previously `redis_configured(config)` returned
    /// `true` and the memory cache was silently used, masking the gap.
    #[test]
    fn build_jwt_validator_no_jti_cache_when_backend_memory_despite_redis_config() {
        let config = config_with_jwt_replay_and_redis_config();
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let validator = build_jwt_validator(&config, &cache, CacheBackendKind::Memory)
            .expect("validator built (jwt_secret set)");
        assert!(
            !validator.has_jti_cache(),
            "jti cache must NOT be wired when actual backend is Memory \
             even though config says redis — per-process tracking cannot stop \
             cross-replica replays"
        );
    }

    /// When the actual backend IS Redis (config matched reality), the jti
    /// replay cache is wired and cross-replica replay protection is active.
    #[test]
    fn build_jwt_validator_jti_cache_wired_when_backend_redis() {
        let config = config_with_jwt_replay_and_redis_config();
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let validator = build_jwt_validator(&config, &cache, CacheBackendKind::Redis)
            .expect("validator built (jwt_secret set)");
        assert!(
            validator.has_jti_cache(),
            "jti cache must be wired when actual backend is Redis"
        );
    }

    /// When `jwt_replay_protection` is disabled, no jti cache is wired
    /// regardless of backend.
    #[test]
    fn build_jwt_validator_no_jti_cache_when_replay_protection_off() {
        let mut config = config_with_jwt_replay_and_redis_config();
        config.general_settings.jwt_replay_protection = Some(false);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let validator = build_jwt_validator(&config, &cache, CacheBackendKind::Redis)
            .expect("validator built (jwt_secret set)");
        assert!(
            !validator.has_jti_cache(),
            "jti cache must not be wired when replay protection is disabled"
        );
    }

    /// LOW: the master-key length guard measures `String::len()` (bytes), not
    /// `chars().count()` (scalar values). The error message must say "bytes"
    /// so a multibyte key (e.g. CJK) is not mislead into thinking it has
    /// `MIN_MASTER_KEY_LEN` *characters* of entropy when it actually has that
    /// many *bytes* (a stricter bar for non-ASCII). This test verifies both
    /// the message wording and that a short multibyte key is rejected.
    #[test]
    fn master_key_too_short_error_reports_bytes_not_chars() {
        // 5 CJK chars = 15 UTF-8 bytes < MIN_MASTER_KEY_LEN (16). Must be
        // rejected, and the error must say "bytes" (not "chars"/"characters").
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("密钥密钥密".to_string()),
                ..Default::default()
            },
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        let err = AppState::new(config)
            .err()
            .expect("15-byte multibyte key must be rejected (under 16-byte floor)");
        let msg = err.to_string();
        assert!(
            msg.contains("bytes"),
            "error must say 'bytes' (the measured unit), got: {msg}"
        );
        assert!(
            !msg.contains("characters"),
            "error must not say 'characters' (misleading for multibyte keys), got: {msg}"
        );
    }

    /// Post-audit fix: `assemble` pre-registers every configured deployment
    /// name and model-group alias in the cardinality-bounded metrics
    /// registry, so they keep their identity even after the registry
    /// saturates. Membership is asserted through the non-admitting test probe
    /// ([`crate::metrics::model_label_registered`]) rather than `model_label`,
    /// which would admit the value on first lookup even if the build wiring
    /// were removed. `serial_test` because the registry is process-global.
    #[test]
    #[serial_test::serial]
    fn assemble_pre_registers_configured_models_and_aliases_as_metric_labels() {
        let configured_model = "assemble-configured-model-probe";
        let configured_alias = "assemble-configured-alias-probe";
        assert!(
            !crate::metrics::model_label_registered(configured_model)
                && !crate::metrics::model_label_registered(configured_alias),
            "precondition: probe labels must not be registered before the build"
        );
        let config = ProxyConfig {
            model_list: vec![crate::config::types::ConfigModel {
                model_name: configured_model.to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings {
                model_group_alias: Some(
                    std::iter::once((
                        configured_alias.to_string(),
                        vec![configured_model.to_string()],
                    ))
                    .collect(),
                ),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let _state = AppState::new(config).expect("state builds");
        assert!(
            crate::metrics::model_label_registered(configured_model),
            "configured model name must be pre-registered by state build"
        );
        assert!(
            crate::metrics::model_label_registered(configured_alias),
            "configured model-group alias must be pre-registered by state build"
        );
    }
}
