//! Shared application state.
//!
//! This module owns [`AppState`] — the struct every handler reads — plus the
//! machinery that builds and hot-reloads it. The implementation is split
//! into focused submodules:
//!
//! - `build` — the canonical construction path: [`StateBuilder`] (with
//!   optional setters for each injectable dependency) and the single
//!   `assemble` function both startup and reload funnel through.
//! - `reload` — `AppState::reload_from_config` plus the `*_config_unchanged`
//!   helpers that decide which infrastructure `Arc`s survive a hot-reload.
//! - `router_config` — `RouterConfig` assembly from `ProxyConfig`
//!   (strategy parsing, fallback rules, routing-group overrides).
//! - `hook_builders` — the hook-chain assembly (`build_hooks`).
//! - `db_models` — DB-stored model merging (`refresh_db_models`), gated on
//!   the `db` feature.
//!
//! Feature-specific constructors live next to the feature they build, not
//! here: guardrails are built in [`crate::guardrails`], the semantic-cache
//! runtime in [`crate::cache::semantic`], the langfuse/alerting hooks in
//! [`crate::hooks::langfuse`] / [`crate::hooks::alerting`]. This module only
//! *calls* them.
//!
//! Design note: `AppState` deliberately keeps its fields flat. Handler and
//! middleware code across the crate reads `state.cache`, `state.router`,
//! `state.deployments`, etc. directly; that access pattern is part of the
//! crate's de-facto API and is preserved on purpose. The cohesive groupings
//! (auth material, guardrails, infrastructure) are expressed at
//! *construction* time in `build::assemble` instead of as nested
//! sub-structs, so a new feature adds a field + one construction site rather
//! than a new constructor wrapper.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use arc_swap::ArcSwap;
use oxllm_core::Cache;
use oxllm_core::Deployment;
use oxllm_core::Hook;
use oxllm_core::ProviderRegistry;
use oxllm_core::router::Router;

use crate::config::ProxyConfig;
use crate::coordination::Coordinator;
use crate::credentials::CredentialResolverRegistry;
use crate::spend_writer::SpendEntry;

mod build;
#[cfg(feature = "db")]
mod db_models;
#[cfg(feature = "db")]
pub(crate) use db_models::db_record_to_deployment;
mod hook_builders;
mod reload;
mod router_config;

pub use build::StateBuilder;
#[cfg(test)]
pub(crate) use build::build_image_http_client;
#[cfg(feature = "db")]
pub(crate) use db_models::merge_db_deployments;
#[cfg(feature = "db")]
pub use db_models::refresh_db_models;

/// When the `db` feature is on, `AppState` may hold an optional database
/// handle. When off, the field is a harmless `Option<()>` that is always
/// `None` — this keeps `AppState` construction uniform across feature configs.
#[cfg(feature = "db")]
pub type DbOpt = Option<std::sync::Arc<oxllm_db::Database>>;
#[cfg(not(feature = "db"))]
pub type DbOpt = Option<()>;

/// The cache backend that was *actually constructed* at startup, independent
/// of what the config string requested. The readiness probe consults this
/// rather than the config string so that a silent fallback (e.g. Redis
/// unreachable at boot → `InMemoryCache`) is reported as a degraded backend,
/// not masked as a healthy `"redis": "ok"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBackendKind {
    Memory,
    Redis,
    S3,
    Gcs,
    Azure,
}

impl CacheBackendKind {
    /// Derive the backend kind from the config's `cache.type` string. Used as
    /// the fallback when no explicit kind was supplied (e.g. test paths that
    /// inject a custom cache without going through [`crate::app::build_cache`]).
    pub(crate) fn from_config(config: &ProxyConfig) -> Self {
        match config
            .oxllm_settings
            .cache
            .as_ref()
            .and_then(|c| c.cache_type.as_deref())
        {
            Some("redis") => Self::Redis,
            Some("s3") => Self::S3,
            Some("gcs") => Self::Gcs,
            Some("azure") => Self::Azure,
            _ => Self::Memory,
        }
    }

    /// Returns `true` for remote/shared backends whose `ping()` is a real
    /// reachability check (not the trait-default no-op).
    pub(crate) fn is_remote(self) -> bool {
        !matches!(self, Self::Memory)
    }

    /// The readiness-probe check label for this backend (`"redis"`, `"s3"`,
    /// ...), or `None` for the in-memory fallback. The readiness handler uses
    /// this so the per-check key reflects the backend actually being pinged
    /// rather than a stale config string after a hot-reload.
    pub(crate) fn label(self) -> Option<&'static str> {
        match self {
            Self::Redis => Some("redis"),
            Self::S3 => Some("s3"),
            Self::Gcs => Some("gcs"),
            Self::Azure => Some("azure"),
            Self::Memory => None,
        }
    }
}

/// A shared, atomically-swappable application state.
///
/// Handlers read `.load_full()` to get the current `Arc<AppState>`. The config
/// watcher calls `.store()` to swap in a new state on hot-reload.
pub type SharedState = Arc<ArcSwap<AppState>>;

/// Shared state accessible to all handlers.
#[derive(Clone)]
pub struct AppState {
    /// The loaded proxy config.
    pub config: Arc<ProxyConfig>,
    /// Pre-converted deployments for quick lookup.
    pub deployments: Vec<Deployment>,
    /// Reused HTTP client (never create per-request).
    pub http_client: reqwest::Client,
    /// Dedicated HTTP client for downloading user-supplied image URLs
    /// during Bedrock image inlining. Always carries the SSRF-aware DNS
    /// resolver ([`crate::security::SsrfResolver`]) regardless of
    /// `allow_private_api_base`: that opt-in governs operator-chosen
    /// provider endpoints, not untrusted image URLs that arrive in request
    /// bodies. Keeping a separate client closes the DNS-rebinding TOCTOU at
    /// the connection layer for image fetches even when the shared
    /// [`AppState::http_client`] has been relaxed for local-Ollama-style
    /// loopback deployments.
    pub image_http_client: reqwest::Client,
    /// SHA-256 of the configured master key, if any.
    pub master_key_hash: Option<[u8; 32]>,
    /// Optional AES-256-GCM cipher for provider API keys stored in the DB
    /// (`models.oxllm_params.api_key`). Built once from `ENCRYPTION_KEY`;
    /// `None` when the env var is unset (plaintext storage, backward
    /// compatible). Env-derived, so preserved across config hot-reloads.
    pub credential_cipher: Option<crate::crypto::CredentialCipher>,
    /// The router for deployment selection.
    pub router: Arc<Router>,
    /// The provider registry: slug → provider impl (chat/embeddings).
    pub registry: Arc<ProviderRegistry>,
    /// The cache (response cache + counters). May be `InMemoryCache` or
    /// `DualCache` (in-memory + Redis) depending on config.
    pub cache: Arc<dyn Cache>,
    /// The backend kind that was *actually constructed* (see
    /// [`CacheBackendKind`]). Diverges from the config string when a remote
    /// backend was requested but fell back to in-memory at startup.
    pub cache_backend: CacheBackendKind,
    /// Process-wide graceful-shutdown flag, flipped to `true` early in the
    /// drain path of [`crate::app::run`]. The liveness/readiness probes
    /// consult it (via [`AppState::is_shutting_down`]) and return
    /// `503 {"status":"shutting_down"}` once set, mirroring the reference
    /// `GracefulShutdownManager.is_shutting_down()` semantics so k8s probes
    /// get a clean 503 (instead of a probe timeout) and stop routing new
    /// traffic into a draining pod. Preserved across hot-reloads: a config
    /// swap never resets the shutdown state.
    pub shutdown: Arc<AtomicBool>,
    /// The hook chain (rate-limit, budget, cache, cost-tracking, ...).
    pub hooks: Vec<Arc<dyn Hook>>,
    /// Shared budget-alert state, wired into both the [`AlertingHook`] and
    /// the [`BudgetResetScheduler`](crate::scheduler::BudgetResetScheduler) so
    /// stale alerts are cleared when a budget window resets. `None` when
    /// alerting is not configured.
    ///
    /// [`AlertingHook`]: crate::hooks::AlertingHook
    pub alert_state: Option<Arc<crate::hooks::AlertState>>,
    /// Slack hook alert state, preserved across hot-reloads when the slack
    /// config is unchanged so in-memory fired markers are not discarded.
    pub slack_alert_state: Option<Arc<crate::hooks::AlertState>>,
    /// PagerDuty hook alert state, preserved across hot-reloads when the
    /// pagerduty config is unchanged.
    pub pagerduty_alert_state: Option<Arc<crate::hooks::AlertState>>,
    /// Email hook alert state, preserved across hot-reloads when the email
    /// alert config is unchanged.
    pub email_alert_state: Option<Arc<crate::hooks::AlertState>>,
    /// The semantic-cache runtime, preserved across config reloads when the
    /// semantic cache config is unchanged so accumulated embeddings are not
    /// discarded. `None` when semantic caching is disabled.
    pub semantic_cache_runtime: Option<Arc<crate::hooks::cache_hook::SemanticCacheRuntime>>,
    /// The langfuse hook Arc, preserved across config reloads when the
    /// langfuse config is unchanged so the spawned flush task is not leaked
    /// on every reload.
    pub langfuse_hook: Option<Arc<crate::hooks::LangfuseHook>>,
    /// The mlflow hook Arc, preserved across config reloads when the mlflow
    /// config is unchanged so the lazily-resolved run_id is not lost on
    /// unrelated reloads.
    pub mlflow_hook: Option<Arc<crate::hooks::MlflowHook>>,
    /// Sender for the batched spend writer. `None` when no database is
    /// configured. Held so the writer's background task stays alive for the
    /// lifetime of this state.
    pub spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
    /// Optional database handle (virtual-key persistence + spend logs). Only
    /// present when the `db` feature is enabled and a `database_url` was
    /// configured at startup.
    pub db: DbOpt,
    /// JWT validator (built from `general_settings.jwt_secret`). `None` when
    /// JWT auth is not configured.
    pub jwt_validator: Option<Arc<crate::auth::jwt::JwtValidator>>,
    /// Per-IP auth-failure brake. `None` when auth brute-force protection is
    /// disabled (e.g. no master key configured).
    pub auth_brake: Option<Arc<crate::security::AuthBrake>>,
    /// Redis-optional coordination layer. When distributed (Redis-backed),
    /// shared state (e.g. the auth-brake counters) is coordinated across
    /// replicas; otherwise behavior is per-process. `None` when no cache was
    /// supplied at construction time.
    pub coordinator: Option<Coordinator>,
    /// Parsed trusted-proxy CIDR ranges. Empty when no proxies are trusted
    /// in that case `X-Forwarded-For` / `X-Real-IP` headers are never honored
    /// and the connection peer IP is used directly.
    pub trusted_proxies: Arc<[ipnet::IpNet]>,
    /// Parsed CIDR allowlist for `general_settings.allowed_ips`. `None` when
    /// the allowlist is not configured (all IPs allowed). `Some([])` is
    /// treated the same as `None` by the middleware.
    pub allowed_ips: Option<Arc<[ipnet::IpNet]>>,
    /// The credential resolver registry used to materialize
    /// [`oxllm_core::provider::ResolvedCredentials`] for each deployment before invoking provider
    /// transforms. Each provider slug maps to a dedicated
    /// [`CredentialResolver`](crate::credentials::CredentialResolver); the
    /// default ([`StaticKeyResolver`](crate::credentials::StaticKeyResolver))
    /// handles every slug without an explicit registration. Cloud resolvers
    /// (Bedrock/Vertex) register by slug in later phases.
    pub credential_resolver: Arc<CredentialResolverRegistry>,
    /// Pre-built PII masking guardrail, built once from
    /// `general_settings.guardrails` at state construction. `None` when PII
    /// masking is disabled. Cached here so chat requests do not rebuild the
    /// regex table per call.
    pub pii_guardrail: Option<crate::guardrails::PiiGuardrail>,
    /// Pre-built prompt-injection guardrail. `None` when injection detection
    /// is disabled. Cached for the same reason as [`Self::pii_guardrail`].
    pub injection_guardrail: Option<crate::guardrails::InjectionGuardrail>,
    /// Pre-built Presidio PII guardrail. `None` when Presidio is not
    /// configured.
    pub presidio_guardrail: Option<crate::guardrails::PresidioGuardrail>,
    /// Pre-built AIM Security guardrail. `None` when AIM is not configured.
    pub aim_guardrail: Option<crate::guardrails::AimGuardrail>,
    /// Pre-built Lakera AI guardrail. `None` when Lakera is not configured.
    pub lakera_guardrail: Option<crate::guardrails::LakeraGuardrail>,
    /// Pre-built Bedrock guardrail. `None` when Bedrock guardrails are not
    /// configured.
    pub bedrock_guardrail: Option<crate::guardrails::BedrockGuardrail>,
    /// Pre-built user-defined guardrail chain (custom regex/keyword/json_path/
    /// http + semantic + MCP), built once from `general_settings.guardrails`
    /// at state construction so chat requests do not recompile regex patterns
    /// or rebuild HTTP clients per call. `None` when no user-defined guardrails
    /// are configured.
    pub custom_guardrail_chain: Option<crate::guardrails::CustomGuardrailChain>,
    /// Pre-built Model Armor guardrail, built once at state construction so
    /// chat requests do not rebuild a `reqwest::Client` per call. `None` when
    /// Model Armor is not configured or disabled.
    pub model_armor: Option<crate::guardrails::ModelArmorGuardrail>,
}

impl AppState {
    /// Returns `true` once graceful shutdown has begun (the drain signal fired
    /// in [`crate::app::run`]). The health/readiness probes consult this so
    /// orchestrators observe a clean 503 instead of a probe timeout during
    /// rolling deploys.
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Returns a clone of this state with `deployments` replaced. Used by
    /// [`refresh_db_models`] to swap in YAML + DB merged deployments without
    /// rebuilding derived fields (router, hooks, credentials, ...).
    #[cfg(feature = "db")]
    fn with_deployments(&self, deployments: Vec<Deployment>) -> Self {
        let mut next = self.clone();
        next.deployments = deployments;
        next
    }
}

/// Returns true when the config's cache section selects the Redis backend
/// (`oxllm_settings.cache.type == "redis"`).
pub(crate) fn redis_configured(config: &ProxyConfig) -> bool {
    config
        .oxllm_settings
        .cache
        .as_ref()
        .and_then(|c| c.cache_type.as_deref())
        == Some("redis")
}
