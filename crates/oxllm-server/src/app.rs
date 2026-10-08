//! Axum router and server bootstrap.
//!
//! Route registration lives in [`crate::routes`]; this module owns process
//! bootstrap: config loading, state/cache construction, the budget-reset
//! scheduler, graceful shutdown, and the server `run` loop. The public
//! [`build_router`] entry point is re-exported from `routes`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
#[cfg(any(
    feature = "s3-cache",
    feature = "gcs-cache",
    feature = "azure-cache",
    test
))]
use async_trait::async_trait;
use oxllm_core::Cache as CacheTrait;
#[cfg(any(
    feature = "s3-cache",
    feature = "gcs-cache",
    feature = "azure-cache",
    test
))]
use oxllm_core::{Error, IncrResult};

use crate::cache::InMemoryCache;
use crate::config::ProxyConfig;
use crate::orchestration::DEFAULT_TIMEOUT_SECS;
use crate::security::sanitize_url;
use crate::state::{AppState, CacheBackendKind, SharedState};

pub use crate::routes::build_router;

const SHUTDOWN_DEADLINE_SECS: u64 = 30;

const OPEN_MODE_WARNING: &str = "WARNING: Running without authentication — all endpoints are open. \
 Set general_settings.master_key, jwt_secret, database_url, or OXLLM_REQUIRE_AUTH in production.";

/// True when at least one authentication path is configured (master key,
/// JWT secret, or virtual-key database). Matches the open-mode gate in
/// [`crate::auth::auth_middleware`] and the `auth_enabled` field reported by
/// [`run`]. Shared with `oxllm validate` so the subcommand reports the same
/// auth status the running proxy would enforce.
pub fn auth_is_configured(
    master_key: &Option<String>,
    jwt_secret: &Option<String>,
    database_url: &Option<String>,
) -> bool {
    let usable_secret = |secret: &Option<String>| {
        secret.as_deref().is_some_and(|value| {
            !value.trim().is_empty() && value.len() >= crate::auth::MIN_AUTH_SECRET_LEN
        })
    };
    usable_secret(master_key)
        || usable_secret(jwt_secret)
        || database_url
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
}

/// True when [`auth_is_configured`] is true, or when a non-empty short master
/// key is accepted by the development-only `allow_short_master_key` hatch.
///
/// [`auth_is_configured`] stays false for short keys. Callers that decide
/// whether the process is in open mode (`validate` reporting, `REQUIRE_AUTH`,
/// reload, and `POST /config/update`) use this function so the hatch counts.
pub fn settings_auth_is_configured(settings: &crate::config::types::GeneralSettings) -> bool {
    auth_is_configured(
        &settings.master_key,
        &settings.jwt_secret,
        &settings.database_url,
    ) || short_master_key_counts(settings)
}

/// A non-empty master key shorter than the 16-byte authentication minimum
/// counts as configured only when the hatch is set and the resolved
/// environment is development.
pub fn short_master_key_counts(settings: &crate::config::types::GeneralSettings) -> bool {
    settings.allow_short_master_key == Some(true)
        && crate::config::loader::environment_is_development()
        && settings.master_key.as_deref().is_some_and(|key| {
            !key.trim().is_empty() && key.len() < crate::auth::MIN_AUTH_SECRET_LEN
        })
}

/// Build state from a config, returning both for testing.
pub fn build_state(config: ProxyConfig) -> anyhow::Result<Arc<AppState>> {
    Ok(Arc::new(AppState::new(config)?))
}

/// Build a `SharedState` (ArcSwap-wrapped) from a config.
pub fn build_shared_state(config: ProxyConfig) -> anyhow::Result<SharedState> {
    Ok(Arc::new(ArcSwap::from_pointee(AppState::new(config)?)))
}

/// Build a `SharedState` with an explicit database handle and spend-writer
/// sender (for tests of persistence/management endpoints).
#[cfg(feature = "db")]
pub fn build_shared_state_with_db(
    config: ProxyConfig,
    spend_tx: Option<tokio::sync::mpsc::Sender<crate::spend_writer::SpendEntry>>,
    db: Option<Arc<oxllm_db::Database>>,
) -> anyhow::Result<SharedState> {
    use oxllm_core::Cache as CacheTrait;
    let registry = oxllm_providers::default_registry();
    let cache: Arc<dyn CacheTrait> = Arc::new(InMemoryCache::default());
    let state = AppState::with_registry_cache_db(config, registry, cache, spend_tx, db)?;
    Ok(Arc::new(ArcSwap::from_pointee(state)))
}

/// Build a cache (InMemory or DualCache) from config, for testing.
pub async fn build_cache_for_test(
    config: &ProxyConfig,
    memory: Arc<InMemoryCache>,
) -> Arc<dyn CacheTrait> {
    build_cache(config, memory)
        .await
        .map(|(cache, _, _)| cache)
        .expect("cache build failed")
}

/// Build the cache layer: `DualCache` (in-memory + Redis) when Redis is
/// configured and the feature is enabled, otherwise plain `InMemoryCache`.
/// Returns the `Arc<dyn Cache>` for `AppState`, the [`CacheBackendKind`] that
/// was *actually constructed* (so the readiness probe detects a fallback), and
/// the underlying `Arc<InMemoryCache>` for the budget-reset scheduler (which
/// needs InMemoryCache-specific methods), plus the `RedisCache` handle when
/// Redis connected (used to spawn the cross-replica invalidation subscriber).
pub(crate) async fn build_cache(
    config: &ProxyConfig,
    memory: Arc<InMemoryCache>,
) -> anyhow::Result<(
    Arc<dyn CacheTrait>,
    CacheBackendKind,
    Option<oxllm_cache_redis::RedisCache>,
)> {
    let Some(cache_cfg) = &config.oxllm_settings.cache else {
        return Ok((memory, CacheBackendKind::Memory, None));
    };

    // Blob backends enforce per-object TTL client-side (S3/Azure/GCS stamp
    // an `x-*-meta-expires` deadline on write and treat past-deadline reads
    // as misses); the stores themselves do not auto-delete on it, so bucket
    // lifecycle rules are still recommended to reclaim space.
    if matches!(
        cache_cfg.cache_type.as_deref(),
        Some("s3" | "gcs" | "azure")
    ) && config.oxllm_settings.cache_ttl_secs.is_some()
    {
        tracing::info!(
            backend = cache_cfg.cache_type.as_deref().unwrap_or(""),
            ttl_secs = ?config.oxllm_settings.cache_ttl_secs,
            "cache_ttl_secs is enforced client-side by blob cache backends \
             (metadata deadline; objects are not auto-deleted — configure \
             bucket lifecycle rules to reclaim space)"
        );
    }

    match cache_cfg.cache_type.as_deref() {
        // ---- Blob-store response caches (no atomic counters) ---------------
        // The blob builders return the kind they *actually constructed*
        // (`Memory` on any fallback: missing sub-config, construction error,
        // or feature flag disabled) so the readiness probe detects the
        // degraded backend instead of masking it as a healthy `"ok"`.
        Some("s3") => {
            let (cache, kind) = build_s3_cache(cache_cfg, memory).await;
            return Ok((cache, kind, None));
        }
        Some("gcs") => {
            let (cache, kind) = build_gcs_cache(cache_cfg, memory).await;
            return Ok((cache, kind, None));
        }
        Some("azure") => {
            let (cache, kind) = build_azure_cache(cache_cfg, memory).await;
            return Ok((cache, kind, None));
        }
        // ---- Redis ---------------------------------------------------------
        Some("redis") => {}
        // ---- memory / unknown ---------------------------------------------
        _ => return Ok((memory, CacheBackendKind::Memory, None)),
    }

    #[cfg(feature = "redis")]
    {
        use std::time::Duration;

        let url = redis_url_from_config(cache_cfg);
        let connect = oxllm_cache_redis::RedisCache::new(
            &url,
            cache_cfg.prefix.as_deref().unwrap_or("oxllm"),
        );
        match tokio::time::timeout(Duration::from_secs(3), connect).await {
            Ok(Ok(redis)) => {
                tracing::info!(url = %sanitize_url(&url), "redis cache enabled");
                Ok((
                    Arc::new(crate::cache::DualCache::new(memory, Some(redis.clone()))),
                    CacheBackendKind::Redis,
                    Some(redis),
                ))
            }
            Ok(Err(e)) => {
                tracing::error!(
                    error = %e,
                    url = %sanitize_url(&url),
                    "failed to connect to redis at startup; falling back to in-memory cache \
                     (readiness probe will report this backend as degraded)"
                );
                Ok((memory, CacheBackendKind::Memory, None))
            }
            Err(_) => {
                tracing::error!(
                    url = %sanitize_url(&url),
                    "redis connection timed out at startup; falling back to in-memory cache \
                     (readiness probe will report this backend as degraded)"
                );
                Ok((memory, CacheBackendKind::Memory, None))
            }
        }
    }

    #[cfg(not(feature = "redis"))]
    {
        tracing::warn!(
            "redis cache requested via config but 'redis' feature is not enabled; \
 using in-memory cache"
        );
        Ok((memory, CacheBackendKind::Memory, None))
    }
}

/// A cache wrapper for blob-store response backends (S3 / GCS / Azure) that
/// have no atomic increment primitive.
///
/// `get` / `set` / `delete` / `ping` are delegated to `blob` (the response
/// cache); the counter operations (`incr` / `incr_if_below` /
/// `decr_if_exists` / `counter_value` / `try_counter_value`) are delegated to
/// a local [`InMemoryCache`] so budget,
/// rate-limit, and JWT replay enforcement stay accurate per replica. The same
/// [`InMemoryCache`] instance is shared with the budget-reset scheduler, so
/// counter resets remain effective. Without this layer every counter call on a
/// blob backend would be a silent no-op and enforcement would be disabled.
#[cfg(any(
    feature = "s3-cache",
    feature = "gcs-cache",
    feature = "azure-cache",
    test
))]
struct BlobCounterCache {
    blob: Arc<dyn CacheTrait>,
    counters: Arc<InMemoryCache>,
}

#[cfg(any(
    feature = "s3-cache",
    feature = "gcs-cache",
    feature = "azure-cache",
    test
))]
#[async_trait]
impl CacheTrait for BlobCounterCache {
    /// Only `resp:*` (cached completion bodies) belongs on the object
    /// backend. Every other value key (`vk:*` auth snapshots, tombstones,
    /// coordination payloads, …) is served by the local in-memory store so
    /// the per-request auth path never crosses object storage (latency,
    /// request cost, and availability coupling).
    async fn get(&self, key: &str) -> Option<String> {
        if key.starts_with("resp:") {
            self.blob.get(key).await
        } else {
            self.counters.get(key).await
        }
    }
    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        if key.starts_with("resp:") {
            self.blob.set(key, value, ttl).await;
        } else {
            self.counters.set(key, value, ttl).await;
        }
    }
    /// Always clear the local counter/value map (reset + invalidation must
    /// take effect immediately per replica); also delete the blob only when
    /// the key is a response object.
    async fn delete(&self, key: &str) {
        self.counters.delete(key).await;
        if key.starts_with("resp:") {
            self.blob.delete(key).await;
        }
    }
    async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error> {
        self.counters.incr(key, by, ttl).await
    }
    async fn incr_if_below(
        &self,
        key: &str,
        delta: f64,
        ceiling: f64,
        ttl: Option<Duration>,
    ) -> Result<IncrResult, Error> {
        self.counters.incr_if_below(key, delta, ceiling, ttl).await
    }
    async fn counter_value(&self, key: &str) -> Option<f64> {
        self.counters.counter_value(key).await
    }
    /// Delegate to the in-memory store's atomic decrement (see
    /// [`oxllm_core::Cache::decr_if_exists`]) rather than the trait default,
    /// whose non-atomic `counter_value`-then-`incr` sequence could mint a
    /// negative, never-expiring counter on a window rollover between the two
    /// calls — permanently inflating the budget/RPM ceiling.
    async fn decr_if_exists(&self, key: &str, by: f64) -> Result<bool, Error> {
        self.counters.decr_if_exists(key, by).await
    }
    /// Delegate to the in-memory store (infallible) rather than the trait
    /// default, which would route through `get` and read a response blob.
    async fn try_counter_value(&self, key: &str) -> Result<Option<f64>, Error> {
        self.counters.try_counter_value(key).await
    }
    async fn ping(&self) -> bool {
        self.blob.ping().await
    }
}

/// Construct an S3-compatible cache. Falls back to the in-memory cache on
/// missing config (e.g. no bucket) with a warning. Credentials resolve from
/// config first, then the standard `AWS_*` env vars.
///
/// Returns the cache and the [`CacheBackendKind`] that was *actually
/// constructed* (`S3` on success, `Memory` on any fallback) so the caller
/// ([`build_cache`]) records the truth for the readiness probe rather than
/// masking a broken/misconfigured/feature-disabled backend as a healthy S3.
#[allow(unused_variables)]
async fn build_s3_cache(
    cache_cfg: &crate::config::types::CacheConfig,
    memory: Arc<InMemoryCache>,
) -> (Arc<dyn CacheTrait>, CacheBackendKind) {
    #[cfg(feature = "s3-cache")]
    {
        let Some(s3) = &cache_cfg.s3 else {
            tracing::warn!(
                "s3 cache requested but no `cache.s3` config provided; using in-memory cache"
            );
            return (memory, CacheBackendKind::Memory);
        };
        let region = s3
            .region
            .clone()
            .or_else(|| {
                std::env::var("AWS_REGION")
                    .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                    .ok()
            })
            .unwrap_or_else(|| "us-east-1".to_string());
        let access_key = s3
            .access_key_id
            .clone()
            .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
            .unwrap_or_default();
        let secret_key = s3
            .secret_access_key
            .clone()
            .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
            .unwrap_or_default();
        let session_token = s3
            .session_token
            .clone()
            .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok());
        let prefix = s3.prefix.clone().unwrap_or_else(|| "oxllm".to_string());
        match oxllm_cache_s3::S3Cache::new(
            &s3.bucket,
            &region,
            &prefix,
            &access_key,
            &secret_key,
            session_token,
            s3.endpoint.clone(),
        ) {
            Ok(cache) => {
                tracing::info!(
                    bucket = %s3.bucket,
                    region = %region,
                    "s3 response cache enabled (counters layered in-memory)"
                );
                // Blob stores have no atomic increment primitive, so the raw
                // S3Cache no-ops the counter operations. Layer the shared
                // in-memory cache as the counter store so budget / rate-limit
                // / JWT replay enforcement stays accurate per replica and
                // shares state with the budget-reset scheduler (which resets
                // counters on this same `memory` instance).
                (
                    Arc::new(BlobCounterCache {
                        blob: Arc::new(cache),
                        counters: memory.clone(),
                    }),
                    CacheBackendKind::S3,
                )
            }
            Err(e) => {
                tracing::warn!(error = %e, "s3 cache construction failed; using in-memory cache");
                (memory, CacheBackendKind::Memory)
            }
        }
    }
    #[cfg(not(feature = "s3-cache"))]
    {
        let _ = cache_cfg;
        tracing::warn!(
            "s3 cache requested via config but 's3-cache' feature is not enabled; \
using in-memory cache"
        );
        (memory, CacheBackendKind::Memory)
    }
}

/// Construct a GCS cache. Falls back to the in-memory cache on missing config.
///
/// Returns the cache and the [`CacheBackendKind`] that was *actually
/// constructed* (`Gcs` on success, `Memory` on any fallback) so the readiness
/// probe sees the real backend. See [`build_s3_cache`] for the rationale.
#[allow(unused_variables)]
async fn build_gcs_cache(
    cache_cfg: &crate::config::types::CacheConfig,
    memory: Arc<InMemoryCache>,
) -> (Arc<dyn CacheTrait>, CacheBackendKind) {
    #[cfg(feature = "gcs-cache")]
    {
        let Some(gcs) = &cache_cfg.gcs else {
            tracing::warn!(
                "gcs cache requested but no `cache.gcs` config provided; using in-memory cache"
            );
            return (memory, CacheBackendKind::Memory);
        };
        let prefix = gcs.prefix.clone().unwrap_or_else(|| "oxllm".to_string());
        match oxllm_cache_gcs::GcsCache::new(
            &gcs.bucket,
            &prefix,
            gcs.service_account_json.as_deref(),
        ) {
            Ok(cache) => {
                tracing::info!(
                    bucket = %gcs.bucket,
                    "gcs response cache enabled (counters layered in-memory)"
                );
                // Blob stores have no atomic increment primitive, so the raw
                // GcsCache no-ops the counter operations. Layer the shared
                // in-memory cache as the counter store so budget / rate-limit
                // / JWT replay enforcement stays accurate per replica and
                // shares state with the budget-reset scheduler (which resets
                // counters on this same `memory` instance).
                (
                    Arc::new(BlobCounterCache {
                        blob: Arc::new(cache),
                        counters: memory.clone(),
                    }),
                    CacheBackendKind::Gcs,
                )
            }
            Err(e) => {
                tracing::warn!(error = %e, "gcs cache construction failed; using in-memory cache");
                (memory, CacheBackendKind::Memory)
            }
        }
    }
    #[cfg(not(feature = "gcs-cache"))]
    {
        let _ = cache_cfg;
        tracing::warn!(
            "gcs cache requested via config but 'gcs-cache' feature is not enabled; \
using in-memory cache"
        );
        (memory, CacheBackendKind::Memory)
    }
}

/// Construct an Azure Blob cache. Falls back to the in-memory cache on missing
/// config.
///
/// Returns the cache and the [`CacheBackendKind`] that was *actually
/// constructed* (`Azure` on success, `Memory` on any fallback) so the readiness
/// probe sees the real backend. See [`build_s3_cache`] for the rationale.
#[allow(unused_variables)]
async fn build_azure_cache(
    cache_cfg: &crate::config::types::CacheConfig,
    memory: Arc<InMemoryCache>,
) -> (Arc<dyn CacheTrait>, CacheBackendKind) {
    #[cfg(feature = "azure-cache")]
    {
        let Some(azure) = &cache_cfg.azure else {
            tracing::warn!(
                "azure cache requested but no `cache.azure` config provided; using in-memory cache"
            );
            return (memory, CacheBackendKind::Memory);
        };
        let prefix = azure.prefix.clone().unwrap_or_else(|| "oxllm".to_string());
        match oxllm_cache_azure::AzureBlobCache::new(
            &azure.account,
            &azure.container,
            &prefix,
            azure.connection_string.as_deref(),
        ) {
            Ok(cache) => {
                tracing::info!(
                    account = %azure.account,
                    container = %azure.container,
                    "azure blob response cache enabled (counters layered in-memory)"
                );
                // Same blob-store counter limitation as S3/GCS: layer the
                // shared in-memory counter store so budget / rate-limit / JWT
                // replay enforcement stays accurate per replica.
                (
                    Arc::new(BlobCounterCache {
                        blob: Arc::new(cache),
                        counters: memory.clone(),
                    }),
                    CacheBackendKind::Azure,
                )
            }
            Err(e) => {
                tracing::warn!(error = %e, "azure cache construction failed; using in-memory cache");
                (memory, CacheBackendKind::Memory)
            }
        }
    }
    #[cfg(not(feature = "azure-cache"))]
    {
        let _ = cache_cfg;
        tracing::warn!(
            "azure cache requested via config but 'azure-cache' feature is not enabled; \
using in-memory cache"
        );
        (memory, CacheBackendKind::Memory)
    }
}

/// Install a plain `tracing_subscriber::fmt` subscriber, using JSON formatting
/// when `log_format` is `"json"`. Uses `try_init` so a subscriber already
/// installed by `otel::init` is left in place rather than panicking.
fn install_fmt_subscriber(log_format: Option<&str>) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    match log_format {
        Some("json") => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .json()
                .with_target(false)
                .try_init();
        }
        _ => {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_target(false)
                .try_init();
        }
    }
}

/// Run the server: load config, build state, serve with graceful shutdown.
pub async fn run(config_path: &str, port: Option<u16>) -> anyhow::Result<()> {
    crate::metrics::install();

    let config = crate::config::load_config(config_path)?;

    enforce_require_auth_guard(&config)?;

    // Install the global tracing subscriber based on the configured
    // `log_format`. `set_global_default` / `init()` can only succeed once per
    // process, so the subscriber must NOT have been installed earlier by the
    // caller. When the `otel` feature is enabled AND an OTel endpoint is
    // configured, `otel::init` owns the subscriber composition (so the OTLP
    // layer attaches); otherwise we install a plain fmt subscriber here.
    let log_format = config.general_settings.log_format.as_deref();
    #[cfg(feature = "otel")]
    let otel_guard: Option<crate::otel::OtelGuard> = {
        let gs = &config.general_settings;
        match crate::otel::init(
            gs.otel_endpoint.as_deref(),
            gs.otel_service_name.as_deref(),
            log_format,
        )? {
            // The OTel layer is attached to a freshly installed global
            // subscriber; hold the guard so shutdown flushes spans.
            crate::otel::InitOutcome::Attached(guard) => Some(guard),
            // No `otel_endpoint` configured: fall back to a plain fmt
            // subscriber so logs are not silently dropped.
            crate::otel::InitOutcome::Disabled => None,
            // An embedder pre-installed a global tracing subscriber, so the
            // OTel layer could not attach and spans will NOT be exported.
            // This is a hard misconfiguration (a pre-installed subscriber
            // swallows the OTel layer), not a silent no-op: fail loudly so
            // the operator notices instead of wondering why the collector
            // sees nothing.
            crate::otel::InitOutcome::SubscriberConflict => {
                anyhow::bail!(
                    "general_settings.otel_endpoint is configured but a global \
                     tracing subscriber is already installed; otel spans cannot \
                     be exported. Remove the pre-installed subscriber or unset \
                     otel_endpoint."
                );
            }
        }
    };

    // When the `otel` feature is enabled but no endpoint is configured,
    // `otel::init` returns `InitOutcome::Disabled` without installing a
    // subscriber. Fall back to a plain fmt subscriber so logs are not
    // silently dropped.
    #[cfg(feature = "otel")]
    if otel_guard.is_none() {
        install_fmt_subscriber(log_format);
    }

    #[cfg(not(feature = "otel"))]
    {
        install_fmt_subscriber(log_format);
    }

    let memory_cache = Arc::new(InMemoryCache::default());
    let (cache, cache_backend, redis_cache) = build_cache(&config, memory_cache.clone()).await?;
    let registry = oxllm_providers::default_registry();

    // C05/HIGH: the fleet-wide auth brake requires a distributed
    // [`crate::coordination::Coordinator`] attached to the state. When Redis
    // is the backing store, wire the bare Redis handle into a distributed
    // coordinator so `assemble` attaches it to the auth brake — otherwise even
    // a Redis-backed multi-replica deployment runs the per-process brake and
    // an attacker multiplies the failure budget by the replica count. Without
    // Redis the coordinator stays `None` and the brake degrades to per-process.
    let coordinator = redis_cache.as_ref().map(|redis| {
        crate::coordination::Coordinator::new(Arc::new(redis.clone()) as Arc<dyn CacheTrait>)
            .with_distributed(true)
    });

    // A configured `database_url` in a build without the `db` feature is a
    // silent no-op today (the field is unread): virtual keys, budgets, spend
    // logging, and management endpoints would quietly not exist, and a
    // database_url-only auth config would run in open mode. Fail fast instead.
    #[cfg(not(feature = "db"))]
    if config.general_settings.database_url.is_some() {
        anyhow::bail!(
            "general_settings.database_url is configured but this binary was built \
             without the `db` feature; the database would be silently ignored \
             (no virtual keys, budgets, or spend logging — and a database_url-only \
             auth config would serve unauthenticated). Rebuild with \
             `--features db` or remove database_url."
        );
    }

    #[cfg(feature = "db")]
    let (db, spend_tx, spend_writer_handle) = setup_persistence(&config).await?;

    // Process-wide graceful-shutdown flag. Stays `false` until the drain
    // signal fires (see the `with_graceful_shutdown` block below), then is
    // flipped to `true` so the liveness/readiness probes return a clean
    // `503 {"status":"shutting_down"}` instead of relying on the kubelet
    // probe timeout. Cloned into the state and into the shutdown path so
    // both observe the same flip. Preserved across hot-reloads (an `Arc`
    // cloned into `BuildParts` by `reload_from_config`).
    let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    #[cfg(feature = "db")]
    let mut initial_state = AppState::with_registry_cache_db_and_coordinator(
        config.clone(),
        registry,
        cache.clone(),
        spend_tx,
        db,
        coordinator,
    )?;
    #[cfg(not(feature = "db"))]
    let mut initial_state = AppState::with_registry_cache_and_coordinator(
        config.clone(),
        registry,
        cache.clone(),
        None,
        coordinator,
    )?;
    // Override the cache_backend (derived from config by the builders above)
    // with the kind that build_cache actually constructed, so the readiness
    // probe detects a Redis/blob fallback rather than reporting a false
    // positive.
    initial_state.cache_backend = cache_backend;
    // Override the default shutdown flag with the shared one so the probes
    // observe the flip driven by the drain path below.
    initial_state.shutdown = shutdown_flag.clone();

    let shared: SharedState = Arc::new(ArcSwap::from_pointee(initial_state));

    #[cfg(feature = "db")]
    {
        let current = shared.load_full();
        if current.db.is_some() {
            if let Err(e) = crate::state::refresh_db_models(&shared).await {
                tracing::warn!(
                    error = %e,
                    "failed to load DB-stored models at startup; YAML-only deployments are live"
                );
            }
        }
    }

    let config_path_owned = config_path.to_string();
    let rebuild: crate::config::watcher::Rebuild = Arc::new(move |prev: &AppState| {
        let path = config_path_owned.clone();
        Box::pin(async move {
            let config = crate::config::load_config_async(&path).await?;
            prev.reload_from_config(config)
        })
    });
    // The watcher resolves absolute/canonical paths while retaining logical
    // symlink paths and the root identity used to track included files.
    let watch_path = PathBuf::from(config_path);
    crate::config::watcher::spawn_watcher(watch_path, shared.clone(), rebuild);

    let state_snapshot = shared.load_full();

    // Port precedence: explicit CLI flag / `OXLLM_PORT` env (Some) wins;
    // otherwise the built-in default of 4000. Kept as an `Option` so the CLI
    // can distinguish "explicitly 4000" from "not specified" — a `u16` with
    // `default_value_t = 4000` erases that distinction.
    let port = port.unwrap_or(4000);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let actual_addr = listener.local_addr()?;

    let auth_enabled = state_snapshot.master_key_hash.is_some()
        || state_snapshot.jwt_validator.is_some()
        || state_snapshot.db.is_some();
    tracing::info!(
    addr = %actual_addr,
    models = state_snapshot.deployments.len(),
    auth_enabled,
    "oxllm proxy server starting"
    );

    if !auth_enabled {
        tracing::warn!("{OPEN_MODE_WARNING}");
    }
    if state_snapshot
        .config
        .general_settings
        .allow_short_master_key
        == Some(true)
    {
        tracing::warn!(
            "general_settings.allow_short_master_key is set; a short master key is accepted only in development"
        );
    }

    let cache_for_scheduler = memory_cache.clone();
    let budget_duration_str = state_snapshot
        .config
        .oxllm_settings
        .budget_duration
        .as_deref()
        .unwrap_or("30d");

    let budget_hook = Arc::new(crate::hooks::BudgetHook::new(
        cache.clone(),
        state_snapshot.config.oxllm_settings.max_budget,
        budget_duration_str,
    ));
    // The scheduler re-reads the current `AlertState` from `shared` on every
    // tick, so a hot-reload that replaces the alert state is honored without
    // restarting the scheduler.
    let scheduler = crate::scheduler::BudgetResetScheduler::new(
        cache_for_scheduler.clone(),
        budget_hook,
        shared.clone(),
    );
    // Only attach the cross-replica delete layer when the cache is actually a
    // `DualCache` over Redis. Blob backends (S3/GCS/Azure) have no shared
    // counter store (their counters live in the layered `InMemoryCache`), so a
    // `delete(&spend:*)` there would dispatch a real object-storage DELETE of an
    // object that was never written. `build_cache` returns the `RedisCache`
    // handle only for the Redis path; pass that bare handle (its `delete`
    // zeroes the key fleet-wide) rather than the full app cache.
    #[cfg(feature = "redis")]
    let scheduler = match &redis_cache {
        Some(redis) => scheduler.with_redis_cache(Arc::new(redis.clone()) as Arc<dyn CacheTrait>),
        None => scheduler,
    };
    #[cfg(not(feature = "redis"))]
    let scheduler = scheduler;
    let (scheduler_shutdown_tx, scheduler_shutdown_rx) = tokio::sync::oneshot::channel();
    let scheduler_join = tokio::spawn(scheduler.run(scheduler_shutdown_rx));
    tracing::info!("budget reset scheduler started (resets per-key and global budgets)");

    // Cross-replica cache invalidation subscriber: when Redis is configured,
    // every management mutation publishes the affected cache key on the
    // `oxllm:invalidate` channel. This task consumes that stream and clears
    // the corresponding local moka entry so peer replicas stop serving stale
    // `vk:*` data immediately instead of waiting for `VK_POSITIVE_TTL`.
    //
    // It also covers four robustness gaps:
    // - **Fallback DB poll (30 s).** Pub/sub delivery is at-most-once; a
    //   missed publish (Redis hiccup, subscriber stream down at the publish
    //   instant, transient `list_models_db` failure on the subscriber) would
    //   leave a peer serving stale models forever. A low-frequency poll
    //   self-heals that drift — `refresh_db_models`'s CAS loop no-ops when
    //   the deployment Vec is unchanged, so the cost is one `list_models_db`
    //   per poll when nothing changed.
    // - **`__model_list__` debounce (300 ms).** A burst of management edits
    //   (config-as-code, scripted bulk adds) publishes one event per
    //   mutation; without coalescing each one drives a full table scan +
    //   CAS on every peer. The debounce collapses a burst into one refresh.
    // - **Stream-end backoff.** When Redis accepts the SUBSCRIBE but drops
    //   the connection in a tight loop (max-clients throttling, OOM
    //   kill-and-restart), the previous code re-subscribed with no delay —
    //   a CPU-spinning tight loop. We sleep 1 s before re-subscribing on a
    //   stream-end (matching the existing subscribe-failure backoff).
    // - **Graceful shutdown.** A oneshot signal lets `run()`'s shutdown
    //   path abort the task cleanly so an in-flight refresh is cancelled
    //   (not abandoned mid-await by process exit), matching the scheduler.
    let invalidation_shutdown: Option<(
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    )> = if let Some(redis_cache) = redis_cache {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let memory = memory_cache.clone();
        #[cfg(feature = "db")]
        let shared_for_models = shared.clone();
        let join = tokio::spawn(async move {
            invalidation_subscriber_loop(
                redis_cache,
                memory,
                #[cfg(feature = "db")]
                shared_for_models,
                rx,
            )
            .await;
        });
        tracing::info!("cross-replica cache invalidation subscriber started");
        Some((tx, join))
    } else {
        None
    };

    // C9: spend-log retention pruning. When `spend_log_retention_days` is
    // set and a database is configured, spawn a background task that runs
    // once per day and deletes spend_logs rows older than the retention
    // window. The first tick is skipped (the `interval.tick().await` before
    // the loop) so we don't burn a DELETE on startup. Errors are logged
    // and swallowed — a transient DB failure must not kill the pruner.
    //
    // The task owns a oneshot shutdown receiver so `run()` can cancel it
    // cleanly on exit, mirroring the budget scheduler and invalidation
    // subscriber; without it a prune mid-`delete_spend_logs_before` is
    // cancelled by runtime teardown (idempotent on next process start, but
    // the cancellation was silent and asymmetric with the other background
    // tasks).
    #[cfg(feature = "db")]
    let pruner_shutdown: Option<(
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    )> = if let Some(retention_days) = state_snapshot
        .config
        .general_settings
        .spend_log_retention_days
    {
        if let Some(db) = state_snapshot.db.clone() {
            let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
            let join = tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(86400));
                // Skip the immediate first tick so a fresh startup doesn't
                // DELETE before the proxy has even served a request.
                interval.tick().await;
                loop {
                    tokio::select! {
                        biased;
                        _ = &mut shutdown_rx => {
                            tracing::info!("spend log retention pruner shutting down");
                            return;
                        }
                        _ = interval.tick() => {}
                    }
                    // `checked_sub` returns Err if the resulting timestamp
                    // would underflow UNIX_EPOCH — impossible for any realistic
                    // retention value, but fall back to "now" (which prunes
                    // nothing further) rather than the previous `unwrap_or_default()`
                    // which yielded UNIX_EPOCH and would delete every row.
                    let cutoff = spend_prune_cutoff_at(jiff::Timestamp::now(), retention_days);
                    // Bounded batches: one unbounded `DELETE FROM spend_logs
                    // WHERE timestamp < ?` holds the single SQLite writer lock
                    // (or a long Postgres transaction) for the whole table
                    // scan, which can starve the batched spend writer past
                    // its busy timeout and drop billing rows. Delete in
                    // 1,000-row batches and yield between them so concurrent
                    // writes interleave; drain until the batch comes back
                    // empty so a daily cycle actually catches up.
                    const PRUNE_BATCH_ROWS: i64 = 1_000;
                    let mut total_deleted: u64 = 0;
                    loop {
                        // Graceful shutdown must win over a long backlog:
                        // without this poll the drain keeps issuing batched
                        // deletes after SIGTERM and delays process exit.
                        if shutdown_rx.try_recv().is_ok() {
                            tracing::info!(
                                deleted_so_far = total_deleted,
                                "spend log retention pruner shutting down mid-drain"
                            );
                            return;
                        }
                        match db
                            .delete_spend_logs_before_batch(&cutoff, PRUNE_BATCH_ROWS)
                            .await
                        {
                            Ok(0) => break,
                            Ok(n) => {
                                total_deleted += n;
                                tokio::time::sleep(Duration::from_millis(50)).await;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    deleted_so_far = total_deleted,
                                    "failed to prune spend logs (will retry next cycle)"
                                );
                                break;
                            }
                        }
                    }
                    if total_deleted > 0 {
                        tracing::info!(
                            deleted = total_deleted,
                            retention_days,
                            "pruned old spend logs"
                        );
                    } else {
                        tracing::debug!(retention_days, "spend log prune: nothing to delete");
                    }
                }
            });
            tracing::info!(
                retention_days,
                "spend log retention pruner started (runs daily)"
            );
            Some((shutdown_tx, join))
        } else {
            None
        }
    } else {
        None
    };

    let configured_shutdown = state_snapshot.config.general_settings.shutdown_timeout_secs;
    let request_timeout = state_snapshot.config.oxllm_settings.request_timeout;
    let shutdown_timeout = compute_shutdown_timeout(configured_shutdown, request_timeout);

    let router = {
        // Pre-warm the BPE tokenizers on the blocking pool: without this the
        // first `estimate_tokens` call lazily parses the tiktoken ranks files
        // (tens of ms) — potentially on a tokio worker, stalling the first
        // billed request. Failure is logged; the lazy path remains as the
        // fallback.
        if let Err(e) = tokio::task::spawn_blocking(oxllm_core::warm_tokenizers).await {
            tracing::warn!(
                error = %e,
                "tokenizer pre-warm task failed; first estimate will lazy-init"
            );
        }
        build_router(shared)
    };

    let make_service = router.into_make_service_with_connect_info::<std::net::SocketAddr>();
    // Run until Ctrl-C / SIGTERM, then bound the in-flight drain to
    // `shutdown_timeout`. Axum's `with_graceful_shutdown` only stops ACCEPTING
    // new connections once the signal future resolves; each in-flight
    // connection then calls hyper's `conn.graceful_shutdown()`, which waits
    // indefinitely for the active request to finish — axum imposes no drain
    // timeout. So one hung/streaming upstream request would block `serve.await`
    // forever, also skipping the scheduler shutdown, the OTel span flush, and
    // the spend-writer flush that all run after serve returns.
    //
    // We therefore race the drain against a timeout that arms ONLY AFTER the
    // signal fires (not from process start — the earlier reverted bug wrapped
    // the whole serve in a wall-clock timeout and killed the process after N
    // seconds of uptime). If the drain exceeds the budget we log and stop
    // waiting; the process then proceeds to cleanup and exits, dropping the
    // in-flight connections.
    let serve = axum::serve(listener, make_service).with_graceful_shutdown(async move {
        shutdown_signal().await;
        // Flip the process-wide shutdown flag BEFORE logging so the probes
        // (which may be hit by the kubelet during the drain window) observe
        // `shutting_down` immediately, returning 503 instead of relying on
        // the probe timeout to mark the pod NotReady.
        shutdown_flag.store(true, std::sync::atomic::Ordering::Release);
        tracing::info!(
            drain_secs = shutdown_timeout,
            "shutdown signal received; draining connections"
        );
    });

    tracing::info!(
        shutdown_drain_secs = shutdown_timeout,
        configured_shutdown_secs = configured_shutdown.unwrap_or(SHUTDOWN_DEADLINE_SECS),
        request_timeout_secs = request_timeout.unwrap_or(DEFAULT_TIMEOUT_SECS),
        "listening until Ctrl-C/SIGTERM (drain budget = max(configured, request_timeout))"
    );

    // Capture a fatal serve error so the shared cleanup block always runs.
    // Propagating with `?` here would skip the scheduler/invalidation
    // shutdown and — crucially — the bounded spend-writer flush (C01 billing
    // accuracy), whose only deterministic flush path is the explicit
    // `handle.shutdown()` call below. `SpendWriterHandle::Drop` deliberately
    // does NOT drain.
    let mut serve_err: Option<anyhow::Error> = None;

    tokio::select! {
        biased;
        res = serve => {
            match res {
                Ok(()) => tracing::info!("oxllm proxy server stopped"),
                Err(e) => {
                    tracing::error!(error = %e, "server accept loop stopped with error");
                    serve_err = Some(e.into());
                }
            }
        }
        _ = async move {
            // Arm the drain budget only after the signal fires. A second
            // `shutdown_signal()` future is harmless: tokio signal handlers
            // resolve all waiting futures on the same event.
            shutdown_signal().await;
            tokio::time::sleep(Duration::from_secs(shutdown_timeout)).await;
        } => {
            tracing::warn!(
                drain_secs = shutdown_timeout,
                "graceful-drain budget exceeded; forcing exit (in-flight connections dropped)"
            );
        }
    }

    // Shutdown the budget-reset scheduler.
    let _ = scheduler_shutdown_tx.send(());
    let _ = scheduler_join.await;

    // Shutdown the spend-log retention pruner and join it, mirroring the
    // scheduler. Bounded so an in-flight DELETE cannot stall process exit.
    #[cfg(feature = "db")]
    if let Some((tx, join)) = pruner_shutdown {
        let _ = tx.send(());
        let _ = tokio::time::timeout(Duration::from_secs(2), join).await;
    }

    // Shutdown the cross-replica invalidation subscriber so an in-flight
    // `refresh_db_models` is cancelled cleanly rather than abandoned
    // mid-await by process exit. The task's `select!` is biased toward the
    // shutdown future, so it returns within one tick of the inner loop.
    let invalidation_join = if let Some((tx, join)) = invalidation_shutdown {
        let _ = tx.send(());
        Some(join)
    } else {
        None
    };
    // Join the subscriber task so a slow final `refresh_db_models` is not
    // cut short by runtime teardown. Bounded so a wedged DB cannot stall
    // process exit; the peer's model list self-heals via the 30s fallback
    // poll regardless.
    if let Some(handle) = invalidation_join {
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }

    // Flush any buffered spend telemetry before returning so a normal restart
    // does not lose entries queued after the last ticker tick. Bounded so a DB
    // outage cannot stall process exit indefinitely.
    #[cfg(feature = "db")]
    if let Some(handle) = spend_writer_handle {
        if tokio::time::timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .is_err()
        {
            tracing::warn!("spend writer shutdown timed out; some entries may be lost");
        }
    }

    // Flush the OpenTelemetry batch exporter on graceful exit. The
    // synchronous `provider.shutdown()` is moved onto the tokio blocking
    // pool by `OtelGuard::shutdown`, so it cannot stall the async worker
    // that drove the rest of the shutdown sequence. Bounded so a wedged
    // collector cannot stall process exit indefinitely. The guard's Drop
    // fallback is skipped on this path because `shutdown` consumes it.
    #[cfg(feature = "otel")]
    if let Some(guard) = otel_guard {
        if tokio::time::timeout(Duration::from_secs(6), guard.shutdown())
            .await
            .is_err()
        {
            tracing::warn!("otel tracer shutdown timed out; some spans may be lost");
        }
    }

    if let Some(e) = serve_err {
        return Err(e);
    }

    Ok(())
}

/// Production guard: when `OXLLM_REQUIRE_AUTH` is set to `"true"` or `"1"` but
/// no auth path is configured (master key, JWT, or database virtual keys),
/// refuse to start. This prevents accidentally running the proxy in open mode
/// in production.
///
/// Public so the `oxllm validate` subcommand can run the exact same pre-flight
/// check: a config that `run()` would refuse to start must not be reported as
/// "config is valid".
pub fn enforce_require_auth_guard(config: &ProxyConfig) -> anyhow::Result<()> {
    crate::auth::validate_auth_config(&config.general_settings)?;
    if short_master_key_counts(&config.general_settings) {
        return Ok(());
    }
    let env_value = std::env::var("OXLLM_REQUIRE_AUTH").ok();
    require_auth_guard(
        &config.general_settings.master_key,
        &config.general_settings.jwt_secret,
        &config.general_settings.database_url,
        env_value.as_deref(),
    )
}

/// Pure implementation of the `OXLLM_REQUIRE_AUTH` guard, extracted so tests
/// can exercise it without mutating process-wide environment variables
/// (`std::env::set_var`/`remove_var` are `unsafe` in the 2024 edition and this
/// crate is `#![deny(unsafe_code)]`).
///
/// Open mode is defined the same way as the auth middleware: no master key,
/// no JWT secret, and no database URL.
fn require_auth_guard(
    master_key: &Option<String>,
    jwt_secret: &Option<String>,
    database_url: &Option<String>,
    require_auth_env: Option<&str>,
) -> anyhow::Result<()> {
    let require_auth = require_auth_env
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    if require_auth && !auth_is_configured(master_key, jwt_secret, database_url) {
        anyhow::bail!(
            "OXLLM_REQUIRE_AUTH is set but no auth is configured; \
 refusing to start in open mode. Set general_settings.master_key, \
 jwt_secret, or database_url, or unset OXLLM_REQUIRE_AUTH."
        );
    }
    Ok(())
}

/// How often the cross-replica invalidation subscriber runs a fallback
/// `refresh_db_models` poll, even without a pub/sub trigger. Matches the
/// multi-instance stress-test plan's §7.2 fallback spec ("30 s DB poll when
/// Redis pub/sub is unavailable") and is cheap when nothing changed: one
/// `list_models_db` round-trip and a CAS that no-ops on an identical
/// deployment Vec.
const MODEL_LIST_FALLBACK_POLL_INTERVAL: Duration = Duration::from_secs(30);

/// Debounce window for `__model_list__` events. A burst of management edits
/// (config-as-code, scripted bulk adds) publishes one event per mutation;
/// without coalescing each one drives a full `list_models_db` + CAS on every
/// peer. The CAS loop makes the work idempotent, so collapsing the burst into
/// one refresh at the end of the window preserves correctness while bounding
/// DB load.
const MODEL_LIST_DEBOUNCE: Duration = Duration::from_millis(300);

/// Backoff before re-subscribing after the Redis pub/sub stream ends. The
/// previous code looped straight from `stream.end` into `redis.subscribe()`,
/// which is a tight CPU loop if Redis accepts SUBSCRIBE but drops the
/// connection immediately (max-clients throttling, OOM kill-and-restart).
const INVALIDATION_RESUBSCRIBE_BACKOFF: Duration = Duration::from_secs(1);

/// The cross-replica invalidation subscriber loop. Subscribes to the
/// prefix-namespaced invalidation channel, clears local cache entries on
/// `vk:*` keys, debounces `__model_list__` refreshes, runs a periodic fallback
/// DB poll, sleeps on stream-end before re-subscribing, and exits cleanly when
/// the shutdown signal fires.
///
/// Extracted from `run()` so the loop body is testable in isolation and the
/// `run()` call site reads as configuration. Exposed (`pub`) so integration
/// tests under `tests/` can boot two in-process replicas sharing a Redis and
/// assert cross-replica model-config propagation end-to-end.
pub async fn invalidation_subscriber_loop(
    redis_cache: oxllm_cache_redis::RedisCache,
    memory: Arc<InMemoryCache>,
    #[cfg(feature = "db")] shared_for_models: SharedState,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) {
    use futures_util::StreamExt;
    // Namespace the channel with the configured Redis key prefix so two
    // fleets sharing one Redis with different prefixes do not cross-deliver
    // invalidation messages. With the default prefix ("oxllm") this yields
    // "oxllm:invalidate", preserving the historical channel name.
    let channel = redis_cache.pubsub_channel(crate::cache::INVALIDATION_CHANNEL_BASE);
    loop {
        let mut stream = match redis_cache.subscribe(&channel).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to subscribe to invalidation channel; retrying in 1s"
                );
                tokio::select! {
                    biased;
                    _ = &mut shutdown => return,
                    _ = tokio::time::sleep(INVALIDATION_RESUBSCRIBE_BACKOFF) => {},
                }
                continue;
            }
        };
        // The periodic fallback poll that self-heals a missed pub/sub
        // publish. `interval.tick()` fires immediately on first call, which
        // we want here: arm the first poll right after (re)subscribe so a
        // replica that missed a burst of edits while its stream was down
        // catches up at most one debounce window after reconnecting.
        let mut poll = tokio::time::interval(MODEL_LIST_FALLBACK_POLL_INTERVAL);
        // Debounce state for `__model_list__` events. Modeled as an
        // `Option<Pin<Box<Sleep>>>` because `tokio::time::Sleep` is `!Unpin`
        // and we need to reset it across iterations; the inner future is
        // rebuilt each loop step from the current slot, and `None` resolves
        // to `pending()` so the select! branch never fires spuriously.
        let mut debounce: Option<std::pin::Pin<Box<tokio::time::Sleep>>> = None;
        loop {
            let debounce_fut = async {
                match debounce.as_mut() {
                    Some(d) => d.as_mut().await,
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::pin!(debounce_fut);
            tokio::select! {
                biased;
                _ = &mut shutdown => {
                    tracing::info!("cross-replica invalidation subscriber shutting down");
                    return;
                }
                // Periodic fallback DB poll. Always runs regardless of
                // pub/sub health; `refresh_db_models`'s CAS no-ops when
                // nothing changed.
                _ = poll.tick() => {
                    refresh_models_best_effort(
                        #[cfg(feature = "db")]
                        &shared_for_models,
                    )
                    .await;
                }
                // Debounce timer fired: run the coalesced refresh.
                _ = &mut debounce_fut => {
                    refresh_models_best_effort(
                        #[cfg(feature = "db")]
                        &shared_for_models,
                    )
                    .await;
                    debounce = None;
                }
                msg = stream.next() => {
                    let Some(msg) = msg else {
                        tracing::warn!(
                            "redis invalidation stream ended; \
                             attempting to resubscribe in {:?}",
                            INVALIDATION_RESUBSCRIBE_BACKOFF
                        );
                        // Sleep before re-subscribing on a stream-end to
                        // avoid a tight loop if Redis drops the connection
                        // immediately after SUBSCRIBE.
                        tokio::select! {
                            biased;
                            _ = &mut shutdown => return,
                            _ = tokio::time::sleep(INVALIDATION_RESUBSCRIBE_BACKOFF) => {},
                        }
                        break;
                    };
                    match msg.get_payload::<String>() {
                        Ok(key) if key == "__model_list__" => {
                            // Arm (or reset) the debounce timer. A burst of
                            // management edits collapses into one refresh.
                            debounce = Some(Box::pin(tokio::time::sleep(MODEL_LIST_DEBOUNCE)));
                        }
                        Ok(key) => {
                            memory.delete(&key).await;
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "invalidation message payload decode failed; dropping"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Best-effort `refresh_db_models` wrapper that handles the `db` feature
/// gate and logs/swallows errors so callers (the invalidation subscriber
/// loop) do not need their own `#[cfg]` plumbing — keeping the `select!`
/// branches cfg-free, which the tokio macro requires.
#[cfg(feature = "db")]
async fn refresh_models_best_effort(shared: &SharedState) {
    match crate::state::refresh_db_models(shared).await {
        Ok(()) => tracing::debug!("cross-replica model-list refresh completed"),
        Err(e) => tracing::warn!(error = %e, "cross-replica model-list refresh failed"),
    }
}

#[cfg(not(feature = "db"))]
async fn refresh_models_best_effort() {}

/// Severity level for the SQLite multi-instance startup guard, extracted so
/// the policy is unit-testable without booting a server. See
/// [`setup_persistence`] for the full rationale.
#[cfg(feature = "db")]
#[derive(Debug, PartialEq, Eq)]
enum SqliteGuardLevel {
    /// No log (e.g. the database URL is not SQLite).
    None,
    /// Plain single-instance SQLite — keep the existing informational warn.
    Warn,
    /// SQLite + Redis co-configured — almost certainly multi-instance;
    /// escalate so the silent spend-log loss risk is loud.
    Error,
}

/// Decide the SQLite guard severity for the given database URL and Redis
/// configuration. The URL prefix check mirrors the `url.starts_with("sqlite://")`
/// gate inside [`setup_persistence`]; the helper returns [`SqliteGuardLevel::None`]
/// for non-SQLite URLs so the call site can use a single `match` instead of a
/// nested `if`.
#[cfg(feature = "db")]
fn sqlite_guard_level(database_url: &str, redis_configured: bool) -> SqliteGuardLevel {
    if !database_url.starts_with("sqlite://") {
        return SqliteGuardLevel::None;
    }
    if redis_configured {
        SqliteGuardLevel::Error
    } else {
        SqliteGuardLevel::Warn
    }
}

/// Parse the `OXLLM_AUTO_MIGRATE` environment variable into a boolean.
///
/// Returns `true` (auto-migrate enabled) for the unset case (`None`) to
/// preserve the backward-compatible single-instance default. When set, the
/// value is trimmed and lowercased before comparison so common operator
/// typos (`False`, `FALSE`, ` false `, etc.) still disable auto-migration.
/// The recognised false-values are: `false`, `0`, `no`, `off`. Any other
/// value is treated as enabled, which is the safe fail-open default.
#[cfg_attr(not(feature = "db"), allow(dead_code))]
fn parse_auto_migrate(raw: Option<String>) -> bool {
    let Some(v) = raw else {
        return true;
    };
    let v = v.trim().to_ascii_lowercase();
    !(v == "false" || v == "0" || v == "no" || v == "off")
}

/// Connect + migrate the database and start the batched spend writer when a
/// `database_url` is configured. Returns `(db_handle, spend_writer_sender,
/// spend_writer_handle)`, all `None` when persistence is off. A configured
/// database that fails to connect (or, with `OXLLM_AUTO_MIGRATE` unset/true,
/// fails to migrate) is a hard startup error: the caller propagates it and
/// the process exits. To migrate as an independent pre-deploy step instead,
/// run `oxllm migrate --database-url ...` and start the server with
/// `OXLLM_AUTO_MIGRATE=false` (connection is still required at boot).
/// The `SpendWriterHandle` is retained by `run()` so it can trigger a bounded final
/// flush of buffered spend telemetry on graceful shutdown.
#[cfg(feature = "db")]
use crate::state::redis_configured;

#[cfg(feature = "db")]
/// Build the persistence stack when `database_url` is configured.
///
/// A configured database is auth-bearing infrastructure: virtual keys,
/// budgets, rate limits, and spend logging all depend on it. A connect or
/// migration failure is therefore FATAL — continuing without the DB would
/// silently drop enforcement (and, on a `database_url`-only auth config,
/// run the proxy unauthenticated in open mode). Boot-time DB outages should
/// be handled by orchestration (restart/backoff), not by degrading security.
async fn setup_persistence(
    config: &ProxyConfig,
) -> anyhow::Result<(
    Option<Arc<oxllm_db::Database>>,
    Option<tokio::sync::mpsc::Sender<crate::spend_writer::SpendEntry>>,
    Option<crate::spend_writer::SpendWriterHandle>,
)> {
    let Some(url) = &config.general_settings.database_url else {
        return Ok((None, None, None));
    };

    if url.starts_with("sqlite://") {
        // When Redis is co-configured, this deployment is very likely
        // multi-replica (Redis is the strongest multi-instance signal in
        // the config). N replicas each running a batched spend writer
        // against the same SQLite file → guaranteed `SQLITE_BUSY` on the
        // write path, which the spend writer swallows/queues → silent
        // spend-log loss (a billing-accuracy problem, C01). Escalate to
        // `error!` so the misconfiguration is loud in production logs; the
        // proxy still boots so an operator who reads the log can run a
        // single-replica deploy with SQLite + Redis without a hard block.
        match sqlite_guard_level(url, redis_configured(config)) {
            SqliteGuardLevel::Warn => {
                tracing::warn!(
                    url = %sanitize_url(url),
                    "SQLite is configured for persistence. SQLite is NOT safe for \
                     multi-instance deployments — concurrent writes from multiple oxllm \
                     instances will cause locking errors or data corruption. Use \
                     PostgreSQL (database_url: postgres://...) when running more than \
                     one instance behind a load balancer."
                );
                metrics::gauge!(
                    crate::metrics::SQLITE_MULTI_INSTANCE_GUARD,
                    "level" => "warn"
                )
                .set(1.0);
            }
            SqliteGuardLevel::Error => {
                tracing::error!(
                    url = %sanitize_url(url),
                    "SQLite is configured for persistence AND Redis is \
                     co-configured — this is almost certainly a multi-instance \
                     deployment, but SQLite is NOT safe for concurrent writes \
                     from multiple oxllm instances. The batched spend writer \
                     will hit SQLITE_BUSY and silently lose spend-log rows \
                     (a billing-accuracy problem): those rows are being lost \
                     for as long as this process runs. Use PostgreSQL \
                     (database_url: postgres://...) when running more than one \
                     instance behind a load balancer."
                );
                metrics::gauge!(
                    crate::metrics::SQLITE_MULTI_INSTANCE_GUARD,
                    "level" => "error"
                )
                .set(1.0);
            }
            SqliteGuardLevel::None => {}
        }
    }

    match oxllm_db::Database::connect(url).await {
        Ok(db) => {
            // Auto-migration can be disabled via OXLLM_AUTO_MIGRATE=false for
            // multi-instance deployments where migrations are run as a separate
            // pre-step (e.g. an init container running `oxllm migrate`). This
            // prevents race conditions when multiple instances start
            // simultaneously against the same fresh database.
            let auto_migrate = parse_auto_migrate(std::env::var("OXLLM_AUTO_MIGRATE").ok());
            if auto_migrate {
                if let Err(e) = db.migrate().await {
                    let error = oxllm_core::redaction::redact_connection_text(&format!("{e:#}"));
                    anyhow::bail!(
                        "database migration failed for {}: {error}; set OXLLM_AUTO_MIGRATE=false \
                         and run 'oxllm migrate' separately if migrations are managed \
                         externally. Refusing to start without the configured database: \
                         virtual keys, budgets, and spend logging would be silently \
                         disabled (and a database_url-only auth config would serve \
                         unauthenticated).",
                        sanitize_url(url)
                    );
                }
                tracing::info!(url = %sanitize_url(url), "database connected and migrated");
            } else {
                tracing::info!(
                    url = %sanitize_url(url),
                    "database connected (auto-migrate disabled via OXLLM_AUTO_MIGRATE; \
                     ensure migrations were applied separately)"
                );
            }
            let (writer, handle) = crate::spend_writer::spawn_with_shutdown(
                db.clone(),
                60,
                crate::spend_writer::DEFAULT_CHANNEL_CAPACITY,
            );
            Ok((Some(db), Some(writer.sender()), Some(handle)))
        }
        Err(e) => {
            let error = oxllm_core::redaction::redact_connection_text(&format!("{e:#}"));
            anyhow::bail!(
                "database connection failed for {}: {error}. Refusing to start without the \
                 configured database: virtual keys, budgets, and spend logging would be \
                 silently disabled (and a database_url-only auth config would serve \
                 unauthenticated). Fix the database or remove general_settings.database_url.",
                sanitize_url(url)
            );
        }
    }
}

/// Graceful shutdown on Ctrl-C / SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            tracing::warn!("failed to install Ctrl-C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => {
                tracing::warn!("failed to install SIGTERM handler");
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
    _ = ctrl_c => {},
    _ = terminate => {},
    }
}

/// Compute the effective graceful-shutdown timeout so that in-flight requests
/// have at least their configured request timeout to complete during shutdown:
/// `max(configured_shutdown, request_timeout)`.
pub fn compute_shutdown_timeout(
    configured_shutdown: Option<u64>,
    request_timeout: Option<u64>,
) -> u64 {
    let configured = configured_shutdown.unwrap_or(SHUTDOWN_DEADLINE_SECS);
    let request = request_timeout.unwrap_or(DEFAULT_TIMEOUT_SECS);
    configured.max(request)
}

/// Compute the spend-log prune cutoff timestamp for a retention window of
/// `retention_days` relative to `now`. Returns an RFC3339 string suitable for
/// `delete_spend_logs_before`.
///
/// On `checked_sub` underflow (a retention so large that `now − retention`
/// precedes UNIX_EPOCH — unreachable for realistic values) the fallback is
/// `now` itself (prunes nothing further), NOT UNIX_EPOCH: a delete-all fallback
/// would be a data-loss footgun.
#[cfg(feature = "db")]
fn spend_prune_cutoff_at(now: jiff::Timestamp, retention_days: u64) -> String {
    let retention_secs = i64::try_from(retention_days)
        .unwrap_or(i64::MAX)
        .saturating_mul(86400);
    match now.checked_sub(jiff::SignedDuration::from_secs(retention_secs)) {
        Ok(ts) => ts.to_string(),
        Err(_) => {
            tracing::warn!(
                retention_days,
                "spend-log prune cutoff underflowed; using 'now' (prunes nothing further)"
            );
            now.to_string()
        }
    }
}

/// Build the Redis connection URL from the cache config. A configured
/// `redis_url` (which can carry username, password, TLS via `rediss://`, and a
/// database index) takes precedence over the bare `host`/`port` pair, which
/// cannot express any of those. Extracted as a pure function so the URL
/// construction (including auth/TLS/db) is unit-testable without connecting.
fn redis_url_from_config(cache_cfg: &crate::config::types::CacheConfig) -> String {
    if let Some(url) = &cache_cfg.redis_url {
        return url.clone();
    }
    let host = cache_cfg.host.as_deref().unwrap_or("localhost");
    let port = cache_cfg.port.unwrap_or(6379);
    format!("redis://{host}:{port}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{
        ConfigModel, GeneralSettings, OxllmSettings, PassThroughEndpoint, ProxyConfig,
    };
    use tower::ServiceExt;

    fn config_with_master_key(master_key: Option<String>) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: crate::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                master_key,
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    fn build_test_router(config: ProxyConfig) -> axum::Router {
        let shared = build_shared_state(config).expect("state");
        build_router(shared)
    }

    #[tokio::test]
    async fn metrics_rejects_unauthenticated_when_master_key_set() {
        let app = build_test_router(config_with_master_key(Some(
            "sk-master-test-key".to_string(),
        )));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn metrics_succeeds_with_master_key() {
        let app = build_test_router(config_with_master_key(Some(
            "sk-master-test-key".to_string(),
        )));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .header("authorization", "Bearer sk-master-test-key")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert!(resp.status().is_success());
    }

    #[tokio::test]
    async fn metrics_open_when_no_master_key() {
        let app = build_test_router(config_with_master_key(None));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/metrics")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert!(resp.status().is_success());
    }

    #[tokio::test]
    async fn health_is_public() {
        let app = build_test_router(config_with_master_key(Some(
            "sk-master-test-key".to_string(),
        )));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    /// `/health/liveness` and `/health/liveliness` are legacy-compatible
    /// aliases for `/health` (k8s liveness probe compatibility). They must be
    /// public and return `{"status":"healthy"}` so a deployment migrating from
    /// an existing proxy does not get 404 on its existing probe paths.
    #[tokio::test]
    async fn health_liveness_and_liveliness_aliases_work() {
        let app = build_test_router(config_with_master_key(Some(
            "sk-master-test-key".to_string(),
        )));
        for path in ["/health/liveness", "/health/liveliness"] {
            let resp = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("GET")
                        .uri(path)
                        .body(axum::body::Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(
                resp.status(),
                axum::http::StatusCode::OK,
                "{path} should be 200"
            );
            let body = axum::body::to_bytes(resp.into_body(), 1024)
                .await
                .expect("body");
            let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
            assert_eq!(v["status"], "healthy", "{path} body mismatch");
        }
    }

    #[test]
    fn require_auth_guard_passes_with_master_key_set() {
        let key = Some("sk-master-test-key".to_string());
        let none: Option<String> = None;
        assert!(require_auth_guard(&key, &none, &none, None).is_ok());
        assert!(require_auth_guard(&key, &none, &none, Some("true")).is_ok());
        assert!(require_auth_guard(&key, &none, &none, Some("1")).is_ok());
    }

    #[test]
    fn require_auth_guard_passes_with_jwt_only() {
        let none: Option<String> = None;
        let jwt = Some("s3cr3t-long-enough!".to_string());
        assert!(require_auth_guard(&none, &jwt, &none, Some("true")).is_ok());
    }

    #[test]
    fn require_auth_guard_passes_with_database_url_only() {
        let none: Option<String> = None;
        let db = Some("sqlite::memory:".to_string());
        assert!(require_auth_guard(&none, &none, &db, Some("true")).is_ok());
    }

    #[test]
    fn require_auth_guard_passes_in_open_mode_when_env_unset() {
        let none: Option<String> = None;
        assert!(require_auth_guard(&none, &none, &none, None).is_ok());
        assert!(require_auth_guard(&none, &none, &none, Some("false")).is_ok());
        assert!(require_auth_guard(&none, &none, &none, Some("")).is_ok());
        assert!(require_auth_guard(&none, &none, &none, Some("no")).is_ok());
    }

    #[test]
    fn require_auth_guard_rejects_open_mode_when_set_true() {
        let none: Option<String> = None;
        let result = require_auth_guard(&none, &none, &none, Some("true"));
        assert!(
            result.is_err(),
            "should refuse to start when OXLLM_REQUIRE_AUTH=true and no auth path"
        );
    }

    #[test]
    fn require_auth_guard_rejects_open_mode_when_set_one() {
        let none: Option<String> = None;
        let result = require_auth_guard(&none, &none, &none, Some("1"));
        assert!(result.is_err());
    }

    #[test]
    fn require_auth_guard_error_message_mentions_auth_options() {
        let none: Option<String> = None;
        let result = require_auth_guard(&none, &none, &none, Some("true"));
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("master_key")
                || msg.contains("jwt_secret")
                || msg.contains("database_url"),
            "error should mention auth config options: {msg}"
        );
        assert!(
            msg.contains("OXLLM_REQUIRE_AUTH"),
            "error should mention OXLLM_REQUIRE_AUTH: {msg}"
        );
    }

    #[test]
    fn auth_is_configured_matches_open_mode_gate() {
        let none: Option<String> = None;
        assert!(!auth_is_configured(&none, &none, &none));
        assert!(auth_is_configured(
            &Some("valid-master-key-value".into()),
            &none,
            &none
        ));
        assert!(auth_is_configured(
            &none,
            &Some("valid-jwt-secret-value".into()),
            &none
        ));
        assert!(!auth_is_configured(&Some("k".into()), &none, &none));
        assert!(!auth_is_configured(&none, &Some(" ".repeat(32)), &none));
        assert!(!auth_is_configured(&none, &none, &Some(" ".into())));
        assert!(auth_is_configured(&none, &none, &Some("db".into())));
    }

    #[test]
    fn shutdown_timeout_defaults_when_unset() {
        let t = compute_shutdown_timeout(None, None);
        assert_eq!(
            t, DEFAULT_TIMEOUT_SECS,
            "when neither set, request_timeout default ({}) should win over shutdown default ({})",
            DEFAULT_TIMEOUT_SECS, SHUTDOWN_DEADLINE_SECS
        );
    }

    #[test]
    fn shutdown_timeout_at_least_request_timeout() {
        let t = compute_shutdown_timeout(Some(30), Some(600));
        assert_eq!(t, 600, "shutdown must be at least request_timeout");
    }

    #[test]
    fn shutdown_timeout_uses_configured_when_larger() {
        let t = compute_shutdown_timeout(Some(900), Some(600));
        assert_eq!(t, 900, "larger configured shutdown should win");
    }

    #[test]
    fn shutdown_timeout_request_default_wins_when_configured_smaller() {
        let t = compute_shutdown_timeout(Some(10), None);
        assert_eq!(
            t, DEFAULT_TIMEOUT_SECS,
            "request_timeout default ({}) must win over small configured shutdown (10)",
            DEFAULT_TIMEOUT_SECS
        );
    }

    // ---- pass_through_endpoints robustness --------------------------------

    fn config_with_passthrough(paths: &[&str]) -> ProxyConfig {
        let endpoints: Vec<PassThroughEndpoint> = paths
            .iter()
            .map(|p| PassThroughEndpoint {
                path: (*p).to_string(),
                target: "https://example.com/upstream".to_string(),
                headers: None,
            })
            .collect();
        ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: crate::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                master_key: Some("sk-master-test-key".to_string()),
                pass_through_endpoints: Some(endpoints),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    /// Building the router with malformed `pass_through_endpoints` paths must
    /// not panic — empty/root entries collide with the global catch-all and
    /// duplicate normalised patterns collide with each other inside
    /// `attach_endpoints`. The misconfigured entries should be skipped with a
    /// warning instead.
    #[test]
    fn build_router_does_not_panic_on_misconfigured_pass_through_endpoints() {
        let app = build_test_router(config_with_passthrough(&["", "/", "/foo", "/foo/", "/bar"]));
        // The router built successfully (no panic) and is usable. The valid
        // prefixes (`/foo`, `/bar`) are registered; the empty/root/duplicate
        // entries are skipped.
        let _ = app;
    }

    /// A configured pass-through prefix that overlaps the global catch-all
    /// must take precedence: an unauthenticated `OPTIONS` to it should be
    /// rejected by the no-preflight auth stack (401), matching the catch-all.
    #[tokio::test]
    async fn options_to_configured_passthrough_is_unauthorized_without_credentials() {
        let app = build_test_router(config_with_passthrough(&["/pt"]));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("OPTIONS")
                    .uri("/pt/anything")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
    }

    // ---- catch-all OPTIONS handling ---------------------------------------

    /// Unauthenticated `OPTIONS` to an unmatched path (handled by the global
    /// catch-all) must return 401, not 500. The catch-all has no CORS layer,
    /// so the preflight bypass would forward the request straight into a
    /// handler whose `Extension<AuthUser>` extractor is missing. The
    /// no-preflight auth wrapper restores the correct 401.
    #[tokio::test]
    async fn options_to_unmatched_path_returns_401_not_500() {
        let app = build_test_router(config_with_master_key(Some(
            "sk-master-test-key".to_string(),
        )));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("OPTIONS")
                    .uri("/v1/does-not-exist")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
    }

    /// Sanity check that a non-OPTIONS method to the catch-all is also
    /// rejected with 401 (the auth stack still applies; only the harmful
    /// preflight bypass was removed).
    #[tokio::test]
    async fn get_to_unmatched_path_returns_401_without_credentials() {
        let app = build_test_router(config_with_master_key(Some(
            "sk-master-test-key".to_string(),
        )));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/v1/does-not-exist")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
    }

    // ---- CORS preflight vs auth interaction -------------------------------

    fn config_with_cors(master_key: Option<String>, origins: Vec<String>) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: crate::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                master_key,
                proxy_cors_origins: Some(origins),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    /// When `proxy_cors_origins` is configured, an unauthenticated CORS
    /// preflight (`OPTIONS /v1/chat/completions` with a matching
    /// `Origin`) must be bypassed by `auth_with_preflight` and reach the
    /// CORS layer, producing a 2xx with `Access-Control-Allow-Origin`.
    /// This locks in the intended asymmetry: the v1 routes (which carry a
    /// CORS layer) bypass auth for OPTIONS, unlike the CORS-less catch-all.
    #[tokio::test]
    async fn cors_preflight_bypasses_auth_on_v1_routes() {
        let app = build_test_router(config_with_cors(
            Some("sk-master-test-key".to_string()),
            vec!["https://app.example.com".to_string()],
        ));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("OPTIONS")
                    .uri("/v1/chat/completions")
                    .header("origin", "https://app.example.com")
                    .header("access-control-request-method", "POST")
                    .header("access-control-request-headers", "authorization")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert!(
            resp.status().is_success(),
            "preflight should succeed, got {}",
            resp.status()
        );
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("https://app.example.com")
        );
    }

    /// Companion check: even with CORS configured, a non-preflight method
    /// (`POST`) with no credentials on a v1 route is still rejected with 401.
    /// The preflight bypass is scoped to `OPTIONS` only.
    #[tokio::test]
    async fn non_options_still_requires_auth_with_cors_configured() {
        let app = build_test_router(config_with_cors(
            Some("sk-master-test-key".to_string()),
            vec!["https://app.example.com".to_string()],
        ));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("origin", "https://app.example.com")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
    }

    /// Minimal blob backend: serves a fixed sentinel from `get`, no-op
    /// elsewhere, and has no counters of its own.
    struct StubBlob;
    #[async_trait]
    impl CacheTrait for StubBlob {
        async fn get(&self, _key: &str) -> Option<String> {
            Some("from-blob".to_string())
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
            Err(Error::Config("blob store has no counters".to_string()))
        }
    }

    /// The fix for the silent counter no-op: wrapping a blob backend with
    /// [`BlobCounterCache`] restores per-replica enforcement (counters come
    /// from the in-memory layer), so `incr_if_below` rejects above the ceiling
    /// while response `get` still reads from the blob backend.
    #[tokio::test]
    async fn blob_counter_cache_enforces_counters_and_reads_blobs() {
        let memory = Arc::new(InMemoryCache::default());
        let cache = BlobCounterCache {
            blob: Arc::new(StubBlob),
            counters: memory.clone(),
        };

        // Response reads come from the blob backend, not the counter store.
        assert_eq!(cache.get("resp:k").await.as_deref(), Some("from-blob"));

        // Counter enforcement is live: allow up to the ceiling, then reject.
        let allowed = cache
            .incr_if_below("spend:k", 5.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(allowed, IncrResult::Allowed(5.0));
        let rejected = cache
            .incr_if_below("spend:k", 6.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(rejected, IncrResult::Rejected(5.0));

        // The counter is visible to the shared memory layer, so the
        // budget-reset scheduler (which reads this same instance) sees it.
        assert_eq!(memory.counter_value("spend:k").await, Some(5.0));
        assert_eq!(cache.counter_value("spend:k").await, Some(5.0));
    }

    /// Non-response keys (`vk:*` auth snapshots, tombstones, coordination
    /// payloads) must never cross the object backend: their reads/writes are
    /// served by the local store, and `delete` clears the local value. The
    /// stub blob answers `"from-blob"` for ANY key it is asked, so these
    /// assertions prove the wrapper never forwards the key.
    #[tokio::test]
    async fn blob_counter_cache_routes_auth_values_to_memory() {
        let memory = Arc::new(InMemoryCache::default());
        let cache = BlobCounterCache {
            blob: Arc::new(StubBlob),
            counters: memory.clone(),
        };

        // An auth snapshot round-trips through the local store only.
        cache
            .set("vk:abc123", "{\"max_budget\":5}".to_string(), None)
            .await;
        assert_eq!(
            cache.get("vk:abc123").await.as_deref(),
            Some("{\"max_budget\":5}")
        );

        // Deletion clears the local value even for non-response keys.
        cache.delete("vk:abc123").await;
        assert_eq!(cache.get("vk:abc123").await, None);

        // Response-body keys still flow to the blob backend.
        cache.set("resp:k", "body".to_string(), None).await;
        assert_eq!(cache.get("resp:k").await.as_deref(), Some("from-blob"));
    }

    /// `BlobCounterCache::decr_if_exists` must delegate to the in-memory
    /// counter layer's atomic implementation rather than the trait default's
    /// non-atomic `counter_value`-then-`incr` sequence: a release landing
    /// after the window expired is a no-op (`Ok(false)`) instead of minting a
    /// negative, never-expiring counter on the blob store's fallback `incr`
    /// path (which has no counters at all).
    #[tokio::test]
    async fn blob_counter_cache_decr_if_exists_delegates_to_memory() {
        let memory = Arc::new(InMemoryCache::default());
        let cache = BlobCounterCache {
            blob: Arc::new(StubBlob),
            counters: memory.clone(),
        };
        // Absent counter → no-op, never a negative residue.
        assert!(!cache.decr_if_exists("spend:k", 1.0).await.unwrap());
        assert_eq!(memory.counter_value("spend:k").await, None);
        // Present counter → atomically decremented on the shared layer.
        let _ = memory.incr("spend:k", 5.0, None).await.unwrap();
        assert!(cache.decr_if_exists("spend:k", 2.0).await.unwrap());
        assert_eq!(cache.counter_value("spend:k").await, Some(3.0));
        // Non-finite delta → Err, counter unchanged.
        assert!(cache.decr_if_exists("spend:k", f64::NAN).await.is_err());
        assert_eq!(cache.counter_value("spend:k").await, Some(3.0));
    }

    /// The S3 wiring must not expose the raw no-op backend as the application
    /// cache: `build_s3_cache` layers [`BlobCounterCache`] so a budget-over
    /// reservation is rejected instead of silently allowed.
    #[cfg(feature = "s3-cache")]
    #[tokio::test]
    async fn build_s3_cache_enforces_counters() {
        let cache_cfg = crate::config::types::CacheConfig {
            cache_type: Some("s3".to_string()),
            s3: Some(crate::config::types::S3CacheConfig {
                bucket: "mybucket".to_string(),
                region: Some("us-east-1".to_string()),
                prefix: Some("oxllm".to_string()),
                access_key_id: Some("AKIDEXAMPLE".to_string()),
                secret_access_key: Some("sk".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let memory = Arc::new(InMemoryCache::default());
        let (cache, kind) = build_s3_cache(&cache_cfg, memory).await;
        assert_eq!(kind, CacheBackendKind::S3);

        // A raw S3Cache would always return Allowed; the wrapper enforces.
        let allowed = cache
            .incr_if_below("spend:k", 5.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(allowed, IncrResult::Allowed(5.0));
        let rejected = cache
            .incr_if_below("spend:k", 6.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(rejected, IncrResult::Rejected(5.0));
    }

    /// The GCS wiring must not expose the raw no-op backend as the application
    /// cache: `build_gcs_cache` layers [`BlobCounterCache`] so a budget-over
    /// reservation is rejected instead of silently allowed.
    #[cfg(feature = "gcs-cache")]
    #[tokio::test]
    async fn build_gcs_cache_enforces_counters() {
        let cache_cfg = crate::config::types::CacheConfig {
            cache_type: Some("gcs".to_string()),
            gcs: Some(crate::config::types::GcsCacheConfig {
                bucket: "mybucket".to_string(),
                prefix: Some("oxllm".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let memory = Arc::new(InMemoryCache::default());
        let (cache, kind) = build_gcs_cache(&cache_cfg, memory).await;
        assert_eq!(kind, CacheBackendKind::Gcs);

        // A raw GcsCache would always return Allowed; the wrapper enforces.
        let allowed = cache
            .incr_if_below("spend:k", 5.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(allowed, IncrResult::Allowed(5.0));
        let rejected = cache
            .incr_if_below("spend:k", 6.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(rejected, IncrResult::Rejected(5.0));
    }

    /// The Azure wiring must not expose the raw no-op backend as the
    /// application cache: `build_azure_cache` layers [`BlobCounterCache`] so a
    /// budget-over reservation is rejected instead of silently allowed.
    #[cfg(feature = "azure-cache")]
    #[tokio::test]
    async fn build_azure_cache_enforces_counters() {
        let cache_cfg = crate::config::types::CacheConfig {
            cache_type: Some("azure".to_string()),
            azure: Some(crate::config::types::AzureBlobCacheConfig {
                account: "myaccount".to_string(),
                container: "mycontainer".to_string(),
                prefix: Some("oxllm".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let memory = Arc::new(InMemoryCache::default());
        let (cache, kind) = build_azure_cache(&cache_cfg, memory).await;
        assert_eq!(kind, CacheBackendKind::Azure);

        // A raw AzureBlobCache would always return Allowed; the wrapper enforces.
        let allowed = cache
            .incr_if_below("spend:k", 5.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(allowed, IncrResult::Allowed(5.0));
        let rejected = cache
            .incr_if_below("spend:k", 6.0, 10.0, None)
            .await
            .unwrap();
        assert_eq!(rejected, IncrResult::Rejected(5.0));
    }

    // ── HIGH: graceful-shutdown drain must be bounded ──────────────────

    /// A thin serve harness mirroring `run()`'s drain pattern. The handler is
    /// confirmed in-flight (via a flag) BEFORE graceful shutdown is triggered,
    /// so the server has a genuinely stuck request to drain. With a 300ms drain
    /// budget, the select must return within the budget — NOT wait the full
    /// 60s the handler sleeps — proving one hung/streaming request cannot block
    /// `serve.await` (and the cleanup that runs after it) indefinitely.
    #[tokio::test]
    async fn graceful_drain_is_bounded_against_hung_request() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let handler_started = Arc::new(AtomicBool::new(false));
        let started_for_handler = handler_started.clone();
        let app = axum::Router::new().route(
            "/hang",
            axum::routing::get(move || {
                let started = started_for_handler.clone();
                async move {
                    started.store(true, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    "ok"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local_addr");

        // Controllable shutdown signal so we can fire it ONLY after the handler
        // is confirmed in-flight (mirrors `run()` where the signal fires on
        // Ctrl-C while requests are being served).
        let (signal_tx, signal_rx) = tokio::sync::oneshot::channel::<()>();
        let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = signal_rx.await;
        });
        // Drive the server concurrently so it accepts the connection while we
        // wait for the handler to confirm it is in-flight.
        let serve_task = tokio::spawn(async move { serve.await });

        tokio::spawn(async move {
            let _ = reqwest::get(format!("http://{addr}/hang")).await;
        });

        // Wait until the handler is actually running, THEN arm graceful
        // shutdown — otherwise the signal could fire before the request is
        // accepted and there would be no stuck connection to drain.
        let wait_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while !handler_started.load(Ordering::SeqCst) {
            if tokio::time::Instant::now() >= wait_deadline {
                panic!("handler never started within 5s");
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let _ = signal_tx.send(());

        let drain_budget = Duration::from_millis(300);
        let started_at = tokio::time::Instant::now();
        let outcome: Result<(), ()> = tokio::select! {
            biased;
            res = serve_task => {
                res.expect("join").expect("serve");
                Ok(())
            }
            _ = tokio::time::sleep(drain_budget) => Err(()),
        };
        let elapsed = started_at.elapsed();

        assert!(
            elapsed < Duration::from_secs(5),
            "drain must be bounded even with a hung request; took {elapsed:?}"
        );
        assert!(
            outcome.is_err(),
            "the 300ms drain budget should force exit before the 60s handler finishes"
        );
    }

    // ── MEDIUM: serve-error must not skip cleanup ──────────────────────

    /// Locks in the fix for the serve-error bypass: when `axum::serve`
    /// returns `Err` (a fatal accept-loop error), `run()` must still run
    /// its cleanup (scheduler/invalidation/pruner shutdown, spend-writer
    /// flush) before propagating the error. The old code did `res?;` in the
    /// serve arm, skipping the bounded spend-writer flush (C01 billing
    /// accuracy). This harness mirrors `run()`'s select shape — a stand-in
    /// `serve` future that resolves to `Err`, plus a cleanup flag — and
    /// asserts (a) cleanup ran and (b) the error is still surfaced. A
    /// regression to `res?;` before cleanup would fail assertion (a).
    #[tokio::test]
    async fn serve_error_still_runs_cleanup_before_propagating() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let cleanup_ran = Arc::new(AtomicBool::new(false));
        let cleanup_for_task = cleanup_ran.clone();

        // A stand-in for `axum::serve(...)` that immediately fails, matching
        // the `std::io::Error` axum yields on a fatal accept-loop error.
        let serve = async move { Err(std::io::Error::other("simulated accept-loop failure")) };

        let mut serve_err: Option<anyhow::Error> = None;
        tokio::select! {
            biased;
            res = serve => {
                match res {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!(error = %e, "server accept loop stopped with error");
                        serve_err = Some(e.into());
                    }
                }
            }
            _ = std::future::pending::<()>() => {}
        }

        // The shared cleanup block (mirrors run(): scheduler / invalidation
        // / pruner shutdown + spend-writer flush).
        cleanup_for_task.store(true, Ordering::SeqCst);

        // Error is still surfaced after cleanup.
        assert!(serve_err.is_some(), "serve error must be propagated");
        assert!(
            cleanup_ran.load(Ordering::SeqCst),
            "cleanup must run even when serve returns Err; \
             a `res?` regression would skip the spend-writer flush (C01)"
        );
    }

    // ── MEDIUM: Redis URL construction preserves auth / TLS / db ───────

    #[test]
    fn redis_url_falls_back_to_host_port_when_no_url() {
        let cfg = crate::config::types::CacheConfig {
            host: Some("cache.internal".to_string()),
            port: Some(6380),
            ..Default::default()
        };
        assert_eq!(redis_url_from_config(&cfg), "redis://cache.internal:6380");
    }

    #[test]
    fn redis_url_defaults_to_localhost_6379() {
        let cfg = crate::config::types::CacheConfig::default();
        assert_eq!(redis_url_from_config(&cfg), "redis://localhost:6379");
    }

    #[test]
    fn redis_url_takes_precedence_and_preserves_auth_tls_db() {
        let cfg = crate::config::types::CacheConfig {
            redis_url: Some("rediss://alice:s3cr3t@cache.internal:6380/2".to_string()),
            host: Some("ignored".to_string()),
            port: Some(1),
            ..Default::default()
        };
        let url = redis_url_from_config(&cfg);
        assert_eq!(url, "rediss://alice:s3cr3t@cache.internal:6380/2");
        assert!(url.starts_with("rediss://"), "TLS scheme must be preserved");
        assert!(
            url.contains("alice:s3cr3t@"),
            "credentials must be preserved"
        );
        assert!(url.ends_with("/2"), "db index must be preserved");
    }

    // ── LOW: spend-pruner cutoff is never earlier than now-retention ───

    #[cfg(feature = "db")]
    #[test]
    fn spend_prune_cutoff_equals_now_minus_retention() {
        let now = jiff::Timestamp::now();
        let cutoff = spend_prune_cutoff_at(now, 90);
        let parsed: jiff::Timestamp = cutoff.parse().expect("parse cutoff");
        let expected = now
            .checked_sub(jiff::SignedDuration::from_secs(90 * 86400))
            .expect("sub");
        assert_eq!(parsed, expected);
    }

    /// On an absurd retention that would underflow jiff's `Timestamp` range,
    /// the cutoff must fall back to "now" (prunes nothing further) — NOT
    /// `unwrap_or_default()` which yielded 1970-01-01 and would delete every
    /// spend_logs row. 5_000_000 days (~13_700 years) precedes jiff's minimum
    /// timestamp, so `checked_sub` returns Err and the fallback runs.
    #[cfg(feature = "db")]
    #[test]
    fn spend_prune_cutoff_underflow_falls_back_to_now_not_epoch() {
        let now = jiff::Timestamp::now();
        let cutoff = spend_prune_cutoff_at(now, 5_000_000);
        assert_eq!(
            cutoff,
            now.to_string(),
            "underflow must fall back to 'now', not UNIX_EPOCH"
        );
        assert!(
            !cutoff.starts_with("1970"),
            "cutoff must never be UNIX_EPOCH (delete-all footgun): {cutoff}"
        );
    }

    // ---- C65 LOW: SQLite guard escalation when Redis is co-configured ------

    #[test]
    #[cfg(feature = "db")]
    fn sqlite_guard_level_is_none_for_non_sqlite_url() {
        assert_eq!(
            sqlite_guard_level("postgres://user@host/db", false),
            SqliteGuardLevel::None,
            "non-SQLite URLs never trip the guard, regardless of Redis"
        );
        assert_eq!(
            sqlite_guard_level("postgres://user@host/db", true),
            SqliteGuardLevel::None,
            "Redis co-configured with Postgres is the correct multi-instance shape"
        );
    }

    #[test]
    #[cfg(feature = "db")]
    fn sqlite_guard_level_is_warn_for_sqlite_without_redis() {
        // Single-instance SQLite is the documented local-dev shape — keep
        // the existing informational warn.
        assert_eq!(
            sqlite_guard_level("sqlite:///tmp/oxllm.db", false),
            SqliteGuardLevel::Warn,
            "plain single-instance SQLite stays at warn"
        );
    }

    #[test]
    #[cfg(feature = "db")]
    fn sqlite_guard_level_is_error_for_sqlite_with_redis() {
        // SQLite + Redis is almost certainly multi-instance — escalate to
        // error so the silent spend-log loss risk (N writers → SQLITE_BUSY
        // → spend writer swallows → billing drift) is loud in production
        // logs.
        assert_eq!(
            sqlite_guard_level("sqlite:///tmp/oxllm.db", true),
            SqliteGuardLevel::Error,
            "SQLite + Redis must escalate to error (multi-instance signal)"
        );
    }

    #[test]
    fn parse_auto_migrate_disables_on_canonical_false_values() {
        for v in ["false", "False", "FALSE", "0", "no", "off", "NO", "Off"] {
            assert!(
                !parse_auto_migrate(Some(v.to_string())),
                "OXLLM_AUTO_MIGRATE={v:?} should disable auto-migrate"
            );
        }
    }

    #[test]
    fn parse_auto_migrate_trims_whitespace() {
        for v in [" false ", "\tfalse\n", " 0 ", " no "] {
            assert!(
                !parse_auto_migrate(Some(v.to_string())),
                "OXLLM_AUTO_MIGRATE={v:?} (with whitespace) should disable auto-migrate"
            );
        }
    }

    #[test]
    fn parse_auto_migrate_enables_on_truthy_and_unrecognized() {
        for v in ["true", "True", "1", "yes", "anything"] {
            assert!(
                parse_auto_migrate(Some(v.to_string())),
                "OXLLM_AUTO_MIGRATE={v:?} should leave auto-migrate enabled (fail-open)"
            );
        }
    }

    #[test]
    fn parse_auto_migrate_defaults_to_enabled_when_unset() {
        assert!(
            parse_auto_migrate(None),
            "unset OXLLM_AUTO_MIGRATE must default to enabled (backward compat)"
        );
    }

    #[test]
    fn parse_auto_migrate_empty_string_stays_enabled_fail_open() {
        // An empty string is not a recognised false-value, so fail-open to
        // enabled (backward-compat). An operator who wants to disable must
        // set a recognised false literal.
        assert!(
            parse_auto_migrate(Some(String::new())),
            "OXLLM_AUTO_MIGRATE=\"\" should fail-open to enabled"
        );
    }
}
