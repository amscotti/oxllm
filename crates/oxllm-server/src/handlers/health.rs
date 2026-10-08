//! `GET /health` and `GET /health/readiness`.

use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde_json::{Map, Value, json};

use crate::state::SharedState;

/// Per-backend ping deadline. A stalled DB or cache backend must not hang the
/// readiness probe — orchestrators and monitors rely on it returning promptly.
const PING_TIMEOUT: Duration = Duration::from_millis(500);

/// Liveness probe. Returns `200 {"status":"healthy"}` while serving, and
/// `503 {"status":"shutting_down"}` once graceful shutdown has begun — mirroring
/// the reference `/health/liveliness`, which consults
/// `GracefulShutdownManager.is_shutting_down()` so the kubelet gets a clean 503
/// (instead of a probe timeout) and stops routing new traffic into a draining
/// pod (`proxy/health_endpoints/_health_endpoints.py:1555-1571`).
pub async fn health(State(shared): State<SharedState>) -> Response {
    let state = shared.load_full();
    if state.is_shutting_down() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "shutting_down" })),
        )
            .into_response();
    }
    Json(json!({ "status": "healthy" })).into_response()
}

/// Readiness probe. Pings every configured backend (DB, cache) and returns
/// `"healthy"` only when all are reachable. On any failure returns 503 with a
/// per-check breakdown.
///
/// Once graceful shutdown has begun it short-circuits to
/// `503 {"status":"shutting_down"}` (without pinging backends), mirroring
/// the reference `/health/readiness`
/// (`proxy/health_endpoints/_health_endpoints.py:1659-1669`).
///
/// The DB and cache pings run concurrently (`tokio::join!`) so the worst-case
/// latency is `PING_TIMEOUT`, not `2 * PING_TIMEOUT`. The cache check consults
/// [`crate::state::AppState::cache_backend`] (the backend that was *actually
/// constructed*) rather than the config string, so a startup fallback (Redis
/// unreachable → `InMemoryCache`) is reported as `"error"`, not masked as a
/// healthy `"ok"`. The per-check label is likewise derived from the live
/// backend kind, so a hot-reload that changed `cache.type` while the backend
/// `Arc` survived does not mislabel the check.
pub async fn readiness(State(shared): State<SharedState>) -> Response {
    let state = shared.load_full();

    if state.is_shutting_down() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "shutting_down" })),
        )
            .into_response();
    }

    // Resolve the cache check up front: whether a remote backend was requested
    // and, if so, whether the runtime cache is actually that remote backend.
    // When config asks for a remote backend but `cache_backend` is `Memory`,
    // the startup `build_cache` fell back to in-memory — the configured
    // backend is degraded and must surface as an error, not a silent "ok".
    let cache_check = state
        .config
        .oxllm_settings
        .cache
        .as_ref()
        .and_then(|cache_cfg| {
            let requested_remote = matches!(
                cache_cfg.cache_type.as_deref(),
                Some("redis" | "s3" | "gcs" | "azure")
            );
            if requested_remote {
                // Label the check by the backend actually being pinged. When
                // the live backend is a real remote, name it; when it fell
                // back to `Memory`, no ping happens so fall back to the
                // config-requested type so an operator sees which configured
                // backend is degraded (e.g. `"redis": "error"`).
                let check_name = state
                    .cache_backend
                    .label()
                    .map(str::to_string)
                    .unwrap_or_else(|| {
                        cache_cfg
                            .cache_type
                            .as_deref()
                            .unwrap_or("cache")
                            .to_string()
                    });
                let actual_remote = state.cache_backend.is_remote();
                Some((check_name, actual_remote))
            } else {
                None
            }
        });

    let cache = state.cache.clone();

    #[cfg(feature = "db")]
    let db = state.db.clone();
    #[cfg(feature = "db")]
    let db_configured = state.config.general_settings.database_url.is_some();

    let cache_future = async {
        match cache_check.as_ref().map(|(_, actual)| *actual) {
            Some(true) => tokio::time::timeout(PING_TIMEOUT, cache.ping())
                .await
                .unwrap_or_default(),
            Some(false) => false,
            None => true,
        }
    };

    #[cfg(feature = "db")]
    let db_future = async {
        match &db {
            Some(db) => tokio::time::timeout(PING_TIMEOUT, db.ping())
                .await
                .unwrap_or_default(),
            // DB was configured but failed to initialise (migration error
            // or connection failure in `setup_persistence`). Report
            // unhealthy so a load balancer pulls the replica out of
            // rotation instead of routing traffic to a degraded, billing-
            // blind proxy.
            None if db_configured => false,
            // DB was never configured — not a health concern.
            None => true,
        }
    };

    // Run the DB and cache pings concurrently so the worst-case latency is
    // one PING_TIMEOUT, not two.
    #[cfg(feature = "db")]
    let (db_ok, cache_ok) = tokio::join!(db_future, cache_future);
    #[cfg(not(feature = "db"))]
    let cache_ok = cache_future.await;

    let mut checks: Map<String, Value> = Map::new();
    let mut all_healthy = true;

    #[cfg(feature = "db")]
    if db.is_some() || db_configured {
        checks.insert("db".to_string(), json!(if db_ok { "ok" } else { "error" }));
        all_healthy = all_healthy && db_ok;
    }

    if let Some((check_name, _)) = &cache_check {
        checks.insert(
            check_name.clone(),
            json!(if cache_ok { "ok" } else { "error" }),
        );
        all_healthy = all_healthy && cache_ok;
    }

    let status = if all_healthy { "healthy" } else { "unhealthy" };
    let body = json!({ "status": status, "checks": checks });
    let code = if all_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(body)).into_response()
}

type Response = axum::response::Response;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
    use crate::state::AppState;
    use async_trait::async_trait;
    use oxllm_core::Cache as CacheTrait;
    use std::time::Duration;

    struct FailCache;

    #[async_trait]
    impl CacheTrait for FailCache {
        async fn get(&self, _key: &str) -> Option<String> {
            None
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(
            &self,
            _key: &str,
            _by: f64,
            _ttl: Option<Duration>,
        ) -> Result<f64, oxllm_core::Error> {
            Ok(0.0)
        }
        async fn ping(&self) -> bool {
            false
        }
    }

    /// A cache stub whose `ping` reports a reachable backend. Used to exercise
    /// the readiness happy path honestly (a remote backend that responds ok),
    /// rather than leaning on the default in-memory cache while labelling it
    /// "redis" — which proved nothing about the redis reachability branch.
    struct OkCache;

    #[async_trait]
    impl CacheTrait for OkCache {
        async fn get(&self, _key: &str) -> Option<String> {
            None
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(
            &self,
            _key: &str,
            _by: f64,
            _ttl: Option<Duration>,
        ) -> Result<f64, oxllm_core::Error> {
            Ok(0.0)
        }
        async fn ping(&self) -> bool {
            true
        }
    }

    fn config_no_backends() -> ProxyConfig {
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
            general_settings: GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    fn make_shared(config: ProxyConfig) -> SharedState {
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(
            AppState::with_registry(config, oxllm_providers::default_registry()).expect("state"),
        ))
    }

    fn make_shared_with_cache(
        config: ProxyConfig,
        cache: std::sync::Arc<dyn CacheTrait>,
    ) -> SharedState {
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(
            AppState::with_registry_and_cache(config, oxllm_providers::default_registry(), cache)
                .expect("state"),
        ))
    }

    /// Build state with an explicit cache backend kind, simulating the
    /// production path where [`crate::app::build_cache`] records what was
    /// *actually constructed* (which may diverge from the config string when a
    /// remote backend falls back to in-memory).
    fn make_shared_with_cache_backend(
        config: ProxyConfig,
        cache: std::sync::Arc<dyn CacheTrait>,
        backend: crate::state::CacheBackendKind,
    ) -> SharedState {
        let state = crate::state::StateBuilder::new(config)
            .with_registry(oxllm_providers::default_registry())
            .with_cache(cache)
            .with_cache_backend(backend)
            .build()
            .expect("state");
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    /// Build state with an explicit cache, backend kind, and shutdown flag —
    /// the three inputs the production bootstrap threads in. Used to exercise
    /// the graceful-shutdown branch of the probes without driving the real
    /// signal handlers.
    fn make_shared_with_cache_backend_and_shutdown(
        config: ProxyConfig,
        cache: std::sync::Arc<dyn CacheTrait>,
        backend: crate::state::CacheBackendKind,
        shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> SharedState {
        let state = crate::state::StateBuilder::new(config)
            .with_registry(oxllm_providers::default_registry())
            .with_cache(cache)
            .with_cache_backend(backend)
            .with_shutdown(shutdown)
            .build()
            .expect("state");
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    #[tokio::test]
    async fn readiness_healthy_when_no_backends_configured() {
        let shared = make_shared(config_no_backends());
        let resp = readiness(State(shared)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_returns_200_when_serving() {
        let shared = make_shared(config_no_backends());
        let resp = health(State(shared)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["status"], "healthy");
    }

    #[tokio::test]
    async fn readiness_reports_redis_check_when_configured() {
        let mut config = config_no_backends();
        config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
            cache_type: Some("redis".to_string()),
            host: Some("127.0.0.1".to_string()),
            port: Some(1),
            semantic_cache: None,
            ..Default::default()
        });
        let shared = make_shared_with_cache(config, std::sync::Arc::new(FailCache));
        let resp = readiness(State(shared)).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["status"], "unhealthy");
        assert_eq!(v["checks"]["redis"], "error");
    }

    #[tokio::test]
    async fn readiness_reports_blob_cache_check_when_configured() {
        // The readiness probe must ping blob-store backends (s3/gcs/azure),
        // not just Redis. Previously the cache ping was gated on a literal
        // `cache_type == "redis"` and silently skipped the blob backends,
        // yielding a false "healthy" when the object store was unreachable.
        for ty in ["s3", "gcs", "azure"] {
            let mut config = config_no_backends();
            config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
                cache_type: Some(ty.to_string()),
                ..Default::default()
            });
            let shared = make_shared_with_cache(config, std::sync::Arc::new(FailCache));
            let resp = readiness(State(shared)).await;
            assert_eq!(
                resp.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{ty} cache failure should flip readiness to 503"
            );
            let body = axum::body::to_bytes(resp.into_body(), 4096)
                .await
                .expect("body");
            let v: Value = serde_json::from_slice(&body).expect("json");
            assert_eq!(v["status"], "unhealthy", "{ty} should be unhealthy");
            assert_eq!(v["checks"][ty], "error", "{ty} check should report error");
        }
    }

    #[tokio::test]
    async fn readiness_times_out_on_stalled_cache_ping() {
        // A stalled backend must not hang the readiness probe indefinitely;
        // the ping deadline flips the check to `error` promptly.
        struct SlowCache;
        #[async_trait]
        impl CacheTrait for SlowCache {
            async fn get(&self, _key: &str) -> Option<String> {
                None
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                _by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, oxllm_core::Error> {
                Ok(0.0)
            }
            async fn ping(&self) -> bool {
                tokio::time::sleep(Duration::from_secs(5)).await;
                true
            }
        }

        let mut config = config_no_backends();
        config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
            cache_type: Some("redis".to_string()),
            ..Default::default()
        });
        let shared = make_shared_with_cache(config, std::sync::Arc::new(SlowCache));

        let start = tokio::time::Instant::now();
        let resp = readiness(State(shared)).await;
        let elapsed = start.elapsed();

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        // The deadline is 500ms; the stalled 5s ping must not be awaited.
        assert!(
            elapsed < Duration::from_secs(2),
            "readiness returned in {elapsed:?}, should be bounded by the ping deadline"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["checks"]["redis"], "error");
    }

    /// Exercises the readiness happy path for a remote backend whose `ping`
    /// succeeds. Previously this was named
    /// `readiness_healthy_when_redis_configured_but_reachable` and routed
    /// through `make_shared`, which (being synchronous, no I/O) defaulted the
    /// cache to a fresh `InMemoryCache` and derived the backend kind from the
    /// config string — so it never touched Redis and only proved the
    /// builder-default path, giving false confidence that real Redis
    /// reachability was covered. It now injects an explicit `OkCache` stub
    /// with `CacheBackendKind::Redis` so the assertion is honest about what is
    /// being exercised: the readiness logic path where a real remote backend
    /// responds ok. Coverage of actual Redis connectivity belongs in an
    /// integration test against a live Redis (not attempted here).
    #[tokio::test]
    async fn readiness_healthy_when_remote_backend_ping_succeeds() {
        let mut config = config_no_backends();
        config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
            cache_type: Some("redis".to_string()),
            ..Default::default()
        });
        let shared = make_shared_with_cache_backend(
            config,
            std::sync::Arc::new(OkCache),
            crate::state::CacheBackendKind::Redis,
        );
        let resp = readiness(State(shared)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["status"], "healthy");
        assert_eq!(v["checks"]["redis"], "ok");
    }

    /// When `build_cache` falls back to `InMemoryCache` at startup because Redis
    /// was unreachable, the config string still says `"redis"` but
    /// `cache_backend` is `Memory`. The readiness probe must surface this as a
    /// degraded backend (`"redis": "error"` → 503), NOT report a false-positive
    /// `"redis": "ok"` from pinging the in-memory fallback.
    #[tokio::test]
    async fn readiness_reports_error_when_redis_fell_back_to_memory() {
        let mut config = config_no_backends();
        config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
            cache_type: Some("redis".to_string()),
            host: Some("127.0.0.1".to_string()),
            port: Some(1),
            ..Default::default()
        });
        // cache_backend = Memory simulates the fallback: config says redis,
        // but build_cache could not connect and returned an InMemoryCache.
        let shared = make_shared_with_cache_backend(
            config,
            std::sync::Arc::new(crate::cache::InMemoryCache::default()),
            crate::state::CacheBackendKind::Memory,
        );
        let resp = readiness(State(shared)).await;
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a Redis fallback must not report readiness 200"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["status"], "unhealthy");
        assert_eq!(v["checks"]["redis"], "error");
    }

    /// When `build_cache` successfully connected to Redis (cache_backend =
    /// Redis), the readiness probe pings the real cache and reports its result.
    #[tokio::test]
    async fn readiness_pings_real_redis_when_backend_kind_is_redis() {
        let mut config = config_no_backends();
        config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
            cache_type: Some("redis".to_string()),
            ..Default::default()
        });
        // FailCache with backend = Redis: actual_remote is true, so the probe
        // pings FailCache → false → error.
        let shared = make_shared_with_cache_backend(
            config,
            std::sync::Arc::new(FailCache),
            crate::state::CacheBackendKind::Redis,
        );
        let resp = readiness(State(shared)).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["checks"]["redis"], "error");
    }

    /// End-to-end through the real [`crate::app::build_cache`]: when a blob
    /// backend is requested (`cache.type: "s3"`/`"gcs"`/`"azure"`) but
    /// `build_cache` falls back to in-memory (missing sub-config / construction
    /// error / feature flag off), it must record `CacheBackendKind::Memory` so
    /// the readiness probe reports a degraded backend instead of masking it as
    /// a healthy `"ok"`. Previously the blob match arms in `build_cache`
    /// unconditionally returned the blob kind while the actual cache object
    /// was a plain `InMemoryCache` (whose `ping` is a no-op `true`), yielding a
    /// false-positive `"s3": "ok"` with HTTP 200. Iterates all three blob
    /// backends; under the default feature set (no `*-cache` features) each
    /// builder takes the feature-off fallback path.
    #[tokio::test]
    async fn readiness_reports_error_when_blob_build_falls_back_to_memory() {
        for ty in ["s3", "gcs", "azure"] {
            let mut config = config_no_backends();
            config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
                cache_type: Some(ty.to_string()),
                // No `cache.s3`/`gcs`/`azure` sub-config → fallback.
                ..Default::default()
            });
            let memory = std::sync::Arc::new(crate::cache::InMemoryCache::default());
            let (cache, backend, _) = crate::app::build_cache(&config, memory)
                .await
                .expect("cache build");
            assert_eq!(
                backend,
                crate::state::CacheBackendKind::Memory,
                "{ty} fallback must record CacheBackendKind::Memory, not the blob kind"
            );
            let shared = make_shared_with_cache_backend(config, cache, backend);
            let resp = readiness(State(shared)).await;
            assert_eq!(
                resp.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{ty} fallback must flip readiness to 503"
            );
            let body = axum::body::to_bytes(resp.into_body(), 4096)
                .await
                .expect("body");
            let v: Value = serde_json::from_slice(&body).expect("json");
            assert_eq!(v["status"], "unhealthy");
            assert_eq!(
                v["checks"][ty], "error",
                "{ty} fallback must surface as a degraded check, not a healthy ok"
            );
        }
    }

    /// Once graceful shutdown begins, both probes must return
    /// `503 {"status":"shutting_down"}` immediately — mirroring the reference
    /// `is_shutting_down()` semantics so the kubelet gets a clean 503 instead
    /// of a probe timeout. Readiness must short-circuit before pinging any
    /// backend (the `checks` map is absent). Verified by flipping the same
    /// `Arc<AtomicBool>` that `app::run` flips in its drain path.
    #[tokio::test]
    async fn health_and_readiness_return_503_when_shutting_down() {
        use std::sync::atomic::AtomicBool;

        let shutdown = std::sync::Arc::new(AtomicBool::new(false));
        let config = config_no_backends();
        let shared = make_shared_with_cache_backend_and_shutdown(
            config,
            std::sync::Arc::new(OkCache),
            crate::state::CacheBackendKind::Redis,
            shutdown.clone(),
        );

        // Pre-flip: both probes are healthy.
        assert_eq!(health(State(shared.clone())).await.status(), StatusCode::OK);
        assert_eq!(
            readiness(State(shared.clone())).await.status(),
            StatusCode::OK
        );

        // Flip the shutdown flag, exactly as `app::run`'s drain path does.
        shutdown.store(true, std::sync::atomic::Ordering::Release);

        let resp = health(State(shared.clone())).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["status"], "shutting_down");

        let resp = readiness(State(shared.clone())).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["status"], "shutting_down");
        // Readiness short-circuited before pinging backends: no `checks` map.
        assert!(
            v.get("checks")
                .and_then(|c| c.as_object())
                .map(|o| o.is_empty())
                .unwrap_or(true),
            "readiness must not emit per-check breakdown while shutting down"
        );
    }

    /// LOW: the readiness check label follows the live backend kind, not the
    /// (possibly stale) config string. Simulates a hot-reload that changed
    /// `cache.type` from `s3` to `redis` while the S3 backend `Arc` was
    /// preserved (cache infrastructure survives a config swap): config says
    /// `redis`, `cache_backend` is still `S3`. The check key must be `"s3"`
    /// (the backend actually being pinged), not the stale `"redis"`.
    #[tokio::test]
    async fn readiness_check_name_follows_backend_not_config_after_reload() {
        let mut config = config_no_backends();
        config.oxllm_settings.cache = Some(crate::config::types::CacheConfig {
            cache_type: Some("redis".to_string()),
            ..Default::default()
        });
        let shared = make_shared_with_cache_backend(
            config,
            std::sync::Arc::new(OkCache),
            crate::state::CacheBackendKind::S3,
        );
        let resp = readiness(State(shared)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert!(
            v["checks"].get("s3").is_some(),
            "check key must follow the live S3 backend, not the stale config"
        );
        assert!(
            v["checks"].get("redis").is_none(),
            "stale config string must not label the check"
        );
    }

    /// C68 MEDIUM: when `database_url` is configured but the DB handle is
    /// absent (migration failure / connection failure in
    /// `setup_persistence`), the readiness probe must report `db: "error"`
    /// and return 503 — not silently treat the missing handle as "no DB
    /// needed" (which routes traffic to a degraded, billing-blind replica).
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn readiness_reports_db_error_when_configured_but_unavailable() {
        let mut config = config_no_backends();
        config.general_settings.database_url = Some("postgres://unreachable/db".to_string());
        // No `.with_db(...)` call: simulates the degraded path where
        // setup_persistence returned (None, None, None).
        let shared = make_shared(config);
        let resp = readiness(State(shared)).await;
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "configured-but-failed DB must flip readiness to 503"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["status"], "unhealthy");
        assert_eq!(v["checks"]["db"], "error");
    }

    /// C68: the readiness probe must NOT report a DB error when no database
    /// was configured at all — the absence of a DB is a valid deployment
    /// shape (master-key-only, no persistence), not a degraded state.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn readiness_omits_db_check_when_not_configured() {
        let shared = make_shared(config_no_backends());
        let resp = readiness(State(shared)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert!(
            v["checks"]
                .as_object()
                .is_none_or(|o| !o.contains_key("db")),
            "no DB check should be emitted when database_url is absent"
        );
    }
}
