//! Shared harness for workspace-level integration tests.
//!
//! Integration tests under `tests/` are compiled as separate crates, so
//! duplication across files (`OxllmHandle`, `start_oxllm`, …) is the rule
//! rather than the exception. This module captures the common shape once;
//! new tests opt in via `mod common; use common::…;` and existing tests can
//! be migrated incrementally.
//!
//! Currently provided:
//!
//! - [`OxllmHandle`] — bound address + shutdown signal for a started proxy.
//! - [`start_oxllm`] — boots oxllm in-process against a caller-supplied
//!   [`ProxyConfig`] (no DB). Returns the handle and the live
//!   [`SharedState`] for direct inspection of router / cache internals.
//! - [`proxy_config`] — minimal one-deployment OpenAI-compatible config
//!   pointing at a mock upstream base URL.
//!
//! Tests that need a DB-backed stack (management endpoints, virtual keys,
//! spend logs) should keep their own `start_oxllm_with_db` helper for now —
//! a DB-aware variant can land here once the migration demonstrates the
//! pattern.

// Not every test pulls in every helper from this module; surface area is
// defined by what is *potentially* useful, not what every file consumes.
#![allow(dead_code)]

use oxllm_server::OxllmParams;
use oxllm_server::SharedState;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};

/// Client for local integration tests. The total deadline covers response
/// bodies/SSE consumption as well as the initial headers, so a stalled test
/// server cannot leave the test process waiting indefinitely.
pub fn http_client() -> reqwest::Client {
    http_client_with_timeout(std::time::Duration::from_secs(30))
}

/// Live/reasoning-provider tests can use a longer, still finite deadline.
pub fn http_client_with_timeout(timeout: std::time::Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(timeout)
        .build()
        .expect("test HTTP client")
}

/// Find a Prometheus sample by metric name and test-controlled labels. Label
/// order is intentionally irrelevant; HELP lines are not samples.
pub fn metric_value(text: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    text.lines().find_map(|line| {
        let (series, value) = line.split_once(' ')?;
        if series != name && !series.starts_with(&format!("{name}{{")) {
            return None;
        }
        if !labels
            .iter()
            .all(|(key, value)| series.contains(&format!("{key}=\"{value}\"")))
        {
            return None;
        }
        value.split_whitespace().next()?.parse().ok()
    })
}

/// An unset/empty Redis URL opts out of Redis integration tests. Once one is
/// configured, connection/ping failures are test failures, never passing skips.
/// Each invocation gets a private namespace so parallel tests/runs cannot
/// consume one another's pub/sub invalidations or cache entries.
pub async fn connect_redis(prefix: &str) -> Option<oxllm_cache_redis::RedisCache> {
    use oxllm_core::Cache as _;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_NAMESPACE: AtomicUsize = AtomicUsize::new(0);
    let url = match std::env::var("OXLLM_REDIS_URL") {
        Ok(value) if !value.trim().is_empty() => value.trim().to_string(),
        Ok(_) | Err(std::env::VarError::NotPresent) => return None,
        Err(std::env::VarError::NotUnicode(_)) => panic!("OXLLM_REDIS_URL must be valid Unicode"),
    };
    let prefix = format!(
        "{prefix}:{}:{}",
        std::process::id(),
        NEXT_NAMESPACE.fetch_add(1, Ordering::Relaxed)
    );
    let cache = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let cache = oxllm_cache_redis::RedisCache::new(&url, &prefix)
            .await
            .expect("OXLLM_REDIS_URL is configured but Redis connection failed");
        assert!(
            cache.ping().await,
            "OXLLM_REDIS_URL is configured but Redis ping failed"
        );
        cache
    })
    .await
    .expect("configured Redis connect/ping timed out after 5 seconds");
    Some(cache)
}

/// Handle to a running in-process oxllm proxy.
///
/// Dropping this value does NOT shut the server down — call
/// [`OxllmHandle::shutdown`] (or send on `shutdown_tx`) to stop the Axum
/// task. The address is logged for convenience; tests format URLs with
/// [`OxllmHandle::url`].
pub struct OxllmHandle {
    /// The bound listener address. Use [`OxllmHandle::url`] to build paths.
    pub addr: std::net::SocketAddr,
    /// Send `()` to trigger graceful shutdown of the Axum serve task.
    pub shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    /// Build a fully-qualified URL for a path on this proxy.
    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    /// Signal the spawned serve task to shut down. Idempotent.
    pub fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Boot oxllm in-process against `config` with no DB backing.
///
/// Spawns an Axum `serve` task on a free port and returns immediately.
/// The returned [`OxllmHandle`] owns the shutdown signal; the
/// [`SharedState`] is the live atomic-swap handle so tests can inspect
/// router state (cooldowns, latency samples, etc.) after a request.
pub async fn start_oxllm(config: ProxyConfig) -> anyhow::Result<(OxllmHandle, SharedState)> {
    let shared = build_shared_state(config)?;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let router = build_router(shared.clone());

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });

    Ok((OxllmHandle { addr, shutdown_tx }, shared))
}

/// Build a one-deployment OpenAI-compatible [`ProxyConfig`] pointing at the
/// supplied `mock_base_url` (typically a `httpmock::MockServer::base_url`).
///
/// The deployment uses the model name `gpt-4o`, API key `sk-test-key`, and
/// no auth (open mode). Callers that need master-key auth or per-deployment
/// overrides should construct their own `ProxyConfig` — this helper is just
/// the most common shape.
pub fn proxy_config(mock_base_url: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}
