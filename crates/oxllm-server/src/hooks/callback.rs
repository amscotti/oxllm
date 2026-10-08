//! Generic HTTP callback hook.
//!
//! A [`Hook`] that POSTs spend events to a configurable HTTP endpoint on
//! every successful upstream response. The fire-and-forget POST happens via
//! a dedicated background task fed by a bounded `mpsc` channel: `post_success`
//! enqueues the payload via `try_send` (never blocks the request path) and
//! the drainer POSTs sequentially. This bounds resident memory (channel
//! capacity) and concurrent in-flight sockets (single drainer) per callback
//! instance. When the channel is full the payload is dropped and counted
//! under `oxllm_telemetry_dropped_total{backend="callback",reason="full"}`;
//! POST failures are logged at `warn` level and never propagate to the
//! request path.
//!
//! Wired in from `oxllm_settings.success_callback` / `callback_settings.generic`.
//!
//! ```yaml
//! oxllm_settings:
//! success_callback: ["generic"]
//! callback_settings:
//! generic:
//! url: https://my-analytics.example.com/spend
//! headers:
//! Authorization: Bearer my-token
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::Hook;
use oxllm_core::{CallContext, ResponseSummary};
use serde::Serialize;
use tokio::sync::mpsc;

use crate::hooks::cost_fallback::fallback_cost;

/// Bounded channel capacity for the per-callback background POST task.
pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 10_000;

/// Payload POSTed to the configured endpoint on each successful response.
#[derive(Debug, Serialize)]
pub struct CallbackPayload {
    pub event: &'static str,
    pub model: String,
    pub provider: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cost: Option<f64>,
    pub request_id: String,
    pub key_id: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
}

/// Static configuration for the generic HTTP callback (the YAML block under
/// `callback_settings.generic`).
#[derive(Debug, Clone)]
pub struct CallbackConfig {
    /// Fully-qualified URL to POST to.
    pub url: String,
    /// Extra headers to attach to each POST (e.g. `Authorization`).
    pub headers: HashMap<String, String>,
}

/// A `Hook` that POSTs a spend event to a configured HTTP endpoint on every
/// successful response.
///
/// The POST is dispatched via a dedicated background task fed by a bounded
/// `mpsc` channel so the request path is never blocked or failed by the
/// callback. Errors are logged at `warn` level; dropped payloads (channel
/// full) are counted under `oxllm_telemetry_dropped_total{backend="callback"}`.
#[derive(Clone)]
pub struct GenericHttpCallback {
    config: Arc<CallbackConfig>,
    /// Bounded sender into the background POST task.
    tx: mpsc::Sender<CallbackPayload>,
}

impl std::fmt::Debug for GenericHttpCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericHttpCallback")
            .field("url", &self.config.url)
            .finish()
    }
}

impl GenericHttpCallback {
    /// Build a callback that posts to `url` with no extra headers and a
    /// sensible default HTTP client (5s timeout, no redirects).
    pub fn new(url: impl Into<String>) -> anyhow::Result<Self> {
        Self::with_config(CallbackConfig {
            url: url.into(),
            headers: HashMap::new(),
        })
    }

    /// Build a callback from a full config block.
    pub fn with_config(config: CallbackConfig) -> anyhow::Result<Self> {
        Self::with_config_capacity(config, DEFAULT_CHANNEL_CAPACITY, false)
    }

    /// Like [`GenericHttpCallback::with_config`] but with an explicit channel
    /// `capacity`, letting tests trade memory for drop-tolerance.
    pub fn with_config_capacity(
        config: CallbackConfig,
        capacity: usize,
        allow_private: bool,
    ) -> anyhow::Result<Self> {
        let client =
            crate::security::build_ssrf_http_client(allow_private, Duration::from_secs(5))?;
        let cap = if capacity == 0 {
            DEFAULT_CHANNEL_CAPACITY
        } else {
            capacity
        };
        let (tx, mut rx) = mpsc::channel::<CallbackPayload>(cap);
        let config = Arc::new(config);
        let drain_config = config.clone();
        tokio::spawn(async move {
            while let Some(payload) = rx.recv().await {
                if let Err(e) = post_payload(&client, &drain_config, &payload).await {
                    tracing::warn!(
                        error = %e.without_url(),
                        url = %crate::security::sanitize_url(&drain_config.url),
                        "generic callback POST failed"
                    );
                    crate::metrics::record_telemetry_dropped("callback", "post_failed", 1);
                }
            }
        });
        Ok(Self { config, tx })
    }

    /// The configured endpoint URL (for inspection / tests).
    pub fn url(&self) -> &str {
        &self.config.url
    }

    fn build_payload(&self, ctx: &CallContext, resp: &ResponseSummary) -> CallbackPayload {
        // U13: key callback configuration
        // (including credentials) was copied into logging metadata. oxllm is
        // safe by design: `CallContext` exposes only `api_key_hash` (a SHA-256
        // digest, never the raw key) plus non-secret identifiers (key_id,
        // team_id, user_id). The callback config itself (`CallbackConfig`:
        // url + headers) is never forwarded into the payload — only the
        // headers are used to *send* the POST, never logged. `ResponseSummary`
        // carries model/cost/usage/provider, none of which are credentials.
        let usage = resp.usage.as_ref();
        CallbackPayload {
            event: "spend",
            model: resp.model.clone(),
            provider: resp.provider.clone(),
            prompt_tokens: usage.map(|u| u.prompt_tokens),
            completion_tokens: usage.map(|u| u.completion_tokens),
            total_tokens: usage.map(|u| u.total_tokens),
            cost: fallback_cost(ctx, resp),
            request_id: ctx.request_id.clone(),
            key_id: ctx.key_id.clone(),
            team_id: ctx.team_id.clone(),
            user_id: ctx.user_id.clone(),
        }
    }

    /// Enqueue the payload onto the bounded channel for the background POST
    /// task. Never blocks: on a full/closed channel the payload is dropped
    /// and counted under `oxllm_telemetry_dropped_total{backend="callback"}`.
    fn dispatch(&self, payload: CallbackPayload) {
        match self.tx.try_send(payload) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!("callback channel full; dropping payload");
                crate::metrics::record_telemetry_dropped("callback", "full", 1);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!("callback channel closed; dropping payload");
                crate::metrics::record_telemetry_dropped("callback", "closed", 1);
            }
        }
    }
}

async fn post_payload(
    client: &reqwest::Client,
    config: &CallbackConfig,
    payload: &CallbackPayload,
) -> Result<(), reqwest::Error> {
    let mut req = client.post(&config.url).json(payload);
    for (k, v) in &config.headers {
        if let Ok(value) = reqwest::header::HeaderValue::from_str(v) {
            if let Ok(name) = reqwest::header::HeaderName::from_bytes(k.as_bytes()) {
                req = req.header(name, value);
            }
        }
    }
    req.send().await?.error_for_status()?;
    Ok(())
}

#[async_trait]
impl Hook for GenericHttpCallback {
    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        if ctx.cache_hit || ctx.is_master {
            return;
        }
        let payload = self.build_payload(ctx, resp);
        self.dispatch(payload);
    }

    async fn post_failure(&self, ctx: &CallContext, _error: &oxllm_core::Error) {
        if ctx.cache_hit || ctx.is_master {
            return;
        }
        self.dispatch(CallbackPayload {
            event: "failure",
            model: ctx.model.clone(),
            provider: ctx
                .pricing_model
                .as_deref()
                .and_then(|model| model.split_once('/').map(|(provider, _)| provider))
                .unwrap_or("")
                .to_string(),
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cost: None,
            request_id: ctx.request_id.clone(),
            key_id: ctx.key_id.clone(),
            team_id: ctx.team_id.clone(),
            user_id: ctx.user_id.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;

    fn make_ctx() -> CallContext {
        CallContext {
            key_id: Some("k-1".to_string()),
            team_id: Some("team-1".to_string()),
            ..CallContext::for_test("gpt-4o")
        }
    }

    fn make_response() -> ResponseSummary {
        ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: Some(0.0025),
            usage: Some(UsageSummary {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    async fn build_payload_carries_usage_and_identity() {
        let hook = GenericHttpCallback::new("https://example.com/spend").expect("build");
        let ctx = make_ctx();
        let resp = make_response();
        let payload = hook.build_payload(&ctx, &resp);
        assert_eq!(payload.event, "spend");
        assert_eq!(payload.model, "gpt-4o");
        assert_eq!(payload.provider, "openai");
        assert_eq!(payload.prompt_tokens, Some(100));
        assert_eq!(payload.completion_tokens, Some(50));
        assert_eq!(payload.total_tokens, Some(150));
        assert_eq!(payload.cost, Some(0.0025));
        assert_eq!(payload.request_id, "req-1");
        assert_eq!(payload.key_id.as_deref(), Some("k-1"));
        assert_eq!(payload.team_id.as_deref(), Some("team-1"));
    }

    // ---- U13: callback payload must never carry credentials -----------------
    //
    // The reference copied key callback config (including
    // credentials) into logging metadata. oxllm's `CallbackPayload` carries
    // only non-secret spend fields. This test serializes the payload to JSON
    // and asserts no credential-bearing key is present, pinning the invariant
    // so a future field addition cannot accidentally leak secrets.

    #[tokio::test]
    async fn u13_payload_json_contains_no_credential_fields() {
        let hook = GenericHttpCallback::with_config(CallbackConfig {
            url: "https://example.com/spend".to_string(),
            headers: HashMap::from([
                (
                    "Authorization".to_string(),
                    "Bearer secret-token".to_string(),
                ),
                ("X-Api-Key".to_string(), "sk-confidential".to_string()),
            ]),
        })
        .expect("build");
        let ctx = make_ctx();
        let resp = make_response();
        let payload = hook.build_payload(&ctx, &resp);
        let json = serde_json::to_string(&payload).expect("serialize");

        // Credential-bearing keys must never appear in the serialized payload.
        for forbidden in [
            "api_key",
            "apikey",
            "authorization",
            "secret",
            "password",
            "credential",
            "headers",
            "callback_config",
            "oxllm_params",
            "api_base",
        ] {
            assert!(
                !json.to_lowercase().contains(forbidden),
                "U13 regression: payload JSON must not contain '{forbidden}', got: {json}"
            );
        }
        // The configured callback headers (used only to *send* the POST) must
        // never leak into the payload body.
        assert!(
            !json.contains("secret-token") && !json.contains("sk-confidential"),
            "U13 regression: callback header credentials leaked into payload: {json}"
        );
    }

    #[tokio::test]
    async fn build_payload_computes_cost_when_response_omits_it() {
        let hook = GenericHttpCallback::new("https://example.com/spend").expect("build");
        let ctx = make_ctx();
        let resp = ResponseSummary {
            cost: None,
            ..make_response()
        };
        let payload = hook.build_payload(&ctx, &resp);
        assert!(payload.cost.unwrap_or(-1.0) > 0.0, "cost should be derived");
    }

    #[tokio::test]
    async fn url_inspector_returns_configured_value() {
        let hook = GenericHttpCallback::new("https://example.com/spend").expect("build");
        assert_eq!(hook.url(), "https://example.com/spend");
    }

    #[tokio::test]
    async fn callback_fires_on_success_httpmock() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/spend";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/spend");
            then.status(202);
        });

        let hook = GenericHttpCallback::new(&url).expect("build");
        let ctx = make_ctx();
        let resp = make_response();
        hook.post_success(&ctx, &resp).await;

        for _ in 0..20 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "callback should have POSTed at least once"
        );
    }

    #[tokio::test]
    async fn callback_does_not_block_request_on_post_failure() {
        let hook =
            GenericHttpCallback::new("http://127.0.0.1:1/this-port-is-closed").expect("build");
        let ctx = make_ctx();
        let resp = make_response();
        let started = std::time::Instant::now();
        hook.post_success(&ctx, &resp).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(100),
            "post_success must return immediately even when the endpoint is unreachable (took {elapsed:?})"
        );
    }

    #[tokio::test]
    async fn cache_hit_skips_callback() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/spend";
        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/spend");
            then.status(202);
        });

        let hook = GenericHttpCallback::new(&url).expect("build");
        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        let resp = make_response();
        hook.post_success(&ctx, &resp).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            post_mock.hits(),
            0,
            "cache hits should not fire the callback"
        );
    }

    #[tokio::test]
    async fn master_key_skips_callback() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/spend";
        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/spend");
            then.status(202);
        });

        let hook = GenericHttpCallback::new(&url).expect("build");
        let mut ctx = make_ctx();
        ctx.is_master = true;
        let resp = make_response();
        hook.post_success(&ctx, &resp).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            post_mock.hits(),
            0,
            "master-key requests should not fire the callback"
        );
    }

    #[tokio::test]
    async fn callback_attaches_configured_headers() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/spend";
        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/spend")
                .header("Authorization", "Bearer my-token")
                .header("X-Custom", "yes");
            then.status(202);
        });

        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer my-token".to_string());
        headers.insert("X-Custom".to_string(), "yes".to_string());
        let hook = GenericHttpCallback::with_config(CallbackConfig {
            url: url.clone(),
            headers,
        })
        .expect("build");

        let ctx = make_ctx();
        let resp = make_response();
        hook.post_success(&ctx, &resp).await;

        for _ in 0..20 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "callback with headers should have hit"
        );
    }
}
