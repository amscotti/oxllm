//! Generic HTTP telemetry hook.
//!
//! Five observability backends (Helicone, LangSmith, Datadog, W&B, MLflow)
//! share an identical skeleton: validate config, build a `reqwest::Client`,
//! and on each successful (non-cache, non-master) request fire-and-forget a
//! POST to a provider-specific endpoint carrying a provider-specific JSON
//! payload. Only two things vary per backend: the wire shape of the payload
//! and the auth header.
//!
//! [`HttpTelemetryHook`] captures that shared skeleton. Each concrete backend
//! supplies:
//!
//! - a fully-resolved `url` (endpoint + path),
//! - an optional `auth_header` — a `(header_name, header_value)` pair, where
//!   the value is already formatted as the backend expects (e.g.
//!   `"Bearer {key}"` for W&B), and
//! - a `build_payload` closure that maps a `(&CallContext, &ResponseSummary)`
//!   pair to the provider's JSON body. The closure is free to capture resolved
//!   config (project name, service name, tags, ...) so the base stays agnostic
//!   to per-backend config fields.
//!
//! ## Backpressure
//!
//! Each hook instance owns a bounded `mpsc` channel of capacity
//! `DEFAULT_CHANNEL_CAPACITY`. `post_success` enqueues the redacted payload
//! via `try_send` (never blocks the request path); a dedicated background
//! task drains the channel and POSTs sequentially. This bounds both resident
//! memory (the channel cannot exceed its capacity) and concurrent outbound
//! sockets (the single drainer keeps one in-flight POST per backend
//! instance). When the channel is full — e.g. the backend is slow or down —
//! the payload is dropped and `oxllm_telemetry_dropped_total{backend,
//! reason="full"}` is incremented so the loss is observable.
//!
//! Backends whose shape is too different (Langfuse batches + flushes; S3Log
//! uses PUT + SigV4) keep their own `Hook` impls and are not built on this
//! type, but follow the same bounded-channel pattern.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::Hook;
use oxllm_core::{CallContext, ResponseSummary, ServiceFailureEvent};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::hooks::redact::{redact_secret_values_in_place, redact_sensitive_in_place};

/// Timeout for the fire-and-forget POST. Matches the per-backend constants the
/// concrete hooks previously each declared independently.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// Bounded channel capacity for the per-instance background POST task. Sized
/// to absorb short bursts without dropping under steady-state load; when
/// exceeded, payloads are dropped and counted under
/// `oxllm_telemetry_dropped_total{backend=...,reason="full"}`.
pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 10_000;
const MAX_QUEUED_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;
const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

struct QueuedPayload {
    payload: Value,
    _memory: OwnedSemaphorePermit,
}

/// Normalise an optional, operator-supplied endpoint URL into the canonical
/// form that is actually POSTed to and that flows through SSRF validation.
///
/// `build_*_hook_opt` and `build()` previously ran slightly different
/// normalisation (one stripped trailing `/` only, the other also rejected
/// whitespace-only strings) so SSRF validation could run against a different
/// URL than the one ultimately used. This helper is the single source of
/// truth so SSRF validation and the live POST target always agree.
pub(crate) fn resolve_endpoint(endpoint: Option<&str>, default: &str) -> String {
    endpoint
        .map(str::trim)
        .map(|s| s.trim_end_matches('/'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

/// Provider-specific payload builder. Captures resolved config so the base
/// does not need to know each backend's config fields. Factored into an alias
/// to keep the struct field readable (and satisfy clippy::type_complexity).
pub type PayloadBuilder = dyn Fn(&CallContext, &ResponseSummary) -> Value + Send + Sync;

/// Optional failure payload. Unset backends stay silent on `post_failure`.
pub type FailurePayloadBuilder = dyn Fn(&CallContext, &oxllm_core::Error) -> Value + Send + Sync;

/// Optional infrastructure failure payload, independent of request callbacks.
pub type ServicePayloadBuilder = dyn Fn(&ServiceFailureEvent) -> Value + Send + Sync;

/// Generic HTTP telemetry hook — fire-and-forget POST to a configurable
/// endpoint. Each concrete backend provides a payload-builder closure.
///
/// See the module docs for the responsibilities split between this type and
/// the per-backend constructors.
pub struct HttpTelemetryHook {
    url: String,
    /// `(header_name, header_value)` sent on every POST, when the backend
    /// requires authentication. `None` for backends with no auth header.
    auth_header: Option<(String, String)>,
    /// Provider-specific payload builder. Captures resolved config so the base
    /// does not need to know each backend's config fields.
    build_payload: Box<PayloadBuilder>,
    /// When set, `post_failure` posts this payload. Other backends leave it
    /// unset and `supports_failure_events` stays false.
    build_failure: Option<Box<FailurePayloadBuilder>>,
    build_service: Option<Box<ServicePayloadBuilder>>,
    /// Bounded sender into the background POST task. Cheap to clone per
    /// request; `try_send` never blocks.
    tx: mpsc::Sender<QueuedPayload>,
    memory: Arc<Semaphore>,
    /// Backend slug, used for tracing and as the `backend` label on
    /// `oxllm_telemetry_dropped_total`.
    event_name: &'static str,
    /// When `true`, apply the opt-in value-pattern redaction pass in addition
    /// to the unconditional key-name denylist.
    redact_secret_values: bool,
}

impl std::fmt::Debug for HttpTelemetryHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpTelemetryHook")
            .field("event_name", &self.event_name)
            .field("url", &self.url)
            .field("has_auth", &self.auth_header.is_some())
            .finish()
    }
}

impl HttpTelemetryHook {
    /// Build a generic telemetry hook.
    ///
    /// `build_payload` may be a plain `fn` (when the payload needs no config)
    /// or a closure capturing resolved config. It is boxed internally so all
    /// concrete hooks share one storage type.
    pub fn new<F>(
        url: String,
        auth_header: Option<(String, String)>,
        build_payload: F,
        event_name: &'static str,
    ) -> anyhow::Result<Self>
    where
        F: Fn(&CallContext, &ResponseSummary) -> Value + Send + Sync + 'static,
    {
        Self::new_full(
            url,
            auth_header,
            build_payload,
            event_name,
            DEFAULT_CHANNEL_CAPACITY,
            false,
            false,
        )
    }

    /// Like [`HttpTelemetryHook::new`] but with an explicit channel
    /// `capacity`, letting tests trade memory for drop-tolerance based on
    /// expected throughput.
    pub fn with_capacity<F>(
        url: String,
        auth_header: Option<(String, String)>,
        build_payload: F,
        event_name: &'static str,
        capacity: usize,
    ) -> anyhow::Result<Self>
    where
        F: Fn(&CallContext, &ResponseSummary) -> Value + Send + Sync + 'static,
    {
        Self::new_full(
            url,
            auth_header,
            build_payload,
            event_name,
            capacity,
            false,
            false,
        )
    }

    /// Full constructor with all knobs: channel `capacity`, SSRF-aware client
    /// gated on `allow_private`, and optional value-pattern redaction gated on
    /// `redact_secret_values`. Used by the per-backend `build()` functions
    /// which read these flags from config.
    pub fn new_full<F>(
        url: String,
        auth_header: Option<(String, String)>,
        build_payload: F,
        event_name: &'static str,
        capacity: usize,
        allow_private: bool,
        redact_secret_values: bool,
    ) -> anyhow::Result<Self>
    where
        F: Fn(&CallContext, &ResponseSummary) -> Value + Send + Sync + 'static,
    {
        if url.trim().is_empty() {
            anyhow::bail!("telemetry hook url must not be empty");
        }
        let client = crate::security::build_ssrf_http_client(allow_private, REQUEST_TIMEOUT)?;
        let cap = if capacity == 0 {
            DEFAULT_CHANNEL_CAPACITY
        } else {
            capacity
        };
        let (tx, mut rx) = mpsc::channel::<QueuedPayload>(cap);
        let memory = Arc::new(Semaphore::new(MAX_QUEUED_PAYLOAD_BYTES));
        let drain_client = client;
        let drain_url = url.clone();
        let drain_auth = auth_header.clone();
        let drain_event = event_name;
        tokio::spawn(async move {
            while let Some(QueuedPayload { payload, _memory }) = rx.recv().await {
                let mut req = drain_client.post(&drain_url).json(&payload);
                if let Some((name, value)) = &drain_auth {
                    req = req.header(name, value);
                }
                if let Err(e) = req.send().await.and_then(|r| r.error_for_status()) {
                    tracing::warn!(
                        error = %e.without_url(),
                        url = %crate::security::sanitize_url(&drain_url),
                        "{drain_event} log request POST failed"
                    );
                    // Surface POST failures as a metric too — fire-and-forget
                    // telemetry is allowed to drop on transient failures, but
                    // the drop must be visible from dashboards/alerts. This is
                    // especially important for backends whose contract the
                    // gateway cannot fully honor from a single POST (e.g.
                    // MLflow `log-batch` targets a run that must pre-exist).
                    crate::metrics::record_telemetry_dropped(drain_event, "post_failed", 1);
                }
            }
        });
        Ok(Self {
            url,
            auth_header,
            build_payload: Box::new(build_payload),
            build_failure: None,
            build_service: None,
            tx,
            memory,
            event_name,
            redact_secret_values,
        })
    }

    /// Attach a failure payload. Only backends that call this emit failure
    /// events; the default stays silent.
    pub fn with_failure_payload<G>(mut self, build_failure: G) -> Self
    where
        G: Fn(&CallContext, &oxllm_core::Error) -> Value + Send + Sync + 'static,
    {
        self.build_failure = Some(Box::new(build_failure));
        self
    }

    /// Attach a service-failure payload to the same bounded telemetry queue.
    pub fn with_service_payload<G>(mut self, build_service: G) -> Self
    where
        G: Fn(&ServiceFailureEvent) -> Value + Send + Sync + 'static,
    {
        self.build_service = Some(Box::new(build_service));
        self
    }

    /// The configured POST URL (for tests).
    pub fn url(&self) -> &str {
        &self.url
    }
}

#[async_trait]
impl Hook for HttpTelemetryHook {
    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        // Cache hits and master-key requests are never logged — the request
        // never reached (or was billed against) an upstream provider.
        if ctx.cache_hit || ctx.is_master {
            return;
        }
        self.enqueue_payload((self.build_payload)(ctx, resp));
    }

    async fn post_failure(&self, ctx: &CallContext, err: &oxllm_core::Error) {
        if ctx.cache_hit || ctx.is_master {
            return;
        }
        let Some(build) = self.build_failure.as_ref() else {
            return;
        };
        self.enqueue_payload(build(ctx, err));
    }

    fn supports_failure_events(&self) -> bool {
        self.build_failure.is_some()
    }

    async fn post_service_failure(&self, event: &ServiceFailureEvent) {
        if let Some(build) = &self.build_service {
            self.enqueue_payload(build(event));
        }
    }

    fn supports_service_events(&self) -> bool {
        self.build_service.is_some()
    }
}

impl HttpTelemetryHook {
    fn enqueue_payload(&self, mut payload: Value) {
        // Defense-in-depth: strip well-known credential-bearing keys from the
        // payload before it leaves the process. Langsmith/Helicone/W&B/
        // Datadog/MLflow all ship request/response bodies to third-party
        // SaaS; an operator who enables one of them alongside Langfuse
        // reasonably expects parity in redaction. Per-backend payload
        // builders may embed the request body verbatim (langsmith `inputs`,
        // helicone `messages`, ...), so redacting here is the single
        // choke point that covers every backend built on this hook.
        redact_sensitive_in_place(&mut payload);
        if self.redact_secret_values {
            redact_secret_values_in_place(&mut payload);
        }
        // Hand the payload to the dedicated background POST task via a
        // bounded channel. `try_send` never blocks the request path; on a
        // full/closed channel the payload is dropped and counted under
        // `oxllm_telemetry_dropped_total` so the loss is observable. This
        // bounds resident memory (channel capacity) and concurrent
        // in-flight POSTs (single drainer) per backend instance.
        if !crate::cache::semantic::json_fits(&payload, MAX_PAYLOAD_BYTES) {
            crate::metrics::record_telemetry_dropped(self.event_name, "oversized", 1);
            return;
        }
        let size = serde_json::to_vec(&payload)
            .map(|bytes| bytes.len().max(256))
            .unwrap_or(MAX_PAYLOAD_BYTES);
        let Ok(memory) = self.memory.clone().try_acquire_many_owned(size as u32) else {
            crate::metrics::record_telemetry_dropped(self.event_name, "full", 1);
            return;
        };
        match self.tx.try_send(QueuedPayload {
            payload,
            _memory: memory,
        }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(
                    backend = self.event_name,
                    "telemetry channel full; dropping payload"
                );
                crate::metrics::record_telemetry_dropped(self.event_name, "full", 1);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!(
                    backend = self.event_name,
                    "telemetry channel closed; dropping payload"
                );
                crate::metrics::record_telemetry_dropped(self.event_name, "closed", 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;
    use serde_json::json;
    /// Build a hook pointed at `url` with the given auth header and a trivial
    /// payload that echoes the model. Payload builders needing config would
    /// close over it; here we exercise the plain-`fn` path.
    fn build_hook(url: String, auth_header: Option<(&str, &str)>) -> HttpTelemetryHook {
        let auth = auth_header.map(|(n, v)| (n.to_string(), v.to_string()));
        HttpTelemetryHook::new(url, auth, test_payload, "test-backend").expect("build")
    }

    fn test_payload(_ctx: &CallContext, resp: &ResponseSummary) -> Value {
        json!({ "model": resp.model })
    }

    fn make_ctx() -> CallContext {
        CallContext::for_test("gpt-4o")
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

    #[test]
    fn rejects_empty_url() {
        let result = HttpTelemetryHook::new(String::new(), None, test_payload, "x");
        assert!(result.is_err());
    }

    /// LOW #12 (endpoint normalization): `resolve_endpoint` is the single
    /// source of truth that both `build()` and `build_*_hook_opt()` consult,
    /// so SSRF validation runs against exactly the URL that will be POSTed
    /// to. Pin the four input shapes that previously diverged between the
    /// two call sites (None, whitespace-only, trailing slash, valid URL).
    #[test]
    fn resolve_endpoint_normalises_consistently() {
        // None → default.
        assert_eq!(
            resolve_endpoint(None, "https://def.example.com"),
            "https://def.example.com"
        );
        // Whitespace-only → default (NOT a malformed URL that SSRF sees and
        // reqwest later rejects).
        assert_eq!(
            resolve_endpoint(Some("   "), "https://def.example.com"),
            "https://def.example.com"
        );
        // Empty → default.
        assert_eq!(
            resolve_endpoint(Some(""), "https://def.example.com"),
            "https://def.example.com"
        );
        // Trailing slash trimmed.
        assert_eq!(
            resolve_endpoint(Some("https://x.example.com/"), "https://def.example.com"),
            "https://x.example.com"
        );
        // Leading/trailing whitespace trimmed.
        assert_eq!(
            resolve_endpoint(Some("  https://x.example.com  "), "https://def.example.com"),
            "https://x.example.com"
        );
        // Valid URL passes through verbatim (modulo trailing slash).
        assert_eq!(
            resolve_endpoint(Some("https://api.example.com"), "https://def.example.com"),
            "https://api.example.com"
        );
    }

    #[tokio::test]
    async fn url_is_exposed() {
        let hook = build_hook("https://example.test/log".to_string(), None);
        assert_eq!(hook.url(), "https://example.test/log");
    }

    #[tokio::test]
    async fn post_success_skips_cache_hit() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let hook = build_hook(server.base_url(), None);

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/");
            then.status(200);
        });

        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        hook.post_success(&ctx, &make_response()).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(post_mock.hits(), 0, "cache hits must not be logged");
    }

    #[tokio::test]
    async fn post_success_skips_master_key() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let hook = build_hook(server.base_url(), None);

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/");
            then.status(200);
        });

        let mut ctx = make_ctx();
        ctx.is_master = true;
        hook.post_success(&ctx, &make_response()).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            post_mock.hits(),
            0,
            "master-key requests must not be logged"
        );
    }

    #[tokio::test]
    async fn post_success_does_not_block_request_on_post_failure() {
        let hook = build_hook("http://127.0.0.1:1".to_string(), None);
        let started = std::time::Instant::now();
        hook.post_success(&make_ctx(), &make_response()).await;
        assert!(
            started.elapsed() < Duration::from_millis(150),
            "post_success must return immediately even when endpoint is unreachable"
        );
    }

    #[tokio::test]
    async fn post_success_posts_payload_with_auth_header() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let hook = build_hook(server.base_url(), Some(("X-Test-Key", "secret")));

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/")
                .header("X-Test-Key", "secret")
                .body_contains("\"model\":\"gpt-4o\"");
            then.status(200);
        });

        hook.post_success(&make_ctx(), &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "payload should be POSTed with the configured auth header"
        );
    }

    /// The payload builder may close over config and is invoked per request.
    /// This guards the `Box<dyn Fn>` plumbing against regressions that would
    /// only surface for config-capturing backends (Datadog, W&B, ...).
    #[tokio::test]
    async fn post_success_invokes_closure_capturing_config() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let service = "my-svc".to_string();
        let hook = HttpTelemetryHook::new(
            server.base_url(),
            None,
            move |_ctx, resp| json!({ "service": service, "model": resp.model }),
            "test-backend",
        )
        .expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/")
                .body_contains("\"service\":\"my-svc\"")
                .body_contains("\"model\":\"gpt-4o\"");
            then.status(200);
        });

        hook.post_success(&make_ctx(), &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(post_mock.hits() >= 1, "captured config must reach the wire");
    }

    /// HIGH #2: every backend built on `HttpTelemetryHook` (Langsmith,
    /// Helicone, Datadog, MLflow, W&B) ships request/response bodies to
    /// third-party SaaS. Well-known credential-bearing keys must be redacted
    /// before the payload leaves the process — parity with the Langfuse hook.
    #[tokio::test]
    async fn post_success_redacts_sensitive_keys_in_payload() {
        use httpmock::MockServer;
        use oxllm_core::UsageSummary;
        use serde_json::json;
        use std::sync::Arc;

        let server = MockServer::start();
        // A payload builder that (like Langsmith/Helicone) embeds the request
        // body verbatim — the worst case for leakage.
        let hook = HttpTelemetryHook::new(
            server.base_url(),
            None,
            |ctx, _resp| json!({ "inputs": (*ctx.request_body).clone() }),
            "test-backend",
        )
        .expect("build");

        let post_mock_redacted = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/")
                .body_contains("\"authorization\":\"[redacted]\"")
                .body_contains("\"api_key\":\"[redacted]\"");
            then.status(200);
        });
        let post_mock_no_leak = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/")
                .body_contains("sk-secret");
            then.status(200);
        });

        let ctx = CallContext {
            request_body: Arc::new(json!({
                "model": "gpt-4o",
                "authorization": "Bearer sk-secret",
                "api_key": "sk-secret",
                "messages": [{"role": "user", "content": "hi"}],
            })),
            ..CallContext::for_test("gpt-4o")
        };
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
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
        };
        hook.post_success(&ctx, &resp).await;

        for _ in 0..40 {
            if post_mock_redacted.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            post_mock_redacted.hits() >= 1,
            "sensitive keys must arrive redacted at the backend"
        );
        assert_eq!(
            post_mock_no_leak.hits(),
            0,
            "raw secret value must never reach the backend"
        );
    }

    /// HIGH #1: when the backend is slow (or down), in-flight payloads must
    /// be bounded by the channel capacity — overflow is dropped and counted
    /// under `oxllm_telemetry_dropped_total{backend=...,reason="full"}` —
    /// instead of accumulating one `tokio::spawn` per request. With a
    /// deliberately tiny capacity we can drive overflow deterministically
    /// without a real load test.
    #[tokio::test]
    async fn post_success_drops_and_counts_when_channel_full() {
        use httpmock::MockServer;
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let server = MockServer::start();
        let url = server.base_url();

        // A mock with a small delay so the single drainer cannot keep up
        // with a burst against a capacity-2 channel, forcing overflow.
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/");
            then.delay(Duration::from_millis(50)).status(200);
        });

        // Side-effect-counting payload builder: proves every `post_success`
        // call ran the builder exactly once (drops happen afterwards).
        let tracker = StdArc::new(AtomicU32::new(0));
        let tracker_for_closure = tracker.clone();

        let hook = HttpTelemetryHook::with_capacity(
            url,
            None,
            move |_ctx, resp| {
                tracker_for_closure.fetch_add(1, Ordering::SeqCst);
                serde_json::json!({ "model": resp.model })
            },
            "bp-backend",
            2,
        )
        .expect("build");

        let _ = crate::metrics::install();

        // Fire a burst larger than the channel capacity while the drainer is
        // still blocked on its first (delayed) POST. `post_success` must not
        // block; the overflow must be dropped, not spawned.
        let ctx = make_ctx();
        let resp = make_response();
        let n: u64 = 50;
        let started = std::time::Instant::now();
        for _ in 0..n {
            hook.post_success(&ctx, &resp).await;
        }
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "post_success burst must not block even when the channel is full"
        );

        // At least (n - capacity) payloads must have been dropped and counted.
        let min_dropped = n.saturating_sub(2);
        let mut dropped = 0u64;
        for _ in 0..80 {
            dropped = telemetry_dropped_value("bp-backend", "full");
            if dropped >= min_dropped {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            dropped >= min_dropped,
            "expected at least {min_dropped} dropped payloads on overflow, got {dropped}"
        );
        assert_eq!(
            tracker.load(Ordering::SeqCst) as u64,
            n,
            "payload builder must be invoked once per post_success call"
        );
    }

    fn telemetry_dropped_value(backend: &str, reason: &str) -> u64 {
        let needle = format!("backend=\"{backend}\",reason=\"{reason}\"");
        for line in crate::metrics::render().lines() {
            if line.starts_with("oxllm_telemetry_dropped_total{") && line.contains(&needle) {
                if let Some(n) = line.rsplit(' ').next() {
                    if let Ok(v) = n.parse::<u64>() {
                        return v;
                    }
                }
            }
        }
        0
    }
}
