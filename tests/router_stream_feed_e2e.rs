//! Integration tests: the streaming host → router-strategy data feed.
//!
//! The latency/usage/in-flight stores must see streaming traffic (not just
//! unary): TTFB latency and the RPM snapshot are recorded at headers time, the
//! in-flight slot stays reserved for the whole stream lifetime (`LeastBusy`
//! counts active streams), and a mid-stream transport failure reaches
//! allowed-fails / percentage cooldown / health — a deployment whose streams
//! consistently die mid-body must not stay permanently healthy.

mod common;

use axum::body::Body;
use axum::body::Bytes;
use axum::http::header;
use common::start_oxllm;
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings, RoutingKnobs,
};
use serde_json::json;
use std::time::Duration;

/// Minimal OpenAI-compatible streaming SSE chunk the proxy forwards verbatim.
const CHUNK_1: &str = "data: {\"id\":\"chatcmpl-s1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n";

/// Slow streaming upstream: sends one chunk, then a second after `delay`.
async fn sse_delayed_response(delay: Duration) -> axum::response::Response {
    use futures_util::StreamExt;
    use futures_util::stream;
    let first = Bytes::from_static(CHUNK_1.as_bytes());
    let done = Bytes::from_static(b"data: [DONE]\n\n");
    let stream =
        stream::iter(vec![Ok::<Bytes, std::io::Error>(first)]).chain(stream::once(async move {
            tokio::time::sleep(delay).await;
            Ok::<Bytes, std::io::Error>(done)
        }));
    axum::response::Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(stream))
        .expect("response builds")
}

/// Streaming upstream that sends 200 + one SSE chunk, then aborts the
/// connection mid-body (the body stream errors) — simulating a connection
/// reset / upstream crash after headers were received. The 300ms delay lets
/// the 200 headers + first chunk flush to the proxy before the abort, so the
/// proxy sees a genuine mid-stream failure (not a connect-time one).
async fn sse_abort_response() -> axum::response::Response {
    use futures_util::StreamExt;
    use futures_util::stream;
    let chunk = Bytes::from_static(CHUNK_1.as_bytes());
    let stream =
        stream::once(async { Ok::<Bytes, Box<dyn std::error::Error + Send + Sync>>(chunk) }).chain(
            stream::once(async {
                tokio::time::sleep(Duration::from_millis(300)).await;
                Err::<Bytes, Box<dyn std::error::Error + Send + Sync>>(
                    "upstream connection aborted".into(),
                )
            }),
        );
    axum::response::Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(stream))
        .expect("response builds")
}

/// Spawn a bare axum upstream on an ephemeral port serving `/chat/completions`
/// with the given route handler. Returns the base URL.
async fn spawn_upstream(route: axum::routing::MethodRouter) -> String {
    let app = axum::Router::new().route("/chat/completions", route);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    format!("http://{addr}")
}

fn config_with(
    api_base: &str,
    routing: RoutingKnobs,
    router_settings: RouterSettings,
) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(api_base.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(10),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            routing,
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings,
        extra: serde_json::Map::new(),
    }
}

/// The stable per-deployment key the stores are fed with (matches the
/// router's `cooldown_key`).
fn store_key(config: &ProxyConfig) -> String {
    let dep: oxllm_core::Deployment = config.model_list[0].clone().into();
    oxllm_core::router::stable_key(&dep)
}

async fn send_streaming(handle: &common::OxllmHandle) -> reqwest::Response {
    reqwest::Client::new()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
        }))
        .send()
        .await
        .expect("request should succeed")
}

async fn wait_until<F>(mut check: F, timeout: Duration, what: &str)
where
    F: FnMut() -> bool,
{
    let deadline = std::time::Instant::now() + timeout;
    while !check() {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn streaming_response_feeds_latency_and_usage() {
    // Slow-but-complete stream: the TTFB is recorded at headers time, the
    // usage snapshot (dep_rpm) at the same point.
    let upstream = spawn_upstream(axum::routing::post(move || {
        sse_delayed_response(Duration::from_millis(50))
    }))
    .await;
    let config = config_with(
        &upstream,
        RoutingKnobs {
            routing_strategy: Some("lowest_latency".to_string()),
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, state) = start_oxllm(config.clone()).await.expect("server starts");

    let resp = send_streaming(&handle).await;
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("stream body");
    assert!(text.contains("[DONE]"), "stream must complete: {text}");

    let key = store_key(&config);
    wait_until(
        || state.load_full().router.latency_store().p50(&key).is_some(),
        Duration::from_secs(3),
        "latency sample for the streaming deployment",
    )
    .await;
    // Usage snapshot (dep_rpm) must be present even when the counter is 0.
    assert!(
        state.load_full().router.usage_store().contains(&key),
        "streaming response must feed the usage store"
    );
    handle.shutdown();
}

#[tokio::test]
async fn stream_holds_in_flight_until_body_completes() {
    let upstream = spawn_upstream(axum::routing::post(move || {
        sse_delayed_response(Duration::from_millis(400))
    }))
    .await;
    let config = config_with(
        &upstream,
        RoutingKnobs {
            routing_strategy: Some("least_busy".to_string()),
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, state) = start_oxllm(config.clone()).await.expect("server starts");
    let key = store_key(&config);

    let resp = send_streaming(&handle).await;
    assert_eq!(resp.status(), 200);

    // While the upstream is still streaming (the body has not drained), the
    // in-flight slot the chat loop reserved must still be held: `LeastBusy`
    // must see active streams, not just connect-time.
    wait_until(
        || state.load_full().router.usage_store().in_flight(&key) > 0,
        Duration::from_secs(3),
        "in-flight slot reserved across the active stream",
    )
    .await;

    let text = resp.text().await.expect("stream body");
    assert!(text.contains("[DONE]"));

    // After the stream completes, the slot is released.
    wait_until(
        || state.load_full().router.usage_store().in_flight(&key) == 0,
        Duration::from_secs(3),
        "in-flight slot released after the stream completes",
    )
    .await;
    handle.shutdown();
}

#[tokio::test]
async fn failed_stream_before_headers_releases_in_flight_slot() {
    // A streaming request whose upstream errors BEFORE any 2xx headers must
    // not leak the in-flight slot the chat loop reserved at pick time (there
    // is no stream task to release it).
    let upstream = spawn_upstream(axum::routing::post(|| async {
        axum::response::Response::builder()
            .status(500)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                "{\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}",
            ))
            .expect("response builds")
    }))
    .await;
    let config = config_with(
        &upstream,
        RoutingKnobs::default(),
        RouterSettings::default(),
    );
    let (handle, state) = start_oxllm(config.clone()).await.expect("server starts");
    let key = store_key(&config);

    let resp = send_streaming(&handle).await;
    assert_eq!(resp.status(), 500);
    wait_until(
        || state.load_full().router.usage_store().in_flight(&key) == 0,
        Duration::from_secs(3),
        "in-flight slot released after a pre-headers stream failure",
    )
    .await;
    handle.shutdown();
}

#[tokio::test]
async fn mid_stream_disconnect_accounts_failure_and_cools() {
    // The upstream returns 200 headers + one SSE chunk, then aborts. With
    // `allowed_fails: 0` (exclusive `>` semantics: tolerate 0, cool on the
    // 1st) the first mid-stream failure must cool the deployment
    // (allowed-fails accounting), and with health-check routing enabled the
    // connection failure must mark it unhealthy.
    let upstream = spawn_upstream(axum::routing::post(sse_abort_response)).await;
    let config = config_with(
        &upstream,
        RoutingKnobs {
            allowed_fails: Some(0),
            ..Default::default()
        },
        RouterSettings {
            enable_health_check_routing: Some(true),
            health_check_failure_threshold: Some(1),
            ..Default::default()
        },
    );
    let (handle, state) = start_oxllm(config.clone()).await.expect("server starts");
    let key = store_key(&config);

    let resp = send_streaming(&handle).await;
    assert_eq!(resp.status(), 200);
    // The proxy surfaces the mid-stream failure as a streaming error chunk
    // followed by [DONE] (no silent-success path).
    let text = resp.text().await.expect("stream body");
    assert!(
        text.contains("\"error\"") && text.contains("[DONE]"),
        "mid-stream failure must be surfaced to the client: {text}"
    );

    wait_until(
        || state.load_full().router.cooldown_store().is_cooling(&key),
        Duration::from_secs(3),
        "deployment cooled after the mid-stream failure",
    )
    .await;

    let binding = state.load_full();
    let health = binding.router.health_store();
    assert!(
        !health.is_healthy(&key, Duration::from_secs(60)),
        "a mid-body connection reset must degrade health routing state"
    );

    // A subsequent request must not be routed to the cooled deployment.
    let resp2 = send_streaming(&handle).await;
    assert_ne!(
        resp2.status(),
        200,
        "a cooled deployment must not serve the next request"
    );
    handle.shutdown();
}

#[tokio::test]
async fn allowed_fails_zero_cools_on_first_mid_stream_failure() {
    // Exclusive `>` parity: with `allowed_fails: 0`, the first mid-stream
    // failure (count=1) exceeds the limit (1 > 0) and cools the deployment.
    // This is the streaming-path equivalent of the unit-level
    // failure-policy transition-point test, which pins the
    // exact (count, limit) transition points for N∈{0,1,2,3} against the
    // `count > limit` comparison. The unit test covers the multi-failure
    // threshold precisely; this e2e test verifies the streaming wiring
    // (TTFB → 200 → mid-body abort → failure recording → cooldown) under
    // the new semantics.
    let upstream = spawn_upstream(axum::routing::post(sse_abort_response)).await;
    let config = config_with(
        &upstream,
        RoutingKnobs {
            allowed_fails: Some(0),
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, state) = start_oxllm(config.clone()).await.expect("server starts");
    let key = store_key(&config);

    let resp = send_streaming(&handle).await;
    assert_eq!(resp.status(), 200);
    let _ = resp.text().await.expect("stream body");
    wait_until(
        || state.load_full().router.cooldown_store().is_cooling(&key),
        Duration::from_secs(3),
        "with allowed_fails: 0, the first mid-stream failure must cool \
         (LiteLLM `>` semantics: count=1 > limit=0)",
    )
    .await;
    handle.shutdown();
}
