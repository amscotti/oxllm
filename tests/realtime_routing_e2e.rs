//! Socket regressions for shared realtime/chat routing, handshake health and
//! cooldown accounting. Hard-task-abort barriers live beside the private
//! realtime lifecycle guards in oxllm-server.

mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::ws::{Message as AxumMessage, WebSocketUpgrade};
use axum::extract::{OriginalUri, Query};
use futures_util::{SinkExt, StreamExt};
use oxllm_core::router::deployment_key;
use oxllm_server::config::types::{ConfigModel, ProxyConfig, RoutingKnobs};
use oxllm_server::state::AppState;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::{CloseFrame, frame::coding::CloseCode};

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct Backend {
    base: String,
    observations: mpsc::UnboundedReceiver<Value>,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn start_backend(name: &'static str) -> Backend {
    let (tx, observations) = mpsc::unbounded_channel();
    let ws_tx = tx.clone();
    let app = axum::Router::new()
        .route(
            "/gateway/realtime",
            axum::routing::get(
                move |upgrade: WebSocketUpgrade,
                      OriginalUri(uri): OriginalUri,
                      Query(query): Query<std::collections::HashMap<String, String>>| {
                    let tx = ws_tx.clone();
                    async move {
                        upgrade.on_upgrade(move |mut socket| async move {
                            let _ = tx.send(json!({"type":"realtime","uri":uri.to_string()}));
                            socket.send(AxumMessage::Text(json!({
                                "type":"session.created", "backend":name,
                                "selector":query.get("model")
                            }).to_string().into())).await.unwrap();
                            while let Some(Ok(message)) = socket.next().await {
                                match message {
                                    AxumMessage::Text(text) => {
                                        let event = serde_json::from_str::<Value>(&text).ok();
                                        if event.as_ref().is_some_and(|event| event["type"] == "response.cancel") {
                                            let _ = socket.send(AxumMessage::Close(None)).await;
                                            break;
                                        }
                                    }
                                    AxumMessage::Close(_) => break,
                                    _ => {}
                                }
                            }
                        })
                    }
                },
            ),
        )
        .route(
            "/gateway/chat/completions",
            axum::routing::post(move |Json(body): Json<Value>| {
                let tx = tx.clone();
                async move {
                    let _ = tx.send(json!({"type":"chat","model":body["model"]}));
                    Json(json!({
                        "id":format!("chat-{name}"),"object":"chat.completion","created":1,
                        "model":"gpt-4o", "choices":[{
                            "index":0,"message":{"role":"assistant","content":name},
                            "finish_reason":"stop"
                        }],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
                    }))
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/gateway", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Backend {
        base,
        observations,
        server,
    }
}

fn routing_config(bases: &[&str]) -> ProxyConfig {
    let mut config = common::proxy_config(bases[0]);
    config.model_list = bases
        .iter()
        .map(|base| ConfigModel {
            model_name: "rt-shared".into(),
            oxllm_params: oxllm_server::OxllmParams {
                model: "openai/gpt-4o".into(),
                api_key: Some("test-key".into()),
                api_base: Some((*base).into()),
                ..Default::default()
            },
            ..Default::default()
        })
        .collect();
    config.router_settings.routing = RoutingKnobs {
        routing_strategy: Some("least-busy".into()),
        allowed_fails: Some(0),
        cooldown_time: Some(60),
        ..Default::default()
    };
    config.router_settings.enable_health_check_routing = Some(true);
    config.router_settings.health_check_failure_threshold = Some(1);
    config
}

async fn open_realtime(handle: &common::OxllmHandle, model: &str) -> ClientSocket {
    tokio::time::timeout(
        Duration::from_secs(3),
        tokio_tungstenite::connect_async(url_for_realtime(handle, model)),
    )
    .await
    .unwrap()
    .unwrap()
    .0
}

fn url_for_realtime(handle: &common::OxllmHandle, model: &str) -> String {
    let mut url = reqwest::Url::parse(&format!("ws://{}/v1/realtime", handle.addr)).unwrap();
    url.query_pairs_mut().append_pair("model", model);
    url.to_string()
}

async fn receive_json(client: &mut ClientSocket) -> Value {
    let frame = tokio::time::timeout(Duration::from_secs(3), client.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    serde_json::from_str(frame.to_text().unwrap()).unwrap()
}

async fn receive_close(client: &mut ClientSocket, expected_code: u16) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(Ok(message)) = client.next().await {
            if let Message::Close(close) = message {
                assert_eq!(
                    close.map(|close| u16::from(close.code)).unwrap_or(1000),
                    expected_code
                );
                return;
            }
        }
        panic!("proxy closed without the expected WebSocket close frame");
    })
    .await
    .unwrap();
}

async fn wait_idle(state: &AppState, key: &str) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while state.router.usage_store().in_flight(key) != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("socket teardown must release router occupancy");
}

#[tokio::test]
async fn open_realtime_sessions_move_chat_and_realtime_to_the_other_least_busy_backend() {
    let mut a = start_backend("a").await;
    let mut b = start_backend("b").await;
    let (handle, shared) = common::start_oxllm(routing_config(&[&a.base, &b.base]))
        .await
        .unwrap();
    let state = shared.load_full();
    let mut first = open_realtime(&handle, "rt-shared").await;
    let first_ready = receive_json(&mut first).await;
    let first_name = first_ready["backend"].as_str().unwrap();
    let first_index = usize::from(first_name == "b");
    let other_index = 1 - first_index;
    let first_key = state.router.cooldown_key(&state.deployments[first_index]);
    let other_key = state.router.cooldown_key(&state.deployments[other_index]);
    assert_eq!(state.router.usage_store().in_flight(&first_key), 1);
    assert_eq!(state.router.usage_store().in_flight(&other_key), 0);
    assert!(state.router.latency_store().p50(&first_key).is_some());
    assert!(state.router.usage_store().contains(&first_key));
    let latency = state.router.latency_store().p50(&first_key);

    let chat: Value = common::http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"rt-shared","messages":[{"role":"user","content":"hello"}]}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_ne!(chat["choices"][0]["message"]["content"], first_name);
    let mut second = open_realtime(&handle, "rt-shared").await;
    let second_ready = receive_json(&mut second).await;
    assert_ne!(second_ready["backend"], first_name);
    assert_eq!(state.router.usage_store().in_flight(&first_key), 1);
    assert_eq!(state.router.usage_store().in_flight(&other_key), 1);
    assert!(state.router.usage_store().contains(&other_key));

    first.close(None).await.unwrap();
    wait_idle(&state, &first_key).await;
    assert_eq!(state.router.usage_store().in_flight(&other_key), 1);
    assert_eq!(
        state.router.latency_store().p50(&first_key),
        latency,
        "session lifetime must never replace the handshake sample"
    );
    assert_eq!(
        state.router.cooldown_store().outcome_counts(&first_key),
        (0, 1)
    );
    let chat: Value = common::http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"rt-shared","messages":[{"role":"user","content":"again"}]}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(chat["choices"][0]["message"]["content"], first_name);
    second.close(None).await.unwrap();
    wait_idle(&state, &other_key).await;
    // Both real upstreams received realtime handshakes, not just HTTP 101s
    // synthesized locally by the proxy.
    for observations in [&mut a.observations, &mut b.observations] {
        tokio::time::timeout(Duration::from_secs(3), async {
            while let Some(event) = observations.recv().await {
                if event["type"] == "realtime" {
                    return;
                }
            }
            panic!("backend did not receive a realtime handshake");
        })
        .await
        .unwrap();
    }
    handle.shutdown();
}

#[tokio::test]
async fn handshake_status_failures_honor_policy_without_cooling_generic_bad_requests() {
    for (status, explicit_policy, cooling) in [
        (400, false, false),
        (400, true, true),
        (409, false, true),
        (401, true, true),
        (403, true, true),
        (404, true, true),
        (424, false, true),
        (429, true, true),
        (503, true, true),
    ] {
        let mock = httpmock::MockServer::start();
        let upstream = mock.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/realtime");
            then.status(status)
                .body("do not leak this provider error body");
        });
        let mut config = routing_config(&[&mock.base_url()]);
        config.router_settings.routing.allowed_fails = Some(0);
        config.router_settings.routing.allowed_fails_policy =
            explicit_policy.then(|| std::collections::HashMap::from([(status.to_string(), 0)]));
        let (handle, shared) = common::start_oxllm(config).await.unwrap();
        let state = shared.load_full();
        let key = state.router.cooldown_key(&state.deployments[0]);
        state
            .router
            .health_store()
            .mark_unhealthy(&key, Some("old transport fault".into()));
        let mut client = open_realtime(&handle, "rt-shared").await;
        receive_close(&mut client, 1011).await;
        wait_idle(&state, &key).await;
        assert!(
            state
                .router
                .health_store()
                .is_healthy(&key, Duration::from_secs(60)),
            "a real HTTP {status} proves reachability"
        );
        assert!(state.router.health_store().reason(&key).is_none());
        assert!(
            state.router.latency_store().p50(&key).is_none(),
            "a rejected handshake must not train latency routing"
        );
        assert_eq!(
            state.router.cooldown_store().is_cooling(&key),
            cooling,
            "status={status}, explicit={explicit_policy}"
        );
        assert_eq!(
            state.router.cooldown_store().failure_count(&key, status),
            u32::from(cooling)
        );
        assert_eq!(
            state.router.cooldown_store().failure_count(&key, 0),
            0,
            "an HTTP rejection must retain its status, rather than become a network failure"
        );
        upstream.assert_hits(1);
        handle.shutdown();
    }
}

async fn scripted_peer(
    closes: Vec<CloseCode>,
) -> (String, Arc<Notify>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let release = Arc::new(Notify::new());
    let peer_release = release.clone();
    let peer = tokio::spawn(async move {
        for code in closes {
            let (stream, _) = listener.accept().await.unwrap();
            let mut upstream = tokio_tungstenite::accept_async(stream).await.unwrap();
            upstream
                .send(Message::Text(r#"{"type":"session.created"}"#.into()))
                .await
                .unwrap();
            peer_release.notified().await;
            upstream
                .close(Some(CloseFrame {
                    code,
                    reason: "scripted terminal result".into(),
                }))
                .await
                .unwrap();
        }
    });
    (base, release, peer)
}

#[tokio::test]
async fn terminal_transport_failures_accumulate_allowed_fails_across_real_handshakes() {
    let (base, release, peer) = scripted_peer(vec![CloseCode::Error, CloseCode::Error]).await;
    let mut config = routing_config(&[&base]);
    config.router_settings.routing.allowed_fails = Some(1);
    let (handle, shared) = common::start_oxllm(config).await.unwrap();
    let state = shared.load_full();
    let key = state.router.cooldown_key(&state.deployments[0]);
    for failure in 1..=2 {
        let mut client = open_realtime(&handle, "rt-shared").await;
        assert_eq!(receive_json(&mut client).await["type"], "session.created");
        assert!(
            state
                .router
                .health_store()
                .is_healthy(&key, Duration::from_secs(60)),
            "handshake reachability must be recorded while the session is alive"
        );
        assert_eq!(state.router.usage_store().in_flight(&key), 1);
        let handshake_latency = state.router.latency_store().p50(&key).unwrap();
        release.notify_one();
        receive_close(&mut client, 1011).await;
        wait_idle(&state, &key).await;
        assert!(
            !state
                .router
                .health_store()
                .is_healthy(&key, Duration::from_secs(60))
        );
        assert_eq!(
            state.router.cooldown_store().failure_count(&key, 0),
            failure
        );
        assert_eq!(state.router.cooldown_store().is_cooling(&key), failure == 2);
        assert_eq!(
            state.router.cooldown_store().outcome_counts(&key),
            (failure, failure),
            "a handshake followed by a broken session is not a successful outcome"
        );
        assert_eq!(
            state.router.latency_store().p50(&key),
            Some(handshake_latency)
        );
    }
    tokio::time::timeout(Duration::from_secs(3), peer)
        .await
        .unwrap()
        .unwrap();
    handle.shutdown();
}

#[tokio::test]
async fn realtime_successes_and_failures_feed_the_shared_percentage_cooldown_window() {
    let (base, release, peer) =
        scripted_peer(vec![CloseCode::Normal, CloseCode::Error, CloseCode::Error]).await;
    let mut config = routing_config(&[&base, &base]);
    // Sibling identity is distinct even though this local peer accepts both.
    config.model_list[1].oxllm_params.api_key = Some("other-test-key".into());
    config.router_settings.enable_health_check_routing = Some(false);
    config.router_settings.routing.allowed_fails = Some(10);
    config.router_settings.routing.failure_threshold_percent = Some(0.5);
    config
        .router_settings
        .routing
        .failure_threshold_min_requests = Some(3);
    let (handle, shared) = common::start_oxllm(config).await.unwrap();
    let state = shared.load_full();
    let key = state.router.cooldown_key(&state.deployments[0]);
    let sibling_key = state.router.cooldown_key(&state.deployments[1]);
    // Keep a sibling occupied so least-busy deterministically picks this
    // deployment instead of its all-idle SimpleShuffle cold-start fallback.
    state.router.record_in_flight(&sibling_key, 1);
    for (failures, total, close_code) in [(0, 1, 1000), (1, 2, 1011), (2, 3, 1011)] {
        let mut client = open_realtime(&handle, "rt-shared").await;
        receive_json(&mut client).await;
        assert_eq!(state.router.usage_store().in_flight(&key), 1);
        release.notify_one();
        receive_close(&mut client, close_code).await;
        wait_idle(&state, &key).await;
        assert_eq!(
            state.router.cooldown_store().outcome_counts(&key),
            (failures, total)
        );
        assert_eq!(
            state.router.cooldown_store().is_cooling(&key),
            total == 3,
            "2/3 failures cools despite remaining below allowed_fails=10"
        );
    }
    assert!(!state.router.cooldown_store().is_cooling(&sibling_key));
    state.router.record_in_flight(&sibling_key, -1);
    tokio::time::timeout(Duration::from_secs(3), peer)
        .await
        .unwrap()
        .unwrap();
    handle.shutdown();
}

#[tokio::test]
async fn wildcard_realtime_uses_a_concrete_bare_selector_and_the_original_router_identity() {
    let mut backend = start_backend("wildcard").await;
    let mut config = routing_config(&[&format!("{}?tenant=test", backend.base)]);
    config.model_list[0].model_name = "*".into();
    config.model_list[0].oxllm_params.model = "openai/*".into();
    let (handle, shared) = common::start_oxllm(config).await.unwrap();
    let state = shared.load_full();
    let selected = &state.deployments[0];
    let key = state.router.cooldown_key(selected);
    let mut client = open_realtime(&handle, "openai/gpt-4o-realtime").await;
    let ready = receive_json(&mut client).await;
    assert_eq!(ready["selector"], "gpt-4o-realtime");
    let observation = backend.observations.recv().await.unwrap();
    let uri = observation["uri"].as_str().unwrap();
    assert!(uri.starts_with("/gateway/realtime?"));
    assert!(uri.contains("tenant=test"));
    assert!(uri.contains("model=gpt-4o-realtime"));
    assert!(!uri.contains("openai%2F"));
    assert_eq!(state.router.usage_store().in_flight(&key), 1);
    assert!(state.router.usage_store().contains(&key));
    assert!(state.router.latency_store().p50(&key).is_some());
    let mut concrete = selected.clone();
    concrete.oxllm_params.model = "openai/gpt-4o-realtime".into();
    assert_eq!(
        state
            .router
            .usage_store()
            .in_flight(&state.router.cooldown_key(&concrete)),
        0
    );
    assert_eq!(
        state
            .cache
            .counter_value(&format!("dep_rpm:{}", deployment_key(selected)))
            .await,
        Some(1.0)
    );

    let rejected = tokio_tungstenite::connect_async(url_for_realtime(&handle, "openai/*"))
        .await
        .unwrap_err();
    assert!(matches!(rejected,
        tokio_tungstenite::tungstenite::Error::Http(response) if response.status().as_u16() == 400));
    assert_eq!(
        state.router.usage_store().in_flight(&key),
        1,
        "a rejected literal wildcard must not release the other live session's slot"
    );
    assert!(backend.observations.try_recv().is_err());
    client.close(None).await.unwrap();
    wait_idle(&state, &key).await;
    handle.shutdown();
}
