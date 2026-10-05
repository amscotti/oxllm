//! Live Realtime WebSocket tests against OpenAI and xAI (Grok Voice).
//!
//! Each test auto-skips when its provider key is unset, so the suite is safe
//! to run in CI without credentials.
//!
//! Run with:
//! `OPENAI_API_KEY=... XAI_API_KEY=... cargo test --test realtime_live -- --nocapture`

use futures_util::{SinkExt, StreamExt};
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

const MASTER_KEY: &str = "sk-live-realtime-master-000000000000";

struct OxllmHandle {
    addr: SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

fn proxy_config(model: &str, api_key: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "rt".to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                api_key: Some(api_key.to_string()),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

async fn start_oxllm(config: ProxyConfig) -> Option<OxllmHandle> {
    let shared = build_shared_state(config).expect("configured live proxy state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind live proxy");
    let addr = listener.local_addr().expect("live proxy address");
    let router = build_router(shared.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await
        .expect("serve live proxy");
    });
    Some(OxllmHandle { addr, shutdown_tx })
}

/// Drive one text turn through the gateway and return the assistant text.
///
/// `session_type` controls whether `session.update` carries
/// `"type": "realtime"` (required by OpenAI; xAI documents the field without
/// it). Text is collected from both the text-output and audio-transcript
/// events so the helper works for either output modality.
async fn roundtrip(handle: &OxllmHandle, session_type: bool, instructions: &str) -> String {
    let url = format!(
        "ws://{}/v1/realtime?model=rt&authorization={MASTER_KEY}",
        handle.addr
    );
    let req = url.into_client_request().expect("ws request");
    let (mut ws, _response) = tokio_tungstenite::connect_async(req)
        .await
        .expect("realtime ws should connect");

    let mut session = json!({
        "output_modalities": ["text"],
        "instructions": instructions,
    });
    if session_type {
        session["type"] = json!("realtime");
    }
    ws.send(Message::Text(
        json!({"type": "session.update", "session": session})
            .to_string()
            .into(),
    ))
    .await
    .expect("session.update");
    ws.send(Message::Text(
        json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with exactly: PONG"}]
            }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("conversation.item.create");
    ws.send(Message::Text(
        json!({"type": "response.create", "response": {"output_modalities": ["text"]}})
            .to_string()
            .into(),
    ))
    .await
    .expect("response.create");

    let mut text = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    loop {
        match tokio::time::timeout_at(deadline, ws.next()).await {
            Err(_) => panic!("timed out waiting for the realtime response (got so far: {text:?})"),
            Ok(None) => break,
            Ok(Some(Err(e))) => panic!("realtime ws error: {e}"),
            Ok(Some(Ok(message))) => {
                let raw = match message {
                    Message::Text(t) => t.to_string(),
                    Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
                    Message::Close(_) => break,
                    _ => continue,
                };
                let Ok(event) = serde_json::from_str::<Value>(&raw) else {
                    continue;
                };
                match event["type"].as_str().unwrap_or_default() {
                    "response.output_text.delta" | "response.output_audio_transcript.delta" => {
                        text.push_str(event["delta"].as_str().unwrap_or_default());
                    }
                    "response.output_text.done" | "response.output_audio_transcript.done"
                        if text.is_empty() =>
                    {
                        text = event["text"].as_str().unwrap_or_default().to_string();
                    }
                    "error" => {
                        panic!("realtime error event: {event}");
                    }
                    "response.done" => break,
                    _ => {}
                }
            }
        }
    }
    let _ = ws.close(None).await;
    text
}

#[tokio::test]
async fn realtime_text_roundtrip_openai() {
    // The binary installs the process-level rustls CryptoProvider at startup;
    // the test harness builds the router directly, and the WS client
    // (tokio-tungstenite over rustls) needs the provider installed too.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let Some(key) = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
    else {
        eprintln!("skipping: OPENAI_API_KEY not set");
        return;
    };
    let Some(handle) = start_oxllm(proxy_config("openai/gpt-realtime-2.1-mini", &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let text = roundtrip(&handle, true, "Answer in as few words as possible.").await;
    assert!(
        !text.trim().is_empty(),
        "expected a text response from the OpenAI realtime session"
    );
    eprintln!("openai realtime response: {text}");
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_text_roundtrip_xai() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let Some(key) = std::env::var("XAI_API_KEY").ok().filter(|k| !k.is_empty()) else {
        eprintln!("skipping: XAI_API_KEY not set");
        return;
    };
    // No `api_base`: exercises the xAI realtime host default
    // (wss://api.x.ai/v1) rather than the OpenAI identity fallback.
    let Some(handle) = start_oxllm(proxy_config("xai/grok-voice-latest", &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    // xAI documents `session.update` without the `type` discriminator.
    let text = roundtrip(&handle, false, "Answer in as few words as possible.").await;
    assert!(
        !text.trim().is_empty(),
        "expected a text response from the xAI realtime session"
    );
    eprintln!("xai realtime response: {text}");
    let _ = handle.shutdown_tx.send(());
}
