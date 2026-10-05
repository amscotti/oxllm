//! Integration tests: per-deployment RPM/TPM limits after routing.

use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RoutingKnobs,
};
use serde_json::json;

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn params(base: &str, key: &str, rpm: Option<u32>) -> OxllmParams {
    OxllmParams {
        model: "openai/gpt-4o".to_string(),
        api_key: Some(key.to_string()),
        api_base: Some(base.to_string()),
        timeout: Some(30.0),
        stream_timeout: Some(30),
        rpm,
        ..Default::default()
    }
}

fn params_with_model(model: &str, base: &str, key: &str, rpm: Option<u32>) -> OxllmParams {
    OxllmParams {
        model: model.to_string(),
        api_key: Some(key.to_string()),
        api_base: Some(base.to_string()),
        timeout: Some(30.0),
        stream_timeout: Some(30),
        rpm,
        ..Default::default()
    }
}

fn params_with_tpm(model: &str, base: &str, key: &str, tpm: Option<u32>) -> OxllmParams {
    OxllmParams {
        model: model.to_string(),
        api_key: Some(key.to_string()),
        api_base: Some(base.to_string()),
        timeout: Some(30.0),
        stream_timeout: Some(30),
        tpm,
        ..Default::default()
    }
}

fn two_deployment_config(mock_a: &str, mock_b: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: params_with_model("openai/gpt-4o-a", mock_a, "sk-a", Some(1)),
                ..Default::default()
            },
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: params_with_model("openai/gpt-4o-b", mock_b, "sk-b", Some(10)),
                ..Default::default()
            },
        ],
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("usage_based".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

async fn start(config: ProxyConfig) -> OxllmHandle {
    let shared = build_shared_state(config).expect("state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let router = build_router(shared);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });
    OxllmHandle { addr, shutdown_tx }
}

fn mock_success<'a>(mock: &'a MockServer, key: &str) -> httpmock::Mock<'a> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer {key}"));
        then.status(200).json_body(json!({
        "id": "chatcmpl-dep",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }));
    })
}

#[tokio::test]
async fn deployment_rpm_fails_over_to_second_deployment() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();
    let handle = start(two_deployment_config(
        &mock_a.base_url(),
        &mock_b.base_url(),
    ))
    .await;

    let ma = mock_success(&mock_a, "sk-a");
    let mb = mock_success(&mock_b, "sk-b");

    let client = reqwest::Client::new();
    let body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}]
    });

    let r1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req1");
    assert_eq!(r1.status(), 200, "first request should succeed");

    let r2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req2");
    assert_eq!(
        r2.status(),
        200,
        "second request should succeed via unlimited deployment"
    );

    assert_eq!(ma.hits(), 1, "rpm-limited deployment should serve once");
    assert_eq!(
        mb.hits(),
        1,
        "second deployment should serve once after first is exhausted"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn single_deployment_rpm_blocks_second_request() {
    let mock = MockServer::start();
    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: params(&mock.base_url(), "sk-only", Some(1)),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start(config).await;
    let m = mock_success(&mock, "sk-only");

    let client = reqwest::Client::new();
    let body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}]
    });

    let r1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req1");
    assert_eq!(r1.status(), 200);

    let r2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req2");
    assert_eq!(
        r2.status(),
        429,
        "only deployment exhausted rpm should return 429"
    );
    assert_eq!(m.hits(), 1, "upstream should not be called twice");

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// TPM (tokens-per-minute) limit tests
// ---------------------------------------------------------------------------

/// Mock that reports a token usage exceeding a low TPM ceiling, so the first
/// successful response exhausts the deployment's per-minute token budget.
fn mock_high_token_usage<'a>(
    mock: &'a MockServer,
    key: &str,
    total_tokens: u64,
) -> httpmock::Mock<'a> {
    // `extract_usage` recomputes `total_tokens` as `prompt + completion`
    // (it does not trust an upstream-supplied total that is inconsistent with
    // the prompt/completion counters — see orchestration/usage.rs). The mock
    // therefore reports prompt/completion that SUM to the requested total so
    // the recorded TPM reflects `total_tokens` rather than a smaller
    // recomputed value.
    let completion: u64 = 5;
    let prompt = total_tokens.saturating_sub(completion);
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer {key}"));
        then.status(200).json_body(json!({
        "id": "chatcmpl-tpm",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total_tokens}
        }));
    })
}

/// A single deployment with `tpm: 50`. The upstream reports 60 total tokens,
/// so the first request exhausts the token budget. The second request is
/// rejected with 429.
#[tokio::test]
async fn single_deployment_tpm_blocks_after_limit() {
    let mock = MockServer::start();
    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: params_with_tpm("openai/gpt-4o", &mock.base_url(), "sk-only", Some(50)),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start(config).await;
    let m = mock_high_token_usage(&mock, "sk-only", 60);

    let client = reqwest::Client::new();
    let body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}]
    });

    // First request: TPM counter is 0 (< 50) → admitted → records 60 tokens.
    let r1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req1");
    assert_eq!(
        r1.status(),
        200,
        "first request should succeed (tpm not yet consumed)"
    );

    // Second request: TPM counter is now 60 (≥ 50) → rejected.
    let r2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req2");
    assert_eq!(
        r2.status(),
        429,
        "second request should be rejected once deployment tpm is exhausted"
    );
    assert_eq!(
        m.hits(),
        1,
        "upstream should not be called after tpm exhaustion"
    );

    let _ = handle.shutdown_tx.send(());
}

/// Two deployments for the same model: first with `tpm: 10` (very low),
/// second effectively unlimited. After the TPM-limited deployment's token
/// budget is consumed, subsequent requests must fail over to the unlimited
/// deployment.
///
/// Both deployments publish an `rpm` limit (A: 1, B: a high ceiling) because
/// `record_deployment_usage` snapshots the `dep_rpm:{key}` counter, which is
/// only incremented for deployments that publish `rpm`. Without it, every
/// `usage_based` pick is a shuffled 0-0 tie and the test is flaky; with it,
/// the first request's usage is always recorded, so the second pick is
/// deterministic regardless of which deployment the initial shuffle chose.
#[tokio::test]
async fn tpm_failover_to_second_deployment() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let config = ProxyConfig {
        model_list: vec![
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    rpm: Some(1),
                    ..params_with_tpm("openai/gpt-4o-a", &mock_a.base_url(), "sk-a", Some(10))
                },
                ..Default::default()
            },
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    rpm: Some(1000),
                    ..params_with_tpm("openai/gpt-4o-b", &mock_b.base_url(), "sk-b", None)
                },
                ..Default::default()
            },
        ],
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("usage_based".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start(config).await;

    let ma = mock_high_token_usage(&mock_a, "sk-a", 60);
    let mb = mock_high_token_usage(&mock_b, "sk-b", 60);

    let client = reqwest::Client::new();
    let body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}]
    });

    // First request: picks a deployment (usage_based, both at 0). Whatever it
    // picks, it succeeds and records 60 tokens.
    let r1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req1");
    assert_eq!(r1.status(), 200, "first request should succeed");

    // Second request: with usage_based routing, the deployment that served
    // req 1 now has higher utilisation, so the router prefers the other.
    // If deployment A (tpm=10) served req 1, its TPM (60 ≥ 10) is now
    // exhausted — any attempt to use A fails over to B via cooldown.
    let r2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("req2");
    assert_eq!(
        r2.status(),
        200,
        "second request should succeed via the other deployment"
    );

    assert_eq!(ma.hits(), 1, "deployment A should serve exactly once");
    assert_eq!(
        mb.hits(),
        1,
        "deployment B should serve once after A's TPM is exhausted"
    );

    let _ = handle.shutdown_tx.send(());
}
