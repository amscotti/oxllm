//! Explicit thinking budgets fail loudly when no valid budget window exists.

mod common;

use std::sync::Arc;
use std::time::Duration;

use httpmock::{Method::POST, MockServer};
use oxllm_server::cache::{SemanticCache, SemanticCacheStore};
use oxllm_server::hooks::CacheHook;
use oxllm_server::hooks::cache_hook::{SemanticCacheRuntime, cache_key_for};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug)]
enum Route {
    Direct,
    Invoke,
    Converse,
    Vertex,
    Foundry,
}

const ROUTES: [Route; 5] = [
    Route::Direct,
    Route::Invoke,
    Route::Converse,
    Route::Vertex,
    Route::Foundry,
];
const APIS: [&str; 2] = ["/v1/chat/completions", "/v1/messages"];
const REASONING: &str = "Considering the answer carefully.";

fn config(upstream: &MockServer, route: Route) -> oxllm_server::ProxyConfig {
    let mut config = common::proxy_config(&upstream.base_url());
    config.model_list[0].model_name = "model".into();
    config.router_settings.routing.num_retries = Some(0);
    config.router_settings.routing.cooldown_time = Some(0);
    let params = &mut config.model_list[0].oxllm_params;
    match route {
        Route::Direct => {
            params.model = "anthropic/claude-haiku-4-5".into();
            params.api_base = Some(upstream.url("/direct"));
        }
        Route::Foundry => {
            params.model = "azure_ai/claude-haiku-4-5".into();
            params.api_base = Some(upstream.url("/foundry/anthropic/v1"));
        }
        Route::Vertex => {
            params.model = "vertex_ai/claude-haiku-4-5".into();
            params.api_base = Some(upstream.url("/vertex"));
            params.api_key = Some("ya29.synthetic-local-token".into());
            params
                .extra
                .insert("vertex_project".into(), json!("local-project"));
            params
                .extra
                .insert("vertex_location".into(), json!("us-central1"));
        }
        Route::Invoke | Route::Converse => {
            params.model = "bedrock/anthropic.claude-haiku-4-5".into();
            params.api_base = Some(upstream.url("/bedrock"));
            params.api_key = None;
            params.aws_access_key_id = Some("AKIDEXAMPLE".into());
            params.aws_secret_access_key = Some("synthetic-local-signing-only".into());
            params
                .extra
                .insert("aws_region_name".into(), json!("us-east-1"));
            params.extra.insert(
                "use_converse_api".into(),
                json!(matches!(route, Route::Converse)),
            );
        }
    }
    config
}

fn path(route: Route) -> &'static str {
    match route {
        Route::Direct => "/direct/messages",
        Route::Foundry => "/foundry/anthropic/v1/messages",
        Route::Vertex => {
            "/vertex/v1/projects/local-project/locations/us-central1/publishers/anthropic/models/claude-haiku-4-5:rawPredict"
        }
        Route::Invoke => "/bedrock/model/anthropic.claude-haiku-4-5/invoke",
        Route::Converse => "/bedrock/model/anthropic.claude-haiku-4-5/converse",
    }
}

fn request(api: &str, max_tokens: u64) -> Value {
    if api == "/v1/responses" {
        json!({"model":"model","input":"Hello","max_output_tokens":max_tokens})
    } else {
        json!({"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":max_tokens})
    }
}

fn wire_body(route: Route, max_tokens: u64, budget: Option<u64>) -> Value {
    if matches!(route, Route::Converse) {
        let mut body = json!({"messages":[{"role":"user","content":[{"text":"Hello"}]}],"inferenceConfig":{"maxTokens":max_tokens}});
        if let Some(budget) = budget {
            body["additionalModelRequestFields"] =
                json!({"thinking":{"type":"enabled","budget_tokens":budget}});
        }
        body
    } else {
        let mut body =
            json!({"messages":[{"role":"user","content":"Hello"}],"max_tokens":max_tokens});
        match route {
            Route::Direct | Route::Foundry => body["model"] = json!("claude-haiku-4-5"),
            Route::Invoke => body["anthropic_version"] = json!("bedrock-2023-05-31"),
            Route::Vertex => body["anthropic_version"] = json!("vertex-2023-10-16"),
            Route::Converse => unreachable!(),
        }
        if let Some(budget) = budget {
            body["thinking"] = json!({"type":"enabled","budget_tokens":budget});
        }
        body
    }
}

fn reply(route: Route, thinking: bool) -> Value {
    let mut response = if matches!(route, Route::Converse) {
        let mut content = vec![];
        if thinking {
            content.push(json!({"reasoningContent":{"reasoningText":{"text":REASONING,"signature":"local-signature"}}}));
        }
        content.push(json!({"text":"Hi"}));
        json!({"output":{"message":{"role":"assistant","content":content}},"stopReason":"end_turn","usage":{"inputTokens":5,"outputTokens":12,"totalTokens":17}})
    } else {
        let mut content = vec![];
        if thinking {
            content.push(
                json!({"type":"thinking","thinking":REASONING,"signature":"local-signature"}),
            );
        }
        content.push(json!({"type":"text","text":"Hi"}));
        json!({"id":"msg-thinking","type":"message","role":"assistant","model":"claude-haiku-4-5","content":content,"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":12}})
    };
    if thinking {
        if matches!(route, Route::Converse) {
            response["usage"]["outputTokensDetails"] = json!({"reasoningTokens":8});
        } else {
            response["usage"]["output_tokens_details"] = json!({"thinking_tokens":8});
        }
    }
    response
}

fn assert_reasoning(api: &str, body: &Value) {
    match api {
        "/v1/messages" => {
            let thinking = body["content"]
                .as_array()
                .unwrap()
                .iter()
                .find(|part| part["type"] == "thinking")
                .expect("signed thinking block");
            assert_eq!(thinking["thinking"], REASONING);
            assert_eq!(thinking["signature"], "local-signature");
            assert_eq!(body["usage"]["output_tokens"], 12);
        }
        _ => {
            let message = &body["choices"][0]["message"];
            assert_eq!(message["reasoning_content"], REASONING);
            assert_eq!(
                message["thinking_blocks"][0]["signature"],
                "local-signature"
            );
            assert!(
                body["usage"]["completion_tokens_details"]["reasoning_tokens"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        }
    }
}

#[tokio::test]
async fn explicit_impossible_thinking_returns_400_before_upstream_including_stream_requests() {
    for route in ROUTES {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST);
                then.status(200).json_body(reply(route, false));
            })
            .await;
        let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
        let client = common::http_client();
        for api in APIS {
            for max_tokens in [100, 1024] {
                for stream in [false, true] {
                    let mut payload = request(api, max_tokens);
                    payload["thinking"] = json!({"type":"enabled","budget_tokens":2000});
                    payload["stream"] = json!(stream);
                    let response = client
                        .post(handle.url(api))
                        .json(&payload)
                        .send()
                        .await
                        .unwrap();
                    let status = response.status();
                    assert_eq!(
                        response.headers()["x-request-id"],
                        response.headers()["x-oxllm-request-id"]
                    );
                    let body: Value = response.json().await.unwrap();
                    assert_eq!(
                        status, 400,
                        "{route:?} {api} max={max_tokens} stream={stream}: {body}"
                    );
                    assert_eq!(body["error"]["type"], "invalid_request_error");
                    let message = body["error"]["message"].as_str().unwrap();
                    assert!(message.contains("thinking.budget_tokens"), "{message}");
                    assert!(message.contains("max_tokens"), "{message}");
                }
            }
        }
        wire.assert_hits_async(0).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn valid_and_clamped_thinking_keeps_native_wire_budgets_and_signed_responses() {
    for route in ROUTES {
        let upstream = MockServer::start_async().await;
        let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
        let client = common::http_client();
        for (max_tokens, budget, expected) in [
            (3000, 2000, 2000),
            (1500, 2000, 1499),
            (1025, 2000, 1024),
            (3000, 512, 1024),
        ] {
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(wire_body(
                        route,
                        max_tokens,
                        Some(expected),
                    ));
                    then.status(200).json_body(reply(route, true));
                })
                .await;
            for api in APIS {
                let mut payload = request(api, max_tokens);
                payload["thinking"] = json!({"type":"enabled","budget_tokens":budget});
                let response = client
                    .post(handle.url(api))
                    .json(&payload)
                    .send()
                    .await
                    .unwrap();
                let status = response.status();
                let body: Value = response.json().await.unwrap();
                assert_eq!(status, 200, "{route:?} {api} max={max_tokens}: {body}");
                assert_reasoning(api, &body);
            }
            wire.assert_hits_async(2).await;
        }
        handle.shutdown();
    }
}

#[tokio::test]
async fn inferred_thinking_remains_lenient_without_increasing_the_output_limit() {
    for route in ROUTES {
        let upstream = MockServer::start_async().await;
        let mut expected = wire_body(route, 100, None);
        if matches!(route, Route::Vertex) {
            expected["anthropic_beta"] = json!(["effort-2025-11-24"]);
        }
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path(path(route)).json_body(expected);
                then.status(200).json_body(reply(route, false));
            })
            .await;
        let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
        let client = common::http_client();
        for api in ["/v1/chat/completions", "/v1/responses"] {
            let mut payload = request(api, 100);
            if api == "/v1/responses" {
                payload["reasoning"] = json!({"effort":"high"});
            } else {
                payload["reasoning_effort"] = json!("high");
            }
            let response = client
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} {api}: {body}");
            match api {
                "/v1/responses" => assert!(
                    body["output"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|item| item["type"] != "reasoning"),
                    "{route:?}: inferred thinking must not appear without a valid budget"
                ),
                _ => assert!(
                    body["choices"][0]["message"]
                        .get("reasoning_content")
                        .is_none(),
                    "{route:?}: inferred thinking must not appear without a valid budget"
                ),
            }
        }
        wire.assert_hits_async(2).await;
        handle.shutdown();
    }
}

fn canonicalize(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.sort_keys();
            object.values_mut().for_each(canonicalize);
        }
        Value::Array(items) => items.iter_mut().for_each(canonicalize),
        _ => {}
    }
}

#[tokio::test]
async fn old_plain_answers_cannot_bypass_thinking_validation_through_exact_or_semantic_caches() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path(path(Route::Direct));
            then.status(200).json_body(reply(Route::Direct, false));
        })
        .await;
    let embeddings = MockServer::start_async().await;
    let embed = embeddings
        .mock_async(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .json_body(json!({"data":[{"index":0,"embedding":[1.0,0.0]}]}));
        })
        .await;
    let (handle, state) = common::start_oxllm(config(&upstream, Route::Direct))
        .await
        .unwrap();
    let semantic = Arc::new(SemanticCache::new(10, 0.95, Duration::from_secs(60)));
    let mut next = (*state.load_full()).clone();
    next.hooks = vec![Arc::new(
        CacheHook::new(next.cache.clone(), Duration::from_secs(60)).with_semantic(Arc::new(
            SemanticCacheRuntime::new(
                semantic.clone(),
                "embedding".into(),
                embeddings.base_url(),
                None,
                common::http_client(),
            ),
        )),
    )];
    state.store(Arc::new(next));
    let mut payload = request(APIS[0], 100);
    payload["thinking"] = json!({"type":"enabled","budget_tokens":2000});
    payload["cache"] = json!({"use_cache":true});
    let principal = oxllm_server::auth::AuthUser::open().key_hash_hex;
    let mut old_request = payload.clone();
    old_request.as_object_mut().unwrap().remove("cache");
    canonicalize(&mut old_request);
    let mut hasher = Sha256::new();
    hasher.update(principal.as_bytes());
    hasher.update(b"\x00");
    hasher.update(serde_json::to_vec(&old_request).unwrap());
    let old_key = format!("resp:{:x}", hasher.finalize());
    assert_ne!(old_key, cache_key_for(&payload, &principal));
    let cached = json!({"id":"old-plain-answer","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]});
    state
        .load_full()
        .cache
        .set(&old_key, cached.to_string(), Some(Duration::from_secs(60)))
        .await;
    old_request["messages"][0]["content"] = json!("");
    let mut old_scope = json!([
        "oxllm-semantic-v2",
        "model",
        "embedding",
        embeddings.base_url(),
        old_request
    ]);
    canonicalize(&mut old_scope);
    let old_scope = format!(
        "sem:v2:{:x}",
        Sha256::digest(serde_json::to_vec(&old_scope).unwrap())
    );
    semantic
        .store(
            vec![1.0, 0.0],
            "Hello".into(),
            cached,
            &principal,
            &old_scope,
        )
        .await;
    let response = common::http_client()
        .post(handle.url(APIS[0]))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let status = response.status();
    assert!(!response.headers().contains_key("x-oxllm-cache-hit"));
    let body: Value = response.json().await.unwrap();
    assert_eq!(status, 400, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("thinking.budget_tokens"), "{message}");
    assert!(message.contains("max_tokens"), "{message}");
    assert!(state.load_full().cache.get(&old_key).await.is_some());
    assert!(
        semantic
            .search(&[1.0, 0.0], &principal, &old_scope)
            .await
            .is_some()
    );
    wire.assert_hits_async(0).await;
    embed.assert_hits_async(1).await;
    handle.shutdown();
}
