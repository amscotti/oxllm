//! Loopback HTTP regressions: inspect the native wire and the TTL actually
//! persisted by the successful chat path, without sleeping through cache TTLs.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::http::{StatusCode, Uri};
use axum::{Json, Router};
use oxllm_core::cache::IncrResult;
use oxllm_core::{Cache, Error};
use serde_json::{Value, json};

use super::{ONE_HOUR_PIN_TTL, PIN_TTL, PromptCachingCheck};
use crate::cache::InMemoryCache;
use crate::state::{AppState, SharedState, StateBuilder};

const ANTHROPIC_MODEL: &str = "anthropic/claude-sonnet-4-5";
const CONVERSE_MODEL: &str = "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0";
const REGIONAL_MODEL: &str = "bedrock/apac.anthropic.claude-haiku-4-5-20251001-v1:0";
const PROFILE: &str =
    "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/pin-test";
const OPEN_KEY_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Clone, Debug)]
struct PinWrite {
    key: String,
    value: String,
    ttl: Option<Duration>,
}

struct SpyCache {
    inner: InMemoryCache,
    pins: Mutex<Vec<PinWrite>>,
}

impl SpyCache {
    fn new() -> Self {
        Self {
            inner: InMemoryCache::new(1_000),
            pins: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Cache for SpyCache {
    async fn get(&self, key: &str) -> Option<String> {
        self.inner.get(key).await
    }

    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        if key.starts_with("pcache:") {
            self.pins.lock().unwrap().push(PinWrite {
                key: key.to_string(),
                value: value.clone(),
                ttl,
            });
        }
        self.inner.set(key, value, ttl).await;
    }

    async fn delete(&self, key: &str) {
        self.inner.delete(key).await;
    }

    async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error> {
        self.inner.incr(key, by, ttl).await
    }

    async fn incr_if_below(
        &self,
        key: &str,
        delta: f64,
        ceiling: f64,
        ttl: Option<Duration>,
    ) -> Result<IncrResult, Error> {
        self.inner.incr_if_below(key, delta, ceiling, ttl).await
    }

    async fn decr_if_exists(&self, key: &str, by: f64) -> Result<bool, Error> {
        self.inner.decr_if_exists(key, by).await
    }

    async fn counter_value(&self, key: &str) -> Option<f64> {
        self.inner.counter_value(key).await
    }

    async fn try_counter_value(&self, key: &str) -> Result<Option<f64>, Error> {
        self.inner.try_counter_value(key).await
    }

    async fn ping(&self) -> bool {
        self.inner.ping().await
    }

    async fn publish_invalidation(&self, key: &str) {
        self.inner.publish_invalidation(key).await;
    }
}

struct HttpServer {
    base: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

async fn serve(router: Router) -> HttpServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown, stopped) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = stopped.await;
            })
            .await
            .unwrap();
    });
    HttpServer {
        base: format!("http://{address}"),
        shutdown: Some(shutdown),
    }
}

struct NativeMock {
    server: HttpServer,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
}

impl NativeMock {
    async fn new(converse: bool, stream: bool, fail: bool) -> Self {
        let payload = if fail {
            json!({"error": {"message": "synthetic overload"}})
                .to_string()
                .into_bytes()
        } else if stream && converse {
            [
                json!({"messageStart":{"role":"assistant"}}),
                json!({"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"wire-ok"}}}),
                json!({"messageStop":{"stopReason":"end_turn"}}),
                json!({"metadata":{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}}),
            ]
            .iter()
            .flat_map(|event| {
                oxllm_providers::bedrock::encode_eventstream_frame(event.to_string().as_bytes())
            })
            .collect::<Vec<_>>()
        } else if stream {
            [
                json!({"type":"message_start", "message":{
                    "id":"msg_pin", "model":"claude-sonnet-4-5", "role":"assistant",
                    "content":[], "usage":{"input_tokens":1,"output_tokens":0}
                }}),
                json!({"type":"content_block_start", "index":0, "content_block":{"type":"text","text":""}}),
                json!({"type":"content_block_delta", "index":0, "delta":{"type":"text_delta","text":"wire-ok"}}),
                json!({"type":"content_block_stop", "index":0}),
                json!({"type":"message_delta", "delta":{"stop_reason":"end_turn"}, "usage":{"output_tokens":1}}),
                json!({"type":"message_stop"}),
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>()
            .into_bytes()
        } else if converse {
            json!({
                "output":{"message":{"role":"assistant","content":[{"text":"wire-ok"}]}},
                "stopReason":"end_turn", "usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}
            })
            .to_string()
            .into_bytes()
        } else {
            json!({
                "id":"msg_pin", "type":"message", "role":"assistant", "model":"claude-sonnet-4-5",
                "content":[{"type":"text","text":"wire-ok"}], "stop_reason":"end_turn",
                "usage":{"input_tokens":1,"output_tokens":1}
            })
            .to_string()
            .into_bytes()
        };
        Self::with_response(
            payload,
            if stream && !fail {
                if converse {
                    "application/vnd.amazon.eventstream"
                } else {
                    "text/event-stream"
                }
            } else {
                "application/json"
            },
            if fail {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::OK
            },
        )
        .await
    }

    async fn openai() -> Self {
        Self::with_response(
            json!({"id":"chatcmpl_custom", "object":"chat.completion", "model":"custom-chat",
                "choices":[{"index":0,"message":{"role":"assistant","content":"wire-ok"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
            }).to_string().into_bytes(),
            "application/json",
            StatusCode::OK,
        )
        .await
    }

    async fn with_response(
        payload: Vec<u8>,
        content_type: &'static str,
        status: StatusCode,
    ) -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let router = Router::new().fallback(move |uri: Uri, Json(body): Json<Value>| {
            let captured = captured.clone();
            let payload = payload.clone();
            async move {
                captured
                    .lock()
                    .unwrap()
                    .push((uri.path().to_string(), body));
                (status, [("content-type", content_type)], payload)
            }
        });
        Self {
            server: serve(router).await,
            requests,
        }
    }

    fn only_body(&self) -> Value {
        let requests = self.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "expected exactly one upstream attempt");
        requests[0].1.clone()
    }
}

fn deployment(model: &str, mock: &NativeMock) -> Value {
    let mut params = json!({
        "model": model, "api_base": mock.server.base, "api_key":"synthetic-key",
        "timeout":5, "stream_timeout":5,
    });
    if model.starts_with("bedrock/") {
        params["aws_access_key_id"] = json!("AKIATESTKEY");
        params["aws_secret_access_key"] = json!("synthetic-secret");
        params["aws_region"] = json!("us-east-1");
        params["use_converse_api"] = json!(true);
    }
    if model.starts_with("vertex_ai/") {
        params["vertex_project"] = json!("pin-test-project");
        params["vertex_location"] = json!("us-central1");
    }
    json!({"model_name":"model", "oxllm_params":params})
}

async fn proxy(models: Vec<Value>, fallbacks: Value) -> (HttpServer, SharedState, Arc<SpyCache>) {
    let config = serde_json::from_value(json!({
        "model_list":models,
        "oxllm_settings":{"prompt_caching_enabled":true},
        "general_settings":{"allow_private_api_base":true},
        "router_settings":{"num_retries":0,"routing_strategy":"round_robin","fallbacks":fallbacks},
    }))
    .unwrap();
    let cache = Arc::new(SpyCache::new());
    let state = StateBuilder::new(config)
        .with_cache(cache.clone())
        .build()
        .unwrap();
    let shared = Arc::new(arc_swap::ArcSwap::from_pointee(state));
    let server = serve(crate::app::build_router(shared.clone())).await;
    (server, shared, cache)
}

fn request() -> Value {
    json!({"model":"model", "max_tokens":32, "messages":[
        {"role":"system", "content":"lorem ipsum dolor sit amet ".repeat(1_000)},
        {"role":"user", "content":"hello"},
    ]})
}

fn control() -> Value {
    json!({"type":"ephemeral","ttl":"1h"})
}

fn tool() -> Value {
    json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}})
}

fn native_tool_config() -> Value {
    json!({"tools":[
        {"toolSpec":{"name":"lookup","inputSchema":{"json":{"type":"object","properties":{}}}}},
        {"cachePoint":{"type":"default","ttl":"1h"}},
    ]})
}

async fn post(server: &HttpServer, body: &Value) -> String {
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
        .post(format!("{}/v1/chat/completions", server.base))
        .json(body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let text = response.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(text.contains("wire-ok"), "{text}");
    text
}

async fn assert_pin(cache: &SpyCache, state: &AppState, body: &Value, index: usize, ttl: Duration) {
    let check = PromptCachingCheck::new(state.cache.clone(), None);
    let (hash, tokens) = check.prompt_hash(body, "model").unwrap();
    assert!(check.engages(&state.deployments[index], tokens));
    let expected_key = format!("pcache:{OPEN_KEY_HASH}:{hash:032x}");
    let expected_value = state.router.cooldown_key(&state.deployments[index]);
    let pins = cache.pins.lock().unwrap().clone();
    assert_eq!(
        pins.len(),
        1,
        "only the serving attempt may write a pin: {pins:?}"
    );
    assert_eq!(pins[0].key, expected_key);
    assert_eq!(pins[0].value, expected_value);
    assert_eq!(
        pins[0].ttl,
        Some(ttl),
        "persisted Cache::set TTL must follow the native wire"
    );
    assert_eq!(cache.get(&expected_key).await, Some(expected_value));
    assert!(
        check
            .preferred_deployment(hash, "another-tenant")
            .await
            .is_none()
    );
}

#[tokio::test]
async fn custom_bedrock_invoke_forged_version_does_not_select_anthropic_pin_wire() {
    for configured in [
        "bedrock/custom.vendor-chat",
        "bedrock/arn:aws:bedrock:us-east-1:123456789012:imported-model/custom-chat",
    ] {
        for version in [Value::Null, json!("bedrock-2023-05-31")] {
            for extra_body in [false, true] {
                let mock = NativeMock::openai().await;
                let mut dep = deployment(configured, &mock);
                dep["oxllm_params"]["use_converse_api"] = json!(false);
                let mut body = request();
                body["messages"][1]["content"] = json!([
                    {"type":"text","text":"hello","cache_control":control()}
                ]);
                if extra_body {
                    dep["oxllm_params"]["extra_body"] = json!({"anthropic_version":version});
                } else {
                    body["anthropic_version"] = version.clone();
                }
                let (server, shared, cache) = proxy(vec![dep], json!([])).await;
                post(&server, &body).await;
                let native = mock.only_body();
                assert_eq!(native.get("anthropic_version"), Some(&version));
                assert_eq!(
                    native["messages"][1]["content"][0]["cache_control"],
                    control()
                );
                assert!(mock.requests.lock().unwrap()[0].0.ends_with("/invoke"));
                assert_pin(&cache, &shared.load_full(), &body, 0, PIN_TTL).await;
            }
        }
    }
}

#[tokio::test]
async fn vertex_openapi_forged_version_does_not_select_anthropic_pin_wire() {
    for configured in ["vertex_ai/custom-chat", "vertex_ai/google/gemini-2.5-flash"] {
        for version in [Value::Null, json!("vertex-2023-10-16")] {
            let mock = NativeMock::openai().await;
            let mut dep = deployment(configured, &mock);
            dep["oxllm_params"]["extra_body"] = json!({
                "anthropic_version":version,
                "system":[{"type":"text","text":"data, not a native system block","cache_control":control()}],
            });
            let (server, shared, cache) = proxy(vec![dep], json!([])).await;
            let body = request();
            post(&server, &body).await;
            let native = mock.only_body();
            assert_eq!(native.get("anthropic_version"), Some(&version));
            assert_eq!(native["system"][0]["cache_control"], control());
            assert!(
                mock.requests.lock().unwrap()[0]
                    .0
                    .ends_with("/openapi/chat/completions")
            );
            assert_pin(&cache, &shared.load_full(), &body, 0, PIN_TTL).await;
        }
    }
}

#[tokio::test]
async fn native_cache_wire_routes_survive_version_field_and_breakpoint_shaping() {
    for (model, converse, suffix) in [
        (ANTHROPIC_MODEL, false, "/messages"),
        (CONVERSE_MODEL, false, "/invoke"),
        (CONVERSE_MODEL, true, "/converse"),
        ("vertex_ai/claude-sonnet-4-5", false, ":rawPredict"),
        ("azure_ai/claude-sonnet-4-5", false, "/messages"),
    ] {
        for wildcard in [false, true] {
            for shape in [
                "unchanged",
                "null version",
                "drop version",
                "native extra breakpoint",
            ] {
                let mock = NativeMock::new(converse, false, false).await;
                let mut dep = deployment(model, &mock);
                if model.starts_with("bedrock/") {
                    dep["oxllm_params"]["use_converse_api"] = json!(converse);
                }
                let mut body = request();
                if wildcard {
                    dep["model_name"] = json!("*");
                    dep["oxllm_params"]["model"] =
                        json!(format!("{}/*", model.split_once('/').unwrap().0));
                    body["model"] = json!(model);
                }
                match shape {
                    "null version" => {
                        dep["oxllm_params"]["extra_body"] = json!({"anthropic_version":null})
                    }
                    "drop version" => {
                        dep["oxllm_params"]["additional_drop_params"] = json!(["anthropic_version"])
                    }
                    "native extra breakpoint" => {
                        dep["oxllm_params"]["extra_body"] = json!({"system":if converse {
                            json!([{"text":"configured"},{"cachePoint":{"type":"default","ttl":"1h"}}])
                        } else {
                            json!([{"type":"text","text":"configured","cache_control":control()}])
                        }})
                    }
                    _ => {}
                }
                if shape != "native extra breakpoint" {
                    body["messages"][0]["cache_control"] = control();
                }
                let (server, shared, cache) = proxy(vec![dep], json!([])).await;
                post(&server, &body).await;
                let native = mock.only_body();
                if converse {
                    assert_eq!(
                        native["system"][1]["cachePoint"]["ttl"], "1h",
                        "{model}, {shape}"
                    );
                } else {
                    assert_eq!(
                        native["system"][0]["cache_control"],
                        control(),
                        "{model}, {shape}"
                    );
                }
                if shape == "drop version" {
                    assert!(native.get("anthropic_version").is_none());
                } else if shape == "null version" {
                    assert_eq!(native.get("anthropic_version"), Some(&Value::Null));
                }
                assert!(mock.requests.lock().unwrap()[0].0.ends_with(suffix));
                assert_pin(&cache, &shared.load_full(), &body, 0, ONE_HOUR_PIN_TTL).await;
            }
        }
    }
}

#[tokio::test]
async fn anthropic_function_level_tool_one_hour_mark_reaches_native_root_and_pin() {
    for rate in [None, Some(0.0)] {
        let mock = NativeMock::new(false, false, false).await;
        let mut dep = deployment(ANTHROPIC_MODEL, &mock);
        if let Some(rate) = rate {
            dep["oxllm_params"]["cache_creation_input_token_cost_above_1hr"] = json!(rate);
        }
        let (server, shared, cache) = proxy(vec![dep], json!([])).await;
        let mut body = request();
        let mut marked = tool();
        marked["function"]["cache_control"] = control();
        body["tools"] = json!([marked]);
        post(&server, &body).await;
        assert_eq!(mock.only_body()["tools"][0]["cache_control"], control());
        assert_pin(&cache, &shared.load_full(), &body, 0, ONE_HOUR_PIN_TTL).await;
    }
}

#[tokio::test]
async fn anthropic_root_tool_one_hour_mark_extends_pin_even_with_zero_contract_rate() {
    let mock = NativeMock::new(false, false, false).await;
    let mut dep = deployment(ANTHROPIC_MODEL, &mock);
    dep["oxllm_params"]["cache_creation_input_token_cost_above_1hr"] = json!(0);
    let (server, shared, cache) = proxy(vec![dep], json!([])).await;
    let mut body = request();
    let mut marked = tool();
    marked["cache_control"] = control();
    body["tools"] = json!([marked]);
    post(&server, &body).await;
    assert_eq!(mock.only_body()["tools"][0]["cache_control"], control());
    assert_pin(&cache, &shared.load_full(), &body, 0, ONE_HOUR_PIN_TTL).await;
}

#[tokio::test]
async fn converse_price_override_cannot_extend_pin_when_wire_omits_ttl() {
    let mock = NativeMock::new(true, false, false).await;
    let mut dep = deployment("bedrock/anthropic.claude-sonnet-4-20250514-v1:0", &mock);
    dep["oxllm_params"]["cache_creation_input_token_cost_above_1hr"] = json!(1e-5);
    let (server, shared, cache) = proxy(vec![dep], json!([])).await;
    let mut body = request();
    body["messages"][0]["cache_control"] = control();
    post(&server, &body).await;
    assert_eq!(
        mock.only_body()["system"][1],
        json!({"cachePoint":{"type":"default"}})
    );
    assert_pin(&cache, &shared.load_full(), &body, 0, PIN_TTL).await;
}

#[tokio::test]
async fn converse_dropped_tool_controls_do_not_become_new_cache_points_or_long_pins() {
    for function_level in [false, true] {
        let mock = NativeMock::new(true, false, false).await;
        let (server, shared, cache) =
            proxy(vec![deployment(CONVERSE_MODEL, &mock)], json!([])).await;
        let mut body = request();
        let mut marked = tool();
        if function_level {
            marked["function"]["cache_control"] = control();
        } else {
            marked["cache_control"] = control();
        }
        body["tools"] = json!([marked]);
        post(&server, &body).await;
        let native = mock.only_body();
        let tools = native["toolConfig"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].get("toolSpec").is_some());
        assert!(tools[0].get("cachePoint").is_none());
        assert_pin(&cache, &shared.load_full(), &body, 0, PIN_TTL).await;
    }
}

#[tokio::test]
async fn converse_native_root_tool_cache_point_extends_persisted_pin() {
    let mock = NativeMock::new(true, false, false).await;
    let (server, shared, cache) = proxy(vec![deployment(CONVERSE_MODEL, &mock)], json!([])).await;
    let mut body = request();
    let point = json!({"cachePoint":{"type":"default","ttl":"1h"}});
    body["tools"] = json!([tool(), point]);
    post(&server, &body).await;
    assert_eq!(mock.only_body()["toolConfig"]["tools"][1], point);
    assert_pin(&cache, &shared.load_full(), &body, 0, ONE_HOUR_PIN_TTL).await;
}

#[tokio::test]
async fn converse_stream_pin_uses_the_native_tool_cache_point() {
    let mock = NativeMock::new(true, true, false).await;
    let (server, shared, cache) = proxy(vec![deployment(CONVERSE_MODEL, &mock)], json!([])).await;
    let mut body = request();
    body["stream"] = json!(true);
    let point = json!({"cachePoint":{"type":"default","ttl":"1h"}});
    body["tools"] = json!([tool(), point]);
    let response = post(&server, &body).await;
    assert!(response.contains("[DONE]"), "{response}");
    assert_eq!(mock.only_body()["toolConfig"]["tools"][1], point);
    assert!(
        mock.requests.lock().unwrap()[0]
            .0
            .ends_with("/converse-stream")
    );
    assert_pin(&cache, &shared.load_full(), &body, 0, ONE_HOUR_PIN_TTL).await;
}

#[tokio::test]
async fn converse_mixed_system_ttls_keep_source_order_and_the_longest_pin() {
    for (first, second) in [("5m", "1h"), ("1h", "5m")] {
        let mock = NativeMock::new(true, false, false).await;
        let (server, shared, cache) =
            proxy(vec![deployment(CONVERSE_MODEL, &mock)], json!([])).await;
        let mut body = request();
        let prefix = body["messages"][0]["content"].clone();
        body["messages"][0]["content"] = json!([
            {"type":"text","text":prefix,"cache_control":{"type":"ephemeral","ttl":first}},
            {"type":"text","text":"tail","cache_control":{"type":"ephemeral","ttl":second}},
        ]);
        post(&server, &body).await;
        let native = mock.only_body();
        let system = native["system"].as_array().unwrap();
        assert_eq!(system.len(), 4);
        assert!(system[0]["text"] == prefix, "the prefix must stay first");
        assert_eq!(
            system[1],
            json!({"cachePoint":{"type":"default","ttl":first}})
        );
        assert_eq!(system[2], json!({"text":"tail"}));
        assert_eq!(
            system[3],
            json!({"cachePoint":{"type":"default","ttl":second}})
        );
        assert_pin(&cache, &shared.load_full(), &body, 0, ONE_HOUR_PIN_TTL).await;
    }
}

#[tokio::test]
async fn converse_regional_wildcard_and_base_model_pins_follow_forwarded_ttl() {
    for (configured, requested, hint) in [
        (REGIONAL_MODEL, "model", None),
        ("bedrock/*", REGIONAL_MODEL, None),
        (PROFILE, "model", Some(("base_model", REGIONAL_MODEL))),
        (PROFILE, "model", Some(("model_info", REGIONAL_MODEL))),
    ] {
        let mock = NativeMock::new(true, false, false).await;
        let mut dep = deployment(configured, &mock);
        if configured.contains('*') {
            dep["model_name"] = json!("*");
        }
        if let Some((location, hint)) = hint {
            if location == "model_info" {
                dep["model_info"] = json!({"base_model":hint});
            } else {
                dep["oxllm_params"][location] = json!(hint);
            }
        }
        let (server, shared, cache) = proxy(vec![dep], json!([])).await;
        let mut body = request();
        body["model"] = json!(requested);
        body["messages"][0]["cache_control"] = control();
        post(&server, &body).await;
        let native = mock.only_body();
        assert_eq!(
            native["system"][1]["cachePoint"]["ttl"], "1h",
            "{configured}, {hint:?}"
        );
        assert!(
            native.get("model").is_none(),
            "runtime identity stays in the URL"
        );
        let runtime = if configured.contains('*') {
            requested
        } else {
            configured
        };
        let path = mock.requests.lock().unwrap()[0].0.clone();
        assert_eq!(
            percent_encoding::percent_decode_str(&path)
                .decode_utf8()
                .unwrap(),
            format!(
                "/model/{}/converse",
                runtime.strip_prefix("bedrock/").unwrap()
            )
        );
        assert_pin(&cache, &shared.load_full(), &body, 0, ONE_HOUR_PIN_TTL).await;
    }
}

#[tokio::test]
async fn native_schema_properties_named_cache_control_or_cache_point_do_not_extend_pin() {
    for converse in [false, true] {
        let mock = NativeMock::new(converse, false, false).await;
        let model = if converse {
            CONVERSE_MODEL
        } else {
            ANTHROPIC_MODEL
        };
        let (server, shared, cache) = proxy(vec![deployment(model, &mock)], json!([])).await;
        let mut body = request();
        let mut definition = tool();
        definition["function"]["parameters"]["properties"] = json!({
            "cache_control":{"type":"object", "default":{"type":"ephemeral","ttl":"1h"}},
            "cachePoint":{"type":"object", "default":{"type":"default","ttl":"1h"}},
            "payload":{"type":"object", "cache_control":{"type":"ephemeral","ttl":"1h"},
                "cachePoint":{"type":"default","ttl":"1h"}},
        });
        body["tools"] = json!([definition]);
        post(&server, &body).await;
        let native = mock.only_body();
        let schema = if converse {
            &native["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
        } else {
            &native["tools"][0]["input_schema"]
        };
        assert_eq!(
            schema["properties"]["cache_control"]["default"]["ttl"],
            "1h"
        );
        assert_eq!(schema["properties"]["cachePoint"]["default"]["ttl"], "1h");
        assert_pin(&cache, &shared.load_full(), &body, 0, PIN_TTL).await;
    }
}

#[tokio::test]
async fn post_transform_drops_and_extra_body_define_the_persisted_pin_ttl() {
    for (label, converse, raw_mark, extra, drops, ttl) in [
        (
            "native system added",
            false,
            false,
            json!({"system":[{"type":"text","text":"configured","cache_control":control()}]}),
            json!([]),
            ONE_HOUR_PIN_TTL,
        ),
        (
            "native system replaced",
            false,
            true,
            json!({"system":"uncached replacement"}),
            json!([]),
            PIN_TTL,
        ),
        (
            "native system dropped",
            false,
            true,
            json!({}),
            json!(["system"]),
            PIN_TTL,
        ),
        (
            "drops beat extra body",
            false,
            true,
            json!({"system":[{"type":"text","text":"configured","cache_control":control()}]}),
            json!(["system"]),
            PIN_TTL,
        ),
        (
            "native tool point added",
            true,
            false,
            json!({"toolConfig":native_tool_config()}),
            json!([]),
            ONE_HOUR_PIN_TTL,
        ),
        (
            "native system replaced",
            true,
            true,
            json!({"system":[{"text":"uncached replacement"}]}),
            json!([]),
            PIN_TTL,
        ),
        (
            "native tool point dropped",
            true,
            false,
            json!({"toolConfig":native_tool_config()}),
            json!(["toolConfig"]),
            PIN_TTL,
        ),
    ] {
        let mock = NativeMock::new(converse, false, false).await;
        let model = if converse {
            CONVERSE_MODEL
        } else {
            ANTHROPIC_MODEL
        };
        let mut dep = deployment(model, &mock);
        dep["oxllm_params"]["extra_body"] = extra.clone();
        dep["oxllm_params"]["additional_drop_params"] = drops.clone();
        let (server, shared, cache) = proxy(vec![dep], json!([])).await;
        let mut body = request();
        if raw_mark {
            body["messages"][0]["cache_control"] = control();
        }
        post(&server, &body).await;
        let native = mock.only_body();
        for key in drops.as_array().unwrap() {
            assert!(
                native.get(key.as_str().unwrap()).is_none(),
                "{label}: {native}"
            );
        }
        for (key, value) in extra.as_object().unwrap() {
            if !drops.as_array().unwrap().contains(&json!(key)) {
                assert_eq!(&native[key], value, "{label}");
            }
        }
        assert_pin(&cache, &shared.load_full(), &body, 0, ttl).await;
    }
}

#[tokio::test]
async fn fallback_and_streaming_pin_only_the_serving_native_attempt() {
    for stream in [false, true] {
        for serving_hour in [false, true] {
            let primary = NativeMock::new(false, stream, true).await;
            let serving = NativeMock::new(false, stream, false).await;
            let mut first = deployment(ANTHROPIC_MODEL, &primary);
            first["model_name"] = json!("primary");
            let mut fallback = deployment(ANTHROPIC_MODEL, &serving);
            fallback["model_name"] = json!("backup");
            let marked_system =
                json!({"system":[{"type":"text","text":"configured","cache_control":control()}]});
            first["oxllm_params"]["extra_body"] = if serving_hour {
                json!({})
            } else {
                marked_system.clone()
            };
            fallback["oxllm_params"]["extra_body"] = if serving_hour {
                marked_system
            } else {
                json!({})
            };
            let (server, shared, cache) =
                proxy(vec![first, fallback], json!([{"primary":["backup"]}])).await;
            let mut body = request();
            body["model"] = json!("primary");
            body["stream"] = json!(stream);
            let response = post(&server, &body).await;
            if stream {
                assert!(
                    response.contains("[DONE]"),
                    "stream must complete: {response}"
                );
            }
            assert_eq!(primary.only_body()["system"].is_array(), !serving_hour);
            assert_eq!(serving.only_body()["system"].is_array(), serving_hour);
            let state = shared.load_full();
            assert!(
                state
                    .router
                    .cooldown_store()
                    .is_cooling(&state.router.cooldown_key(&state.deployments[0]))
            );
            assert!(
                !state
                    .router
                    .cooldown_store()
                    .is_cooling(&state.router.cooldown_key(&state.deployments[1]))
            );
            assert_pin(
                &cache,
                &state,
                &body,
                1,
                if serving_hour {
                    ONE_HOUR_PIN_TTL
                } else {
                    PIN_TTL
                },
            )
            .await;
        }
    }
}

#[tokio::test]
async fn fallback_between_anthropic_and_converse_observes_only_the_serving_wire() {
    for stream in [false, true] {
        for serving_converse in [false, true] {
            let primary = NativeMock::new(!serving_converse, stream, true).await;
            let serving = NativeMock::new(serving_converse, stream, false).await;
            let mut first = deployment(
                if serving_converse {
                    ANTHROPIC_MODEL
                } else {
                    CONVERSE_MODEL
                },
                &primary,
            );
            first["model_name"] = json!("primary");
            let mut fallback = deployment(
                if serving_converse {
                    CONVERSE_MODEL
                } else {
                    ANTHROPIC_MODEL
                },
                &serving,
            );
            fallback["model_name"] = json!("backup");
            let (server, shared, cache) =
                proxy(vec![first, fallback], json!([{"primary":["backup"]}])).await;
            let mut body = request();
            body["model"] = json!("primary");
            body["stream"] = json!(stream);
            let mut marked = tool();
            marked["cache_control"] = control();
            body["tools"] = json!([marked]);
            post(&server, &body).await;
            let first = primary.only_body();
            let served = serving.only_body();
            let (anthropic, converse) = if serving_converse {
                (&first, &served)
            } else {
                (&served, &first)
            };
            assert_eq!(anthropic["tools"][0]["cache_control"], control());
            let tools = converse["toolConfig"]["tools"].as_array().unwrap();
            assert_eq!(tools.len(), 1);
            assert!(tools[0].get("toolSpec").is_some());
            assert!(tools[0].get("cachePoint").is_none());
            assert_pin(
                &cache,
                &shared.load_full(),
                &body,
                1,
                if serving_converse {
                    PIN_TTL
                } else {
                    ONE_HOUR_PIN_TTL
                },
            )
            .await;
        }
    }
}

#[tokio::test]
async fn selected_deployment_and_locally_failed_attempt_clear_prior_pin_ttl_observations() {
    let mock = NativeMock::new(false, false, false).await;
    let (_server, shared, cache) = proxy(vec![deployment(ANTHROPIC_MODEL, &mock)], json!([])).await;
    let state = shared.load_full();
    let deployment = &state.deployments[0];
    let mut ctx = oxllm_core::CallContext::for_test("model");
    assert!(ctx.upstream_prompt_cache_pin_ttl.is_none());
    ctx.upstream_prompt_cache_pin_ttl = Some(ONE_HOUR_PIN_TTL);
    crate::pipeline::prepare_deployment_pricing(&state, &mut ctx, deployment, &request())
        .await
        .unwrap();
    assert!(
        ctx.upstream_prompt_cache_pin_ttl.is_none(),
        "a new pick clears the previous attempt"
    );

    ctx.upstream_prompt_cache_pin_ttl = Some(ONE_HOUR_PIN_TTL);
    let provider = crate::orchestration::upstream::resolve_provider(&state, deployment).unwrap();
    let mut invalid = request();
    invalid["timeout"] = json!(-1);
    assert!(
        crate::orchestration::upstream::attempt_upstream_with_context(
            &state,
            deployment,
            &invalid,
            false,
            &provider,
            None,
            Some(&mut ctx),
        )
        .await
        .is_err()
    );
    assert!(
        ctx.upstream_prompt_cache_pin_ttl.is_none(),
        "even failure before shaping clears a previous observation"
    );
    assert!(mock.requests.lock().unwrap().is_empty());
    assert!(cache.pins.lock().unwrap().is_empty());
}
