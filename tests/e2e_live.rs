//! Live end-to-end tests: start a real oxllm server and make actual API
//! calls through it to verify the full gateway pipeline.
//!
//! These tests are **auto-ignored** unless the `live-tests` feature is
//! enabled:
//!
//! ```sh
//! # Normal dev — tests compile but are skipped (shown as "ignored")
//! cargo test
//!
//! # Run live tests for providers whose API keys are in your environment:
//! cargo test --features live-tests --test e2e_live
//!
//! # Explicit key:
//! DEEPSEEK_API_KEY=sk-... cargo test --features live-tests --test e2e_live
//! ```
//!
//! Each test self-gates on its provider's API key env var. A missing key
//! causes a silent skip (the test passes with a printed message). Only
//! providers with keys actually execute.
//!
//! ## CI/CD
//!
//! ```yaml
//! on:
//!   schedule: [{cron: '0 6 * * *'}]  # daily 6am UTC
//!   workflow_dispatch: {}
//! jobs:
//!   live-e2e:
//!     steps:
//!       - run: cargo test --features live-tests --test e2e_live
//!     env:
//!       DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
//!       OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
//!       # ... only providers with configured secrets run
//! ```

mod common;

use common::start_oxllm;
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read an API key from the environment. Returns `None` if unset or empty.
fn key(env_var: &str) -> Option<String> {
    std::env::var(env_var).ok().filter(|s| !s.is_empty())
}

/// Build a minimal one-deployment config pointing at a real provider.
fn live_config(model_name: &str, provider_model: &str, api_key: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: provider_model.to_string(),
                api_key: Some(api_key.to_string()),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            drop_params: Some(true),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            master_key: Some(AUTH.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: RouterSettings::default(),
        extra: serde_json::Map::new(),
    }
}

/// The master key used for auth in all tests (must be ≥16 chars per the
/// production safety guard added in the C05 security review).
const AUTH: &str = "sk-e2e-live-test-master-key";

fn live_client() -> reqwest::Client {
    common::http_client_with_timeout(std::time::Duration::from_secs(180))
}

/// An unavailable feature is reported only for a parameter-specific response,
/// not for an arbitrary 400/5xx/auth failure.
fn unsupported_parameter(status: reqwest::StatusCode, body: &Value, params: &[&str]) -> bool {
    if status != 400 && status != 422 {
        return false;
    }
    let error = &body["error"];
    let detail = format!(
        "{} {}",
        error["message"].as_str().unwrap_or(""),
        error["param"].as_str().unwrap_or("")
    )
    .to_ascii_lowercase();
    let unsupported = error["code"] == "unsupported_parameter"
        || [
            "not supported",
            "unsupported",
            "does not support",
            "not available",
            "unavailable",
            "no native representation",
            "no lossless",
        ]
        .iter()
        .any(|phrase| detail.contains(*phrase));
    unsupported && params.iter().any(|param| detail.contains(*param))
}

#[test]
fn native_mode_unavailability_is_not_confused_with_schema_or_auth_failures() {
    let unavailable = json!({"error":{"message":"response_format json_object has no native representation without a schema; use json_schema"}});
    assert!(unsupported_parameter(
        reqwest::StatusCode::BAD_REQUEST,
        &unavailable,
        &["json_object", "response_format"]
    ));
    assert!(!unsupported_parameter(
        reqwest::StatusCode::BAD_REQUEST,
        &json!({"error":{"message":"invalid JSON schema"}}),
        &["json_schema", "response_format"]
    ));
    assert!(!unsupported_parameter(
        reqwest::StatusCode::UNAUTHORIZED,
        &unavailable,
        &["json_object", "response_format"]
    ));
}

/// Run the full e2e suite against one provider: chat, streaming, tool calls,
/// response headers, and pricing verification.
async fn run_suite(label: &str, config: ProxyConfig, model: &str, max_tokens: u64) {
    let (handle, _state) = start_oxllm(config).await.expect("server should start");
    let client = live_client();
    let url = handle.url("/v1/chat/completions");

    // 1. Chat
    {
        let resp = client
            .post(&url)
            .bearer_auth(AUTH)
            .json(&json!({
                "model": model,
                "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
                "max_tokens": max_tokens,
            }))
            .send()
            .await
            .expect("chat request");
        let status = resp.status();
        let cost_header = resp.headers().get("x-oxllm-response-cost").cloned();
        let body: Value = resp.json().await.expect("chat json");
        assert_eq!(
            status, 200,
            "[{label}] chat: expected 200, got {status}\n{body}"
        );
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(!content.is_empty(), "[{label}] chat: empty content");
        let pt = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        let ct = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        assert!(pt > 0, "[{label}] chat: no prompt tokens in usage");
        assert!(ct > 0, "[{label}] chat: no completion tokens in usage");
        println!(
            "  [{label}] chat ✅  model={} tokens={pt}+{ct}",
            body["model"]
        );
        // Presence is contractual when token pricing is available for the
        // returned model/provider. Provider cache/reasoning buckets can alter
        // the amount, so this smoke check does not invent an exact price.
        let priced = oxllm_core::calculate_cost_with_provider(
            body["model"].as_str().expect("response model"),
            &oxllm_core::UsageSummary {
                prompt_tokens: pt,
                completion_tokens: ct,
                total_tokens: pt + ct,
                ..Default::default()
            },
            label,
        )
        .is_some();
        if priced {
            assert!(
                cost_header.is_some(),
                "[{label}] priced response is missing its cost header"
            );
        }
        if let Some(cost) = cost_header {
            let cost_val: f64 = cost
                .to_str()
                .expect("ASCII cost header")
                .parse()
                .expect("numeric cost header");
            assert!(
                cost_val.is_finite() && cost_val >= 0.0,
                "[{label}] invalid cost header: {cost_val}"
            );
            println!("  [{label}] cost header validated: ${cost_val:.6}");
        } else {
            println!("  [{label}] cost: SKIP (returned model has no token-pricing entry)");
        }
    }

    // 2. Streaming
    {
        let resp = client
            .post(&url)
            .bearer_auth(AUTH)
            .json(&json!({
                "model": model,
                "stream": true,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": max_tokens.min(100),
            }))
            .send()
            .await
            .expect("stream request");
        assert_eq!(resp.status(), 200, "[{label}] stream: bad status");
        let text = resp.text().await.expect("stream body");
        assert!(
            text.contains("data:"),
            "[{label}] stream: no SSE data lines"
        );
        assert!(text.contains("[DONE]"), "[{label}] stream: no [DONE]");
        let chunk_count = text.matches("data:").count();
        println!("  [{label}] stream ✅  chunks={chunk_count}");
    }

    // 3. Tool calls
    {
        let resp = client
            .post(&url)
            .bearer_auth(AUTH)
            .json(&json!({
                "model": model,
                "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }],
                "max_tokens": max_tokens,
            }))
            .send()
            .await
            .expect("tool request");

        let status = resp.status();
        let body: Value = resp.json().await.expect("tool json");

        if unsupported_parameter(status, &body, &["tools", "tool_choice"]) {
            println!("  [{label}] tools: SKIP (upstream explicitly rejects tool support)");
        } else {
            assert_eq!(
                status, 200,
                "[{label}] tools: expected 200, got {status}\n{body}"
            );
            let finish = body["choices"][0]["finish_reason"].as_str().unwrap_or("");
            let msg = &body["choices"][0]["message"];
            let has_tool_calls = msg["tool_calls"]
                .as_array()
                .is_some_and(|calls| !calls.is_empty());
            let has_content = msg["content"].as_str().is_some_and(|s| !s.is_empty());
            assert!(
                has_tool_calls || has_content,
                "[{label}] tools: no tool_calls and no content in response\n{body}"
            );
            if has_tool_calls {
                for tc in msg["tool_calls"].as_array().unwrap() {
                    assert_eq!(
                        tc["function"]["name"], "get_weather",
                        "[{label}] unexpected tool name"
                    );
                    let arguments: Value = serde_json::from_str(
                        tc["function"]["arguments"]
                            .as_str()
                            .expect("tool arguments string"),
                    )
                    .expect("tool arguments must be JSON");
                    assert!(
                        arguments["location"].is_string(),
                        "[{label}] missing tool location"
                    );
                }
                println!("  [{label}] tools ✅  get_weather arguments validated; finish={finish}");
            } else {
                println!("  [{label}] tools ✅  (content response, no tool call) finish={finish}");
            }
        }
    }

    // 4. Test schema output and JSON mode independently. Unsupported schema
    // constraints must not turn into a pass using a weaker output request.
    {
        let schema_request = json!({
            "model": model,
            "messages": [{"role": "user", "content": "Generate a person named Alice who is 30 years old. Respond as JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        },
                        "required": ["name", "age"],
                        "additionalProperties": false
                    }
                }
            },
            "max_tokens": max_tokens,
        });
        let object_request = json!({
            "model": model,
            "messages": [{"role": "user", "content": "Generate a person named Alice who is 30 years old. Respond as JSON with 'name' and 'age' fields."}],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
        });

        for (fmt_used, request) in [
            ("json_schema", schema_request),
            ("json_object", object_request),
        ] {
            let structured = {
                let resp = client
                    .post(&url)
                    .bearer_auth(AUTH)
                    .json(&request)
                    .send()
                    .await
                    .expect("structured output request");
                let status = resp.status();
                let body: Value = resp.json().await.expect("structured output json");
                if status.is_success() {
                    Some((fmt_used, body))
                } else if unsupported_parameter(status, &body, &[fmt_used, "response_format"]) {
                    println!("  [{label}] {fmt_used}: SKIP (native format unavailable)");
                    None
                } else {
                    panic!("[{label}] {fmt_used} request failed with {status}: {body}");
                }
            };

            if let Some((fmt_used, body)) = structured {
                let choice = &body["choices"][0];
                let message = &choice["message"];
                // Strict schema guarantees do not cover truncation or refusal.
                // Report these as unevaluated, never as a valid structured result.
                if matches!(
                    choice["finish_reason"].as_str(),
                    Some("length" | "content_filter" | "refusal")
                ) || message["refusal"].as_str().is_some_and(|s| !s.is_empty())
                {
                    println!("  [{label}] structured output: SKIP (truncated/refused response)");
                } else {
                    let content = message["content"]
                        .as_str()
                        .expect("structured text content");
                    let parsed: Value = serde_json::from_str(content)
                        .unwrap_or_else(|e| panic!("[{label}] invalid {fmt_used} content: {e}"));
                    assert!(
                        parsed.is_object(),
                        "[{label}] {fmt_used} must produce a JSON object"
                    );
                    if fmt_used == "json_schema" {
                        assert!(
                            parsed["name"].is_string(),
                            "[{label}] schema requires name:string"
                        );
                        // JSON Schema integer includes 30.0 as well as 30.
                        assert!(
                            parsed["age"].as_f64().is_some_and(|age| age.fract() == 0.0),
                            "[{label}] schema requires age:integer"
                        );
                        assert_eq!(
                            parsed.as_object().unwrap().len(),
                            2,
                            "additionalProperties=false"
                        );
                    }
                    println!("  [{label}] structured output ✅  {fmt_used} contract validated");
                }
            }
        }
    }

    // 5. Reasoning-effort response contract. Effort is a control signal, not
    // a guarantee that two random generations have monotonic token counts.
    {
        let question = "How many r's are in strawberry? Think step by step.";
        let extract_reasoning = |body: &Value| -> (u64, u64, bool) {
            let u = &body["usage"];
            let reasoning = u["completion_tokens_details"]["reasoning_tokens"]
                .as_u64()
                .or_else(|| u["reasoning_tokens"].as_u64())
                .unwrap_or(0);
            let completion = u["completion_tokens"].as_u64().unwrap_or(0);
            // Some providers (Anthropic adaptive thinking) don't report
            // reasoning_tokens separately but do produce longer completions.
            // Check reasoning_content in the message as a secondary signal.
            let has_reasoning_content = body["choices"][0]["message"]["reasoning_content"]
                .as_str()
                .is_some_and(|s| !s.is_empty());
            (reasoning, completion, has_reasoning_content)
        };

        let mut responses = Vec::new();
        for effort in ["low", "high"] {
            let response = client
                .post(&url)
                .bearer_auth(AUTH)
                .json(&json!({
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "reasoning_effort": effort,
                    "max_tokens": max_tokens,
                }))
                .send()
                .await
                .expect("reasoning request");
            let status = response.status();
            let body: Value = response.json().await.expect("reasoning JSON");
            if unsupported_parameter(status, &body, &["reasoning_effort", "effort"]) {
                println!("  [{label}] reasoning: SKIP (effort explicitly unsupported)");
                responses.clear();
                break;
            }
            assert_eq!(
                status, 200,
                "[{label}] {effort} reasoning request failed: {body}"
            );
            assert!(
                body["choices"][0]["message"].is_object(),
                "[{label}] missing reasoning response message"
            );
            assert!(
                body["usage"]["completion_tokens"].as_u64().is_some(),
                "[{label}] invalid completion usage"
            );
            for field in [
                &body["usage"]["completion_tokens_details"]["reasoning_tokens"],
                &body["usage"]["reasoning_tokens"],
            ] {
                assert!(
                    field.is_null() || field.as_u64().is_some(),
                    "[{label}] invalid reasoning usage"
                );
            }
            responses.push(body);
        }
        if let [low_body, high_body] = responses.as_slice() {
            let (low_r, low_c, low_rc) = extract_reasoning(low_body);
            let (high_r, high_c, high_rc) = extract_reasoning(high_body);
            println!(
                "  [{label}] reasoning observations: low={low_r}/{low_c}, high={high_r}/{high_c} reasoning/completion tokens; reasoning content low={low_rc} high={high_rc}"
            );
        }
    }

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// Provider tests
// ---------------------------------------------------------------------------

/// The attribute that auto-ignores tests unless `--features live-tests` is set.
/// Tests are always compiled (catching bit-rot) but only execute when the
/// feature is enabled AND the provider's API key is in the environment.
macro_rules! live_test {
    ($item:item) => {
        #[cfg_attr(
            not(feature = "live-tests"),
            ignore = "enable with --features live-tests"
        )]
        $item
    };
}

live_test! {
    #[tokio::test]
    async fn e2e_deepseek() {
        let Some(key) = key("DEEPSEEK_API_KEY") else {
            eprintln!("skipping: DEEPSEEK_API_KEY not set");
            return;
        };
        println!("=== DeepSeek v4 Flash ===");
        run_suite(
            "deepseek",
            live_config("ds-flash", "deepseek/deepseek-v4-flash", &key),
            "ds-flash",
            500, // reasoning model: needs headroom for thinking tokens
        )
        .await;
    }
}

live_test! {
    #[tokio::test]
    async fn e2e_anthropic() {
        let Some(key) = key("ANTHROPIC_API_KEY") else {
            eprintln!("skipping: ANTHROPIC_API_KEY not set");
            return;
        };
        println!("=== Anthropic Claude Sonnet 5 ===");
        run_suite(
            "anthropic",
            live_config("claude", "anthropic/claude-sonnet-5", &key),
            "claude",
            200,
        )
        .await;
    }
}

live_test! {
    #[tokio::test]
    async fn e2e_openai() {
        let Some(key) = key("OPENAI_API_KEY") else {
            eprintln!("skipping: OPENAI_API_KEY not set");
            return;
        };
        println!("=== OpenAI GPT-5.6 ===");
        run_suite(
            "openai",
            live_config("gpt", "openai/gpt-5.6-luna", &key),
            "gpt",
            500, // reasoning model: needs headroom
        )
        .await;
    }
}

live_test! {
    #[tokio::test]
    async fn e2e_meta() {
        let Some(key) = key("META_API_KEY") else {
            eprintln!("skipping: META_API_KEY not set");
            return;
        };
        println!("=== Meta Muse Spark 1.2 ===");
        run_suite(
            "meta",
            live_config("muse", "meta/muse-spark-1.2", &key),
            "muse",
            500, // reasoning model
        )
        .await;
    }
}

live_test! {
    #[tokio::test]
    async fn e2e_bedrock() {
        // Bedrock uses AWS SigV4, not a bearer API key. The credentials are
        // read from the standard AWS env vars by the Bedrock credential
        // resolver.
        let Some(_) = key("AWS_ACCESS_KEY_ID") else {
            eprintln!("skipping: AWS_ACCESS_KEY_ID not set (Bedrock)");
            return;
        };
        let Some(_) = key("AWS_SECRET_ACCESS_KEY") else {
            eprintln!("skipping: AWS_SECRET_ACCESS_KEY not set (Bedrock)");
            return;
        };
        let region = key("AWS_REGION")
            .or_else(|| key("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|| "us-east-1".to_string());
        println!("=== Bedrock Claude Sonnet 4.6 (region={region}) ===");
        // Note: us.anthropic.claude-sonnet-5 exists on Bedrock but requires
        // Anthropic use-case form approval. Using 4-6 until access is granted.
        // Sonnet 5 is tested via the direct Anthropic API in e2e_anthropic.
        run_suite(
            "bedrock",
            live_config("bedrock-claude", "bedrock/us.anthropic.claude-sonnet-4-6", "dummy-not-used"),
            "bedrock-claude",
            200,
        )
        .await;
    }
}

live_test! {
    #[tokio::test]
    async fn e2e_xai() {
        let Some(key) = key("XAI_API_KEY") else {
            eprintln!("skipping: XAI_API_KEY not set");
            return;
        };
        println!("=== xAI Grok 4.6 ===");
        run_suite(
            "xai",
            live_config("grok", "xai/grok-4.6", &key),
            "grok",
            500,
        )
        .await;
    }
}

live_test! {
    #[tokio::test]
    async fn e2e_gemini() {
        let Some(key) = key("GEMINI_API_KEY") else {
            eprintln!("skipping: GEMINI_API_KEY not set");
            return;
        };
        println!("=== Google Gemini 3.8 Flash ===");
        run_suite(
            "gemini",
            live_config("gemini-flash", "gemini/gemini-3.8-flash", &key),
            "gemini-flash",
            500,
        )
        .await;
    }
}

// ---------------------------------------------------------------------------
// Cross-format test: /v1/messages (Anthropic format) → any OpenAI-compat
// ---------------------------------------------------------------------------

live_test! {
    #[tokio::test]
    async fn e2e_cross_format_messages_endpoint() {
        // Test the /v1/messages endpoint with a DeepSeek (OpenAI-compat)
        // backend. Verifies oxllm translates Anthropic Messages format →
        // OpenAI → response back to Anthropic format.
        let Some(key) = key("DEEPSEEK_API_KEY") else {
            eprintln!("skipping: DEEPSEEK_API_KEY not set (cross-format test)");
            return;
        };
        println!("=== Cross-format: /v1/messages → DeepSeek ===");
        let config = live_config("ds-flash", "deepseek/deepseek-v4-flash", &key);
        let (handle, _state) = start_oxllm(config).await.expect("server starts");
        let client = live_client();

        let resp = client
            .post(handle.url("/v1/messages"))
            .header("x-api-key", AUTH)
            .header("anthropic-version", "2023-06-01")
            .json(&json!({
                "model": "ds-flash",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": "Say hello in exactly 3 words"}],
            }))
            .send()
            .await
            .expect("messages request");

        let status = resp.status();
        let body: Value = resp.json().await.expect("messages json");
        assert_eq!(status, 200, "cross-format: expected 200, got {status}\n{body}");

        // Anthropic Messages response shape: content array with text blocks
        assert_eq!(body["type"], "message", "cross-format: missing type=message");
        let content = body["content"].as_array().expect("cross-format: content not array");
        assert!(!content.is_empty(), "cross-format: empty content array");
        // A reasoning upstream (deepseek-v4-flash emits reasoning_content)
        // legitimately converts to a leading `thinking` block, matching
        // Claude's own Messages shape — pick the first *text* block instead
        // of blindly indexing content[0].
        let text = content
            .iter()
            .find(|b| b["type"] == "text")
            .and_then(|b| b["text"].as_str())
            .unwrap_or("");
        assert!(!text.is_empty(), "cross-format: empty text in content block");
        let input_tokens = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
        let output_tokens = body["usage"]["output_tokens"].as_u64().unwrap_or(0);
        assert!(input_tokens > 0, "cross-format: no input_tokens");
        assert!(output_tokens > 0, "cross-format: no output_tokens");
        println!(
            "  [cross-format] ✅  model={} text=\"{text}\" tokens={input_tokens}+{output_tokens}",
            body["model"]
        );

        handle.shutdown();
    }
}
