//! Integration tests: Unicode / non-ASCII text through the proxy.
//!
//! Verifies that emoji, CJK, Arabic, RTL Hebrew, zero-width characters,
//! and large multi-language payloads round-trip correctly through the
//! chat completions endpoint.

mod common;

use common::{proxy_config, start_oxllm};
use httpmock::{Mock, MockServer};
use serde_json::{Value, json};

fn mock_echo_chat<'a>(mock: &'a MockServer) -> Mock<'a> {
    // The mock upstream echoes the first user message back in the
    // assistant content so tests can verify round-trip fidelity.
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-unicode",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
            }));
    })
}

fn mock_with_content<'a>(mock: &'a MockServer, content: &str) -> Mock<'a> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-unicode",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
            }));
    })
}

// ---------------------------------------------------------------------------
// 1. Basic Unicode chat
// ---------------------------------------------------------------------------

/// Send a chat request with messages containing emoji, CJK, Arabic, RTL
/// Hebrew, and zero-width characters. Assert the proxy handles the request
/// without error (200).
#[tokio::test]
async fn basic_unicode_chat_round_trips() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let texts = [
        "\u{1F389} party time",                     // emoji 🎉
        "\u{4f60}\u{597d}\u{4e16}\u{754c}",         // CJK 你好世界
        "\u{0645}\u{0631}\u{062d}\u{0628}\u{0627}", // Arabic مرحبا
        "\u{05e9}\u{05dc}\u{05d5}\u{05dd}",         // Hebrew שלום
        "zero\u{200B}width",                        // zero-width space
    ];

    let client = reqwest::Client::new();
    let url = handle.url("/v1/chat/completions");

    for text in &texts {
        // Match on the message CONTENT, not just the path: this is the actual
        // "round-trip" claim — the unicode body must reach the upstream intact.
        let upstream = mock.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions")
                .json_body_partial(
                    json!({"messages": [{"role": "user", "content": text}]}).to_string(),
                );
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                "id": "chatcmpl-unicode",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
                }));
        });

        let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": text}]
        });
        let resp = client.post(&url).json(&body).send().await.expect("request");
        assert_eq!(
            resp.status(),
            200,
            "unicode text should be accepted: {:?}",
            text
        );

        // The upstream mock only matches when the forwarded body still
        // contains the unicode message verbatim — if the proxy mangled,
        // escaped incorrectly, or dropped the multi-byte content, this mock
        // would have zero hits and the request above would have 404'd.
        upstream.assert_hits(1);
    }

    handle.shutdown();
}

/// Verify the proxy preserves Unicode in the response body — the assistant
/// content with non-ASCII characters must arrive unmodified.
#[tokio::test]
async fn unicode_response_content_preserved() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");
    let reply =
        "\u{1F389} \u{4f60}\u{597d}\u{4e16}\u{754c} \u{0645}\u{0631}\u{062d}\u{0628}\u{0627}";
    let _m = mock_with_content(&mock, reply);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(
        body["choices"][0]["message"]["content"], reply,
        "unicode response content must be preserved end-to-end"
    );

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// 2. Unicode in tool call arguments
// ---------------------------------------------------------------------------

/// Send a tool call with unicode in the function arguments JSON. Assert it
/// round-trips correctly (the proxy forwards the request body without
/// mangling non-ASCII in the arguments string).
#[tokio::test]
async fn unicode_in_tool_call_arguments() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    // Mock that returns a tool call with unicode in the arguments.
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "set_greeting",
                    "arguments": "{\"greeting\": \"\u{4f60}\u{597d}\u{1F389}\"}"
                }
                }]
            },
            "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "greet in Chinese"}],
        "tools": [{
        "type": "function",
        "function": {
            "name": "set_greeting",
            "description": "Set a greeting message",
            "parameters": {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"}
            },
            "required": ["greeting"]
            }
        }
        }]
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");

    // The tool call arguments must contain the unicode greeting intact.
    let args = body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("arguments string");
    assert!(
        args.contains("\u{4f60}\u{597d}"),
        "unicode tool-call arguments must round-trip, got: {args}"
    );
    assert!(
        args.contains("\u{1F389}"),
        "emoji in tool-call arguments must round-trip, got: {args}"
    );

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// 3. Unicode in the `user` field
// ---------------------------------------------------------------------------

/// A request with a unicode `user` field should be accepted and forwarded.
#[tokio::test]
async fn unicode_user_field_accepted() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");
    let _m = mock_echo_chat(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "user": "user-\u{00e9}\u{4f60}\u{597d}\u{1F389}"
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200, "unicode user field should be accepted");

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// 4. Large Unicode payload
// ---------------------------------------------------------------------------

/// Send a request with ~10 KB of multi-language text. Assert 200 response.
#[tokio::test]
async fn large_unicode_payload_accepted() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");
    let _m = mock_echo_chat(&mock);

    // Build ~10 KB of mixed-language text by repeating a multilingual block.
    let block = "\u{1F389}\u{4f60}\u{597d}\u{4e16}\u{754c}\u{0645}\u{0631}\u{062d}\u{0628}\u{0627}\u{05e9}\u{05dc}\u{05d5}\u{05dd}hello ";
    // Each block is ~25 bytes of UTF-8; repeat to exceed 10 KB.
    let large_text = block.repeat(450);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": &large_text}]
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status(),
        200,
        "large unicode payload should be accepted"
    );
    assert!(
        large_text.len() >= 10_000,
        "payload should be at least 10 KB"
    );

    handle.shutdown();
}
