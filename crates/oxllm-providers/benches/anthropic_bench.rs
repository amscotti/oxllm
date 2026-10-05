use criterion::{Criterion, criterion_group, criterion_main};
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, ResolvedCredentials};
use oxllm_providers::anthropic::AnthropicProvider;
use serde_json::json;

fn make_deployment() -> Deployment {
    Deployment {
        model_name: "claude-haiku".to_string(),
        oxllm_params: OxllmParams {
            model: "anthropic/claude-haiku-4-5-20251001".to_string(),
            api_key: Some("test-key".to_string()),
            extra: Default::default(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn simple_request() -> serde_json::Value {
    json!({
    "model": "claude-haiku",
    "messages": [
    {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
    })
}

fn complex_request() -> serde_json::Value {
    json!({
    "model": "claude-haiku",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
    })
}

fn request_with_tools() -> serde_json::Value {
    json!({
    "model": "claude-haiku",
    "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    "max_tokens": 100,
    "tools": [{
    "type": "function",
    "function": {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
    "type": "object",
    "properties": {
    "city": {"type": "string"},
    "unit": {"type": "string", "enum": ["c", "f"]}
    },
    "required": ["city"]
    }
    }
    }],
    "tool_choice": "auto"
    })
}

fn anthropic_response() -> serde_json::Value {
    json!({
    "id": "msg_0123456789",
    "type": "message",
    "role": "assistant",
    "content": [
    {"type": "text", "text": "The capital of France is Paris. It is known as the City of Light."}
    ],
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "end_turn",
    "usage": {
    "input_tokens": 25,
    "output_tokens": 15
    }
    })
}

fn anthropic_tool_response() -> serde_json::Value {
    json!({
    "id": "msg_0123456789",
    "type": "message",
    "role": "assistant",
    "content": [
    {"type": "text", "text": "I'll check the weather for you."},
    {"type": "tool_use", "id": "toolu_01abc", "name": "get_weather", "input": {"city": "Tokyo", "unit": "c"}}
    ],
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "tool_use",
    "usage": {
    "input_tokens": 30,
    "output_tokens": 20
    }
    })
}

fn stream_chunk(event_type: &str) -> String {
    match event_type {
        "delta" => {
            let data = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "The capital of France is Paris."}
            });
            format!("event: content_block_delta\ndata: {data}\n\n")
        }
        "start" => {
            let data = json!({
            "type": "message_start",
            "message": {
            "id": "msg_0123456789",
            "role": "assistant",
            "content": [],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": null,
            "usage": {"input_tokens": 25, "output_tokens": 1}
            }
            });
            format!("event: message_start\ndata: {data}\n\n")
        }
        "stop" => {
            let data = json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15}
            });
            format!("event: message_delta\ndata: {data}\n\n")
        }
        _ => String::new(),
    }
}

fn transform_request_benches(c: &mut Criterion) {
    let provider = AnthropicProvider;
    let deployment = make_deployment();
    c.bench_function("transform_request_simple", |b| {
        let req = simple_request();
        assert!(provider.transform_request(&req, &deployment).is_ok());
        b.iter(|| provider.transform_request(&req, &deployment));
    });
    c.bench_function("transform_request_complex", |b| {
        let req = complex_request();
        assert!(provider.transform_request(&req, &deployment).is_ok());
        b.iter(|| provider.transform_request(&req, &deployment));
    });
    c.bench_function("transform_request_with_tools", |b| {
        let req = request_with_tools();
        assert!(provider.transform_request(&req, &deployment).is_ok());
        b.iter(|| provider.transform_request(&req, &deployment));
    });
}

fn transform_response_benches(c: &mut Criterion) {
    let provider = AnthropicProvider;
    c.bench_function("transform_response_text", |b| {
        let resp = anthropic_response();
        assert!(provider.transform_response(&resp, "claude-haiku").is_ok());
        b.iter(|| provider.transform_response(&resp, "claude-haiku"));
    });
    c.bench_function("transform_response_tool_use", |b| {
        let resp = anthropic_tool_response();
        assert!(provider.transform_response(&resp, "claude-haiku").is_ok());
        b.iter(|| provider.transform_response(&resp, "claude-haiku"));
    });
}

fn transform_stream_chunk_benches(c: &mut Criterion) {
    let provider = AnthropicProvider;
    c.bench_function("transform_stream_chunk_delta", |b| {
        let chunk = stream_chunk("delta");
        b.iter(|| provider.transform_stream_chunk(&chunk));
    });
    c.bench_function("transform_stream_chunk_start", |b| {
        let chunk = stream_chunk("start");
        b.iter(|| provider.transform_stream_chunk(&chunk));
    });
    c.bench_function("transform_stream_chunk_stop", |b| {
        let chunk = stream_chunk("stop");
        b.iter(|| provider.transform_stream_chunk(&chunk));
    });
}

fn transform_stream_chunk_stateful_benches(c: &mut Criterion) {
    let provider = AnthropicProvider;
    let request = simple_request();
    c.bench_function("transform_stream_chunk_stateful_delta", |b| {
        let chunk = stream_chunk("delta");
        let mut state = None;
        b.iter(|| provider.transform_stream_chunk_stateful(&chunk, &mut state, &request));
    });
    c.bench_function("transform_stream_chunk_stateful_start", |b| {
        let chunk = stream_chunk("start");
        let mut state = None;
        b.iter(|| provider.transform_stream_chunk_stateful(&chunk, &mut state, &request));
    });
    c.bench_function("transform_stream_chunk_stateful_stop", |b| {
        let chunk = stream_chunk("stop");
        let mut state = None;
        b.iter(|| provider.transform_stream_chunk_stateful(&chunk, &mut state, &request));
    });
}

fn auth_and_endpoint_benches(c: &mut Criterion) {
    let provider = AnthropicProvider;
    let deployment = make_deployment();
    let creds = ResolvedCredentials::from_key("sk-ant-test-key");
    c.bench_function("apply_auth_headers", |b| {
        let mut headers = std::collections::HashMap::new();
        b.iter(|| {
            headers.clear();
            provider.apply_auth(&deployment, &creds, &mut headers);
        });
    });
    c.bench_function("endpoint_url", |b| {
        b.iter(|| provider.endpoint_url(&deployment));
    });
}

criterion_group!(
    benches,
    transform_request_benches,
    transform_response_benches,
    transform_stream_chunk_benches,
    transform_stream_chunk_stateful_benches,
    auth_and_endpoint_benches,
);
criterion_main!(benches);
