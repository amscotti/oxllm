use criterion::{Criterion, criterion_group, criterion_main};
use oxllm_core::tokens::{estimate_request_tokens, estimate_tokens};

fn estimate_tokens_benches(c: &mut Criterion) {
    c.bench_function("estimate_tokens_short", |b| {
        let text = "Hello, world! This is a short test message.";
        b.iter(|| estimate_tokens(text, "gpt-4o"));
    });
    c.bench_function("estimate_tokens_medium", |b| {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        b.iter(|| estimate_tokens(&text, "gpt-4o"));
    });
    c.bench_function("estimate_tokens_large", |b| {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(500);
        b.iter(|| estimate_tokens(&text, "gpt-4o"));
    });
    c.bench_function("estimate_tokens_non_openai", |b| {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        b.iter(|| estimate_tokens(&text, "claude-3-haiku"));
    });
}

fn estimate_request_tokens_benches(c: &mut Criterion) {
    c.bench_function("estimate_request_tokens_small", |b| {
        let request = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"}
        ]
        });
        b.iter(|| estimate_request_tokens(&request));
    });
    c.bench_function("estimate_request_tokens_large", |b| {
        let long_content = "Lorem ipsum dolor sit amet. ".repeat(100);
        let request = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "system", "content": &long_content},
        {"role": "user", "content": &long_content}
        ]
        });
        b.iter(|| estimate_request_tokens(&request));
    });
}

fn estimate_request_tokens_tools_benches(c: &mut Criterion) {
    c.bench_function("estimate_request_tokens_with_tools", |b| {
        let request = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": [{
        "type": "function",
        "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city",
        "parameters": {
        "type": "object",
        "properties": {
        "city": {"type": "string", "description": "The city name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "forecast": {
        "type": "array",
        "items": {"type": "object"},
        "description": "Optional forecast days"
        }
        },
        "required": ["city"]
        }
        }
        }, {
        "type": "function",
        "function": {
        "name": "get_time",
        "description": "Get the current time in a timezone",
        "parameters": {
        "type": "object",
        "properties": {
        "timezone": {"type": "string"}
        },
        "required": ["timezone"]
        }
        }
        }],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        });
        b.iter(|| estimate_request_tokens(&request));
    });
    c.bench_function("estimate_request_tokens_with_image", |b| {
        let request = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{
        "role": "user",
        "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg", "detail": "low"}}
        ]
        }]
        });
        b.iter(|| estimate_request_tokens(&request));
    });
}

criterion_group!(
    benches,
    estimate_tokens_benches,
    estimate_request_tokens_benches,
    estimate_request_tokens_tools_benches,
);
criterion_main!(benches);
