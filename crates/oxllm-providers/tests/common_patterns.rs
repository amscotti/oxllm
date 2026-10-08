//! Cross-provider test patterns ported from the reference
//! `tests/llm_translation/base_llm_unit_tests.py`.
//!
//! The reference defines a `BaseLLMChatTest` abstract class whose
//! `_usage_format_tests` helper enforces the OpenAI usage invariant
//! (`total == prompt + completion`, `cached ≤ prompt`). Every provider's
//! `transform_response` must produce a body that obeys the same invariant.
//!
//! These integration tests exercise the same invariant plus the basic
//! OpenAI `ChatCompletion` response shape across every provider that
//! publishes a non-identity transform, plus the identity providers as a
//! sanity check.

use oxllm_core::provider::ChatProvider;
use oxllm_providers::anthropic::AnthropicProvider;
use oxllm_providers::cohere::CohereProvider;
use oxllm_providers::gemini::PROVIDER as GEMINI;
use oxllm_providers::ollama::PROVIDER as OLLAMA;
use oxllm_providers::openai::OpenAiProvider;
use serde_json::{Value, json};

/// A provider paired with a provider-native sample response body that
/// represents a successful `"Hello"` completion with usage
/// `{prompt: 5, completion: 2, total: 7}`.
struct ProviderCase {
    name: &'static str,
    provider: &'static dyn ChatProvider,
    /// Provider-native response body (i.e. what the upstream API returns).
    body: Value,
    /// Expected assistant text after transform.
    expected_content: &'static str,
    /// Expected prompt_tokens after transform.
    expected_prompt: u64,
    /// Expected completion_tokens after transform.
    expected_completion: u64,
}

/// Build the suite of provider cases. Each body is in the provider's native
/// upstream format; the assertions verify the post-transform OpenAI shape.
fn cases() -> Vec<ProviderCase> {
    vec![
        ProviderCase {
            name: "openai",
            provider: &OpenAiProvider,
            body: json!({
                "id": "chatcmpl-openai",
                "object": "chat.completion",
                "created": 1_700_000_000u64,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }),
            expected_content: "Hello",
            expected_prompt: 5,
            expected_completion: 2,
        },
        ProviderCase {
            name: "gemini",
            // Gemini uses the OpenAI-compat identity transform.
            provider: &GEMINI,
            body: json!({
                "id": "chatcmpl-gemini",
                "object": "chat.completion",
                "created": 1_700_000_000u64,
                "model": "gemini-pro",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }),
            expected_content: "Hello",
            expected_prompt: 5,
            expected_completion: 2,
        },
        ProviderCase {
            name: "ollama",
            provider: &OLLAMA,
            body: json!({
                "id": "chatcmpl-ollama",
                "object": "chat.completion",
                "created": 1_700_000_000u64,
                "model": "llama3",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }),
            expected_content: "Hello",
            expected_prompt: 5,
            expected_completion: 2,
        },
        ProviderCase {
            name: "anthropic",
            provider: &AnthropicProvider,
            body: json!({
                "id": "msg_anthropic",
                "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }),
            expected_content: "Hello",
            expected_prompt: 5,
            expected_completion: 2,
        },
        ProviderCase {
            name: "cohere",
            provider: &CohereProvider,
            body: json!({
                "id": "cohere-1",
                "finish_reason": "COMPLETE",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}],
                },
                "usage": {
                    "billed_units": {"input_tokens": 5, "output_tokens": 2},
                },
            }),
            expected_content: "Hello",
            expected_prompt: 5,
            expected_completion: 2,
        },
    ]
}

/// Ported from `_usage_format_tests` in base_llm_unit_tests.py.
///
/// Invariant: `total_tokens == prompt_tokens + completion_tokens`.
/// Verifies the OpenAI usage shape is consistent across every provider.
#[test]
fn all_producers_total_equals_prompt_plus_completion() {
    for case in cases() {
        let out = case
            .provider
            .transform_response(&case.body, "test-model")
            .unwrap_or_else(|e| panic!("{}: transform_response failed: {e}", case.name));

        let prompt = out["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: prompt_tokens missing or non-u64", case.name));
        let completion = out["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: completion_tokens missing or non-u64", case.name));
        let total = out["usage"]["total_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: total_tokens missing or non-u64", case.name));

        assert_eq!(
            total,
            prompt + completion,
            "{}: total_tokens ({total}) must equal prompt ({prompt}) + completion ({completion})",
            case.name,
        );
        assert_eq!(
            prompt, case.expected_prompt,
            "{}: prompt_tokens mismatch",
            case.name
        );
        assert_eq!(
            completion, case.expected_completion,
            "{}: completion_tokens mismatch",
            case.name
        );
    }
}

/// Ported from the response-shape checks scattered across
/// `test_convert_to_model_response_object_*`. Every provider's
/// `transform_response` must yield a body with the canonical OpenAI
/// `ChatCompletion` top-level fields: `id`, `object`, `created`, `model`,
/// `choices[*].{index, message, finish_reason}`, `usage`.
#[test]
fn all_producers_emit_openai_chat_completion_shape() {
    for case in cases() {
        let out = case
            .provider
            .transform_response(&case.body, "test-model")
            .unwrap_or_else(|e| panic!("{}: transform_response failed: {e}", case.name));

        assert!(
            out.get("id").is_some(),
            "{}: response must have top-level `id`",
            case.name
        );
        assert_eq!(
            out["object"], "chat.completion",
            "{}: object must be chat.completion",
            case.name
        );
        assert!(
            out["created"].as_u64().is_some(),
            "{}: created must be a unix timestamp",
            case.name
        );
        assert!(
            out.get("model").is_some(),
            "{}: model must be present",
            case.name
        );

        let choices = out["choices"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: choices must be an array", case.name));
        assert_eq!(
            choices.len(),
            1,
            "{}: expected exactly one choice",
            case.name
        );
        let choice = &choices[0];
        assert_eq!(choice["index"], 0, "{}: choice index must be 0", case.name);
        assert_eq!(
            choice["message"]["role"], "assistant",
            "{}: message.role must be assistant",
            case.name
        );
        assert_eq!(
            choice["message"]["content"], case.expected_content,
            "{}: message.content mismatch",
            case.name
        );
        assert_eq!(
            choice["finish_reason"], "stop",
            "{}: finish_reason must be stop for a normal completion",
            case.name
        );
    }
}

/// Ported from `test_content_list_handling` in base_llm_unit_tests.py: a
/// user message whose content is an array of `{type: "text", text: ...}`
/// blocks must be accepted by every provider's `transform_request`.
#[test]
fn all_producers_accept_content_array_in_request() {
    fn request() -> Value {
        json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }],
            "max_tokens": 16,
        })
    }

    let cases: Vec<(&str, &dyn ChatProvider)> = vec![
        ("openai", &OpenAiProvider),
        ("gemini", &GEMINI),
        ("ollama", &OLLAMA),
        ("anthropic", &AnthropicProvider),
        ("cohere", &CohereProvider),
    ];

    for (name, provider) in cases {
        let out = provider
            .transform_request(&request(), &minimal_deployment(name))
            .unwrap_or_else(|e| panic!("{name}: transform_request failed: {e}"));
        // Every provider must keep at least one message in the request —
        // dropping the user's content would be a protocol violation.
        let messages = out["messages"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: messages must be an array"));
        assert!(
            !messages.is_empty(),
            "{name}: messages array must not be empty after transform"
        );
    }
}

/// Build a minimal `Deployment` for the given provider slug. This mirrors
/// the per-provider `deployment()` helpers in the unit tests.
fn minimal_deployment(provider: &str) -> oxllm_core::deployment::Deployment {
    use oxllm_core::deployment::{Deployment, OxllmParams};
    Deployment {
        model_name: format!("{provider}-model"),
        oxllm_params: OxllmParams {
            model: format!("{provider}/test-model"),
            api_key: Some("sk-test".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}
