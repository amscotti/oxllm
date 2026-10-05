//! Gemini provider — uses Google's OpenAI-compatible endpoint.
//!
//! Wire target:
//! `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`.
//!
//! Because Gemini exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `gemini/` provider prefix from the model and
//! forward the rest unchanged. See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;
#[cfg(test)]
use serde_json::Value;

/// The default Gemini OpenAI-compatible base URL.
pub const DEFAULT_GEMINI_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/openai";

const STRIP: &str = "gemini";

/// The subset of the shared OpenAI-compat param list
/// ([`SUPPORTED_CHAT_PARAMS`](crate::openai_compat::SUPPORTED_CHAT_PARAMS))
/// that Google's Gemini OpenAI-compat surface accepts. The shared list forwards
/// several OpenAI-only fields (`logit_bias`, `store`, `metadata`) that Gemini
/// rejects; Gemini does accept `modalities`, `audio` (on TTS/audio models), and
/// `service_tier` (`"priority"` / `"flex"`), so those are kept — matching
/// LiteLLM's `get_supported_openai_params` for Gemini.
///
/// This override drives **both** param-handling paths: the host's
/// `drop_params` stripping (under `drop_params = true` keys outside this list
/// are stripped before forwarding, preventing the upstream `400` those fields
/// would trigger; under the default `drop_params = false` they are only warned
/// about) **and** [`OpenAiCompatProvider::transform_request`], which rebuilds
/// the body from this list via [`build_chat_request_with_params`]. A field
/// omitted here is therefore dropped regardless of `drop_params`. Kept in
/// lock-step with the shared list by `supported_params_excludes_openai_only_fields`
/// in `tests.rs`.
const GEMINI_SUPPORTED_CHAT_PARAMS: &[&str] = &[
    "model",
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "stop",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "user",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "seed",
    "logprobs",
    "top_logprobs",
    "modalities",
    "audio",
    "service_tier",
];

/// The configured Gemini provider. Gemini publishes a Gemini-narrowed
/// `supported_params` list (the shared OpenAI-compat list minus the
/// OpenAI-only fields Gemini rejects), so the host's `drop_params` mechanism
/// owns param stripping with proper warnings. Serves embeddings via the same
/// OpenAI-compat endpoint.
pub const PROVIDER: OpenAiCompatProvider =
    OpenAiCompatProvider::new("gemini", DEFAULT_GEMINI_BASE, STRIP, true, true, true)
        .with_supported_params(GEMINI_SUPPORTED_CHAT_PARAMS);

/// Backward-compatible handle to the Gemini provider.
///
/// Historically `GeminiProvider` was a unit struct; it is now a `const` of the
/// shared [`OpenAiCompatProvider`] so existing call sites keep working.
#[allow(non_upper_case_globals)]
pub const GeminiProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `gemini/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_gemini_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
