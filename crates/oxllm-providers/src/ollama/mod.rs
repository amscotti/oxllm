//! Ollama provider — uses the local OpenAI-compatible server.
//!
//! Wire target: `http://localhost:11434/v1/chat/completions`.
//!
//! Ollama exposes an OpenAI-compatible API; no auth required. The transform
//! is a near-identity with the `ollama/` provider prefix stripped from the
//! model name.

use crate::openai_compat::OpenAiCompatProvider;
#[cfg(test)]
use serde_json::Value;

/// The default Ollama base URL.
pub const DEFAULT_OLLAMA_BASE: &str = "http://localhost:11434/v1";

const STRIP: &str = "ollama";

/// The configured Ollama provider. Ollama requires no auth, publishes the
/// shared OpenAI-compat `supported_params` list (so the host's `drop_params`
/// mechanism owns param stripping with proper warnings, instead of
/// `build_chat_request` silently filtering), and serves embeddings via the
/// OpenAI-compat `/v1/embeddings` endpoint.
pub const PROVIDER: OpenAiCompatProvider =
    OpenAiCompatProvider::new("ollama", DEFAULT_OLLAMA_BASE, STRIP, true, true, false);

/// Backward-compatible handle to the Ollama provider.
///
/// Historically `OllamaProvider` was a unit struct; it is now a `const` of the
/// shared [`OpenAiCompatProvider`] so existing call sites keep working.
#[allow(non_upper_case_globals)]
pub const OllamaProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `ollama/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_ollama_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
