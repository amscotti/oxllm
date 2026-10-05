//! AI21 provider — uses AI21's OpenAI-compatible Chat Completions endpoint.
//!
//! Wire target: `https://api.ai21.com/studio/v1/chat/completions`.
//!
//! Because AI21 exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `ai21/` provider prefix from the model and
//! forward the rest unchanged. See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;

/// The default AI21 OpenAI-compatible base URL.
pub const DEFAULT_AI21_BASE: &str = "https://api.ai21.com/studio/v1";

const STRIP: &str = "ai21";

/// The configured AI21 provider.
pub const PROVIDER: OpenAiCompatProvider =
    OpenAiCompatProvider::new("ai21", DEFAULT_AI21_BASE, STRIP, false, true, true);

/// Backward-compatible handle to the AI21 provider.
///
/// Historically `AI21Provider` was a unit struct; it is now a `const` of the
/// shared [`OpenAiCompatProvider`] so existing call sites keep working.
#[allow(non_upper_case_globals)]
pub const AI21Provider: OpenAiCompatProvider = PROVIDER;

/// Strip the `ai21/` provider prefix from a model id. Kept as a free function
/// for backward compatibility with existing tests.
pub fn strip_ai21_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
