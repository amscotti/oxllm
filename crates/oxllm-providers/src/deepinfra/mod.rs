//! DeepInfra provider — uses DeepInfra's OpenAI-compatible Chat Completions
//! endpoint.
//!
//! Wire targets:
//! - chat: `https://api.deepinfra.com/v1/openai/chat/completions`
//! - embeddings: `https://api.deepinfra.com/v1/openai/embeddings`
//!
//! Because DeepInfra exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `deepinfra/` provider prefix from the model and
//! forward the rest unchanged. See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;

/// The default DeepInfra OpenAI-compatible base URL.
pub const DEFAULT_DEEPINFRA_BASE: &str = "https://api.deepinfra.com/v1/openai";

const STRIP: &str = "deepinfra";

/// The configured DeepInfra provider. Supports both chat completions and
/// embeddings.
pub const PROVIDER: OpenAiCompatProvider =
    OpenAiCompatProvider::new("deepinfra", DEFAULT_DEEPINFRA_BASE, STRIP, true, true, true);

/// Backward-compatible handle to the DeepInfra provider.
///
/// Historically `DeepInfraProvider` was a unit struct; it is now a `const` of
/// the shared [`OpenAiCompatProvider`] so existing call sites keep working.
#[allow(non_upper_case_globals)]
pub const DeepInfraProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `deepinfra/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_deepinfra_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
