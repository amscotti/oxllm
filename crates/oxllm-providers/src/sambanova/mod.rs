//! SambaNova provider — uses SambaNova's OpenAI-compatible endpoints.
//!
//! Wire targets:
//! - chat: `https://api.sambanova.ai/v1/chat/completions`
//! - embeddings: `https://api.sambanova.ai/v1/embeddings`
//!
//! Because SambaNova exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `sambanova/` provider prefix from the model and
//! forward the rest unchanged. See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;
/// The default SambaNova OpenAI-compatible base URL.
pub const DEFAULT_SAMBANOVA_BASE: &str = "https://api.sambanova.ai/v1";

const STRIP: &str = "sambanova";

/// The configured SambaNova provider. Supports both chat completions and
/// embeddings.
pub const PROVIDER: OpenAiCompatProvider =
    OpenAiCompatProvider::new("sambanova", DEFAULT_SAMBANOVA_BASE, STRIP, true, true, true);

/// Backward-compatible handle to the SambaNova provider.
#[allow(non_upper_case_globals)]
pub const SambanovaProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `sambanova/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_sambanova_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
