//! Meta provider — Meta's official OpenAI-compatible Chat Completions API
//! (`api.meta.ai`).
//!
//! Wire target: `https://api.meta.ai/v1/chat/completions`.
//!
//! Because Meta exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `meta/` provider prefix from the model and
//! forward the rest unchanged. Current models (e.g. `muse-spark-1.1`) are
//! reasoning models that emit `reasoning_tokens` usage details and may leave
//! `content` null when `max_tokens` is exhausted by reasoning.
//! See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;

/// The default Meta OpenAI-compatible base URL. Meta's API is served under
/// the `/v1` prefix, so the base includes it (the shared compat provider
/// appends `/chat/completions`).
pub const DEFAULT_META_BASE: &str = "https://api.meta.ai/v1";

const STRIP: &str = "meta";

/// The configured Meta provider. Supports chat completions only.
pub const PROVIDER: OpenAiCompatProvider =
    OpenAiCompatProvider::new("meta", DEFAULT_META_BASE, STRIP, false, true, true);

/// Strip the `meta/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_meta_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
