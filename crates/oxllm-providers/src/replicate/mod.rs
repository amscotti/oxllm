//! Replicate provider — uses Replicate's OpenAI-compatible Chat Completions
//! endpoint.
//!
//! Wire target: `https://api.replicate.com/v1/chat/completions`.
//!
//! Replicate exposes an OpenAI-compatible surface, so the transform is a
//! near-identity: we strip the `replicate/` provider prefix from the model and
//! forward the rest unchanged.
//!
//! Peculiarity: Replicate model ids use an `owner/model-name` structure (e.g.
//! `meta/llama-3.1-8b-instruct`). Only the leading `replicate/` segment is
//! stripped; the underlying `owner/model` path is preserved (see
//! [`OpenAiCompatProvider::strip_one_segment`], which drops a single segment).
//! See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;
#[cfg(test)]
use serde_json::Value;

/// The default Replicate OpenAI-compatible base URL.
pub const DEFAULT_REPLICATE_BASE: &str = "https://api.replicate.com/v1";

const STRIP: &str = "replicate";

/// The configured Replicate provider.
pub const PROVIDER: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "replicate",
    DEFAULT_REPLICATE_BASE,
    STRIP,
    false,
    true,
    true,
);

/// Backward-compatible handle to the Replicate provider.
///
/// Historically `ReplicateProvider` was a unit struct; it is now a `const` of
/// the shared [`OpenAiCompatProvider`] so existing call sites
/// (`Arc::new(replicate::ReplicateProvider)`, `let p = ReplicateProvider;`)
/// keep working.
#[allow(non_upper_case_globals)]
pub const ReplicateProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `replicate/` provider prefix from a model id. Only the first
/// segment is dropped, so the underlying `owner/model-name` structure (e.g.
/// `meta/llama-3.1-8b-instruct`) survives intact. Kept as a free function for
/// backward compatibility with existing tests.
pub fn strip_replicate_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
