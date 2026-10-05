//! Bedrock Mantle provider — uses AWS Bedrock Mantle's OpenAI-compatible
//! Chat Completions endpoint.
//!
//! Wire target: `https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions`.
//!
//! Bedrock Mantle is a distributed inference engine for Bedrock-hosted models
//! that exposes an OpenAI-compatible surface. The transform is a near-identity:
//! we strip the `bedrock_mantle/` provider prefix from the model and forward
//! the rest unchanged. See `docs/adding-a-provider.md`.
//!
//! The base URL is region-specific (`https://bedrock-mantle.{region}.api.aws/v1`);
//! the default below uses `us-east-1`. Override the region via `api_base` in
//! the deployment config. Auth is a standard Bearer token whose value looks
//! like `bedrock-api-key-<base64>`.

use crate::openai_compat::OpenAiCompatProvider;

/// The default Bedrock Mantle OpenAI-compatible base URL (`us-east-1`).
pub const DEFAULT_BEDROCK_MANTLE_BASE: &str = "https://bedrock-mantle.us-east-1.api.aws/v1";

const STRIP: &str = "bedrock_mantle";

/// The configured Bedrock Mantle provider. Supports chat completions only —
/// Mantle does not yet expose embeddings or rerank.
pub const PROVIDER: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "bedrock_mantle",
    DEFAULT_BEDROCK_MANTLE_BASE,
    STRIP,
    false,
    true,
    true,
);

/// Backward-compatible handle to the Bedrock Mantle provider.
///
/// Historically `<Slug>Provider` was a unit struct; it is now a `const` of
/// the shared [`OpenAiCompatProvider`] so existing call sites keep working.
#[allow(non_upper_case_globals)]
pub const BedrockMantleProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `bedrock_mantle/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_bedrock_mantle_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
