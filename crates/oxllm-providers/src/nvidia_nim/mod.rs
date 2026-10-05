//! NVIDIA NIM provider — uses NVIDIA's OpenAI-compatible endpoints.
//!
//! Wire targets:
//! - chat: `https://integrate.api.nvidia.com/v1/chat/completions`
//! - embeddings: `https://integrate.api.nvidia.com/v1/embeddings`
//!
//! Because NVIDIA NIM exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `nvidia_nim/` provider prefix from the model
//! and forward the rest unchanged. See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;
/// The default NVIDIA NIM OpenAI-compatible base URL.
pub const DEFAULT_NVIDIA_NIM_BASE: &str = "https://integrate.api.nvidia.com/v1";

const STRIP: &str = "nvidia_nim";

/// The configured NVIDIA NIM provider. Supports both chat completions and
/// embeddings.
pub const PROVIDER: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "nvidia_nim",
    DEFAULT_NVIDIA_NIM_BASE,
    STRIP,
    true,
    true,
    true,
);

/// Backward-compatible handle to the NVIDIA NIM provider.
///
/// Historically `NvidiaNimProvider` was a unit struct; it is now a `const` of
/// the shared [`OpenAiCompatProvider`] so existing call sites keep working.
#[allow(non_upper_case_globals)]
pub const NvidiaNimProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `nvidia_nim/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_nvidia_nim_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
