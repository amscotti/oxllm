//! Cloudflare Workers AI provider — uses Cloudflare's OpenAI-compatible Chat
//! Completions endpoint.
//!
//! Wire target: `<api_base>/chat/completions`.
//!
//! Because Cloudflare exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `cloudflare/` provider prefix from the model and
//! forward the rest unchanged. The base URL embeds an `account_id` segment, so
//! the full account-scoped URL is supplied via `api_base` on the deployment
//! rather than relying on the generic default below.
//!
//! **Footgun:** [`DEFAULT_CLOUDFLARE_BASE`] is a non-functional template — it
//! carries the literal `{account_id}` placeholder and will not route if a
//! deployment forgets to set `api_base`. A deployment without `api_base`
//! produces an unusable URL and the upstream 404s with a confusing error.
//! Always set `api_base` for Cloudflare.
//! See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;
/// The generic Cloudflare Workers AI base URL template.
///
/// **This is a non-functional template.** The `{account_id}` segment is a
/// literal placeholder and is not substituted at request time; every
/// Cloudflare deployment must supply its account-scoped URL via `api_base`.
/// Kept as a non-empty sentinel so misconfiguration surfaces the missing
/// account id in the request URL rather than collapsing to a relative path.
pub const DEFAULT_CLOUDFLARE_BASE: &str =
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1";

const STRIP: &str = "cloudflare";

/// The configured Cloudflare provider. Supports chat completions only.
pub const PROVIDER: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "cloudflare",
    DEFAULT_CLOUDFLARE_BASE,
    STRIP,
    false,
    true,
    true,
);

/// Backward-compatible handle to the Cloudflare provider.
#[allow(non_upper_case_globals)]
pub const CloudflareProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `cloudflare/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_cloudflare_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
