//! Databricks provider — uses Databricks' OpenAI-compatible serving endpoints.
//!
//! Wire targets:
//! - chat: `<api_base>/chat/completions`
//! - embeddings: `<api_base>/embeddings`
//!
//! Because Databricks exposes an OpenAI-compatible surface, the transform is a
//! near-identity: we strip the `databricks/` provider prefix from the model and
//! forward the rest unchanged. Databricks requires the per-workspace URL, so
//! `api_base` is typically set per-deployment rather than relying on the
//! generic default below. See `docs/adding-a-provider.md`.

use crate::openai_compat::OpenAiCompatProvider;
/// The generic Databricks serving-endpoints base URL. In practice the
/// workspace-specific URL (e.g. `https://<workspace>.cloud.databricks.com/serving-endpoints`)
/// is supplied via `api_base` on the deployment.
pub const DEFAULT_DATABRICKS_BASE: &str = "https://api.databricks.com/serving-endpoints";

const STRIP: &str = "databricks";

/// The configured Databricks provider. Supports both chat completions and
/// embeddings.
pub const PROVIDER: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "databricks",
    DEFAULT_DATABRICKS_BASE,
    STRIP,
    true,
    true,
    true,
);

/// Backward-compatible handle to the Databricks provider.
#[allow(non_upper_case_globals)]
pub const DatabricksProvider: OpenAiCompatProvider = PROVIDER;

/// Strip the `databricks/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_databricks_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
