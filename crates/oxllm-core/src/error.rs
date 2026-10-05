//! Error types for oxllm.
//!
//! Every fallible path returns a typed [`enum@Error`]. No `unwrap`/`expect`/`panic`
//! on user or provider input.

use thiserror::Error;

/// The canonical error type for all oxllm crates.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum Error {
    /// The request was malformed or missing required fields.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// Authentication failed (invalid or missing API key).
    #[error("invalid api key")]
    Unauthorized,

    /// The requested model is not configured.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// A rate limit was exceeded.
    #[error("{0}")]
    RateLimited(String),

    /// A spend budget (key/team/org/project/model/deployment/provider cap) was
    /// exhausted. Distinct from [`Error::RateLimited`]: budget windows reset on
    /// daily/monthly horizons, so a 60s `Retry-After` would mislead clients
    /// into hot-retrying a cap that won't lift for hours/days.
    #[error("{0}")]
    BudgetExceeded(String),

    /// The upstream provider returned an error.
    #[error("upstream error ({status}): {message}")]
    Upstream {
        /// HTTP status code from the upstream provider.
        status: u16,
        /// Error message from the provider (forwarded verbatim to the client;
        /// size-bounded at extraction). No redaction/sanitization pass is
        /// applied: callers must not route secrets through this variant
        /// expecting them to be scrubbed.
        message: String,
    },

    /// A network or I/O error talking to the upstream provider.
    #[error("network error: {0}")]
    Network(String),

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// A configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// A routing pre-call check failed at runtime. Distinct from
    /// [`Error::Config`]: the check itself is configured fine, its
    /// execution failed.
    #[error("pre-call check failed: {0}")]
    PreCallCheckFailed(String),

    /// A provider transformation error.
    #[error("transform error: {0}")]
    Transform(String),

    /// An internal error that should not leak detail to the client.
    #[error("internal error")]
    Internal,
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
