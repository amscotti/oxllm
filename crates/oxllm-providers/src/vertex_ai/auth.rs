//! Credential helpers for the Vertex AI provider.
//!
//! The host owns the actual credential resolution (env vars, ADC, service
//! accounts) and hands the provider a pre-resolved
//! [`ResolvedCredentials::GcpBearer`]. This module only documents the contract
//! and provides constants for the env vars the host reads.
//!
//! ## Current state
//!
//! The initial implementation supports **pre-configured access tokens** only:
//! the host reads `GCP_ACCESS_TOKEN` (or `VERTEX_AI_ACCESS_TOKEN`) from the
//! environment and forwards it verbatim as a bearer token. Tokens expire after
//! roughly one hour, so operators are responsible for refreshing them out of
//! band.
//!
//! ## Project / location configuration
//!
//! The GCP project id and location are **not** read from dedicated env vars by
//! the host. They must be supplied per-deployment via `oxllm_params.extra`
//! (`vertex_project` / `vertex_location`). To source them from the process
//! environment, use the config loader's `os.environ/` substitution in YAML,
//! e.g. `vertex_project: os.environ/VERTEX_AI_PROJECT`.
//!
//! ## Future: Application Default Credentials
//!
//! Full ADC support (read `GOOGLE_APPLICATION_CREDENTIALS` → service-account
//! JSON → JWT-sign an OAuth2 assertion → exchange for a short-lived access
//! token, with caching) is a planned enhancement. It requires a JWT/RSA
//! signing dependency and a token-cache with expiry, neither of which belongs
//! in the pure provider crate.

/// The primary env var the host reads to obtain a pre-exchanged GCP bearer
/// token for Vertex AI.
pub const ENV_GCP_ACCESS_TOKEN: &str = "GCP_ACCESS_TOKEN";

/// Alternative env var name (mirrors LiteLLM's convention).
pub const ENV_VERTEX_ACCESS_TOKEN: &str = "VERTEX_AI_ACCESS_TOKEN";
