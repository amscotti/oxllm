//! Credential helpers for the Vertex AI provider.
//!
//! The host owns the actual credential resolution (env vars, ADC, service
//! accounts) and hands the provider a pre-resolved
//! [`ResolvedCredentials::GcpBearer`]. This module only documents the contract
//! and provides constants for the env vars the host reads.
//!
//! ## Current state
//!
//! The host accepts explicit access tokens or renewable Google credentials.
//! `vertex_credentials` accepts a service-account or authorized-user JSON
//! object, JSON string, or file path. `GOOGLE_APPLICATION_CREDENTIALS` selects
//! an ambient credential file. OAuth tokens are cached and refreshed on demand
//! before expiry; concurrent exchanges for one identity are coalesced.
//! Static `GCP_ACCESS_TOKEN` / `VERTEX_AI_ACCESS_TOKEN` values are forwarded
//! verbatim and need external rotation.
//!
//! ## Project / location configuration
//!
//! The GCP project id and location are **not** read from dedicated env vars by
//! the host. They must be supplied per-deployment via `oxllm_params.extra`
//! (`vertex_project` / `vertex_location`). To source them from the process
//! environment, use the config loader's `os.environ/` substitution in YAML,
//! e.g. `vertex_project: os.environ/VERTEX_AI_PROJECT`.
//!
//! ## Application Default Credentials subset
//!
//! The host exchanges service-account RS256 assertions and authorized-user
//! refresh tokens. Metadata-server discovery, gcloud default-file discovery,
//! federation, impersonation and non-default universes remain unsupported.

/// The primary env var the host reads to obtain a pre-exchanged GCP bearer
/// token for Vertex AI.
pub const ENV_GCP_ACCESS_TOKEN: &str = "GCP_ACCESS_TOKEN";

/// Alternative env var name (retained for compatibility).
pub const ENV_VERTEX_ACCESS_TOKEN: &str = "VERTEX_AI_ACCESS_TOKEN";
