//! The oxllm proxy HTTP server.
//!
//! This crate is the **host**: it owns all I/O (HTTP, config loading, cache).
//! Endpoints and middleware live here; all pure logic is in `oxllm-core`.

// Production code is `forbid(unsafe_code)`. The crate-level attribute is
// test-scoped off because several unit-test modules mutate process
// environment variables via `std::env::set_var`/`remove_var`, which are
// `unsafe` in edition 2024. Shipped (non-test) code contains no `unsafe`.
#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod app;
pub mod auth;
pub mod bedrock_sigv4;
pub mod cache;
pub mod config;
pub mod coordination;
pub mod credentials;
pub mod crypto;
pub mod error;
pub mod guardrails;
pub mod handlers;
pub mod hooks;
pub mod metrics;
pub mod orchestration;
#[cfg(feature = "otel")]
pub mod otel;
pub mod pipeline;
pub mod pre_call_checks;
pub mod realtime;
pub mod routes;
pub mod scheduler;
pub mod security;
#[cfg(feature = "db")]
pub mod skills;
pub mod spend_writer;
pub(crate) mod sse;
pub mod state;
pub mod tokens;

pub use app::run;
pub use cache::{DualCache, InMemoryCache};
pub use config::{ProxyConfig, load_config, parse_config_str};
pub use credentials::{
    BedrockCredentialResolver, BedrockMantleCredentialResolver, CredentialResolver,
    CredentialResolverRegistry, StaticKeyResolver,
};
pub use error::ApiError;
pub use oxllm_core::Cache;
pub use oxllm_core::deployment::{Deployment, ModelInfo, OxllmParams};
pub use oxllm_core::{calculate_cost, estimate_request_tokens, estimate_tokens};
pub use state::{AppState, SharedState};

#[cfg(feature = "db")]
pub use oxllm_db::{
    Database, KeyRecord, ModelConfigRecord, SpendFilter, SpendLog, TeamRecord, UserRecord,
};
#[cfg(feature = "db")]
pub use oxllm_db::{compute_since_iso, now_iso};
#[cfg(feature = "db")]
pub use spend_writer::SpendWriter;
