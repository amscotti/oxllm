//! Serde types for the proxy YAML config, grouped by domain.
//!
//! Submodules:
//! - [`base`] — top-level containers ([`ProxyConfig`], [`OxllmSettings`],
//!   [`GeneralSettings`], [`ConfigModel`]).
//! - [`router`] — routing knobs, router settings, routing-group overrides.
//! - [`cache`] — response-cache and semantic-cache configuration.
//! - [`alerting`] — Slack / PagerDuty / webhook / email alerting config.
//! - [`telemetry`] — observability backends (Langfuse, Datadog, W&B, S3, …).
//! - [`guardrails`] — pre-call guardrail configuration.
//! - `tests` — inline unit tests covering all of the above.

pub mod alerting;
pub mod base;
pub mod cache;
pub mod guardrails;
pub mod router;
pub mod telemetry;

#[cfg(test)]
mod tests;

pub use self::{alerting::*, base::*, cache::*, guardrails::*, router::*, telemetry::*};
