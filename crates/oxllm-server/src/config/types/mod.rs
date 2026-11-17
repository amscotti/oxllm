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

/// Marker written in `Debug` output for secret-bearing config fields.
const REDACTED: &str = "<redacted>";

pub(super) fn redact_opt(opt: &Option<String>) -> Option<&'static str> {
    opt.as_ref().map(|_| REDACTED)
}

pub(super) fn redact_str(s: &str) -> &'static str {
    if s.is_empty() { "" } else { REDACTED }
}

pub(super) struct RedactedHeaders<'a>(pub(super) &'a std::collections::HashMap<String, String>);

impl std::fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for name in self.0.keys() {
            map.entry(name, &REDACTED);
        }
        map.finish()
    }
}

pub(super) struct RedactedValue<'a>(pub(super) &'a serde_json::Value);

impl std::fmt::Debug for RedactedValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            serde_json::Value::String(_) => f.write_str(REDACTED),
            serde_json::Value::Object(m) => {
                let mut map = f.debug_map();
                for (key, value) in m {
                    map.entry(key, &RedactedValue(value));
                }
                map.finish()
            }
            serde_json::Value::Array(items) => {
                let mut list = f.debug_list();
                for item in items {
                    list.entry(&RedactedValue(item));
                }
                list.finish()
            }
            other => std::fmt::Debug::fmt(other, f),
        }
    }
}

pub(super) struct RedactedExtra<'a>(pub(super) &'a serde_json::Map<String, serde_json::Value>);

impl std::fmt::Debug for RedactedExtra<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (key, value) in self.0 {
            map.entry(key, &RedactedValue(value));
        }
        map.finish()
    }
}

pub(super) struct RedactedValueMap<'a>(
    pub(super) &'a std::collections::HashMap<String, serde_json::Value>,
);

impl std::fmt::Debug for RedactedValueMap<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (key, value) in self.0 {
            map.entry(key, &RedactedValue(value));
        }
        map.finish()
    }
}
