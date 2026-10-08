//! Built-in hook implementations.
//!
//! These are the typed analogues of the reference `proxy/hooks/` chain. They share
//! the in-memory cache (or Redis, when wired in) for spend/RPM/TPM counters and
//! cached responses.

pub mod adaptive_hook;
pub mod alerting;
pub mod budget;
pub mod cache_hook;
pub mod callback;
pub mod cost;
pub mod cost_fallback;
pub mod datadog;
pub mod email_alert;
pub mod helicone;
pub mod http_telemetry;
pub mod langfuse;
pub mod langsmith;
pub mod metrics;
pub mod mlflow;
pub mod pagerduty_alert;
pub mod rate_limit;
pub mod redact;
pub mod runner;
pub mod s3_log;
pub mod slack_alert;
pub mod spend_keys;
pub mod wandb;

pub use adaptive_hook::AdaptiveHook;
pub use alerting::{AlertState, AlertingHook, BudgetAlert, scope_for_spend_key};
pub use budget::BudgetHook;
pub use cache_hook::CacheHook;
pub use callback::{CallbackConfig, GenericHttpCallback};
pub use cost::CostTrackingHook;
pub use email_alert::EmailAlertingHook;
pub use http_telemetry::HttpTelemetryHook;
pub use langfuse::{LangfuseEvent, LangfuseHook};
pub use metrics::MetricsHook;
pub use mlflow::MlflowHook;
pub use pagerduty_alert::PagerDutyAlertingHook;
pub use rate_limit::RateLimitHook;
pub use runner::{
    run_post_failure_hooks, run_post_success_hooks, run_pre_call_hooks, run_release_concurrency,
};
pub use s3_log::S3LogHook;
pub use slack_alert::SlackAlertingHook;
