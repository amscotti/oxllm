//! Alerting config: Slack, PagerDuty, generic webhook, and email (SMTP).

use serde::{Deserialize, Serialize};

fn default_thresholds() -> Vec<f64> {
    vec![0.8, 0.9, 1.0]
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AlertingConfig {
    #[serde(default)]
    pub webhook: Option<WebhookAlertConfig>,
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebhookAlertConfig {
    pub url: String,
    #[serde(default)]
    pub headers: Option<std::collections::HashMap<String, String>>,
    #[serde(default = "default_thresholds")]
    pub thresholds: Vec<f64>,
    #[serde(default)]
    pub webhook_type: Option<String>,
    /// Optional soft-budget fraction of each scope's `max_budget`. When set
    /// (e.g. `0.8`), crossing `max_budget * soft_budget_pct` fires a
    /// non-blocking `"soft_budget_exceeded"` alert alongside the ratio
    /// thresholds. `None` disables soft-budget alerts.
    #[serde(default)]
    pub soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for WebhookAlertConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookAlertConfig")
            .field("url", &super::redact_str(&self.url))
            .field(
                "headers",
                &self.headers.as_ref().map(super::RedactedHeaders),
            )
            .field("thresholds", &self.thresholds)
            .field("webhook_type", &self.webhook_type)
            .field("soft_budget_pct", &self.soft_budget_pct)
            .finish()
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlackAlertConfig {
    pub webhook_url: String,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default = "default_thresholds")]
    pub thresholds: Vec<f64>,
    /// Optional soft-budget fraction of each scope's `max_budget`. See
    /// [`WebhookAlertConfig::soft_budget_pct`].
    #[serde(default)]
    pub soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for SlackAlertConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlackAlertConfig")
            .field("webhook_url", &super::redact_str(&self.webhook_url))
            .field("channel", &self.channel)
            .field("thresholds", &self.thresholds)
            .field("soft_budget_pct", &self.soft_budget_pct)
            .finish()
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagerDutyConfig {
    pub routing_key: String,
    #[serde(default = "default_thresholds")]
    pub thresholds: Vec<f64>,
    /// Optional Events API v2 endpoint override. Defaults to
    /// `https://events.pagerduty.com/v2/enqueue`; set this to target a
    /// regional (EU, GOV) or self-hosted PagerDuty instance. When set,
    /// the value is run through `validate_api_base` at config load and
    /// the SSRF-aware HTTP client is used for the POST.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Optional soft-budget fraction of each scope's `max_budget`. See
    /// [`WebhookAlertConfig::soft_budget_pct`].
    #[serde(default)]
    pub soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for PagerDutyConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagerDutyConfig")
            .field("routing_key", &super::redact_str(&self.routing_key))
            .field("thresholds", &self.thresholds)
            .field("endpoint", &self.endpoint)
            .field("soft_budget_pct", &self.soft_budget_pct)
            .finish()
    }
}

/// Configuration for the email (SMTP) budget-threshold alerting hook.
///
/// ```yaml
/// oxllm_settings:
///   email_alert_config:
///     smtp_host: smtp.example.com
///     smtp_port: 587
///     smtp_username: alerts@example.com
///     smtp_password: ...
///     from: alerts@example.com
///     to: ["oncall@example.com"]
///     ssl: true
///     thresholds: [0.8, 0.9, 1.0]
/// ```
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmailAlertConfig {
    /// SMTP server hostname.
    pub smtp_host: String,
    /// SMTP server port. Defaults to 465 (implicit TLS).
    #[serde(default = "default_smtp_port")]
    pub smtp_port: u16,
    /// SMTP auth username. Optional (some relays are unauthenticated).
    #[serde(default)]
    pub smtp_username: Option<String>,
    /// SMTP auth password.
    #[serde(default)]
    pub smtp_password: Option<String>,
    /// `From:` address for alert emails.
    pub from: String,
    /// `To:` recipient list.
    pub to: Vec<String>,
    /// Use TLS (`true` for implicit-TLS on port 465, or STARTTLS on 587).
    /// Defaults to `true`.
    #[serde(default = "default_ssl_true")]
    pub ssl: bool,
    /// Spend-percentage thresholds at which to fire an alert. Defaults to
    /// `[0.8, 0.9, 1.0]` (same default as the Slack/PagerDuty hooks).
    #[serde(default = "default_thresholds")]
    pub thresholds: Vec<f64>,
    /// Optional soft-budget fraction of each scope's `max_budget`. See
    /// [`WebhookAlertConfig::soft_budget_pct`].
    #[serde(default)]
    pub soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for EmailAlertConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmailAlertConfig")
            .field("smtp_host", &self.smtp_host)
            .field("smtp_port", &self.smtp_port)
            .field("smtp_username", &self.smtp_username)
            .field("smtp_password", &super::redact_opt(&self.smtp_password))
            .field("from", &self.from)
            .field("to", &self.to)
            .field("ssl", &self.ssl)
            .field("thresholds", &self.thresholds)
            .field("soft_budget_pct", &self.soft_budget_pct)
            .finish()
    }
}

fn default_smtp_port() -> u16 {
    465
}

fn default_ssl_true() -> bool {
    true
}
