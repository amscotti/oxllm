//! Email (SMTP) budget-threshold alerting hook.
//!
//! A [`Hook`] that sends alert emails via SMTP when a key/team/user/org/
//! project/model spend crosses a configurable percentage of its
//! `max_budget`. It is the email analogue of
//! [`SlackAlertingHook`](crate::hooks::SlackAlertingHook): same
//! threshold-evaluation logic and the same dedup via [`AlertState`], but the
//! notification goes out as an email instead of a Slack webhook.
//!
//! SMTP transport is provided by [`lettre`] (rustls backend). The send is
//! fire-and-forget: it runs in a detached `tokio::spawn` task so the request
//! path is never blocked; failures are logged at `warn` level.
//!
//! Wired in from `oxllm_settings.email_alert_config`. The hook owns its own
//! [`AlertState`] so it can coexist with the Slack and generic webhook
//! alerting hooks (all fire independently when configured together).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lettre::message::Mailbox;
use lettre::message::header::ContentType;
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use oxllm_core::{Cache, CallContext, Hook, ResponseSummary};

use crate::config::types::EmailAlertConfig;
use crate::hooks::alerting::{self, AlertState, SOFT_BUDGET_SENTINEL};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

fn now_iso() -> String {
    jiff::Timestamp::now().to_string()
}

/// A `Hook` that fires SMTP email alerts when budget thresholds are crossed.
pub struct EmailAlertingHook {
    cache: Arc<dyn Cache>,
    config: EmailAlertConfig,
    thresholds: Vec<f64>,
    state: Arc<AlertState>,
    transport: AsyncSmtpTransport<Tokio1Executor>,
    global_max_budget: Option<f64>,
    /// Per-model global spend caps, mirrored from
    /// `oxllm_settings.model_max_budget`.
    model_max_budget: Arc<HashMap<String, f64>>,
    /// Optional soft-budget percentage of each scope's `max_budget`. When
    /// set (e.g. `0.8`), crossing it fires a non-blocking soft-budget
    /// alert email (subject prefixed `soft_budget_exceeded`). Mirrors the
    /// same field on [`AlertingHook`](crate::hooks::AlertingHook). `None`
    /// disables soft-budget alerts.
    soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for EmailAlertingHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmailAlertingHook")
            .field("smtp_host", &self.config.smtp_host)
            .field("smtp_port", &self.config.smtp_port)
            .field("from", &self.config.from)
            .field("thresholds", &self.thresholds)
            .finish()
    }
}

impl EmailAlertingHook {
    /// Build an email alerting hook from a config, threshold list, shared
    /// alert state, and the global/model budget ceilings. Constructs the SMTP
    /// transport up front so misconfiguration surfaces at startup.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache: Arc<dyn Cache>,
        config: EmailAlertConfig,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
        global_max_budget: Option<f64>,
        model_max_budget: HashMap<String, f64>,
    ) -> anyhow::Result<Self> {
        if config.smtp_host.trim().is_empty() {
            anyhow::bail!("email_alert_config.smtp_host must be set");
        }
        if config.from.trim().is_empty() {
            anyhow::bail!("email_alert_config.from must be set");
        }
        if config.to.is_empty() {
            anyhow::bail!("email_alert_config.to must have at least one recipient");
        }
        let transport = build_transport(&config)?;

        Ok(Self {
            cache,
            config,
            thresholds,
            state,
            transport,
            global_max_budget,
            model_max_budget: Arc::new(model_max_budget),
            soft_budget_pct: None,
        })
    }

    /// Builder: set the soft-budget percentage (e.g. `Some(0.8)` → fire a
    /// non-blocking soft-budget alert email when a scope's spend crosses
    /// 80% of its `max_budget`). `None` (the default) disables
    /// soft-budget alerts. Parity with
    /// [`AlertingHook::with_soft_budget_pct`](crate::hooks::AlertingHook::with_soft_budget_pct).
    pub fn with_soft_budget_pct(mut self, pct: Option<f64>) -> Self {
        self.soft_budget_pct = pct;
        self
    }

    /// The configured `from` address (for tests).
    pub fn from_addr(&self) -> &str {
        &self.config.from
    }

    /// Evaluate a single budget scope, firing any newly-crossed thresholds.
    async fn eval_scope(
        &self,
        scope: &str,
        spend_cache_key: &str,
        scope_id: &str,
        limit: Option<f64>,
        ctx: &CallContext,
    ) {
        let Some(limit) = limit else { return };
        let soft_budget = self
            .soft_budget_pct
            .filter(|pct| pct > &0.0)
            .map(|pct| limit * pct);
        for hit in self
            .state
            .eval_thresholds_with_soft(
                &self.cache,
                scope,
                spend_cache_key,
                limit,
                &self.thresholds,
                soft_budget,
            )
            .await
        {
            let (subject, body) = build_email_content(
                scope,
                scope_id,
                hit.threshold,
                hit.current_spend,
                limit,
                &now_iso(),
                ctx,
            );
            self.dispatch(subject, body);
        }
    }

    fn dispatch(&self, subject: String, body: String) {
        let transport = self.transport.clone();
        let from = self.config.from.clone();
        let to = self.config.to.clone();
        tokio::spawn(async move {
            if let Err(e) = send_email(&transport, &from, &to, &subject, &body).await {
                tracing::warn!(
                    error = %e,
                    from = %from,
                    "email alert SMTP send failed"
                );
            }
        });
    }
}

/// Construct the lettre async SMTP transport from the config block.
/// Applies [`REQUEST_TIMEOUT`] to every relay mode so a slow or unresponsive
/// SMTP server cannot hold the spawned send task open beyond the configured
/// bound (lettre's own default is 60s, which would let a budget-alert storm
/// pile up many long-lived tasks each holding a TCP connection).
fn build_transport(
    config: &EmailAlertConfig,
) -> anyhow::Result<AsyncSmtpTransport<Tokio1Executor>> {
    let mut builder = if config.ssl {
        if config.smtp_port == 465 {
            // Implicit TLS (SMTPS) on port 465.
            AsyncSmtpTransport::<Tokio1Executor>::relay(&config.smtp_host)?
        } else {
            // STARTTLS upgrade (e.g. port 587).
            AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&config.smtp_host)?
        }
    } else {
        AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(&config.smtp_host)
            .tls(lettre::transport::smtp::client::Tls::None)
    };
    builder = builder
        .port(config.smtp_port)
        .timeout(Some(REQUEST_TIMEOUT));
    if let (Some(user), Some(pass)) = (
        config.smtp_username.as_deref(),
        config.smtp_password.as_deref(),
    ) {
        if !user.trim().is_empty() {
            builder = builder.credentials(Credentials::new(user.to_string(), pass.to_string()));
        }
    }
    Ok(builder.build())
}

/// Build the subject + plain-text body for a single threshold crossing.
/// A `threshold` of [`SOFT_BUDGET_SENTINEL`] renders the soft-budget
/// variant (subject prefixed `soft_budget_exceeded`; the body omits the
/// `Threshold: N%` line because the soft ceiling is an absolute USD
/// amount, not a ratio).
#[allow(clippy::too_many_arguments)]
fn build_email_content(
    scope: &str,
    scope_id: &str,
    threshold: f64,
    current_spend: f64,
    max_budget: f64,
    timestamp: &str,
    ctx: &CallContext,
) -> (String, String) {
    let is_soft = threshold == SOFT_BUDGET_SENTINEL;
    let pct = (threshold * 100.0).round() as i64;
    if is_soft {
        let subject = format!("Budget alert: {scope} crossed its soft budget");
        let body = format!(
            "A soft-budget ceiling was crossed.\n\n\
             Event: soft_budget_exceeded\n\
             Scope: {scope}\n\
             Scope ID: {scope_id}\n\
             Current spend: ${current_spend:.2}\n\
             Max budget: ${max_budget:.2}\n\
             Request ID: {req_id}\n\
             Fired at: {timestamp}\n",
            req_id = ctx.request_id,
        );
        (subject, body)
    } else {
        let subject = format!("Budget alert: {scope} reached {pct}% of its limit");
        let body = format!(
            "A budget threshold was crossed.\n\n\
             Scope: {scope}\n\
             Scope ID: {scope_id}\n\
             Threshold: {pct}%\n\
             Current spend: ${current_spend:.2}\n\
             Max budget: ${max_budget:.2}\n\
             Request ID: {req_id}\n\
             Fired at: {timestamp}\n",
            req_id = ctx.request_id,
        );
        (subject, body)
    }
}

/// Build a JSON representation of an alert email (for inspection / tests).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn build_email_payload(
    config: &EmailAlertConfig,
    scope: &str,
    scope_id: &str,
    threshold: f64,
    current_spend: f64,
    max_budget: f64,
    timestamp: &str,
    ctx: &CallContext,
) -> serde_json::Value {
    let (subject, body) = build_email_content(
        scope,
        scope_id,
        threshold,
        current_spend,
        max_budget,
        timestamp,
        ctx,
    );
    serde_json::json!({
        "from": config.from,
        "to": config.to,
        "subject": subject,
        "body": body,
    })
}

async fn send_email(
    transport: &AsyncSmtpTransport<Tokio1Executor>,
    from: &str,
    to: &[String],
    subject: &str,
    body: &str,
) -> Result<(), EmailError> {
    let mut builder = Message::builder()
        .from(parse_mailbox(from)?)
        .subject(subject)
        .header(ContentType::TEXT_PLAIN);
    for recipient in to {
        builder = builder.to(parse_mailbox(recipient)?);
    }
    let email = builder
        .body(body.to_string())
        .map_err(EmailError::BuildMessage)?;
    transport
        .send(email)
        .await
        .map_err(|e| EmailError::Send(Box::new(e)))?;
    Ok(())
}

fn parse_mailbox(addr: &str) -> Result<Mailbox, EmailError> {
    addr.parse::<Mailbox>()
        .map_err(|e| EmailError::BadAddress(addr.to_string(), e.to_string()))
}

/// Errors from an SMTP send attempt.
#[derive(Debug)]
enum EmailError {
    /// A `From:`/`To:` address could not be parsed.
    BadAddress(String, String),
    /// The `lettre::Message` could not be assembled.
    BuildMessage(lettre::error::Error),
    /// The SMTP transport rejected the message.
    Send(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for EmailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmailError::BadAddress(a, e) => write!(f, "bad email address {a:?}: {e}"),
            EmailError::BuildMessage(e) => write!(f, "failed to build email message: {e}"),
            EmailError::Send(e) => write!(f, "smtp send failed: {e}"),
        }
    }
}

impl std::error::Error for EmailError {}

/// Build the shared [`AlertState`] for the email hook when email alerting is
/// configured, else `None`. Preserved across hot-reloads when the email config
/// is unchanged so in-memory fired markers survive an unrelated config edit.
pub(crate) fn build_email_alert_state(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
) -> Option<Arc<AlertState>> {
    let email_cfg = config.oxllm_settings.email_alert_config.as_ref()?;
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if crate::config::validate_host_port(&email_cfg.smtp_host, email_cfg.smtp_port, allow_private)
        .is_err()
    {
        return None;
    }
    let thresholds = email_cfg.thresholds.clone();
    let budget_window = crate::scheduler::parse_budget_duration(
        config
            .oxllm_settings
            .budget_duration
            .as_deref()
            .unwrap_or("30d"),
    );
    let is_redis = crate::state::redis_configured(config);
    let alert_cache = if is_redis { Some(cache) } else { None };
    Some(Arc::new(
        AlertState::new()
            .with_name("email")
            .with_thresholds(thresholds)
            .with_budget_window(budget_window)
            .with_cache(alert_cache),
    ))
}

/// Build an [`EmailAlertingHook`] from `oxllm_settings.email_alert_config`
/// using the provided [`AlertState`] (constructed via
/// [`build_email_alert_state`] and preserved across hot-reloads). Returns
/// `None` when email alerting is not configured or misconfigured (logged at
/// warn level).
///
/// **SMTP SSRF note:** the smtp host is validated at config load via
/// [`crate::config::validate_host_port`], but unlike the HTTP-backed
/// alerting hooks there is no runtime DNS-resolver defence — lettre's SMTP
/// transport has no `dns_resolver` hook equivalent to reqwest's. The blast
/// radius is small (the email payload carries alert data only, no
/// secrets), and the load-time check closes the common case, but operators
/// who need stronger guarantees against DNS-rebinding should pin the SMTP
/// host to an IP literal in the config.
pub(crate) fn build_email_hook_opt(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
    state: Arc<AlertState>,
) -> Option<Arc<EmailAlertingHook>> {
    let email_cfg = config.oxllm_settings.email_alert_config.as_ref()?;

    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if let Err(e) =
        crate::config::validate_host_port(&email_cfg.smtp_host, email_cfg.smtp_port, allow_private)
    {
        tracing::warn!(
            error = %e,
            host = %email_cfg.smtp_host,
            port = email_cfg.smtp_port,
            "email_alert_config smtp_host failed SSRF validation; skipping"
        );
        return None;
    }

    let thresholds = email_cfg.thresholds.clone();
    let global_max_budget = config.oxllm_settings.max_budget;
    let model_max_budget = config
        .oxllm_settings
        .model_max_budget
        .clone()
        .unwrap_or_default();

    match EmailAlertingHook::new(
        cache,
        email_cfg.clone(),
        thresholds,
        state,
        global_max_budget,
        model_max_budget,
    ) {
        Ok(hook) => Some(Arc::new(
            hook.with_soft_budget_pct(email_cfg.soft_budget_pct),
        )),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build email alerting hook; skipping");
            None
        }
    }
}

#[async_trait]
impl Hook for EmailAlertingHook {
    async fn post_success(&self, ctx: &CallContext, _resp: &ResponseSummary) {
        if ctx.is_master || ctx.cache_hit {
            return;
        }
        for (scope, spend_cache_key, scope_id, limit) in
            alerting::budget_scopes(ctx, self.global_max_budget, &self.model_max_budget)
        {
            self.eval_scope(&scope, &spend_cache_key, &scope_id, Some(limit), ctx)
                .await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use oxllm_core::UsageSummary;
    use std::collections::HashMap;

    fn email_config() -> EmailAlertConfig {
        EmailAlertConfig {
            smtp_host: "smtp.example.com".to_string(),
            smtp_port: 587,
            smtp_username: Some("alerts@example.com".to_string()),
            smtp_password: Some("pass".to_string()),
            from: "alerts@example.com".to_string(),
            to: vec!["oncall@example.com".to_string()],
            ssl: true,
            thresholds: vec![0.8],
            soft_budget_pct: None,
        }
    }

    fn make_ctx() -> CallContext {
        CallContext {
            key_id: Some("k-1".to_string()),
            ..CallContext::for_test("gpt-4o")
        }
    }

    fn make_response() -> ResponseSummary {
        ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: Some(0.0025),
            usage: Some(UsageSummary {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[test]
    fn parses_from_yaml() {
        let yaml = "\
smtp_host: smtp.example.com
smtp_port: 587
smtp_username: alerts@example.com
smtp_password: pass
from: alerts@example.com
to: [oncall@example.com, ops@example.com]
ssl: true
thresholds: [0.8, 0.9, 1.0]
";
        let cfg: EmailAlertConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.smtp_host, "smtp.example.com");
        assert_eq!(cfg.smtp_port, 587);
        assert_eq!(cfg.from, "alerts@example.com");
        assert_eq!(cfg.to.len(), 2);
        assert!(cfg.ssl);
        assert_eq!(cfg.thresholds, vec![0.8, 0.9, 1.0]);
    }

    #[test]
    fn parses_defaults_when_omitted() {
        let yaml = "\
smtp_host: smtp.example.com
from: alerts@example.com
to: [oncall@example.com]
";
        let cfg: EmailAlertConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.smtp_port, 465, "default port is implicit-TLS 465");
        assert!(cfg.ssl, "ssl defaults to true");
        assert_eq!(cfg.thresholds, vec![0.8, 0.9, 1.0]);
    }

    #[test]
    fn build_email_payload_has_correct_shape() {
        let ctx = make_ctx();
        let payload = build_email_payload(
            &email_config(),
            "key:abc",
            "abc",
            0.8,
            80.0,
            100.0,
            "2024-01-01T00:00:00Z",
            &ctx,
        );
        assert_eq!(payload["from"], "alerts@example.com");
        assert_eq!(payload["to"][0], "oncall@example.com");
        assert_eq!(
            payload["subject"],
            "Budget alert: key:abc reached 80% of its limit"
        );
        let body = payload["body"].as_str().unwrap();
        assert!(body.contains("Scope: key:abc"), "body: {body}");
        assert!(body.contains("Threshold: 80%"), "body: {body}");
        assert!(body.contains("Current spend: $80.00"), "body: {body}");
        assert!(body.contains("Max budget: $100.00"), "body: {body}");
        assert!(body.contains("Request ID: req-1"), "body: {body}");
    }

    fn build_hook(
        cache: Arc<dyn Cache>,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
    ) -> EmailAlertingHook {
        EmailAlertingHook::new(
            cache,
            email_config(),
            thresholds,
            state,
            None,
            HashMap::new(),
        )
        .expect("build hook")
    }

    #[test]
    fn construction_rejects_empty_recipients() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let cfg = EmailAlertConfig {
            to: vec![],
            ..email_config()
        };
        let res = EmailAlertingHook::new(
            cache,
            cfg,
            vec![0.8],
            Arc::new(AlertState::new()),
            None,
            HashMap::new(),
        );
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn post_success_skips_cache_hit() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), vec![0.8], state.clone());

        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 90.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            !state.was_fired("key:abc", 0.8).await,
            "cache hits must not fire alerts"
        );
    }

    #[tokio::test]
    async fn post_success_skips_master_key() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), vec![0.8], state.clone());

        let mut ctx = make_ctx();
        ctx.is_master = true;
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 90.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            !state.was_fired("key:abc", 0.8).await,
            "master key must not fire alerts"
        );
    }

    #[tokio::test]
    async fn post_success_fires_on_threshold_cross() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), vec![0.8], state.clone());

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            state.was_fired("key:abc", 0.8).await,
            "threshold cross must fire an alert"
        );
    }

    #[tokio::test]
    async fn post_success_fires_on_model_budget_threshold_cross() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let mut model_budgets = HashMap::new();
        model_budgets.insert("gpt-4o".to_string(), 100.0);
        let hook = EmailAlertingHook::new(
            cache.clone(),
            email_config(),
            vec![0.8],
            state.clone(),
            None,
            model_budgets,
        )
        .expect("build hook");

        let ctx = make_ctx();
        let _ = cache.incr("spend:model:gpt-4o", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            state.was_fired("model:gpt-4o", 0.8).await,
            "per-model budget threshold must fire alerts"
        );
    }

    #[tokio::test]
    async fn alert_does_not_refire_for_same_threshold() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), vec![0.8], state.clone());

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        // Second call: spend went up but still in the same threshold band.
        let _ = cache.incr("spend:abc", 5.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        let fired_again = state.mark_fired_if_new("key:abc", 0.8).await;
        assert!(
            !fired_again,
            "email alert must not refire for the same threshold until budget resets"
        );
    }

    #[tokio::test]
    async fn post_success_does_not_block_request_on_send_failure() {
        // The transport points at an unroutable host; the spawned send task
        // fails fast and the request path must return immediately.
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let cfg = EmailAlertConfig {
            // RFC 5737 documentation address; resolves nowhere useful and the
            // SMTP connect will fail — but only inside the spawned task.
            smtp_host: "192.0.2.1".to_string(),
            smtp_port: 25,
            ssl: false,
            ..email_config()
        };
        let hook = EmailAlertingHook::new(
            cache.clone(),
            cfg,
            vec![0.8],
            state.clone(),
            None,
            HashMap::new(),
        )
        .expect("build hook");

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        let started = std::time::Instant::now();
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "post_success must return immediately even when SMTP is unreachable"
        );
    }

    /// The transport built by `build_transport` must apply
    /// [`REQUEST_TIMEOUT`] (15s) to the SMTP connect, not lettre's 60s
    /// default. The fixture points the transport at `192.0.2.1` (RFC 5737
    /// documentation range — unrouted in every real network, so the TCP
    /// SYN is blackholed and the connect only returns once the timeout
    /// fires). A regression that drops the `.timeout(Some(...))` call
    /// would let lettre fall back to its 60s default; the test bounds
    /// elapsed at 25s so a regression fails here in 25s rather than 60s.
    #[tokio::test]
    async fn smtp_send_completes_within_configured_timeout() {
        let cfg = EmailAlertConfig {
            smtp_host: "192.0.2.1".to_string(),
            smtp_port: 25,
            ssl: false,
            ..email_config()
        };
        let transport = build_transport(&cfg).expect("build transport");
        let started = std::time::Instant::now();
        let outcome = tokio::time::timeout(
            Duration::from_secs(25),
            send_email(
                &transport,
                "alerts@example.com",
                &["oncall@example.com".to_string()],
                "subject",
                "body",
            ),
        )
        .await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(25),
            "send_email must complete within 25s (15s lettre timeout + slack); took {elapsed:?}"
        );
        // The send must have completed (with an error, since the host is
        // unroutable) rather than been cut off by the test's outer bound.
        assert!(
            outcome.is_ok(),
            "send_email must return before the 25s test bound (regression: lettre 60s default)"
        );
    }

    fn proxy_with_email(yaml: &str) -> crate::config::ProxyConfig {
        crate::config::parse_config_str(yaml).expect("parse config")
    }

    #[tokio::test]
    async fn build_email_hook_opt_rejects_private_smtp_host() {
        for host in ["127.0.0.1", "169.254.169.254", "10.0.0.5"] {
            let yaml = format!(
                "\
model_list: []
oxllm_settings:
  email_alert_config:
    smtp_host: {host}
    smtp_port: 25
    from: alerts@example.com
    to: [oncall@example.com]
    ssl: false
"
            );
            let err = crate::config::parse_config_str(&yaml)
                .expect_err("private smtp_host must be SSRF-rejected at config load");
            let full = format!("{err:#}");
            assert!(
                full.contains("private"),
                "expected private-IP error for smtp_host {host}, got: {full}"
            );
        }
    }

    #[tokio::test]
    async fn build_email_hook_opt_allows_private_smtp_host_when_opted_in() {
        let yaml = "\
model_list: []
oxllm_settings:
  email_alert_config:
    smtp_host: 127.0.0.1
    smtp_port: 25
    from: alerts@example.com
    to: [oncall@example.com]
    ssl: false
general_settings:
  allow_private_api_base: true
";
        let config = proxy_with_email(yaml);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = build_email_alert_state(&config, cache.clone()).expect("state built");
        assert!(
            build_email_hook_opt(&config, cache, state).is_some(),
            "allow_private_api_base: true must permit a private smtp_host"
        );
    }

    /// Soft-budget parity with the generic-webhook hook
    /// (`hook_fires_soft_budget_exceeded_event`): with `soft_budget_pct`
    /// set, crossing the derived ceiling fires an alert email whose body
    /// identifies the soft-budget event. The state marker uses the same
    /// sentinel so dedup semantics match the generic hook.
    #[tokio::test]
    async fn email_hook_fires_soft_budget_exceeded_event() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), vec![], state.clone()).with_soft_budget_pct(Some(0.8));

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "soft_budget marker must be set when spend crosses the derived ceiling"
        );
    }

    /// Without `soft_budget_pct`, the email hook must not fire a soft event
    /// even when spend is high.
    #[tokio::test]
    async fn email_hook_without_soft_budget_pct_does_not_fire_soft_event() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), vec![0.8], state.clone());

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(state.was_fired("key:abc", 0.8).await);
        assert!(
            !state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "soft_budget must not fire when pct is unset on the email hook"
        );
    }

    #[test]
    fn email_soft_budget_payload_omits_threshold_line() {
        let ctx = make_ctx();
        let payload = build_email_payload(
            &email_config(),
            "key:abc",
            "abc",
            SOFT_BUDGET_SENTINEL,
            85.0,
            100.0,
            "2024-01-01T00:00:00Z",
            &ctx,
        );
        let body = payload["body"].as_str().unwrap();
        assert!(
            body.contains("soft_budget_exceeded"),
            "soft email body must identify the event: {body}"
        );
        assert!(
            !body.contains("Threshold:"),
            "soft email body must omit the Threshold line: {body}"
        );
    }
}
