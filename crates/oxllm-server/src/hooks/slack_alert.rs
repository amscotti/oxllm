//! Slack budget-threshold alerting hook.
//!
//! A [`Hook`] that fires [Slack incoming webhook](https://api.slack.com/messaging/webhooks)
//! notifications — formatted as [Block Kit](https://api.slack.com/block-kit)
//! messages — when a key/team/user/org/project/model spend crosses a
//! configurable percentage of its `max_budget`. It is the Slack analogue of
//! [`AlertingHook`](crate::hooks::AlertingHook): same threshold-evaluation
//! logic and the same dedup via [`AlertState`], but the notification goes to
//! Slack as a Block Kit message instead of a plain JSON webhook.
//!
//! Wired in from `oxllm_settings.slack_alerting_config`. The Slack hook
//! owns its own [`AlertState`] so it can coexist with the generic webhook
//! alerting hook (both fire independently when configured together).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::{Cache, CallContext, Hook, ResponseSummary};
use serde_json::{Value, json};

use crate::config::types::SlackAlertConfig;
use crate::hooks::alerting::{self, AlertState, SOFT_BUDGET_SENTINEL};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

fn now_iso() -> String {
    jiff::Timestamp::now().to_string()
}

/// A `Hook` that fires Slack Block Kit alerts when budget thresholds are
/// crossed.
pub struct SlackAlertingHook {
    cache: Arc<dyn Cache>,
    config: SlackAlertConfig,
    thresholds: Vec<f64>,
    state: Arc<AlertState>,
    client: reqwest::Client,
    global_max_budget: Option<f64>,
    /// Per-model global spend caps, mirrored from
    /// `oxllm_settings.model_max_budget`.
    model_max_budget: Arc<HashMap<String, f64>>,
    /// Optional soft-budget percentage of each scope's `max_budget`. When
    /// set (e.g. `0.8`), a scope's soft ceiling is `max_budget * pct`;
    /// crossing it fires a non-blocking `"soft_budget_exceeded"` Slack
    /// alert, deduped per scope via [`AlertState`]. Mirrors the same field
    /// on [`AlertingHook`](crate::hooks::AlertingHook). `None` disables
    /// soft-budget alerts.
    soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for SlackAlertingHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlackAlertingHook")
            .field(
                "webhook_url",
                &alerting::redact_url(&self.config.webhook_url),
            )
            .field("thresholds", &self.thresholds)
            .finish()
    }
}

impl SlackAlertingHook {
    /// Build a Slack alerting hook from a webhook config, threshold list,
    /// shared alert state, and the global/model budget ceilings. The HTTP
    /// client carries the connect-time [`crate::security::SsrfResolver`]
    /// unless `allow_private` is set, so a DNS-rebinding attack on the
    /// operator-configured webhook after config load cannot steer the
    /// connection to a private address.
    pub fn new(
        cache: Arc<dyn Cache>,
        config: SlackAlertConfig,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
        global_max_budget: Option<f64>,
        model_max_budget: HashMap<String, f64>,
        allow_private: bool,
    ) -> anyhow::Result<Self> {
        let client = crate::security::build_ssrf_http_client(allow_private, REQUEST_TIMEOUT)?;
        Ok(Self {
            cache,
            config,
            thresholds,
            state,
            client,
            global_max_budget,
            model_max_budget: Arc::new(model_max_budget),
            soft_budget_pct: None,
        })
    }

    /// Builder: set the soft-budget percentage (e.g. `Some(0.8)` → fire a
    /// non-blocking `"soft_budget_exceeded"` Slack alert when a scope's
    /// spend crosses 80% of its `max_budget`). `None` (the default)
    /// disables soft-budget alerts. Parity with
    /// [`AlertingHook::with_soft_budget_pct`](crate::hooks::AlertingHook::with_soft_budget_pct).
    pub fn with_soft_budget_pct(mut self, pct: Option<f64>) -> Self {
        self.soft_budget_pct = pct;
        self
    }

    /// The configured webhook URL (for tests).
    pub fn webhook_url(&self) -> &str {
        &self.config.webhook_url
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
            let payload = build_slack_payload(
                &self.config,
                scope,
                scope_id,
                hit.threshold,
                hit.current_spend,
                limit,
                &now_iso(),
                ctx,
            );
            self.dispatch(payload);
        }
    }

    fn dispatch(&self, payload: Value) {
        let client = self.client.clone();
        let url = self.config.webhook_url.clone();
        tokio::spawn(async move {
            if let Err(e) = send_slack(&client, &url, &payload).await {
                tracing::warn!(
                    error = %e,
                    url = %alerting::redact_url(&url),
                    "slack alert webhook POST failed"
                );
            }
        });
    }
}

#[async_trait]
impl Hook for SlackAlertingHook {
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

/// Build a Slack incoming-webhook payload (Block Kit) for a single threshold
/// crossing. A `threshold` of [`SOFT_BUDGET_SENTINEL`] renders the
/// soft-budget variant (header text and field set differ; no percentage is
/// shown because the soft ceiling is an absolute USD amount, not a ratio).
#[allow(clippy::too_many_arguments)]
fn build_slack_payload(
    config: &SlackAlertConfig,
    scope: &str,
    scope_id: &str,
    threshold: f64,
    current_spend: f64,
    max_budget: f64,
    timestamp: &str,
    ctx: &CallContext,
) -> Value {
    let is_soft = threshold == SOFT_BUDGET_SENTINEL;
    let pct = (threshold * 100.0).round() as i64;
    let header_text = if is_soft {
        format!("Budget alert: {scope} crossed its soft budget")
    } else {
        format!("Budget alert: {scope} reached {pct}% of its limit")
    };

    let fields = if is_soft {
        json!([
            { "type": "mrkdwn", "text": format!("*Scope*\n`{scope}`") },
            { "type": "mrkdwn", "text": format!("*Scope ID*\n`{scope_id}`") },
            { "type": "mrkdwn", "text": "*Event*\nsoft_budget_exceeded" },
            { "type": "mrkdwn", "text": format!("*Current spend*\n${current_spend:.2}") },
            { "type": "mrkdwn", "text": format!("*Max budget*\n${max_budget:.2}") },
            { "type": "mrkdwn", "text": format!("*Request ID*\n`{}`", ctx.request_id) },
        ])
    } else {
        json!([
            { "type": "mrkdwn", "text": format!("*Scope*\n`{scope}`") },
            { "type": "mrkdwn", "text": format!("*Scope ID*\n`{scope_id}`") },
            { "type": "mrkdwn", "text": format!("*Threshold*\n{pct}%") },
            { "type": "mrkdwn", "text": format!("*Current spend*\n${current_spend:.2}") },
            { "type": "mrkdwn", "text": format!("*Max budget*\n${max_budget:.2}") },
            { "type": "mrkdwn", "text": format!("*Request ID*\n`{}`", ctx.request_id) },
        ])
    };

    let mut payload = json!({
        "blocks": [
            {
                "type": "header",
                "text": { "type": "plain_text", "text": header_text }
            },
            {
                "type": "section",
                "fields": fields
            },
            {
                "type": "context",
                "elements": [
                    { "type": "mrkdwn", "text": format!("Fired at {timestamp}") }
                ]
            }
        ]
    });

    if let Some(channel) = config.channel.as_deref() {
        if !channel.is_empty() {
            payload["channel"] = json!(channel);
        }
    }
    payload
}

async fn send_slack(
    client: &reqwest::Client,
    url: &str,
    payload: &Value,
) -> Result<(), reqwest::Error> {
    client
        .post(url)
        .json(payload)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

/// Build a [`SlackAlertingHook`] from `oxllm_settings.slack_alerting_config`
/// when present. The hook owns its own [`AlertState`] (so it fires
/// independently of the generic webhook alerting hook). Returns `None` when
/// slack alerting is not configured or misconfigured (logged at warn level).
/// Build the shared [`AlertState`] for the Slack hook when slack alerting is
/// configured, else `None`. The returned `Arc` is preserved across hot-reloads
/// when the slack config is unchanged (see `state::reload`) so in-memory fired
/// markers are not discarded on an unrelated config edit.
pub(crate) fn build_slack_alert_state(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
) -> Option<Arc<AlertState>> {
    let slack_cfg = config.oxllm_settings.slack_alerting_config.as_ref()?;
    if slack_cfg.webhook_url.is_empty() {
        return None;
    }
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if crate::config::validate_api_base(&slack_cfg.webhook_url, allow_private).is_err() {
        return None;
    }
    let thresholds = slack_cfg.thresholds.clone();
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
            .with_name("slack")
            .with_thresholds(thresholds)
            .with_budget_window(budget_window)
            .with_cache(alert_cache),
    ))
}

/// Build a [`SlackAlertingHook`] from `oxllm_settings.slack_alerting_config`
/// using the provided [`AlertState`] (constructed via [`build_slack_alert_state`]
/// and preserved across hot-reloads). Returns `None` when slack alerting is not
/// configured or misconfigured (logged at warn level).
pub(crate) fn build_slack_hook_opt(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
    state: Arc<AlertState>,
) -> Option<Arc<SlackAlertingHook>> {
    let slack_cfg = config.oxllm_settings.slack_alerting_config.as_ref()?;
    if slack_cfg.webhook_url.is_empty() {
        tracing::warn!("slack_alerting_config.webhook_url is empty; skipping slack hook");
        return None;
    }

    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if let Err(e) = crate::config::validate_api_base(&slack_cfg.webhook_url, allow_private) {
        tracing::warn!(
            error = %e,
            "slack_alerting_config webhook URL '{}' failed SSRF validation; skipping",
            alerting::redact_url(&slack_cfg.webhook_url)
        );
        return None;
    }

    let thresholds = slack_cfg.thresholds.clone();
    let global_max_budget = config.oxllm_settings.max_budget;
    let model_max_budget = config
        .oxllm_settings
        .model_max_budget
        .clone()
        .unwrap_or_default();

    match SlackAlertingHook::new(
        cache,
        slack_cfg.clone(),
        thresholds,
        state,
        global_max_budget,
        model_max_budget,
        allow_private,
    ) {
        Ok(hook) => Some(Arc::new(
            hook.with_soft_budget_pct(slack_cfg.soft_budget_pct),
        )),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build slack alerting hook; skipping");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use crate::hooks::callback::CallbackConfig;
    use oxllm_core::UsageSummary;
    use std::collections::HashMap;

    fn slack_config(url: &str) -> SlackAlertConfig {
        SlackAlertConfig {
            webhook_url: url.to_string(),
            channel: None,
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

    // Silence unused-import warning for `CallbackConfig` when the test below
    // that uses it is compiled out on some feature sets.
    const _: fn() = || {
        let _ = CallbackConfig {
            url: String::new(),
            headers: HashMap::new(),
        };
    };

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
webhook_url: https://hooks.slack.com/services/T/B/X
channel: '#alerts'
thresholds: [0.8, 0.9, 1.0]
";
        let cfg: SlackAlertConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.webhook_url, "https://hooks.slack.com/services/T/B/X");
        assert_eq!(cfg.channel.as_deref(), Some("#alerts"));
        assert_eq!(cfg.thresholds, vec![0.8, 0.9, 1.0]);
    }

    #[test]
    fn parses_default_thresholds_when_omitted() {
        let yaml = "webhook_url: https://hooks.slack.com/services/T/B/X\n";
        let cfg: SlackAlertConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.thresholds, vec![0.8, 0.9, 1.0]);
    }

    #[test]
    fn slack_payload_is_block_kit() {
        let ctx = make_ctx();
        let payload = build_slack_payload(
            &slack_config("https://hooks.slack.com/x"),
            "key:abc",
            "abc",
            0.8,
            80.0,
            100.0,
            "2024-01-01T00:00:00Z",
            &ctx,
        );
        assert_eq!(payload["blocks"][0]["type"], "header");
        assert_eq!(
            payload["blocks"][0]["text"]["text"],
            "Budget alert: key:abc reached 80% of its limit"
        );
        assert_eq!(payload["blocks"][1]["type"], "section");
        let fields = payload["blocks"][1]["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[0]["text"], "*Scope*\n`key:abc`");
        assert_eq!(fields[3]["text"], "*Current spend*\n$80.00");
        assert_eq!(fields[4]["text"], "*Max budget*\n$100.00");
        assert_eq!(fields[5]["text"], "*Request ID*\n`req-1`");
        assert!(payload.get("channel").is_none(), "no channel override");
    }

    #[test]
    fn slack_payload_includes_channel_override() {
        let cfg = SlackAlertConfig {
            webhook_url: "https://hooks.slack.com/x".to_string(),
            channel: Some("#ops".to_string()),
            thresholds: vec![0.8],
            soft_budget_pct: None,
        };
        let ctx = make_ctx();
        let payload = build_slack_payload(&cfg, "key:abc", "abc", 0.8, 80.0, 100.0, "t", &ctx);
        assert_eq!(payload["channel"], "#ops");
    }

    /// LOW: the manual Debug impl must not leak the secret-bearing webhook
    /// URL (Slack tokens live in the path). A debug-printed hook must never
    /// contain the token substring.
    #[test]
    fn slack_hook_debug_impl_masks_webhook_url() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let token = "SECRET_TOKEN_XYZ";
        let hook = SlackAlertingHook::new(
            cache,
            SlackAlertConfig {
                webhook_url: format!("https://hooks.slack.com/services/T/B/{token}"),
                channel: None,
                thresholds: vec![0.8],
                soft_budget_pct: None,
            },
            vec![0.8],
            Arc::new(AlertState::new()),
            None,
            HashMap::new(),
            true,
        )
        .expect("build hook");
        let dbg = format!("{hook:?}");
        assert!(
            !dbg.contains(token),
            "Debug impl must not leak the webhook token, got: {dbg}"
        );
    }

    fn build_hook(
        cache: Arc<dyn Cache>,
        url: &str,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
    ) -> SlackAlertingHook {
        SlackAlertingHook::new(
            cache,
            slack_config(url),
            thresholds,
            state,
            None,
            HashMap::new(),
            true,
        )
        .expect("build hook")
    }

    #[tokio::test]
    async fn post_success_skips_cache_hit() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(
            cache.clone(),
            "http://example.invalid/x",
            vec![0.8],
            state.clone(),
        );

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
        let hook = build_hook(
            cache.clone(),
            "http://example.invalid/x",
            vec![0.8],
            state.clone(),
        );

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
        let hook = build_hook(
            cache.clone(),
            "http://example.invalid/x",
            vec![0.8],
            state.clone(),
        );

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
        let hook = SlackAlertingHook::new(
            cache.clone(),
            slack_config("http://example.invalid/x"),
            vec![0.8],
            state.clone(),
            None,
            model_budgets,
            true,
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
    async fn webhook_fires_when_threshold_crossed_httpmock() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/slack";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/slack")
                .body_contains("\"type\":\"header\"")
                .body_contains("Budget alert");
            then.status(200);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), &url, vec![0.8], state.clone());

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "slack webhook should have been called when threshold crossed"
        );
    }

    #[tokio::test]
    async fn webhook_does_not_refire_for_same_threshold() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/slack";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/slack");
            then.status(200);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), &url, vec![0.8], state.clone());

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        // Second call: spend went up but still in the same threshold band.
        let _ = cache.incr("spend:abc", 5.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            post_mock.hits(),
            1,
            "slack webhook must fire exactly once per threshold until budget resets"
        );
    }

    #[tokio::test]
    async fn post_success_does_not_block_request_on_post_failure() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(
            cache.clone(),
            "http://127.0.0.1:1/x",
            vec![0.8],
            state.clone(),
        );

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        let started = std::time::Instant::now();
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "post_success must return immediately even when endpoint is unreachable"
        );
    }

    /// Soft-budget parity with the generic-webhook hook
    /// (`hook_fires_soft_budget_exceeded_event`): with `soft_budget_pct`
    /// set, crossing the derived ceiling fires a Slack alert whose header
    /// text identifies the soft-budget event. Mirrors the reference
    /// `soft_budget_crossed` Slack signal.
    #[tokio::test]
    async fn slack_hook_fires_soft_budget_exceeded_event() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/slack";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/slack")
                .body_contains("soft_budget_exceeded");
            then.status(200);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook =
            build_hook(cache.clone(), &url, vec![], state.clone()).with_soft_budget_pct(Some(0.8));

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "soft_budget_exceeded slack alert must fire when spend crosses 80% of max_budget"
        );
        assert!(
            state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "sentinel marker must be set"
        );
    }

    /// Without `soft_budget_pct`, the Slack hook must not fire a soft event
    /// even when spend is high.
    #[tokio::test]
    async fn slack_hook_without_soft_budget_pct_does_not_fire_soft_event() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(
            cache.clone(),
            "http://example.invalid/x",
            vec![0.8],
            state.clone(),
        );

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(state.was_fired("key:abc", 0.8).await);
        assert!(
            !state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "soft_budget must not fire when pct is unset on the slack hook"
        );
    }

    /// MEDIUM: `soft_budget_pct` set in the Slack config YAML must reach the
    /// built hook. Previously the builder never called `with_soft_budget_pct`,
    /// leaving the whole soft-budget code path as dead code. This verifies the
    /// full config → hook wiring end-to-end through `build_slack_hook_opt`
    /// via an httpmock POST carrying the `soft_budget_exceeded` event.
    #[tokio::test]
    async fn build_slack_hook_opt_wires_soft_budget_pct_from_config() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/slack")
                .body_contains("soft_budget_exceeded");
            then.status(200);
        });

        let yaml = format!(
            "\
model_list: []
oxllm_settings:
  slack_alerting_config:
    webhook_url: http://127.0.0.1:{port}/slack
    thresholds: []
    soft_budget_pct: 0.8
general_settings:
  allow_private_api_base: true
",
            port = server.port()
        );
        let config = crate::config::parse_config_str(&yaml).expect("parse config");
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = build_slack_alert_state(&config, cache.clone()).expect("state built");
        let hook = build_slack_hook_opt(&config, cache.clone(), state).expect("hook built");

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "soft_budget_pct from config must be wired: soft_budget_exceeded slack POST must fire"
        );
    }
}
