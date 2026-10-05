//! PagerDuty budget-threshold alerting hook.
//!
//! A [`Hook`] that fires [PagerDuty Events API v2](https://developer.pagerduty.com/docs/send-alert-event)
//! events when a key/team/user/org/project/model spend crosses a
//! configurable percentage of its `max_budget`. It is the PagerDuty analogue
//! of [`SlackAlertingHook`](crate::hooks::SlackAlertingHook): same
//! threshold-evaluation logic and the same dedup via [`AlertState`], but the
//! notification goes to PagerDuty as a trigger event instead of a Slack
//! Block Kit message.
//!
//! Wired in from `oxllm_settings.pagerduty_config`. The hook owns its own
//! [`AlertState`] so it can coexist with the Slack and generic webhook
//! alerting hooks (all fire independently when configured together).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::{Cache, CallContext, Hook, ResponseSummary};
use serde_json::{Value, json};

use crate::config::types::PagerDutyConfig;
use crate::hooks::alerting::{self, AlertState, SOFT_BUDGET_SENTINEL};

const DEFAULT_ENDPOINT: &str = "https://events.pagerduty.com/v2/enqueue";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

fn now_iso() -> String {
    jiff::Timestamp::now().to_string()
}

/// A `Hook` that fires PagerDuty Events API v2 alerts when budget
/// thresholds are crossed.
pub struct PagerDutyAlertingHook {
    cache: Arc<dyn Cache>,
    config: PagerDutyConfig,
    thresholds: Vec<f64>,
    state: Arc<AlertState>,
    client: reqwest::Client,
    endpoint: String,
    global_max_budget: Option<f64>,
    /// Per-model global spend caps, mirrored from
    /// `oxllm_settings.model_max_budget`.
    model_max_budget: Arc<HashMap<String, f64>>,
    /// Optional soft-budget percentage of each scope's `max_budget`. When
    /// set (e.g. `0.8`), crossing it fires a non-blocking PagerDuty
    /// `severity: warning` trigger with a `dedup_key` distinct from the
    /// ratio-threshold events. Mirrors the same field on
    /// [`AlertingHook`](crate::hooks::AlertingHook). `None` disables
    /// soft-budget alerts.
    soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for PagerDutyAlertingHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagerDutyAlertingHook")
            .field("endpoint", &self.endpoint)
            .field("thresholds", &self.thresholds)
            .finish()
    }
}

impl PagerDutyAlertingHook {
    /// Build a PagerDuty alerting hook from a config, threshold list, shared
    /// alert state, and the global/model budget ceilings. The endpoint is
    /// taken from `config.endpoint` when set, otherwise
    /// `DEFAULT_ENDPOINT`. When `allow_private` is `false`, the HTTP
    /// client carries the connect-time [`crate::security::SsrfResolver`]
    /// so a DNS-rebinding attack on a custom operator-configured endpoint
    /// cannot steer the POST to a private address (the default
    /// `events.pagerduty.com` endpoint is a literal so its surface is
    /// negligible, but a custom endpoint is operator-controlled).
    pub fn new(
        cache: Arc<dyn Cache>,
        config: PagerDutyConfig,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
        global_max_budget: Option<f64>,
        model_max_budget: HashMap<String, f64>,
        allow_private: bool,
    ) -> anyhow::Result<Self> {
        let client = crate::security::build_ssrf_http_client(allow_private, REQUEST_TIMEOUT)?;
        let endpoint = config
            .endpoint
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        Ok(Self {
            cache,
            config,
            thresholds,
            state,
            client,
            endpoint,
            global_max_budget,
            model_max_budget: Arc::new(model_max_budget),
            soft_budget_pct: None,
        })
    }

    /// Builder: set the soft-budget percentage (e.g. `Some(0.8)` → fire a
    /// non-blocking PagerDuty trigger when a scope's spend crosses 80% of
    /// its `max_budget`). `None` (the default) disables soft-budget
    /// alerts. Parity with
    /// [`AlertingHook::with_soft_budget_pct`](crate::hooks::AlertingHook::with_soft_budget_pct).
    pub fn with_soft_budget_pct(mut self, pct: Option<f64>) -> Self {
        self.soft_budget_pct = pct;
        self
    }

    /// The configured endpoint URL (for tests).
    pub fn endpoint(&self) -> &str {
        &self.endpoint
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
            let payload = build_pagerduty_payload(
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
        let url = self.endpoint.clone();
        tokio::spawn(async move {
            if let Err(e) = send_pagerduty(&client, &url, &payload).await {
                tracing::warn!(
                    error = %e,
                    url = %url,
                    "pagerduty event enqueue POST failed"
                );
            }
        });
    }
}

#[async_trait]
impl Hook for PagerDutyAlertingHook {
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

/// Build a PagerDuty Events API v2 trigger payload for a single threshold
/// crossing. A `threshold` of [`SOFT_BUDGET_SENTINEL`] renders the
/// soft-budget variant: `severity: warning`, a `dedup_key` distinct from
/// the ratio-threshold events (so PagerDuty does not collapse them), and
/// `threshold_pct` omitted from `custom_details` (the soft ceiling is an
/// absolute USD amount, not a ratio).
#[allow(clippy::too_many_arguments)]
fn build_pagerduty_payload(
    config: &PagerDutyConfig,
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
    let severity = if is_soft {
        "warning"
    } else if threshold >= 1.0 {
        "critical"
    } else {
        "warning"
    };
    let summary = if is_soft {
        format!("Budget alert: {scope} crossed its soft budget")
    } else {
        format!("Budget alert: {scope} reached {pct}% of its limit")
    };
    // Distinct dedup_key for soft-budget events so PagerDuty does not
    // collapse them with the ratio-threshold incident for the same scope.
    let dedup_key = if is_soft {
        format!("{scope}:soft")
    } else {
        format!("{scope}:{pct}")
    };
    let mut custom_details = json!({
        "event": if is_soft { "soft_budget_exceeded" } else { "budget_threshold_crossed" },
        "scope": scope,
        "scope_id": scope_id,
        "current_spend": current_spend,
        "max_budget": max_budget,
        "request_id": ctx.request_id,
    });
    if !is_soft {
        custom_details["threshold_pct"] = json!(pct);
    }

    json!({
        "routing_key": config.routing_key,
        "event_action": "trigger",
        "dedup_key": dedup_key,
        "payload": {
            "summary": summary,
            "timestamp": timestamp,
            "severity": severity,
            "source": "oxllm",
            "component": "budget",
            "custom_details": custom_details,
        },
        "client": "oxllm",
    })
}

async fn send_pagerduty(
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

/// Build a [`PagerDutyAlertingHook`] from `oxllm_settings.pagerduty_config`
/// when present. The hook owns its own [`AlertState`] (so it fires
/// independently of the Slack and generic webhook alerting hooks). Returns
/// `None` when pagerduty is not configured or misconfigured (logged at warn
/// level).
/// Build the shared [`AlertState`] for the PagerDuty hook when pagerduty is
/// configured, else `None`. Preserved across hot-reloads when the pagerduty
/// config is unchanged so in-memory fired markers survive an unrelated config
/// edit.
pub(crate) fn build_pagerduty_alert_state(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
) -> Option<Arc<AlertState>> {
    let pd_cfg = config.oxllm_settings.pagerduty_config.as_ref()?;
    if pd_cfg.routing_key.is_empty() {
        return None;
    }
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if let Some(endpoint) = pd_cfg.endpoint.as_deref() {
        if !endpoint.trim().is_empty()
            && crate::config::validate_api_base(endpoint, allow_private).is_err()
        {
            return None;
        }
    }
    let thresholds = pd_cfg.thresholds.clone();
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
            .with_name("pagerduty")
            .with_thresholds(thresholds)
            .with_budget_window(budget_window)
            .with_cache(alert_cache),
    ))
}

/// Build a [`PagerDutyAlertingHook`] from `oxllm_settings.pagerduty_config`
/// using the provided [`AlertState`] (constructed via
/// [`build_pagerduty_alert_state`] and preserved across hot-reloads). Returns
/// `None` when pagerduty is not configured or misconfigured (logged at warn
/// level).
pub(crate) fn build_pagerduty_hook_opt(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
    state: Arc<AlertState>,
) -> Option<Arc<PagerDutyAlertingHook>> {
    let pd_cfg = config.oxllm_settings.pagerduty_config.as_ref()?;
    if pd_cfg.routing_key.is_empty() {
        tracing::warn!("pagerduty_config.routing_key is empty; skipping pagerduty hook");
        return None;
    }
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if let Some(endpoint) = pd_cfg.endpoint.as_deref() {
        if !endpoint.trim().is_empty() {
            if let Err(e) = crate::config::validate_api_base(endpoint, allow_private) {
                tracing::warn!(
                    error = %e,
                    "pagerduty_config endpoint '{endpoint}' failed SSRF validation; skipping"
                );
                return None;
            }
        }
    }

    let thresholds = pd_cfg.thresholds.clone();
    let global_max_budget = config.oxllm_settings.max_budget;
    let model_max_budget = config
        .oxllm_settings
        .model_max_budget
        .clone()
        .unwrap_or_default();

    match PagerDutyAlertingHook::new(
        cache,
        pd_cfg.clone(),
        thresholds,
        state,
        global_max_budget,
        model_max_budget,
        allow_private,
    ) {
        Ok(hook) => Some(Arc::new(hook.with_soft_budget_pct(pd_cfg.soft_budget_pct))),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build pagerduty alerting hook; skipping");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use oxllm_core::UsageSummary;
    use std::collections::HashMap;

    fn pd_config() -> PagerDutyConfig {
        PagerDutyConfig {
            routing_key: "rk-key".to_string(),
            thresholds: vec![0.8],
            endpoint: None,
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
routing_key: rk-key
thresholds: [0.8, 0.9, 1.0]
";
        let cfg: PagerDutyConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.routing_key, "rk-key");
        assert_eq!(cfg.thresholds, vec![0.8, 0.9, 1.0]);
        assert!(
            cfg.endpoint.is_none(),
            "endpoint defaults to None (DEFAULT_ENDPOINT)"
        );
    }

    #[test]
    fn parses_default_thresholds_when_omitted() {
        let yaml = "routing_key: rk-key\n";
        let cfg: PagerDutyConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.thresholds, vec![0.8, 0.9, 1.0]);
    }

    #[test]
    fn parses_custom_endpoint_override() {
        let yaml = "\
routing_key: rk-key
endpoint: https://events.eu.pagerduty.com/v2/enqueue
";
        let cfg: PagerDutyConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(
            cfg.endpoint.as_deref(),
            Some("https://events.eu.pagerduty.com/v2/enqueue")
        );
    }

    #[test]
    fn pagerduty_hook_uses_default_endpoint_when_unset() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = PagerDutyAlertingHook::new(
            cache,
            pd_config(),
            vec![0.8],
            Arc::new(AlertState::new()),
            None,
            HashMap::new(),
            true,
        )
        .expect("build hook");
        assert_eq!(hook.endpoint(), DEFAULT_ENDPOINT);
    }

    #[test]
    fn pagerduty_hook_uses_custom_endpoint_when_set() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let cfg = PagerDutyConfig {
            routing_key: "rk-key".to_string(),
            thresholds: vec![0.8],
            endpoint: Some("https://events.example.com/v2/enqueue".to_string()),
            soft_budget_pct: None,
        };
        let hook = PagerDutyAlertingHook::new(
            cache,
            cfg,
            vec![0.8],
            Arc::new(AlertState::new()),
            None,
            HashMap::new(),
            true,
        )
        .expect("build hook");
        assert_eq!(hook.endpoint(), "https://events.example.com/v2/enqueue");
    }

    #[test]
    fn pagerduty_payload_has_correct_shape() {
        let ctx = make_ctx();
        let payload = build_pagerduty_payload(
            &pd_config(),
            "key:abc",
            "abc",
            0.8,
            80.0,
            100.0,
            "2024-01-01T00:00:00Z",
            &ctx,
        );
        assert_eq!(payload["routing_key"], "rk-key");
        assert_eq!(payload["event_action"], "trigger");
        assert_eq!(payload["dedup_key"], "key:abc:80");
        assert_eq!(
            payload["payload"]["summary"],
            "Budget alert: key:abc reached 80% of its limit"
        );
        assert_eq!(payload["payload"]["severity"], "warning");
        assert_eq!(payload["payload"]["timestamp"], "2024-01-01T00:00:00Z");
        assert_eq!(payload["payload"]["source"], "oxllm");
        assert_eq!(payload["payload"]["component"], "budget");
        assert_eq!(payload["payload"]["custom_details"]["scope"], "key:abc");
        assert_eq!(payload["payload"]["custom_details"]["threshold_pct"], 80);
        assert_eq!(payload["payload"]["custom_details"]["current_spend"], 80.0);
        assert_eq!(payload["payload"]["custom_details"]["max_budget"], 100.0);
        assert_eq!(payload["payload"]["custom_details"]["request_id"], "req-1");
        assert_eq!(payload["client"], "oxllm");
    }

    #[test]
    fn pagerduty_payload_severity_critical_at_full_budget() {
        let ctx = make_ctx();
        let payload =
            build_pagerduty_payload(&pd_config(), "key:abc", "abc", 1.0, 100.0, 100.0, "t", &ctx);
        assert_eq!(payload["payload"]["severity"], "critical");
    }

    fn build_hook(
        cache: Arc<dyn Cache>,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
    ) -> PagerDutyAlertingHook {
        PagerDutyAlertingHook::new(
            cache,
            pd_config(),
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
        let hook = PagerDutyAlertingHook::new(
            cache.clone(),
            pd_config(),
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

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v2/enqueue")
                .body_contains("\"routing_key\":\"rk-key\"")
                .body_contains("\"event_action\":\"trigger\"")
                .body_contains("Budget alert");
            then.status(202);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let mut hook = build_hook(cache.clone(), vec![0.8], state.clone());
        // The child `tests` module can override the private endpoint field to
        // point at the mock server.
        hook.endpoint = server.base_url() + "/v2/enqueue";

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
            "pagerduty event should have been POSTed when threshold crossed"
        );
    }

    #[tokio::test]
    async fn webhook_does_not_refire_for_same_threshold() {
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

        assert!(
            state.was_fired("key:abc", 0.8).await,
            "alert state must be set"
        );
        // A second call that crosses the same threshold must not re-mark;
        // dedup is enforced by AlertState::mark_fired_if_new.
        let fired_again = state.mark_fired_if_new("key:abc", 0.8).await;
        assert!(
            !fired_again,
            "pagerduty must not refire for the same threshold until budget resets"
        );
    }

    #[tokio::test]
    async fn post_success_does_not_block_request_on_post_failure() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = build_hook(cache.clone(), vec![0.8], state.clone());

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
    /// set, crossing the derived ceiling fires a PagerDuty trigger whose
    /// `dedup_key` is distinct from any ratio-threshold event for the
    /// same scope and whose severity is `warning`.
    #[tokio::test]
    async fn pagerduty_hook_fires_soft_budget_exceeded_event() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let mut hook =
            build_hook(cache.clone(), vec![], state.clone()).with_soft_budget_pct(Some(0.8));
        hook.endpoint = server.base_url() + "/v2/enqueue";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v2/enqueue")
                .body_contains("soft_budget_exceeded");
            then.status(202);
        });

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
            "soft_budget_exceeded pagerduty event must fire when spend crosses 80% of max_budget"
        );
        assert!(
            state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "sentinel marker must be set"
        );
    }

    /// Without `soft_budget_pct`, the PagerDuty hook must not fire a soft
    /// event even when spend is high.
    #[tokio::test]
    async fn pagerduty_hook_without_soft_budget_pct_does_not_fire_soft_event() {
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
            "soft_budget must not fire when pct is unset on the pagerduty hook"
        );
    }

    #[test]
    fn pagerduty_soft_budget_payload_has_distinct_dedup_key_and_no_threshold_pct() {
        let ctx = make_ctx();
        let payload = build_pagerduty_payload(
            &pd_config(),
            "key:abc",
            "abc",
            SOFT_BUDGET_SENTINEL,
            85.0,
            100.0,
            "2024-01-01T00:00:00Z",
            &ctx,
        );
        assert_eq!(payload["dedup_key"], "key:abc:soft", "distinct dedup_key");
        assert_eq!(payload["payload"]["severity"], "warning");
        assert!(
            payload["payload"]["custom_details"]
                .get("threshold_pct")
                .is_none(),
            "soft-budget payload must omit threshold_pct"
        );
        // Sanity: a ratio-threshold payload for the same scope produces a
        // different dedup_key, so PagerDuty does not collapse them.
        let ratio_payload = build_pagerduty_payload(
            &pd_config(),
            "key:abc",
            "abc",
            0.8,
            80.0,
            100.0,
            "2024-01-01T00:00:00Z",
            &ctx,
        );
        assert_ne!(
            payload["dedup_key"], ratio_payload["dedup_key"],
            "soft and ratio dedup_keys must not collide"
        );
    }
}
