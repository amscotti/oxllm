//! Langfuse observability hook.
//!
//! A [`Hook`] that batches `trace-create` + `generation-create` events and
//! POSTs them to the Langfuse ingestion API (`POST {host}/api/public/ingestion`)
//! using HTTP Basic auth.
//!
//! Flushing happens either when the in-memory buffer reaches `max_batch_size`
//! or on a periodic timer (`flush_interval_secs`). Both paths hand the batch
//! to a single background drainer task fed by a bounded `mpsc` channel, so the
//! request path is never blocked and at most one POST is in flight per hook
//! instance (matching [`crate::hooks::http_telemetry::HttpTelemetryHook`]'s
//! backpressure contract). Failures are logged at `warn` level and counted
//! under `oxllm_telemetry_dropped_total{backend="langfuse",reason="..."}`.
//!
//! Wired in from `oxllm_settings.langfuse_config`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::Hook;
use oxllm_core::{CallContext, ResponseSummary};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::config::types::LangfuseConfig;
use crate::hooks::cost_fallback::fallback_cost;
use crate::hooks::redact::{
    redact_secret_values_in_place, redact_sensitive, redact_sensitive_in_place,
};

const DEFAULT_FLUSH_INTERVAL_SECS: u64 = 5;
const DEFAULT_MAX_BATCH_SIZE: usize = 50;
const DEFAULT_LOG_INPUTS: bool = true;
const DEFAULT_LOG_OUTPUTS: bool = true;
/// Bounded channel capacity for the background POST drainer. Sized to absorb
/// short bursts of `max_batch_size`-triggered flushes without dropping under
/// steady-state load; when exceeded, batches are dropped and counted under
/// `oxllm_telemetry_dropped_total{backend="langfuse",reason="full"}`.
const DEFAULT_CHANNEL_CAPACITY: usize = 256;
/// Request timeout for a single ingestion POST.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// One entry in the Langfuse ingestion `batch` array. The `body` field is
/// polymorphic across event types (`trace-create` vs `generation-create`),
/// hence `serde_json::Value`.
#[derive(Debug, Clone, Serialize)]
pub struct LangfuseEvent {
    /// Unique event id (UUID). Langfuse dedupes on this.
    pub id: String,
    /// Event type: `"trace-create"` or `"generation-create"`.
    #[serde(rename = "type")]
    pub event_type: &'static str,
    /// Type-specific body.
    pub body: Value,
}

/// Internal snapshot of the resolved configuration.
#[derive(Debug, Clone)]
struct ResolvedConfig {
    host: String,
    public_key: String,
    secret_key: String,
    flush_interval: Duration,
    max_batch_size: usize,
    log_inputs: bool,
    log_outputs: bool,
    redact_secret_values: bool,
}

/// A `Hook` that batches Langfuse ingestion events and POSTs them in the
/// background. See the module docs for the wire format.
///
/// The hook holds only a bounded channel sender. A single background drainer
/// task owns an internal buffer, a periodic flush ticker, and the channel
/// receiver. The drainer exits when all senders are dropped (i.e. when every
/// `Arc<LangfuseHook>` is dropped), so hot-reloads that rebuild the hook
/// cleanly terminate the old drainer — no leaked tasks, no leaked `Arc<Self>`.
pub struct LangfuseHook {
    config: ResolvedConfig,
    ingestion_url: String,
    /// Bounded sender into the single background POST drainer. Cheap to clone
    /// per request; `try_send` never blocks.
    tx: mpsc::Sender<Vec<LangfuseEvent>>,
}

impl std::fmt::Debug for LangfuseHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangfuseHook")
            .field("host", &self.config.host)
            .field("max_batch_size", &self.config.max_batch_size)
            .field("flush_interval", &self.config.flush_interval)
            .field("log_inputs", &self.config.log_inputs)
            .field("log_outputs", &self.config.log_outputs)
            .finish()
    }
}

impl LangfuseHook {
    /// Build a hook from a raw config block. The SSRF-aware DNS resolver is
    /// installed on the reqwest client when `allow_private` is `false`.
    /// When `redact_secret_values` is `true`, the opt-in value-pattern
    /// redaction pass is applied to input/output in addition to the
    /// unconditional key-name denylist.
    pub fn new(
        config: &LangfuseConfig,
        allow_private: bool,
        redact_secret_values: bool,
    ) -> anyhow::Result<Self> {
        Self::with_capacity(
            config,
            DEFAULT_CHANNEL_CAPACITY,
            allow_private,
            redact_secret_values,
        )
    }

    /// Like [`LangfuseHook::new`] but with an explicit channel `capacity`,
    /// letting tests trade memory for drop-tolerance.
    pub fn with_capacity(
        config: &LangfuseConfig,
        capacity: usize,
        allow_private: bool,
        redact_secret_values: bool,
    ) -> anyhow::Result<Self> {
        if config.flush_interval_secs == Some(0) {
            anyhow::bail!("langfuse_config.flush_interval_secs must be positive");
        }
        if config.host.trim().is_empty() {
            anyhow::bail!("langfuse_config.host is empty");
        }
        if config.public_key.trim().is_empty() || config.secret_key.trim().is_empty() {
            anyhow::bail!("langfuse_config.public_key / secret_key must be set");
        }
        let host = config.host.trim_end_matches('/').to_string();

        let resolved = ResolvedConfig {
            host: host.clone(),
            public_key: config.public_key.clone(),
            secret_key: config.secret_key.clone(),
            flush_interval: Duration::from_secs(
                config
                    .flush_interval_secs
                    .unwrap_or(DEFAULT_FLUSH_INTERVAL_SECS),
            ),
            max_batch_size: config.max_batch_size.unwrap_or(DEFAULT_MAX_BATCH_SIZE),
            log_inputs: config.log_inputs.unwrap_or(DEFAULT_LOG_INPUTS),
            log_outputs: config.log_outputs.unwrap_or(DEFAULT_LOG_OUTPUTS),
            redact_secret_values,
        };

        let client = crate::security::build_ssrf_http_client(allow_private, REQUEST_TIMEOUT)?;

        let ingestion_url = format!("{host}/api/public/ingestion");

        let cap = if capacity == 0 {
            DEFAULT_CHANNEL_CAPACITY
        } else {
            capacity
        };
        let (tx, mut rx) = mpsc::channel::<Vec<LangfuseEvent>>(cap);
        let drain_user = resolved.public_key.clone();
        let drain_pass = resolved.secret_key.clone();
        let drain_url = ingestion_url.clone();
        let flush_interval = resolved.flush_interval;
        let max_batch_size = resolved.max_batch_size;
        tokio::spawn(async move {
            // Single drainer with an internal buffer + periodic ticker.
            // The drainer exits when all senders are dropped (rx.recv()
            // returns None), which happens when every Arc<LangfuseHook>
            // is dropped. This eliminates the separate Arc<Self>-holding
            // flush task that previously leaked on hot-reload.
            let mut buffer: Vec<LangfuseEvent> = Vec::new();
            let mut ticker = tokio::time::interval(flush_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // Discard the immediate first tick so the first flush window
            // does not expire at t=0 (tokio::interval fires immediately on
            // the first tick).
            ticker.tick().await;
            loop {
                tokio::select! {
                    biased;
                    maybe_events = rx.recv() => {
                        match maybe_events {
                            Some(events) => {
                                buffer.extend(events);
                                if buffer.len() >= max_batch_size {
                                    let batch = std::mem::take(&mut buffer);
                                    if let Err(e) =
                                        post_batch(&client, &drain_url, &drain_user, &drain_pass, &batch).await
                                    {
                                        let n = batch.len();
                                        tracing::warn!(
                                            error = %e.without_url(),
                                            url = %crate::security::sanitize_url(&drain_url),
                                            events = n,
                                            "langfuse ingestion POST failed"
                                        );
                                        crate::metrics::record_telemetry_dropped(
                                            "langfuse",
                                            "post_failed",
                                            n as u64,
                                        );
                                    }
                                }
                            }
                            None => {
                                // All senders dropped: final flush + exit.
                                if !buffer.is_empty() {
                                    let batch = std::mem::take(&mut buffer);
                                    if let Err(e) = post_batch(
                                        &client,
                                        &drain_url,
                                        &drain_user,
                                        &drain_pass,
                                        &batch,
                                    )
                                    .await
                                    {
                                        let n = batch.len();
                                        tracing::warn!(
                                            error = %e.without_url(),
                                            url = %crate::security::sanitize_url(&drain_url),
                                            events = n,
                                            "langflush ingestion POST failed on shutdown flush"
                                        );
                                        crate::metrics::record_telemetry_dropped(
                                            "langfuse",
                                            "post_failed",
                                            n as u64,
                                        );
                                    }
                                }
                                break;
                            }
                        }
                    }
                    _ = ticker.tick() => {
                        if !buffer.is_empty() {
                            let batch = std::mem::take(&mut buffer);
                            if let Err(e) =
                                post_batch(&client, &drain_url, &drain_user, &drain_pass, &batch).await
                            {
                                let n = batch.len();
                                tracing::warn!(
                                    error = %e.without_url(),
                                    url = %crate::security::sanitize_url(&drain_url),
                                    events = n,
                                    "langfuse ingestion POST failed"
                                );
                                crate::metrics::record_telemetry_dropped(
                                    "langfuse",
                                    "post_failed",
                                    n as u64,
                                );
                            }
                        }
                    }
                }
            }
        });

        let hook = Self {
            config: resolved,
            ingestion_url,
            tx,
        };

        Ok(hook)
    }

    /// The configured ingestion URL (for tests).
    pub fn ingestion_url(&self) -> &str {
        &self.ingestion_url
    }

    /// The configured max batch size (for tests).
    pub fn max_batch_size(&self) -> usize {
        self.config.max_batch_size
    }

    /// The configured flush interval (for tests).
    pub fn flush_interval(&self) -> Duration {
        self.config.flush_interval
    }

    /// Build the `trace-create` + `generation-create` event pair for a single
    /// request, redacting sensitive fields from input/output as configured.
    ///
    /// Event ids (and the generation id that links the gen event to its trace)
    /// are suffixed with a fresh UUID per call. The orchestrator reuses
    /// `request_id` across retry / fallback attempts; Langfuse dedupes
    /// ingestion on event id, so deterministic ids would silently drop every
    /// attempt after the first from a trace. The per-call suffix keeps each
    /// attempt a distinct generation under the same trace (the common reason
    /// operators turn on tracing).
    fn build_events(&self, ctx: &CallContext, resp: &ResponseSummary) -> Vec<LangfuseEvent> {
        let trace_id = ctx.request_id.clone();
        let attempt = uuid::Uuid::new_v4().to_string();
        let generation_id = format!("{trace_id}-gen-{attempt}");
        // MEDIUM #5: derive `startTime` / `endTime` from the wall-clock
        // window stamped on the CallContext so generation latency surfaces
        // correctly in Langfuse dashboards. The previous code set both to
        // `now_iso()` and every generation reported zero duration. Fall
        // back to a single `now()` sample when `start_time` is missing
        // (tests, synthetic contexts) so the fields stay well-formed.
        let end = jiff::Timestamp::now();
        let start = ctx.start_time.and_then(|s| {
            let start_system = std::time::SystemTime::now().checked_sub(s.elapsed())?;
            jiff::Timestamp::try_from(start_system).ok()
        });
        let (start_iso, end_iso) = match start {
            Some(s) => (s.to_string(), end.to_string()),
            None => (end.to_string(), end.to_string()),
        };

        let trace_metadata = json!({
            "key_id": ctx.key_id,
            "team_id": ctx.team_id,
            "user_id": ctx.user_id,
            "request_id": ctx.request_id,
        });

        let trace_event = LangfuseEvent {
            id: format!("{trace_id}-trace-event-{attempt}"),
            event_type: "trace-create",
            body: json!({
                "id": trace_id.clone(),
                "name": ctx.model,
                "metadata": trace_metadata,
            }),
        };

        let input = if self.config.log_inputs {
            // Arc::clone is cheap; redact_sensitive consumes the cloned Value
            // and produces a new (owned) redacted Value.
            let mut v = redact_sensitive((*ctx.request_body).clone());
            if self.config.redact_secret_values {
                redact_secret_values_in_place(&mut v);
            }
            v
        } else {
            Value::Null
        };
        let output = if self.config.log_outputs {
            let mut out = resp.body.clone().unwrap_or(Value::Null);
            redact_sensitive_in_place(&mut out);
            if self.config.redact_secret_values {
                redact_secret_values_in_place(&mut out);
            }
            out
        } else {
            Value::Null
        };

        let cost = fallback_cost(ctx, resp);

        let usage = resp.usage.as_ref().map(|u| {
            json!({
                "promptTokens": u.prompt_tokens,
                "completionTokens": u.completion_tokens,
                "total": u.total_tokens,
            })
        });

        let generation_metadata = json!({
            "cost": cost,
            "provider": resp.provider,
            "key_id": ctx.key_id,
            "team_id": ctx.team_id,
            "request_id": ctx.request_id,
        });

        let mut gen_body = json!({
            "id": generation_id,
            "traceId": trace_id,
            "name": resp.model,
            "model": resp.model,
            "input": input,
            "output": output,
            "metadata": generation_metadata,
            "startTime": start_iso,
            "endTime": end_iso,
        });
        if let Some(u) = usage {
            gen_body["usage"] = u;
        }

        let gen_event = LangfuseEvent {
            id: format!("{trace_id}-gen-event-{attempt}"),
            event_type: "generation-create",
            body: gen_body,
        };

        vec![trace_event, gen_event]
    }
}

#[async_trait]
impl Hook for LangfuseHook {
    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        if ctx.cache_hit {
            return;
        }
        if ctx.is_master {
            return;
        }
        let events = self.build_events(ctx, resp);
        let n_events = events.len();
        match self.tx.try_send(events) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!("langfuse drainer channel full; dropping {n_events} events");
                crate::metrics::record_telemetry_dropped("langfuse", "full", n_events as u64);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!("langfuse drainer channel closed; dropping {n_events} events");
                crate::metrics::record_telemetry_dropped("langfuse", "closed", n_events as u64);
            }
        }
    }
}

async fn post_batch(
    client: &reqwest::Client,
    url: &str,
    public_key: &str,
    secret_key: &str,
    events: &[LangfuseEvent],
) -> Result<(), reqwest::Error> {
    let body = json!({ "batch": events });
    client
        .post(url)
        .basic_auth(public_key, Some(secret_key))
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

/// Build a [`LangfuseHook`] from `oxllm_settings.langfuse_config` and spawn
/// its periodic flush task. Returns `None` when langfuse is not configured or
/// fails SSRF validation (mirroring the previous inline behavior).
pub(crate) fn build_langfuse_hook_opt(
    config: &crate::config::ProxyConfig,
) -> Option<Arc<LangfuseHook>> {
    let lf_cfg = config.oxllm_settings.langfuse_config.as_ref()?;
    match build_langfuse_hook(lf_cfg, &config.general_settings) {
        Ok(hook) => Some(hook),
        Err(e) => {
            tracing::warn!(
                error = %e,
                "failed to build langfuse hook; skipping"
            );
            None
        }
    }
}

/// Construct a [`LangfuseHook`] from `oxllm_settings.langfuse_config`.
/// The periodic flush ticker runs inside the single background drainer task
/// (no separate Arc<Self>-holding task), so dropping the last `Arc<LangfuseHook>`
/// cleanly terminates both the drainer and the ticker.
pub(crate) fn build_langfuse_hook(
    config: &LangfuseConfig,
    general: &crate::config::types::GeneralSettings,
) -> anyhow::Result<Arc<LangfuseHook>> {
    let allow_private = general.allow_private_api_base.unwrap_or(false);
    let redact_secret_values = general.redact_secret_values.unwrap_or(false);
    if let Err(e) = crate::config::validate_api_base(&config.host, allow_private) {
        anyhow::bail!(
            "langfuse_config host '{}' failed SSRF validation: {e}",
            config.host
        );
    }
    let hook = LangfuseHook::new(config, allow_private, redact_secret_values)?;
    Ok(Arc::new(hook))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;

    fn raw_config() -> LangfuseConfig {
        LangfuseConfig {
            host: "https://langfuse.example.com".to_string(),
            public_key: "pk-lf-abc".to_string(),
            secret_key: "sk-lf-xyz".to_string(),
            flush_interval_secs: None,
            max_batch_size: None,
            log_inputs: None,
            log_outputs: None,
        }
    }

    fn make_ctx() -> CallContext {
        CallContext {
            request_body: std::sync::Arc::new(json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "authorization": "Bearer sk-secret"
            })),
            key_id: Some("k-1".to_string()),
            team_id: Some("team-1".to_string()),
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
                browser_open_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: Some(json!({
                "id": "chatcmpl-7",
                "choices": [{"message": {"content": "hi"}}],
            })),
            provider: "openai".to_string(),
        }
    }

    #[test]
    fn parses_from_yaml() {
        let yaml = "\
host: https://cloud.langfuse.com
public_key: pk-test
secret_key: sk-test
flush_interval_secs: 10
max_batch_size: 25
log_inputs: false
log_outputs: true
";
        let cfg: LangfuseConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.host, "https://cloud.langfuse.com");
        assert_eq!(cfg.public_key, "pk-test");
        assert_eq!(cfg.secret_key, "sk-test");
        assert_eq!(cfg.flush_interval_secs, Some(10));
        assert_eq!(cfg.max_batch_size, Some(25));
        assert_eq!(cfg.log_inputs, Some(false));
        assert_eq!(cfg.log_outputs, Some(true));
    }

    #[tokio::test]
    async fn construction_applies_defaults() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        assert_eq!(
            hook.ingestion_url(),
            "https://langfuse.example.com/api/public/ingestion"
        );
        assert_eq!(hook.max_batch_size(), 50);
        assert_eq!(hook.flush_interval(), Duration::from_secs(5));
    }

    #[test]
    fn construction_rejects_empty_host() {
        let cfg = LangfuseConfig {
            host: "".to_string(),
            ..raw_config()
        };
        assert!(LangfuseHook::new(&cfg, false, false).is_err());
    }

    #[test]
    fn construction_rejects_missing_keys() {
        let cfg = LangfuseConfig {
            public_key: "".to_string(),
            ..raw_config()
        };
        assert!(LangfuseHook::new(&cfg, false, false).is_err());
    }

    #[tokio::test]
    async fn construction_trims_trailing_slash() {
        let cfg = LangfuseConfig {
            host: "https://langfuse.example.com/".to_string(),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");
        assert_eq!(
            hook.ingestion_url(),
            "https://langfuse.example.com/api/public/ingestion"
        );
    }

    #[tokio::test]
    async fn build_events_emits_trace_and_generation() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let events = hook.build_events(&make_ctx(), &make_response());
        assert_eq!(events.len(), 2, "expected trace-create + generation-create");
        assert_eq!(events[0].event_type, "trace-create");
        assert_eq!(events[1].event_type, "generation-create");
    }

    #[tokio::test]
    async fn trace_event_has_correct_shape() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let events = hook.build_events(&make_ctx(), &make_response());
        let trace = &events[0];
        assert_eq!(trace.body["id"], "req-1");
        assert_eq!(trace.body["name"], "gpt-4o");
        assert_eq!(trace.body["metadata"]["key_id"], "k-1");
        assert_eq!(trace.body["metadata"]["team_id"], "team-1");
        assert_eq!(trace.body["metadata"]["request_id"], "req-1");
    }

    #[tokio::test]
    async fn generation_event_has_correct_shape() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let events = hook.build_events(&make_ctx(), &make_response());
        let gen_event = &events[1];
        let gen_id = gen_event.body["id"].as_str().expect("id is string");
        assert!(
            gen_id.starts_with("req-1-gen-"),
            "generation id must be suffixed with a per-attempt uuid: {gen_id}"
        );
        assert_eq!(gen_event.body["traceId"], "req-1");
        assert_eq!(gen_event.body["model"], "gpt-4o");
        assert_eq!(gen_event.body["name"], "gpt-4o");
        assert_eq!(gen_event.body["usage"]["promptTokens"], 100);
        assert_eq!(gen_event.body["usage"]["completionTokens"], 50);
        assert_eq!(gen_event.body["usage"]["total"], 150);
        assert_eq!(gen_event.body["metadata"]["provider"], "openai");
        assert_eq!(gen_event.body["metadata"]["cost"], 0.0025);
        assert!(gen_event.body["startTime"].is_string());
        assert!(gen_event.body["endTime"].is_string());
    }

    #[tokio::test]
    async fn generation_input_includes_request_body_by_default() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let events = hook.build_events(&make_ctx(), &make_response());
        let input = &events[1].body["input"];
        assert_eq!(input["model"], "gpt-4o");
    }

    #[tokio::test]
    async fn generation_input_redacts_authorization_header() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let events = hook.build_events(&make_ctx(), &make_response());
        let input = &events[1].body["input"];
        assert_eq!(input["authorization"], "[redacted]");
    }

    #[tokio::test]
    async fn log_inputs_false_nulls_input() {
        let cfg = LangfuseConfig {
            log_inputs: Some(false),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");
        let events = hook.build_events(&make_ctx(), &make_response());
        assert!(events[1].body["input"].is_null());
    }

    #[tokio::test]
    async fn log_outputs_false_nulls_output() {
        let cfg = LangfuseConfig {
            log_outputs: Some(false),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");
        let events = hook.build_events(&make_ctx(), &make_response());
        assert!(events[1].body["output"].is_null());
    }

    #[test]
    fn redacts_x_api_key_case_insensitive() {
        let value = json!({
            "X-API-KEY": "sk-secret",
            "nested": {"Api-Key": "sk-inner"},
        });
        let redacted = redact_sensitive(value);
        assert_eq!(redacted["X-API-KEY"], "[redacted]");
        assert_eq!(redacted["nested"]["Api-Key"], "[redacted]");
    }

    #[tokio::test]
    async fn output_redacts_authorization_header() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let ctx = make_ctx();
        let mut resp = make_response();
        resp.body = Some(json!({
            "choices": [{"message": {"content": "hi"}}],
            "authorization": "Bearer sk-leak"
        }));
        let events = hook.build_events(&ctx, &resp);
        let output = &events[1].body["output"];
        assert_eq!(output["authorization"], "[redacted]");
    }

    #[tokio::test]
    async fn output_redacts_nested_api_key() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let ctx = make_ctx();
        let mut resp = make_response();
        resp.body = Some(json!({
            "data": [{"api_key": "sk-secret"}]
        }));
        let events = hook.build_events(&ctx, &resp);
        let output = &events[1].body["output"];
        assert_eq!(output["data"][0]["api_key"], "[redacted]");
    }

    /// LOW #11: the orchestrator reuses `request_id` across retry / fallback
    /// attempts. Langfuse dedupes ingestion on event id, so two attempts
    /// with the same trace id must still produce distinct event ids (and a
    /// distinct generation id linking the gen event to its trace) or the
    /// later attempt would be silently dropped.
    #[tokio::test]
    async fn event_ids_are_distinct_across_attempts() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let ctx = make_ctx();
        let resp = make_response();
        let first = hook.build_events(&ctx, &resp);
        let second = hook.build_events(&ctx, &resp);
        let first_ids: std::collections::HashSet<&str> =
            first.iter().map(|e| e.id.as_str()).collect();
        let second_ids: std::collections::HashSet<&str> =
            second.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            first_ids.intersection(&second_ids).count(),
            0,
            "event ids must differ across attempts for the same request_id"
        );
        let first_gen = first[1].body["id"].as_str().expect("gen id");
        let second_gen = second[1].body["id"].as_str().expect("gen id");
        assert_ne!(first_gen, second_gen);
        assert_eq!(first[1].body["traceId"], "req-1");
        assert_eq!(second[1].body["traceId"], "req-1");
    }

    /// MEDIUM #5: when `CallContext::start_time` is set, the generation's
    /// `startTime` must be at least that far in the past and `endTime` must
    /// be the current "now", so generation latency surfaces correctly in
    /// Langfuse dashboards. The previous code set both fields to the same
    /// `now_iso()` and every generation reported zero duration. The
    /// divergence (`endTime - startTime`) must be ≥ the sleep window.
    #[tokio::test]
    async fn generation_endtime_is_later_than_starttime_when_start_time_set() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let mut ctx = make_ctx();
        ctx.start_time = Some(std::time::Instant::now());
        // Sleep long enough that the elapsed between start_time and the
        // build_events call is non-trivial (well above timestamp rounding).
        tokio::time::sleep(Duration::from_millis(250)).await;
        let events = hook.build_events(&ctx, &make_response());
        let generation = &events[1];
        let start_str = generation.body["startTime"]
            .as_str()
            .expect("startTime is string");
        let end_str = generation.body["endTime"]
            .as_str()
            .expect("endTime is string");
        let start_ts: jiff::Timestamp = start_str.parse().expect("startTime parses");
        let end_ts: jiff::Timestamp = end_str.parse().expect("endTime parses");
        let duration_ms = end_ts
            .since(start_ts)
            .expect("end ≥ start")
            .get_milliseconds();
        assert!(
            duration_ms >= 200,
            "endTime - startTime must be ≥ the 250ms sleep window, got {duration_ms}ms"
        );
    }

    /// MEDIUM #5: when no `start_time` is available (synthetic contexts,
    /// tests), `startTime` and `endTime` fall back to the same `now()` so
    /// the fields stay well-formed (matches the legacy behaviour rather
    /// than emitting `null`).
    #[tokio::test]
    async fn generation_starttime_equals_endtime_when_start_time_absent() {
        let hook = LangfuseHook::new(&raw_config(), false, false).expect("build");
        let ctx = make_ctx();
        assert!(ctx.start_time.is_none(), "test ctx has no start_time");
        let events = hook.build_events(&ctx, &make_response());
        let generation = &events[1];
        // Without a start_time we cannot meaningfully order the two, but
        // they must both be present and parse as ISO timestamps.
        assert!(generation.body["startTime"].is_string());
        assert!(generation.body["endTime"].is_string());
    }

    /// With the new design, events are sent directly to the drainer's
    /// internal buffer. When below `max_batch_size`, no POST should fire.
    #[tokio::test]
    async fn buffer_batches_events_until_flush() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = LangfuseConfig {
            host: server.base_url(),
            max_batch_size: Some(50),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion");
            then.status(200);
        });

        // One post_success produces 2 events; below max_batch_size (50) so
        // nothing should be flushed immediately (the ticker has not fired
        // yet either).
        hook.post_success(&make_ctx(), &make_response()).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            post_mock.hits(),
            0,
            "events should be buffered, not flushed"
        );
    }

    #[tokio::test]
    async fn flushes_when_buffer_reaches_max_batch_size() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = LangfuseConfig {
            host: server.base_url(),
            max_batch_size: Some(2),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion");
            then.status(200).json_body(json!({"success": true}));
        });

        // max_batch_size == 2 → a single post_success (2 events) triggers a flush.
        hook.post_success(&make_ctx(), &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "buffer full should trigger an immediate POST"
        );
    }

    #[tokio::test]
    async fn post_success_skips_cache_hits() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = LangfuseConfig {
            host: server.base_url(),
            max_batch_size: Some(2),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion");
            then.status(200);
        });

        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        hook.post_success(&ctx, &make_response()).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(post_mock.hits(), 0, "cache hits must not be logged");
    }

    #[tokio::test]
    async fn post_success_skips_master_key() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = LangfuseConfig {
            host: server.base_url(),
            max_batch_size: Some(2),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion");
            then.status(200);
        });

        let mut ctx = make_ctx();
        ctx.is_master = true;
        hook.post_success(&ctx, &make_response()).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            post_mock.hits(),
            0,
            "master-key requests must not be logged"
        );
    }

    #[tokio::test]
    async fn ingestion_uses_basic_auth() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = LangfuseConfig {
            host: server.base_url(),
            max_batch_size: Some(2),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");

        // base64("pk-lf-abc:sk-lf-xyz") = "cGstbGYtYWJjOnNrLWxmLXh5eg=="
        let expected_auth = "Basic cGstbGYtYWJjOnNrLWxmLXh5eg==";
        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion")
                .header("Authorization", expected_auth);
            then.status(200);
        });

        hook.post_success(&make_ctx(), &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(post_mock.hits(), 1, "Basic auth header must be sent");
    }

    #[tokio::test]
    async fn ingestion_batch_contains_correct_envelope() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = LangfuseConfig {
            host: server.base_url(),
            max_batch_size: Some(2),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion")
                .body_contains("\"type\":\"trace-create\"")
                .body_contains("\"type\":\"generation-create\"");
            then.status(200);
        });

        hook.post_success(&make_ctx(), &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "batch should contain both event types"
        );
    }

    #[tokio::test]
    async fn post_success_does_not_block_request_on_post_failure() {
        let cfg = LangfuseConfig {
            host: "http://127.0.0.1:1".to_string(),
            max_batch_size: Some(2),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");
        let started = std::time::Instant::now();
        hook.post_success(&make_ctx(), &make_response()).await;
        assert!(
            started.elapsed() < Duration::from_millis(150),
            "post_success must return immediately even when endpoint is unreachable"
        );
    }

    /// LOW #8: a failed ingestion POST must increment
    /// `oxllm_telemetry_dropped_total{backend="langfuse",reason="post_failed"}`
    /// by the batch size, so the silent loss is observable (mirroring the
    /// spend-writer drop accounting).
    #[tokio::test]
    async fn flush_failure_increments_drop_counter() {
        use httpmock::MockServer;

        let _ = crate::metrics::install();

        let server = MockServer::start();
        let cfg = LangfuseConfig {
            host: server.base_url(),
            max_batch_size: Some(2),
            ..raw_config()
        };
        let hook = LangfuseHook::new(&cfg, false, false).expect("build");

        // Force every ingestion POST to fail.
        let _post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion");
            then.status(500);
        });

        // One post_success produces 2 events → triggers a flush (max_batch=2).
        hook.post_success(&make_ctx(), &make_response()).await;

        // The drop counter is bumped inside the spawned task; poll for it.
        let mut dropped = 0u64;
        for _ in 0..40 {
            dropped = telemetry_dropped_value("langfuse", "post_failed");
            if dropped >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            dropped >= 2,
            "expected >=2 langfuse events counted as dropped on POST failure, got {dropped}"
        );
    }

    /// HIGH #2: under sustained load against a slow Langfuse backend, the
    /// number of concurrently in-flight ingestion POSTs must be bounded by
    /// the single-drainer design (one POST at a time per hook instance). The
    /// legacy `tokio::spawn`-per-flush path could fan out an unbounded number
    /// of concurrent POSTs and never reported a "full" drop. This test pins
    /// the new behaviour by exercising a tiny-capacity drainer against a slow
    /// backend: the channel saturates, overflow is dropped and counted under
    /// `oxllm_telemetry_dropped_total{backend="langfuse",reason="full"}`,
    /// and `post_success` never blocks.
    #[tokio::test]
    async fn flush_drainer_bounds_in_flight_posts_via_single_drainer() {
        use httpmock::MockServer;

        let _ = crate::metrics::install();

        let server = MockServer::start();
        // Each POST delays so the single drainer cannot keep up with a burst
        // against a capacity-2 channel, forcing overflow.
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/public/ingestion");
            then.delay(Duration::from_millis(100))
                .status(200)
                .json_body(json!({"success": true}));
        });

        let cfg = LangfuseConfig {
            host: server.base_url(),
            // Tiny batch size: every post_success (2 events) triggers a flush.
            max_batch_size: Some(2),
            ..raw_config()
        };
        // Tiny drainer capacity: the drainer fills immediately and overflow
        // is dropped + counted.
        let hook = LangfuseHook::with_capacity(&cfg, 2, false, false).expect("build");

        let ctx = make_ctx();
        let resp = make_response();
        // 100 post_success calls → 100 flushes; channel capacity is 2 and
        // each drain takes ≥100ms, so the overwhelming majority must be
        // dropped on the floor (counted) rather than spawned as concurrent
        // tasks. Under the legacy spawn-per-flush design this would have
        // produced ~100 in-flight POSTs and zero "full" drops.
        let n: u64 = 100;
        let started = std::time::Instant::now();
        for _ in 0..n {
            hook.post_success(&ctx, &resp).await;
        }
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "post_success burst must not block, took {:?}",
            started.elapsed()
        );

        // The drop counter for "full" must be non-zero — proof that the
        // drainer channel saturated and overflow was shed by the bounded
        // channel rather than fanned out via tokio::spawn.
        let mut dropped = 0u64;
        for _ in 0..80 {
            dropped = telemetry_dropped_value("langfuse", "full");
            if dropped > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            dropped > 0,
            "bounded drainer must drop+count overflow under sustained load; the legacy spawn-per-flush path would not"
        );
    }

    /// MEDIUM #4: dropping the last `Arc<LangfuseHook>` must terminate the
    /// background drainer (and its periodic flush ticker) cleanly. The
    /// previous design held an `Arc<Self>` in a separate flush task whose
    /// `JoinHandle` was immediately dropped, so neither the flush loop nor
    /// the ingestion drainer could ever exit — leaking permanently on every
    /// hot-reload. The new merged-drainer design exits when `rx.recv()`
    /// returns `None` (all senders dropped). We verify by dropping the hook
    /// and asserting the `Arc` strong count falls to zero within a bounded
    /// window (proving the drainer task released its reference).
    #[tokio::test]
    async fn dropping_hook_terminates_drainer() {
        let cfg = LangfuseConfig {
            max_batch_size: Some(2),
            flush_interval_secs: Some(1),
            ..raw_config()
        };
        let hook = Arc::new(LangfuseHook::new(&cfg, false, false).expect("build"));
        let weak = Arc::downgrade(&hook);
        drop(hook);
        // The drainer task exits when rx.recv() returns None. Give it a
        // bounded window to wind down.
        let mut ok = false;
        for _ in 0..40 {
            if weak.strong_count() == 0 {
                ok = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            ok,
            "drainer task must release its Arc<LangfuseHook> after all senders are dropped"
        );
    }

    /// Parse `oxllm_telemetry_dropped_total{backend="...",reason="..."}` out
    /// of the rendered exposition text. Mirrors the spend_writer test helper.
    fn telemetry_dropped_value(backend: &str, reason: &str) -> u64 {
        let needle = format!("backend=\"{backend}\",reason=\"{reason}\"");
        for line in crate::metrics::render().lines() {
            if line.starts_with("oxllm_telemetry_dropped_total{") && line.contains(&needle) {
                if let Some(n) = line.rsplit(' ').next() {
                    if let Ok(v) = n.parse::<u64>() {
                        return v;
                    }
                }
            }
        }
        0
    }
}
