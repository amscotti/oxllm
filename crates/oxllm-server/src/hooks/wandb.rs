//! Weights & Biases spend-logging hook.
//!
//! Built on [`HttpTelemetryHook`]: each successful (non-cache, non-master)
//! request's spend data (model, tokens, cost, latency) is POSTed to the W&B
//! REST API (`POST {endpoint}/v1/oxllm/spend`) under an
//! `Authorization: Bearer {api_key}` header. The POST is fire-and-forget.
//! Wired in from `oxllm_settings.wandb_config`.

use std::sync::Arc;

use oxllm_core::{CallContext, ResponseSummary};
use serde_json::{Value, json};

use crate::config::types::WandbConfig;
use crate::hooks::cost_fallback::fallback_cost;
use crate::hooks::http_telemetry::{HttpTelemetryHook, resolve_endpoint};

const DEFAULT_ENDPOINT: &str = "https://api.wandb.ai";

/// Build a W&B telemetry hook from its config block.
///
/// Validates the API key and project, resolves the endpoint (defaulting to
/// W&B cloud), and constructs the spend URL. The project, entity, and tags are
/// captured into the payload-builder closure.
pub fn build(
    config: &WandbConfig,
    allow_private: bool,
    redact_secret_values: bool,
) -> anyhow::Result<HttpTelemetryHook> {
    if config.api_key.trim().is_empty() {
        anyhow::bail!("wandb_config.api_key must be set");
    }
    if config.project.trim().is_empty() {
        anyhow::bail!("wandb_config.project must be set");
    }
    let endpoint = resolve_endpoint(config.endpoint.as_deref(), DEFAULT_ENDPOINT);
    let url = format!("{endpoint}/v1/oxllm/spend");
    let project = config.project.clone();
    let entity = config.entity.clone();
    let tags = config.tags.clone();
    HttpTelemetryHook::new_full(
        url,
        Some((
            "Authorization".to_string(),
            format!("Bearer {}", config.api_key),
        )),
        move |ctx, resp| wandb_payload(&project, &entity, &tags, ctx, resp),
        "wandb",
        crate::hooks::http_telemetry::DEFAULT_CHANNEL_CAPACITY,
        allow_private,
        redact_secret_values,
    )
}

/// Build a W&B hook from `oxllm_settings.wandb_config` when present.
/// Returns `None` when W&B is not configured or fails SSRF validation
/// (misconfiguration is logged at warn level).
pub(crate) fn build_wandb_hook_opt(
    config: &crate::config::ProxyConfig,
) -> Option<Arc<HttpTelemetryHook>> {
    let cfg = config.oxllm_settings.wandb_config.as_ref()?;
    let endpoint = resolve_endpoint(cfg.endpoint.as_deref(), DEFAULT_ENDPOINT);
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    let redact_secret_values = config
        .general_settings
        .redact_secret_values
        .unwrap_or(false);
    if let Err(e) = crate::config::validate_api_base(&endpoint, allow_private) {
        tracing::warn!(
            error = %e,
            "wandb_config endpoint '{endpoint}' failed SSRF validation; skipping"
        );
        return None;
    }
    match build(cfg, allow_private, redact_secret_values) {
        Ok(hook) => Some(Arc::new(hook)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build wandb hook; skipping");
            None
        }
    }
}

/// Build the W&B spend payload for a single request. Free function so the
/// payload-shape test can exercise it directly; the wire body is the payload
/// itself (no envelope).
fn wandb_payload(
    project: &str,
    entity: &Option<String>,
    tags: &[String],
    ctx: &CallContext,
    resp: &ResponseSummary,
) -> Value {
    let usage = resp.usage.as_ref();
    let cost = fallback_cost(ctx, resp);

    let usage_json = usage.map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    });

    // MEDIUM #5: derive `latency_ms` from the wall-clock window stamped on
    // the CallContext so the W&B spend payload carries an actual duration
    // (the previous code hardcoded `null`). `None` when no `start_time` is
    // available (tests, synthetic contexts).
    let latency_ms = ctx.start_time.map(|s| s.elapsed().as_millis() as u64);

    json!({
        "project": project,
        "entity": entity,
        "tags": tags,
        "model": resp.model,
        "provider": resp.provider,
        "usage": usage_json,
        "cost": cost,
        "latency_ms": latency_ms,
        "request_id": ctx.request_id,
        "key_id": ctx.key_id,
        "team_id": ctx.team_id,
        "user_id": ctx.user_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;
    use std::sync::Arc;
    use std::time::Duration;

    fn raw_config() -> WandbConfig {
        WandbConfig {
            api_key: "wb-key".to_string(),
            project: "oxllm".to_string(),
            entity: Some("my-team".to_string()),
            tags: vec!["prod".to_string()],
            endpoint: Some("https://wandb.example.com".to_string()),
        }
    }

    fn make_ctx() -> CallContext {
        CallContext {
            request_body: Arc::new(json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
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
                cache_creation_1h_input_tokens: None,
            }),
            body: Some(json!({
                "choices": [{"message": {"content": "hi"}}],
            })),
            provider: "openai".to_string(),
        }
    }

    #[test]
    fn parses_from_yaml() {
        let yaml = "\
api_key: wb-key
project: oxllm
entity: my-team
tags: [prod]
endpoint: https://wandb.example.com
";
        let cfg: WandbConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.api_key, "wb-key");
        assert_eq!(cfg.project, "oxllm");
        assert_eq!(cfg.entity.as_deref(), Some("my-team"));
        assert_eq!(cfg.tags, vec!["prod"]);
    }

    #[tokio::test]
    async fn construction_applies_default_endpoint() {
        let cfg = WandbConfig {
            endpoint: None,
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://api.wandb.ai/v1/oxllm/spend");
    }

    #[tokio::test]
    async fn construction_trims_trailing_slash() {
        let cfg = WandbConfig {
            endpoint: Some("https://wandb.example.com/".to_string()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://wandb.example.com/v1/oxllm/spend");
    }

    #[test]
    fn construction_rejects_empty_api_key() {
        let cfg = WandbConfig {
            api_key: "".to_string(),
            ..raw_config()
        };
        assert!(build(&cfg, false, false).is_err());
    }

    #[test]
    fn construction_rejects_empty_project() {
        let cfg = WandbConfig {
            project: "".to_string(),
            ..raw_config()
        };
        assert!(build(&cfg, false, false).is_err());
    }

    /// LOW #10: an empty/whitespace `endpoint` must fall back to the default
    /// W&B endpoint rather than producing a path-only URL.
    #[tokio::test]
    async fn construction_applies_default_when_endpoint_empty() {
        let cfg = WandbConfig {
            endpoint: Some("   ".to_string()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://api.wandb.ai/v1/oxllm/spend");
    }

    #[test]
    fn build_log_has_correct_shape() {
        let payload = wandb_payload(
            "oxllm",
            &Some("my-team".to_string()),
            &["prod".to_string()],
            &make_ctx(),
            &make_response(),
        );
        assert_eq!(payload["project"], "oxllm");
        assert_eq!(payload["entity"], "my-team");
        assert_eq!(payload["tags"][0], "prod");
        assert_eq!(payload["model"], "gpt-4o");
        assert_eq!(payload["provider"], "openai");
        assert_eq!(payload["usage"]["prompt_tokens"], 100);
        assert_eq!(payload["usage"]["completion_tokens"], 50);
        assert_eq!(payload["usage"]["total_tokens"], 150);
        assert_eq!(payload["cost"], 0.0025);
        assert_eq!(payload["request_id"], "req-1");
        assert_eq!(payload["key_id"], "k-1");
        assert_eq!(payload["team_id"], "team-1");
    }

    #[test]
    fn build_log_computes_cost_when_response_omits_it() {
        let resp = ResponseSummary {
            cost: None,
            ..make_response()
        };
        let payload = wandb_payload("oxllm", &None, &[], &make_ctx(), &resp);
        assert!(
            payload["cost"].as_f64().unwrap_or(-1.0) > 0.0,
            "cost should be derived"
        );
    }

    /// MEDIUM #5: when `CallContext::start_time` is set, the W&B payload's
    /// `latency_ms` must reflect the wall-clock elapsed since start (the
    /// previous code hardcoded `null`). Sleep long enough that the elapsed
    /// is unambiguous, then assert `latency_ms` is non-null and ≥ the
    /// sleep window.
    #[test]
    fn build_log_carries_latency_ms_when_start_time_set() {
        let mut ctx = make_ctx();
        ctx.start_time = Some(std::time::Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(120));
        let payload = wandb_payload("oxllm", &None, &[], &ctx, &make_response());
        let latency_ms = payload["latency_ms"]
            .as_u64()
            .expect("latency_ms is a number");
        assert!(
            latency_ms >= 100,
            "latency_ms must be ≥ the 120ms sleep window, got {latency_ms}"
        );
    }

    /// MEDIUM #5: when no `start_time` is available (synthetic contexts),
    /// `latency_ms` remains `null` rather than synthesising a value.
    #[test]
    fn build_log_latency_ms_is_null_when_start_time_absent() {
        let ctx = make_ctx();
        assert!(ctx.start_time.is_none(), "test ctx has no start_time");
        let payload = wandb_payload("oxllm", &None, &[], &ctx, &make_response());
        assert!(
            payload["latency_ms"].is_null(),
            "latency_ms must be null when start_time is not set"
        );
    }

    #[tokio::test]
    async fn log_posts_with_auth_header() {
        use httpmock::MockServer;
        use oxllm_core::Hook;

        let server = MockServer::start();
        let cfg = WandbConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/oxllm/spend")
                .header("Authorization", "Bearer wb-key")
                .body_contains("\"model\":\"gpt-4o\"")
                .body_contains("\"prompt_tokens\":100")
                .body_contains("\"project\":\"oxllm\"");
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
            "wandb should have POSTed a log with the auth header"
        );
    }
}
