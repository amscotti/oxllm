//! Datadog logging hook.
//!
//! Built on [`HttpTelemetryHook`]: each successful (non-cache, non-master)
//! request is POSTed as a Datadog log entry to the logs intake API
//! (`POST {endpoint}/api/v2/logs`) under a `DD-API-KEY` header. The POST is
//! fire-and-forget. Wired in from `oxllm_settings.datadog_config`.

use std::sync::Arc;

use oxllm_core::{CallContext, ResponseSummary};
use serde_json::{Value, json};

use crate::config::types::DatadogConfig;
use crate::hooks::cost_fallback::fallback_cost;
use crate::hooks::http_telemetry::{HttpTelemetryHook, resolve_endpoint};

const DEFAULT_ENDPOINT: &str = "https://http-intake.logs.datadoghq.com";
const DEFAULT_SERVICE: &str = "oxllm";

/// Build a Datadog telemetry hook from its config block.
///
/// Validates the API key, resolves the endpoint (defaulting to Datadog logs
/// intake) and service name (defaulting to `oxllm`), and constructs the intake
/// URL. The service name is captured into the payload-builder closure.
pub fn build(
    config: &DatadogConfig,
    allow_private: bool,
    redact_secret_values: bool,
) -> anyhow::Result<HttpTelemetryHook> {
    if config.api_key.trim().is_empty() {
        anyhow::bail!("datadog_config.api_key must be set");
    }
    let endpoint = resolve_endpoint(config.endpoint.as_deref(), DEFAULT_ENDPOINT);
    let service = config
        .service
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| DEFAULT_SERVICE.to_string());
    let url = format!("{endpoint}/api/v2/logs");
    HttpTelemetryHook::new_full(
        url,
        Some(("DD-API-KEY".to_string(), config.api_key.clone())),
        move |ctx, resp| json!([datadog_log(&service, ctx, resp)]),
        "datadog",
        crate::hooks::http_telemetry::DEFAULT_CHANNEL_CAPACITY,
        allow_private,
        redact_secret_values,
    )
}

/// Build a Datadog hook from `oxllm_settings.datadog_config` when present.
/// Returns `None` when datadog is not configured or fails SSRF validation
/// (misconfiguration is logged at warn level).
pub(crate) fn build_datadog_hook_opt(
    config: &crate::config::ProxyConfig,
) -> Option<Arc<HttpTelemetryHook>> {
    let cfg = config.oxllm_settings.datadog_config.as_ref()?;
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
            "datadog_config endpoint '{endpoint}' failed SSRF validation; skipping"
        );
        return None;
    }
    match build(cfg, allow_private, redact_secret_values) {
        Ok(hook) => Some(Arc::new(hook)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build datadog hook; skipping");
            None
        }
    }
}

/// Build the Datadog log entry for a single request. Free function so the
/// payload-shape test can exercise it directly; the wire body wraps it in a
/// JSON array (`[log]`) as the Datadog intake expects.
fn datadog_log(service: &str, ctx: &CallContext, resp: &ResponseSummary) -> Value {
    let usage = resp.usage.as_ref();
    let prompt_tokens = usage.map(|u| u.prompt_tokens);
    let completion_tokens = usage.map(|u| u.completion_tokens);
    let total_tokens = usage.map(|u| u.total_tokens);

    let cost = fallback_cost(ctx, resp);

    json!({
        "ddsource": "oxllm",
        "service": service,
        "status": "info",
        "message": "LLM request completed",
        "hostname": service,
        "attributes": {
            "model": resp.model,
            "provider": resp.provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "key_id": ctx.key_id,
            "team_id": ctx.team_id,
            "user_id": ctx.user_id,
            "request_id": ctx.request_id,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;
    use std::time::Duration;

    fn raw_config() -> DatadogConfig {
        DatadogConfig {
            api_key: "dd-key".to_string(),
            service: Some("my-svc".to_string()),
            endpoint: Some("https://datadog.example.com".to_string()),
        }
    }

    fn make_ctx() -> CallContext {
        CallContext {
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
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[test]
    fn parses_from_yaml() {
        let yaml = "\
api_key: dd-key
service: my-svc
endpoint: https://datadog.example.com
";
        let cfg: DatadogConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.api_key, "dd-key");
        assert_eq!(cfg.service.as_deref(), Some("my-svc"));
        assert_eq!(cfg.endpoint.as_deref(), Some("https://datadog.example.com"));
    }

    #[tokio::test]
    async fn construction_applies_default_endpoint_and_service() {
        let cfg = DatadogConfig {
            api_key: "dd-key".to_string(),
            service: None,
            endpoint: None,
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(
            hook.url(),
            "https://http-intake.logs.datadoghq.com/api/v2/logs"
        );
    }

    #[tokio::test]
    async fn construction_trims_trailing_slash() {
        let cfg = DatadogConfig {
            endpoint: Some("https://datadog.example.com/".to_string()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://datadog.example.com/api/v2/logs");
    }

    #[test]
    fn construction_rejects_empty_api_key() {
        let cfg = DatadogConfig {
            api_key: "".to_string(),
            ..raw_config()
        };
        assert!(build(&cfg, false, false).is_err());
    }

    /// LOW #10: an empty/whitespace `endpoint` must fall back to the default
    /// Datadog logs intake rather than producing a path-only URL.
    #[tokio::test]
    async fn construction_applies_default_when_endpoint_empty() {
        let cfg = DatadogConfig {
            api_key: "dd-key".to_string(),
            service: None,
            endpoint: Some("   ".to_string()),
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(
            hook.url(),
            "https://http-intake.logs.datadoghq.com/api/v2/logs"
        );
    }

    #[test]
    fn build_log_has_correct_shape() {
        let log = datadog_log("my-svc", &make_ctx(), &make_response());
        assert_eq!(log["service"], "my-svc");
        assert_eq!(log["status"], "info");
        assert_eq!(log["message"], "LLM request completed");
        assert_eq!(log["attributes"]["model"], "gpt-4o");
        assert_eq!(log["attributes"]["provider"], "openai");
        assert_eq!(log["attributes"]["prompt_tokens"], 100);
        assert_eq!(log["attributes"]["completion_tokens"], 50);
        assert_eq!(log["attributes"]["total_tokens"], 150);
        assert_eq!(log["attributes"]["cost"], 0.0025);
        assert_eq!(log["attributes"]["key_id"], "k-1");
        assert_eq!(log["attributes"]["team_id"], "team-1");
    }

    #[test]
    fn build_log_applies_default_service_when_omitted() {
        let log = datadog_log("oxllm", &make_ctx(), &make_response());
        assert_eq!(log["service"], "oxllm");
    }

    /// LOW #9: `hostname` must be a stable host-level value (the service
    /// name), not the per-request `ctx.request_id`. Stuffing a unique UUID
    /// into `hostname` defeats Datadog's host-based correlation and risks
    /// hitting custom-metric/host cardinality caps.
    #[test]
    fn build_log_uses_stable_hostname_not_request_id() {
        let log = datadog_log("my-svc", &make_ctx(), &make_response());
        assert_eq!(
            log["hostname"], "my-svc",
            "hostname must be the stable service name, not the per-request id"
        );
        assert_ne!(log["hostname"], "req-1");
    }

    #[tokio::test]
    async fn logs_intake_posts_with_api_key_header() {
        use httpmock::MockServer;
        use oxllm_core::Hook;

        let server = MockServer::start();
        let cfg = DatadogConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/api/v2/logs")
                .header("DD-API-KEY", "dd-key")
                .body_contains("\"service\":\"my-svc\"")
                .body_contains("\"message\":\"LLM request completed\"");
            then.status(202);
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
            "datadog should have POSTed a log with the api key"
        );
    }
}
