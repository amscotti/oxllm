//! Helicone logging hook.
//!
//! Built on [`HttpTelemetryHook`]: each successful (non-cache, non-master)
//! request is POSTed to the Helicone request logging API
//! (`POST {endpoint}/v1/log/request`) under a `Helicone-Auth` header. The POST
//! is fire-and-forget. Wired in from `oxllm_settings.helicone_config`.

use std::sync::Arc;

use oxllm_core::{CallContext, ResponseSummary};
use serde_json::{Value, json};

use crate::config::types::HeliconeConfig;
use crate::hooks::cost_fallback::fallback_cost;
use crate::hooks::http_telemetry::{HttpTelemetryHook, resolve_endpoint};

const DEFAULT_ENDPOINT: &str = "https://api.helicone.ai";

/// Build a Helicone telemetry hook from its config block.
///
/// Validates the API key, resolves the endpoint (defaulting to Helicone cloud),
/// and constructs the ingestion URL. The payload builder is the plain
/// `helicone_payload` function — Helicone's payload needs no config fields.
pub fn build(
    config: &HeliconeConfig,
    allow_private: bool,
    redact_secret_values: bool,
) -> anyhow::Result<HttpTelemetryHook> {
    if config.api_key.trim().is_empty() {
        anyhow::bail!("helicone_config.api_key must be set");
    }
    let endpoint = resolve_endpoint(config.endpoint.as_deref(), DEFAULT_ENDPOINT);
    let url = format!("{endpoint}/v1/log/request");
    HttpTelemetryHook::new_full(
        url,
        Some(("Helicone-Auth".to_string(), config.api_key.clone())),
        helicone_payload,
        "helicone",
        crate::hooks::http_telemetry::DEFAULT_CHANNEL_CAPACITY,
        allow_private,
        redact_secret_values,
    )
}

/// Build a Helicone hook from `oxllm_settings.helicone_config` when present.
/// Returns `None` when helicone is not configured or fails SSRF validation
/// (misconfiguration is logged at warn level).
pub(crate) fn build_helicone_hook_opt(
    config: &crate::config::ProxyConfig,
) -> Option<Arc<HttpTelemetryHook>> {
    let cfg = config.oxllm_settings.helicone_config.as_ref()?;
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
            "helicone_config endpoint '{endpoint}' failed SSRF validation; skipping"
        );
        return None;
    }
    match build(cfg, allow_private, redact_secret_values) {
        Ok(hook) => Some(Arc::new(hook)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build helicone hook; skipping");
            None
        }
    }
}

/// Build the Helicone log payload for a single request. Free function so the
/// payload-shape test can exercise it without constructing a hook.
fn helicone_payload(ctx: &CallContext, resp: &ResponseSummary) -> Value {
    let usage = resp.usage.as_ref();
    let cost = fallback_cost(ctx, resp);

    let messages = ctx
        .request_body
        .get("messages")
        .cloned()
        .unwrap_or(Value::Null);

    let usage_json = usage.map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    });

    // MEDIUM #3: Helicone's ingestion API expects usage at the top level of
    // the payload only; embedding it inside `response.usage` is redundant and
    // risks double-counting if Helicone sums both. The reference helicone helper
    // sends usage only at the top level. Keep `response` as just the raw
    // upstream body — never synthesise a `usage` field inside it.
    let response = resp.body.clone().unwrap_or(Value::Null);

    json!({
        "model": resp.model,
        "messages": messages,
        "response": response,
        "usage": usage_json,
        "cost": cost,
        "latency_ms": null,
        "request_id": ctx.request_id,
        "key_id": ctx.key_id,
        "team_id": ctx.team_id,
        "user_id": ctx.user_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::Hook;
    use oxllm_core::UsageSummary;
    use std::sync::Arc;
    use std::time::Duration;

    fn raw_config() -> HeliconeConfig {
        HeliconeConfig {
            api_key: "hc-key".to_string(),
            endpoint: Some("https://helicone.example.com".to_string()),
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
api_key: hc-key
endpoint: https://helicone.example.com
";
        let cfg: HeliconeConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.api_key, "hc-key");
        assert_eq!(
            cfg.endpoint.as_deref(),
            Some("https://helicone.example.com")
        );
    }

    #[tokio::test]
    async fn construction_applies_default_endpoint() {
        let cfg = HeliconeConfig {
            endpoint: None,
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://api.helicone.ai/v1/log/request");
    }

    #[tokio::test]
    async fn construction_trims_trailing_slash() {
        let cfg = HeliconeConfig {
            endpoint: Some("https://helicone.example.com/".to_string()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://helicone.example.com/v1/log/request");
    }

    #[test]
    fn construction_rejects_empty_api_key() {
        let cfg = HeliconeConfig {
            api_key: "".to_string(),
            ..raw_config()
        };
        assert!(build(&cfg, false, false).is_err());
    }

    /// LOW #10: an empty/whitespace `endpoint` must fall back to the default
    /// endpoint rather than producing a path-only URL.
    #[tokio::test]
    async fn construction_applies_default_when_endpoint_empty() {
        let cfg = HeliconeConfig {
            endpoint: Some("   ".to_string()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://api.helicone.ai/v1/log/request");
    }

    #[test]
    fn build_log_has_correct_shape() {
        let payload = helicone_payload(&make_ctx(), &make_response());
        assert_eq!(payload["model"], "gpt-4o");
        assert_eq!(payload["messages"][0]["role"], "user");
        assert_eq!(
            payload["response"]["choices"][0]["message"]["content"],
            "hi"
        );
        assert_eq!(payload["usage"]["prompt_tokens"], 100);
        assert_eq!(payload["usage"]["completion_tokens"], 50);
        assert_eq!(payload["usage"]["total_tokens"], 150);
        assert_eq!(payload["cost"], 0.0025);
    }

    /// LOW #6: Helicone must include `request_id` and identity fields
    /// (`key_id`, `team_id`, `user_id`) for parity with every other backend,
    /// so an operator can correlate a Helicone log entry back to an oxllm
    /// request and to a key/team for attribution.
    #[test]
    fn build_log_includes_request_id_and_identity_fields() {
        let payload = helicone_payload(&make_ctx(), &make_response());
        assert_eq!(payload["request_id"], "req-1");
        assert_eq!(payload["key_id"], "k-1");
        assert_eq!(payload["team_id"], "team-1");
    }

    /// MEDIUM #3: Helicone's ingestion API expects usage at the top level of
    /// the payload only. The legacy `helicone_payload` also synthesised a
    /// `response.usage` field, which is redundant and risks double-counting
    /// if Helicone sums both locations. Pin that `response` carries only the
    /// raw upstream body (no synthesised `usage`).
    #[test]
    fn helicone_payload_does_not_embed_usage_in_response() {
        let payload = helicone_payload(&make_ctx(), &make_response());
        assert!(
            payload["response"]["usage"].is_null(),
            "response.usage must not be synthesised; got: {}",
            payload["response"]
        );
        assert_eq!(
            payload["usage"]["prompt_tokens"], 100,
            "top-level usage must still carry the value"
        );
    }

    #[tokio::test]
    async fn log_posts_with_auth_header() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = HeliconeConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/log/request")
                .header("Helicone-Auth", "hc-key")
                .body_contains("\"model\":\"gpt-4o\"")
                .body_contains("\"prompt_tokens\":100");
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
            "helicone should have POSTed a log with the auth header"
        );
    }
}
