//! LangSmith tracing hook.
//!
//! Built on [`HttpTelemetryHook`]: each successful (non-cache, non-master)
//! request is POSTed as a "run" to the LangSmith batch ingestion API
//! (`POST {endpoint}/v1/runs/batch`) under an `x-api-key` header. The POST is
//! fire-and-forget. Wired in from `oxllm_settings.langsmith_config`.

use std::sync::Arc;

use oxllm_core::{CallContext, ResponseSummary};
use serde_json::{Value, json};

use crate::config::types::LangsmithConfig;
use crate::hooks::cost_fallback::fallback_cost;
use crate::hooks::http_telemetry::{HttpTelemetryHook, resolve_endpoint};

const DEFAULT_ENDPOINT: &str = "https://api.langsmith.ai";

/// Build a LangSmith telemetry hook from its config block.
///
/// Validates the API key and project, resolves the endpoint (defaulting to
/// LangSmith cloud), and constructs the batch ingestion URL. The project name
/// is captured into the payload-builder closure so `langsmith_run` needs no
/// config plumbing of its own.
pub fn build(
    config: &LangsmithConfig,
    allow_private: bool,
    redact_secret_values: bool,
) -> anyhow::Result<HttpTelemetryHook> {
    if config.api_key.trim().is_empty() {
        anyhow::bail!("langsmith_config.api_key must be set");
    }
    if config.project.trim().is_empty() {
        anyhow::bail!("langsmith_config.project must be set");
    }
    let endpoint = resolve_endpoint(config.endpoint.as_deref(), DEFAULT_ENDPOINT);
    let url = format!("{endpoint}/v1/runs/batch");
    let project = config.project.clone();
    HttpTelemetryHook::new_full(
        url,
        Some(("x-api-key".to_string(), config.api_key.clone())),
        move |ctx, resp| json!({ "post": [langsmith_run(&project, ctx, resp)] }),
        "langsmith",
        crate::hooks::http_telemetry::DEFAULT_CHANNEL_CAPACITY,
        allow_private,
        redact_secret_values,
    )
}

/// Build a LangSmith hook from `oxllm_settings.langsmith_config` when
/// present. Returns `None` when langsmith is not configured or fails SSRF
/// validation (misconfiguration is logged at warn level).
pub(crate) fn build_langsmith_hook_opt(
    config: &crate::config::ProxyConfig,
) -> Option<Arc<HttpTelemetryHook>> {
    let cfg = config.oxllm_settings.langsmith_config.as_ref()?;
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
            "langsmith_config endpoint '{endpoint}' failed SSRF validation; skipping"
        );
        return None;
    }
    match build(cfg, allow_private, redact_secret_values) {
        Ok(hook) => Some(Arc::new(hook)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build langsmith hook; skipping");
            None
        }
    }
}

/// Build the LangSmith run object for a single request. Free function so the
/// payload-shape test can exercise it directly; the wire body wraps it as
/// `{ "post": [run] }`.
fn langsmith_run(project: &str, ctx: &CallContext, resp: &ResponseSummary) -> Value {
    let run_id = uuid::Uuid::new_v4();
    // MEDIUM #5: derive `start_time` / `end_time` from the wall-clock window
    // stamped on the CallContext so the run's reported duration matches the
    // actual upstream time. The previous code set both to `now_iso()` and
    // every run surfaced as zero-duration in LangSmith. Fall back to a
    // single `now()` sample when `start_time` is missing.
    let end = jiff::Timestamp::now();
    let start = ctx.start_time.and_then(|s| {
        let start_system = std::time::SystemTime::now().checked_sub(s.elapsed())?;
        jiff::Timestamp::try_from(start_system).ok()
    });
    let (start_iso, end_iso) = match start {
        Some(s) => (s.to_string(), end.to_string()),
        None => (end.to_string(), end.to_string()),
    };

    let cost = fallback_cost(ctx, resp);

    let extra = json!({
        "key_id": ctx.key_id,
        "team_id": ctx.team_id,
        "user_id": ctx.user_id,
        "request_id": ctx.request_id,
        "provider": resp.provider,
        "cost": cost,
    });

    let usage = resp.usage.as_ref().map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    });

    let mut outputs = resp.body.clone().unwrap_or(Value::Null);
    if let Some(u) = usage {
        if let Value::Object(map) = &mut outputs {
            map.insert("usage".to_string(), u);
        } else {
            outputs = json!({ "body": outputs, "usage": u });
        }
    }

    json!({
        "id": run_id.to_string(),
        "name": resp.model,
        "run_type": "llm",
        "inputs": (*ctx.request_body).clone(),
        "outputs": outputs,
        "extra": extra,
        "session_name": project,
        "start_time": start_iso,
        "end_time": end_iso,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;
    use std::sync::Arc;
    use std::time::Duration;

    fn raw_config() -> LangsmithConfig {
        LangsmithConfig {
            api_key: "ls-key".to_string(),
            project: "my-project".to_string(),
            endpoint: Some("https://langsmith.example.com".to_string()),
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
                browser_open_requests: None,
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
api_key: ls-key
project: my-project
endpoint: https://langsmith.example.com
";
        let cfg: LangsmithConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.api_key, "ls-key");
        assert_eq!(cfg.project, "my-project");
        assert_eq!(
            cfg.endpoint.as_deref(),
            Some("https://langsmith.example.com")
        );
    }

    #[tokio::test]
    async fn construction_applies_default_endpoint() {
        let cfg = LangsmithConfig {
            endpoint: None,
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://api.langsmith.ai/v1/runs/batch");
    }

    #[tokio::test]
    async fn construction_trims_trailing_slash() {
        let cfg = LangsmithConfig {
            endpoint: Some("https://langsmith.example.com/".to_string()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://langsmith.example.com/v1/runs/batch");
    }

    #[test]
    fn construction_rejects_empty_api_key() {
        let cfg = LangsmithConfig {
            api_key: "".to_string(),
            ..raw_config()
        };
        assert!(build(&cfg, false, false).is_err());
    }

    #[test]
    fn construction_rejects_empty_project() {
        let cfg = LangsmithConfig {
            project: "".to_string(),
            ..raw_config()
        };
        assert!(build(&cfg, false, false).is_err());
    }

    /// LOW #10: an empty/whitespace `endpoint` must not produce a path-only
    /// URL (`/v1/runs/batch`) that reqwest would fail on per-request; it
    /// must fall back to the default endpoint.
    #[tokio::test]
    async fn construction_applies_default_when_endpoint_empty() {
        let cfg = LangsmithConfig {
            endpoint: Some("   ".to_string()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");
        assert_eq!(hook.url(), "https://api.langsmith.ai/v1/runs/batch");
    }

    #[test]
    fn build_run_has_correct_shape() {
        let run = langsmith_run("my-project", &make_ctx(), &make_response());
        assert_eq!(run["name"], "gpt-4o");
        assert_eq!(run["run_type"], "llm");
        assert_eq!(run["inputs"]["model"], "gpt-4o");
        assert_eq!(run["outputs"]["choices"][0]["message"]["content"], "hi");
        assert_eq!(run["outputs"]["usage"]["prompt_tokens"], 100);
        assert_eq!(run["outputs"]["usage"]["completion_tokens"], 50);
        assert_eq!(run["outputs"]["usage"]["total_tokens"], 150);
        assert_eq!(run["extra"]["key_id"], "k-1");
        assert_eq!(run["extra"]["team_id"], "team-1");
        assert_eq!(run["extra"]["provider"], "openai");
        assert_eq!(run["extra"]["cost"], 0.0025);
        assert_eq!(run["session_name"], "my-project");
        assert!(run["id"].is_string());
        assert!(run["start_time"].is_string());
        assert!(run["end_time"].is_string());
    }

    #[tokio::test]
    async fn runs_batch_posts_with_api_key_header() {
        use httpmock::MockServer;
        use oxllm_core::Hook;

        let server = MockServer::start();
        let cfg = LangsmithConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = build(&cfg, false, false).expect("build");

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/runs/batch")
                .header("x-api-key", "ls-key")
                .body_contains("\"run_type\":\"llm\"")
                .body_contains("\"session_name\":\"my-project\"");
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
            "langsmith should have POSTed a run with the api key"
        );
    }
}
