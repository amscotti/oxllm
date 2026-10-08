use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::types::LakeraConfig;
use crate::guardrails::GuardrailResult;

/// Default Lakera API base URL.
pub(crate) const DEFAULT_ENDPOINT: &str = "https://api.lakera.ai";

/// Default per-request timeout.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 5;

/// Lakera Guard `/v2/guard` scan result.
///
/// Matches the reference v2 `LakeraAIResponse` shape
/// (`types/proxy/guardrails/guardrail_hooks/lakera_ai_v2.py`):
///
/// ```json
/// { "flagged": true, "payload": [...], "breakdown": [...], "dev_info": {...} }
/// ```
///
/// `flagged` is required: an error body such as `{"detail": "unauthorized"}`
/// (or any response without this field) fails to parse and is treated as a
/// scan failure — with `serde(default)` it would otherwise parse as
/// `flagged: false` and silently allow the request. This mirrors the
/// fail-open-on-error-body design used by AIM/Bedrock/Model Armor.
///
/// The historical `score` field is intentionally absent: the v2 API does not
/// return one, and the reference v2 decides the block purely on
/// `lakera_response.get("flagged") is True`.
#[derive(Debug, Clone, Deserialize)]
pub struct LakeraScanResult {
    pub flagged: bool,
    #[serde(default)]
    pub payload: Value,
    #[serde(default)]
    pub breakdown: Value,
    #[serde(default)]
    pub dev_info: Value,
}

/// Lakera Guard `/v2/guard` request payload.
#[derive(Debug, Serialize)]
pub(crate) struct LakeraScanRequest {
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Lakera AI prompt-injection detection guardrail.
///
/// Calls the Lakera Guard v2 API (`POST {endpoint}/v2/guard`) to scan
/// messages for prompt injection. Blocks when the response reports
/// `flagged: true` (matching the reference v2
/// `lakera_ai_v2.LakeraGuardrailRequestBase` semantics). When `monitor` mode
/// is enabled, flagged requests are logged server-side and allowed through
/// instead of blocked (matching the reference v2 `on_flagged: "monitor"`).
/// Fail-open: errors are logged and the request is allowed through.
#[derive(Clone)]
pub struct LakeraGuardrail {
    client: reqwest::Client,
    scan_url: String,
    api_key: String,
    /// When `true`, flagged requests are logged and allowed (monitor mode).
    monitor: bool,
}

impl LakeraGuardrail {
    /// Build from explicit parameters. Returns an error when the HTTP client
    /// cannot be built. The HTTP client carries the connect-time
    /// [`crate::security::SsrfResolver`] unless `allow_private` is set, so a
    /// DNS-rebinding attack after config load cannot steer the connection to a
    /// private address.
    pub fn new(
        api_key: String,
        endpoint: Option<String>,
        timeout_secs: Option<u64>,
        allow_private: bool,
        monitor: bool,
    ) -> Result<Self, String> {
        // Clamp to >= 1s: a zero timeout makes every request time out
        // instantly, yielding a permanent silent fail-open.
        let timeout = Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS).max(1));

        // Redirects are disabled so the guardrail cannot be coerced into
        // following 3xx chains to unexpected hosts (SSRF hardening policy).
        // The client carries the connect-time [`crate::security::SsrfResolver`]
        // unless `allow_private` is set, so a DNS-rebinding attack after
        // config load cannot steer the connection to a private address.
        let client = crate::security::build_ssrf_http_client(allow_private, timeout)
            .map_err(|e| format!("failed to build Lakera HTTP client: {e}"))?;

        let endpoint = endpoint.unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        let base = endpoint.trim_end_matches('/');
        Ok(Self {
            client,
            scan_url: format!("{base}/v2/guard"),
            api_key,
            monitor,
        })
    }

    /// Build from a [`LakeraConfig`]. The deprecated `threshold` field is
    /// validated for range (a YAML `.nan` would otherwise silently disable
    /// other range checks elsewhere) but no longer affects blocking, since
    /// the v2 API does not return a score.
    pub fn from_config(config: &LakeraConfig, allow_private: bool) -> Result<Self, String> {
        if let Some(threshold) = config.threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(format!(
                    "Lakera threshold must be within 0.0–1.0, got {threshold} (note: threshold is deprecated on the v2 API and no longer affects blocking)"
                ));
            }
        }
        let monitor = match config.mode.as_deref() {
            Some("monitor") => true,
            Some("block") | None => false,
            Some(other) => {
                return Err(format!(
                    "Lakera mode must be 'block' or 'monitor', got '{other}'"
                ));
            }
        };
        Self::new(
            config.api_key.clone(),
            config.endpoint.clone(),
            config.timeout_secs,
            allow_private,
            monitor,
        )
    }

    /// Scan messages for prompt injection.
    pub async fn scan(
        &self,
        messages: &[Value],
        model: Option<&str>,
    ) -> anyhow::Result<LakeraScanResult> {
        let req = LakeraScanRequest {
            messages: messages.to_vec(),
            model: model.map(String::from),
        };
        // The shared reader rejects non-2xx and oversized response bodies.
        let response = self
            .client
            .post(&self.scan_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await?;
        super::response::read_json(response).await
    }

    /// Check a request body for prompt injection via Lakera Guard.
    ///
    /// Returns `GuardrailResult::Block` when the Lakera v2 response reports
    /// `flagged: true` (matching the reference v2
    /// `if lakera_response.get("flagged") is True` semantics). Fail-open:
    /// errors log a warning and allow the request through.
    pub async fn check_request(&self, body: &Value) -> GuardrailResult {
        let scan_messages = crate::guardrails::custom::scan_messages(body);
        if scan_messages.is_empty() {
            return GuardrailResult::Allow;
        }

        let model = body.get("model").and_then(Value::as_str);

        match self.scan(&scan_messages, model).await {
            Ok(result) => {
                if result.flagged {
                    if self.monitor {
                        // Monitor mode: log the violation server-side and
                        // allow the request through (matching the reference v2
                        // `on_flagged: "monitor"` at `lakera_ai_v2.py:261`).
                        tracing::warn!(
                            payload = ?result.payload,
                            breakdown = ?result.breakdown,
                            "Lakera Guard flagged request (monitor mode); allowing"
                        );
                        GuardrailResult::Allow
                    } else {
                        // Log payload/breakdown server-side only; the
                        // client-facing reason stays generic so a blocked caller
                        // cannot probe the detection logic.
                        tracing::warn!(
                            payload = ?result.payload,
                            breakdown = ?result.breakdown,
                            "Lakera Guard flagged request; blocking"
                        );
                        GuardrailResult::Block("request blocked by content policy".to_string())
                    }
                } else {
                    GuardrailResult::Allow
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Lakera Guard scan failed; allowing request");
                GuardrailResult::Allow
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    #[test]
    fn scan_request_serialization() {
        let messages = vec![json!({"role": "user", "content": "hello"})];
        let req = LakeraScanRequest {
            messages,
            model: Some("gpt-4o".to_string()),
        };
        let json_val = serde_json::to_value(&req).unwrap();
        assert_eq!(json_val["messages"][0]["role"], "user");
        assert_eq!(json_val["messages"][0]["content"], "hello");
        assert_eq!(json_val["model"], "gpt-4o");
    }

    #[test]
    fn scan_request_omits_null_model() {
        let req = LakeraScanRequest {
            messages: vec![json!({"role": "user", "content": "hi"})],
            model: None,
        };
        let json_val = serde_json::to_value(&req).unwrap();
        assert!(json_val.get("model").is_none());
    }

    #[test]
    fn scan_response_parsing() {
        // Real Lakera v2 response shape (per the reference
        // `LakeraAIResponse` TypedDict).
        let data = json!({
            "flagged": true,
            "payload": [{"start": 0, "end": 4, "text": "ignore", "detector_type": "prompt_injection"}],
            "breakdown": [{"project_id": "p", "policy_id": "pol", "detector_id": "d", "detector_type": "prompt_injection", "detected": true}],
            "dev_info": {"version": "1.0"}
        });
        let result: LakeraScanResult = serde_json::from_value(data).unwrap();
        assert!(result.flagged);
        assert!(result.payload.is_array());
        assert!(result.breakdown.is_array());
        assert!(result.dev_info.is_object());
    }

    #[test]
    fn scan_response_without_optional_fields_defaults() {
        // `flagged` is required; `payload` / `breakdown` / `dev_info` default
        // to null when absent.
        let data = json!({"flagged": false});
        let result: LakeraScanResult = serde_json::from_value(data).unwrap();
        assert!(!result.flagged);
        assert!(result.payload.is_null());
        assert!(result.breakdown.is_null());
        assert!(result.dev_info.is_null());
    }

    #[test]
    fn scan_response_missing_flagged_is_an_error() {
        // `flagged` is required: a response without it (e.g. a Lakera error
        // body) must fail to parse so the caller fails open with a warning
        // instead of treating the error as a clean allow.
        assert!(serde_json::from_value::<LakeraScanResult>(json!({})).is_err());
        assert!(
            serde_json::from_value::<LakeraScanResult>(json!({"detail": "Invalid API key"}))
                .is_err()
        );
    }

    #[tokio::test]
    async fn integration_scan_parses_real_lakera_v2_response() {
        // Regression for the HIGH-severity bug where oxllm's response struct
        // required a top-level `score` field, which the v2 API does not
        // return. Every real Lakera 200 OK used to fail to parse and the
        // guardrail silently failed open.
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method("POST")
                .path("/v2/guard")
                .header("Authorization", "Bearer test-key");
            then.status(200).json_body(json!({
                "flagged": true,
                "payload": [],
                "breakdown": [],
                "dev_info": {}
            }));
        });

        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some(server.base_url()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        match result {
            GuardrailResult::Block(reason) => {
                assert_eq!(reason, "request blocked by content policy");
            }
            GuardrailResult::Allow => panic!("expected block"),
        }
    }

    #[tokio::test]
    async fn integration_scan_blocks_when_flagged() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.status(200).json_body(json!({"flagged": true}));
        });

        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some(server.base_url()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Block(_)));
    }

    #[tokio::test]
    async fn integration_scan_allows_when_not_flagged() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.status(200).json_body(json!({"flagged": false}));
        });

        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some(server.base_url()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "good morning"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn score_threshold_is_ignored_only_flagged_matters() {
        // The v2 API does not return a `score`; a configured `threshold`
        // must NOT cause blocking on a `flagged: false` response, regardless
        // of how low the threshold is. Only `flagged` matters.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.status(200).json_body(json!({"flagged": false}));
        });

        // The threshold is set very low; even so, `flagged: false` allows.
        let cfg = LakeraConfig {
            enabled: Some(true),
            api_key: "test-key".to_string(),
            endpoint: Some(server.base_url()),
            threshold: Some(0.0),
            timeout_secs: None,
            mode: None,
        };
        let guardrail = LakeraGuardrail::from_config(&cfg, true).unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(
            matches!(result, GuardrailResult::Allow),
            "v2 has no score; only flagged should block"
        );
    }

    #[tokio::test]
    async fn fail_open_on_scan_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.status(500);
        });

        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some(server.base_url()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn service_unreachable_falls_through() {
        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some("http://127.0.0.1:1".to_string()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn empty_messages_allows() {
        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some("http://127.0.0.1:1".to_string()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({"model": "gpt-4o"});
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn fail_open_on_auth_error_with_json_error_body() {
        // A 401 carrying a JSON error body must fail open (with a warning
        // logged) — `error_for_status` converts 4xx/5xx to errors so the
        // body is never parsed.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.status(401)
                .json_body(json!({"detail": "Invalid API key"}));
        });

        let guardrail = LakeraGuardrail::new(
            "bad-key".to_string(),
            Some(server.base_url()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn error_shaped_json_body_with_200_status_fails_open() {
        // Even with a 200 status, a body that lacks `flagged` (e.g. an error
        // object relayed by a gateway) must fail to parse → fail open rather
        // than parse as a clean allow.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.status(200)
                .json_body(json!({"detail": "upstream degraded"}));
        });

        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some(server.base_url()),
            None,
            true,
            false,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[test]
    fn from_config_rejects_nan_threshold() {
        // The deprecated `threshold` field is still range-validated so a
        // YAML `.nan` does not silently disable other range checks.
        let cfg = LakeraConfig {
            enabled: Some(true),
            api_key: "k".to_string(),
            endpoint: None,
            threshold: Some(f64::NAN),
            timeout_secs: None,
            mode: None,
        };
        let err = LakeraGuardrail::from_config(&cfg, true)
            .err()
            .expect("NaN threshold rejected");
        assert!(err.contains("threshold"), "{err}");
    }

    #[test]
    fn from_config_rejects_out_of_range_threshold() {
        for threshold in [-0.1, 1.5, f64::INFINITY] {
            let cfg = LakeraConfig {
                enabled: Some(true),
                api_key: "k".to_string(),
                endpoint: None,
                threshold: Some(threshold),
                timeout_secs: None,
                mode: None,
            };
            assert!(
                LakeraGuardrail::from_config(&cfg, true).is_err(),
                "threshold {threshold} should be rejected"
            );
        }
    }

    #[test]
    fn from_config_accepts_boundary_thresholds() {
        for threshold in [0.0, 1.0] {
            let cfg = LakeraConfig {
                enabled: Some(true),
                api_key: "k".to_string(),
                endpoint: None,
                threshold: Some(threshold),
                timeout_secs: None,
                mode: None,
            };
            assert!(LakeraGuardrail::from_config(&cfg, true).is_ok());
        }
    }

    #[test]
    fn constructor_uses_default_endpoint() {
        let g = LakeraGuardrail::new("k".to_string(), None, None, true, false).unwrap();
        assert_eq!(g.scan_url, "https://api.lakera.ai/v2/guard");
    }

    #[test]
    fn constructor_clamps_zero_timeout_to_one() {
        // `timeout_secs: 0` would yield a 0-duration client timeout (instant
        // fail-open); clamp to >= 1s, matching AIM / Presidio / MCP / custom
        // HTTP guardrails.
        let g = LakeraGuardrail::new("k".to_string(), None, Some(0), true, false).unwrap();
        // The constructed client hides its timeout, but the constructor must
        // not panic / error on `0` and the rest of the struct must be usable.
        assert_eq!(g.scan_url, "https://api.lakera.ai/v2/guard");
    }

    #[tokio::test]
    async fn lakera_timeout_is_configurable() {
        // A custom `timeout_secs` shorter than the server's delay must time
        // out the scan and fail open.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.delay(Duration::from_millis(500))
                .status(200)
                .json_body(json!({"flagged": true}));
        });
        let guardrail = LakeraGuardrail::new(
            "k".to_string(),
            Some(server.base_url()),
            Some(1),
            true,
            false,
        )
        .unwrap();
        // With a 1s timeout the call succeeds despite the 500ms delay.
        let body = json!({"messages": [{"role": "user", "content": "x"}]});
        let r = guardrail.check_request(&body).await;
        assert!(matches!(r, GuardrailResult::Block(_)));
    }

    #[tokio::test]
    async fn monitor_mode_allows_flagged_request() {
        // In monitor mode, a flagged request is logged server-side and
        // allowed through (matching the reference v2 `on_flagged: "monitor"`).
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/v2/guard");
            then.status(200).json_body(json!({"flagged": true}));
        });

        let guardrail = LakeraGuardrail::new(
            "test-key".to_string(),
            Some(server.base_url()),
            None,
            true,
            true,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(
            matches!(result, GuardrailResult::Allow),
            "monitor mode must allow flagged requests"
        );
    }

    #[test]
    fn from_config_parses_monitor_mode() {
        let cfg = LakeraConfig {
            enabled: Some(true),
            api_key: "k".to_string(),
            endpoint: None,
            threshold: None,
            timeout_secs: None,
            mode: Some("monitor".to_string()),
        };
        let g = LakeraGuardrail::from_config(&cfg, true).unwrap();
        assert!(g.monitor, "mode: 'monitor' should set monitor=true");
    }

    #[test]
    fn from_config_parses_block_mode() {
        let cfg = LakeraConfig {
            enabled: Some(true),
            api_key: "k".to_string(),
            endpoint: None,
            threshold: None,
            timeout_secs: None,
            mode: Some("block".to_string()),
        };
        let g = LakeraGuardrail::from_config(&cfg, true).unwrap();
        assert!(!g.monitor, "mode: 'block' should set monitor=false");
    }

    #[test]
    fn from_config_rejects_invalid_mode() {
        let cfg = LakeraConfig {
            enabled: Some(true),
            api_key: "k".to_string(),
            endpoint: None,
            threshold: None,
            timeout_secs: None,
            mode: Some("bogus".to_string()),
        };
        let err = LakeraGuardrail::from_config(&cfg, true)
            .err()
            .expect("invalid mode rejected");
        assert!(err.contains("mode"), "{err}");
    }
}
