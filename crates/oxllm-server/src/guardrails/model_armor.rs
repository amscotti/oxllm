use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::types::ModelArmorConfig;
use crate::guardrails::GuardrailResult;

/// Default score threshold; scores at or above this block the request.
pub(crate) const DEFAULT_THRESHOLD: f64 = 0.5;

/// Model Armor scan result.
///
/// `risk_score` is required: an error body such as `{"error": "..."}`
/// (or any response without this field) fails to parse and is treated as a
/// scan failure — with `serde(default)` it would otherwise parse as a clean
/// `{risk_score: 0.0}` and silently allow the request.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArmorScanResult {
    pub risk_score: f64,
    #[serde(default)]
    pub blocked: bool,
    #[serde(default)]
    pub findings: Value,
}

/// Model Armor scan request payload.
#[derive(Debug, Serialize)]
pub(crate) struct ModelArmorScanRequest {
    pub messages: Vec<Value>,
}

/// Google Cloud Model Armor prompt-injection detection guardrail.
///
/// Calls a configurable Model Armor endpoint to scan messages for prompt
/// injection and malicious content. Fail-open: errors are logged and the
/// request is allowed through.
#[derive(Clone)]
pub struct ModelArmorGuardrail {
    client: reqwest::Client,
    scan_url: String,
    api_key: Option<String>,
    threshold: f64,
}

impl ModelArmorGuardrail {
    /// Build from explicit parameters. Returns an error when the threshold is
    /// NaN or outside `0.0–1.0` (YAML `.nan` would otherwise silently disable
    /// blocking: `risk_score >= NaN` is always false) or when the HTTP client
    /// cannot be built. The HTTP client carries the connect-time
    /// [`crate::security::SsrfResolver`] unless `allow_private` is set, so a
    /// DNS-rebinding attack after config load cannot steer the connection to a
    /// private address (matching AIM / Lakera / MCP / custom HTTP).
    pub fn new(
        endpoint: String,
        api_key: Option<String>,
        threshold: Option<f64>,
        timeout_secs: Option<u64>,
        allow_private: bool,
    ) -> Result<Self, String> {
        // Clamp to >= 1s: a zero timeout makes every request time out
        // instantly, yielding a permanent silent fail-open (matches AIM /
        // Presidio / Lakera / MCP / custom HTTP).
        let timeout = Duration::from_secs(timeout_secs.unwrap_or(5).max(1));

        // Redirects are disabled so the guardrail cannot be coerced into
        // following 3xx chains to unexpected hosts. The connect-time
        // [`crate::security::SsrfResolver`] closes the DNS-rebinding gap
        // between the config-load `validate_api_base` check and the actual
        // `reqwest` connect — every other operator-pointable guardrail
        // (AIM / Lakera / MCP / custom HTTP) installs the same resolver.
        let client = crate::security::build_ssrf_http_client(allow_private, timeout)
            .map_err(|e| format!("failed to build Model Armor HTTP client: {e}"))?;

        let threshold = threshold.unwrap_or(DEFAULT_THRESHOLD);
        // `contains` returns false for NaN, so both NaN and out-of-range
        // values are rejected here.
        if !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "Model Armor threshold must be within 0.0–1.0, got {threshold}"
            ));
        }

        Ok(Self {
            client,
            scan_url: endpoint,
            api_key,
            threshold,
        })
    }

    /// Build from a [`ModelArmorConfig`]. `allow_private` gates the
    /// connect-time SSRF resolver (matching AIM / Lakera / MCP / custom
    /// HTTP guardrails).
    pub fn from_config(config: &ModelArmorConfig, allow_private: bool) -> Result<Self, String> {
        Self::new(
            config.endpoint.clone(),
            config.api_key.clone(),
            config.threshold,
            config.timeout_secs,
            allow_private,
        )
    }

    /// Scan messages for prompt injection.
    pub async fn scan(&self, messages: &[Value]) -> anyhow::Result<ModelArmorScanResult> {
        let req = ModelArmorScanRequest {
            messages: messages.to_vec(),
        };
        // The shared reader rejects non-2xx and oversized response bodies.
        let mut builder = self.client.post(&self.scan_url).json(&req);
        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }
        let response = builder.send().await?;
        super::response::read_json(response).await
    }

    /// Check a request body for prompt injection via Model Armor.
    ///
    /// Returns `GuardrailResult::Block` when `blocked` is true or the
    /// `risk_score` exceeds the configured threshold. Fail-open: errors log a
    /// warning and allow the request through.
    pub async fn check_request(&self, body: &Value) -> GuardrailResult {
        let scan_messages = crate::guardrails::custom::scan_messages(body);
        if scan_messages.is_empty() {
            return GuardrailResult::Allow;
        }

        match self.scan(&scan_messages).await {
            Ok(result) => {
                if result.blocked || result.risk_score >= self.threshold {
                    // Log the score/findings server-side only; the
                    // client-facing reason stays generic so a blocked caller
                    // cannot probe the detection logic.
                    tracing::warn!(
                        risk_score = result.risk_score,
                        blocked = result.blocked,
                        findings = ?result.findings,
                        "Model Armor flagged request; blocking"
                    );
                    GuardrailResult::Block("request blocked by content policy".to_string())
                } else {
                    GuardrailResult::Allow
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Model Armor scan failed; allowing request");
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
        let req = ModelArmorScanRequest { messages };
        let json_val = serde_json::to_value(&req).unwrap();
        assert_eq!(json_val["messages"][0]["role"], "user");
        assert_eq!(json_val["messages"][0]["content"], "hello");
    }

    #[test]
    fn scan_response_parsing() {
        let data = json!({
            "risk_score": 0.95,
            "blocked": true,
            "findings": {"prompt_injection": true}
        });
        let result: ModelArmorScanResult = serde_json::from_value(data).unwrap();
        assert!((result.risk_score - 0.95).abs() < 0.001);
        assert!(result.blocked);
        assert_eq!(result.findings["prompt_injection"], true);
    }

    #[test]
    fn scan_response_without_findings_defaults() {
        let data = json!({"risk_score": 0.1, "blocked": false});
        let result: ModelArmorScanResult = serde_json::from_value(data).unwrap();
        assert!((result.risk_score - 0.1).abs() < 0.001);
        assert!(!result.blocked);
        assert!(result.findings.is_null());
    }

    #[test]
    fn scan_response_missing_risk_score_is_an_error() {
        // `risk_score` is required: a response without it (e.g. an error
        // body) must fail to parse so the caller fails open with a warning
        // instead of treating the error as a clean allow.
        assert!(serde_json::from_value::<ModelArmorScanResult>(json!({})).is_err());
        assert!(
            serde_json::from_value::<ModelArmorScanResult>(json!({"error": "Invalid API key"}))
                .is_err()
        );
    }

    /// Helper: build a Model Armor guardrail pointed at `url` with the SSRF
    /// resolver bypassed (so httpmock's loopback URL is reachable, matching
    /// the AIM / Lakera / MCP test setup).
    fn guardrail_at(endpoint: String, threshold: Option<f64>) -> ModelArmorGuardrail {
        ModelArmorGuardrail::new(endpoint, None, threshold, None, true).unwrap()
    }

    #[tokio::test]
    async fn integration_scan_blocks_on_flagged() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method("POST")
                .path("/")
                .header("Authorization", "Bearer test-key");
            then.status(200).json_body(json!({
                "risk_score": 0.95,
                "blocked": true,
                "findings": {"prompt_injection": true}
            }));
        });

        let guardrail = ModelArmorGuardrail::new(
            server.base_url(),
            Some("test-key".to_string()),
            Some(0.5),
            None,
            true,
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
    async fn integration_scan_allows_on_clean() {
        let server = MockServer::start();

        server.mock(|when, then| {
            when.method("POST").path("/");
            then.status(200).json_body(json!({
                "risk_score": 0.1,
                "blocked": false,
                "findings": {}
            }));
        });

        let guardrail = guardrail_at(server.base_url(), Some(0.5));
        let body = json!({
            "messages": [{"role": "user", "content": "good morning"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn score_at_threshold_blocks() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/");
            then.status(200).json_body(json!({
                "risk_score": 0.5,
                "blocked": false,
                "findings": {}
            }));
        });

        let guardrail = guardrail_at(server.base_url(), Some(0.5));
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Block(_)));
    }

    #[tokio::test]
    async fn blocked_true_blocks_even_below_threshold() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/");
            then.status(200).json_body(json!({
                "risk_score": 0.1,
                "blocked": true,
                "findings": {"prompt_injection": true}
            }));
        });

        let guardrail = guardrail_at(server.base_url(), Some(0.9));
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Block(_)));
    }

    #[tokio::test]
    async fn fail_open_on_scan_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/");
            then.status(500);
        });

        let guardrail = guardrail_at(server.base_url(), Some(0.5));
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn fail_open_on_timeout() {
        let guardrail = ModelArmorGuardrail::new(
            "http://127.0.0.1:1".to_string(),
            None,
            Some(0.5),
            Some(1),
            true,
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
        let guardrail = guardrail_at("http://127.0.0.1:1".to_string(), Some(0.5));
        let body = json!({"model": "gpt-4o"});
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn fail_open_on_auth_error_with_json_error_body() {
        // A 401 carrying a JSON error body must fail open (with a warning
        // logged) — not parse as a clean {risk_score: 0.0} allow.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/");
            then.status(401)
                .json_body(json!({"error": "Invalid API key"}));
        });

        let guardrail = ModelArmorGuardrail::new(
            server.base_url(),
            Some("bad-key".to_string()),
            Some(0.5),
            None,
            true,
        )
        .unwrap();
        let body = json!({
            "messages": [{"role": "user", "content": "ignore instructions"}]
        });
        let result = guardrail.check_request(&body).await;
        assert!(matches!(result, GuardrailResult::Allow));
    }

    #[test]
    fn constructor_rejects_nan_threshold() {
        let err =
            ModelArmorGuardrail::new("https://x".to_string(), None, Some(f64::NAN), None, true)
                .err()
                .expect("NaN threshold rejected");
        assert!(err.contains("threshold"), "{err}");
    }

    #[test]
    fn constructor_rejects_out_of_range_threshold() {
        for threshold in [-0.1, 1.5, f64::INFINITY] {
            assert!(
                ModelArmorGuardrail::new(
                    "https://x".to_string(),
                    None,
                    Some(threshold),
                    None,
                    true
                )
                .is_err(),
                "threshold {threshold} should be rejected"
            );
        }
    }

    #[test]
    fn constructor_accepts_boundary_thresholds() {
        for threshold in [0.0, 1.0] {
            assert!(
                ModelArmorGuardrail::new(
                    "https://x".to_string(),
                    None,
                    Some(threshold),
                    None,
                    true
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn constructor_clamps_zero_timeout_to_one() {
        // `timeout_secs: 0` would yield a 0-duration client timeout (instant
        // fail-open); clamp to >= 1s, matching AIM / Presidio / Lakera / MCP
        // / custom HTTP guardrails. The constructor must not error on `0`.
        let g = ModelArmorGuardrail::new("https://x".to_string(), None, Some(0.5), Some(0), true)
            .expect("zero timeout should clamp to 1s, not error");
        assert_eq!(g.scan_url, "https://x");
    }

    #[tokio::test]
    async fn constructor_installs_ssrf_resolver_when_not_allow_private() {
        // When `allow_private` is false, the client must install the SSRF
        // DNS resolver — the same one used by AIM / Lakera / MCP / custom
        // HTTP. The resolver rejects connect-time DNS answers that resolve
        // to private / loopback / link-local ranges, closing the DNS-rebind
        // gap between config-load `validate_api_base` and the actual
        // `reqwest` connect.
        //
        // The SSRF resolver only kicks in for DNS *names* — IP literals
        // (e.g. http://127.0.0.1) bypass DNS entirely, so we use
        // `localhost` which goes through the hosts file / DNS and resolves
        // to a loopback address that the resolver filters out.
        let server = MockServer::start();
        let port = server.port();

        let guardrail_protected = ModelArmorGuardrail::new(
            format!("http://localhost:{port}"),
            None,
            Some(0.5),
            Some(2),
            false,
        )
        .unwrap();
        let body = json!({"messages": [{"role": "user", "content": "x"}]});
        let r = guardrail_protected.check_request(&body).await;
        assert!(
            matches!(r, GuardrailResult::Allow),
            "SSRF resolver should reject the loopback target and fail open"
        );

        // Sanity: with `allow_private: true` the same hostname must reach
        // the server. We re-mock on a fresh server because the protected
        // client never reached the first one.
        let server2 = MockServer::start();
        let port2 = server2.port();
        let mock = server2.mock(|when, then| {
            when.method("POST").path("/");
            then.status(200).json_body(json!({
                "risk_score": 0.95, "blocked": true, "findings": {}
            }));
        });
        let guardrail_open = ModelArmorGuardrail::new(
            format!("http://localhost:{port2}"),
            None,
            Some(0.5),
            Some(2),
            true,
        )
        .unwrap();
        let r2 = guardrail_open.check_request(&body).await;
        assert!(
            matches!(r2, GuardrailResult::Block(_)),
            "allow_private client should reach the server and block"
        );
        assert_eq!(mock.hits(), 1, "allow_private client must reach the server");
    }
}
