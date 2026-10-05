use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::types::PresidioConfig;

/// A detected PII entity returned by Presidio's `/analyze` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PresidioEntity {
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_explanation: Option<Value>,
}

/// Presidio `/analyze` request payload.
///
/// `language` is required by Presidio's analyzer — requests without it are
/// rejected with 400 — so it is always serialized.
#[derive(Debug, Serialize)]
pub(crate) struct AnalyzeRequest {
    pub text: String,
    pub language: String,
}

/// Presidio `/anonymize` request payload.
#[derive(Debug, Serialize)]
pub(crate) struct AnonymizeRequest {
    pub text: String,
    pub analyzer_results: Vec<AnalyzerResultItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anonymizers: Option<Value>,
}

/// One item in the `analyzer_results` array for the anonymize request.
#[derive(Debug, Serialize)]
pub(crate) struct AnalyzerResultItem {
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub score: f64,
}

/// Presidio `/anonymize` response payload.
#[derive(Debug, Deserialize)]
pub(crate) struct AnonymizeResponse {
    pub text: String,
}

/// Presidio PII guardrail.
///
/// Calls the Presidio service to detect and redact PII from text.
/// Fail-open: errors are logged and the request is allowed through.
#[derive(Clone)]
pub struct PresidioGuardrail {
    client: reqwest::Client,
    analyze_url: String,
    anonymize_url: String,
    language: String,
    anonymizers: Option<Value>,
}

impl PresidioGuardrail {
    /// Build from a [`PresidioConfig`]. Returns an error when the HTTP client
    /// cannot be built (propagated by the caller so a config hot-reload is
    /// rejected instead of panicking).
    ///
    /// `allow_private` gates private/loopback targets at the DNS connect layer
    /// via [`crate::security::build_ssrf_http_client`]: when `false`, the
    /// client installs the [`crate::security::SsrfResolver`] so a DNS-rebinding
    /// attack after config load cannot steer the connection to a private
    /// address. Defaults to `true` for Presidio (which typically runs on
    /// localhost), but should be `false` when the Presidio URL is on a public
    /// host.
    pub fn new(config: &PresidioConfig, allow_private: bool) -> Result<Self, String> {
        let analyze = config
            .analyze_endpoint
            .clone()
            .unwrap_or_else(|| "/analyze".into());
        let anonymize = config
            .anonymize_endpoint
            .clone()
            .unwrap_or_else(|| "/anonymize".into());
        // Clamp to >= 1s: a zero timeout makes every request time out
        // instantly, yielding a permanent silent fail-open.
        let timeout = Duration::from_secs(config.timeout_secs.unwrap_or(5).max(1));

        // Redirects are disabled so the guardrail cannot be coerced into
        // following 3xx chains to unexpected hosts (SSRF hardening policy).
        // The client carries the connect-time [`crate::security::SsrfResolver`]
        // unless `allow_private` is set, matching AIM / Lakera / ModelArmor /
        // MCP / custom HTTP.
        let client = crate::security::build_ssrf_http_client(allow_private, timeout)
            .map_err(|e| format!("failed to build Presidio HTTP client: {e}"))?;

        let base = config.url.trim_end_matches('/');
        Ok(Self {
            client,
            analyze_url: format!("{base}{analyze}"),
            anonymize_url: format!("{base}{anonymize}"),
            language: config.language.clone().unwrap_or_else(|| "en".to_string()),
            anonymizers: config.anonymizers.clone(),
        })
    }

    /// Analyze text for PII entities.
    pub async fn analyze(&self, text: &str) -> anyhow::Result<Vec<PresidioEntity>> {
        let req = AnalyzeRequest {
            text: text.to_string(),
            language: self.language.clone(),
        };
        let response = self
            .client
            .post(&self.analyze_url)
            .json(&req)
            .send()
            .await?;
        super::response::read_json(response).await
    }

    /// Anonymize text by replacing detected PII entities with placeholders.
    pub async fn anonymize(
        &self,
        text: &str,
        entities: &[PresidioEntity],
    ) -> anyhow::Result<String> {
        let analyzer_results: Vec<AnalyzerResultItem> = entities
            .iter()
            .map(|e| AnalyzerResultItem {
                entity_type: e.entity_type.clone(),
                start: e.start,
                end: e.end,
                score: e.score,
            })
            .collect();

        let req = AnonymizeRequest {
            text: text.to_string(),
            analyzer_results,
            anonymizers: Some(self.anonymizers.clone().unwrap_or_else(|| {
                serde_json::json!({
                    "DEFAULT": {
                        "type": "replace",
                        "new_value": "<REDACTED>"
                    }
                })
            })),
        };

        let response = self
            .client
            .post(&self.anonymize_url)
            .json(&req)
            .send()
            .await?;
        let resp: AnonymizeResponse = super::response::read_json(response).await?;
        Ok(resp.text)
    }

    /// Run the full PII redaction pipeline on a single text segment.
    /// Fail-open: errors log a warning and return the original text.
    pub async fn apply(&self, text: &str) -> String {
        match self.try_apply(text).await {
            Ok(redacted) => redacted,
            Err(e) => {
                tracing::warn!(error = %e, "Presidio redaction failed; falling through");
                text.to_string()
            }
        }
    }

    async fn try_apply(&self, text: &str) -> anyhow::Result<String> {
        let entities = self.analyze(text).await?;
        if entities.is_empty() {
            Ok(text.to_string())
        } else {
            self.anonymize(text, &entities).await
        }
    }

    /// Apply Presidio PII redaction to a chat-shaped request body.
    ///
    /// Covers `messages[].content` (both the legacy string shape and the
    /// array-of-parts shape), `messages[].name`, assistant
    /// `tool_calls[].function.description` / `arguments` (JSON argument
    /// strings are parsed, redacted recursively, and re-serialized), and
    /// top-level wrapped or flat tool descriptions and `parameters` /
    /// `input_schema` string leaves — mirroring the field coverage of
    /// [`crate::guardrails::PiiGuardrail::mask_request_body`].
    /// Fail-open: errors log a warning and leave the body unchanged. Changes
    /// are staged until every segment succeeds, including when the enclosing
    /// remote-guardrail deadline cancels this future midway through a request.
    ///
    /// Identical text segments are deduplicated via a per-request memo so a
    /// tool catalog with many identical JSON-schema `description` fields
    /// results in a single Presidio round-trip rather than N.
    pub async fn apply_to_body(&self, body: &mut Value) {
        let mut staged = body.clone();
        match self.try_apply_to_body(&mut staged).await {
            Ok(()) => *body = staged,
            Err(e) => {
                tracing::warn!(error = %e, "Presidio redaction failed; leaving request unchanged");
            }
        }
    }

    async fn try_apply_to_body(&self, body: &mut Value) -> anyhow::Result<()> {
        let mut memo: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
            for msg in messages.iter_mut() {
                if let Some(content) = msg.get_mut("content") {
                    match content {
                        Value::String(s) => {
                            self.apply_segment(s, &mut memo).await?;
                        }
                        Value::Array(parts) => {
                            for part in parts.iter_mut() {
                                if let Some(Value::String(t)) = part.get_mut("text") {
                                    self.apply_segment(t, &mut memo).await?;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(Value::String(name)) = msg.get_mut("name") {
                    self.apply_segment(name, &mut memo).await?;
                }
                if let Some(Value::Array(tool_calls)) = msg.get_mut("tool_calls") {
                    for call in tool_calls.iter_mut() {
                        if let Some(Value::Object(func)) = call.get_mut("function") {
                            if let Some(Value::String(desc)) = func.get_mut("description") {
                                self.apply_segment(desc, &mut memo).await?;
                            }
                            if let Some(args) = func.get_mut("arguments") {
                                self.apply_to_arguments(args, &mut memo).await?;
                            }
                        }
                    }
                }
            }
        }
        if let Some(Value::Array(tools)) = body.get_mut("tools") {
            for tool in tools.iter_mut() {
                if let Some(func) = super::custom::tool_definition_mut(tool) {
                    if let Some(Value::String(desc)) = func.get_mut("description") {
                        self.apply_segment(desc, &mut memo).await?;
                    }
                    if let Some(params) = super::custom::tool_schema_mut(func) {
                        self.apply_to_value_recursive(params, &mut memo).await?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Redact a single text segment in place. Empty/whitespace-only segments
    /// are skipped: Presidio's `/analyze` rejects them with 422. Identical
    /// segments are served from `memo` so Presidio is called at most once per
    /// distinct text value per request. Returns whether the text changed.
    async fn apply_segment(
        &self,
        text: &mut String,
        memo: &mut std::collections::HashMap<String, String>,
    ) -> anyhow::Result<bool> {
        if text.trim().is_empty() {
            return Ok(false);
        }
        if let Some(redacted) = memo.get(text.as_str()) {
            let changed = redacted.as_str() != text.as_str();
            if changed {
                *text = redacted.clone();
            }
            return Ok(changed);
        }
        let redacted = self.try_apply(text).await?;
        let changed = redacted != *text;
        memo.insert(text.clone(), redacted.clone());
        if changed {
            *text = redacted;
        }
        Ok(changed)
    }

    /// Redact a tool-call `arguments` payload: when the string parses as
    /// JSON, redact every string leaf and re-serialize only when a leaf changes.
    /// Clean arguments retain their original formatting and numeric precision;
    /// non-JSON arguments are redacted as raw strings.
    async fn apply_to_arguments(
        &self,
        args: &mut Value,
        memo: &mut std::collections::HashMap<String, String>,
    ) -> anyhow::Result<()> {
        let Value::String(s) = args else {
            return Ok(());
        };
        if s.trim().is_empty() {
            return Ok(());
        }
        let Ok(mut parsed) = serde_json::from_str::<Value>(s) else {
            return self.apply_segment(s, memo).await.map(|_| ());
        };
        if self.apply_to_value_recursive(&mut parsed, memo).await? {
            *s = parsed.to_string();
        }
        Ok(())
    }

    /// Redact every string leaf in a JSON value, returning whether any changed.
    async fn apply_to_value_recursive(
        &self,
        value: &mut Value,
        memo: &mut std::collections::HashMap<String, String>,
    ) -> anyhow::Result<bool> {
        let mut leaves = Vec::new();
        Self::collect_strings(value, &mut leaves);
        let mut changed = false;
        for s in leaves {
            changed |= self.apply_segment(s, memo).await?;
        }
        Ok(changed)
    }

    /// Collect mutable references to every string leaf in `value`.
    fn collect_strings<'a>(value: &'a mut Value, out: &mut Vec<&'a mut String>) {
        match value {
            Value::String(s) => out.push(s),
            Value::Array(arr) => {
                for v in arr {
                    Self::collect_strings(v, out);
                }
            }
            Value::Object(map) => {
                for v in map.values_mut() {
                    Self::collect_strings(v, out);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    #[test]
    fn analyze_request_serialization() {
        let req = AnalyzeRequest {
            text: "email me at test@example.com".to_string(),
            language: "en".to_string(),
        };
        let json_val = serde_json::to_value(&req).unwrap();
        assert_eq!(json_val["text"], "email me at test@example.com");
        // `language` is required by Presidio's /analyze — always serialized.
        assert_eq!(json_val["language"], "en");
    }

    #[test]
    fn analyze_response_parsing() {
        let data = json!([
            {"entity_type": "EMAIL_ADDRESS", "start": 11, "end": 27, "score": 0.99}
        ]);
        let entities: Vec<PresidioEntity> = serde_json::from_value(data).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "EMAIL_ADDRESS");
        assert_eq!(entities[0].start, 11);
        assert_eq!(entities[0].end, 27);
        assert!((entities[0].score - 0.99).abs() < 0.001);
    }

    #[test]
    fn analyze_response_with_optional_fields() {
        let data = json!([
            {
                "entity_type": "CREDIT_CARD",
                "start": 0,
                "end": 19,
                "score": 0.95,
                "analysis_explanation": {"recognizer": "CreditCardRecognizer"}
            }
        ]);
        let entities: Vec<PresidioEntity> = serde_json::from_value(data).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "CREDIT_CARD");
        assert!(entities[0].analysis_explanation.is_some());
    }

    #[test]
    fn anonymize_request_serialization() {
        let entities = [PresidioEntity {
            entity_type: "EMAIL_ADDRESS".to_string(),
            start: 11,
            end: 27,
            score: 0.99,
            analysis_explanation: None,
        }];
        let analyzer_results: Vec<AnalyzerResultItem> = entities
            .iter()
            .map(|e| AnalyzerResultItem {
                entity_type: e.entity_type.clone(),
                start: e.start,
                end: e.end,
                score: e.score,
            })
            .collect();
        let req = AnonymizeRequest {
            text: "email me at test@example.com".to_string(),
            analyzer_results,
            anonymizers: Some(json!({
                "DEFAULT": {"type": "replace", "new_value": "<REDACTED>"}
            })),
        };
        let json_val = serde_json::to_value(&req).unwrap();
        assert_eq!(json_val["text"], "email me at test@example.com");
        assert_eq!(
            json_val["analyzer_results"][0]["entity_type"],
            "EMAIL_ADDRESS"
        );
        assert_eq!(json_val["analyzer_results"][0]["start"], 11);
        assert!(json_val["anonymizers"].is_object());
    }

    #[test]
    fn anonymize_response_parsing() {
        let data = json!({"text": "email me at <REDACTED>"});
        let resp: AnonymizeResponse = serde_json::from_value(data).unwrap();
        assert_eq!(resp.text, "email me at <REDACTED>");
    }

    #[tokio::test]
    async fn integration_analyze_and_anonymize() {
        let server = MockServer::start();

        let analyze_mock = server.mock(|when, then| {
            when.method("POST")
                .path("/analyze")
                .header("content-type", "application/json")
                // Presidio requires `language` in the analyze request body.
                .body_contains("\"language\":\"en\"");
            then.status(200).json_body(json!([
                {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17, "score": 0.99}
            ]));
        });

        let anonymize_mock = server.mock(|when, then| {
            when.method("POST")
                .path("/anonymize")
                .header("content-type", "application/json");
            then.status(200).json_body(json!({"text": "<REDACTED>"}));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let result = guardrail.apply("user@example.com").await;
        assert_eq!(result, "<REDACTED>");
        assert_eq!(analyze_mock.hits(), 1);
        assert_eq!(anonymize_mock.hits(), 1);
    }

    #[tokio::test]
    async fn no_entities_skips_anonymize() {
        let server = MockServer::start();
        let analyze_mock = server.mock(|when, then| {
            when.method("POST").path("/analyze");
            then.status(200).json_body(json!([]));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let result = guardrail.apply("hello world").await;
        assert_eq!(result, "hello world");
        assert_eq!(analyze_mock.hits(), 1);
    }

    #[tokio::test]
    async fn fail_open_on_analyze_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/analyze");
            then.status(500);
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let result = guardrail.apply("user@example.com").await;
        assert_eq!(result, "user@example.com");
    }

    #[tokio::test]
    async fn fail_open_on_anonymize_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/analyze");
            then.status(200).json_body(json!([
                {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17, "score": 0.99}
            ]));
        });
        server.mock(|when, then| {
            when.method("POST").path("/anonymize");
            then.status(500);
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let result = guardrail.apply("user@example.com").await;
        assert_eq!(result, "user@example.com");
    }

    #[tokio::test]
    async fn later_failure_rolls_back_earlier_redactions() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.path("/analyze").body_contains("first");
            then.status(200).json_body(json!([
                {"entity_type":"PERSON","start":0,"end":5,"score":1.0}
            ]));
        });
        let anonymize = server.mock(|when, then| {
            when.path("/anonymize");
            then.status(200).json_body(json!({"text":"[REDACTED]"}));
        });
        server.mock(|when, then| {
            when.path("/analyze").body_contains("second");
            then.status(500);
        });
        let config: PresidioConfig =
            serde_json::from_value(json!({"url":server.base_url()})).unwrap();
        let guard = PresidioGuardrail::new(&config, true).unwrap();
        let mut body = json!({"messages":[
            {"role":"user","content":"first"}, {"role":"user","content":"second"}
        ]});
        let original = body.clone();
        guard.apply_to_body(&mut body).await;
        assert_eq!(
            anonymize.hits(),
            1,
            "first segment must have been redacted in staging"
        );
        assert_eq!(body, original);
    }

    #[tokio::test]
    async fn cancelled_redaction_leaves_the_whole_body_unchanged() {
        use std::sync::Arc;
        let entered_second = Arc::new(tokio::sync::Notify::new());
        let signal = entered_second.clone();
        let app = axum::Router::new()
            .route(
                "/analyze",
                axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                    let signal = signal.clone();
                    async move {
                        if body["text"] == "second" {
                            signal.notify_one();
                            std::future::pending::<()>().await;
                        }
                        axum::Json(json!([{"entity_type":"PERSON","start":0,"end":5,"score":1.0}]))
                    }
                }),
            )
            .route(
                "/anonymize",
                axum::routing::post(|| async { axum::Json(json!({"text":"[REDACTED]"})) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config: PresidioConfig = serde_json::from_value(json!({
            "url":format!("http://{}", listener.local_addr().unwrap())
        }))
        .unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let guard = PresidioGuardrail::new(&config, true).unwrap();
        let mut body = json!({"messages":[
            {"role":"user","content":"first"}, {"role":"user","content":"second"}
        ]});
        let original = body.clone();
        {
            let work = guard.apply_to_body(&mut body);
            tokio::pin!(work);
            tokio::select! {
                _ = &mut work => panic!("second segment must still be pending"),
                ready = tokio::time::timeout(Duration::from_secs(3), entered_second.notified()) => {
                    ready.expect("first redaction must finish before cancellation");
                }
            }
            // Dropping the future is exactly what the outer guardrail timeout does.
        }
        server.abort();
        assert_eq!(body, original);
    }

    #[tokio::test]
    async fn analyze_timeout_after_a_staged_redaction_leaves_the_body_unchanged() {
        // A per-request Presidio timeout on a later segment must discard the
        // redactions already staged in the clone (copy-then-commit) and leave
        // the caller's body byte-for-byte unchanged. Unlike the 500-error and
        // future-drop tests, this exercises the client's own total timeout.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let anonymize_calls = Arc::new(AtomicUsize::new(0));
        let counter = anonymize_calls.clone();
        let app = axum::Router::new()
            .route(
                "/analyze",
                axum::routing::post(|axum::Json(body): axum::Json<Value>| async move {
                    if body["text"] == "second" {
                        // Never answers: the client's total timeout fires.
                        std::future::pending::<()>().await;
                    }
                    axum::Json(json!([{"entity_type":"PERSON","start":0,"end":5,"score":1.0}]))
                }),
            )
            .route(
                "/anonymize",
                axum::routing::post(move || {
                    let counter = counter.clone();
                    async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        axum::Json(json!({"text":"[REDACTED]"}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config: PresidioConfig = serde_json::from_value(json!({
            "url": format!("http://{}", listener.local_addr().unwrap()),
            "timeout_secs": 1
        }))
        .unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let guard = PresidioGuardrail::new(&config, true).unwrap();
        let mut body = json!({"messages":[
            {"role":"user","content":"first"}, {"role":"user","content":"second"}
        ]});
        let original = body.clone();
        guard.apply_to_body(&mut body).await;
        assert_eq!(
            anonymize_calls.load(Ordering::SeqCst),
            1,
            "the first segment must have been redacted in staging before the timeout"
        );
        assert_eq!(
            body, original,
            "a timeout must discard every staged redaction"
        );
        server.abort();
    }

    #[tokio::test]
    async fn apply_to_body_processes_all_messages() {
        let server = MockServer::start();

        // The first message "user@example.com" — return PII entities.
        let analyze_user = server.mock(|when, then| {
            when.method("POST")
                .path("/analyze")
                .body_contains("user@example.com");
            then.status(200).json_body(json!([
                {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17, "score": 0.99}
            ]));
        });
        // The second message "hello" — return no entities.
        let analyze_hello = server.mock(|when, then| {
            when.method("POST").path("/analyze").body_contains("hello");
            then.status(200).json_body(json!([]));
        });

        let anonymize = server.mock(|when, then| {
            when.method("POST").path("/anonymize");
            then.status(200).json_body(json!({"text": "<REDACTED>"}));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "user@example.com"},
                {"role": "assistant", "content": "hello"}
            ]
        });
        guardrail.apply_to_body(&mut body).await;
        assert_eq!(body["messages"][0]["content"], "<REDACTED>");
        assert_eq!(body["messages"][1]["content"], "hello");
        assert_eq!(analyze_user.hits(), 1);
        assert_eq!(analyze_hello.hits(), 1);
        assert_eq!(anonymize.hits(), 1);
    }

    #[tokio::test]
    async fn service_unreachable_falls_through() {
        let config = PresidioConfig {
            enabled: Some(true),
            url: "http://127.0.0.1:1".to_string(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(1),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let result = guardrail.apply("hello").await;
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn custom_endpoints_used_when_configured() {
        let server = MockServer::start();
        let analyze_mock = server.mock(|when, then| {
            when.method("POST").path("/custom/analyze");
            then.status(200).json_body(
                json!([{"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17, "score": 0.99}]),
            );
        });
        let anonymize_mock = server.mock(|when, then| {
            when.method("POST").path("/custom/anonymize");
            then.status(200).json_body(json!({"text": "<REDACTED>"}));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: Some("/custom/analyze".into()),
            anonymize_endpoint: Some("/custom/anonymize".into()),
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        guardrail.apply("test@example.com").await;
        assert_eq!(analyze_mock.hits(), 1);
        assert_eq!(anonymize_mock.hits(), 1);
    }

    #[tokio::test]
    async fn empty_segments_are_not_sent_to_analyze() {
        // Empty/whitespace-only segments would make Presidio return
        // 422 "Invalid input, text can not be empty" — skip them instead.
        let server = MockServer::start();
        let analyze = server.mock(|when, then| {
            when.method("POST").path("/analyze");
            then.status(200).json_body(json!([]));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let mut body = json!({
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "   "},
                {"role": "assistant", "content": null}
            ]
        });
        guardrail.apply_to_body(&mut body).await;
        assert_eq!(analyze.hits(), 0);
        assert_eq!(body["messages"][0]["content"], "");
        assert_eq!(body["messages"][1]["content"], "   ");
        assert_eq!(body["messages"][2]["content"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn apply_to_body_covers_name_tool_calls_and_tools() {
        let server = MockServer::start();

        // Any analyze call reports an email entity spanning the whole input;
        // anonymize replaces it with <REDACTED>.
        let analyze = server.mock(|when, then| {
            when.method("POST").path("/analyze");
            then.status(200).json_body(json!([
                {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17, "score": 0.99}
            ]));
        });
        let anonymize = server.mock(|when, then| {
            when.method("POST").path("/anonymize");
            then.status(200).json_body(json!({"text": "<REDACTED>"}));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let mut body = json!({
            "messages": [{
                "role": "assistant",
                "content": null,
                "name": "john@example.com",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "description": "send to john@example.com",
                        "arguments": "{\"to\": \"john@example.com\", \"note\": \"urgent\"}"
                    }
                }]
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "call john@example.com for help",
                    "parameters": {"type": "object", "properties": {"q": {"description": "mail john@example.com"}}}
                }
            }]
        });
        guardrail.apply_to_body(&mut body).await;

        assert_eq!(body["messages"][0]["name"], "<REDACTED>");
        let func = &body["messages"][0]["tool_calls"][0]["function"];
        assert_eq!(func["description"], "<REDACTED>");
        let args: Value = serde_json::from_str(func["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["to"], "<REDACTED>");
        assert_eq!(args["note"], "<REDACTED>");
        let tool_func = &body["tools"][0]["function"];
        assert_eq!(tool_func["description"], "<REDACTED>");
        assert_eq!(
            tool_func["parameters"]["properties"]["q"]["description"],
            "<REDACTED>"
        );
        assert!(analyze.hits() >= 6);
        assert!(anonymize.hits() >= 6);
    }

    #[tokio::test]
    async fn dedup_collapses_identical_segments() {
        // A body with many identical string leaves under
        // `tools[].function.parameters` must result in a single Presidio
        // analyze call per distinct value, not one per leaf.
        let server = MockServer::start();
        let analyze = server.mock(|when, then| {
            when.method("POST").path("/analyze");
            then.status(200).json_body(json!([]));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: None,
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();

        // 50 identical descriptions under tools[].function.parameters.
        let mut props = serde_json::Map::new();
        for i in 0..50 {
            props.insert(format!("p{i}"), json!({"description": "same text"}));
        }
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "t",
                    "parameters": {"type": "object", "properties": props}
                }
            }]
        });
        guardrail.apply_to_body(&mut body).await;
        // 3 distinct strings: "hi", "same text", and "object" (the
        // parameters `type` field). Dedup collapses 50 identical "same text"
        // descriptions into one call.
        assert_eq!(
            analyze.hits(),
            3,
            "expected 3 distinct segments, got {}",
            analyze.hits()
        );
    }

    #[tokio::test]
    async fn custom_anonymizers_are_passed_through() {
        // When a custom `anonymizers` config is set, it must appear verbatim
        // in the /anonymize request body instead of the default.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/analyze");
            then.status(200).json_body(json!([
                {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17, "score": 0.99}
            ]));
        });
        let anonymize = server.mock(|when, then| {
            when.method("POST")
                .path("/anonymize")
                .body_contains("\"EMAIL_ADDRESS\"");
            then.status(200).json_body(json!({"text": "[MASKED]"}));
        });

        let config = PresidioConfig {
            enabled: Some(true),
            url: server.base_url(),
            language: None,
            analyze_endpoint: None,
            anonymize_endpoint: None,
            anonymizers: Some(json!({
                "DEFAULT": {"type": "replace", "new_value": "<REDACTED>"},
                "EMAIL_ADDRESS": {"type": "mask"}
            })),
            timeout_secs: Some(10),
        };
        let guardrail = PresidioGuardrail::new(&config, true).unwrap();
        let result = guardrail.apply("user@example.com").await;
        assert_eq!(result, "[MASKED]");
        assert_eq!(anonymize.hits(), 1);
    }
}
