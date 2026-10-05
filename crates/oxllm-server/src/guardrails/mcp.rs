//! MCP (Model Context Protocol) security scanning guardrail.
//!
//! Posts the request text to an MCP server's JSON-RPC `tools/call` endpoint
//! and blocks when the server returns `{ "result": { "blocked": true } }`.
//! This is a thin specialization of the `http` custom-guardrail pattern with
//! MCP-specific request/response shaping.
//!
//! Request:
//! ```jsonc
//! {
//!   "jsonrpc": "2.0", "id": 1,
//!   "method": "tools/call",
//!   "params": { "name": "scan_prompt", "arguments": { "content": "<text>" } }
//! }
//! ```
//! Response (inside the JSON-RPC `result`):
//! ```jsonc
//! { "blocked": true, "reason": "prompt injection detected" }
//! ```
//!
//! Fail-open: on network/parse errors the request is allowed through (unless
//! `fail_open: false`).

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::types::McpGuardrailConfig;
use crate::guardrails::GuardrailResult;
use crate::guardrails::custom::all_text;

/// Default tool name invoked on the MCP server.
const DEFAULT_TOOL: &str = "scan_prompt";
/// Fixed JSON-RPC request id. The proxy does not multiplex calls, so a
/// constant id is sufficient and keeps responses easy to inspect.
const JSONRPC_ID: i64 = 1;

/// JSON-RPC `tools/call` request payload.
#[derive(Debug, Serialize)]
struct McpRequest {
    jsonrpc: &'static str,
    id: i64,
    method: &'static str,
    params: McpParams,
}

#[derive(Debug, Serialize)]
struct McpParams {
    name: String,
    arguments: McpArguments,
}

#[derive(Debug, Serialize)]
struct McpArguments {
    content: String,
}

/// The verdict returned inside the JSON-RPC `result`.
#[derive(Debug, Deserialize)]
struct McpVerdict {
    blocked: bool,
    #[serde(default)]
    reason: Option<String>,
}

/// The JSON-RPC envelope.
#[derive(Debug, Deserialize)]
struct McpResponse {
    #[serde(default)]
    result: Option<McpVerdict>,
    /// An MCP server may return a JSON-RPC error object instead of a result.
    #[serde(default)]
    error: Option<Value>,
}

/// An MCP-server-backed security guardrail.
#[derive(Debug, Clone)]
pub struct McpGuardrail {
    pub name: String,
    client: reqwest::Client,
    url: String,
    tool_name: String,
    fail_open: bool,
    host: String,
    allow_private: bool,
}

impl McpGuardrail {
    pub(crate) fn is_fail_closed(&self) -> bool {
        !self.fail_open
    }

    /// Build from explicit parameters. Performs no SSRF validation (mirrors
    /// the Lakera/AIM pattern); SSRF is enforced by the caller at config-build
    /// time and re-checked via [`crate::security::is_private_host_async`] at
    /// request time when `allow_private` is `false`. The HTTP client carries
    /// the connect-time [`crate::security::SsrfResolver`] unless
    /// `allow_private` is set, so a DNS-rebinding attack after config load
    /// cannot steer the connection to a private address.
    pub fn new(
        name: &str,
        server_url: String,
        tool_name: Option<String>,
        timeout_secs: Option<u64>,
        fail_open: bool,
        allow_private: bool,
    ) -> Result<Self, String> {
        let client = crate::security::build_ssrf_http_client(
            allow_private,
            Duration::from_secs(timeout_secs.unwrap_or(5).max(1)),
        )
        .map_err(|e| format!("mcp guardrail '{name}': HTTP client build failed: {e}"))?;
        let url = server_url.trim_end_matches('/').to_string();
        let host = url::Url::parse(&url)
            .ok()
            .and_then(|u| u.host_str().map(String::from))
            .unwrap_or_default();
        Ok(Self {
            name: name.to_string(),
            client,
            url: format!("{url}/mcp"),
            tool_name: tool_name.unwrap_or_else(|| DEFAULT_TOOL.to_string()),
            fail_open,
            host,
            allow_private,
        })
    }

    /// Build from a [`McpGuardrailConfig`].
    pub fn from_config(config: &McpGuardrailConfig, allow_private: bool) -> Result<Self, String> {
        Self::new(
            &config.name,
            config.server_url.clone(),
            config.tool_name.clone(),
            config.timeout_secs,
            config.fail_open.unwrap_or(true),
            allow_private,
        )
    }

    /// Forward `content` to the MCP server and parse the verdict.
    async fn scan(&self, content: &str) -> anyhow::Result<McpResponse> {
        let req = McpRequest {
            jsonrpc: "2.0",
            id: JSONRPC_ID,
            method: "tools/call",
            params: McpParams {
                name: self.tool_name.clone(),
                arguments: McpArguments {
                    content: content.to_string(),
                },
            },
        };
        let response = self.client.post(&self.url).json(&req).send().await?;
        super::response::read_json(response).await
    }

    /// Block when the MCP server reports `blocked: true`. Fail-open otherwise.
    pub async fn apply(&self, body: &mut Value) -> GuardrailResult {
        if !self.allow_private
            && !self.host.is_empty()
            && crate::security::is_private_host_async(&self.host).await
        {
            tracing::warn!(
                guardrail = %self.name,
                host = %self.host,
                "mcp guardrail target resolves to private host; skipping send"
            );
            return self.allow_or_close();
        }
        let text = all_text(body);
        match self.scan(&text).await {
            Ok(resp) => {
                if let Some(err) = resp.error {
                    tracing::warn!(
                        guardrail = %self.name,
                        error = ?err,
                        "mcp guardrail: server returned JSON-RPC error"
                    );
                    return self.allow_or_close();
                }
                match resp.result {
                    Some(v) if v.blocked => {
                        tracing::warn!(
                            guardrail = %self.name,
                            reason = ?v.reason,
                            "mcp guardrail blocked request"
                        );
                        GuardrailResult::Block("request blocked by content policy".to_string())
                    }
                    Some(_) => GuardrailResult::Allow,
                    None => self.allow_or_close(),
                }
            }
            Err(e) => {
                if self.fail_open {
                    tracing::warn!(
                        guardrail = %self.name,
                        error = %e,
                        "mcp guardrail: scan failed; allowing (fail_open)"
                    );
                    GuardrailResult::Allow
                } else {
                    GuardrailResult::Block("request blocked: guardrail unavailable".to_string())
                }
            }
        }
    }

    fn allow_or_close(&self) -> GuardrailResult {
        if self.fail_open {
            GuardrailResult::Allow
        } else {
            GuardrailResult::Block("request blocked: guardrail unavailable".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    fn mcp_response_payload(blocked: bool, reason: Option<&str>) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("blocked".to_string(), json!(blocked));
        if let Some(r) = reason {
            m.insert("reason".to_string(), json!(r));
        }
        json!({ "jsonrpc": "2.0", "id": 1, "result": Value::Object(m) })
    }

    #[test]
    fn request_envelope_shape() {
        let req = McpRequest {
            jsonrpc: "2.0",
            id: JSONRPC_ID,
            method: "tools/call",
            params: McpParams {
                name: "scan_prompt".to_string(),
                arguments: McpArguments {
                    content: "hi".to_string(),
                },
            },
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["method"], "tools/call");
        assert_eq!(v["params"]["name"], "scan_prompt");
        assert_eq!(v["params"]["arguments"]["content"], "hi");
    }

    #[test]
    fn verdict_parses_blocked() {
        let v: McpVerdict =
            serde_json::from_value(json!({"blocked": true, "reason": "injection"})).unwrap();
        assert!(v.blocked);
        assert_eq!(v.reason.as_deref(), Some("injection"));
    }

    #[test]
    fn verdict_requires_explicit_boolean() {
        for value in [
            json!({}),
            json!({"blocked": null}),
            json!({"blocked": "false"}),
        ] {
            assert!(serde_json::from_value::<McpVerdict>(value).is_err());
        }
    }

    #[tokio::test]
    async fn blocks_when_server_reports_blocked() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/mcp");
            then.status(200)
                .json_body(mcp_response_payload(true, Some("bad")));
        });
        let g = McpGuardrail::new("m", server.base_url(), None, None, true, true).unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "bad"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn allows_when_server_reports_clean() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/mcp");
            then.status(200)
                .json_body(mcp_response_payload(false, None));
        });
        let g = McpGuardrail::new("m", server.base_url(), None, None, true, true).unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "ok"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn fail_open_on_5xx() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/mcp");
            then.status(500);
        });
        let g = McpGuardrail::new("m", server.base_url(), None, None, true, true).unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn fail_closed_blocks_on_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/mcp");
            then.status(500);
        });
        let g = McpGuardrail::new("m", server.base_url(), None, None, false, true).unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn missing_invalid_and_oversized_verdicts_honor_fail_closed() {
        let responses = [
            "{}".to_string(),
            // A well-formed JSON-RPC envelope with *neither* result nor error:
            // the guardrail cannot learn a verdict and must honor fail_open.
            r#"{"jsonrpc":"2.0","id":1}"#.to_string(),
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#.to_string(),
            r#"{"result":{"blocked":"false"}}"#.to_string(),
            json!({"result":{"blocked":false,"padding":"x".repeat(super::super::response::MAX_RESPONSE_BYTES)}}).to_string(),
        ];
        for response in responses {
            let server = MockServer::start();
            server.mock(|when, then| {
                when.method(POST).path("/mcp");
                then.status(200).body(&response);
            });
            for fail_open in [false, true] {
                let guard =
                    McpGuardrail::new("m", server.base_url(), None, None, fail_open, true).unwrap();
                let mut body = json!({"messages":[{"role":"user","content":"hello"}]});
                let result = guard.apply(&mut body).await;
                if fail_open {
                    assert!(
                        matches!(result, GuardrailResult::Allow),
                        "response {response} must fail open, got {result:?}"
                    );
                } else {
                    assert!(
                        matches!(result, GuardrailResult::Block(_)),
                        "response {response} must fail closed, got {result:?}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn jsonrpc_error_fails_open() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/mcp");
            then.status(200).json_body(json!({
                "jsonrpc": "2.0", "id": 1,
                "error": {"code": -32601, "message": "method not found"}
            }));
        });
        let g = McpGuardrail::new("m", server.base_url(), None, None, true, true).unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn default_tool_name_is_scan_prompt() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST).path("/mcp").body_contains("scan_prompt");
            then.status(200)
                .json_body(mcp_response_payload(false, None));
        });
        let g = McpGuardrail::new("m", server.base_url(), None, None, true, true).unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
        let _ = g.apply(&mut body).await;
        m.assert();
    }

    #[tokio::test]
    async fn custom_tool_name_is_used() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST).path("/mcp").body_contains("custom_scan");
            then.status(200)
                .json_body(mcp_response_payload(false, None));
        });
        let g = McpGuardrail::new(
            "m",
            server.base_url(),
            Some("custom_scan".to_string()),
            None,
            true,
            true,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
        let _ = g.apply(&mut body).await;
        m.assert();
    }

    /// C05/MEDIUM: a guardrail targeting a private literal host with
    /// `allow_private = false` must skip the send entirely (fail-open) at
    /// request time — the request-time re-check does a real resolution rather
    /// than a cache-only, fail-open lookup.
    #[tokio::test]
    async fn skips_private_literal_host_without_allow_private() {
        let g = McpGuardrail::new(
            "m",
            "http://127.0.0.1:9".to_string(),
            None,
            None,
            true,
            false,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
        assert!(
            matches!(g.apply(&mut body).await, GuardrailResult::Allow),
            "a private-literal MCP target must be skipped (fail-open) when allow_private is false"
        );
    }
}
