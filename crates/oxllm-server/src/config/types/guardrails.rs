//! Pre-call guardrail configuration: PII masking, prompt-injection
//! detection, Presidio, AIM, Lakera, Bedrock, Model Armor, custom guards,
//! semantic guards, and MCP-backed guards.

use serde::{Deserialize, Serialize};

/// Top-level guardrails container, mounted at
/// `general_settings.guardrails`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardrailsConfig {
    /// Rule-based PII masking (email / phone / credit card / SSN / API key).
    #[serde(default)]
    pub pii_masking: Option<PiiMaskingConfig>,
    /// Heuristic prompt-injection detection.
    #[serde(default)]
    pub prompt_injection: Option<PromptInjectionConfig>,
    /// Presidio PII redaction service (remote API).
    #[serde(default)]
    pub presidio: Option<PresidioConfig>,
    /// AIM Security prompt-injection detection (remote API).
    #[serde(default)]
    pub aim: Option<AimConfig>,
    /// Lakera AI prompt-injection detection (remote API).
    #[serde(default)]
    pub lakera: Option<LakeraConfig>,
    /// AWS Bedrock Guardrails (remote SigV4-signed API).
    #[serde(default)]
    pub bedrock_guardrail: Option<BedrockGuardrailConfig>,
    /// Google Cloud Model Armor prompt-injection detection (remote API).
    #[serde(default)]
    pub model_armor: Option<ModelArmorConfig>,
    /// User-defined custom guardrails (`regex` / `keyword` / `json_path` /
    /// `http`). Each entry is compiled into a [`crate::guardrails::CustomGuardrail`]
    /// and applied in the standard guardrail chain. See [`CustomGuardrailConfig`].
    #[serde(default)]
    pub custom_guardrails: Option<Vec<CustomGuardrailConfig>>,
    /// Semantic (similarity-based) blocked-phrase guard. Uses a local MinHash
    /// approximation of n-gram Jaccard similarity rather than embedding calls
    /// (the guardrail path is synchronous and pre-upstream, so per-request
    /// embedding round-trips are impractical). See [`SemanticGuardConfig`].
    #[serde(default)]
    pub semantic_guard: Option<SemanticGuardConfig>,
    /// MCP-server-backed security scanning guardrails. Each entry posts the
    /// request text to an MCP server's `tools/call` endpoint (JSON-RPC over
    /// HTTP) and blocks when the server returns `blocked: true`. See
    /// [`McpGuardrailConfig`].
    #[serde(default)]
    pub mcp_guardrails: Option<Vec<McpGuardrailConfig>>,
}

/// Configuration for the Presidio PII redaction API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresidioConfig {
    /// Master switch. The guardrail is only constructed when
    /// `enabled: true`; mere presence of this section is not enough.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Presidio service URL (e.g. `http://localhost:5000`).
    pub url: String,
    /// Language sent with every `/analyze` request (default `"en"`).
    /// Presidio's analyzer requires this field; requests without it are
    /// rejected with 400.
    #[serde(default)]
    pub language: Option<String>,
    /// Analyzer endpoint path (default `/analyze`).
    #[serde(default)]
    pub analyze_endpoint: Option<String>,
    /// Anonymizer endpoint path (default `/anonymize`).
    #[serde(default)]
    pub anonymize_endpoint: Option<String>,
    /// HTTP request timeout in seconds (default 5, minimum 1).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Custom anonymizer configuration passed through to Presidio's
    /// `/anonymize` endpoint. When unset, defaults to
    /// `{"DEFAULT":{"type":"replace","new_value":"<REDACTED>"}}`. Operators
    /// can use this to configure per-entity anonymizers (e.g.
    /// `{"EMAIL_ADDRESS":{"type":"mask"}}`).
    #[serde(default)]
    pub anonymizers: Option<serde_json::Value>,
}

/// Configuration for the AIM Security prompt-injection detection API.
///
/// AIM Security's "firewall" endpoint (`/fw/v1/analyze`) is policy-driven
/// rather than score-driven: the response carries a
/// `required_action.action_type` of `"monitor_action"`, `"block_action"`,
/// or `"anonymize_action"`. Oxllm blocks when the policy reports
/// `block_action` (matching the verified reference integration).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AimConfig {
    /// Master switch. The guardrail is only constructed when
    /// `enabled: true`; mere presence of this section is not enough.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// AIM Security service URL (e.g. `https://api.aim.security`).
    pub url: String,
    /// API key for authentication.
    pub api_key: String,
    /// **Deprecated:** risk score threshold (0.0–1.0, default 0.7). The
    /// firewall API is policy-driven and does not return a score, so this
    /// field no longer affects blocking. It is retained for backward compat
    /// with existing configs and validated for range only.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// HTTP request timeout in seconds (default 5, minimum 1).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

/// Configuration for the Lakera AI prompt-injection detection API.
///
/// Lakera Guard calls `POST {endpoint}/v2/guard` with the chat messages and
/// blocks when the response reports `flagged: true` (matching the reference v2
/// `lakera_ai_v2.LakeraAIResponse` shape and `if lakera_response.get("flagged")
/// is True` semantics). Fail-open: on service errors the request is allowed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LakeraConfig {
    /// Master switch. Defaults to `true` when the section is present.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// API key for authentication.
    pub api_key: String,
    /// Lakera service base URL (default `https://api.lakera.ai`).
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Risk score threshold (0.0–1.0, default 0.5). **Deprecated:** the v2
    /// Lakera API does not return a score, so blocking is based solely on
    /// `flagged: true`. The field is retained for backward-compat with
    /// existing configs and validated for range only.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// HTTP request timeout in seconds (default 5, minimum 1).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Action when Lakera flags a request: `"block"` (default) rejects with
    /// 400; `"monitor"` logs the violation server-side and allows the request
    /// (matching the reference v2 `on_flagged: "monitor"` at
    /// `lakera_ai_v2.py:261`). Useful for gradual rollouts where operators
    /// want to observe detections before enforcing blocks.
    #[serde(default)]
    pub mode: Option<String>,
}

/// Configuration for the AWS Bedrock Guardrails integration.
///
/// Uses the Bedrock `ApplyGuardrail` API, signed with AWS SigV4. AWS
/// credentials are read from the standard environment variables
/// (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`).
/// Fail-open: on service errors the request is allowed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BedrockGuardrailConfig {
    /// Master switch. Defaults to `true` when the section is present.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// The Bedrock guardrail identifier.
    pub guardrail_id: String,
    /// The Bedrock guardrail version (e.g. `"1"` or `"DRAFT"`).
    pub guardrail_version: String,
    /// AWS region (default `us-east-1`).
    #[serde(default)]
    pub region: Option<String>,
    /// HTTP request timeout in seconds (default 5, minimum 1).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

/// Configuration for the Google Cloud Model Armor prompt-injection detection
/// API.
///
/// Calls `POST {endpoint}` with `{ messages: [...] }` and blocks when the
/// response reports `blocked: true` or the `risk_score` clears the configured
/// threshold. Fail-open: on service errors the request is allowed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArmorConfig {
    /// Master switch. Defaults to `true` when the section is present.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Model Armor service endpoint (full URL).
    pub endpoint: String,
    /// Optional API key for authentication (sent as `Bearer` token).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Risk score threshold (0.0–1.0, default 0.5). Scores at or above this
    /// value cause the request to be blocked (in addition to `blocked: true`).
    #[serde(default)]
    pub threshold: Option<f64>,
    /// HTTP request timeout in seconds (default 5, minimum 1).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

/// Configuration for the PII masking guardrail.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PiiMaskingConfig {
    /// Master switch.
    #[serde(default)]
    pub enabled: bool,
    /// PII types to leave unmasked. Recognized slugs: `"email"`, `"phone"`,
    /// `"credit_card"`, `"ssn"`, `"api_key"`.
    #[serde(default)]
    pub exclude: Option<Vec<String>>,
}

/// Configuration for the prompt-injection detection guardrail.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptInjectionConfig {
    /// Master switch.
    #[serde(default)]
    pub enabled: bool,
    /// `"block"` (default) rejects the request with 400; `"log"` allows the
    /// request through while emitting a `tracing::warn!`.
    #[serde(default)]
    pub action: Option<String>,
    /// Minimum confidence (0.0–1.0) required to trigger. Defaults to `0.7`.
    #[serde(default)]
    pub threshold: Option<f64>,
}

/// One user-defined custom guardrail, declared under
/// `general_settings.guardrails.custom_guardrails`.
///
/// The `config` payload is an arbitrary JSON object whose shape depends on
/// `type` (parsed by [`crate::guardrails::CustomGuardrail::from_config`]):
///
/// - `regex`     — `{ patterns: [{ pattern, action, replacement? }], scope? }`
/// - `keyword`   — `{ blocklist: [], allowlist: [], action? }`
/// - `json_path` — `{ rules: [{ path, operator, value?, action? }] }`
/// - `http`      — `{ url, method?, headers?, timeout_secs?, block_on: { field, value }, fail_open? }`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomGuardrailConfig {
    /// Human-readable name used in logs.
    pub name: String,
    /// Guardrail kind: `"regex"`, `"keyword"`, `"json_path"`, or `"http"`.
    #[serde(rename = "type")]
    pub guardrail_type: String,
    /// Type-specific configuration (an arbitrary JSON object).
    #[serde(default)]
    pub config: serde_json::Value,
}

/// Semantic-similarity blocked-phrase guard configuration, declared under
/// `general_settings.guardrails.semantic_guard`.
///
/// **Design note:** true semantic similarity requires an embedding round-trip
/// per request, which is impractical in the synchronous, pre-upstream
/// guardrail path. The implementation therefore uses a local MinHash
/// approximation of n-gram Jaccard similarity (no network). Set
/// `embedding_model` for forward-compatibility; it is currently unused by the
/// local approximation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticGuardConfig {
    /// Master switch. Defaults to `true` when the section is present.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Phrases whose similarity to the user message triggers a block.
    #[serde(default)]
    pub blocked_phrases: Vec<String>,
    /// Jaccard-similarity cutoff in `[0.0, 1.0]` (default `0.6`). A request is
    /// blocked when its estimated similarity to any blocked phrase is `>=`
    /// this value.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Embedding model slug. Recorded for future use; the local MinHash
    /// approximation does not call it.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// When `true` (default), any internal error allows the request through.
    #[serde(default)]
    pub fail_open: Option<bool>,
}

/// One MCP-server-backed guardrail, declared under
/// `general_settings.guardrails.mcp_guardrails`.
///
/// The proxy sends the request text to the MCP server's `tools/call` endpoint
/// as a JSON-RPC `tools/call` request and blocks when the server returns
/// `{ "result": { "blocked": true } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpGuardrailConfig {
    /// Human-readable name used in logs.
    pub name: String,
    /// MCP server base URL (JSON-RPC over HTTP).
    pub server_url: String,
    /// Tool name to invoke (default `"scan_prompt"`).
    #[serde(default)]
    pub tool_name: Option<String>,
    /// HTTP request timeout in seconds (default 5, minimum 1).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// When `true` (default), network/parse errors allow the request through.
    #[serde(default)]
    pub fail_open: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guardrail_container_rejects_misspelled_sections() {
        let error = serde_json::from_value::<GuardrailsConfig>(serde_json::json!({
            "prompt_injecton": {"enabled": true}
        }))
        .unwrap_err();
        assert!(error.to_string().contains("prompt_injecton"));
        assert!(
            serde_json::from_value::<GuardrailsConfig>(serde_json::json!({
                "prompt_injection": {"enabled": true}
            }))
            .unwrap()
            .prompt_injection
            .unwrap()
            .enabled
        );
    }

    #[test]
    fn nested_yaml_guardrail_typos_fail_proxy_config_deserialization() {
        let error = serde_yml::from_str::<crate::config::ProxyConfig>(
            "model_list: []\ngeneral_settings:\n  guardrails:\n    prompt_injecton:\n      enabled: true\n",
        ).unwrap_err();
        assert!(error.to_string().contains("prompt_injecton"));
    }
}
