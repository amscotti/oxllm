//! Shared payload-redaction helpers for telemetry/exporter hooks.
//!
//! Every external exporter (Langfuse, Langsmith, Helicone, Datadog, MLflow,
//! W&B, S3, generic callback) ships request/response bodies to a third-party
//! SaaS or to long-term object storage. Prompt bodies routinely embed
//! credentials, API keys, and PII; redacting well-known credential-bearing
//! keys before the payload leaves the process is the lowest-cost defense
//! against accidental leakage.
//!
//! Two passes are available:
//!
//! - **Key-name denylist** (unconditional): object keys matching
//!   [`SENSITIVE_BODY_KEYS`] case-insensitively are replaced with
//!   `"[redacted]"`. Values are never inspected.
//! - **Value-pattern masking** (opt-in via [`redact_secret_values_in_place`]):
//!   string values anywhere in the tree are scanned for well-known secret
//!   shapes (`sk-...`, `AKIA...`, `Bearer ...`, `ghp_...`, `xoxb-...`) and
//!   matching substrings are replaced with `"[redacted]"`. This is gated
//!   behind a config flag (`redact_secret_values`) so prompt content that
//!   legitimately discusses these patterns is preserved unless the operator
//!   opts in.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

/// HTTP-header-style keys whose values are stripped from logged request and
/// response bodies before being sent to any external telemetry backend. The
/// list covers the credential-bearing field names that gateway-compatible
/// agent / tool-using workloads routinely embed in structured prompts (config
/// objects, tool I/O, bearer-token challenge responses, ...). Matching is
/// exact (case-insensitive) on the object KEY only — values are never
/// inspected, so a prompt that merely discusses these words in a content
/// string is preserved verbatim.
pub const SENSITIVE_BODY_KEYS: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "bearer",
    "session_id",
    "session",
    "cookie",
    "set-cookie",
    "private_key",
    "client_secret",
    "id_token",
    "openai_api_key",
    "anthropic_api_key",
    "litellm_api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "stripe_key",
    "github_token",
    "secret_access_key",
    "access_key_id",
    "private_token",
    "auth_token",
];

/// Compiled regex patterns matching well-known secret value shapes. Each
/// pattern is applied to string values throughout the JSON tree by
/// [`redact_secret_values_in_place`]. Matching substrings are replaced with
/// `"[redacted]"`.
static SECRET_VALUE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        // OpenAI-style keys: sk-[A-Za-z0-9]{20,}
        Regex::new(r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}").expect("valid regex"),
        // AWS access key ids: AKIA[0-9A-Z]{16}
        Regex::new(r"AKIA[0-9A-Z]{16}").expect("valid regex"),
        // Bearer tokens: Bearer <token>
        Regex::new(r"(?i)Bearer\s+\S+").expect("valid regex"),
        // Slack bot tokens: xoxb-...
        Regex::new(r"xox[bpoa]-[A-Za-z0-9-]{10,}").expect("valid regex"),
        // GitHub personal access tokens: ghp_..., gho_..., ghs_...
        Regex::new(r"gh[pousr]_[A-Za-z0-9]{36,}").expect("valid regex"),
    ]
});

/// Return a copy of `value` with every object key matching (case-insensitively)
/// an entry in [`SENSITIVE_BODY_KEYS`] replaced by `"[redacted]"`, recursing
/// into nested objects and arrays.
pub fn redact_sensitive(value: Value) -> Value {
    let mut v = value;
    redact_sensitive_in_place(&mut v);
    v
}

/// In-place variant of [`redact_sensitive`] for callers that already own the
/// `Value` (e.g. a freshly cloned response body).
pub fn redact_sensitive_in_place(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let keys_to_redact: Vec<String> = map
                .keys()
                .filter(|k| {
                    SENSITIVE_BODY_KEYS
                        .iter()
                        .any(|s| s.eq_ignore_ascii_case(k))
                })
                .cloned()
                .collect();
            for k in keys_to_redact {
                map.insert(k, Value::String("[redacted]".to_string()));
            }
            for v in map.values_mut() {
                redact_sensitive_in_place(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                redact_sensitive_in_place(v);
            }
        }
        _ => {}
    }
}

/// Opt-in value-pattern redaction: scans all string values in the JSON tree
/// for well-known secret shapes (`sk-...`, `AKIA...`, `Bearer ...`, `xoxb-...`,
/// `ghp_...`) and replaces matching substrings with `"[redacted]"`. Unlike
/// [`redact_sensitive_in_place`], which only inspects object **keys**, this
/// pass inspects **values** and is therefore gated behind a config flag
/// (`redact_secret_values`, default off) so prompt content that legitimately
/// discusses these patterns is preserved unless the operator opts in.
pub fn redact_secret_values_in_place(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for v in map.values_mut() {
                redact_secret_values_in_place(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                redact_secret_values_in_place(v);
            }
        }
        Value::String(s) => {
            let mut masked = s.clone();
            for re in SECRET_VALUE_PATTERNS.iter() {
                masked = re.replace_all(&masked, "[redacted]").to_string();
            }
            if masked != *s {
                *s = masked;
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_top_level_authorization() {
        let v = json!({ "authorization": "Bearer sk-x", "keep": "ok" });
        let r = redact_sensitive(v);
        assert_eq!(r["authorization"], "[redacted]");
        assert_eq!(r["keep"], "ok");
    }

    #[test]
    fn redacts_all_known_keys_case_insensitively() {
        let v = json!({
            "X-API-KEY": "a",
            "Api-Key": "b",
            "api_key": "c",
            "apikey": "d",
            "authorization": "e",
            "PASSWORD": "f",
            "Secret": "g",
            "Token": "h",
            "access_token": "i",
            "refresh_token": "j",
            "BEARER": "k",
            "session_id": "l",
            "SESSION": "m",
            "Cookie": "n",
            "set-cookie": "o",
            "private_key": "p",
            "client_secret": "q",
            "id_token": "r"
        });
        let r = redact_sensitive(v);
        for k in [
            "X-API-KEY",
            "Api-Key",
            "api_key",
            "apikey",
            "authorization",
            "PASSWORD",
            "Secret",
            "Token",
            "access_token",
            "refresh_token",
            "BEARER",
            "session_id",
            "SESSION",
            "Cookie",
            "set-cookie",
            "private_key",
            "client_secret",
            "id_token",
        ] {
            assert_eq!(r[k], "[redacted]", "key {k} should be redacted");
        }
    }

    /// MEDIUM #2 (regression guard): a prompt payload that carries common
    /// credential-bearing object keys (typical of agent / tool-using
    /// workloads passing config objects as messages) must have each of those
    /// keys' values replaced by `"[redacted]"` before the payload leaves the
    /// process via any telemetry/exporter hook.
    #[test]
    fn redacts_structured_credential_payload() {
        let v = json!({
            "tool_input": {
                "password": "hunter2",
                "session_id": "sess-abc",
                "cookie": "session=xyz",
                "bearer": "Bearer abc123",
                "client_secret": "shh",
            },
            "messages": [{"role": "user", "content": "talk about tokens"}]
        });
        let r = redact_sensitive(v);
        for k in [
            "password",
            "session_id",
            "cookie",
            "bearer",
            "client_secret",
        ] {
            assert_eq!(
                r["tool_input"][k], "[redacted]",
                "credential key {k} must be redacted in nested objects"
            );
        }
        // Content that merely *discusses* tokens is preserved verbatim.
        assert_eq!(
            r["messages"][0]["content"], "talk about tokens",
            "non-key occurrences of sensitive words must be preserved"
        );
    }

    #[test]
    fn redacts_nested_in_arrays_and_objects() {
        let v = json!({
            "data": [{"api_key": "sk-inner"}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let r = redact_sensitive(v);
        assert_eq!(r["data"][0]["api_key"], "[redacted]");
        assert_eq!(r["messages"][0]["content"], "hi");
    }

    #[test]
    fn leaves_non_object_values_untouched() {
        let v = json!(["api_key", 42, "Bearer x"]);
        let r = redact_sensitive(v);
        assert_eq!(r, json!(["api_key", 42, "Bearer x"]));
    }

    /// MEDIUM #3: common provider-prefixed credential key names
    /// (`openai_api_key`, `aws_secret_access_key`, ...) that agent /
    /// tool-using workloads routinely embed must be caught by the
    /// denylist alongside the original 17 names.
    #[test]
    fn redacts_provider_prefixed_credential_keys() {
        let v = json!({
            "openai_api_key": "sk-x",
            "anthropic_api_key": "sk-ant-y",
            "aws_secret_access_key": "z",
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_session_token": "t",
            "stripe_key": "sk_live_z",
            "github_token": "ghp_x",
            "secret_access_key": "s",
            "access_key_id": "a",
        });
        let r = redact_sensitive(v);
        for k in [
            "openai_api_key",
            "anthropic_api_key",
            "aws_secret_access_key",
            "aws_access_key_id",
            "aws_session_token",
            "stripe_key",
            "github_token",
            "secret_access_key",
            "access_key_id",
        ] {
            assert_eq!(r[k], "[redacted]", "key {k} should be redacted");
        }
    }

    /// MEDIUM #3 (value-pattern pass): a secret living under a
    /// non-matching key (e.g. `{"key": "sk-..."}`) is masked by the opt-in
    /// value-pattern pass but preserved when the pass is not applied.
    #[test]
    fn value_pattern_redaction_masks_openai_key_under_neutral_key() {
        let v = json!({
            "key": "sk-abcdefghijklmnopqrstuvwxyz1234567890",
            "label": "my-api-key"
        });
        let mut r = v.clone();
        redact_secret_values_in_place(&mut r);
        assert!(
            r["key"].as_str().unwrap().contains("[redacted]"),
            "sk-... value must be masked, got: {}",
            r["key"]
        );
        assert!(
            !r["key"]
                .as_str()
                .unwrap()
                .contains("sk-abcdefghijklmnopqrstuvwxyz"),
            "raw secret must not survive"
        );
    }

    #[test]
    fn value_pattern_redaction_masks_aws_bearer_and_github() {
        let v = json!({
            "a": "AKIAIOSFODNN7EXAMPLE",
            "b": "Bearer abc123def456",
            "c": "ghp_1234567890abcdefghijklmnopqrstuvwxyz1234",
            "d": "xoxb-1234567890-abcdefghijklmnop",
        });
        let mut r = v;
        redact_secret_values_in_place(&mut r);
        assert!(r["a"].as_str().unwrap().contains("[redacted]"));
        assert!(r["b"].as_str().unwrap().contains("[redacted]"));
        assert!(r["c"].as_str().unwrap().contains("[redacted]"));
        assert!(r["d"].as_str().unwrap().contains("[redacted]"));
    }

    #[test]
    fn value_pattern_redaction_preserves_non_secret_strings() {
        let v = json!({
            "message": "Please use the sk- prefix for your API key",
            "model": "gpt-4o"
        });
        let mut r = v.clone();
        redact_secret_values_in_place(&mut r);
        // Short "sk-" mentions (< 20 chars after prefix) are not matched.
        assert_eq!(r["message"], "Please use the sk- prefix for your API key");
        assert_eq!(r["model"], "gpt-4o");
    }
}
