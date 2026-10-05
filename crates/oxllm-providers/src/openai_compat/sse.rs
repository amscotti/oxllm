//! Shared OpenAI Server-Sent-Event line parser.
//!
//! Every OpenAI-compatible upstream (Groq, xAI, Together, Gemini's OpenAI
//! shim, Mistral, …) emits SSE chunks in the identical shape:
//!
//! ```text
//! data: { …json… }\n\n
//! data: [DONE]\n\n
//! ```
//!
//! Rather than copy-pasting the parser into each provider, it lives here and
//! is reused by [`super::OpenAiCompatProvider`] (and the reference
//! [`crate::openai::OpenAiProvider`] where it still inlines the same logic).

use oxllm_core::Result;
use serde_json::Value;

/// Parse one raw SSE chunk into a JSON value.
///
/// Returns:
/// - `Ok(None)` for empty input or the terminal `data: [DONE]` marker.
/// - `Ok(Some(value))` for a parseable JSON payload (after stripping the
///   optional `data: ` / `data:` prefix).
/// - `Err` when the payload is present but not valid JSON.
pub(crate) fn parse_openai_stream_chunk(chunk: &str) -> Result<Option<Value>> {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    // Walk the event block field-by-field per the SSE spec (HTML §9.2.6):
    // keep-alive comments (`:`) and field lines other than `data` (`event:`,
    // `id:`, `retry:`, unknown) are metadata and ignored; only `data:` values
    // are dispatched, joined with `\n` when several appear in one block. This
    // preserves the single-line OpenAI shape while handling multi-line blocks.
    // Lines are split on any of `\n`, `\r\n`, `\r` so that events framed with
    // any of the three SSE line-ending styles — including the lone-CR style
    // the host framer (`oxllm_server::sse::pop_sse_event`) recognises — parse
    // identically. (`str::lines` splits only on `\n` and silently drops
    // multi-field lone-CR events.)
    let mut data_parts: Vec<&str> = Vec::new();
    for line in trimmed.split(['\n', '\r']) {
        // Trim per-line leading/trailing whitespace before the field-name check
        // so a stray space injected by an upstream proxy (or a padded shim)
        // cannot hide a `data:` line. Parity with the sibling framer
        // `oxllm_server::sse::extract_sse_data`.
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("data:") {
            // A single leading space after the colon is framing, not value.
            data_parts.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    // Single `data:` line is the common case — borrow it directly to avoid an
    // allocation; only the rare multi-`data` block needs to own a join.
    let joined;
    let json_str: &str = match data_parts.len() {
        0 => {
            // No `data:` framing. A bare JSON object (some shims emit one
            // without the prefix) is parsed directly; any other metadata-only
            // block carries no payload.
            if !trimmed.starts_with('{') {
                return Ok(None);
            }
            trimmed
        }
        1 => data_parts[0],
        _ => {
            joined = data_parts.join("\n");
            &joined
        }
    };
    if json_str.trim().is_empty() {
        return Ok(None);
    }
    // `[DONE]` is checked on the joined form so `data: [DONE]`, `data:[DONE]`,
    // and a bare `[DONE]`-only data line all terminate the stream.
    if json_str.trim() == "[DONE]" {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(json_str)
        .map_err(|e| oxllm_core::Error::Transform(format!("invalid stream chunk: {e}")))?;
    if let Some(error) = value.get("error").filter(|e| match e {
        // Match LiteLLM's Python truthiness (`data.get("error", False)`): only
        // a non-empty string or a non-empty object is a real upstream error.
        // `null`/`false`/`0`/`""`/`[]`/`{}` are "no error" and the chunk flows
        // through as normal content, so a falsy field can never drop the
        // payload by masquerading as a spurious 500.
        Value::String(s) => !s.is_empty(),
        Value::Object(o) => !o.is_empty(),
        _ => false,
    }) {
        // Bound the message to spec-sanctioned shapes so a missing/non-string
        // `message` never causes the whole error object (which may carry
        // `request_id`/internal diagnostics) to be stringified to the client.
        // A bare-string error is safe to forward verbatim (LiteLLM does the
        // same in its chunk handlers); only the object case is field-bounded.
        let message = error.as_str().map(str::to_string).unwrap_or_else(|| {
            error
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    error
                        .get("type")
                        .and_then(Value::as_str)
                        .map(|t| format!("upstream error: {t}"))
                        .unwrap_or_else(|| "upstream error".to_string())
                })
        });
        // Resolve an HTTP status: numeric `code` first, then a named/numeric
        // string `code`, then `type`, finally 500. Numeric strings are filtered
        // to the HTTP range so a stray `"9999"` cannot masquerade as a status.
        let status = error
            .get("code")
            .and_then(Value::as_u64)
            .filter(|code| (400..600).contains(code))
            .and_then(|code| u16::try_from(code).ok())
            .or_else(|| {
                error.get("code").and_then(Value::as_str).and_then(|code| {
                    named_error_status(code)
                        .or_else(|| code.parse::<u16>().ok().filter(|c| (400..600).contains(c)))
                })
            })
            .or_else(|| {
                error
                    .get("type")
                    .and_then(Value::as_str)
                    .and_then(named_error_status)
            })
            .unwrap_or(500);
        return Err(oxllm_core::Error::Upstream { status, message });
    }
    Ok(Some(value))
}

/// Map a known OpenAI-style error `code` or `type` string to an HTTP status.
/// Shared across the `code` and `type` lookup paths.
fn named_error_status(s: &str) -> Option<u16> {
    match s {
        "rate_limit_exceeded" | "rate_limit" | "rate_limit_error" | "insufficient_quota" => {
            Some(429)
        }
        "invalid_request_error" | "context_length_exceeded" | "content_policy_violation" => {
            Some(400)
        }
        "invalid_api_key" | "invalid_api_key_error" => Some(401),
        "model_not_found" | "model_not_found_error" => Some(404),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chunk_returns_none() {
        assert!(parse_openai_stream_chunk(" ").expect("ok").is_none());
    }

    #[test]
    fn done_marker_returns_none() {
        assert!(
            parse_openai_stream_chunk("data: [DONE]")
                .expect("ok")
                .is_none()
        );
    }

    #[test]
    fn done_marker_without_space_returns_none() {
        // M1: `data:[DONE]` (no space after the colon) must also be recognized
        // as the terminal marker, not parsed as JSON.
        assert!(
            parse_openai_stream_chunk("data:[DONE]")
                .expect("ok")
                .is_none()
        );
    }

    #[test]
    fn done_marker_bare_returns_none() {
        // M1: a bare `[DONE]` (no `data:` prefix at all) is also terminal.
        assert!(parse_openai_stream_chunk("[DONE]").expect("ok").is_none());
    }

    #[test]
    fn sse_comment_returns_none() {
        assert!(
            parse_openai_stream_chunk(": keep-alive\n\n")
                .expect("ok")
                .is_none()
        );
    }

    #[test]
    fn event_line_returns_none() {
        assert!(
            parse_openai_stream_chunk("event: ping\n\n")
                .expect("ok")
                .is_none()
        );
    }

    #[test]
    fn parses_data_prefixed_json() {
        let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
        let out = parse_openai_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        assert_eq!(out["id"], "x");
        assert_eq!(out["choices"][0]["delta"]["content"], "hi");
    }

    #[test]
    fn parses_bare_json_without_prefix() {
        let raw = "{\"id\":\"y\"}";
        let out = parse_openai_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        assert_eq!(out["id"], "y");
    }

    #[test]
    fn invalid_json_is_an_error() {
        let raw = "data: {not valid";
        assert!(parse_openai_stream_chunk(raw).is_err());
    }

    #[test]
    fn in_body_error_is_an_upstream_error() {
        let raw = r#"data: {"error":{"message":"capacity exhausted","code":429}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 429, message }
                if message == "capacity exhausted"
        ));
    }

    #[test]
    fn malformed_in_body_error_defaults_to_500() {
        let raw = r#"data: {"error":"provider failed"}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        // A bare-string error is forwarded verbatim (parity with LiteLLM's
        // chunk handlers, which `raise` the string); status stays 500 since a
        // string carries no `code`/`type`.
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 500, message }
                if message == "provider failed"
        ));
    }

    #[test]
    fn null_error_field_is_not_treated_as_error() {
        let raw = r#"data: {"id":"x","choices":[{"delta":{"content":"hi"}}],"error":null}"#;
        let out = parse_openai_stream_chunk(raw)
            .expect("null error must not surface as Upstream error")
            .expect("should produce value");
        assert_eq!(out["id"], "x");
        assert_eq!(out["choices"][0]["delta"]["content"], "hi");
        assert!(out["error"].is_null());
    }

    /// L6: a bare `data:` line with no payload must not error — it carries no
    /// JSON and must be skipped. Previously this hit `serde_json::from_str("")`
    /// and surfaced a spurious `Transform` error.
    #[test]
    fn empty_data_line_returns_none() {
        assert!(parse_openai_stream_chunk("data:").expect("ok").is_none());
        assert!(parse_openai_stream_chunk("data: ").expect("ok").is_none());
        assert!(
            parse_openai_stream_chunk("data:   \n\n")
                .expect("ok")
                .is_none()
        );
    }

    /// L5: an unknown SSE field line (not `data:`/`event:`/`id:`/`retry:`/`:`
    /// and not a bare JSON object) must be ignored per the SSE spec rather
    /// than surfacing a parse error.
    #[test]
    fn unknown_sse_field_returns_none() {
        assert!(
            parse_openai_stream_chunk("foo: bar").expect("ok").is_none(),
            "unknown field line must be ignored"
        );
        assert!(
            parse_openai_stream_chunk("baz:qux").expect("ok").is_none(),
            "unknown field line without space must be ignored"
        );
    }

    #[test]
    fn string_error_code_rate_limit_maps_to_429() {
        // M3: OpenAI emits `code` as a string. `rate_limit_exceeded` → 429.
        let raw = r#"data: {"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 429, .. }
        ));
    }

    #[test]
    fn string_error_code_rate_limit_short_maps_to_429() {
        // M3: alternate short spelling `rate_limit`.
        let raw = r#"data: {"error":{"code":"rate_limit","message":"slow down"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 429, .. }
        ));
    }

    #[test]
    fn string_error_code_invalid_request_maps_to_400() {
        // M3: `invalid_request_error` → 400.
        let raw = r#"data: {"error":{"code":"invalid_request_error","message":"bad input"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 400, .. }
        ));
    }

    #[test]
    fn error_type_rate_limit_error_maps_to_429() {
        // M3: when `code` is absent, fall back to `error.type`. Some providers
        // surface `rate_limit_error` here.
        let raw = r#"data: {"error":{"type":"rate_limit_error","message":"slow down"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 429, message }
                if message == "slow down"
        ));
    }

    #[test]
    fn unknown_string_error_code_defaults_to_500() {
        // unknown string codes still default to 500.
        let raw = r#"data: {"error":{"code":"some_weird_code","message":"boom"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 500, .. }
        ));
    }

    #[test]
    fn string_error_code_numeric_429_maps_to_429() {
        let raw = r#"data: {"error":{"code":"429","message":"slow down"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 429, .. }
        ));
    }

    #[test]
    fn string_error_code_numeric_503_maps_to_503() {
        let raw = r#"data: {"error":{"code":"503","message":"down"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 503, .. }
        ));
    }

    #[test]
    fn string_error_code_numeric_out_of_range_defaults_to_500() {
        let raw = r#"data: {"error":{"code":"9999","message":"boom"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 500, .. }
        ));
    }

    #[test]
    fn named_error_codes_map_to_expected_statuses() {
        for (code, expected) in [
            ("insufficient_quota", 429),
            ("model_not_found", 404),
            ("model_not_found_error", 404),
            ("context_length_exceeded", 400),
            ("content_policy_violation", 400),
            ("invalid_api_key", 401),
            ("invalid_api_key_error", 401),
        ] {
            let raw = format!(r#"data: {{"error":{{"code":"{code}","message":"x"}}}}"#);
            let err = parse_openai_stream_chunk(&raw).expect_err("must surface provider error");
            assert!(
                matches!(err, oxllm_core::Error::Upstream { status, .. } if status == expected),
                "code {code} should map to {expected}"
            );
        }
    }

    #[test]
    fn named_error_type_insufficient_quota_maps_to_429() {
        let raw = r#"data: {"error":{"type":"insufficient_quota","message":"x"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        assert!(matches!(
            err,
            oxllm_core::Error::Upstream { status: 429, .. }
        ));
    }

    #[test]
    fn error_without_message_does_not_leak_object() {
        // A missing `message` must not stringify the whole object: diagnostic
        // fields like `request_id`/`debug` must never reach the client.
        let raw = r#"data: {"error":{"code":"weird","debug":{"request_id":"abc-xyz","internal":"secret"}}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        match err {
            oxllm_core::Error::Upstream { message, .. } => {
                assert!(
                    !message.contains("abc-xyz"),
                    "message must not leak request_id: {message}"
                );
                assert!(
                    !message.contains("secret"),
                    "message must not leak internals: {message}"
                );
            }
            other => panic!("expected Upstream, got {other:?}"),
        }
    }

    #[test]
    fn error_with_non_string_message_uses_fallback() {
        let raw = r#"data: {"error":{"message":42}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        match err {
            oxllm_core::Error::Upstream { message, .. } => {
                assert_eq!(message, "upstream error");
            }
            other => panic!("expected Upstream, got {other:?}"),
        }
    }

    #[test]
    fn error_with_type_and_no_message_surfaces_type() {
        let raw = r#"data: {"error":{"type":"rate_limit_error"}}"#;
        let err = parse_openai_stream_chunk(raw).expect_err("must surface provider error");
        match err {
            oxllm_core::Error::Upstream { status, message } => {
                assert_eq!(status, 429);
                assert_eq!(message, "upstream error: rate_limit_error");
            }
            other => panic!("expected Upstream, got {other:?}"),
        }
    }

    #[test]
    fn multiline_event_block_with_data_is_parsed() {
        // An `event:` line followed by a `data:` line must not be dropped: only
        // the non-`data` field is metadata.
        let raw = "event: ping\ndata: {\"id\":\"x\"}\n\n";
        let out = parse_openai_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        assert_eq!(out["id"], "x");
    }

    #[test]
    fn multiline_data_block_is_concatenated() {
        // Multiple `data:` lines in one block are joined with `\n` per spec.
        let raw = "data: {\"a\":\ndata: 1}";
        let out = parse_openai_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        assert_eq!(out["a"], 1);
    }

    #[test]
    fn lone_cr_multifield_event_block_with_data_is_parsed() {
        // Parity with `multiline_event_block_with_data_is_parsed` but framed
        // with lone-CR (`\r`) line endings — the style the host framer
        // (`oxllm_server::sse::pop_sse_event`) recognises via `\r\r`. The
        // `data:` line must still be found; previously `str::lines` (which
        // splits only on `\n`) treated the whole block as one line and the
        // payload was silently dropped.
        let raw = "event: ping\rdata: {\"id\":\"x\"}\r\r";
        let out = parse_openai_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        assert_eq!(out["id"], "x");
    }

    /// Leading whitespace before the `data:` field name (e.g. injected by a
    /// misbehaving proxy) must not hide the payload. Parity with the sibling
    /// `oxllm_server::sse::extract_sse_data`, which trims each line before the
    /// prefix check.
    #[test]
    fn data_line_with_leading_whitespace_is_parsed() {
        let raw = "  data: {\"id\":\"x\"}\n\n";
        let out = parse_openai_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        assert_eq!(out["id"], "x");
    }

    /// A falsy but non-null `error` field (LiteLLM uses Python truthiness:
    /// `data.get("error", False)`) must not surface as a spurious upstream
    /// error and drop the chunk's real content.
    #[test]
    fn falsy_error_field_is_not_treated_as_error() {
        for raw in [
            r#"data: {"error": false, "choices": []}"#,
            r#"data: {"error": 0, "choices": []}"#,
            r#"data: {"error": "", "choices": []}"#,
            r#"data: {"error": [], "choices": []}"#,
            r#"data: {"error": {}, "choices": []}"#,
            r#"data: {"error": null, "choices": []}"#,
        ] {
            let out = parse_openai_stream_chunk(raw)
                .unwrap_or_else(|_| panic!("falsy error must not surface as Upstream: {raw}"))
                .unwrap_or_else(|| panic!("falsy error must yield a value: {raw}"));
            assert!(out["choices"].is_array(), "payload preserved for: {raw}");
        }
    }
}
