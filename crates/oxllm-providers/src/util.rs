//! Small shared helpers used across provider modules.

#[cfg(any(
    test,
    feature = "openai_compat",
    feature = "voyage",
    feature = "jina_ai"
))]
use std::collections::HashMap;

#[cfg(any(feature = "cohere", feature = "azure_ai"))]
use oxllm_core::Deployment;
#[cfg(any(
    test,
    feature = "openai_compat",
    feature = "voyage",
    feature = "jina_ai"
))]
use oxllm_core::provider::ResolvedCredentials;
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
use serde_json::Map;
#[cfg(any(test, feature = "anthropic", feature = "bedrock", feature = "cohere"))]
use serde_json::Value;
#[cfg(any(test, feature = "anthropic", feature = "bedrock"))]
use serde_json::json;

/// Insert an `Authorization: Bearer <token>` header from resolved credentials.
///
/// The bearer is derived via [`ResolvedCredentials::best_effort_bearer`], which
/// forwards the [`ResolvedCredentials::ApiKey`] bearer verbatim and falls back
/// to a best-effort token for the other bearer-like variants. When no bearer
/// can be derived the call degrades to "no auth header" rather than inserting
/// a bogus value, matching the contract every bespoke provider already
/// honours.
#[cfg(any(
    test,
    feature = "openai_compat",
    feature = "voyage",
    feature = "jina_ai"
))]
pub(crate) fn apply_bearer_auth(
    creds: &ResolvedCredentials,
    headers: &mut HashMap<String, String>,
) {
    let Some(bearer) = creds.best_effort_bearer() else {
        return;
    };
    headers.insert("authorization".to_string(), format!("Bearer {bearer}"));
}

/// Append `operation` to `base` when the URL path does not already end in it.
///
/// The suffix match uses the path only. A query string or fragment is kept,
/// so `…/chat/completions?api-version=2024-05-01-preview` stays one path and
/// does not lose `api-version`. A base that already ends in `/messages` is
/// used verbatim instead of becoming `/messages/messages`.
pub(crate) fn append_operation_path(base: &str, operation: &str) -> String {
    let (without_fragment, fragment) = match base.split_once('#') {
        Some((path, fragment)) => (path, Some(fragment)),
        None => (base, None),
    };
    let (path, query) = match without_fragment.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (without_fragment, None),
    };
    let path = path.trim_end_matches('/');
    let operation = operation.trim_matches('/');
    let mut url = if path_ends_with_operation(path, operation) {
        path.to_string()
    } else {
        format!("{path}/{operation}")
    };
    if let Some(query) = query.filter(|query| !query.is_empty()) {
        url.push('?');
        url.push_str(query);
    }
    if let Some(fragment) = fragment {
        url.push('#');
        url.push_str(fragment);
    }
    url
}

/// Identify the unified Azure model-inference host without matching lookalike
/// domains, query strings, or classic deployment-scoped OpenAI paths.
#[cfg(any(feature = "cohere", feature = "azure_ai"))]
pub(crate) fn is_foundry_base(base: &str) -> bool {
    let path = base.split(['?', '#']).next().unwrap_or(base);
    let Some((_, rest)) = path.split_once("://") else {
        return false;
    };
    let authority = rest.split('/').next().unwrap_or("");
    let host = authority
        .rsplit('@')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("");
    host.to_ascii_lowercase()
        .ends_with(".services.ai.azure.com")
        && !rest.contains("/openai/deployments/")
}

/// Explicit `api_type: azure_ai` also supports custom gateways and local fixtures.
#[cfg(feature = "cohere")]
pub(crate) fn uses_foundry_inference(deployment: &Deployment) -> bool {
    deployment.provider_slug() == "azure_ai"
        || deployment
            .oxllm_params
            .extra
            .get("api_type")
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, "azure_ai" | "foundry"))
        || deployment
            .oxllm_params
            .api_base
            .as_deref()
            .is_some_and(is_foundry_base)
}

/// Build Foundry chat, native Anthropic, or embedding URLs, retaining
/// explicit query versions and normalizing full operation paths for reuse.
#[cfg(any(feature = "cohere", feature = "azure_ai"))]
pub(crate) fn foundry_endpoint_url(
    deployment: &Deployment,
    operation: &str,
    anthropic: bool,
) -> String {
    let base = deployment
        .oxllm_params
        .api_base
        .as_deref()
        .unwrap_or("https://azure-ai-api-base-not-configured.invalid");
    let (base, fragment) = base
        .split_once('#')
        .map_or((base, None), |(base, fragment)| (base, Some(fragment)));
    let (base, query) = base
        .split_once('?')
        .map_or((base, None), |(base, query)| (base, Some(query)));
    let mut path = base.trim_end_matches('/').to_string();
    for suffix in [
        "/messages/count_tokens",
        "/chat/completions",
        "/images/embeddings",
        "/embeddings",
        "/embed",
        "/messages",
    ] {
        if let Some(root) = path.strip_suffix(suffix) {
            path = root.to_string();
            break;
        }
    }
    if is_foundry_base(base) {
        if anthropic {
            if let Some(root) = path.strip_suffix("/models") {
                path = format!("{root}/anthropic/v1");
            } else if path.ends_with("/anthropic") {
                path.push_str("/v1");
            } else if path
                .split_once("://")
                .is_some_and(|(_, rest)| !rest.contains('/'))
            {
                path.push_str("/anthropic/v1");
            }
        } else if !path.ends_with("/models") && !path.ends_with("/openai/v1") {
            path.push_str("/models");
        }
    }
    let mut url = append_operation_path(&path, operation);
    if let Some(query) = query.filter(|query| !query.is_empty()) {
        url.push('?');
        url.push_str(query);
    }
    let has_version = query.is_some_and(|query| {
        query
            .split('&')
            .any(|param| param.split('=').next() == Some("api-version"))
    });
    if !has_version
        && let Some(version) = deployment
            .oxllm_params
            .api_version
            .as_deref()
            .filter(|version| !version.is_empty())
    {
        const QUERY: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
            .add(b' ')
            .add(b'&')
            .add(b'=')
            .add(b'#')
            .add(b'+')
            .add(b'%')
            .add(b'?');
        url.push(if url.contains('?') { '&' } else { '?' });
        url.push_str("api-version=");
        url.push_str(&percent_encoding::utf8_percent_encode(version, QUERY).to_string());
    }
    if let Some(fragment) = fragment {
        url.push('#');
        url.push_str(fragment);
    }
    url
}

/// True when `path` is exactly `operation` or ends in `/operation`.
fn path_ends_with_operation(path: &str, operation: &str) -> bool {
    path == operation
        || path
            .strip_suffix(operation)
            .is_some_and(|prefix| prefix.ends_with('/'))
}

/// Convert a `data:<media-type>;base64,<payload>` URL into the Anthropic
/// Messages image block shape. Returns `None` for non-`data:` URLs, data URLs
/// without a `;base64` marker (URL-encoded payloads are not forwarded as
/// base64), or malformed data URLs (caller leaves the original block
/// untouched). The media type is forwarded verbatim and not validated as an
/// image type; Anthropic rejects unsupported media types upstream.
///
/// Shared by Anthropic and Bedrock transforms to keep image conversion identical.
#[cfg(any(test, feature = "anthropic", feature = "bedrock"))]
pub(crate) fn data_url_to_anthropic_image(url: &str) -> Option<Value> {
    let (media_type, data) = parse_base64_data_url(url)?;
    Some(json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    }))
}

/// Borrow a data URL's media type and base64 payload. In particular, a
/// percent-encoded data URL must never be labelled as base64 on the wire.
#[cfg(any(test, feature = "anthropic", feature = "bedrock"))]
pub(crate) fn parse_base64_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (header, data) = rest.split_once(',')?;
    if data.is_empty() {
        return None;
    }
    let mut parts = header.split(';');
    let media_type = parts.next().filter(|s| !s.is_empty())?;
    if !parts.any(|seg| seg == "base64") {
        return None;
    }
    Some((media_type, data))
}

/// Extract all SSE data fields from one event, including lone-CR framing.
/// Vendor dispatch must inspect the same joined payload as its final parser.
#[cfg(any(feature = "anthropic", feature = "bedrock", feature = "cohere"))]
pub(crate) fn extract_sse_data(raw: &str) -> Option<String> {
    let parts: Vec<&str> = raw
        .split(['\n', '\r'])
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(|data| data.strip_prefix(' ').unwrap_or(data))
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

/// Accept the OpenAI schema wrapper and the historical direct-schema alias,
/// but never mistake a wrapper with a missing schema for the schema itself.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn response_format_schema(
    format: &Map<String, Value>,
) -> oxllm_core::Result<Option<Value>> {
    let Some(value) = format.get("json_schema") else {
        return Ok(None);
    };
    let schema = match value.as_object() {
        Some(wrapper) if wrapper.contains_key("schema") => &wrapper["schema"],
        Some(wrapper) if !wrapper.contains_key("name") && !wrapper.contains_key("strict") => value,
        _ => {
            return Err(oxllm_core::Error::BadRequest(
                "response_format.json_schema requires an object schema".into(),
            ));
        }
    };
    if !schema.is_object() {
        return Err(oxllm_core::Error::BadRequest(
            "response_format.json_schema.schema must be an object".into(),
        ));
    }
    Ok(Some(schema.clone()))
}

/// Translate only a caller-supplied schema. Schema-less JSON mode has no
/// equivalent on these native APIs; never invent a schema or a tool fallback.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn native_response_schema(request: &Value) -> oxllm_core::Result<Option<Value>> {
    let invalid = |message: &str| oxllm_core::Error::BadRequest(message.into());
    let Some(format) = request
        .get("response_format")
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let format = format
        .as_object()
        .ok_or_else(|| invalid("response_format must be an object"))?;
    match format.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some(kind @ ("json_schema" | "json_object")) => {
            if let Some(strict) = format
                .get("json_schema")
                .and_then(|value| value.get("strict"))
                .filter(|value| !value.is_null())
                && !strict.is_boolean()
            {
                return Err(invalid(
                    "response_format.json_schema.strict must be a boolean",
                ));
            }
            let schema = response_format_schema(format)?;
            if schema.is_none() {
                return Err(invalid(if kind == "json_object" {
                    "response_format json_object has no native representation without a schema; use json_schema"
                } else {
                    "response_format json_schema requires an object schema"
                }));
            }
            Ok(schema)
        }
        _ => Err(invalid(
            "unsupported native response_format.type; use text or json_schema",
        )),
    }
}

/// Resolve equivalent native schema envelopes without changing their contents.
/// A legacy output_format is translated to the stable format container, not a
/// beta request or a synthetic tool. Multiple definitions must agree exactly.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn native_output_format(
    request: &Value,
    native_paths: &[(&str, &str)],
) -> oxllm_core::Result<Option<Value>> {
    // A bare JSON-mode intent can be satisfied by a schema the caller has
    // explicitly supplied through another envelope. Never synthesize one.
    let json_mode = request.get("response_format").is_some_and(|format| {
        format.get("type").and_then(Value::as_str) == Some("json_object")
            && format.get("json_schema").is_none()
    });
    let mut selected = if json_mode {
        None
    } else {
        native_response_schema(request)?
    }
    .map(|schema| {
        Value::Object(Map::from_iter([
            ("type".into(), Value::String("json_schema".into())),
            ("schema".into(), schema),
        ]))
    });
    let mut source = "response_format";
    for &(path, label) in native_paths {
        if let Some((parent, _)) = path.rsplit_once('/')
            && !parent.is_empty()
            && request
                .pointer(parent)
                .is_some_and(|value| !value.is_null() && !value.is_object())
        {
            return Err(oxllm_core::Error::BadRequest(format!(
                "{} must be an object",
                parent.trim_start_matches('/')
            )));
        }
        let Some(format) = request.pointer(path).filter(|value| !value.is_null()) else {
            continue;
        };
        if request
            .pointer("/response_format/type")
            .and_then(Value::as_str)
            == Some("text")
        {
            return Err(oxllm_core::Error::BadRequest(format!(
                "response_format text conflicts with {label}"
            )));
        }
        if !format.is_object() {
            return Err(oxllm_core::Error::BadRequest(format!(
                "{label} must be an object"
            )));
        }
        if path != "/output_format"
            && format.get("type").and_then(Value::as_str) == Some("json_object")
        {
            return Err(oxllm_core::Error::BadRequest(format!(
                "{label} json_object has no native representation; use json_schema with an object schema"
            )));
        }
        if path == "/output_format"
            && (format.get("type").and_then(Value::as_str) != Some("json_schema")
                || !format.get("schema").is_some_and(Value::is_object)
                || format.as_object().is_some_and(|format| {
                    format
                        .keys()
                        .any(|key| !matches!(key.as_str(), "type" | "schema"))
                }))
        {
            return Err(oxllm_core::Error::BadRequest("output_format requires json_schema with only type and an object schema; use output_config.format for native options".into()));
        }
        if let Some(selected) = &selected {
            if selected != format {
                return Err(oxllm_core::Error::BadRequest(format!(
                    "{source} conflicts with {label}"
                )));
            }
        } else {
            selected = Some(format.clone());
            source = label;
        }
    }
    if json_mode
        && !selected.as_ref().is_some_and(|format| {
            format.get("type").and_then(Value::as_str) == Some("json_schema")
                && format.get("schema").is_some_and(Value::is_object)
        })
    {
        return Err(oxllm_core::Error::BadRequest("response_format json_object has no native representation without a schema; use json_schema".into()));
    }
    Ok(selected)
}

/// Strictness is transport metadata, never a JSON Schema keyword. Explicit
/// false/null stay non-strict; malformed or misplaced flags must not disappear.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn strict_function_tool(function: &Value) -> oxllm_core::Result<bool> {
    if [
        function.get("parameters"),
        function.get("input_schema"),
        function.pointer("/inputSchema/json"),
    ]
    .into_iter()
    .flatten()
    .any(|schema| schema.get("strict").is_some())
    {
        return Err(oxllm_core::Error::BadRequest("tool schema must not contain a root strict field; put strict on the function/tool definition".into()));
    }
    match function.get("strict") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => Ok(false),
        Some(Value::Bool(true)) => Ok(true),
        Some(_) => Err(oxllm_core::Error::BadRequest(
            "tool strict must be a boolean".into(),
        )),
    }
}

/// Resolve transport-metadata aliases without ignoring either strict flag or
/// schema carrier. Contradictory declarations are malformed, not precedence.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn strict_tool_definition(tool: &Value) -> oxllm_core::Result<bool> {
    let root = strict_function_tool(tool)?;
    let Some(function) = tool.get("function") else {
        return Ok(root);
    };
    let nested = strict_function_tool(function)?;
    match (
        tool.get("strict").filter(|value| !value.is_null()),
        function.get("strict").filter(|value| !value.is_null()),
    ) {
        (Some(root), Some(nested)) if root != nested => Err(oxllm_core::Error::BadRequest(
            "tool strict conflicts with function.strict".into(),
        )),
        (Some(_), _) => Ok(root),
        _ => Ok(nested),
    }
}

/// A `thinking` block the API rejects on replay: the text is missing or
/// whitespace AND it carries no signature. A signed blank block is valid —
/// models returning `display: omitted` produce exactly that shape, and the
/// signature binds it to the original (omitted) text, so it must be replayed
/// unchanged rather than dropped or refilled. `redacted_thinking` carries
/// opaque data, not text, so it is never blank.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn is_blank_unsigned_thinking_block(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("thinking")
        && match block.get("thinking") {
            None | Some(Value::Null) => true,
            Some(Value::String(text)) => text.trim().is_empty(),
            _ => false,
        }
        && match block.get("signature") {
            None | Some(Value::Null) => true,
            Some(Value::String(signature)) => signature.is_empty(),
            _ => false,
        }
}

/// Normalize signed missing/null text to the provider's empty-text replay
/// shape. Explicit strings (including whitespace) are signature-bound bytes;
/// malformed field types must remain available to validation, not be repaired.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn normalize_thinking_block(block: &mut Value) {
    if block.get("type").and_then(Value::as_str) == Some("thinking")
        && block.get("thinking").is_none_or(Value::is_null)
        && block
            .get("signature")
            .and_then(Value::as_str)
            .is_some_and(|signature| !signature.is_empty())
    {
        block["thinking"] = json!("");
    }
}

/// Inline content is authoritative for a mirrored signed/redacted block,
/// even when its text spelling or metadata differs from the extension copy.
/// Signatures identify thinking blocks; redacted data is itself opaque identity.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
fn is_inline_thinking_mirror(inline: &Value, block: &Value) -> bool {
    if inline == block {
        return true;
    }
    match block.get("type").and_then(Value::as_str) {
        Some("thinking") => {
            let Some(signature) = block
                .get("signature")
                .and_then(Value::as_str)
                .filter(|signature| !signature.is_empty())
            else {
                return false;
            };
            (inline.get("type").and_then(Value::as_str) == Some("thinking")
                && inline.get("signature").and_then(Value::as_str) == Some(signature))
                || inline
                    .pointer("/reasoningContent/reasoningText/signature")
                    .and_then(Value::as_str)
                    == Some(signature)
        }
        Some("redacted_thinking") => {
            (inline.get("type").and_then(Value::as_str) == Some("redacted_thinking")
                && inline.get("data") == block.get("data"))
                || inline.pointer("/reasoningContent/redactedContent") == block.get("data")
        }
        _ => false,
    }
}

/// Restore provider-extension thinking blocks before assistant text/tool uses.
/// Content-array blocks take precedence when clients also send their mirrored
/// `thinking_blocks` field, so a signed block is not replayed twice.
///
/// An unsigned thinking block with no text is omitted. A signed blank block
/// — the `display: omitted` replay shape — is kept unchanged: its signature
/// binds the original text, so refilling or dropping it would break replay.
/// Only the gateway's signature-only extension spelling (no `thinking` key)
/// may be refilled from non-empty `reasoning_content`, with exactly one block
/// and no inline thinking association. Explicit empty/whitespace/null text is
/// never refilled; several blocks are not split. A non-blank block still needs
/// its signature when required.
#[cfg(any(feature = "anthropic", feature = "bedrock"))]
pub(crate) fn thinking_blocks_for_replay(
    message: &Value,
    require_signature: bool,
) -> oxllm_core::Result<Vec<Value>> {
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Ok(Vec::new());
    }
    let content = message.get("content").and_then(Value::as_array);
    let native_thinking = content.is_some_and(|blocks| {
        blocks.iter().any(|block| {
            matches!(
                block.get("type").and_then(Value::as_str),
                Some("thinking" | "redacted_thinking")
            ) || block.get("reasoningContent").is_some()
        })
    });
    let mut replay = Vec::new();
    if let Some(value) = message.get("thinking_blocks").filter(|v| !v.is_null()) {
        let blocks = value.as_array().ok_or_else(|| {
            oxllm_core::Error::BadRequest("assistant thinking_blocks must be an array".into())
        })?;
        for block in blocks {
            let signature_only = block.get("type").and_then(Value::as_str) == Some("thinking")
                && block.get("thinking").is_none()
                && block
                    .get("signature")
                    .and_then(Value::as_str)
                    .is_some_and(|signature| !signature.is_empty());
            let mut block = block.clone();
            normalize_thinking_block(&mut block);
            // An unsigned blank block cannot be replayed (the signature is
            // what makes a blank `display: omitted` block acceptable); a
            // signed blank stays, whatever its text.
            if is_blank_unsigned_thinking_block(&block) {
                continue;
            }
            let valid = match block.get("type").and_then(Value::as_str) {
                Some("thinking") => {
                    block.get("thinking").and_then(Value::as_str).is_some()
                        && (!require_signature
                            || block.get("signature").and_then(Value::as_str).is_some())
                }
                Some("redacted_thinking") => block.get("data").and_then(Value::as_str).is_some(),
                _ => false,
            };
            if !valid {
                return Err(oxllm_core::Error::BadRequest(
                    "assistant thinking_blocks must contain complete provider thinking blocks; \
                     replay thinking text with its original signature or redacted_thinking data"
                        .into(),
                ));
            }
            // Validate before suppressing a mirror so precedence cannot hide
            // malformed extension fields. Raw JSON equality is insufficient:
            // missing/null/empty text and streamed signature-only copies can
            // all refer to the same inline signed block.
            if content.is_some_and(|blocks| {
                blocks
                    .iter()
                    .any(|inline| is_inline_thinking_mirror(inline, &block))
            }) {
                continue;
            }
            // The gateway stream emits text and signature separately. Refill
            // only that spelling, never explicit signature-bound text, and do
            // not assign concatenated reasoning to one of several blocks.
            // Even whitespace-only stream text is signature-bound bytes.
            if signature_only && blocks.len() == 1 && !native_thinking {
                if let Some(text) = message
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                {
                    block["thinking"] = json!(text);
                }
            }
            if let Some(object) = block.as_object_mut() {
                // `index` is a stream-accumulator hint, not an Anthropic
                // request field; never replay it to the provider.
                object.remove("index");
            }
            replay.push(block);
        }
    }
    if require_signature
        && replay.is_empty()
        && !native_thinking
        && message
            .get("reasoning_content")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
    {
        return Err(oxllm_core::Error::BadRequest(
            "assistant reasoning_content alone cannot be replayed to Claude; \
             include the original signed thinking_blocks"
                .into(),
        ));
    }
    Ok(replay)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::provider::ResolvedCredentials;

    #[cfg(any(feature = "anthropic", feature = "bedrock"))]
    mod thinking_replay {
        use super::*;

        fn signed(text: Option<Value>) -> Value {
            let mut block = json!({"type": "thinking", "signature": "sig"});
            if let Some(text) = text {
                block["thinking"] = text;
            }
            block
        }

        #[test]
        fn mixed_missing_null_empty_and_text_mirrors_respect_inline_authority() {
            let spellings = [
                None,
                Some(Value::Null),
                Some(json!("")),
                Some(json!(" \t\r\n")),
                Some(json!("plan")),
            ];
            for inline in &spellings {
                for mirror in &spellings {
                    let message = json!({
                        "role": "assistant", "content": [signed(inline.clone())],
                        "thinking_blocks": [signed(mirror.clone())],
                        "reasoning_content": "different unsigned text",
                    });
                    assert!(
                        thinking_blocks_for_replay(&message, true)
                            .unwrap()
                            .is_empty(),
                        "{message}"
                    );
                }
            }
        }

        #[test]
        fn only_one_gateway_signature_only_extension_can_be_refilled() {
            let reasoning = " \tplan\r\n ";
            for spelling in [
                None,
                Some(Value::Null),
                Some(json!("")),
                Some(json!(" \t\r\n")),
                Some(json!("signed text")),
            ] {
                let expected_text = match &spelling {
                    None => json!(reasoning),
                    Some(Value::Null) => json!(""),
                    Some(text) => text.clone(),
                };
                let message = json!({
                    "role": "assistant", "content": "answer", "thinking_blocks": [signed(spelling)],
                    "reasoning_content": reasoning,
                });
                assert_eq!(
                    thinking_blocks_for_replay(&message, true).unwrap(),
                    vec![signed(Some(expected_text))]
                );
            }
            for reasoning in [Value::Null, json!(""), json!(" \t\r\n")] {
                let message = json!({
                    "role": "assistant", "content": "answer", "thinking_blocks": [signed(None)],
                    "reasoning_content": reasoning,
                });
                let expected_text = reasoning.as_str().unwrap_or("");
                assert_eq!(
                    thinking_blocks_for_replay(&message, true).unwrap(),
                    vec![signed(Some(json!(expected_text)))]
                );
            }
        }

        #[test]
        fn stream_index_is_removed_before_replay() {
            let mut block = signed(Some(json!("thought")));
            block["index"] = json!(3);
            let message = json!({
                "role": "assistant", "content": "answer", "thinking_blocks": [block]
            });
            let replay = thinking_blocks_for_replay(&message, true).unwrap();
            assert_eq!(replay.len(), 1);
            assert!(replay[0].get("index").is_none());
        }

        #[test]
        fn multiple_associations_never_guess_a_reasoning_split() {
            let first = signed(None);
            let second = json!({"type": "thinking", "signature": "second-sig"});
            let redacted = json!({"type": "redacted_thinking", "data": "AA=="});
            for blocks in [json!([first, second]), json!([first, redacted])] {
                let message = json!({
                    "role": "assistant", "content": "answer", "thinking_blocks": blocks,
                    "reasoning_content": "concatenated reasoning with no block boundaries",
                });
                let replay = thinking_blocks_for_replay(&message, true).unwrap();
                assert_eq!(replay.len(), 2);
                assert_eq!(replay[0], signed(Some(json!(""))));
                assert!(
                    replay
                        .iter()
                        .all(|block| block.get("thinking").is_none_or(|text| text == ""))
                );
            }
            // A single extension is not a single association when inline
            // thinking also exists under a different signature.
            let message = json!({
                "role": "assistant", "content": [second], "thinking_blocks": [first],
                "reasoning_content": "cannot assign this to either signature",
            });
            assert_eq!(
                thinking_blocks_for_replay(&message, true).unwrap(),
                vec![signed(Some(json!("")))]
            );
        }

        #[test]
        fn redacted_mirrors_use_opaque_data_without_losing_distinct_blocks() {
            let block = json!({"type": "redacted_thinking", "data": "AA=="});
            let distinct = json!({"type": "redacted_thinking", "data": "AQ=="});
            for inline in [
                json!({"type": "redacted_thinking", "data": "AA==", "cache_control": {"type": "ephemeral"}}),
                json!({"reasoningContent": {"redactedContent": "AA=="}}),
            ] {
                let message = json!({
                    "role": "assistant", "content": [inline], "thinking_blocks": [block, distinct],
                });
                assert_eq!(
                    thinking_blocks_for_replay(&message, true).unwrap(),
                    vec![distinct.clone()]
                );
            }
        }

        #[test]
        fn malformed_extension_fields_cannot_be_repaired_or_hidden_by_inline_content() {
            let inline = signed(Some(json!("authoritative")));
            for thinking in [json!(false), json!(42), json!([]), json!({})] {
                let mut malformed = signed(Some(thinking));
                let original = malformed.clone();
                normalize_thinking_block(&mut malformed);
                assert_eq!(malformed, original);
                assert!(!is_blank_unsigned_thinking_block(&malformed));
                let message = json!({
                    "role": "assistant", "content": [inline], "thinking_blocks": [malformed],
                    "reasoning_content": "not a schema repair",
                });
                assert!(matches!(
                    thinking_blocks_for_replay(&message, true),
                    Err(oxllm_core::Error::BadRequest(_))
                ));
            }
            for malformed in [
                json!({"type": "thinking", "thinking": "", "signature": false}),
                json!({"type": "thinking", "signature": 42}),
                json!({"type": "thinking", "thinking": "text", "signature": []}),
                json!({"type": "redacted_thinking", "data": null}),
                json!({"type": "redacted_thinking", "data": 42}),
                json!({"type": "redacted_thinking"}),
                json!({"type": "unknown", "signature": "sig"}),
                Value::Null,
            ] {
                let message = json!({
                    "role": "assistant", "content": [inline], "thinking_blocks": [malformed],
                });
                assert!(
                    matches!(
                        thinking_blocks_for_replay(&message, true),
                        Err(oxllm_core::Error::BadRequest(_))
                    ),
                    "{message}"
                );
            }
            for malformed in [json!({}), json!("blocks"), json!(42)] {
                let message =
                    json!({"role": "assistant", "content": [inline], "thinking_blocks": malformed});
                assert!(matches!(
                    thinking_blocks_for_replay(&message, true),
                    Err(oxllm_core::Error::BadRequest(_))
                ));
            }
        }

        #[test]
        fn unsigned_errors_and_non_claude_signature_policy_are_preserved() {
            let unsigned = json!({"type": "thinking", "thinking": "unsigned text"});
            let message =
                json!({"role": "assistant", "content": "answer", "thinking_blocks": [unsigned]});
            assert!(matches!(
                thinking_blocks_for_replay(&message, true),
                Err(oxllm_core::Error::BadRequest(_))
            ));
            assert_eq!(
                thinking_blocks_for_replay(&message, false).unwrap(),
                vec![unsigned]
            );
            for thinking_blocks in [
                Value::Null,
                json!([]),
                json!([{"type": "thinking"}]),
                json!([{"type": "thinking", "thinking": " \t\n"}]),
            ] {
                let message = json!({
                    "role": "assistant", "content": "answer", "thinking_blocks": thinking_blocks,
                    "reasoning_content": "unsigned reasoning",
                });
                assert!(matches!(
                    thinking_blocks_for_replay(&message, true),
                    Err(oxllm_core::Error::BadRequest(_))
                ));
            }
            let message =
                json!({"role": "assistant", "content": "answer", "thinking_blocks": null});
            assert!(
                thinking_blocks_for_replay(&message, true)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    // --- apply_bearer_auth ---

    #[test]
    fn apply_bearer_auth_api_key_uses_best_effort_path() {
        let creds = ResolvedCredentials::ApiKey {
            bearer: "sk-test".to_string(),
        };
        let mut headers = HashMap::new();
        apply_bearer_auth(&creds, &mut headers);
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer sk-test")
        );
    }

    #[test]
    fn apply_bearer_auth_best_effort_variants_yield_bearer() {
        for (name, creds, expected) in [
            (
                "azure",
                ResolvedCredentials::Azure {
                    api_key: "az-key".to_string(),
                },
                "Bearer az-key",
            ),
            (
                "azure-ad",
                ResolvedCredentials::AzureAd {
                    token: "aad-tok".to_string(),
                },
                "Bearer aad-tok",
            ),
            (
                "gcp",
                ResolvedCredentials::GcpBearer {
                    token: "gcp-tok".to_string(),
                },
                "Bearer gcp-tok",
            ),
        ] {
            let mut headers = HashMap::new();
            apply_bearer_auth(&creds, &mut headers);
            assert_eq!(
                headers.get("authorization").map(String::as_str),
                Some(expected),
                "{name}"
            );
        }
    }

    #[test]
    fn apply_bearer_auth_without_usable_bearer_sets_no_header() {
        for (name, creds) in [
            (
                "sigv4",
                ResolvedCredentials::AwsSigV4 {
                    access_key: "AKIA".to_string(),
                    secret_key: "s".to_string(),
                    session_token: None,
                    region: "us-east-1".to_string(),
                },
            ),
            (
                "presigned",
                ResolvedCredentials::PreSignedHeaders(HashMap::new()),
            ),
        ] {
            let mut headers = HashMap::new();
            headers.insert("authorization".to_string(), "pre-existing".to_string());
            apply_bearer_auth(&creds, &mut headers);
            // No bearer derivable: no header is inserted, so a pre-existing
            // value (if any) is left untouched rather than clobbered.
            assert_eq!(
                headers.get("authorization").map(String::as_str),
                Some("pre-existing"),
                "{name} must not insert a header"
            );
        }
    }

    #[test]
    fn apply_bearer_auth_overwrites_existing_authorization() {
        // When a bearer IS derivable, the helper inserts unconditionally,
        // overwriting any prior `authorization` header. This pins that the
        // single-call contract always wins.
        let creds = ResolvedCredentials::ApiKey {
            bearer: "new".to_string(),
        };
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "old".to_string());
        apply_bearer_auth(&creds, &mut headers);
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer new")
        );
    }

    // --- data_url_to_anthropic_image ---

    #[test]
    fn data_url_to_anthropic_image_valid_base64() {
        let out =
            data_url_to_anthropic_image("data:image/png;base64,AAA=").expect("valid data url");
        assert_eq!(out["type"], "image");
        assert_eq!(out["source"]["type"], "base64");
        assert_eq!(out["source"]["media_type"], "image/png");
        assert_eq!(out["source"]["data"], "AAA=");
    }

    #[test]
    fn data_url_to_anthropic_image_handles_params_after_base64() {
        // `data:image/png;foo=bar;base64,AAA` → media_type is the first header
        // segment, base64 marker still recognized anywhere in the header.
        let out = data_url_to_anthropic_image("data:image/png;foo=bar;base64,AAA")
            .expect("base64 present after params");
        assert_eq!(out["source"]["media_type"], "image/png");
        assert_eq!(out["source"]["data"], "AAA");
    }

    #[test]
    fn data_url_to_anthropic_image_url_encoded_returns_none() {
        // No `;base64` marker: a percent-encoded payload must not be forwarded
        // as base64.
        assert_eq!(
            data_url_to_anthropic_image("data:image/png,%E2%9C%93"),
            None
        );
    }

    #[test]
    fn data_url_to_anthropic_image_rejects_malformed_inputs() {
        for (name, url) in [
            ("non-data-url", "https://example.com/x.png"),
            ("missing-prefix", "image/png;base64,AAA"),
            ("no-comma", "data:image/png;base64"),
            ("empty-data", "data:image/png;base64,"),
            ("empty-media-type", "data:;base64,AAA"),
        ] {
            assert_eq!(
                data_url_to_anthropic_image(url),
                None,
                "{name}: `{url}` should map to None"
            );
        }
    }
}
