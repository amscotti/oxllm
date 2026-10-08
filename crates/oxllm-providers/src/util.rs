//! Small shared helpers used across provider modules.

use std::collections::HashMap;

#[cfg(any(feature = "cohere", feature = "azure_ai"))]
use oxllm_core::Deployment;
use oxllm_core::provider::ResolvedCredentials;
use serde_json::{Map, Value, json};

/// Insert an `Authorization: Bearer <token>` header from resolved credentials.
///
/// The bearer is derived via [`ResolvedCredentials::best_effort_bearer`], which
/// forwards the [`ResolvedCredentials::ApiKey`] bearer verbatim and falls back
/// to a best-effort token for the other bearer-like variants. When no bearer
/// can be derived the call degrades to "no auth header" rather than inserting
/// a bogus value, matching the contract every bespoke provider already
/// honours.
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

/// Build Foundry chat, native Anthropic, or text-embedding URLs, retaining
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
    for suffix in ["/chat/completions", "/embeddings", "/embed", "/messages"] {
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
/// Shared by Anthropic and Bedrock transforms so neither feature depends on
/// the other.
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

/// A `thinking` block the API rejects: the text is missing or whitespace.
/// `redacted_thinking` carries opaque data, not text, so it is never blank.
pub(crate) fn is_blank_thinking_block(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("thinking")
        && block
            .get("thinking")
            .and_then(Value::as_str)
            .is_none_or(|text| text.trim().is_empty())
}

/// Restore provider-extension thinking blocks before assistant text/tool uses.
/// Content-array blocks take precedence when clients also send their mirrored
/// `thinking_blocks` field, so a signed block is not replayed twice.
///
/// A thinking block with no text is omitted. When there is exactly one such
/// block, non-empty `reasoning_content` fills it; several blank blocks are
/// not split. A non-blank block still needs its signature when required.
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
            let mut block = block.clone();
            // Some streaming clients aggregate one signature block alongside
            // reasoning_content. With exactly one block the association is
            // unambiguous; never guess how to split text across several blocks.
            if blocks.len() == 1 && is_blank_thinking_block(&block) {
                if let Some(text) = message
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                {
                    block["thinking"] = json!(text);
                }
            }
            if is_blank_thinking_block(&block) {
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
            if !content.is_some_and(|blocks| blocks.contains(&block)) {
                replay.push(block);
            }
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

/// Maximum recursion depth for [`inject_additional_properties_false`].
/// Matches serde_json's default 128-level recursion limit with headroom for
/// the breadth of combiners walked per level.
const SCHEMA_MAX_DEPTH: u32 = 64;

/// Recursively inject `additionalProperties: false` on object nodes of a JSON
/// Schema. Anthropic (native + Bedrock Converse) requires this annotation.
///
/// Shared by Anthropic and Bedrock so neither feature depends on the other.
pub(crate) fn inject_additional_properties_false(schema: &mut Value) {
    inject_additional_properties_false_inner(schema, 0);
}

fn inject_additional_properties_false_inner(schema: &mut Value, depth: u32) {
    if depth >= SCHEMA_MAX_DEPTH {
        return;
    }
    let Some(map) = schema.as_object_mut() else {
        return;
    };
    let is_object = map
        .get("type")
        .and_then(Value::as_str)
        .map(|t| t == "object")
        .unwrap_or(false);
    if is_object {
        // Anthropic requires additionalProperties to be exactly false on
        // object schemas. Always overwrite — even if the client explicitly
        // sent true — because Anthropic 400s on anything else. (The reference
        // commit 46751ad83e)
        map.insert("additionalProperties".to_string(), Value::Bool(false));
    }
    if let Some(Value::Object(props)) = map.get_mut("properties") {
        for (_, child) in props.iter_mut() {
            inject_additional_properties_false_inner(child, depth + 1);
        }
    }
    if let Some(items) = map.get_mut("items") {
        inject_additional_properties_false_inner(items, depth + 1);
    }
    // Walk `$defs` / `definitions` (referenced sub-schemas) so named
    // definitions get the same annotation. Matches the reference
    // `_add_additional_properties_to_schema`.
    for defs_key in ["$defs", "definitions"] {
        if let Some(Value::Object(defs)) = map.get_mut(defs_key) {
            for (_, child) in defs.iter_mut() {
                inject_additional_properties_false_inner(child, depth + 1);
            }
        }
    }
    for combiner in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(arr)) = map.get_mut(combiner) {
            for child in arr.iter_mut() {
                inject_additional_properties_false_inner(child, depth + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::provider::ResolvedCredentials;

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

    // --- inject_additional_properties_false ---

    #[test]
    fn inject_object_node_gets_annotation() {
        let mut schema = json!({"type": "object", "properties": {}});
        inject_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn inject_non_object_type_is_skipped() {
        let mut schema = json!({"type": "string"});
        inject_additional_properties_false(&mut schema);
        assert!(schema.get("additionalProperties").is_none());
    }

    #[test]
    fn inject_does_not_annotate_untyped_schema() {
        // A node with no explicit `type` is not an object per the reference
        // `result.get("type") == "object"` predicate.
        let mut schema = json!({"description": "x"});
        inject_additional_properties_false(&mut schema);
        assert!(schema.get("additionalProperties").is_none());
    }

    #[test]
    fn inject_does_not_annotate_combiner_root() {
        // A bare combiner node has no `type: object`; the top-level must NOT
        // gain `additionalProperties` (it would reject every instance). The
        // object child inside the combiner IS still annotated.
        let mut schema = json!({"anyOf": [{"type": "object", "properties": {}}]});
        inject_additional_properties_false(&mut schema);
        assert!(
            schema.get("additionalProperties").is_none(),
            "combiner root must not be annotated"
        );
        assert_eq!(schema["anyOf"][0]["additionalProperties"], false);
    }

    #[test]
    fn inject_skips_type_array() {
        // `type: ["object","null"]` is not the string "object"; the reference skips
        // it. oxllm now matches (previously injected).
        let mut schema = json!({"type": ["object", "null"], "properties": {}});
        inject_additional_properties_false(&mut schema);
        assert!(
            schema.get("additionalProperties").is_none(),
            "array-typed node must not be annotated"
        );
    }

    #[test]
    fn inject_always_forces_additional_properties_false() {
        // Even when the client explicitly sends true, it must be overwritten
        // to false — Anthropic 400s on additionalProperties: true.
        let mut schema = json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true,
        });
        inject_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], false);

        // Absent key still gets injected.
        let mut schema = json!({
            "type": "object",
            "properties": {},
        });
        inject_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn inject_walks_items_and_combiners_and_defs() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "list": {"type": "array", "items": {"type": "object", "properties": {}}},
                "choice": {"oneOf": [{"type": "object", "properties": {}}]}
            },
            "$defs": {
                "named": {"type": "object", "properties": {}}
            },
            "definitions": {
                "legacy": {"type": "object", "properties": {}}
            }
        });
        inject_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["list"]["items"]["additionalProperties"],
            false
        );
        assert_eq!(
            schema["properties"]["choice"]["oneOf"][0]["additionalProperties"],
            false
        );
        assert_eq!(schema["$defs"]["named"]["additionalProperties"], false);
        assert_eq!(
            schema["definitions"]["legacy"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn inject_caps_recursion_at_max_depth() {
        // A chain of nested objects deeper than SCHEMA_MAX_DEPTH (64). Only the
        // first SCHEMA_MAX_DEPTH levels may be annotated; deeper levels are
        // skipped and the walk terminates without overflowing the stack.
        let depth: u32 = 80;
        let mut schema = json!({"type": "object"});
        for _ in 0..depth {
            schema = json!({"type": "object", "properties": {"next": schema}});
        }
        inject_additional_properties_false(&mut schema);
        let mut node = &schema;
        let mut annotated = 0u32;
        loop {
            if node.get("additionalProperties") == Some(&json!(false)) {
                annotated += 1;
            } else {
                break;
            }
            match node.get("properties").and_then(|p| p.get("next")) {
                Some(child) => node = child,
                None => break,
            }
        }
        assert_eq!(annotated, SCHEMA_MAX_DEPTH);
    }
}
