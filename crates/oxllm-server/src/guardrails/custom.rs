//! User-defined custom guardrails.
//!
//! Lets operators define guardrails without writing Rust, via
//! `general_settings.guardrails.custom_guardrails: [...]`. Four types are
//! supported:
//!
//! - `regex`     — match patterns in request text and block/mask/log on match.
//! - `keyword`   — simple keyword blocklist with an allowlist override.
//! - `json_path` — evaluate path/operator rules against the request body.
//! - `http`      — forward request text to an external HTTP endpoint for a
//!   classification verdict (follows the Lakera/AIM pattern).
//!
//! The polymorphic `config` blob on each entry is parsed here into typed
//! structs. Invalid entries are skipped (with a warning) rather than aborting
//! state construction, matching the fail-open posture of the remote
//! guardrails.

use std::collections::HashSet;
use std::time::Duration;

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

use crate::config::types::CustomGuardrailConfig;
use crate::guardrails::GuardrailResult;

/// Default replacement string for `mask` actions when none is configured.
const DEFAULT_MASK: &str = "[REDACTED]";

/// Which message roles a text-scoped guardrail inspects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CustomScope {
    All,
    System,
    User,
    Assistant,
}

impl CustomScope {
    fn from_str(s: Option<&str>) -> Self {
        match s.map(|x| x.to_ascii_lowercase()).as_deref() {
            Some("system") => Self::System,
            Some("user") => Self::User,
            Some("assistant") => Self::Assistant,
            _ => Self::All,
        }
    }

    fn matches_role(self, role: &str) -> bool {
        match self {
            Self::All => true,
            Self::System => role.eq_ignore_ascii_case("system"),
            Self::User => role.eq_ignore_ascii_case("user"),
            Self::Assistant => role.eq_ignore_ascii_case("assistant"),
        }
    }
}

/// Collect every string leaf from a JSON value (used to extract text from
/// tool-call argument JSON and tool parameter schemas).
fn collect_string_leaves(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(s) => out.push(s.clone()),
        Value::Array(arr) => {
            for v in arr {
                collect_string_leaves(v, out);
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                collect_string_leaves(v, out);
            }
        }
        _ => {}
    }
}

/// Collect attacker-relevant text from a message's
/// `tool_calls[].function`: each `description` and the string leaves of the
/// parsed `arguments` JSON string. Mirrors the surface that
/// [`crate::guardrails::PiiGuardrail::mask_arguments`] walks.
pub(crate) fn collect_tool_call_texts(msg: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(Value::Array(tool_calls)) = msg.get("tool_calls") else {
        return out;
    };
    for call in tool_calls {
        let Some(Value::Object(func)) = call.get("function") else {
            continue;
        };
        if let Some(Value::String(desc)) = func.get("description") {
            out.push(desc.clone());
        }
        if let Some(args) = func.get("arguments") {
            match args {
                Value::String(s) => match serde_json::from_str::<Value>(s) {
                    Ok(parsed) => collect_string_leaves(&parsed, &mut out),
                    Err(_) => out.push(s.clone()),
                },
                value => collect_string_leaves(value, &mut out),
            }
        }
    }
    out
}

/// The definition of an OpenAI-wrapped or flat native tool. A present but
/// malformed `function` is not treated as a flat definition.
fn tool_definition(tool: &Value) -> Option<&serde_json::Map<String, Value>> {
    tool.get("function").unwrap_or(tool).as_object()
}

/// Mutable counterpart of [`tool_definition`] for string-only masking.
pub(crate) fn tool_definition_mut(tool: &mut Value) -> Option<&mut serde_json::Map<String, Value>> {
    if tool.get("function").is_some() {
        tool.get_mut("function").and_then(Value::as_object_mut)
    } else {
        tool.as_object_mut()
    }
}

/// Match the schema precedence used by [`tools_texts`], without rewriting
/// structural fields such as function names or object keys.
pub(crate) fn tool_schema_mut(
    definition: &mut serde_json::Map<String, Value>,
) -> Option<&mut Value> {
    let field = if definition.contains_key("parameters") {
        "parameters"
    } else {
        "input_schema"
    };
    definition.get_mut(field)
}

/// Collect attacker-relevant text from wrapped and flat top-level tools:
/// each `description` and the string leaves of `parameters` / `input_schema`.
/// A client-controlled catalog can smuggle injection payloads inside JSON-schema
/// descriptions, so these must be scanned by blocking guardrails.
pub(crate) fn tools_texts(body: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(Value::Array(tools)) = body.get("tools") else {
        return out;
    };
    for tool in tools {
        let Some(func) = tool_definition(tool) else {
            continue;
        };
        if let Some(Value::String(desc)) = func.get("description") {
            out.push(desc.clone());
        }
        if let Some(params) = func.get("parameters").or_else(|| func.get("input_schema")) {
            collect_string_leaves(params, &mut out);
        }
    }
    out
}

/// Collect the text segments of every in-scope message. Handles both the
/// legacy string-content shape (`"content": "hi"`) and the array-of-parts
/// shape (`"content": [{"type": "text", "text": "hi"}]`). Also includes
/// `messages[].name` and `tool_calls[].function.{description,arguments}` text
/// so injection payloads hidden in tool-call arguments are scanned.
pub(crate) fn scoped_texts(body: &Value, scope: CustomScope) -> Vec<String> {
    let mut out = Vec::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return out;
    };
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or_default();
        if !scope.matches_role(role) {
            continue;
        }
        push_text(&mut out, msg.get("content"));
        if let Some(Value::String(name)) = msg.get("name") {
            out.push(name.clone());
        }
        out.extend(collect_tool_call_texts(msg));
    }
    out
}

/// A role-scoped rule only inspects that role's messages. The all scope also
/// covers the role-less tool catalog; use the same surface when rewriting.
fn local_texts(body: &Value, scope: CustomScope) -> Vec<String> {
    let mut texts = scoped_texts(body, scope);
    if scope == CustomScope::All {
        texts.extend(tools_texts(body));
    }
    texts
}

fn push_text(out: &mut Vec<String>, content: Option<&Value>) {
    match content {
        Some(Value::String(s)) => out.push(s.clone()),
        Some(Value::Array(parts)) => {
            for p in parts {
                if let Some(Value::String(t)) = p.get("text") {
                    out.push(t.clone());
                }
            }
        }
        _ => {}
    }
}

/// Maximum size (in bytes) of the joined text forwarded to remote
/// classifiers (custom HTTP / MCP). The classifier is making a block/allow
/// decision, not re-constructing the conversation, so the first N KiB is
/// sufficient; capping bounds upstream bandwidth/billing amplification from
/// an attacker stuffing the request body (which can be up to the server's
/// body cap, default several MiB).
pub(crate) const MAX_REMOTE_PAYLOAD_BYTES: usize = 64 * 1024;

/// Truncation marker appended when the joined text exceeds
/// [`MAX_REMOTE_PAYLOAD_BYTES`], so a downstream classifier can tell it is
/// looking at a truncated payload.
const REMOTE_PAYLOAD_TRUNCATION_MARKER: &str = "\n[truncated by oxllm guardrail]";

/// Concatenate all message text (every role) plus top-level `tools[]` text
/// for forwarding to remote classifiers (http / mcp), capped at
/// [`MAX_REMOTE_PAYLOAD_BYTES`] bytes (with a marker appended on truncation)
/// to bound upstream bandwidth/billing amplification.
pub(crate) fn all_text(body: &Value) -> String {
    let mut texts = scoped_texts(body, CustomScope::All);
    texts.extend(tools_texts(body));
    let joined = texts.join("\n");
    cap_remote_payload(&joined)
}

/// Cap `text` at [`MAX_REMOTE_PAYLOAD_BYTES`] bytes (on a char boundary)
/// and append the truncation marker when truncation occurs. No-op for
/// texts already under the cap.
pub(crate) fn cap_remote_payload(text: &str) -> String {
    if text.len() <= MAX_REMOTE_PAYLOAD_BYTES {
        return text.to_string();
    }
    let mut end = MAX_REMOTE_PAYLOAD_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + REMOTE_PAYLOAD_TRUNCATION_MARKER.len());
    out.push_str(&text[..end]);
    out.push_str(REMOTE_PAYLOAD_TRUNCATION_MARKER);
    out
}

/// Cap the total bytes of text forwarded across multiple segments at
/// [`MAX_REMOTE_PAYLOAD_BYTES`], appending the truncation marker at the
/// segment where the cap is hit and clearing all subsequent segments.
///
/// Used by the built-in remote guardrails (AIM / Lakera / ModelArmor /
/// Bedrock) that forward the full messages array (or all text segments) —
/// matching the cap already applied to custom HTTP / MCP guardrails via
/// [`all_text`] / [`cap_remote_payload`]. Bounds upstream bandwidth/billing
/// amplification from large request bodies (which can be up to the server's
/// body cap, default several MiB).
pub(crate) fn cap_forwarded_text_segments(segments: &mut [String]) {
    cap_string_iter(segments.iter_mut());
}

/// Cap the total text content forwarded in an array of scan messages (each
/// `{ "role": ..., "content": "..." }`) at [`MAX_REMOTE_PAYLOAD_BYTES`].
/// Convenience wrapper around [`cap_forwarded_text_segments`] for the
/// AIM / Lakera / ModelArmor guardrails, which build scan messages as
/// `Vec<Value>` rather than `Vec<String>`.
pub(crate) fn cap_scan_messages_content(messages: &mut [Value]) {
    let mut content_texts = Vec::new();
    for message in messages {
        match message.get_mut("content") {
            Some(Value::String(text)) => content_texts.push(text),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(Value::String(text)) = part.get_mut("text") {
                        content_texts.push(text);
                    }
                }
            }
            _ => {}
        }
    }
    cap_string_iter(content_texts.into_iter());
}

/// Join scan text without building an unbounded intermediate payload. Keep
/// the first character crossing the cap so the shared capper can append its
/// marker exactly once, including at a multibyte boundary.
fn join_scan_texts(texts: impl IntoIterator<Item = String>) -> String {
    let mut joined = String::new();
    for (index, text) in texts.into_iter().enumerate() {
        if index > 0 {
            joined.push('\n');
            if joined.len() > MAX_REMOTE_PAYLOAD_BYTES {
                break;
            }
        }
        let remaining = MAX_REMOTE_PAYLOAD_BYTES.saturating_sub(joined.len());
        if text.len() > remaining {
            let mut end = remaining + 1;
            while !text.is_char_boundary(end) {
                end += 1;
            }
            joined.push_str(&text[..end]);
            break;
        }
        joined.push_str(&text);
    }
    joined
}

/// Build the common AIM / Lakera / Model Armor scan surface. Preserve message
/// roles and content shapes, fold tool-call text into its message, and append
/// the role-less catalog as a scan-only user message. Catalog text participates
/// in the same 64 KiB budget as the conversation and is never written into the
/// original request (or into role-scoped local/semantic checks).
pub(crate) fn scan_messages(body: &Value) -> Vec<Value> {
    let mut scan_messages: Vec<Value> = body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|msg| {
            let mut message = serde_json::Map::new();
            if let Some(role) = msg.get("role") {
                message.insert("role".to_string(), role.clone());
            }
            let tool_texts = collect_tool_call_texts(msg);
            if tool_texts.is_empty() {
                if let Some(content) = msg.get("content") {
                    message.insert("content".to_string(), content.clone());
                }
            } else {
                let content_text = match msg.get("content") {
                    Some(Value::String(text)) => text.clone(),
                    Some(Value::Array(parts)) => join_scan_texts(parts.iter().filter_map(|part| {
                        part.get("text").and_then(Value::as_str).map(String::from)
                    })),
                    _ => String::new(),
                };
                let combined = if content_text.is_empty() {
                    join_scan_texts(tool_texts)
                } else {
                    join_scan_texts(std::iter::once(content_text).chain(tool_texts))
                };
                message.insert("content".to_string(), Value::String(combined));
            }
            Value::Object(message)
        })
        .collect();
    let catalog_texts = tools_texts(body);
    if !catalog_texts.is_empty() {
        scan_messages.push(serde_json::json!({
            "role": "user",
            "content": join_scan_texts(catalog_texts)
        }));
    }
    cap_scan_messages_content(&mut scan_messages);
    scan_messages
}

#[cfg(test)]
mod scan_cap_regressions {
    use super::*;

    #[test]
    fn array_content_shares_the_remote_text_budget() {
        let mut messages = vec![serde_json::json!({"role":"user","content":[
            {"type":"text","text":"a".repeat(MAX_REMOTE_PAYLOAD_BYTES + 1)},
            {"type":"text","text":"later text"}
        ]})];
        cap_scan_messages_content(&mut messages);
        assert!(
            messages[0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .ends_with(REMOTE_PAYLOAD_TRUNCATION_MARKER)
        );
        assert_eq!(messages[0]["content"][1]["text"], "");
    }

    #[test]
    fn scan_text_join_bounds_work_at_utf8_cap_boundaries() {
        for prefix_len in [
            0,
            MAX_REMOTE_PAYLOAD_BYTES - 3,
            MAX_REMOTE_PAYLOAD_BYTES - 1,
        ] {
            let text = format!(
                "{}{}",
                "a".repeat(prefix_len),
                "😀".repeat(MAX_REMOTE_PAYLOAD_BYTES)
            );
            let joined = join_scan_texts([text.clone(), "unforwarded-tail".to_string()]);
            assert!(joined.len() > MAX_REMOTE_PAYLOAD_BYTES);
            assert!(joined.len() <= MAX_REMOTE_PAYLOAD_BYTES + 4);
            assert_eq!(cap_remote_payload(&joined), cap_remote_payload(&text));
            assert!(!joined.contains("unforwarded-tail"));
        }
    }

    #[test]
    fn scan_messages_preserves_roles_content_shapes_and_tool_call_text() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "system text"},
                    {"type": "image_url", "image_url": {"url": "image-source"}}
                ]},
                {"role": "assistant", "content": null, "name": "not-forwarded", "tool_calls": [
                    {"function": {"name": "not-forwarded", "description": "call description",
                        "arguments": "{\"q\":[\"argument leaf\",7,true,null]}"}}
                ]}
            ],
            "tools": [{"name": "not-forwarded", "description": "catalog description",
                "input_schema": {"examples": ["schema leaf", 7, false, null]}}]
        });
        let original = body.clone();
        assert_eq!(
            scan_messages(&body),
            vec![
                body["messages"][0].clone(),
                serde_json::json!({"role": "assistant", "content": "call description\nargument leaf"}),
                serde_json::json!({"role": "user", "content": "catalog description\nschema leaf"})
            ]
        );
        assert_eq!(body, original);
    }
}

fn cap_string_iter<'a>(iter: impl Iterator<Item = &'a mut String>) {
    let mut budget = MAX_REMOTE_PAYLOAD_BYTES;
    let mut truncated = false;
    for s in iter {
        if truncated {
            s.clear();
            continue;
        }
        if s.len() > budget {
            let mut end = budget;
            while end > 0 && !s.is_char_boundary(end) {
                end -= 1;
            }
            let mut capped = String::with_capacity(end + REMOTE_PAYLOAD_TRUNCATION_MARKER.len());
            capped.push_str(&s[..end]);
            capped.push_str(REMOTE_PAYLOAD_TRUNCATION_MARKER);
            *s = capped;
            truncated = true;
        }
        budget = budget.saturating_sub(s.len());
    }
}

/// Rewrite every in-scope text segment in place using `f`. Mirrors the
/// content-shape handling of [`local_texts`], including tool argument JSON.
pub(crate) fn rewrite_scoped_texts(
    body: &mut Value,
    scope: CustomScope,
    mut f: impl FnMut(&str) -> String,
) {
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for msg in messages.iter_mut() {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or_default();
            if !scope.matches_role(role) {
                continue;
            }
            if let Some(content) = msg.get_mut("content") {
                match content {
                    Value::String(s) => *s = f(s),
                    Value::Array(parts) => {
                        for p in parts.iter_mut() {
                            if let Some(Value::String(t)) = p.get_mut("text") {
                                *t = f(t);
                            }
                        }
                    }
                    _ => {}
                }
            }
            if let Some(Value::String(name)) = msg.get_mut("name") {
                *name = f(name);
            }
            if let Some(Value::Array(calls)) = msg.get_mut("tool_calls") {
                for call in calls {
                    if let Some(func) = call.get_mut("function") {
                        if let Some(Value::String(description)) = func.get_mut("description") {
                            *description = f(description);
                        }
                        if let Some(arguments) = func.get_mut("arguments") {
                            match arguments {
                                Value::String(s) => match serde_json::from_str::<Value>(s) {
                                    Ok(mut parsed) => {
                                        if rewrite_string_leaves(&mut parsed, &mut f) {
                                            *s = parsed.to_string();
                                        }
                                    }
                                    Err(_) => *s = f(s),
                                },
                                value => {
                                    rewrite_string_leaves(value, &mut f);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if scope == CustomScope::All {
        if let Some(Value::Array(tools)) = body.get_mut("tools") {
            for tool in tools {
                if let Some(func) = tool_definition_mut(tool) {
                    if let Some(Value::String(description)) = func.get_mut("description") {
                        *description = f(description);
                    }
                    if let Some(parameters) = tool_schema_mut(func) {
                        rewrite_string_leaves(parameters, &mut f);
                    }
                }
            }
        }
    }
}

fn rewrite_string_leaves(value: &mut Value, f: &mut impl FnMut(&str) -> String) -> bool {
    match value {
        Value::String(s) => {
            let rewritten = f(s);
            let changed = rewritten != *s;
            if changed {
                *s = rewritten;
            }
            changed
        }
        Value::Array(values) => {
            let mut changed = false;
            for value in values {
                changed |= rewrite_string_leaves(value, f);
            }
            changed
        }
        Value::Object(values) => {
            let mut changed = false;
            for value in values.values_mut() {
                changed |= rewrite_string_leaves(value, f);
            }
            changed
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// regex
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegexAction {
    Block,
    Mask,
    Log,
}

impl RegexAction {
    fn from_str(s: Option<&str>) -> Self {
        match s.map(|x| x.to_ascii_lowercase()).as_deref() {
            Some("mask") => Self::Mask,
            Some("log") => Self::Log,
            _ => Self::Block,
        }
    }
}

#[derive(Debug, Clone)]
struct CompiledRegexRule {
    regex: Regex,
    action: RegexAction,
    replacement: String,
}

/// A `regex`-type custom guardrail.
#[derive(Debug, Clone)]
pub struct RegexGuardrail {
    pub name: String,
    rules: Vec<CompiledRegexRule>,
    scope: CustomScope,
}

/// A `keyword`-type custom guardrail.
#[derive(Debug, Clone)]
pub struct KeywordGuardrail {
    pub name: String,
    blocklist: Vec<String>,
    allowlist: HashSet<String>,
    block: bool, // true => block, false => log
    scope: CustomScope,
}

/// An `http`-type custom guardrail: forward request text to an external
/// classifier and block on its verdict. Fail-open like Lakera/AIM.
#[derive(Debug, Clone)]
pub struct HttpCustomGuardrail {
    pub name: String,
    client: reqwest::Client,
    url: String,
    method: reqwest::Method,
    headers: Vec<(String, String)>,
    block_field: String,
    block_value: Value,
    fail_open: bool,
    host: String,
    allow_private: bool,
}

/// A `json_path`-type custom guardrail.
#[derive(Debug, Clone)]
pub struct JsonPathGuardrail {
    pub name: String,
    rules: Vec<JsonPathRule>,
}

#[derive(Debug, Clone)]
struct JsonPathRule {
    segments: Vec<PathSegment>,
    op: JsonOp,
    value: Value,
    action: JsonAction,
    raw_path: String,
}

#[derive(Debug, Clone)]
enum PathSegment {
    Key(String),
    Index(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonOp {
    Exists,
    Equals,
    Contains,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonAction {
    Block,
    Log,
}

/// The unified custom-guardrail enum, applied in the standard chain.
#[derive(Debug, Clone)]
pub enum CustomGuardrail {
    Regex(RegexGuardrail),
    Keyword(KeywordGuardrail),
    Http(HttpCustomGuardrail),
    JsonPath(JsonPathGuardrail),
}

impl CustomGuardrail {
    /// True when this remote HTTP guardrail is configured to fail closed.
    pub(crate) fn remote_fail_closed(&self) -> bool {
        match self {
            Self::Http(g) => !g.fail_open,
            _ => false,
        }
    }

    /// Human-readable name (for logs).
    pub fn name(&self) -> &str {
        match self {
            Self::Regex(g) => &g.name,
            Self::Keyword(g) => &g.name,
            Self::Http(g) => &g.name,
            Self::JsonPath(g) => &g.name,
        }
    }

    /// Compile a [`CustomGuardrailConfig`] into a runnable guardrail. The
    /// returned `Err` is logged by the caller and the entry skipped.
    /// `allow_private` is forwarded to HTTP guardrails so they can
    /// re-validate the SSRF guard at request time.
    pub fn from_config(cfg: &CustomGuardrailConfig, allow_private: bool) -> Result<Self, String> {
        match cfg.guardrail_type.as_str() {
            "regex" => Ok(Self::Regex(RegexGuardrail::from_config(
                &cfg.config,
                &cfg.name,
            )?)),
            "keyword" => Ok(Self::Keyword(KeywordGuardrail::from_config(
                &cfg.config,
                &cfg.name,
            )?)),
            "json_path" => Ok(Self::JsonPath(JsonPathGuardrail::from_config(
                &cfg.config,
                &cfg.name,
            )?)),
            "http" => Ok(Self::Http(HttpCustomGuardrail::from_config(
                &cfg.config,
                &cfg.name,
                allow_private,
            )?)),
            other => Err(format!(
                "custom guardrail '{}': unknown type '{other}' (expected regex|keyword|json_path|http)",
                cfg.name
            )),
        }
    }

    /// Apply the guardrail to `body` in place. Returns `Block` to reject.
    pub async fn apply(&self, body: &mut Value) -> GuardrailResult {
        match self {
            Self::Regex(g) => g.apply(body),
            Self::Keyword(g) => g.apply(body),
            Self::JsonPath(g) => g.apply(body),
            Self::Http(g) => g.apply(body).await,
        }
    }

    /// Synchronous local-only apply: regex / keyword / json_path only. Http
    /// variants return `Allow` (they belong to the async remote path). Used
    /// by `CustomGuardrailChain::apply_local` so the deterministic local
    /// checks can run outside the remote-guardrail wall-clock budget.
    pub fn apply_sync(&self, body: &mut Value) -> GuardrailResult {
        match self {
            Self::Regex(g) => g.apply(body),
            Self::Keyword(g) => g.apply(body),
            Self::JsonPath(g) => g.apply(body),
            Self::Http(_) => GuardrailResult::Allow,
        }
    }
}

impl RegexGuardrail {
    fn from_config(config: &Value, name: &str) -> Result<Self, String> {
        let parsed: RegexConfig = serde_json::from_value(config.clone())
            .map_err(|e| format!("custom guardrail '{name}': invalid regex config: {e}"))?;
        let scope = CustomScope::from_str(parsed.scope.as_deref());
        let mut rules = Vec::with_capacity(parsed.patterns.len());
        for p in parsed.patterns {
            let regex = Regex::new(&p.pattern).map_err(|e| {
                format!(
                    "custom guardrail '{name}': invalid regex '{}': {e}",
                    p.pattern
                )
            })?;
            rules.push(CompiledRegexRule {
                regex,
                action: RegexAction::from_str(p.action.as_deref()),
                replacement: p
                    .replacement
                    .clone()
                    .unwrap_or_else(|| DEFAULT_MASK.to_string()),
            });
        }
        Ok(Self {
            name: name.to_string(),
            rules,
            scope,
        })
    }

    fn apply(&self, body: &mut Value) -> GuardrailResult {
        let texts = local_texts(body, self.scope);
        // 1. Block rules take precedence and are evaluated on the original
        //    text (before any masking).
        for rule in &self.rules {
            if rule.action == RegexAction::Block {
                for t in &texts {
                    if rule.regex.is_match(t) {
                        return GuardrailResult::Block(
                            "request blocked by content policy".to_string(),
                        );
                    }
                }
            }
        }
        // 2. Log rules emit a warning but do not block.
        for rule in &self.rules {
            if rule.action == RegexAction::Log {
                for t in &texts {
                    if rule.regex.is_match(t) {
                        tracing::warn!(
                            guardrail = %self.name,
                            "custom regex guardrail matched (action=log)"
                        );
                    }
                }
            }
        }
        // 3. Mask rules rewrite text in place.
        if self.rules.iter().any(|r| r.action == RegexAction::Mask) {
            let rules = self.rules.clone();
            rewrite_scoped_texts(body, self.scope, move |text| {
                let mut out = text.to_string();
                for rule in &rules {
                    if rule.action == RegexAction::Mask {
                        out = rule.regex.replace_all(&out, &rule.replacement).to_string();
                    }
                }
                out
            });
        }
        GuardrailResult::Allow
    }
}

impl KeywordGuardrail {
    fn from_config(config: &Value, name: &str) -> Result<Self, String> {
        let parsed: KeywordConfig = serde_json::from_value(config.clone())
            .map_err(|e| format!("custom guardrail '{name}': invalid keyword config: {e}"))?;
        let block = !parsed
            .action
            .as_deref()
            .map(|a| a.eq_ignore_ascii_case("log"))
            .unwrap_or(false);
        Ok(Self {
            name: name.to_string(),
            blocklist: parsed.blocklist,
            // Stored lowercased so the allowlist override matches the
            // case-insensitive semantics of the blocklist: a blocklist keyword
            // is matched with `to_ascii_lowercase()`, so an allowlisted keyword
            // must exempt it regardless of case.
            allowlist: parsed
                .allowlist
                .into_iter()
                .map(|s| s.to_ascii_lowercase())
                .collect(),
            block,
            scope: CustomScope::from_str(parsed.scope.as_deref()),
        })
    }

    fn apply(&self, body: &mut Value) -> GuardrailResult {
        let texts = local_texts(body, self.scope);
        for kw in &self.blocklist {
            // allowlist wins: an explicitly allowed keyword is never a trigger.
            if self.allowlist.contains(&kw.to_ascii_lowercase()) {
                continue;
            }
            let needle = kw.to_ascii_lowercase();
            for t in &texts {
                if t.to_ascii_lowercase().contains(&needle) {
                    if self.block {
                        return GuardrailResult::Block(
                            "request blocked by content policy".to_string(),
                        );
                    }
                    tracing::warn!(
                        guardrail = %self.name,
                        keyword = %kw,
                        "custom keyword guardrail matched (action=log)"
                    );
                }
            }
        }
        GuardrailResult::Allow
    }
}

impl HttpCustomGuardrail {
    /// Build from explicit parameters. Performs no SSRF validation at build
    /// time (mirrors Lakera/AIM where `new` only builds the client); SSRF is
    /// enforced by the caller at config-build time and re-checked via
    /// [`crate::security::is_private_host_async`] at request time when
    /// `allow_private` is `false`. The HTTP client carries the connect-time
    /// [`crate::security::SsrfResolver`] unless `allow_private` is set, so a
    /// DNS-rebinding attack after config load cannot steer the connection to a
    /// private address.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        url: String,
        method: Option<String>,
        headers: Vec<(String, String)>,
        block_field: String,
        block_value: Value,
        timeout_secs: Option<u64>,
        fail_open: bool,
        allow_private: bool,
    ) -> Result<Self, String> {
        let method = method
            .as_deref()
            .map(|m| reqwest::Method::from_bytes(m.as_bytes()))
            .transpose()
            .map_err(|e| format!("custom guardrail '{name}': invalid method: {e}"))?
            .unwrap_or(reqwest::Method::POST);
        let client = crate::security::build_ssrf_http_client(
            allow_private,
            Duration::from_secs(timeout_secs.unwrap_or(5).max(1)),
        )
        .map_err(|e| format!("custom guardrail '{name}': HTTP client build failed: {e}"))?;
        let host = url::Url::parse(&url)
            .ok()
            .and_then(|u| u.host_str().map(String::from))
            .unwrap_or_default();
        Ok(Self {
            name: name.to_string(),
            client,
            url,
            method,
            headers,
            block_field,
            block_value,
            fail_open,
            host,
            allow_private,
        })
    }

    fn from_config(config: &Value, name: &str, allow_private: bool) -> Result<Self, String> {
        let parsed: HttpConfig = serde_json::from_value(config.clone())
            .map_err(|e| format!("custom guardrail '{name}': invalid http config: {e}"))?;
        let block_on = parsed
            .block_on
            .ok_or_else(|| format!("custom guardrail '{name}': http config requires 'block_on'"))?;
        let headers = parsed
            .headers
            .map(|m| {
                m.into_iter()
                    .map(|(k, v)| {
                        let vs = v
                            .as_str()
                            .map(String::from)
                            .unwrap_or_else(|| v.to_string());
                        (k, vs)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self::new(
            name,
            parsed.url,
            parsed.method,
            headers,
            block_on.field,
            block_on.value,
            parsed.timeout_secs,
            parsed.fail_open.unwrap_or(true),
            allow_private,
        )
    }

    pub(crate) async fn apply(&self, body: &mut Value) -> GuardrailResult {
        if !self.allow_private
            && !self.host.is_empty()
            && crate::security::is_private_host_async(&self.host).await
        {
            tracing::warn!(
                guardrail = %self.name,
                host = %self.host,
                "custom http guardrail target resolves to private host; skipping send"
            );
            return if self.fail_open {
                GuardrailResult::Allow
            } else {
                GuardrailResult::Block("request blocked: guardrail unavailable".to_string())
            };
        }
        let text = all_text(body);
        // Send only the joined `content` text — the classifier is making a
        // block/allow decision, not re-constructing the conversation, so the
        // redundant deep-clone of `messages` is dropped to halve the
        // upstream payload (a 4 MiB chat body no longer becomes an 8 MiB
        // POST). `content` is already capped at MAX_REMOTE_PAYLOAD_BYTES.
        let payload = serde_json::json!({ "content": text });
        let mut req = self
            .client
            .request(self.method.clone(), &self.url)
            .json(&payload);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        match req.send().await {
            Ok(resp) => match resp.error_for_status() {
                Ok(resp) => match super::response::read_json::<Value>(resp).await {
                    Ok(v) => {
                        if v.get(&self.block_field).is_none() && !self.fail_open {
                            return GuardrailResult::Block(
                                "request blocked: guardrail returned no verdict".into(),
                            );
                        }
                        if v.get(&self.block_field) == Some(&self.block_value) {
                            tracing::warn!(
                                guardrail = %self.name,
                                "custom http guardrail blocked request"
                            );
                            return GuardrailResult::Block(
                                "request blocked by content policy".to_string(),
                            );
                        }
                        GuardrailResult::Allow
                    }
                    Err(e) => {
                        if self.fail_open {
                            tracing::warn!(
                                guardrail = %self.name,
                                error = %e,
                                "custom http guardrail: invalid or oversized response; allowing (fail_open)"
                            );
                            GuardrailResult::Allow
                        } else {
                            GuardrailResult::Block(
                                "request blocked: guardrail unavailable".to_string(),
                            )
                        }
                    }
                },
                Err(e) => {
                    if self.fail_open {
                        tracing::warn!(
                            guardrail = %self.name,
                            error = %e,
                            "custom http guardrail: non-2xx response; allowing (fail_open)"
                        );
                        GuardrailResult::Allow
                    } else {
                        GuardrailResult::Block("request blocked: guardrail unavailable".to_string())
                    }
                }
            },
            Err(e) => {
                if self.fail_open {
                    tracing::warn!(
                        guardrail = %self.name,
                        error = %e,
                        "custom http guardrail: request failed; allowing (fail_open)"
                    );
                    GuardrailResult::Allow
                } else {
                    GuardrailResult::Block("request blocked: guardrail unavailable".to_string())
                }
            }
        }
    }
}

impl JsonPathGuardrail {
    fn from_config(config: &Value, name: &str) -> Result<Self, String> {
        let parsed: JsonPathConfig = serde_json::from_value(config.clone())
            .map_err(|e| format!("custom guardrail '{name}': invalid json_path config: {e}"))?;
        let mut rules = Vec::with_capacity(parsed.rules.len());
        for r in parsed.rules {
            let segments = parse_path(&r.path).ok_or_else(|| {
                format!("custom guardrail '{name}': invalid json_path '{}'", r.path)
            })?;
            let op = match r.operator.as_deref().unwrap_or("exists") {
                "exists" => JsonOp::Exists,
                "equals" => JsonOp::Equals,
                "contains" => JsonOp::Contains,
                other => {
                    return Err(format!(
                        "custom guardrail '{name}': unknown operator '{other}'"
                    ));
                }
            };
            let action = if r.action.as_deref() == Some("log") {
                JsonAction::Log
            } else {
                JsonAction::Block
            };
            rules.push(JsonPathRule {
                segments,
                op,
                value: r.value.unwrap_or(Value::Null),
                action,
                raw_path: r.path,
            });
        }
        Ok(Self {
            name: name.to_string(),
            rules,
        })
    }

    fn apply(&self, body: &mut Value) -> GuardrailResult {
        for rule in &self.rules {
            let resolved = resolve_path(body, &rule.segments);
            let matched = match rule.op {
                JsonOp::Exists => resolved.is_some(),
                JsonOp::Equals => resolved == Some(&rule.value),
                JsonOp::Contains => resolved.is_some_and(|v| match v {
                    Value::String(s) => {
                        rule.value.as_str().is_some_and(|needle| s.contains(needle))
                    }
                    Value::Array(arr) => arr.iter().any(|x| x == &rule.value),
                    _ => false,
                }),
            };
            if matched {
                match rule.action {
                    JsonAction::Block => {
                        tracing::warn!(
                            guardrail = %self.name,
                            path = %rule.raw_path,
                            "custom json_path guardrail blocked request"
                        );
                        return GuardrailResult::Block(
                            "request blocked by content policy".to_string(),
                        );
                    }
                    JsonAction::Log => {
                        tracing::warn!(
                            guardrail = %self.name,
                            path = %rule.raw_path,
                            "custom json_path guardrail matched (action=log)"
                        );
                    }
                }
            }
        }
        GuardrailResult::Allow
    }
}

/// Parse a dotted path like `messages.0.content` into segments. Numeric
/// segments index into arrays; all others index object keys. A leading `$`
/// (JSONPath root marker) is tolerated and dropped.
fn parse_path(path: &str) -> Option<Vec<PathSegment>> {
    let trimmed = path.trim_start_matches('$').trim_start_matches('.');
    if trimmed.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for seg in trimmed.split('.') {
        if seg.is_empty() {
            return None;
        }
        if let Ok(idx) = seg.parse::<usize>() {
            out.push(PathSegment::Index(idx));
        } else {
            out.push(PathSegment::Key(seg.to_string()));
        }
    }
    Some(out)
}

fn resolve_path<'a>(value: &'a Value, segments: &[PathSegment]) -> Option<&'a Value> {
    let mut cur = value;
    for seg in segments {
        cur = match (cur, seg) {
            (Value::Object(map), PathSegment::Key(k)) => map.get(k)?,
            (Value::Array(arr), PathSegment::Index(i)) => arr.get(*i)?,
            (Value::Object(map), PathSegment::Index(i)) => map.get(&i.to_string())?,
            _ => return None,
        };
    }
    Some(cur)
}

// ---- typed config payloads (parsed from the polymorphic `config` blob) ----

#[derive(Debug, Deserialize)]
struct RegexConfig {
    #[serde(default)]
    patterns: Vec<RegexPatternCfg>,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RegexPatternCfg {
    pattern: String,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    replacement: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct KeywordConfig {
    #[serde(default)]
    blocklist: Vec<String>,
    #[serde(default)]
    allowlist: Vec<String>,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HttpConfig {
    url: String,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    headers: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    block_on: Option<BlockOnCfg>,
    #[serde(default)]
    fail_open: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct BlockOnCfg {
    field: String,
    #[serde(default)]
    value: Value,
}

#[derive(Debug, Default, Deserialize)]
struct JsonPathConfig {
    #[serde(default)]
    rules: Vec<JsonPathRuleCfg>,
}

#[derive(Debug, Deserialize)]
struct JsonPathRuleCfg {
    path: String,
    #[serde(default)]
    operator: Option<String>,
    #[serde(default)]
    value: Option<Value>,
    #[serde(default)]
    action: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    // ---- helpers / scope + walkers ----

    #[tokio::test]
    async fn scoped_texts_filters_by_role() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
            ]
        });
        assert_eq!(scoped_texts(&body, CustomScope::User), vec!["hello"]);
        assert_eq!(scoped_texts(&body, CustomScope::System), vec!["sys"]);
        assert_eq!(
            scoped_texts(&body, CustomScope::Assistant),
            vec!["hi".to_string()]
        );
        assert_eq!(scoped_texts(&body, CustomScope::All).len(), 3);
    }

    #[tokio::test]
    async fn all_text_joins_segments() {
        let body = json!({"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]});
        assert_eq!(all_text(&body), "a\nb");
    }

    #[tokio::test]
    async fn rewrite_scoped_texts_masks_only_user() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "secret"},
                {"role": "user", "content": "secret here"}
            ]
        });
        rewrite_scoped_texts(&mut body, CustomScope::User, |t| t.replace("secret", "[X]"));
        assert_eq!(body["messages"][0]["content"], "secret");
        assert_eq!(body["messages"][1]["content"], "[X] here");
    }

    // ---- regex ----

    fn regex_guard(pattern: &str, action: &str) -> CustomGuardrail {
        CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "r".to_string(),
                guardrail_type: "regex".to_string(),
                config: json!({
                    "patterns": [{"pattern": pattern, "action": action}],
                    "scope": "user"
                }),
            },
            false,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn regex_block_matches() {
        let g = regex_guard(r"forbidden", "block");
        let mut body =
            json!({"messages": [{"role": "user", "content": "this is forbidden stuff"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn regex_block_no_match_allows() {
        let g = regex_guard(r"forbidden", "block");
        let mut body = json!({"messages": [{"role": "user", "content": "all good"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn regex_scope_skips_other_roles() {
        let g = regex_guard(r"forbidden", "block");
        // 'forbidden' only in assistant -> user-scoped guard ignores it.
        let mut body = json!({"messages": [{"role": "assistant", "content": "forbidden"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn regex_mask_redacts_in_place() {
        let g = regex_guard(r"\b\d{16}\b", "mask");
        let mut body =
            json!({"messages": [{"role": "user", "content": "card 1234567890123456 ok"}]});
        let res = g.apply(&mut body).await;
        assert!(matches!(res, GuardrailResult::Allow));
        assert_eq!(body["messages"][0]["content"], "card [REDACTED] ok");
    }

    #[tokio::test]
    async fn regex_mask_custom_replacement() {
        let g = CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "r".to_string(),
                guardrail_type: "regex".to_string(),
                config: json!({"patterns": [{"pattern": "secret", "action": "mask", "replacement": "[HIDDEN]"}], "scope": "all"}),
            },
            false,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "top secret info"}]});
        let _ = g.apply(&mut body).await;
        assert_eq!(body["messages"][0]["content"], "top [HIDDEN] info");
    }

    #[test]
    fn regex_masks_tool_arguments_as_json_and_respects_role_scope() {
        let mut body = json!({
            "messages": [{"role":"assistant", "tool_calls":[{"function":{
                "name":"secret", "description":"secret", "arguments":
                "{\"secret\": [\"secret\", {\"nested\": \"secret\"}], \"n\": 2, \"b\": true}"
            }}]}],
            "tools":[{"function":{"name":"secret", "description":"secret",
                "parameters":{"type":"object", "properties":{"x":{"description":"secret"}}}
            }}]
        });
        let guard = |scope| {
            CustomGuardrail::from_config(&CustomGuardrailConfig {
            name: "redact".into(), guardrail_type: "regex".into(),
            config: json!({"scope":scope,"patterns":[{"pattern":"secret","action":"mask"}]}),
        }, false).unwrap()
        };
        let original = body.clone();
        assert!(matches!(
            guard("user").apply_sync(&mut body),
            GuardrailResult::Allow
        ));
        assert_eq!(
            body, original,
            "user scope must not rewrite assistant tools or the catalog"
        );
        assert!(matches!(
            guard("all").apply_sync(&mut body),
            GuardrailResult::Allow
        ));
        let function = &body["messages"][0]["tool_calls"][0]["function"];
        let arguments: Value =
            serde_json::from_str(function["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(
            arguments,
            json!({"secret":["[REDACTED]",{"nested":"[REDACTED]"}],"n":2,"b":true})
        );
        assert_eq!(
            function["name"], "secret",
            "function identifiers are structural"
        );
        assert_eq!(function["description"], "[REDACTED]");
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["properties"]["x"]["description"],
            "[REDACTED]"
        );
    }

    #[test]
    fn local_block_rules_scan_catalog_without_messages() {
        for (kind, config) in [
            ("regex", json!({"patterns":[{"pattern":"forbidden"}]})),
            ("keyword", json!({"blocklist":["forbidden"]})),
        ] {
            let guard = CustomGuardrail::from_config(
                &CustomGuardrailConfig {
                    name: "catalog".into(),
                    guardrail_type: kind.into(),
                    config,
                },
                false,
            )
            .unwrap();
            let mut body = json!({"tools":[{"function":{"parameters":{
                "properties":{"q":{"description":"forbidden"}}
            }}}]});
            assert!(matches!(
                guard.apply_sync(&mut body),
                GuardrailResult::Block(_)
            ));
        }
    }

    #[test]
    fn local_block_rules_scan_top_level_tool_descriptions() {
        // `tools[].function.description` is attacker-controlled but carries no
        // `messages[]` text: a block rule matching only there must still block,
        // for both deterministic local rule types.
        for (kind, config) in [
            (
                "regex",
                json!({"patterns":[{"pattern":"forbidden-tool-desc"}]}),
            ),
            ("keyword", json!({"blocklist":["forbidden-tool-desc"]})),
        ] {
            let guard = CustomGuardrail::from_config(
                &CustomGuardrailConfig {
                    name: "catalog-desc".into(),
                    guardrail_type: kind.into(),
                    config: config.clone(),
                },
                false,
            )
            .unwrap();
            let mut body = json!({"messages":[{"role":"user","content":"hello"}],"tools":[{
                "type":"function",
                "function":{
                    "name":"lookup",
                    "description":"forbidden-tool-desc",
                    "parameters":{"type":"object","properties":{}}
                }
            }]});
            assert!(
                matches!(guard.apply_sync(&mut body), GuardrailResult::Block(_)),
                "{kind} block rule must inspect tools[].function.description"
            );

            // Documented limitation: a non-`all` scope only inspects that
            // role's messages, so the role-less catalog is exempt.
            let mut user_config = config;
            user_config["scope"] = json!("user");
            let user_guard = CustomGuardrail::from_config(
                &CustomGuardrailConfig {
                    name: "catalog-desc-user".into(),
                    guardrail_type: kind.into(),
                    config: user_config,
                },
                false,
            )
            .unwrap();
            assert!(
                matches!(user_guard.apply_sync(&mut body), GuardrailResult::Allow),
                "{kind}: scoped rule must not inspect the role-less tool catalog"
            );
        }
    }

    #[test]
    fn regex_mask_rewrites_non_json_and_non_string_tool_arguments() {
        // Mask rules reach tool-call arguments through every branch of
        // `rewrite_scoped_texts`: a malformed JSON string is rewritten as a
        // raw string, and a structured (non-string) arguments value has its
        // string leaves rewritten recursively.
        let guard = CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "redact".into(),
                guardrail_type: "regex".into(),
                config: json!({"patterns":[{"pattern":"secret","action":"mask"}]}),
            },
            false,
        )
        .unwrap();
        let mut body = json!({"messages":[{"role":"assistant","tool_calls":[
            {"function":{"name":"a","arguments":"not json secret tail"}},
            {"function":{"name":"b","arguments":{"nested":["secret", 2]}}}
        ]}]});
        assert!(matches!(
            guard.apply_sync(&mut body),
            GuardrailResult::Allow
        ));
        let calls = &body["messages"][0]["tool_calls"];
        assert_eq!(
            calls[0]["function"]["arguments"], "not json [REDACTED] tail",
            "a non-JSON arguments string must be masked as a raw string"
        );
        assert_eq!(
            calls[1]["function"]["arguments"],
            json!({"nested": ["[REDACTED]", 2]}),
            "structured arguments must have their string leaves masked"
        );
    }

    #[tokio::test]
    async fn regex_log_action_allows() {
        let g = regex_guard(r"forbidden", "log");
        let mut body = json!({"messages": [{"role": "user", "content": "forbidden"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn regex_block_takes_precedence_over_mask() {
        let g = CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "r".to_string(),
                guardrail_type: "regex".to_string(),
                config: json!({"patterns": [
                    {"pattern": "x", "action": "mask"},
                    {"pattern": "x", "action": "block"}
                ], "scope": "all"}),
            },
            false,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "x"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn regex_invalid_pattern_rejected_at_build() {
        let res = CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "r".to_string(),
                guardrail_type: "regex".to_string(),
                config: json!({"patterns": [{"pattern": "(unclosed", "action": "block"}]}),
            },
            false,
        );
        assert!(res.is_err());
    }

    // ---- keyword ----

    fn kw_guard(blocklist: &[&str], allowlist: &[&str], action: &str) -> CustomGuardrail {
        CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "kw".to_string(),
                guardrail_type: "keyword".to_string(),
                config: json!({
                    "blocklist": blocklist,
                    "allowlist": allowlist,
                    "action": action,
                    "scope": "all"
                }),
            },
            false,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn keyword_blocklist_blocks_case_insensitive() {
        let g = kw_guard(&["badword"], &[], "block");
        let mut body = json!({"messages": [{"role": "user", "content": "this is BadWord here"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn keyword_allowlist_overrides_blocklist() {
        let g = kw_guard(&["secret"], &["secret"], "block");
        let mut body = json!({"messages": [{"role": "user", "content": "share the secret"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn keyword_allowlist_is_case_insensitive() {
        // The blocklist matches case-insensitively, so the allowlist override
        // must too: an allowlist entry in a different case still exempts the
        // keyword. Without this, "SECRET" on the allowlist would fail to exempt
        // a "secret" blocklist entry and the request would be wrongly blocked.
        let g = kw_guard(&["secret"], &["SECRET"], "block");
        let mut body = json!({"messages": [{"role": "user", "content": "share the secret"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn keyword_log_action_allows() {
        let g = kw_guard(&["badword"], &[], "log");
        let mut body = json!({"messages": [{"role": "user", "content": "badword"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn keyword_no_match_allows() {
        let g = kw_guard(&["badword"], &[], "block");
        let mut body = json!({"messages": [{"role": "user", "content": "clean text"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    // ---- json_path ----

    fn jp_guard(rules: &[Value]) -> CustomGuardrail {
        CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "jp".to_string(),
                guardrail_type: "json_path".to_string(),
                config: json!({"rules": rules}),
            },
            false,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn json_path_exists_blocks() {
        let g = jp_guard(&[json!({"path": "messages.0.content", "operator": "exists"})]);
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn json_path_equals_blocks() {
        let g = jp_guard(&[json!({"path": "model", "operator": "equals", "value": "gpt-4o"})]);
        let mut body = json!({"model": "gpt-4o", "messages": []});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn json_path_contains_blocks_on_string() {
        let g = jp_guard(&[
            json!({"path": "messages.0.content", "operator": "contains", "value": "bomb"}),
        ]);
        let mut body = json!({"messages": [{"role": "user", "content": "build a bomb"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn json_path_log_action_allows() {
        let g = jp_guard(&[json!({"path": "model", "operator": "exists", "action": "log"})]);
        let mut body = json!({"model": "x"});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn json_path_dollar_prefix_tolerated() {
        let g = jp_guard(&[json!({"path": "$.model", "operator": "equals", "value": "x"})]);
        let mut body = json!({"model": "x"});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn json_path_missing_path_allows() {
        let g = jp_guard(&[json!({"path": "messages.0.content", "operator": "exists"})]);
        let mut body = json!({"model": "x"});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn json_path_invalid_path_rejected_at_build() {
        let res = CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "jp".to_string(),
                guardrail_type: "json_path".to_string(),
                config: json!({"rules": [{"path": ""}]}),
            },
            false,
        );
        assert!(res.is_err());
    }

    // ---- http ----

    #[tokio::test]
    async fn http_blocks_when_field_matches() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/classify");
            then.status(200).json_body(json!({"blocked": true}));
        });
        let g = HttpCustomGuardrail::new(
            "h",
            format!("{}/classify", server.base_url()),
            None,
            vec![],
            "blocked".to_string(),
            json!(true),
            None,
            true,
            true,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "anything"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn http_allows_when_field_absent() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/classify");
            then.status(200).json_body(json!({"blocked": false}));
        });
        let g = HttpCustomGuardrail::new(
            "h",
            format!("{}/classify", server.base_url()),
            None,
            vec![],
            "blocked".to_string(),
            json!(true),
            None,
            true,
            true,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn http_fail_open_on_5xx() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/classify");
            then.status(500);
        });
        let g = HttpCustomGuardrail::new(
            "h",
            format!("{}/classify", server.base_url()),
            None,
            vec![],
            "blocked".to_string(),
            json!(true),
            None,
            true,
            true,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(matches!(g.apply(&mut body).await, GuardrailResult::Allow));
    }

    #[tokio::test]
    async fn http_fail_closed_blocks_on_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/classify");
            then.status(500);
        });
        let g = HttpCustomGuardrail::new(
            "h",
            format!("{}/classify", server.base_url()),
            None,
            vec![],
            "blocked".to_string(),
            json!(true),
            None,
            false,
            true,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(matches!(
            g.apply(&mut body).await,
            GuardrailResult::Block(_)
        ));
    }

    #[tokio::test]
    async fn http_sends_custom_headers() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/classify")
                .header("x-api-key", "secret");
            then.status(200).json_body(json!({"blocked": false}));
        });
        let g = HttpCustomGuardrail::new(
            "h",
            format!("{}/classify", server.base_url()),
            None,
            vec![("x-api-key".to_string(), "secret".to_string())],
            "blocked".to_string(),
            json!(true),
            None,
            true,
            true,
        )
        .unwrap();
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let _ = g.apply(&mut body).await;
        m.assert();
    }

    #[tokio::test]
    async fn unknown_type_rejected() {
        let res = CustomGuardrail::from_config(
            &CustomGuardrailConfig {
                name: "x".to_string(),
                guardrail_type: "unknown".to_string(),
                config: json!({}),
            },
            false,
        );
        assert!(res.is_err());
    }

    #[test]
    fn cap_remote_payload_passes_short_text_through() {
        let s = cap_remote_payload("hello");
        assert_eq!(s, "hello");
    }

    #[test]
    fn cap_remote_payload_truncates_long_text_with_marker() {
        // A payload larger than MAX_REMOTE_PAYLOAD_BYTES is truncated to
        // <= MAX_REMOTE_PAYLOAD_BYTES (plus the marker) on a char boundary.
        let big = "a".repeat(MAX_REMOTE_PAYLOAD_BYTES + 4096);
        let capped = cap_remote_payload(&big);
        assert!(capped.len() < big.len(), "must truncate");
        assert!(
            capped.len() <= MAX_REMOTE_PAYLOAD_BYTES + REMOTE_PAYLOAD_TRUNCATION_MARKER.len(),
            "must not exceed cap + marker: got {} bytes",
            capped.len()
        );
        assert!(
            capped.ends_with(REMOTE_PAYLOAD_TRUNCATION_MARKER),
            "must end with truncation marker"
        );
    }

    #[test]
    fn cap_remote_payload_truncates_on_char_boundary() {
        // A multibyte character at the cap boundary must not be split.
        let head = "a".repeat(MAX_REMOTE_PAYLOAD_BYTES - 1);
        let big = format!("{head}あいうえお");
        let capped = cap_remote_payload(&big);
        // The capped string must be valid UTF-8 (cap_remote_payload returns
        // a String, so this is enforced by the type, but assert to be
        // explicit).
        assert!(std::str::from_utf8(capped.as_bytes()).is_ok());
        assert!(capped.ends_with(REMOTE_PAYLOAD_TRUNCATION_MARKER));
    }

    #[tokio::test]
    async fn http_guardrail_truncates_large_body() {
        // A 1 MiB user message must be capped before being POSTed to the
        // upstream classifier — the received body should contain the
        // truncation marker and be roughly MAX_REMOTE_PAYLOAD_BYTES (not
        // 1 MiB). We assert presence of the truncation marker (httpmock
        // matchers verify the body substring without exposing the raw
        // request).
        let server = MockServer::start();
        let truncated_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/classify")
                .body_contains("truncated by oxllm guardrail");
            then.status(200).json_body(json!({"blocked": false}));
        });
        // Sanity: a fully-sent 1 MiB body would NOT contain the marker; if
        // it did, that would still satisfy the truncated_mock above. To
        // also assert the body is small, we register a fall-through mock
        // that matches when the marker is absent and fails the test by
        // returning a different status. We rely on the first matching
        // mock being served.
        let untruncated_mock = server.mock(|when, then| {
            when.method(POST).path("/classify");
            then.status(200).json_body(json!({"blocked": false}));
        });

        let g = HttpCustomGuardrail::new(
            "h",
            format!("{}/classify", server.base_url()),
            None,
            vec![],
            "blocked".to_string(),
            json!(true),
            None,
            true,
            true,
        )
        .unwrap();
        let big = "x".repeat(1024 * 1024);
        let mut body = json!({"messages": [{"role": "user", "content": big}]});
        let _ = g.apply(&mut body).await;

        assert_eq!(
            truncated_mock.hits(),
            1,
            "the upstream classifier should have received the truncation marker"
        );
        assert_eq!(
            untruncated_mock.hits(),
            0,
            "the untruncated branch must not match — the body must have been capped"
        );
    }

    #[test]
    fn cap_forwarded_text_segments_passes_short_segments() {
        let mut segs = vec!["hello".to_string(), "world".to_string()];
        cap_forwarded_text_segments(&mut segs);
        assert_eq!(segs, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn cap_forwarded_text_segments_truncates_at_boundary() {
        // Total content exceeds the cap → the segment at the boundary gets
        // the marker; subsequent segments are cleared.
        let mut segs = vec![
            "a".repeat(MAX_REMOTE_PAYLOAD_BYTES),
            "b".repeat(4096),
            "c".to_string(),
        ];
        cap_forwarded_text_segments(&mut segs);
        // First segment: budget was MAX, it is exactly MAX, so it survives
        // intact and budget drops to 0.
        assert_eq!(segs[0].len(), MAX_REMOTE_PAYLOAD_BYTES);
        // Second segment: budget is 0, len > 0, so it gets truncated to the
        // marker only.
        assert!(segs[1].ends_with(REMOTE_PAYLOAD_TRUNCATION_MARKER));
        assert!(segs[1].len() <= REMOTE_PAYLOAD_TRUNCATION_MARKER.len());
        // Third segment: truncated flag is set, so it is cleared.
        assert_eq!(segs[2], "");
    }

    #[test]
    fn cap_forwarded_text_segments_handles_multibyte_boundary() {
        // A multibyte character straddling the cap boundary must not be split.
        let head = "a".repeat(MAX_REMOTE_PAYLOAD_BYTES - 1);
        let mut segs = vec![format!("{head}あいうえお")];
        cap_forwarded_text_segments(&mut segs);
        assert!(std::str::from_utf8(segs[0].as_bytes()).is_ok());
        assert!(segs[0].ends_with(REMOTE_PAYLOAD_TRUNCATION_MARKER));
    }
}
