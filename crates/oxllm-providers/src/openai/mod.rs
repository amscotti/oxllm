//! OpenAI provider — the identity transform.
//!
//! Since oxllm speaks the OpenAI wire format, the OpenAI provider's transforms
//! are largely pass-through. This is also the reference implementation showing
//! the expected shape of a provider module.

use std::collections::HashMap;

use oxllm_core::Error;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use serde_json::{Value, json};

/// The OpenAI provider. Implements `ChatProvider` as a near-identity transform.
#[derive(Debug, Clone, Default)]
pub struct OpenAiProvider;

/// Strip a single leading `openai/` prefix from a model id.
fn strip_openai_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("openai", rest)) => rest,
        _ => model,
    }
}

/// Rewrite the request body's `model` field from the deployment's configured
/// upstream model id (with `openai/` stripped).
fn rewrite_model(request: &Value, deployment: &Deployment) -> Result<Value> {
    let mut body = request.clone();
    if let Some(obj) = body.as_object_mut() {
        let configured = strip_openai_prefix(&deployment.oxllm_params.model);
        // Wildcard configured models: `*` (any) or `<provider>/*` (provider
        // wildcard). Extract the actual model id from the request body.
        let is_wildcard = configured == "*" || configured.ends_with("/*");
        let upstream = if is_wildcard {
            match request
                .get("model")
                .and_then(Value::as_str)
                .map(strip_openai_prefix)
            {
                Some(m) => m,
                None => {
                    return Err(Error::BadRequest(format!(
                        "wildcard deployment '{configured}' requires a `model` field in the request body"
                    )));
                }
            }
        } else {
            configured
        };
        obj.insert("model".to_string(), Value::String(upstream.to_string()));
        // OpenAI's API only accepts `metadata` when `store` is explicitly
        // `true` — any other request carrying a non-empty `metadata` object
        // fails with "'metadata' parameter is only allowed when 'store' is
        // true". The gateway consumes `metadata.tags` and `metadata.user_id`
        // itself (routing/metering + session extraction), and `metadata`
        // carries no model behavior, so drop it unless the caller opted
        // into storage. (The `metadata.tags` list is already removed one
        // layer up by the host's gateway-control strip; this drops the
        // remainder so it cannot 400.)
        let store_on = obj.get("store").and_then(Value::as_bool).unwrap_or(false);
        if !store_on {
            obj.remove("metadata");
        }
    }
    Ok(body)
}

/// Whether a model is an OpenAI reasoning model that rejects a non-default
/// `temperature` (o1, o3, o4, gpt-5*, gpt-6*, and any model tagged
/// `-reasoning`).
///
/// The `gpt-5-chat*` / `gpt-6-chat*` families are regular chat models: they
/// accept `temperature` and do not take `reasoning_effort`, so they stay off
/// this path.
fn is_reasoning_model(model: &str) -> bool {
    let leaf = model_leaf(model);
    is_o_series_lowered(&leaf) || is_gpt_reasoning_series(&leaf) || leaf.contains("-reasoning")
}

#[cfg(feature = "azure")]
pub(crate) fn has_known_reasoning_family(model: &str) -> bool {
    let leaf = model_leaf(model);
    is_o_series_lowered(&leaf) || is_gpt_reasoning_series(&leaf)
}

/// Whether a model is an OpenAI o-series reasoning model (o1, o3, o4) — the
/// strict subset that always rejects sampling/logprob params. GPT-5 only
/// rejects them when `reasoning_effort` is set to a non-`none` value.
fn is_o_series(model: &str) -> bool {
    is_o_series_lowered(&model.to_lowercase())
}

fn is_o_series_lowered(lower: &str) -> bool {
    matches_model_segment(lower, "o1")
        || matches_model_segment(lower, "o3")
        || matches_model_segment(lower, "o4")
}

/// True if `s` equals `prefix`, or begins with `prefix` followed by `-`.
/// Avoids false positives on names like `o1foo` or `o3foo` while still
/// matching `o1`, `o1-mini`, `o3-custom`, etc.
fn matches_model_segment(s: &str, prefix: &str) -> bool {
    match s.strip_prefix(prefix) {
        Some(rest) => rest.is_empty() || rest.starts_with('-'),
        None => false,
    }
}

/// Extract the effective `reasoning_effort` level as a string, accepting both
/// the plain string form (`"none"`) and the Responses-API dict form
/// (`{"effort": "none", "summary": "detailed"}`). Returns `None` when the field
/// is absent or has an unrecognized shape.
fn effort_level(body: &Value) -> Option<&str> {
    match body.get("reasoning_effort")? {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => o.get("effort").and_then(Value::as_str),
        _ => None,
    }
}

/// Model id used for capability checks: last path segment, Mantle's
/// `openai.` prefix removed, and a trailing snapshot date removed.
fn model_leaf(model: &str) -> String {
    let mut leaf = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    if let Some((_, rest)) = leaf.rsplit_once(".openai.") {
        leaf = rest.to_string();
    } else if let Some(rest) = leaf.strip_prefix("openai.") {
        leaf = rest.to_string();
    }
    if let Some(base) = strip_snapshot_date(&leaf) {
        leaf = base;
    }
    leaf
}

fn strip_snapshot_date(model: &str) -> Option<String> {
    let bytes = model.as_bytes();
    let is_digit = |b: u8| b.is_ascii_digit();
    if bytes.len() >= 12 {
        let s = &bytes[bytes.len() - 11..];
        let iso = s[0] == b'-'
            && is_digit(s[1])
            && is_digit(s[2])
            && is_digit(s[3])
            && is_digit(s[4])
            && s[5] == b'-'
            && is_digit(s[6])
            && is_digit(s[7])
            && s[8] == b'-'
            && is_digit(s[9])
            && is_digit(s[10]);
        if iso {
            return Some(model[..model.len() - 11].to_string());
        }
    }
    if bytes.len() >= 10 {
        let s = &bytes[bytes.len() - 9..];
        let compact = s[0] == b'-' && s[1..].iter().all(|b| is_digit(*b));
        if compact {
            return Some(model[..model.len() - 9].to_string());
        }
    }
    None
}

fn is_gpt_chat_model(leaf: &str) -> bool {
    (leaf.starts_with("gpt-5") || leaf.starts_with("gpt-6"))
        && leaf.split('-').any(|segment| segment == "chat")
}

fn is_gpt_reasoning_series(leaf: &str) -> bool {
    (leaf.starts_with("gpt-5") || leaf.starts_with("gpt-6")) && !is_gpt_chat_model(leaf)
}

fn leaf_excludes_none(leaf: &str) -> bool {
    leaf.contains("codex") || leaf.ends_with("-pro") || leaf.contains("-chat")
}

/// `(major, minor)` for `gpt-5`, `gpt-5.4-mini`, `gpt-6-luna`. Minor is
/// absent when the name has no `.` version (`gpt-5`, `gpt-6-luna`).
fn gpt_series_version(leaf: &str) -> Option<(u32, Option<u32>)> {
    let rest = leaf.strip_prefix("gpt-")?;
    let major_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if major_end == 0 {
        return None;
    }
    let major: u32 = rest[..major_end].parse().ok()?;
    let after = &rest[major_end..];
    let minor = if let Some(after) = after.strip_prefix('.') {
        let n = after
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after.len());
        if n == 0 {
            None
        } else {
            after[..n].parse().ok()
        }
    } else {
        None
    };
    Some((major, minor))
}

pub(crate) fn supports_explicit_prompt_cache(model: &str) -> bool {
    gpt_series_version(&model_leaf(model))
        .is_some_and(|(major, minor)| (major, minor.unwrap_or(0)) >= (5, 6))
}

fn effort_resolves_to_none(leaf: &str, body: &Value) -> bool {
    match effort_level(body) {
        Some(level) => level == "none",
        None => defaults_to_none_effort(leaf),
    }
}

/// Chat Completions rejects function tools on the gpt-5.6 family unless
/// reasoning is explicitly off. Those models default reasoning on, so an
/// omitted `reasoning_effort` plus a non-empty `tools` array is sent as
/// `reasoning_effort: "none"`. An explicit effort is left unchanged.
fn chat_tools_reject_default_reasoning(leaf: &str) -> bool {
    leaf.starts_with("gpt-5.6") && supports_none_reasoning_effort(leaf)
}

fn request_has_tools(body: &Value) -> bool {
    body.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
}

/// Drop parameters the OpenAI reasoning APIs reject, and map `max_tokens`
/// onto `max_completion_tokens`. No-op for non-reasoning models.
pub(crate) fn apply_openai_reasoning_limits(body: &mut Value, model: &str) {
    if !is_reasoning_model(model) {
        return;
    }
    let leaf = model_leaf(model);
    if chat_tools_reject_default_reasoning(&leaf)
        && effort_level(body).is_none()
        && request_has_tools(body)
    {
        if let Some(obj) = body.as_object_mut() {
            obj.insert(
                "reasoning_effort".to_string(),
                Value::String("none".to_string()),
            );
        }
    }
    let strip_temp = body
        .get("temperature")
        .is_some_and(|t| t.as_f64() != Some(1.0))
        && should_strip_temperature(model, body);
    let strip_sampling = should_strip_reasoning_unsupported(model, body);
    let drop_always = is_o_series_lowered(&leaf) || is_gpt_reasoning_series(&leaf);
    apply_openai_token_limit_alias(body);
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    if strip_temp {
        obj.remove("temperature");
    }
    if drop_always {
        for key in ALWAYS_DROPPED_ON_REASONING_SERIES {
            obj.remove(*key);
        }
    }
    if strip_sampling {
        for key in SAMPLING_PARAMS_WHEN_REASONING {
            obj.remove(*key);
        }
    }
    if let Some(Value::Object(dict)) = obj.get("reasoning_effort") {
        if let Some(Value::String(s)) = dict.get("effort") {
            let s = s.clone();
            obj.insert("reasoning_effort".to_string(), Value::String(s));
        }
    }
    if is_o_series(&leaf) {
        if let Some(messages) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
            for msg in messages.iter_mut() {
                if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
                    msg["role"] = Value::String("developer".to_string());
                }
            }
        }
    }
}

/// Map a token-limit alias after the caller has identified a compatible API.
/// Token-limit support and reasoning/sampling capabilities are independent.
pub(crate) fn apply_openai_token_limit_alias(body: &mut Value) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    if object.contains_key("max_completion_tokens") {
        object.remove("max_tokens");
    } else if let Some(tokens) = object.remove("max_tokens") {
        object.insert("max_completion_tokens".into(), tokens);
    }
}

/// Whether the model accepts `reasoning_effort: "none"`.
///
/// Accepting `"none"` is not the same as defaulting to it. gpt-5.1 / 5.2 /
/// 5.4 default to `"none"`; gpt-5.5, gpt-5.6, gpt-6-luna, and gpt-6-sol accept
/// `"none"` but default to reasoning on. gpt-6-astra does not accept `"none"`.
/// Codex, pro, and chat variants do not accept it.
fn supports_none_reasoning_effort(model: &str) -> bool {
    let leaf = model_leaf(model);
    if leaf_excludes_none(&leaf) {
        return false;
    }
    if leaf.starts_with("gpt-6-luna")
        || leaf.starts_with("gpt-6-sol")
        || leaf.starts_with("gpt-6.1-sol")
    {
        return true;
    }
    if leaf.starts_with("gpt-6") {
        return false;
    }
    matches!(gpt_series_version(&leaf), Some((5, Some(minor))) if minor >= 1)
}

/// Whether an omitted `reasoning_effort` means reasoning is off.
///
/// Only gpt-5.1, gpt-5.2, and gpt-5.4 (including mini / nano / dated
/// snapshots, excluding codex, pro, and chat) default to `"none"`. Every
/// later model that merely *accepts* `"none"` still reasons unless the
/// caller sets it.
fn defaults_to_none_effort(model: &str) -> bool {
    let leaf = model_leaf(model);
    supports_none_reasoning_effort(&leaf)
        && matches!(gpt_series_version(&leaf), Some((5, Some(minor))) if matches!(minor, 1 | 2 | 4))
}

/// Whether a non-default `temperature` must be stripped.
///
/// A temperature of exactly 1 is kept by the caller. Otherwise:
/// o-series always strip; a gpt-5/gpt-6 model keeps it only when reasoning
/// resolves to `"none"` on a model that accepts that level; every other
/// reasoning model strips.
fn should_strip_temperature(model: &str, body: &Value) -> bool {
    let leaf = model_leaf(model);
    if is_o_series_lowered(&leaf) {
        return true;
    }
    if is_gpt_reasoning_series(&leaf) {
        return !(supports_none_reasoning_effort(&leaf) && effort_resolves_to_none(&leaf, body));
    }
    true
}

/// Whether `top_p` / `logprobs` / `top_logprobs` must be stripped.
///
/// o-series always strip. gpt-5/gpt-6 models that do not accept `"none"`
/// always strip, including when the caller sends `reasoning_effort: "none"`.
/// Models that accept `"none"` keep these params only while effort resolves
/// to `"none"`.
fn should_strip_reasoning_unsupported(model: &str, body: &Value) -> bool {
    let leaf = model_leaf(model);
    if is_o_series_lowered(&leaf) {
        return true;
    }
    if is_gpt_reasoning_series(&leaf) {
        if !supports_none_reasoning_effort(&leaf) {
            return true;
        }
        return !effort_resolves_to_none(&leaf, body);
    }
    false
}

/// `top_p` and logprob params. Gated by [`should_strip_reasoning_unsupported`].
const SAMPLING_PARAMS_WHEN_REASONING: &[&str] = &["logprobs", "top_logprobs", "top_p"];

/// Params the gpt-5/gpt-6 and o-series APIs reject even when reasoning is off.
const ALWAYS_DROPPED_ON_REASONING_SERIES: &[&str] =
    &["frequency_penalty", "presence_penalty", "logit_bias"];

/// Normalize null token counters in the response's `usage` object to `0`.
///
/// Some upstreams (notably OpenAICompat shims) return `"prompt_tokens": null`
/// when a request fails or is rate-limited. Downstream `extract_usage` treats
/// `null` as "missing" and substitutes `0` already, but other consumers
/// (proxies, logging hooks) read the raw body; rewriting `null` → `0` here
/// keeps the wire shape consistent across providers.
pub(crate) fn sanitize_null_tokens(body: &mut Value) {
    let Some(usage) = body.get_mut("usage").and_then(|u| u.as_object_mut()) else {
        return;
    };
    for key in &["prompt_tokens", "completion_tokens", "total_tokens"] {
        if usage.get(*key).map(|v| v.is_null()).unwrap_or(false) {
            usage.insert(key.to_string(), json!(0));
        }
    }
    for detail_key in &["prompt_tokens_details", "completion_tokens_details"] {
        if let Some(details) = usage.get_mut(*detail_key).and_then(|d| d.as_object_mut()) {
            for (k, v) in details.clone() {
                if v.is_null() {
                    details.insert(k, json!(0));
                }
            }
        }
    }
}

impl ChatProvider for OpenAiProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        crate::util::append_operation_path(base, "chat/completions")
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        let bearer = match creds {
            ResolvedCredentials::ApiKey { bearer } => bearer.as_str(),
            other => match other.best_effort_bearer() {
                Some(b) => b,
                None => return,
            },
        };
        headers.insert("authorization".to_string(), format!("Bearer {bearer}"));
        oxllm_core::provider::apply_auth_overrides(deployment, creds, headers);

        // Optional OpenAI multi-organization / project routing. These are
        // captured by `OxllmParams.extra` (a flattened serde_json::Map) when
        // users set `organization` / `project` on a deployment. Header names
        // are lowercase because the request pipeline lowercases all header
        // keys before sending.
        if let Some(org) = deployment
            .oxllm_params
            .extra
            .get("organization")
            .and_then(Value::as_str)
        {
            headers.insert("openai-organization".to_string(), org.to_string());
        }
        if let Some(project) = deployment
            .oxllm_params
            .extra
            .get("project")
            .and_then(Value::as_str)
        {
            headers.insert("openai-project".to_string(), project.to_string());
        }
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let mut out = rewrite_model(request, deployment)?;
        let model = out
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        apply_openai_reasoning_limits(&mut out, &model);
        crate::prompt_cache::apply(&mut out, request, deployment);
        Ok(out)
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let mut out = body.clone();
        sanitize_null_tokens(&mut out);
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        crate::openai_compat::sse::parse_openai_stream_chunk(chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        None
    }
}

impl EmbeddingProvider for OpenAiProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        crate::util::append_operation_path(base, "embeddings")
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        ChatProvider::apply_auth(self, deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        rewrite_model(request, deployment)
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let mut out = body.clone();
        sanitize_null_tokens(&mut out);
        Ok(out)
    }
}

impl Provider for OpenAiProvider {
    fn id(&self) -> ProviderId {
        "openai"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(
            endpoint,
            oxllm_core::Endpoint::ChatCompletions
                | oxllm_core::Endpoint::Embeddings
                | oxllm_core::Endpoint::ImagesGenerations
                | oxllm_core::Endpoint::AudioSpeech
                | oxllm_core::Endpoint::AudioTranscriptions
                | oxllm_core::Endpoint::Moderations
        )
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        Some(self)
    }
}

#[cfg(all(test, feature = "openai"))]
mod tests;
