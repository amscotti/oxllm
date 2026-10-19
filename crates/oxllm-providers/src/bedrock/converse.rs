//! AWS Bedrock **Converse API** transform — OpenAI ⇄ Converse.
//!
//! Converse is Bedrock's unified chat surface: one request/response shape
//! across every model family (Anthropic, Meta, Mistral, Nova, AI21, …).
//! This replaces the per-vendor `InvokeModel` body munging with a single
//! canonical mapping. The host still owns SigV4 signing and the wire
//! framing; this module is pure (no I/O).
//!
//! Wire targets (URL is assembled by the parent `BedrockProvider`):
//!   * non-streaming: `POST …/model/{id}/converse`
//!   * streaming:     `POST …/model/{id}/converse-stream`
//!
//! Streaming wire format is the binary `application/vnd.amazon.eventstream`
//! framing — the same family InvokeModel uses. The host strips the framing
//! and feeds this transform one event payload at a time. Each payload is
//! a JSON object whose top-level shape identifies the event type
//! (`messageStart`, `contentBlockDelta`, `messageStop`, …). When the
//! payload arrives boto-SDK-style wrapped as `{"<eventType>": {…}}`, the
//! wrapper is unwrapped transparently.
//!
//! Reference: AWS Bedrock Runtime `Converse` / `ConverseStream` API.
//! Field names follow the AWS docs (camelCase: `toolUse`, `toolResult`,
//! `inferenceConfig`, `stopReason`, `outputTokens`, …).

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

use oxllm_core::Result;
use serde_json::{Map, Value, json};

use crate::anthropic::sanitize_tool_input_schema;

// =============================================================================
// Request transform
// =============================================================================

/// Transform an OpenAI Chat Completion request body into a Bedrock Converse
/// request body.
///
/// `extra` is the deployment's `oxllm_params.extra` map — the source of
/// deployment-level configuration like `guardrail_id` / `guardrail_version`
/// and `performance_config`. These are injected into the Converse body (not
/// URL query params, which is the InvokeModel convention).
///
/// Mapping summary (see `docs/plans/bedrock-converse-plan.md`):
///
/// | OpenAI                          | Converse                                  |
/// |---------------------------------|-------------------------------------------|
/// | `messages[role=system/developer]`| `system[]: [{text: …}]`                   |
/// | `messages[role=user]`           | `messages[]: {role:user, content:[…]}`    |
/// | `messages[role=assistant]`      | `messages[]: {role:assistant, …}`         |
/// | `messages[role=tool]`           | `messages[]: {role:user, toolResult: …}`  |
/// | `max_tokens`/`temperature`/…    | `inferenceConfig`                         |
/// | `tools[]`                       | `toolConfig.tools[]` (`toolSpec`)         |
/// | `tool_choice`                   | `toolConfig.toolChoice`                   |
/// | `reasoning_effort` (Anthropic)  | `additionalModelRequestFields.thinking`   |
/// | `response_format: json_schema`  | `outputConfig.textFormat` (provider-validated) |
/// | `response_format: json_object`  | native supplied schema, or unsupported error |
/// | `extra.guardrail_id`            | `guardrailConfig`                         |
/// | `extra.performance_config`      | `performanceConfig`                       |
///
/// `model` is the resolved Bedrock model id (already stripped of any
/// `bedrock/` prefix). The model id is NOT placed in the body — it lives
/// in the URL path (`/model/{id}/converse`), matching the Converse spec.
/// An opaque inference profile may declare its underlying model in
/// `extra.base_model`; the provider also accepts `model_info.base_model`.
/// These hints select feature mappings only, never the runtime model id.
pub fn transform_request_converse(
    request: &Value,
    model: &str,
    extra: &Map<String, Value>,
) -> Result<Value> {
    transform_request_converse_with_base_model(request, model, extra, None)
}

pub(super) fn transform_request_converse_with_base_model(
    request: &Value,
    runtime_model: &str,
    extra: &Map<String, Value>,
    metadata_base_model: Option<&str>,
) -> Result<Value> {
    let resolved_feature_model = resolve_feature_model(runtime_model, extra, metadata_base_model)?;
    let feature_model = resolved_feature_model.as_deref().unwrap_or("");
    if request
        .get("tools")
        .is_some_and(|tools| !tools.is_null() && !tools.is_array())
    {
        return Err(oxllm_core::Error::BadRequest(
            "tools must be an array".into(),
        ));
    }
    validate_profile_reasoning(request, runtime_model, feature_model)?;
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut converse_messages: Vec<Value> = Vec::new();

    // When guardrails are configured, the trailing run of consecutive user
    // messages is auto-tagged so its text blocks are wrapped in
    // `guardContent` for selective guardrail scanning. Mirrors the reference
    // `_convert_consecutive_user_messages_to_guarded_text`.
    let guardrails_active = build_guardrail_config(extra).is_some();
    let mut request_messages: Vec<Value> = request
        .get("messages")
        .and_then(Value::as_array)
        .map(|a| a.to_vec())
        .unwrap_or_default();
    if guardrails_active {
        tag_trailing_user_messages_as_guarded(&mut request_messages);
    }

    for msg in &request_messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        match role {
            "system" | "developer" => {
                accumulate_system_blocks(msg, &mut system_blocks, feature_model)
            }
            "user" => {
                let blocks = convert_user_content(msg, feature_model)?;
                if !blocks.is_empty() {
                    converse_messages.push(json!({"role": "user", "content": blocks}));
                }
            }
            "assistant" => {
                let blocks = convert_assistant_content(
                    msg,
                    is_anthropic_model(feature_model),
                    feature_model,
                )?;
                if !blocks.is_empty() {
                    converse_messages.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            "tool" => {
                let mut content = convert_tool_result(msg, feature_model)?;
                // A `cache_control` hint on the tool message (message-
                // level or inside its content array) injects a trailing
                // `cachePoint` sibling block — matches the reference tool-
                // result cache handling.
                if has_cache_control(msg) || tool_result_content_has_cache_control(msg) {
                    content.push(build_cache_point_block(msg, feature_model));
                }
                converse_messages.push(json!({"role": "user", "content": content}));
            }
            _ => {
                // Unknown role — treat as user (lenient fallback).
                let blocks = convert_user_content(msg, feature_model)?;
                if !blocks.is_empty() {
                    converse_messages.push(json!({"role": "user", "content": blocks}));
                }
            }
        }
    }

    let modify_params = extra.get("modify_params").and_then(Value::as_bool) == Some(true);
    let converse_messages = merge_consecutive_messages(converse_messages, modify_params);
    let mut out = Map::new();
    out.insert("messages".to_string(), Value::Array(converse_messages));
    if !system_blocks.is_empty() {
        out.insert("system".to_string(), Value::Array(system_blocks));
    }
    // The reference always emits `inferenceConfig` (possibly empty) — match that
    // so request shapes are stable across deployments.
    out.insert(
        "inferenceConfig".to_string(),
        build_inference_config(request, feature_model),
    );
    apply_response_format(request, &mut out)?;
    if let Some(tc) = build_tool_config(request, feature_model)? {
        out.insert("toolConfig".to_string(), tc);
    }
    // Deployment-level guardrail configuration (Converse puts these in the
    // body, unlike InvokeModel which uses URL query params).
    if let Some(mut gc) = build_guardrail_config(extra) {
        if let Some(trace) = build_guardrail_trace(extra) {
            gc["trace"] = trace;
        }
        out.insert("guardrailConfig".to_string(), gc);
    }
    // Deployment-level performance configuration.
    if let Some(pc) = build_performance_config(extra) {
        out.insert("performanceConfig".to_string(), pc);
    }
    // `reasoning_effort` → `additionalModelRequestFields.thinking` (Anthropic
    // only). Skipped when the caller already set `thinking` directly.
    apply_reasoning_effort(request, feature_model, &mut out);
    // `service_tier` (OpenAI string) → top-level `serviceTier: {type: …}`.
    apply_service_tier(request, &mut out);
    // `context_management` → `additionalModelRequestFields.context_management`.
    apply_context_management(request, &mut out);
    // Tool name sanitization (Converse: `^[a-zA-Z0-9_-]{1,64}$`). Build the
    // forward map from the caller's tools and rewrite
    // names in the already-built `toolConfig` + assistant `toolUse` blocks.
    // The reverse map is NOT stashed here: the response/stream transforms
    // rebuild it from the original request via [`reverse_map_from_request`]
    // (see [`crate::bedrock::BedrockProvider::transform_response_with_request`]
    // and [`crate::bedrock::BedrockProvider::transform_stream_chunk_stateful`])
    // so it survives concurrent and inter-thread request flows — a process-
    // wide cell cannot, since two in-flight requests with different tool sets
    // would clobber each other's reverse map.
    let forward = request
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| build_tool_name_maps(tools).0)
        .unwrap_or_default();
    // Pass-through Converse-native top-level fields when the caller has
    // pre-shaped them. These run last so explicit caller values take
    // precedence over the feature-level defaults above. Each entry maps the
    // request key (left) onto the canonical Converse field name (right) so
    // snake_case aliases (`guardrail_config`, …) are RENAMED instead of
    // forwarded verbatim — the Converse API only accepts the camelCase form
    // and would reject the snake_case key with a ValidationException.
    for (key, canonical) in PASS_THROUGH_TOP_LEVEL_KEYS {
        if let Some(v) = request.get(key) {
            if !v.is_null() {
                if *canonical == "outputConfig" {
                    if let Some(caller) = v.as_object() {
                        let mut caller = caller.clone();
                        caller.remove("format");
                        if caller.get("textFormat").is_some_and(Value::is_null) {
                            caller.remove("textFormat");
                        }
                        if caller.contains_key("textFormat")
                            && request
                                .pointer("/response_format/type")
                                .and_then(Value::as_str)
                                == Some("text")
                        {
                            return Err(oxllm_core::Error::BadRequest(format!(
                                "response_format text conflicts with {key}.textFormat"
                            )));
                        }
                        if let Some(native) = caller.get_mut("textFormat")
                            && let Some(mapped) = out
                                .get("outputConfig")
                                .and_then(|value| value.get("textFormat"))
                        {
                            *native = merge_text_formats(mapped, native).ok_or_else(|| {
                                oxllm_core::Error::BadRequest(format!(
                                    "requested output format conflicts with {key}.textFormat"
                                ))
                            })?;
                        }
                        let output = out
                            .entry(canonical.to_string())
                            .or_insert_with(|| json!({}));
                        if let Some(output) = output.as_object_mut() {
                            output.extend(caller);
                        }
                        continue;
                    }
                }
                out.insert(canonical.to_string(), v.clone());
            }
        }
    }
    // `thinking` is special: it is an advertised supported param for Claude
    // extended thinking, but the Converse API only accepts it nested under
    // `additionalModelRequestFields.thinking` — a top-level `thinking` field
    // is a ValidationException. Map it there (after the rename loop, so it
    // also lands after `apply_reasoning_effort`'s default, which already
    // defers to an explicit caller `thinking`).
    if let Some(t) = request.get("thinking") {
        if !t.is_null() {
            let amrf = out
                .entry("additionalModelRequestFields".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(map) = amrf {
                map.insert("thinking".to_string(), t.clone());
            }
        }
    }
    // A caller-supplied `additionalModelRequestFields` (or its snake_case
    // alias) is merged key-by-key into the object built above rather than
    // replacing it, so feature-mapped keys (`reasoning_effort` → `thinking`,
    // `context_management`) survive alongside the caller's own fields while
    // the caller's values win on conflict.
    for amrf_key in [
        "additionalModelRequestFields",
        "additional_model_request_fields",
    ] {
        if let Some(Value::Object(caller_amrf)) = request.get(amrf_key) {
            let amrf = out
                .entry("additionalModelRequestFields".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(map) = amrf {
                for (k, v) in caller_amrf {
                    if !v.is_null() {
                        map.insert(k.clone(), v.clone());
                    }
                }
            }
        }
    }
    // Model-specific choices are merged above; encode them in the same caller
    // vocabulary as tool specifications and the common Converse choice.
    if !forward.is_empty() {
        apply_tool_name_forward_map(&mut out, &forward);
    }
    apply_top_k(request, feature_model, &mut out)?;
    // Enforce the same budget window as the Anthropic adapter after merging
    // all explicit thinking aliases. Only inferred thinking may be omitted.
    clamp_thinking_budget(request, &mut out)?;
    // Auto-set `inferenceConfig.maxTokens` when thinking is enabled but no
    // explicit maxTokens was provided (Bedrock requires maxTokens >
    // budget_tokens).
    auto_set_max_tokens_for_thinking(&mut out);
    // Claude thinking accepts only temperature 1. Drop any other value
    // after thinking has been placed on the Converse body.
    drop_temperature_incompatible_with_thinking(&mut out);
    // Express serial-tool requests for Anthropic-family models through the
    // model-specific `additionalModelRequestFields`; other families have no
    // equivalent field.
    apply_parallel_tool_calls(feature_model, request, &mut out, &forward)?;

    if modify_params {
        repair_converse_history(request, &mut out);
    }

    Ok(Value::Object(out))
}

/// Group adjacent turns after role mapping, so several tool results from one
/// assistant turn share one user turn. Content metadata is retained, with
/// signed reasoning kept ahead of assistant text as required by the wire API.
fn merge_consecutive_messages(messages: Vec<Value>, modify_params: bool) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for mut message in messages {
        let assistant = message.get("role").and_then(Value::as_str) == Some("assistant");
        let tool_results = |message: &Value| {
            message
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|blocks| blocks.iter().any(|block| block.get("toolResult").is_some()))
        };
        let separate_results = modify_params
            && !assistant
            && merged.last().is_some_and(|previous| {
                previous.get("role") == message.get("role")
                    && tool_results(previous) != tool_results(&message)
            });
        if separate_results {
            merged.push(json!({"role": "assistant", "content": [{"text": "Continue."}]}));
        }
        if let Some(previous) = merged.last_mut()
            && previous.get("role") == message.get("role")
            && let Some(content) = previous.get_mut("content").and_then(Value::as_array_mut)
            && let Some(Value::Array(blocks)) = message.get_mut("content")
        {
            content.append(blocks);
            if assistant {
                content.sort_by_key(|block| block.get("reasoningContent").is_none());
            }
        } else {
            merged.push(message);
        }
    }
    merged
}

/// Bedrock requires tool configuration even when the tools only occur in
/// history. Recover minimal declarations and disable thinking for unsigned
/// historical tool-use turns rather than sending an invalid thinking request.
fn repair_converse_history(request: &Value, out: &mut Map<String, Value>) {
    let Some(messages) = out.get("messages").and_then(Value::as_array) else {
        return;
    };
    let mut names: Vec<&str> = Vec::new();
    let mut unsigned_tool_turn = false;
    for message in messages {
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        let has_reasoning = blocks
            .iter()
            .any(|block| block.get("reasoningContent").is_some());
        for block in blocks {
            if let Some(name) = block
                .get("toolUse")
                .and_then(|tool| tool.get("name"))
                .and_then(Value::as_str)
            {
                if !name.is_empty() && !names.contains(&name) {
                    names.push(name);
                }
                unsigned_tool_turn |= !has_reasoning;
            }
        }
    }
    let tools = names
        .into_iter()
        .map(|name| {
            json!({"toolSpec": {
                "name": name, "description": "Historical function declaration.",
                "inputSchema": {"json": {"type": "object"}}
            }})
        })
        .collect::<Vec<_>>();
    let explicit_none = request.get("tool_choice").is_some_and(|choice| {
        choice.as_str() == Some("none")
            || choice.get("type").and_then(Value::as_str) == Some("none")
    });
    if !tools.is_empty() && !out.contains_key("toolConfig") && !explicit_none {
        out.insert("toolConfig".into(), json!({"tools": tools}));
    }
    if unsigned_tool_turn
        && let Some(fields) = out
            .get_mut("additionalModelRequestFields")
            .and_then(Value::as_object_mut)
        && fields
            .get("thinking")
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, "enabled" | "adaptive"))
    {
        fields.insert("thinking".into(), json!({"type": "disabled"}));
    }
}

/// Profiles hide their source model. An explicit, concrete base-model hint is
/// an operator assertion about that source, not a replacement for the ARN.
fn resolve_feature_model<'a>(
    runtime_model: &'a str,
    extra: &'a Map<String, Value>,
    metadata_base_model: Option<&'a str>,
) -> Result<Option<Cow<'a, str>>> {
    let known_id =
        super::model_id_for_family(runtime_model).filter(|id| is_recognized_feature_model(id));
    if let Some(id) = known_id {
        return Ok(Some(Cow::Borrowed(id)));
    }
    if !is_inference_profile_arn(runtime_model) {
        return Ok(Some(Cow::Borrowed(runtime_model)));
    }
    let invalid_hint = || {
        oxllm_core::Error::Config(
            "Bedrock model_info.base_model/base_model must identify a recognized concrete underlying model, not an alias, wildcard, or opaque profile ARN".into(),
        )
    };
    let hint = match metadata_base_model {
        Some(hint) => Some(hint),
        None => match extra.get("base_model").filter(|value| !value.is_null()) {
            Some(value) => Some(value.as_str().ok_or_else(invalid_hint)?),
            None => None,
        },
    };
    let Some(hint) = hint else {
        return Ok(None);
    };
    if hint.is_empty() || hint.contains('*') || hint.chars().any(char::is_whitespace) {
        return Err(invalid_hint());
    }
    let hint = hint.strip_prefix("bedrock/").unwrap_or(hint);
    let normalized = if let Some(model) = hint.strip_prefix("anthropic/") {
        format!("anthropic.{model}")
    } else if hint.starts_with("claude-") {
        format!("anthropic.{hint}")
    } else {
        hint.to_string()
    };
    let normalized = super::strip_model_suffixes(&super::strip_model_suffixes(&normalized));
    let id = super::model_id_for_family(&normalized)
        .filter(|id| is_recognized_feature_model(id))
        .ok_or_else(invalid_hint)?;
    Ok(Some(Cow::Owned(id.to_string())))
}

fn is_recognized_feature_model(model: &str) -> bool {
    if !model
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | ':'))
    {
        return false;
    }
    if let Some(claude) = model.strip_prefix("anthropic.claude-") {
        let version = ["sonnet-", "opus-", "haiku-"]
            .iter()
            .find_map(|tier| claude.strip_prefix(*tier))
            .unwrap_or(claude);
        return version.starts_with(|ch: char| ch.is_ascii_digit());
    }
    [
        "amazon.nova-",
        "amazon.titan-",
        "meta.llama",
        "mistral.",
        "ai21.",
        "cohere.",
        "deepseek.",
        "qwen.",
        "zai.glm-",
        "openai.gpt-",
        "xai.grok-",
    ]
    .iter()
    .any(|prefix| {
        model
            .strip_prefix(*prefix)
            .is_some_and(|rest| !rest.is_empty())
    })
}

fn is_inference_profile_arn(model: &str) -> bool {
    let model = model.strip_prefix("bedrock/").unwrap_or(model);
    let mut parts = model.splitn(6, ':');
    parts.next() == Some("arn")
        && parts.next().is_some()
        && parts.next() == Some("bedrock")
        && parts.next().is_some()
        && parts.next().is_some()
        && parts.next().is_some_and(|resource| {
            resource.starts_with("application-inference-profile/")
                || resource.starts_with("inference-profile/")
        })
}

fn validate_profile_reasoning(
    request: &Value,
    runtime_model: &str,
    feature_model: &str,
) -> Result<()> {
    if !is_inference_profile_arn(runtime_model) {
        return Ok(());
    }
    let has_thinking = request
        .get("thinking")
        .is_some_and(|value| !value.is_null())
        || [
            "additionalModelRequestFields",
            "additional_model_request_fields",
        ]
        .iter()
        .any(|key| {
            request
                .get(*key)
                .and_then(|fields| fields.get("thinking"))
                .is_some_and(|value| !value.is_null())
        });
    let has_effort = request
        .get("reasoning_effort")
        .is_some_and(|value| !value.is_null());
    if (has_thinking && !is_anthropic_model(feature_model))
        || (has_effort && !(is_anthropic_model(feature_model) || is_nova2_model(feature_model)))
    {
        return Err(oxllm_core::Error::BadRequest(
            "thinking/reasoning_effort requires a supported underlying model; configure model_info.base_model or base_model for an opaque Bedrock profile".into(),
        ));
    }
    Ok(())
}

/// Caller-supplied request keys forwarded onto canonical top-level Converse
/// fields. Each entry is `(request key, Converse field)`. Most of these
/// correspond to Batch 3 features (`guardrailConfig`, `performanceConfig`,
/// `requestMetadata`, `outputConfig`, `serviceTier`). The snake_case forms on
/// the left are accepted as aliases and RENAMED to the camelCase Converse
/// field (right) — forwarding them verbatim would trigger a
/// ValidationException. Two deliberately absent keys:
///
/// - `thinking`: NOT a top-level Converse field. Claude extended thinking is
///   configured under `additionalModelRequestFields.thinking`; the transform
///   maps a caller `thinking` there (and `clamp_thinking_budget` /
///   `auto_set_max_tokens_for_thinking` then see it).
/// - `additionalModelRequestFields` (and its snake_case alias): merged
///   key-by-key into the object the feature mappings above build, instead of
///   being forwarded as a separate top-level key.
///
/// `service_tier` (snake_case) is deliberately absent — it is mapped to
/// `serviceTier` by [`apply_service_tier`] rather than forwarded verbatim.
const PASS_THROUGH_TOP_LEVEL_KEYS: &[(&str, &str)] = &[
    ("guardrailConfig", "guardrailConfig"),
    ("guardrail_config", "guardrailConfig"),
    ("performanceConfig", "performanceConfig"),
    ("performance_config", "performanceConfig"),
    ("requestMetadata", "requestMetadata"),
    ("request_metadata", "requestMetadata"),
    ("outputConfig", "outputConfig"),
    ("output_config", "outputConfig"),
    ("serviceTier", "serviceTier"),
];

/// Use the same known-family detection for all Anthropic-specific fields.
/// Opaque application profile ARNs cannot establish Anthropic support.
fn is_anthropic_model(model: &str) -> bool {
    super::detect_bedrock_provider(model) == super::BedrockVendor::Anthropic
}

/// Is this a known Bedrock Amazon Nova model id?
/// Used to gate Nova-only features like web-search grounding
/// (`web_search_options` → `systemTool: {name: "nova_grounding"}`).
fn is_nova_model(model: &str) -> bool {
    super::detect_bedrock_provider(model) == super::BedrockVendor::Nova
}

/// Heuristic: is this a Bedrock Amazon **Nova 2** model id? Nova 2 models
/// carry `nova-2` in the id (e.g. `amazon.nova-2-lite-v1:0`,
/// `us.amazon.nova-2-pro-preview-20251202-v1:0`). Used to gate the Nova 2
/// `reasoningConfig` mapping (Nova 2 uses a different reasoning shape than
/// Anthropic's `thinking`). Opaque ARNs do not establish this capability.
fn is_nova2_model(model: &str) -> bool {
    super::model_id_for_family(model).is_some_and(|id| id.starts_with("amazon.nova-2-"))
}

/// Converse's common inferenceConfig has no top-k member. Claude accepts the
/// flat model-native top_k; Nova v1 text models use inferenceConfig.topK in
/// additionalModelRequestFields. Other families/versions need a verified
/// mapping before this adapter can advertise the same behavior.
fn apply_top_k(request: &Value, model: &str, out: &mut Map<String, Value>) -> Result<()> {
    // Post-4.6 Claude models reject top_k outright; omit it like the
    // temperature/topP gates in build_inference_config.
    if crate::anthropic::model_rejects_sampling_params(model) {
        return Ok(());
    }
    let Some(value) = request.get("top_k").filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let top_k = value.as_u64().ok_or_else(|| {
        oxllm_core::Error::BadRequest("Bedrock top_k must be a non-negative integer".into())
    })?;
    let nova_top_k = super::model_id_for_family(model).is_some_and(|id| {
        [
            "amazon.nova-micro-v1",
            "amazon.nova-lite-v1",
            "amazon.nova-pro-v1",
            "amazon.nova-premier-v1",
        ]
        .iter()
        .any(|prefix| {
            id.strip_prefix(*prefix)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with(':'))
        })
    });
    if !is_anthropic_model(model) && !nova_top_k {
        return Err(oxllm_core::Error::BadRequest(format!(
            "top_k has no supported Bedrock Converse mapping for model `{model}`; an opaque profile requires model_info.base_model or base_model identifying a supported underlying model"
        )));
    }
    let fields = out
        .entry("additionalModelRequestFields".to_string())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| {
            oxllm_core::Error::BadRequest("additionalModelRequestFields must be an object".into())
        })?;
    if nova_top_k {
        let inference = fields
            .entry("inferenceConfig".to_string())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                oxllm_core::Error::BadRequest(
                    "Nova additionalModelRequestFields.inferenceConfig must be an object".into(),
                )
            })?;
        inference
            .entry("topK".to_string())
            .or_insert_with(|| json!(top_k));
    } else {
        fields
            .entry("top_k".to_string())
            .or_insert_with(|| json!(top_k));
    }
    Ok(())
}

/// Build the Converse `guardrailConfig` body field from the deployment's
/// `extra` params. Accepts snake_case (`guardrail_id`,
/// `guardrail_version`) or AWS-native camelCase (`guardrailIdentifier`,
/// `guardrailVersion`). Returns `None` unless both id and version are
/// present (Bedrock requires both).
fn build_guardrail_config(extra: &Map<String, Value>) -> Option<Value> {
    let id = extra
        .get("guardrail_id")
        .and_then(Value::as_str)
        .or_else(|| extra.get("guardrailIdentifier").and_then(Value::as_str))?;
    let version = extra
        .get("guardrail_version")
        .and_then(Value::as_str)
        .or_else(|| extra.get("guardrailVersion").and_then(Value::as_str))?;
    Some(json!({
        "guardrailIdentifier": id,
        "guardrailVersion": version,
    }))
}

/// Build the Converse top-level `trace` field when the deployment enables
/// guardrail tracing. Accepts `guardrail_trace` (Converse-specific) or
/// `trace` (shared with the InvokeModel path). Returns `None` when the flag
/// is absent or false.
fn build_guardrail_trace(extra: &Map<String, Value>) -> Option<Value> {
    let enabled = extra
        .get("guardrail_trace")
        .and_then(Value::as_bool)
        .or_else(|| extra.get("trace").and_then(Value::as_bool))?;
    if enabled {
        Some(json!("enabled"))
    } else {
        None
    }
}

/// Build the Converse `performanceConfig` body field from the deployment's
/// `extra` params. Accepts `performance_config` (snake_case) or
/// `performanceConfig` (camelCase). Returns `None` when absent or null.
fn build_performance_config(extra: &Map<String, Value>) -> Option<Value> {
    let pc = extra
        .get("performance_config")
        .or_else(|| extra.get("performanceConfig"))?;
    if pc.is_null() { None } else { Some(pc.clone()) }
}

/// Map an OpenAI `reasoning_effort` string onto an Anthropic extended-thinking
/// token budget. Returns `None` for unknown values (no mapping applied).
///
/// | effort   | budget_tokens |
/// |----------|---------------|
/// | `low`    | 1024          |
/// | `medium` | 4096          |
/// | `high`   | 10000         |
fn reasoning_effort_to_budget(effort: &str) -> Option<u64> {
    match effort {
        "low" => Some(1024),
        "medium" => Some(4096),
        "high" => Some(10000),
        _ => None,
    }
}

/// Map OpenAI `reasoning_effort` → Converse
/// `additionalModelRequestFields.thinking` for Anthropic models, or
/// `additionalModelRequestFields.reasoningConfig` for Nova 2 models.
///
/// - Nova 2: `reasoning_effort` → `{reasoningConfig: {type:"enabled",
///   maxReasoningEffort: "low"|"medium"|"high"}}`. Nova 2 doesn't support
///   `max`/`xhigh`/unknown values — they clamp to `high`.
/// - Anthropic: `reasoning_effort` → `{thinking: {type:"enabled",
///   budget_tokens: …}}`. Skipped when the caller already set `thinking`
///   directly.
/// - Other vendors: silently dropped (reasoning config is vendor-specific).
fn apply_reasoning_effort(request: &Value, model: &str, out: &mut Map<String, Value>) {
    let Some(effort) = request.get("reasoning_effort").and_then(Value::as_str) else {
        return;
    };

    // Nova 2: map to reasoningConfig. Nova 2 only accepts low/medium/high;
    // clamp max/xhigh/unknown down to high.
    if is_nova2_model(model) {
        let clamped = match effort {
            "low" | "medium" | "high" => effort,
            _ => "high",
        };
        let amrf = out
            .entry("additionalModelRequestFields".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(map) = amrf {
            map.insert(
                "reasoningConfig".to_string(),
                json!({"type": "enabled", "maxReasoningEffort": clamped}),
            );
        }
        return;
    }

    if !is_anthropic_model(model) {
        return;
    }
    // Don't override an explicit `thinking` from the caller.
    if request
        .get("thinking")
        .is_some_and(|value| !value.is_null())
    {
        return;
    }
    let Some(budget) = reasoning_effort_to_budget(effort) else {
        return;
    };
    // Merge into any existing `additionalModelRequestFields` rather than
    // clobbering caller-supplied keys. Don't override an explicit `thinking`
    // nested there either.
    let amrf = out
        .entry("additionalModelRequestFields".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(map) = amrf {
        if !map.contains_key("thinking") {
            map.insert(
                "thinking".to_string(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
    }
}

/// Map OpenAI `service_tier` (string) onto a top-level Converse `serviceTier`
/// block. Bedrock models accept `{type: "default" | "flex" | "priority"}`
/// (no `"auto"`); OpenAI's `"auto"` maps onto `"default"`. Unknown values are
/// dropped so Bedrock doesn't reject the whole request. Skipped when the
/// caller already supplied a pre-shaped `serviceTier` object (forwarded by
/// `PASS_THROUGH_TOP_LEVEL_KEYS`).
///
/// Reference behavior: `_map_service_tier_param`.
fn apply_service_tier(request: &Value, out: &mut Map<String, Value>) {
    // Don't override an explicit pre-shaped `serviceTier` object.
    if request.get("serviceTier").is_some() {
        return;
    }
    let Some(tier) = request.get("service_tier").and_then(Value::as_str) else {
        return;
    };
    let bedrock_tier = match tier {
        "auto" => "default",
        "default" | "flex" | "priority" => tier,
        // Unknown — skip; Bedrock would reject it anyway.
        _ => return,
    };
    out.insert("serviceTier".to_string(), json!({"type": bedrock_tier}));
}

/// Pass `context_management` through into
/// `additionalModelRequestFields.context_management`. Converse's format is
/// close to the Anthropic/OpenAI list form; the reference runs an OpenAI→Anthropic
/// mapper for list inputs and passes dicts through unchanged — we do the
/// same basic passthrough here (both shapes forwarded verbatim).
///
/// Reference behavior: `_map_context_management_param`.
fn apply_context_management(request: &Value, out: &mut Map<String, Value>) {
    let Some(cm) = request.get("context_management") else {
        return;
    };
    if cm.is_null() {
        return;
    }
    let amrf = out
        .entry("additionalModelRequestFields".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(map) = amrf {
        map.insert("context_management".to_string(), cm.clone());
    }
}

/// Clamp manual thinking into `[1024, maxTokens - 1]` without increasing an
/// explicit output limit. An impossible explicit request is an error; an
/// inferred budget can be omitted. Without a limit, clamp only the minimum and
/// let `auto_set_max_tokens_for_thinking` retain its existing default policy.
fn clamp_thinking_budget(request: &Value, out: &mut Map<String, Value>) -> Result<()> {
    let max_tokens = out
        .get("inferenceConfig")
        .and_then(|config| config.get("maxTokens"))
        .and_then(crate::anthropic::json_number_as_u64)
        .unwrap_or(u64::MAX);
    let Some(Value::Object(amrf)) = out.get_mut("additionalModelRequestFields") else {
        return Ok(());
    };
    let Some(thinking) = amrf.get("thinking") else {
        return Ok(());
    };
    match crate::anthropic::clamp_thinking_budget(thinking, max_tokens) {
        Some(thinking) => {
            amrf.insert("thinking".into(), thinking);
        }
        None => {
            let explicit = [
                request.get("thinking"),
                request.pointer("/additionalModelRequestFields/thinking"),
                request.pointer("/additional_model_request_fields/thinking"),
            ]
            .into_iter()
            .flatten()
            .any(|value| !value.is_null());
            if explicit {
                return Err(crate::anthropic::thinking_budget_error());
            }
            amrf.remove("thinking");
            if amrf.is_empty() {
                out.remove("additionalModelRequestFields");
            }
        }
    }
    Ok(())
}

/// When `additionalModelRequestFields.thinking` is present, Bedrock requires
/// `inferenceConfig.maxTokens` to exceed `budget_tokens`. If the caller did
/// not set `maxTokens`, default it to `budget_tokens + 4096` so the request
/// has room for both the reasoning trace and the visible answer. No-op when
/// thinking is absent or maxTokens is already set.
fn auto_set_max_tokens_for_thinking(out: &mut Map<String, Value>) {
    let Some(budget) = out
        .get("additionalModelRequestFields")
        .and_then(|f| f.get("thinking"))
        .and_then(|t| t.get("budget_tokens"))
        .and_then(Value::as_u64)
    else {
        return;
    };
    let max_tokens_set = out
        .get("inferenceConfig")
        .and_then(|c| c.get("maxTokens"))
        .is_some();
    if max_tokens_set {
        return;
    }
    if let Some(Value::Object(ic)) = out.get_mut("inferenceConfig") {
        // `budget` is caller-controlled (forwarded verbatim by
        // `PASS_THROUGH_TOP_LEVEL_KEYS`), so plain `+` could overflow `u64`
        // on a hostile value like `u64::MAX` — panicking in debug, wrapping
        // in release. Saturate so the transform never panics on user input.
        ic.insert("maxTokens".to_string(), json!(budget.saturating_add(4096)));
    }
}

/// Claude thinking accepts only temperature 1. Drop any other value from
/// `inferenceConfig` after thinking or output effort has been placed on the
/// body. `topP` is left alone.
fn drop_temperature_incompatible_with_thinking(out: &mut Map<String, Value>) {
    let temperature_ok = out
        .get("inferenceConfig")
        .and_then(|config| config.get("temperature"))
        .is_none_or(|temp| temp.as_f64() == Some(1.0));
    if temperature_ok {
        return;
    }
    let thinking_on = out
        .get("additionalModelRequestFields")
        .and_then(|fields| fields.get("thinking"))
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "enabled" || kind == "adaptive");
    let effort_set = out
        .get("outputConfig")
        .and_then(|cfg| cfg.get("effort"))
        .is_some_and(|effort| !effort.is_null());
    if thinking_on || effort_set {
        if let Some(Value::Object(config)) = out.get_mut("inferenceConfig") {
            config.remove("temperature");
        }
    }
}

/// Converse's native `toolConfig.toolChoice` has no parallel-use flag.
/// Anthropic-family models accept the flag through the model-specific
/// `additionalModelRequestFields.tool_choice` object (with the required
/// `type` discriminator), mirroring how the reference serves
/// `enable_parallel_tool_use` on Bedrock Claude. Other model families have no
/// equivalent field, so an explicit `parallel_tool_calls: false` is rejected
/// rather than silently ignored. An already-native model-specific choice is
/// preserved, and duplicate native/canonical selections must agree.
fn apply_parallel_tool_calls(
    model: &str,
    request: &Value,
    out: &mut Map<String, Value>,
    forward: &HashMap<String, String>,
) -> Result<()> {
    let inline_serial = request
        .pointer("/tool_choice/disable_parallel_tool_use")
        .filter(|value| !value.is_null());
    if inline_serial.is_some_and(|value| !value.is_boolean()) {
        return Err(oxllm_core::Error::BadRequest(
            "tool_choice.disable_parallel_tool_use must be a boolean".into(),
        ));
    }
    let serial = request.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false)
        || inline_serial.and_then(Value::as_bool) == Some(true);
    if serial && inline_serial.and_then(Value::as_bool) == Some(false) {
        return Err(oxllm_core::Error::BadRequest(
            "parallel_tool_calls false conflicts with tool_choice.disable_parallel_tool_use".into(),
        ));
    }
    let native = out
        .get("additionalModelRequestFields")
        .and_then(|fields| fields.get("tool_choice"))
        .filter(|value| !value.is_null())
        .cloned();
    let normalize = |value: &Value| {
        let mut choice = value.clone();
        remap_native_choice_name(&mut choice, forward);
        choice
    };
    let camel = request
        .pointer("/additionalModelRequestFields/tool_choice")
        .filter(|value| !value.is_null())
        .map(normalize);
    let snake = request
        .pointer("/additional_model_request_fields/tool_choice")
        .filter(|value| !value.is_null())
        .map(normalize);
    if camel
        .zip(snake)
        .is_some_and(|(camel, snake)| camel != snake)
    {
        return Err(oxllm_core::Error::BadRequest("additionalModelRequestFields.tool_choice conflicts with additional_model_request_fields.tool_choice".into()));
    }
    let canonical_none = request.get("tool_choice").is_some_and(|choice| {
        choice.as_str() == Some("none")
            || choice.get("type").and_then(Value::as_str) == Some("none")
    });
    if canonical_none
        && native
            .as_ref()
            .is_some_and(|choice| choice.get("type").and_then(Value::as_str) != Some("none"))
    {
        return Err(oxllm_core::Error::BadRequest(
            "tool_choice none conflicts with additionalModelRequestFields.tool_choice".into(),
        ));
    }
    if !out.contains_key("toolConfig") || (!serial && native.is_none()) {
        return Ok(());
    }
    if !is_anthropic_model(model) {
        if !serial {
            return Ok(());
        }
        return Err(oxllm_core::Error::BadRequest(
            "parallel_tool_calls false is unsupported by the Bedrock Converse adapter for this \
             model; use a Claude model, or configure model_info.base_model/base_model for a \
             verified Claude-backed inference profile"
                .into(),
        ));
    }
    // Only derive a selection when none was supplied in the model-specific
    // native container. Adding the serial flag must not replace a caller's
    // required/named choice with an automatic default.
    let mapped = match out.get("toolConfig").and_then(|tc| tc.get("toolChoice")) {
        Some(Value::Object(t)) if t.contains_key("tool") => {
            let name = t["tool"].get("name").cloned().unwrap_or(Value::Null);
            Some(json!({"type": "tool", "name": name}))
        }
        Some(Value::Object(t)) if t.contains_key("any") => Some(json!({"type": "any"})),
        Some(Value::Object(t)) if t.contains_key("auto") => Some(json!({"type": "auto"})),
        _ => None,
    };
    if let (Some(native), Some(mapped)) = (&native, &mapped)
        && (native.get("type") != mapped.get("type") || native.get("name") != mapped.get("name"))
    {
        return Err(oxllm_core::Error::BadRequest(
            "tool_choice conflicts with additionalModelRequestFields.tool_choice".into(),
        ));
    }
    let mut choice = native.or(mapped).unwrap_or_else(|| json!({"type":"auto"}));
    // A caller-supplied native forced choice in
    // additionalModelRequestFields bypasses the canonical downgrade in
    // build_tool_config; apply the same forced-tool policy to the merged
    // result so `any`/`tool` cannot reach a model that rejects them.
    if crate::anthropic::model_rejects_forced_tool_use(model)
        && matches!(
            choice.get("type").and_then(Value::as_str),
            Some("any" | "tool")
        )
    {
        let mut downgraded = Map::new();
        downgraded.insert("type".to_string(), json!("auto"));
        if let Some(parallel) = choice.get("disable_parallel_tool_use") {
            downgraded.insert("disable_parallel_tool_use".to_string(), parallel.clone());
        }
        choice = Value::Object(downgraded);
    }
    if serial && choice.get("type").and_then(Value::as_str) != Some("none") {
        if choice
            .get("disable_parallel_tool_use")
            .is_some_and(|value| !value.is_null() && value.as_bool() != Some(true))
        {
            return Err(oxllm_core::Error::BadRequest("parallel_tool_calls false conflicts with native tool_choice.disable_parallel_tool_use".into()));
        }
        choice
            .as_object_mut()
            .ok_or_else(|| {
                oxllm_core::Error::BadRequest(
                    "native tool_choice must be an object to preserve parallel_tool_calls false"
                        .into(),
                )
            })?
            .insert("disable_parallel_tool_use".to_string(), json!(true));
    }
    // Remove the Bedrock-native choice so the request does not carry two
    // `tool_choice` encodings for the same call.
    if let Some(tc) = out.get_mut("toolConfig").and_then(Value::as_object_mut) {
        tc.remove("toolChoice");
    }
    let additional = out
        .entry("additionalModelRequestFields".to_string())
        .or_insert_with(|| json!({}));
    additional
        .as_object_mut()
        .ok_or_else(|| {
            oxllm_core::Error::BadRequest(
                "additionalModelRequestFields must be an object to preserve tool_choice".into(),
            )
        })?
        .insert("tool_choice".to_string(), choice);
    Ok(())
}

/// Converse has one structured-output wire shape across model families:
/// `outputConfig.textFormat.structure.jsonSchema.schema` is a JSON string.
/// Forward caller-supplied schemas without a model-name gate or repairs.
/// Unsupported models/schema features are rejected by Bedrock itself.
fn apply_response_format(request: &Value, out: &mut Map<String, Value>) -> Result<()> {
    let paths = [
        ("/output_config/format", "output_config.format"),
        ("/outputConfig/format", "outputConfig.format"),
        ("/output_format", "output_format"),
    ];
    let bare_json = request.get("response_format").is_some_and(|format| {
        format.get("type").and_then(Value::as_str) == Some("json_object")
            && format.get("json_schema").is_none()
    });
    if bare_json
        && !paths
            .iter()
            .any(|(path, _)| request.pointer(path).is_some_and(|value| !value.is_null()))
        && ["/outputConfig/textFormat", "/output_config/textFormat"]
            .iter()
            .any(|path| {
                request.pointer(path).is_some_and(|format| {
                    format.get("type").and_then(Value::as_str) == Some("json_schema")
                        && format
                            .pointer("/structure/jsonSchema/schema")
                            .and_then(Value::as_str)
                            .is_some_and(|schema| serde_json::from_str::<Value>(schema).is_ok())
                })
            })
    {
        // The caller already supplied a complete native schema. Preserve its
        // string/options in the pass-through merge below, without rebuilding it.
        return Ok(());
    }
    if let Some(format) = crate::util::native_output_format(request, &paths)? {
        if format.get("type").and_then(Value::as_str) != Some("json_schema")
            || !format.get("schema").is_some_and(Value::is_object)
            || format.as_object().is_some_and(|format| {
                format
                    .keys()
                    .any(|key| !matches!(key.as_str(), "type" | "schema"))
            })
        {
            let source = paths
                .iter()
                .find(|(path, _)| request.pointer(path).is_some_and(|value| !value.is_null()))
                .map(|(_, name)| *name)
                .unwrap_or("response_format");
            return Err(oxllm_core::Error::BadRequest(format!(
                "{source} has no lossless native Converse mapping; use json_schema with only type and an object schema, or outputConfig.textFormat for Converse-native options"
            )));
        }
        let mut json_schema = json!({"schema": format["schema"].to_string()});
        if let Some(wrapper) = request
            .pointer("/response_format/json_schema")
            .filter(|wrapper| wrapper.get("schema").is_some())
        {
            for key in ["name", "description"] {
                if let Some(value) = wrapper.get(key).and_then(Value::as_str) {
                    json_schema[key] = json!(value);
                }
            }
        }
        out.insert(
            "outputConfig".to_string(),
            json!({
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": json_schema
                    }
                }
            }),
        );
        return Ok(());
    }
    Ok(())
}

/// Normalize schema serialization only for comparison. Keep the caller's native
/// schema string verbatim and merge optional, explicitly requested labels.
fn merge_text_formats(mapped: &Value, native: &Value) -> Option<Value> {
    if mapped == native {
        return Some(native.clone());
    }
    const SCHEMA: &str = "/structure/jsonSchema/schema";
    const CONFIG: &str = "/structure/jsonSchema";
    let normalize = |format: &Value| {
        if format.get("type").and_then(Value::as_str) != Some("json_schema") {
            return None;
        }
        let schema: Value = serde_json::from_str(format.pointer(SCHEMA)?.as_str()?).ok()?;
        let mut normalized = format.clone();
        *normalized.pointer_mut(SCHEMA)? = schema;
        Some(normalized)
    };
    let mut left = normalize(mapped)?;
    let mut right = normalize(native)?;
    for key in ["name", "description"] {
        let l = left.pointer(CONFIG)?.get(key).cloned();
        let r = right.pointer(CONFIG)?.get(key).cloned();
        match (l, r) {
            (Some(value), None) => {
                right
                    .pointer_mut(CONFIG)?
                    .as_object_mut()?
                    .insert(key.into(), value);
            }
            (None, Some(value)) => {
                left.pointer_mut(CONFIG)?
                    .as_object_mut()?
                    .insert(key.into(), value);
            }
            _ => {}
        }
    }
    if left != right {
        return None;
    }
    let mut merged = native.clone();
    for key in ["name", "description"] {
        if let Some(value) = mapped.pointer(CONFIG)?.get(key) {
            merged
                .pointer_mut(CONFIG)?
                .as_object_mut()?
                .entry(key)
                .or_insert_with(|| value.clone());
        }
    }
    Some(merged)
}

/// Does this JSON value carry a non-null OpenAI-style `cache_control` hint?
/// Used to decide whether to inject a Converse `cachePoint` breakpoint after
/// the carrying message or content block. Reference behavior:
/// `_get_cache_point_block` / `_build_cache_point_block`.
fn has_cache_control(obj: &Value) -> bool {
    obj.get("cache_control").is_some_and(|v| !v.is_null())
}

/// Does any element of a tool message's `content` array carry a
/// `cache_control` hint? The reference checks both message-level and content-level
/// `cache_control` on tool-result messages.
fn tool_result_content_has_cache_control(msg: &Value) -> bool {
    msg.get("content")
        .and_then(Value::as_array)
        .is_some_and(|arr| arr.iter().any(has_cache_control))
}

/// Build a Converse `cachePoint` content block from an OpenAI-style
/// `cache_control` hint. The `type` is always `"default"` (the only value
/// Bedrock's Converse API accepts). A `ttl` of `"5m"`/`"1h"` is forwarded for
/// models whose card publishes a 1h cache-write rate (reference
/// `_build_cache_point_block`); other models keep the plain form.
fn build_cache_point_block(carrier: &Value, model: &str) -> Value {
    let mut block = json!({"cachePoint": {"type": "default"}});
    // Bedrock accepts an explicit `ttl` ("5m"/"1h") on `cachePoint` for
    // models whose card publishes a 1h cache-write rate (reference
    // `_build_cache_point_block`). Anything else keeps the plain form.
    if let Some(ttl) = cache_control_ttl(carrier)
        && matches!(ttl, "5m" | "1h")
        && bedrock_cache_point_ttl_supported(model)
    {
        block["cachePoint"]["ttl"] = json!(ttl);
    }
    block
}

/// The `cache_control.ttl` marked on the carrier itself or on one of its
/// content parts — the same two levels `has_cache_control` scans.
fn cache_control_ttl(carrier: &Value) -> Option<&str> {
    let direct = carrier
        .get("cache_control")
        .and_then(|control| control.get("ttl"))
        .and_then(Value::as_str);
    direct.or_else(|| {
        carrier
            .get("content")
            .and_then(Value::as_array)
            .and_then(|parts| {
                parts.iter().find_map(|part| {
                    part.get("cache_control")
                        .and_then(|control| control.get("ttl"))
                        .and_then(Value::as_str)
                })
            })
    })
}

/// Whether the Bedrock model accepts a `cachePoint` `ttl`, decided by a
/// published 1h cache-write rate on its card (presence-based, matching the
/// reference `supports_bedrock_prompt_cache_ttl`).
fn bedrock_cache_point_ttl_supported(model: &str) -> bool {
    oxllm_core::lookup_pricing_with_provider(model, "bedrock")
        .and_then(|pricing| pricing.cache_creation_input_token_cost_above_1hr)
        .is_some()
}

/// Push every text fragment found in a system message's `content` into
/// `out` as Converse `{"text": "…"}` system blocks. Empty strings are
/// dropped (Bedrock rejects empty text blocks). When a message or content
/// block carries an OpenAI-style `cache_control` hint, a Converse
/// `{cachePoint: {type: "default"}}` block is injected immediately after
/// the text block — this is the standard Anthropic prompt-caching breakpoint
/// pattern on Converse.
fn accumulate_system_blocks(msg: &Value, out: &mut Vec<Value>, model: &str) {
    let content = &msg["content"];
    match content {
        Value::String(s) => {
            if !s.is_empty() {
                out.push(json!({"text": s}));
                if has_cache_control(msg) {
                    out.push(build_cache_point_block(msg, model));
                }
            }
        }
        Value::Array(arr) => {
            for elem in arr {
                let Some(obj) = elem.as_object() else {
                    continue;
                };
                // Only `text` system blocks are derived here. `cachePoint`
                // blocks are injected from `cache_control` hints below;
                // `guardContent` is a separate concern.
                let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("text");
                if kind == "text" {
                    if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                        if !t.is_empty() {
                            out.push(json!({"text": t}));
                            if has_cache_control(elem) {
                                out.push(build_cache_point_block(elem, model));
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Convert an OpenAI user-message (string or array of typed blocks) into
/// Converse content blocks. Recognises:
///
/// * plain string → `{text: "…"}`
/// * `{type:"text", text:"…"}` → `{text: "…"}`
/// * `{type:"guarded_text", text:"…"}` → `{guardContent: {text: {text: "…"}}}`
///   (selective guardrail scanning; auto-injected for trailing user messages
///   when `guardrailConfig` is set — see `tag_trailing_user_messages_as_guarded`)
/// * `{type:"image_url", image_url:{url:"data:…"}}` → `{image:{format,source:{bytes}}}`
///
/// Non-`data:` URLs (HTTP(S)) are rejected here — Bedrock cannot fetch remote
/// URLs. The server layer is responsible for inlining HTTP image URLs into
/// `data:` URLs before this transform runs (see
/// `oxllm_server::pre_call_checks::image_inlining`); a raw HTTP URL reaching
/// this point indicates the request did not route through a Bedrock-targeted
/// call path. Empty text is dropped to avoid emitting blank content blocks
/// (which Bedrock rejects). A `cache_control` hint on the message or a
/// content block injects a trailing `{cachePoint: {type: "default"}}` block.
fn convert_user_content(msg: &Value, model: &str) -> Result<Vec<Value>> {
    let content = &msg["content"];
    let mut blocks = Vec::new();
    match content {
        Value::String(s) => {
            if !s.is_empty() {
                blocks.push(json!({"text": s}));
                if has_cache_control(msg) {
                    blocks.push(build_cache_point_block(msg, model));
                }
            }
        }
        Value::Array(arr) => {
            for elem in arr {
                let Some(obj) = elem.as_object() else {
                    continue;
                };
                let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("text");
                match kind {
                    "text" => {
                        if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                            if !t.is_empty() {
                                blocks.push(json!({"text": t}));
                                if has_cache_control(elem) {
                                    blocks.push(build_cache_point_block(elem, model));
                                }
                            }
                        }
                    }
                    "guarded_text" => {
                        // `guarded_text` → Converse `guardContent` block for
                        // selective guardrail scanning. The Bedrock schema
                        // nests a `GuardrailConverseTextBlock` (`{text:{text}}`)
                        // — matches the reference prompt-template factory.
                        if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                            if !t.is_empty() {
                                blocks.push(json!({"guardContent": {"text": {"text": t}}}));
                            }
                        }
                    }
                    "image_url" => {
                        blocks.push(data_url_to_converse_media(obj)?);
                        if has_cache_control(elem) {
                            blocks.push(build_cache_point_block(elem, model));
                        }
                    }
                    "file" => {
                        blocks.push(openai_file_to_converse_document(obj)?);
                        if has_cache_control(elem) {
                            blocks.push(build_cache_point_block(elem, model));
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    ensure_document_messages_have_text(&mut blocks);
    Ok(blocks)
}

/// Rewrite the trailing run of consecutive `role:"user"` messages so their
/// text content is tagged for selective guardrail scanning. For each user
/// message in that trailing run:
///
/// * array content: every `{type:"text"}` block has its `type` flipped to
///   `"guarded_text"` (other blocks like images are left untouched);
/// * string content: converted to `[{type:"guarded_text", text: <string>}]`;
/// * a message that already contains a `guarded_text` block is skipped
///   entirely (caller override).
///
/// Non-trailing user messages (anything before a non-user message from the
/// end) are left alone. No-op when there is no trailing user run.
///
/// The downstream [`convert_user_content`] then maps `guarded_text` →
/// `guardContent`. Mirrors the reference
/// `_convert_consecutive_user_messages_to_guarded_text`
/// (`converse_transformation.py:141`).
fn tag_trailing_user_messages_as_guarded(messages: &mut [Value]) {
    // Walk backwards to find the start index of the trailing user run.
    let mut start = messages.len();
    while start > 0 {
        let prev = start - 1;
        let is_user = messages[prev]
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|r| r == "user");
        if is_user {
            start = prev;
        } else {
            break;
        }
    }

    for msg in messages[start..].iter_mut() {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        match obj.get_mut("content") {
            Some(Value::Array(arr)) => {
                // Caller override: if any block is already guarded_text,
                // leave the whole message untouched.
                let already_guarded = arr
                    .iter()
                    .any(|b| b.get("type").and_then(Value::as_str) == Some("guarded_text"));
                if already_guarded {
                    continue;
                }
                for block in arr.iter_mut() {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(bo) = block.as_object_mut() {
                            bo.insert(
                                "type".to_string(),
                                Value::String("guarded_text".to_string()),
                            );
                        }
                    }
                }
            }
            Some(Value::String(s)) => {
                let text = s.clone();
                obj.insert(
                    "content".to_string(),
                    json!([{"type": "guarded_text", "text": text}]),
                );
            }
            _ => {}
        }
    }
}

/// Convert an OpenAI assistant message (`content` + `tool_calls`) into
/// Converse assistant content blocks. Text fragments become `{text: …}`
/// blocks (in order); tool calls become `{toolUse: {toolUseId, name, input}}`
/// blocks appended afterwards.
///
/// `tool_calls[].function.arguments` is a JSON-encoded string in OpenAI;
/// Converse wants `input` as a structured JSON object. We parse the string
/// and, on failure, fall back to `{}` so the block remains well-formed
/// (Bedrock rejects non-object `input`). A `cache_control` hint on a text
/// content block injects a trailing `{cachePoint: {type: "default"}}` block.
fn convert_assistant_content(
    msg: &Value,
    require_signature: bool,
    model: &str,
) -> Result<Vec<Value>> {
    let content = &msg["content"];
    let tool_calls = &msg["tool_calls"];
    let mut blocks: Vec<Value> = crate::util::thinking_blocks_for_replay(msg, require_signature)?
        .iter()
        .filter_map(thinking_block_to_converse)
        .filter(|block| {
            !content
                .as_array()
                .is_some_and(|content| content.contains(block))
        })
        .collect();
    match content {
        Value::String(s) => {
            if !s.is_empty() {
                blocks.push(json!({"text": s}));
                if has_cache_control(msg) {
                    blocks.push(build_cache_point_block(msg, model));
                }
            }
        }
        Value::Array(arr) => {
            for elem in arr {
                if let Some(reasoning) = thinking_block_to_converse(elem) {
                    blocks.push(reasoning);
                    continue;
                }
                if elem.get("reasoningContent").is_some() || elem.get("toolUse").is_some() {
                    blocks.push(elem.clone());
                    continue;
                }
                let Some(obj) = elem.as_object() else {
                    continue;
                };
                let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("text");
                if kind == "text" {
                    if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                        if !t.is_empty() {
                            blocks.push(json!({"text": t}));
                            if has_cache_control(elem) {
                                blocks.push(build_cache_point_block(elem, model));
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    if let Some(tcs) = tool_calls.as_array() {
        for tc in tcs {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("call_bedrock");
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args_str = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let input: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
            blocks.push(json!({
                "toolUse": {
                    "toolUseId": id,
                    "name": name,
                    "input": input,
                }
            }));
        }
    }
    Ok(blocks)
}

fn thinking_block_to_converse(block: &Value) -> Option<Value> {
    match block.get("type").and_then(Value::as_str)? {
        "thinking" => {
            let mut block = block.clone();
            crate::util::normalize_thinking_block(&mut block);
            if crate::util::is_blank_unsigned_thinking_block(&block) {
                return None;
            }
            // Preserve malformed types for provider validation rather than
            // silently repairing them to signed empty text.
            let mut text_block = json!({"text": block.get("thinking").unwrap_or(&Value::Null)});
            if let Some(signature) = block.get("signature") {
                text_block["signature"] = signature.clone();
            }
            Some(json!({"reasoningContent": {"reasoningText": text_block}}))
        }
        "redacted_thinking" => Some(json!({
            "reasoningContent": {"redactedContent": block.get("data")?}
        })),
        _ => None,
    }
}

/// Convert an OpenAI tool-role message (`{role:"tool", tool_call_id, content}`)
/// into Converse user content. GPT models require tool-result images beside
/// the result; other families keep images nested in their original order.
fn convert_tool_result(msg: &Value, model: &str) -> Result<Vec<Value>> {
    let tool_call_id = msg
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mut content_blocks = Vec::new();
    let mut sibling_images = Vec::new();
    let images_beside_result = bedrock_gpt_version(model).is_some();
    match &msg["content"] {
        Value::String(text) if !text.is_empty() => content_blocks.push(json!({"text": text})),
        Value::Array(parts) => {
            for part in parts {
                let Some(object) = part.as_object() else {
                    continue;
                };
                match object.get("type").and_then(Value::as_str).unwrap_or("text") {
                    "text" => {
                        if let Some(text) = object.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            content_blocks.push(json!({"text": text}));
                        }
                    }
                    "tool_reference" => {
                        if let Some(name) = object.get("tool_name").and_then(Value::as_str)
                            && !name.is_empty()
                        {
                            content_blocks.push(json!({"text": name}));
                        }
                    }
                    "image_url" => {
                        let media = data_url_to_converse_media(object)?;
                        if images_beside_result && media.get("image").is_some() {
                            sibling_images.push(media);
                        } else {
                            content_blocks.push(media);
                        }
                    }
                    "file" => content_blocks.push(openai_file_to_converse_document(object)?),
                    _ => {}
                }
            }
        }
        _ => {}
    }
    if content_blocks.is_empty() {
        let text = if sibling_images.is_empty() {
            "Tool returned no content."
        } else {
            "Tool returned an image. See the image blocks in this message."
        };
        content_blocks.push(json!({"text": text}));
    }
    let mut result = json!({
        "toolResult": {
            "toolUseId": tool_call_id,
            "content": content_blocks,
        }
    });
    if let Some(status) = msg.get("status").filter(|value| !value.is_null()) {
        result["toolResult"]["status"] = status.clone();
    }
    let mut blocks = vec![result];
    blocks.append(&mut sibling_images);
    Ok(blocks)
}

/// Recognize numeric GPT versions without mistaking GPT-OSS or an opaque
/// profile name for a GPT model. Family detection handles regional IDs and ARNs.
fn bedrock_gpt_version(model: &str) -> Option<(u32, u32)> {
    let lower = model.to_ascii_lowercase();
    let model = super::model_id_for_family(&lower)?;
    let version = model
        .strip_prefix("openai.gpt-")?
        .split(['-', ':', '['])
        .next()?;
    let parse_component = |part: &str| {
        part.bytes()
            .all(|byte| byte.is_ascii_digit())
            .then(|| part.parse::<u32>().ok())
            .flatten()
    };
    let mut components = version.split('.');
    let major = parse_component(components.next()?)?;
    let minor = match components.next() {
        Some(part) => parse_component(part)?,
        None => 0,
    };
    components.next().is_none().then_some((major, minor))
}

fn bedrock_rejects_stop_sequences(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let grok = super::model_id_for_family(&lower).is_some_and(|model| {
        model
            .strip_prefix("xai.grok-")
            .is_some_and(|suffix| !suffix.is_empty())
    });
    grok || bedrock_gpt_version(model).is_some_and(|version| version >= (5, 6))
}

fn bedrock_requires_min_max_tokens(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("xai.grok-")
        || lower
            .match_indices("openai.gpt-")
            .any(|(i, needle)| lower[i + needle.len()..].starts_with(|c: char| c.is_ascii_digit()))
}

/// GPT-5 and newer on Converse reject `temperature` and `top_p`.
fn bedrock_gpt_rejects_sampling(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    [
        "openai.gpt-5",
        "openai.gpt-6",
        "openai.gpt-7",
        "openai.gpt-8",
        "openai.gpt-9",
    ]
    .iter()
    .any(|prefix| lower.contains(prefix))
}

/// Build the `inferenceConfig` object from sampling params.
///
/// Always returns an object (possibly empty) so the request shape is stable.
fn build_inference_config(request: &Value, model: &str) -> Value {
    let mut ic = Map::new();
    // Honour the modern OpenAI alias (`max_completion_tokens`) when the
    // classic `max_tokens` is absent — the InvokeModel vendor path and the
    // Anthropic transform both do, so a Converse caller that only sets the
    // alias must not silently lose its output limit.
    let max_tokens = request
        .get("max_tokens")
        .and_then(crate::anthropic::json_number_as_u64)
        .or_else(|| {
            request
                .get("max_completion_tokens")
                .and_then(crate::anthropic::json_number_as_u64)
        });
    if let Some(mt) = max_tokens {
        // Converse rejects maxTokens below 16 for OpenAI GPT and xAI Grok.
        let mt = if bedrock_requires_min_max_tokens(model) {
            mt.max(16)
        } else {
            mt
        };
        ic.insert("maxTokens".to_string(), json!(mt));
    }
    if !bedrock_gpt_rejects_sampling(model) {
        // Claude models released after Opus 4.6 removed sampling control:
        // forward only the values the API still accepts (temperature 1,
        // top_p >= 0.99) and omit the rest, matching the silent-omit
        // pattern above. The Anthropic InvokeModel/native path enforces
        // the same policy with drop_params-aware errors.
        let claude_rejects_sampling = crate::anthropic::model_rejects_sampling_params(model);
        if let Some(t) = request.get("temperature").and_then(|v| v.as_f64()) {
            if !claude_rejects_sampling || t == 1.0 {
                ic.insert("temperature".to_string(), json!(t));
            }
        }
        if let Some(p) = request.get("top_p").and_then(|v| v.as_f64()) {
            if !claude_rejects_sampling || p >= 0.99 {
                ic.insert("topP".to_string(), json!(p));
            }
        }
    }
    // Grok and GPT 5.6+ reject stop sequences on Converse.
    if !bedrock_rejects_stop_sequences(model) {
        match request.get("stop") {
            Some(Value::Array(arr)) => {
                let strs: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                if !strs.is_empty() {
                    ic.insert("stopSequences".to_string(), json!(strs));
                }
            }
            Some(Value::String(s)) if !s.is_empty() => {
                ic.insert("stopSequences".to_string(), json!([s]));
            }
            _ => {}
        }
    }
    Value::Object(ic)
}

/// Build the Converse `toolConfig` object from OpenAI `tools[]` and
/// `tool_choice`. Returns `None` when there are no usable tools, or when
/// the caller explicitly disables tools via `tool_choice: "none"`.
///
/// Tools already in Bedrock shape (`systemTool`, `toolSpec`, `cachePoint`)
/// are forwarded verbatim — this is how Nova grounding / cache points ride
/// through. OpenAI function tools become `{toolSpec: {name, description,
/// inputSchema:{json}}}`. Anthropic-style tools (`{name, input_schema}`)
/// are also recognised.
///
fn build_tool_config(request: &Value, model: &str) -> Result<Option<Value>> {
    // Honour `tool_choice: "none"` by omitting toolConfig entirely. OpenAI
    // accepts both the object form (`{"type":"none"}`) and the bare string
    // (`"none"`); recognise both so a caller asking for "no tools" actually
    // gets them suppressed upstream.
    if let Some(choice) = request.get("tool_choice") {
        let is_none = choice.as_str() == Some("none")
            || choice.get("type").and_then(|v| v.as_str()) == Some("none");
        if is_none {
            if choice
                .as_object()
                .is_some_and(|choice| choice.keys().any(|key| key != "type"))
            {
                return Err(oxllm_core::Error::BadRequest("tool choice none has no lossless native Converse mapping for additional options".into()));
            }
            if let Some(tools) = request.get("tools").and_then(Value::as_array) {
                for tool in tools {
                    crate::util::strict_tool_definition(tool)?;
                }
            }
            return Ok(None);
        }
    }

    let mut tool_specs: Vec<Value> = Vec::new();
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        // Counter for unnamed tools so each gets a unique placeholder name
        // (collisions would make tool-result routing ambiguous).
        let mut unnamed_counter: usize = 0;
        for tool in tools {
            // Pre-formatted Bedrock tool blocks (e.g. Nova grounding systemTool)
            // pass through unchanged.
            if tool.get("systemTool").is_some()
                || tool.get("toolSpec").is_some()
                || tool.get("cachePoint").is_some()
            {
                validate_native_tool(tool)?;
                let mut tool = tool.clone();
                if let Some(spec) = tool.get_mut("toolSpec").and_then(Value::as_object_mut)
                    && spec
                        .get("strict")
                        .is_some_and(|value| value.is_null() || value.as_bool() == Some(false))
                {
                    spec.remove("strict");
                }
                tool_specs.push(tool);
                continue;
            }
            let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if !(matches!(tool_type, "" | "function")
                || tool_type == "custom" && tool.get("input_schema").is_some())
            {
                return Err(unsupported_hosted_tool(tool_type));
            }
            // OpenAI `{type:"function", function:{name, description, parameters}}`
            // or Anthropic `{name, description, input_schema}`.
            let function = if tool.get("function").is_some() {
                &tool["function"]
            } else if tool.get("input_schema").is_some() {
                tool
            } else {
                continue;
            };
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from)
                .unwrap_or_else(|| {
                    unnamed_counter = unnamed_counter.saturating_add(1);
                    format!("litellm_unnamed_tool_{unnamed_counter}")
                });
            let description = function
                .get("description")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(&name);
            // Advisory tools retain legacy compatibility normalization;
            // explicitly strict schemas bypass it without constraint changes.
            let strict = crate::util::strict_tool_definition(tool)?;
            let schema = function
                .get("parameters")
                .or_else(|| function.get("input_schema"))
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            let schema = if strict {
                schema
            } else {
                sanitize_tool_input_schema(schema)
            };
            if !is_tool_input_object_schema(&schema) {
                return Err(oxllm_core::Error::BadRequest(
                    "Bedrock Converse function tools require an object input schema".into(),
                ));
            }
            let mut tool_spec = json!({
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": schema},
                }
            });
            if strict {
                tool_spec["toolSpec"]["strict"] = json!(true);
            }
            tool_specs.push(tool_spec);
        }
    }
    // Nova web-search grounding: when `web_search_options` is present and the
    // model is a Nova model, inject a `systemTool` for `nova_grounding`.
    // Nova doesn't support the OpenAI web_search_options sub-fields, so we
    // just enable grounding with no options (mirrors the reference
    // `_map_web_search_options`). Non-Nova models are skipped.
    if is_nova_model(model) && request.get("web_search_options").is_some() {
        tool_specs.push(json!({"systemTool": {"name": "nova_grounding"}}));
    }
    if tool_specs.is_empty() {
        return Ok(None);
    }

    let mut tc = Map::new();
    tc.insert("tools".to_string(), Value::Array(tool_specs));
    if let Some(choice) = request.get("tool_choice") {
        if let Some(mut mapped) = map_tool_choice(choice) {
            // Forced tool use (`any`/`tool`) 400s on Fable/Mythos 5.1 and
            // the Opus/Sonnet 5.5 point releases; downgrade to auto, the
            // silent-omit pattern this transform uses for other
            // model-rejected fields.
            if crate::anthropic::model_rejects_forced_tool_use(model)
                && (mapped.get("any").is_some() || mapped.get("tool").is_some())
            {
                mapped = json!({"auto": {}});
            }
            tc.insert("toolChoice".to_string(), mapped);
        } else if !choice.is_null() {
            return Err(oxllm_core::Error::BadRequest("tool choice has no lossless native Converse mapping; use auto, required or a named function".into()));
        }
    }
    Ok(Some(Value::Object(tc)))
}

/// Map an OpenAI `tool_choice` onto a Converse `toolChoice`.
///
/// | OpenAI                                          | Converse               |
/// |-------------------------------------------------|------------------------|
/// | `"auto"` / `{type:"auto"}`                      | `{auto: {}}`           |
/// | `"required"` / `{type:"required"}`              | `{any: {}}`            |
/// | `{type:"function", function:{name:"x"}}`        | `{tool: {name:"x"}}`   |
/// | `"none"` / `{type:"none"}`                      | (handled by caller)    |
///
/// Both the bare-string and object forms are recognised (OpenAI's
/// `tool_choice` accepts either), matching the reference normalisation.
fn map_tool_choice(choice: &Value) -> Option<Value> {
    // Bare-string form: `"auto"` / `"required"` / `"none"`.
    if let Some(s) = choice.as_str() {
        return Some(match s {
            "auto" => json!({"auto": {}}),
            "required" | "any" => json!({"any": {}}),
            // `"none"` is handled by the caller (drops toolConfig); map to
            // auto defensively rather than dropping silently here.
            "none" => json!({"auto": {}}),
            _ => return None,
        });
    }
    let t = choice.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if choice.as_object().is_some_and(|object| {
        object.keys().any(|key| {
            !(matches!(key.as_str(), "type" | "disable_parallel_tool_use")
                || t == "function" && key == "function")
        })
    }) || choice
        .get("function")
        .and_then(Value::as_object)
        .is_some_and(|function| function.keys().any(|key| key != "name"))
    {
        return None;
    }
    Some(match t {
        "auto" => json!({"auto": {}}),
        "required" | "any" => json!({"any": {}}),
        "function" => {
            let name = choice
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            json!({"tool": {"name": name}})
        }
        _ => return None,
    })
}

fn unsupported_hosted_tool(tool_type: &str) -> oxllm_core::Error {
    oxllm_core::Error::BadRequest(format!(
        "tool type `{tool_type}` has no supported Bedrock Converse representation; use InvokeModel on a supported Claude model with use_converse_api/use_converse disabled"
    ))
}

fn is_tool_input_object_schema(schema: &Value) -> bool {
    schema.is_object()
        && schema
            .get("type")
            .is_none_or(|kind| kind.as_str() == Some("object"))
}

/// The AWS Tool union contains toolSpec, systemTool, or cachePoint, never an
/// Anthropic hosted type disguised as a client-defined function.
fn validate_native_tool(tool: &Value) -> Result<()> {
    let invalid =
        || oxllm_core::Error::BadRequest("invalid Bedrock Converse native tool shape".into());
    let object = tool.as_object().ok_or_else(invalid)?;
    if object.len() != 1 {
        return Err(invalid());
    }
    let valid_name = |name: &Value| {
        name.as_str().is_some_and(|name| {
            !name.is_empty() && name.len() <= 64 && name.chars().all(is_valid_tool_name_char)
        })
    };
    if let Some(spec) = object.get("toolSpec") {
        if let Some(kind) = spec.get("type").and_then(Value::as_str) {
            return Err(if matches!(kind, "function" | "custom") {
                invalid()
            } else {
                unsupported_hosted_tool(kind)
            });
        }
        crate::util::strict_function_tool(spec)?;
        let spec = spec.as_object().ok_or_else(invalid)?;
        let schema = spec
            .get("inputSchema")
            .and_then(Value::as_object)
            .ok_or_else(invalid)?;
        if !spec.get("name").is_some_and(valid_name)
            || schema.len() != 1
            || !schema.get("json").is_some_and(is_tool_input_object_schema)
            || spec.keys().any(|key| {
                !matches!(
                    key.as_str(),
                    "name" | "description" | "inputSchema" | "strict"
                )
            })
            || spec
                .get("description")
                .is_some_and(|value| value.as_str().is_none_or(str::is_empty))
        {
            return Err(invalid());
        }
    } else if let Some(system) = object.get("systemTool") {
        let system = system.as_object().ok_or_else(invalid)?;
        if system.len() != 1 || !system.get("name").is_some_and(valid_name) {
            return Err(invalid());
        }
    } else if let Some(cache) = object.get("cachePoint") {
        if !cache.is_object() || cache.get("type").and_then(Value::as_str) != Some("default") {
            return Err(invalid());
        }
    } else {
        return Err(invalid());
    }
    Ok(())
}

/// Convert an OpenAI `image_url` content block with a `data:` URL into a
/// Converse image block:
///
/// ```json
/// {"image": {"format": "png", "source": {"bytes": "<base64>"}}}
/// ```
///
/// Rejects remote or malformed URLs instead of losing their content.
/// Convert an OpenAI `image_url` data-URL into a Converse `image` or
/// `document` block. PDFs (and other Bedrock document types) must go out
/// as `{document: …}` — wrapping them as `image` 400s on Converse.
fn data_url_to_converse_media(block: &Map<String, Value>) -> Result<Value> {
    let url = block
        .get("image_url")
        .and_then(|image| {
            image
                .as_str()
                .or_else(|| image.get("url").and_then(Value::as_str))
        })
        .unwrap_or("");
    let (media_type, data) = crate::util::parse_base64_data_url(url).ok_or_else(|| {
        oxllm_core::Error::BadRequest(
            "Bedrock Converse image_url requires a non-empty base64 data URL; remote URLs must be inlined before translation".into(),
        )
    })?;
    if let Some(format) = media_type.strip_prefix("image/") {
        return Ok(json!({
            "image": {
                "format": format,
                "source": {"bytes": data},
            }
        }));
    }
    let format = document_format_from_media_type(media_type).ok_or_else(|| {
        oxllm_core::Error::BadRequest("unsupported Bedrock Converse media type".into())
    })?;
    Ok(json!({
        "document": {
            "format": format,
            "name": format!("document.{format}"),
            "source": {"bytes": data},
        }
    }))
}

fn document_format_from_media_type(media_type: &str) -> Option<&'static str> {
    match media_type {
        "application/pdf" => Some("pdf"),
        "text/csv" => Some("csv"),
        "application/msword" => Some("doc"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Some("docx"),
        "application/vnd.ms-excel" => Some("xls"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Some("xlsx"),
        "text/html" => Some("html"),
        "text/plain" => Some("txt"),
        "text/markdown" => Some("md"),
        _ => None,
    }
}

fn openai_file_to_converse_document(block: &Map<String, Value>) -> Result<Value> {
    use base64::Engine as _;
    let invalid_file = || {
        oxllm_core::Error::BadRequest(
        "Bedrock Converse file requires non-empty base64 file_data; file IDs and non-base64 data URLs are unsupported".into(),
    )
    };
    let file = block
        .get("file")
        .and_then(Value::as_object)
        .ok_or_else(invalid_file)?;
    let data = file
        .get("file_data")
        .and_then(Value::as_str)
        .or_else(|| file.get("data").and_then(Value::as_str))
        .ok_or_else(invalid_file)?;
    let (media_type, bytes) = if data.starts_with("data:") {
        crate::util::parse_base64_data_url(data).ok_or_else(invalid_file)?
    } else {
        ("application/pdf", data)
    };
    if bytes.is_empty()
        || base64::engine::general_purpose::STANDARD
            .decode(bytes)
            .is_err()
    {
        return Err(invalid_file());
    }
    let format = document_format_from_media_type(media_type).ok_or_else(|| {
        oxllm_core::Error::BadRequest("unsupported Bedrock Converse document media type".into())
    })?;
    let name = file
        .get("filename")
        .or_else(|| file.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("document");
    Ok(json!({
        "document": {
            "format": format,
            "name": name,
            "source": {"bytes": bytes},
        }
    }))
}

const BEDROCK_DOCUMENT_PLACEHOLDER_TEXT: &str = ".";

/// Bedrock Converse rejects a user message that carries a document block
/// without a sibling text block ("A text block must be included when using
/// documents"). Claude Code sends PDF-only user turns; inject a placeholder
/// ahead of a trailing `cachePoint` so the cache boundary stays last.
fn ensure_document_messages_have_text(blocks: &mut Vec<Value>) {
    let has_document = blocks.iter().any(|b| b.get("document").is_some());
    let has_text = blocks.iter().any(|b| b.get("text").is_some());
    if !has_document || has_text {
        return;
    }
    let placeholder = json!({"text": BEDROCK_DOCUMENT_PLACEHOLDER_TEXT});
    if blocks.last().is_some_and(|b| b.get("cachePoint").is_some()) {
        let cache = blocks.pop();
        blocks.push(placeholder);
        if let Some(cache) = cache {
            blocks.push(cache);
        }
    } else {
        blocks.push(placeholder);
    }
}

// =============================================================================
// Tool name sanitization
// =============================================================================
//
// Converse uses Anthropic's tool-name charset with a shorter 64-character
// limit (`^[a-zA-Z0-9_-]{1,64}$`). OpenAI callers can pass arbitrary names
// (e.g. `my.tool.v2`), so we sanitize them before they go upstream and keep
// a reverse map to restore caller names on the response / stream way back.
// This mirrors the Anthropic provider's `build_tool_name_maps` logic.

// Per-thread reverse tool-name map (sanitized → original). Seeded by
// `BedrockProvider::transform_response_with_request` right before the
// synchronous read inside `transform_response_converse`, so the write and
// read always run on the same thread with no `await` between them (the
// non-streaming response transform is fully synchronous). The streaming
// path does NOT use this — it carries its own per-stream reverse map via
// `BedrockStreamState` because a tokio task can migrate across worker
// threads between chunks, orphaning a `thread_local!`.
thread_local! {
    static TOOL_NAME_MAP: RefCell<HashMap<String, String>> =
        RefCell::new(HashMap::new());
}

/// Build the reverse (sanitized → original) tool-name map from an OpenAI-shape
/// request. Used by the stateful response/stream transforms to recover caller
/// tool names without depending on a process-wide cell (which two concurrent
/// requests with different tool sets would clobber). Replays the same
/// [`build_tool_name_maps`] pipeline [`transform_request_converse`] uses, so
/// the sanitized forms agree with whatever was sent upstream.
pub(crate) fn reverse_map_from_request(request: &Value) -> HashMap<String, String> {
    let tools: Vec<Value> = request
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| a.to_vec())
        .unwrap_or_default();
    build_tool_name_maps(&tools).1
}

/// Seed the per-thread reverse tool-name map. Called by the stateful response
/// transform on the calling thread immediately before the synchronous read.
pub(crate) fn seed_tool_name_map(reverse: HashMap<String, String>) {
    TOOL_NAME_MAP.with(|cell| *cell.borrow_mut() = reverse);
}

/// Whether a character is allowed in a Converse tool name
/// (`^[a-zA-Z0-9_-]{1,64}$`).
fn is_valid_tool_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Truncate a string to at most `max` chars (by Unicode scalar value).
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Build `(forward, reverse)` tool-name maps for a list of OpenAI-format
/// tools. Ported from the Anthropic provider (`build_tool_name_maps`):
///
/// - `forward`: original caller name → Converse-safe name to send upstream.
/// - `reverse`: Converse-safe name → original caller name.
///
/// Replacement rule: invalid chars → `_`. Names longer than 64 chars are
/// truncated. Collisions are broken by appending `_2`, `_3`, … Names that
/// are already valid and unique are recorded as identity mappings so the
/// reverse lookup table is exhaustive over everything sent upstream.
fn build_tool_name_maps(tools: &[Value]) -> (HashMap<String, String>, HashMap<String, String>) {
    let mut forward: HashMap<String, String> = HashMap::new();
    let mut reverse: HashMap<String, String> = HashMap::new();

    for tool in tools {
        // OpenAI-format caller tools carry the name under `function.name`.
        // Pre-formatted Bedrock tools (`systemTool` / `toolSpec`) are already
        // Converse-safe and have no `function.name` / top-level `name`, so
        // they are skipped here.
        let name = tool
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .or_else(|| tool.get("name").and_then(|n| n.as_str()))
            .unwrap_or("");
        if name.is_empty() || forward.contains_key(name) {
            continue;
        }

        let needs_sanitization =
            name.len() > 64 || name.chars().any(|c| !is_valid_tool_name_char(c));

        // An already-valid name is only an identity mapping when nothing has
        // claimed that name yet: with ["my.tool", "my_tool"] the identity
        // insert would clobber the reverse entry for the first tool's
        // sanitized form and both tools would go upstream under the same
        // name. Treat the clash as a collision and fall through to the
        // suffix-disambiguation loop below.
        if !needs_sanitization && !reverse.contains_key(name) {
            forward.insert(name.to_string(), name.to_string());
            reverse.insert(name.to_string(), name.to_string());
            continue;
        }

        let base: String = name
            .chars()
            .map(|c| if is_valid_tool_name_char(c) { c } else { '_' })
            .collect();

        let mut candidate = truncate_chars(&base, 64);
        let mut counter = 2u32;
        while reverse.contains_key(&candidate) && reverse.get(&candidate) != Some(&name.to_string())
        {
            let suffix = format!("_{counter}");
            let max_base = 64usize.saturating_sub(suffix.len());
            candidate = format!("{}{}", truncate_chars(&base, max_base), suffix);
            counter += 1;
        }
        if candidate.len() > 64 {
            candidate = truncate_chars(&candidate, 64);
        }

        forward.insert(name.to_string(), candidate.clone());
        reverse.insert(candidate, name.to_string());
    }

    (forward, reverse)
}

/// Walk the Converse request body and rewrite tool names in place using the
/// forward map: `toolConfig.tools[].toolSpec.name`,
/// `toolConfig.toolChoice.tool.name`, and `toolUse.name` blocks inside
/// assistant messages. Identity mappings (the common case) leave `out`
/// untouched.
fn apply_tool_name_forward_map(out: &mut Map<String, Value>, forward: &HashMap<String, String>) {
    let remap =
        |name: &str| -> Option<String> { forward.get(name).filter(|m| *m != name).cloned() };

    if let Some(Value::Object(tc)) = out.get_mut("toolConfig") {
        if let Some(Value::Array(tools)) = tc.get_mut("tools") {
            for tool in tools.iter_mut() {
                if let Some(Value::Object(spec)) = tool.get_mut("toolSpec") {
                    if let Some(Value::String(name)) = spec.get_mut("name") {
                        if let Some(renamed) = remap(name) {
                            *name = renamed;
                        }
                    }
                }
            }
        }
        if let Some(Value::Object(choice)) = tc.get_mut("toolChoice") {
            if let Some(Value::Object(tool)) = choice.get_mut("tool") {
                if let Some(Value::String(name)) = tool.get_mut("name") {
                    if let Some(renamed) = remap(name) {
                        *name = renamed;
                    }
                }
            }
        }
    }
    if let Some(choice) = out
        .get_mut("additionalModelRequestFields")
        .and_then(|fields| fields.get_mut("tool_choice"))
    {
        remap_native_choice_name(choice, forward);
    }

    if let Some(Value::Array(messages)) = out.get_mut("messages") {
        for msg in messages.iter_mut() {
            let Some(Value::Array(blocks)) = msg.get_mut("content") else {
                continue;
            };
            for block in blocks.iter_mut() {
                if let Some(Value::Object(tu)) = block.get_mut("toolUse") {
                    if let Some(Value::String(name)) = tu.get_mut("name") {
                        if let Some(renamed) = remap(name) {
                            *name = renamed;
                        }
                    }
                }
            }
        }
    }
}

fn remap_native_choice_name(choice: &mut Value, forward: &HashMap<String, String>) {
    if choice.get("type").and_then(Value::as_str) == Some("tool")
        && let Some(name) = choice
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned)
        && let Some(mapped) = forward.get(&name)
    {
        choice["name"] = json!(mapped);
    }
}

/// In-place reverse-map of `tool_calls[].function.name` on an OpenAI-format
/// response. Reads the per-thread map seeded by [`seed_tool_name_map`].
fn reverse_map_tool_call_names(tool_calls: &mut [Value]) {
    let map = TOOL_NAME_MAP.with(|c| c.borrow().clone());
    if map.is_empty() {
        return;
    }
    reverse_map_tool_call_names_in(tool_calls, &map);
}

/// In-place reverse-map of `tool_calls[].function.name` on an OpenAI-format
/// response, using an explicit reverse map (used by the streaming path, which
/// carries its own per-stream map).
fn reverse_map_tool_call_names_in(tool_calls: &mut [Value], reverse_map: &HashMap<String, String>) {
    if reverse_map.is_empty() {
        return;
    }
    for tc in tool_calls.iter_mut() {
        let Some(Value::Object(func)) = tc.get_mut("function") else {
            continue;
        };
        let Some(Value::String(name)) = func.get_mut("name") else {
            continue;
        };
        if let Some(original) = reverse_map.get(name) {
            if original != name {
                *name = original.clone();
            }
        }
    }
}

// =============================================================================
// Response transform
// =============================================================================

/// Transform a Bedrock Converse response body into an OpenAI Chat
/// Completion response.
///
/// Converse response shape (abbreviated):
/// ```json
/// {
///   "output": {"message": {"role": "assistant", "content": [...]}},
///   "stopReason": "end_turn",
///   "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
/// }
/// ```
///
/// Content block mapping:
/// * `{text: "…"}`             → concatenated into `message.content`
/// * `{toolUse: {…}}`          → `message.tool_calls[]`
/// * `{reasoningContent: {…}}` → `message.reasoning_content` + `thinking_blocks`
///   (so reasoning survives a subsequent tool-call turn)
pub fn transform_response_converse(body: &Value, model: &str) -> Result<Value> {
    let message = body.get("output").and_then(|o| o.get("message"));

    let mut content_str = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();

    if let Some(message) = message {
        if let Some(content) = message.get("content").and_then(Value::as_array) {
            // Track a separate tool-call counter so `tool_calls[].index`
            // reflects the position within the `tool_calls` array (0, 1, 2,
            // …) rather than the content-block index, which OpenAI's
            // non-streaming `chat.completion` shape expects. The reference does the
            // same in `_get_indexed_tool_calls`.
            let mut tool_call_idx: usize = 0;
            for block in content.iter() {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    content_str.push_str(text);
                }
                if let Some(tu) = block.get("toolUse") {
                    let id = tu
                        .get("toolUseId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("call_bedrock");
                    let name = tu.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = tu.get("input").cloned().unwrap_or_else(|| json!({}));
                    // OpenAI tool_call.function.arguments is a JSON-encoded
                    // string; serialise the structured `input` to match.
                    let arguments =
                        serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                    tool_calls.push(json!({
                        "index": tool_call_idx,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }));
                    tool_call_idx += 1;
                }
                if let Some(rc) = block.get("reasoningContent") {
                    collect_reasoning_block(rc, &mut reasoning_parts, &mut thinking_blocks);
                }
            }
        }
    }

    let finish_reason = map_finish_reason(body.get("stopReason").and_then(Value::as_str));

    // Restore caller tool names: `transform_request_converse` may have
    // sanitized them to fit Converse's `[a-zA-Z0-9_-]{1,64}` rule. Look
    // each name up in the reverse map and rewrite in place.
    if !tool_calls.is_empty() {
        reverse_map_tool_call_names(&mut tool_calls);
    }

    let usage = body.get("usage");
    let cache_read = usage
        .and_then(|u| u.get("cacheReadInputTokens"))
        .and_then(Value::as_u64);
    let cache_write = usage
        .and_then(|u| u.get("cacheWriteInputTokens"))
        .and_then(Value::as_u64);
    let input_tokens = usage
        .and_then(|u| u.get("inputTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = usage
        .and_then(|u| u.get("outputTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    // Inflate prompt_tokens by cache read + write so the OpenAI invariant
    // `cached_tokens ⊆ prompt_tokens` holds. Mirrors the Anthropic
    // provider's usage normalization (`anthropic/mod.rs:680`). Saturating
    // addition guards against adversarial upstream totals (e.g.
    // `u64::MAX` in both `inputTokens` and `cacheReadInputTokens`) that
    // would otherwise panic in debug / wrap in release.
    let prompt_tokens = input_tokens
        .saturating_add(cache_read.unwrap_or(0))
        .saturating_add(cache_write.unwrap_or(0));
    let total_tokens = usage
        .and_then(|u| u.get("totalTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(prompt_tokens.saturating_add(completion_tokens));

    let mut message_obj = Map::new();
    message_obj.insert("role".to_string(), json!("assistant"));
    message_obj.insert("content".to_string(), json!(content_str));
    let reasoning_text = reasoning_parts.join("");
    // Count by Unicode scalar values (not bytes) for the token estimate —
    // multi-byte reasoning text (CJK/emoji) would otherwise overestimate
    // token count by up to 4×, driving `text_tokens` to near-zero.
    let reasoning_chars = reasoning_text.chars().count();
    if !reasoning_text.is_empty() {
        message_obj.insert("reasoning_content".to_string(), json!(reasoning_text));
    }
    if !thinking_blocks.is_empty() {
        message_obj.insert("thinking_blocks".to_string(), Value::Array(thinking_blocks));
    }
    if !tool_calls.is_empty() {
        message_obj.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }

    let mut out = Map::new();
    out.insert("object".to_string(), json!("chat.completion"));
    out.insert(
        "id".to_string(),
        json!(format!("chatcmpl-bedrock-converse-{model}")),
    );
    out.insert(
        "created".to_string(),
        json!(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        ),
    );
    out.insert("model".to_string(), Value::String(model.to_string()));
    out.insert(
        "choices".to_string(),
        json!([{
            "index": 0,
            "message": Value::Object(message_obj),
            "finish_reason": finish_reason,
        }]),
    );
    let mut usage_obj = Map::new();
    usage_obj.insert("prompt_tokens".to_string(), json!(prompt_tokens));
    usage_obj.insert("completion_tokens".to_string(), json!(completion_tokens));
    usage_obj.insert("total_tokens".to_string(), json!(total_tokens));
    if let Some(v) = cache_read {
        let details = usage_obj
            .entry("prompt_tokens_details".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(details_map) = details {
            details_map.insert("cached_tokens".to_string(), json!(v));
        }
    }
    if let Some(v) = cache_write {
        // Subset form only (no top-level `cache_creation_input_tokens`
        // mirror): `prompt_tokens` above already folds the cache-write
        // tokens in, and the host usage extractor treats the top-level
        // anthropic form as ADDITIVE (outside `prompt_tokens`) — emitting
        // both would bill every cache-write token twice. Matching the
        // direct-Anthropic transform's shape keeps usage identical across
        // the two Claude surfaces.
        let details = usage_obj
            .entry("prompt_tokens_details".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(details_map) = details {
            details_map.insert("cache_creation_tokens".to_string(), json!(v));
        }
    }
    // completion_tokens_details: surface reasoning token breakdown.
    // Prefer Bedrock's explicit `outputTokensDetails.reasoningTokens` when
    // present; otherwise estimate from the reasoning text (~4 chars/token).
    let reasoning_tokens = usage
        .and_then(|u| u.get("outputTokensDetails"))
        .and_then(|d| d.get("reasoningTokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            if reasoning_chars > 0 {
                Some((reasoning_chars as u64).div_ceil(4))
            } else {
                None
            }
        });
    if let Some(rt) = reasoning_tokens {
        let text_tokens = completion_tokens.saturating_sub(rt);
        usage_obj.insert(
            "completion_tokens_details".to_string(),
            json!({"reasoning_tokens": rt, "text_tokens": text_tokens}),
        );
    }
    out.insert("usage".to_string(), Value::Object(usage_obj));

    // Surface the service tier Bedrock reports back.
    if let Some(st) = body
        .get("serviceTier")
        .and_then(|v| v.get("type"))
        .and_then(Value::as_str)
    {
        out.insert("service_tier".to_string(), json!(st));
    }
    // Surface the guardrail trace block verbatim.
    if let Some(trace) = body.get("trace") {
        out.insert("trace".to_string(), trace.clone());
    }

    Ok(Value::Object(out))
}

/// Pull reasoning text + raw thinking blocks out of a Converse
/// `reasoningContent` block. Bedrock emits reasoning two ways:
///
/// * `{reasoningText: {text, signature?}}` — visible thinking
/// * `{redactedContent: "<opaque>"}`       — safety-redacted thinking
///
/// Visible text is concatenated into `reasoning_parts`; both shapes are
/// preserved in `thinking_blocks` for round-trip into subsequent tool-call
/// turns (matching the InvokeModel Anthropic-on-Bedrock behaviour).
fn collect_reasoning_block(
    rc: &Value,
    reasoning_parts: &mut Vec<String>,
    thinking_blocks: &mut Vec<Value>,
) {
    if let Some(rt) = rc.get("reasoningText") {
        if let Some(t) = rt.get("text").and_then(Value::as_str) {
            reasoning_parts.push(t.to_string());
        }
        let mut tb = Map::new();
        tb.insert("type".to_string(), json!("thinking"));
        if let Some(t) = rt.get("text") {
            tb.insert("thinking".to_string(), t.clone());
        }
        if let Some(s) = rt.get("signature") {
            tb.insert("signature".to_string(), s.clone());
        }
        thinking_blocks.push(Value::Object(tb));
    }
    if let Some(redacted) = rc.get("redactedContent") {
        thinking_blocks.push(json!({
            "type": "redacted_thinking",
            "data": redacted.clone(),
        }));
    }
}

// =============================================================================
// Streaming chunk transform
// =============================================================================

/// State belongs to one stream and travels with the host's task. Converse
/// does not send an OpenAI completion id or timestamp, so create them once.
pub(super) struct ConverseStreamState {
    id: String,
    model: String,
    created: u64,
    thinking_blocks: HashMap<u64, Value>,
    tool_arguments: HashMap<u64, bool>,
}

impl ConverseStreamState {
    pub(super) fn new(model: &str) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            model: model.to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0),
            thinking_blocks: HashMap::new(),
            tool_arguments: HashMap::new(),
        }
    }
}

// Compatibility for direct, single-threaded callers of the older entrypoints.
// The production host supplies explicit state via BedrockStreamState instead.
thread_local! {
    static STREAM_STATE: RefCell<ConverseStreamState> =
        RefCell::new(ConverseStreamState::new("bedrock"));
}

/// Transform a single ConverseStream event payload into an OpenAI-shaped
/// streaming chunk.
///
/// Accepts the event either as a bare JSON object (raw eventstream payload
/// — the wire path) or as a boto-SDK-style single-key wrapper
/// (`{"messageStart": {…}}`). Returns `Ok(None)` for events that produce
/// no OpenAI output (ordinary content-block stops, unrecognised shapes,
/// parse failures).
///
/// Event mapping:
///
/// | Converse event            | OpenAI chunk                                  |
/// |---------------------------|-----------------------------------------------|
/// | `messageStart`            | `delta:{role:"assistant"}`                    |
/// | `contentBlockStart` (tool)| `delta:{tool_calls:[{index,id,name,args:""}]}`|
/// | `contentBlockDelta` (text)| `delta:{content:"…"}`                         |
/// | `contentBlockDelta` (tool)| `delta:{tool_calls:[{index, function.args}]}` |
/// | `contentBlockDelta` (think)| `delta:{reasoning_content:"…"}`              |
/// | `contentBlockStop`        | completed thinking block / empty tool args     |
/// | `messageStop`             | `finish_reason` only                          |
/// | `metadata`                | `usage` chunk (with empty choices)            |
pub fn transform_stream_chunk_converse(line: &str) -> Result<Option<Value>> {
    // Read the per-thread map seeded by the stateful response transform.
    // The streaming path normally goes through
    // `transform_stream_chunk_converse_with_state` instead (with host-owned
    // state); this entry point covers direct callers and compatibility tests.
    let reverse_map = TOOL_NAME_MAP.with(|c| c.borrow().clone());
    transform_stream_chunk_converse_stateful(line, &reverse_map)
}

/// Compatibility entrypoint with an explicit reverse-name map. Other stream
/// state is thread-local here; the production ChatProvider implementation
/// uses transform_stream_chunk_converse_with_state and host-owned state.
pub fn transform_stream_chunk_converse_stateful(
    line: &str,
    reverse_map: &HashMap<String, String>,
) -> Result<Option<Value>> {
    STREAM_STATE.with(|state| {
        transform_stream_chunk_converse_with_state(line, reverse_map, &mut state.borrow_mut())
    })
}

pub(super) fn transform_stream_chunk_converse_with_state(
    line: &str,
    reverse_map: &HashMap<String, String>,
    state: &mut ConverseStreamState,
) -> Result<Option<Value>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(None);
    }
    let payload_str = strip_data_prefix(trimmed);
    let value: Value = match serde_json::from_str(&payload_str) {
        Ok(v) => v,
        Err(_) => {
            // Not JSON — likely a partial frame or keep-alive. Skip.
            return Ok(None);
        }
    };
    if let Some(error) = super::bedrock_stream_error(&value) {
        return Err(error);
    }

    let (event_type, payload) = classify_event(&value);
    let index = payload
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mut chunk = match event_type {
        "messageStart" => {
            *state = ConverseStreamState::new(&state.model);
            Ok(Some(delta_chunk(json!({"role": "assistant"}), Value::Null)))
        }
        "contentBlockStart" => {
            if payload
                .pointer("/start/toolUse")
                .is_some_and(Value::is_object)
            {
                state.tool_arguments.insert(index, false);
            }
            handle_block_start(payload, reverse_map)
        }
        "contentBlockDelta" => {
            if let Some(received) = state.tool_arguments.get_mut(&index) {
                *received |= payload
                    .pointer("/delta/toolUse/input")
                    .and_then(Value::as_str)
                    .is_some_and(|input| !input.is_empty());
            }
            if let Some(reasoning) = payload.pointer("/delta/reasoningContent") {
                let block = state
                    .thinking_blocks
                    .entry(index)
                    .or_insert_with(|| json!({"type": "thinking", "thinking": ""}));
                if let Some(data) = reasoning.get("redactedContent") {
                    if block.get("data").is_some() {
                        // Each blob is independently base64 encoded; a
                        // string append would corrupt the opaque bytes.
                        return Err(oxllm_core::Error::Transform(
                                "fragmented Bedrock redactedContent cannot be replayed losslessly by this adapter".into(),
                            ));
                    }
                    *block = json!({"type": "redacted_thinking", "data": data});
                } else {
                    for (source, target) in [("text", "thinking"), ("signature", "signature")] {
                        if let Some(fragment) = reasoning.get(source).and_then(Value::as_str) {
                            if let Some(object) = block.as_object_mut() {
                                let value = object
                                    .entry(target.to_string())
                                    .or_insert_with(|| json!(""));
                                if let Value::String(text) = value {
                                    text.push_str(fragment);
                                }
                            }
                        }
                    }
                }
            }
            handle_block_delta(payload)
        }
        "contentBlockStop" => {
            if state.tool_arguments.remove(&index) == Some(false) {
                Ok(Some(delta_chunk(
                    json!({"tool_calls":[{"index":index,"function":{"arguments":"{}"}}]}),
                    Value::Null,
                )))
            } else if let Some(block) = state.thinking_blocks.remove(&index) {
                Ok(Some(delta_chunk(
                    json!({"thinking_blocks": [block]}),
                    Value::Null,
                )))
            } else {
                Ok(None)
            }
        }
        "messageStop" => {
            let finish = map_finish_reason(payload.get("stopReason").and_then(Value::as_str));
            Ok(Some(delta_chunk(Value::Object(Map::new()), finish)))
        }
        "metadata" => handle_metadata(payload),
        _ => Ok(None),
    }?;
    if let Some(chunk) = chunk.as_mut().and_then(Value::as_object_mut) {
        chunk.insert("id".into(), json!(state.id));
        chunk.insert("model".into(), json!(state.model));
        chunk.insert("created".into(), json!(state.created));
    }
    Ok(chunk)
}

/// Handle a `contentBlockStart` event. Tool-use starts emit a tool_call
/// delta carrying the call id and name (arguments empty — filled in by
/// subsequent `contentBlockDelta` events). Reasoning-content starts
/// (redacted thinking only — visible thinking arrives via deltas) emit
/// an empty `reasoning_content` marker plus the redacted block.
fn handle_block_start(
    payload: &Value,
    reverse_map: &HashMap<String, String>,
) -> Result<Option<Value>> {
    let Some(start) = payload.get("start") else {
        return Ok(None);
    };
    let idx = payload
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if let Some(tu) = start.get("toolUse") {
        let id = tu
            .get("toolUseId")
            .and_then(|v| v.as_str())
            .unwrap_or("call_bedrock");
        let raw_name = tu.get("name").and_then(|v| v.as_str()).unwrap_or("");
        // Restore the caller's original tool name if it was sanitized on the
        // request side, using the explicit reverse map threaded in by the
        // caller (per-thread for the non-stateful path, per-stream for the
        // stateful path).
        let name = reverse_map
            .get(raw_name)
            .cloned()
            .unwrap_or_else(|| raw_name.to_string());
        return Ok(Some(delta_chunk(
            json!({
                "tool_calls": [{
                    "index": idx,
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }]
            }),
            Value::Null,
        )));
    }
    if let Some(rc) = start.get("reasoningContent") {
        if let Some(redacted) = rc.get("redactedContent") {
            return Ok(Some(delta_chunk(
                json!({
                    "reasoning_content": "",
                    "thinking_blocks": [{
                        "type": "redacted_thinking",
                        "data": redacted.clone(),
                    }],
                }),
                Value::Null,
            )));
        }
    }
    Ok(None)
}

/// Handle a `contentBlockDelta` event. Three payload shapes:
///
/// * `{text: "…"}`                              → content delta
/// * `{toolUse: {input: "…"}}`                  → tool_call argument delta
/// * `{reasoningContent: {text: "…"}}`          → reasoning_content delta
/// * `{reasoningContent: {signature | redactedContent}}` → empty marker
fn handle_block_delta(payload: &Value) -> Result<Option<Value>> {
    let Some(delta) = payload.get("delta") else {
        return Ok(None);
    };
    let idx = payload
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if let Some(text) = delta.get("text").and_then(Value::as_str) {
        return Ok(Some(delta_chunk(json!({"content": text}), Value::Null)));
    }
    if let Some(tu) = delta.get("toolUse") {
        let args = tu.get("input").and_then(Value::as_str).unwrap_or("");
        return Ok(Some(delta_chunk(
            json!({
                "tool_calls": [{
                    "index": idx,
                    "function": {"arguments": args},
                }]
            }),
            Value::Null,
        )));
    }
    if let Some(rc) = delta.get("reasoningContent") {
        if let Some(text) = rc.get("text").and_then(Value::as_str) {
            return Ok(Some(delta_chunk(
                json!({"reasoning_content": text}),
                Value::Null,
            )));
        }
        // signature-only or redactedContent delta — emit empty marker so
        // downstream consumers can flag thinking without text content.
        if rc.get("signature").is_some() || rc.get("redactedContent").is_some() {
            return Ok(Some(delta_chunk(
                json!({"reasoning_content": ""}),
                Value::Null,
            )));
        }
    }
    Ok(None)
}

/// Handle a `metadata` event — emits a chunk carrying the final `usage`
/// totals. Bedrock emits this once, near the end of the stream.
fn handle_metadata(payload: &Value) -> Result<Option<Value>> {
    let Some(usage) = payload.get("usage") else {
        return Ok(None);
    };
    let cache_read = usage.get("cacheReadInputTokens").and_then(Value::as_u64);
    let cache_write = usage.get("cacheWriteInputTokens").and_then(Value::as_u64);
    let input_tokens = usage
        .get("inputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion = usage
        .get("outputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let prompt = input_tokens
        .saturating_add(cache_read.unwrap_or(0))
        .saturating_add(cache_write.unwrap_or(0));
    let total = usage
        .get("totalTokens")
        .and_then(Value::as_u64)
        .unwrap_or(prompt.saturating_add(completion));
    let mut usage_obj = Map::new();
    usage_obj.insert("prompt_tokens".to_string(), json!(prompt));
    usage_obj.insert("completion_tokens".to_string(), json!(completion));
    usage_obj.insert("total_tokens".to_string(), json!(total));
    if let Some(v) = cache_read {
        let details = usage_obj
            .entry("prompt_tokens_details".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(details_map) = details {
            details_map.insert("cached_tokens".to_string(), json!(v));
        }
    }
    if let Some(v) = cache_write {
        // Subset form only — same double-bill rationale as the non-streaming
        // Converse transform above (`prompt_tokens` already folds the write
        // tokens in; the host extractor bills a top-level anthropic-form
        // count additively on top).
        let details = usage_obj
            .entry("prompt_tokens_details".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(details_map) = details {
            details_map.insert("cache_creation_tokens".to_string(), json!(v));
        }
    }
    // completion_tokens_details: surface the reasoning-token breakdown when
    // Bedrock reports `outputTokensDetails.reasoningTokens`, mirroring the
    // non-streaming Converse response transform. Reasoning tokens are already
    // counted inside `completion_tokens`; this only adds the detail split.
    if let Some(rt) = usage
        .get("outputTokensDetails")
        .and_then(|d| d.get("reasoningTokens"))
        .and_then(Value::as_u64)
    {
        let text_tokens = completion.saturating_sub(rt);
        usage_obj.insert(
            "completion_tokens_details".to_string(),
            json!({"reasoning_tokens": rt, "text_tokens": text_tokens}),
        );
    }
    let mut chunk = Map::new();
    chunk.insert("object".to_string(), json!("chat.completion.chunk"));
    chunk.insert("choices".to_string(), json!([]));
    chunk.insert("usage".to_string(), Value::Object(usage_obj));
    Ok(Some(Value::Object(chunk)))
}

/// Identify a ConverseStream event by its top-level shape. Handles both
/// the raw-eventstream form (event type inferred from payload keys) and
/// the boto-SDK wrapper form (`{"<eventType>": {…}}`).
///
/// Returns `("unknown", value)` for unrecognised shapes.
fn classify_event(value: &Value) -> (&'static str, &Value) {
    // boto-SDK-style single-key wrapper
    if let Some(obj) = value.as_object() {
        if obj.len() == 1 {
            if let Some((k, v)) = obj.iter().next() {
                match k.as_str() {
                    "messageStart" => return ("messageStart", v),
                    "contentBlockStart" => return ("contentBlockStart", v),
                    "contentBlockDelta" => return ("contentBlockDelta", v),
                    "contentBlockStop" => return ("contentBlockStop", v),
                    "messageStop" => return ("messageStop", v),
                    "metadata" => return ("metadata", v),
                    _ => {}
                }
            }
        }
    }
    // Raw eventstream payload — infer from characteristic keys. The order
    // matters: `contentBlockIndex` alone is the `contentBlockStop` payload,
    // but it also appears alongside `start`/`delta` for start/delta events.
    if value.get("role").is_some() {
        return ("messageStart", value);
    }
    if value.get("stopReason").is_some() {
        return ("messageStop", value);
    }
    if value.get("usage").is_some() || value.get("metrics").is_some() {
        return ("metadata", value);
    }
    if value.get("delta").is_some() {
        return ("contentBlockDelta", value);
    }
    if value.get("start").is_some() {
        return ("contentBlockStart", value);
    }
    if value.get("contentBlockIndex").is_some() {
        return ("contentBlockStop", value);
    }
    ("unknown", value)
}

/// Strip an optional SSE `data:` prefix. Handles both `data: ` (with space)
/// and `data:` (without space) forms, mirroring the InvokeModel chunk
/// extractor in `mod.rs`.
fn strip_data_prefix(s: &str) -> String {
    super::extract_payload(s)
}

/// Map a Converse `stopReason` onto an OpenAI `finish_reason`. Unknown
/// values default to `stop` (matching the reference finish-reason mapping).
pub(crate) fn map_finish_reason(reason: Option<&str>) -> Value {
    let mapped = match reason {
        Some("end_turn") | Some("stop_sequence") | Some("stop") => "stop",
        Some("max_tokens" | "length" | "model_context_window_exceeded") => "length",
        Some("tool_use") | Some("tool_calls") => "tool_calls",
        Some("content_filter")
        | Some("content_filtered")
        | Some("guardrail_intervened")
        | Some("refusal")
        | Some("guardrailIntervened") => "content_filter",
        Some(_) | None => "stop",
    };
    json!(mapped)
}

/// Build an OpenAI-shaped streaming chunk with a single choice.
fn delta_chunk(delta: Value, finish_reason: Value) -> Value {
    json!({
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    })
}

/// Heuristic: does this parsed chunk look like a Converse event? Used by
/// the parent provider to dispatch between the Converse SSE parser and
/// the legacy InvokeModel eventstream parser without needing deployment
/// context (the streaming chunk trait method only receives the raw line).
///
/// # Anthropic vs. Converse disambiguation
///
/// Anthropic `InvokeModel` streaming events carry a top-level `"type"`
/// field whose value is snake_case (`message_start`, `content_block_delta`,
/// `message_delta`, `message_stop`, `ping`, `error`, …). Converse's wire
/// shape **never** uses a top-level snake_case `type` discriminator — it
/// identifies events either by a boto-style single-key wrapper
/// (`{"messageStart": …}`) or by characteristic camelCase fields
/// (`stopReason`, `contentBlockIndex`, …).
///
/// This matters because Anthropic events *also* happen to carry `delta`
/// and `usage` fields that `classify_event` keys on. Without this guard,
/// `{"type":"content_block_delta",...,"delta":{...}}` is misclassified as
/// Converse `contentBlockDelta`, and `{"type":"message_delta","delta":{...},
/// "usage":{...}}` is misclassified as Converse `metadata` — silently
/// dropping the finish_reason and zeroing usage. A snake_case `type` is
/// therefore a strong "this is Anthropic, not Converse" signal and must
/// short-circuit the heuristic before the field-shape checks run.
pub(crate) fn looks_like_converse_event(value: &Value) -> bool {
    // (1) Anthropic snake_case `type` discriminator → not Converse.
    if let Some(t) = value.get("type").and_then(Value::as_str) {
        if t.contains('_') {
            return false;
        }
    }
    // (2/3) Converse boto wrapper (`{"messageStart": …}`) or characteristic
    // camelCase fields (`role`, `stopReason`, `usage` w/o `type`,
    // `contentBlockIndex`, `delta`, `start`). (4) Default: unknown → false.
    let (kind, _) = classify_event(value);
    kind != "unknown"
}

/// Heuristic: does this response body look like a Converse response?
/// Used by the parent provider's `transform_response` to dispatch.
pub(crate) fn looks_like_converse_response(body: &Value) -> bool {
    body.get("output").and_then(|o| o.get("message")).is_some()
}

// =============================================================================
// Converse beta-header filtering
// =============================================================================
//
// Bedrock Converse rejects a subset of the beta features that native
// Anthropic accepts. The host layer is responsible for setting HTTP headers;
// this helper strips the unsupported values from the `anthropic-beta` header
// (case-insensitive key lookup) so the host can call it just before the
// request goes out on the Converse path. Reference behavior:
// `UNSUPPORTED_BEDROCK_CONVERSE_BETA_PATTERNS`
// (`converse_transformation.py:97`).

/// `anthropic-beta` value prefixes that Bedrock Converse rejects. Any header
/// value starting with one of these (after trimming) is dropped; other
/// values in the comma-separated list are kept.
const UNSUPPORTED_CONVERSE_BETA_PREFIXES: &[&str] =
    &["advanced-tool-use", "prompt-caching", "compact-"];

/// Strip beta-header values that Bedrock Converse rejects from the
/// `anthropic-beta` header in `headers`.
///
/// The header value is a comma-separated list of beta feature tokens. Each
/// token whose prefix matches `UNSUPPORTED_CONVERSE_BETA_PREFIXES` is
/// dropped; surviving tokens are re-joined. When the list ends up empty the
/// whole header is removed. No-op when the header is absent.
pub fn filter_converse_beta_headers(headers: &mut HashMap<String, String>) {
    let key = headers
        .keys()
        .find(|k| k.eq_ignore_ascii_case("anthropic-beta"))
        .cloned();
    let Some(key) = key else {
        return;
    };
    let Some(value) = headers.get(&key).cloned() else {
        return;
    };
    let kept: Vec<&str> = value
        .split(',')
        .map(|s| s.trim())
        .filter(|s| {
            !s.is_empty()
                && !UNSUPPORTED_CONVERSE_BETA_PREFIXES
                    .iter()
                    .any(|p| s.starts_with(p))
        })
        .collect();
    if kept.is_empty() {
        headers.remove(&key);
    } else {
        headers.insert(key, kept.join(","));
    }
}

#[cfg(test)]
#[path = "converse_compatibility_tests.rs"]
mod compatibility_tests;
