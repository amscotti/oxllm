//! AWS Bedrock provider — translates OpenAI ⇄ Bedrock InvokeModel / Converse.
//!
//! Wire target:
//! `POST https://bedrock-runtime.{region}.amazonaws.com/model/{model-id}/invoke`
//! (non-streaming) or `/invoke-with-response-stream` (streaming). Converse
//! deployments use `/converse` or `/converse-stream` instead.
//!
//! Auth is AWS SigV4 (handled by the host before the request goes out — see
//! `oxllm_server::bedrock_sigv4`). The provider transform only owns body and
//! URL construction; the SigV4 signature is added in the upstream pipeline.
//!
//! Bedrock hosts models from several vendors behind a uniform `InvokeModel`
//! surface. Each vendor accepts its own native request body: Claude models
//! take the Anthropic Messages body (with `anthropic_version` inlined),
//! Llama/Mistral take their own JSON shape, Titan takes Amazon's. This
//! provider maps the known vendor bodies and uses Converse for model families
//! that require its unified messages surface. Unknown custom InvokeModel
//! deployments retain an OpenAI-shaped passthrough body.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde_json::{Map, Value, json};

use crate::anthropic::AnthropicProvider;
use crate::util::data_url_to_anthropic_image;

/// Characters that must be percent-encoded inside a single URL path segment.
/// Includes `/` so application-inference-profile ARNs (which contain `/`)
/// stay one segment instead of being split into multiple path parts.
const MODEL_PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'<')
    .add(b'>')
    .add(b'`')
    .add(b'#')
    .add(b'?')
    .add(b'{')
    .add(b'}')
    .add(b'/')
    .add(b'\\')
    .add(b'%');

/// Characters that must be percent-encoded inside a query-string value so
/// guardrail identifiers cannot inject extra query parameters or a fragment.
const QUERY_VALUE: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'<')
    .add(b'>')
    .add(b'&')
    .add(b'=')
    .add(b'+')
    .add(b'%');

pub mod converse;
pub mod realtime;

pub use realtime::BedrockRealtimeTransform;

/// The `anthropic_version` value Bedrock requires in the request body for
/// Anthropic models. (Native Anthropic uses the `anthropic-version` header
/// instead; Bedrock inlines it.)
pub const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// The AWS Bedrock provider.
///
/// See the module docs for the wire target and the Anthropic-on-Bedrock focus.
#[derive(Debug, Clone, Default)]
pub struct BedrockProvider;

/// The underlying vendor behind a Bedrock model id, used for routing
/// decisions on the InvokeModel path. Named `BedrockVendor` rather than
/// `BedrockProvider` to avoid colliding with the provider struct above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedrockVendor {
    /// Anthropic Claude models (`anthropic.*`).
    Anthropic,
    /// Amazon Nova models (`amazon.nova*`).
    Nova,
    /// Amazon Titan models (`amazon.titan*`).
    Titan,
    /// Other Amazon models not covered by Nova/Titan.
    Amazon,
    /// Meta Llama models (`meta.*`).
    Llama,
    /// Mistral models (`mistral.*`).
    Mistral,
    /// AI21 Jamba models (`ai21.*`).
    AI21,
    /// Cohere models (`cohere.*`).
    Cohere,
    /// DeepSeek models (`deepseek.*`).
    DeepSeek,
    /// Qwen models (`qwen.*`).
    Qwen,
    /// Vendor could not be determined from the model id.
    Unknown,
}

/// Model id prefixes that MUST use the Converse API because InvokeModel
/// does not support them (or supports them poorly). Mirrors LiteLLM's
/// `bedrock_converse_models` list. Matched with `starts_with` against the
/// base model id (after stripping routing prefixes and version/throughput
/// suffixes) so versioned variants like `deepseek.r1-v1:0` also match.
///
/// `amazon.nova` — Nova's InvokeModel body is its own `messages-v1` schema,
/// NOT OpenAI-shaped, so the Unknown-vendor passthrough forwards keys Nova
/// rejects (`extraneous key [max_tokens]`). Converse is the documented path
/// (verified live 2026-08: nova-micro/lite Converse OK, InvokeModel
/// passthrough 400s).
///
/// `deepseek.` — the Bedrock DeepSeek bodies require an OpenAI-style
/// `messages` array; the legacy `prompt` InvokeModel shape fails with
/// `missing field 'messages'` (verified live on deepseek.v3.2; Converse OK).
///
/// New-generation Mistral (`ministral-3`, `mistral-large-3`, `devstral`,
/// `magistral`) — same `missing field 'messages'` failure via the legacy
/// `prompt` shape (verified live on ministral-3-8b). The LEGACY Mistral ids
/// (`mistral.mistral-7b`, `mistral.mixtral-*`, `mistral.mistral-*-2402`,
/// `mistral.mistral-small-*`) still accept `prompt` on InvokeModel and keep
/// that path.
const MUST_USE_CONVERSE: &[&str] = &[
    "ai21.jamba",
    "qwen.qwen3",
    "amazon.nova",
    "deepseek.",
    "openai.gpt-5",
    "openai.gpt-6",
    "mistral.ministral-3",
    "mistral.mistral-large-3",
    "mistral.devstral",
    "mistral.magistral",
];

impl BedrockProvider {
    /// Returns true when the deployment targets an Anthropic model on Bedrock
    /// (model id starts with `anthropic.`, also behind a cross-region
    /// inference prefix like `us.` / `eu.` / `apac.`). Those models take the
    /// Anthropic Messages body shape with `anthropic_version` inlined.
    fn is_anthropic_model(model: &str) -> bool {
        detect_bedrock_provider(model) == BedrockVendor::Anthropic
    }

    /// Resolve the configured model id, honouring wildcards. `bedrock/*` and
    /// `*` mean "route whatever the client sent" — the actual model is read
    /// from the request body (with any leading `bedrock/` stripped).
    fn resolve_model<'a>(request: &'a Value, deployment: &'a Deployment) -> &'a str {
        let configured = deployment
            .oxllm_params
            .model
            .strip_prefix("bedrock/")
            .unwrap_or(&deployment.oxllm_params.model);
        if configured.contains('*') {
            request
                .get("model")
                .and_then(|v| v.as_str())
                .map(|m| m.strip_prefix("bedrock/").unwrap_or(m))
                .unwrap_or(configured)
        } else {
            configured
        }
    }

    /// Resolve inline guardrail configuration from the deployment's `extra`
    /// params. Accepts either LiteLLM-style snake_case (`guardrail_id`,
    /// `guardrail_version`) or AWS-native camelCase (`guardrailIdentifier`,
    /// `guardrailVersion`). Returns `(id, version, trace)` when both id and
    /// version are present — Bedrock requires both as URL query parameters.
    fn guardrail_config(deployment: &Deployment) -> Option<(String, String, bool)> {
        let extra = &deployment.oxllm_params.extra;
        let id = extra
            .get("guardrail_id")
            .and_then(Value::as_str)
            .or_else(|| extra.get("guardrailIdentifier").and_then(Value::as_str))?;
        let version = extra
            .get("guardrail_version")
            .and_then(Value::as_str)
            .or_else(|| extra.get("guardrailVersion").and_then(Value::as_str))?;
        let trace = extra.get("trace").and_then(Value::as_bool).unwrap_or(false);
        Some((id.to_string(), version.to_string(), trace))
    }

    /// Resolve the `performanceConfig` body field from `extra.performance_config`
    /// or `extra.performanceConfig`. Returns the value verbatim — callers are
    /// expected to pass through shapes like `{"latency": "optimized"}`.
    fn performance_config_value(deployment: &Deployment) -> Option<Value> {
        let extra = &deployment.oxllm_params.extra;
        extra
            .get("performance_config")
            .or_else(|| extra.get("performanceConfig"))
            .cloned()
    }

    /// Read the Converse-API opt-in flag from the deployment's `extra`
    /// params. LiteLLM accepts either `use_converse_api` (canonical) or
    /// `use_converse` (alias). Defaults to `false` — InvokeModel stays the
    /// primary path until Batch 4 flips the default.
    fn use_converse(deployment: &Deployment) -> bool {
        deployment
            .oxllm_params
            .extra
            .get("use_converse_api")
            .or_else(|| deployment.oxllm_params.extra.get("use_converse"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}

impl ChatProvider for BedrockProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        // Default host must match the region used for SigV4. Prefer
        // `api_base`, then `extra.aws_region`, else us-east-1.
        let region = deployment
            .oxllm_params
            .extra
            .get("aws_region_name")
            .or_else(|| deployment.oxllm_params.extra.get("aws_region"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("us-east-1");
        let default_base = format!("https://bedrock-runtime.{region}.amazonaws.com");
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(&default_base)
            .trim_end_matches('/');
        let model = &deployment.oxllm_params.model;
        let model_stripped = model.strip_prefix("bedrock/").unwrap_or(model);
        // Strip LiteLLM throughput/context-window suffixes (`:51k`, `[1m]`)
        // so the upstream model id is clean — Bedrock would reject the
        // suffix as part of the path.
        let model_stripped = strip_model_suffixes(model_stripped);
        // Percent-encode as a single path segment so ARNs with `/` (e.g.
        // application-inference-profile/…) are not split into extra segments.
        let model_encoded = utf8_percent_encode(&model_stripped, MODEL_PATH_SEGMENT);
        // Converse uses a different path suffix (`/converse`) than InvokeModel
        // (`/invoke`). The streaming variant (`/converse-stream`) is derived
        // by the host's URL rewriter — see `build_upstream_url`. Inline
        // guardrails (when configured) attach as query params on either path.
        let converse = should_use_converse(&model_stripped, Self::use_converse(deployment));
        let suffix = if converse { "/converse" } else { "/invoke" };
        let mut url = format!("{base}/model/{model_encoded}{suffix}");
        // AWS Bedrock applies inline guardrails differently depending on the
        // API surface: InvokeModel uses URL query parameters, while Converse
        // takes a `guardrailConfig` body field (injected by the Converse
        // transform). Appending query params on the Converse path would both
        // be rejected by Bedrock AND break the host's `/converse` →
        // `/converse-stream` streaming rewrite (the URL would no longer end
        // with `/converse`).
        if !converse {
            if let Some((id, version, trace)) = Self::guardrail_config(deployment) {
                let id_enc = utf8_percent_encode(&id, QUERY_VALUE);
                let version_enc = utf8_percent_encode(&version, QUERY_VALUE);
                url.push_str("?guardrailIdentifier=");
                url.push_str(&id_enc.to_string());
                url.push_str("&guardrailVersion=");
                url.push_str(&version_enc.to_string());
                if trace {
                    url.push_str("&trace=enabled");
                }
            }
        }
        url
    }

    fn endpoint_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        let mut resolved = deployment.clone();
        resolved.oxllm_params.model =
            format!("bedrock/{}", Self::resolve_model(request, deployment));
        self.endpoint_url(&resolved)
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        _creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        // SigV4 signing lives in the host (`oxllm_server::bedrock_sigv4`),
        // which has access to the serialized body and full URL. `apply_auth`
        // is called before the body is serialized, so we cannot sign here.
        //
        // One thing we DO own here: stripping beta-header values that the
        // Bedrock Converse API rejects (`advanced-tool-use-*`,
        // `prompt-caching-*`, `compact-*`). These arrive via the host's
        // header pass-through and would cause a 400 on the Converse path.
        let cfg_model = deployment
            .oxllm_params
            .model
            .strip_prefix("bedrock/")
            .unwrap_or(&deployment.oxllm_params.model);
        let cfg_model = strip_model_suffixes(cfg_model);
        if should_use_converse(&cfg_model, Self::use_converse(deployment)) {
            converse::filter_converse_beta_headers(headers);
        }
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let model = Self::resolve_model(request, deployment);
        if model.is_empty()
            || model.contains('*')
            || model
                .split(['/', '\\'])
                .any(|segment| segment == "." || segment == "..")
        {
            return Err(oxllm_core::Error::BadRequest(
                "Bedrock requires a concrete model id without traversal segments".into(),
            ));
        }

        // Converse-API path: route every model family through the unified
        // Converse transform when the deployment opts in via
        // `use_converse_api: true`, or when auto-routing kicks in (inference
        // profile ARNs, must-use-Converse models like `deepseek.r1`). Skips
        // the Anthropic / passthrough split entirely — Converse standardises
        // across all vendors.
        if should_use_converse(model, Self::use_converse(deployment)) {
            return converse::transform_request_converse_with_base_model(
                request,
                model,
                &deployment.oxllm_params.extra,
                deployment
                    .model_info
                    .as_ref()
                    .and_then(|info| info.base_model.as_deref()),
            );
        }

        if Self::is_anthropic_model(model) {
            // Run the standard Anthropic transform, then inline
            // `anthropic_version` and strip `stream` (Bedrock encodes streaming
            // in the URL path, not the body).
            let anthropic = AnthropicProvider;
            let mut body = anthropic.transform_request(request, deployment)?;
            let message_config_used = crate::anthropic::uses_message_output_config(&body);
            if let Value::Object(map) = &mut body {
                map.insert("anthropic_version".to_string(), json!(ANTHROPIC_VERSION));
                map.remove("stream");
                // InvokeModel carries beta selectors in the native body. Keep
                // explicit tool betas alongside any message-config beta.
                let mut betas = match request
                    .get("anthropic_beta")
                    .filter(|value| !value.is_null())
                {
                    Some(Value::Array(betas))
                        if betas
                            .iter()
                            .all(|value| value.as_str().is_some_and(|beta| !beta.is_empty())) =>
                    {
                        betas.clone()
                    }
                    Some(_) => {
                        return Err(oxllm_core::Error::BadRequest(
                            "Bedrock anthropic_beta must be an array of non-empty strings".into(),
                        ));
                    }
                    None => Vec::new(),
                };
                if message_config_used {
                    let beta = crate::anthropic::BETA_MID_CONVERSATION_OUTPUT_CONFIG;
                    if !betas.iter().any(|value| value.as_str() == Some(beta)) {
                        betas.push(json!(beta));
                    }
                }
                if !betas.is_empty() {
                    map.insert("anthropic_beta".to_string(), Value::Array(betas));
                }
                // Bedrock's Anthropic Messages adapter rejects a `model` body
                // field ("model: Extra inputs are not permitted") — the model
                // id lives in the URL path. Wildcard deployments resolve that
                // path through endpoint_url_for_request, so their bodies also
                // omit model.
                map.remove("model");
            }
            // Ensure any remaining data-URL image_url blocks use the native
            // Anthropic image shape before InvokeModel serialization.
            convert_image_url_blocks(&mut body);
            Ok(body)
        } else {
            // Non-Anthropic Bedrock models: dispatch to per-vendor
            // InvokeModel transforms. Each vendor accepts its own native
            // request body shape (Titan's `inputText`/`textGenerationConfig`,
            // Llama's `prompt`/`max_gen_len`, AI21's camelCase params, …),
            // distinct from the OpenAI Chat Completions body. Unknown vendors
            // (custom/imported models without a recognised prefix, plus
            // Amazon Nova which is handled via the Converse path) fall
            // through to a passthrough that forwards the OpenAI-shaped body
            // verbatim with the resolved model id.
            let messages: &[Value] = request
                .get("messages")
                .and_then(Value::as_array)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let vendor = detect_invoke_vendor(model);
            let normalized_messages;
            let messages = if matches!(vendor, InvokeVendor::Unknown) {
                messages
            } else {
                normalized_messages = normalize_invoke_prompt_messages(request, messages)?;
                &normalized_messages
            };
            // Common optional OpenAI inference params, hoisted out so each
            // vendor arm only pulls the fields it actually accepts.
            let max_tokens = request
                .get("max_tokens")
                .or_else(|| request.get("max_completion_tokens"));
            let temperature = request.get("temperature");
            let top_p = request.get("top_p");
            let top_k = request.get("top_k");
            let stop = request
                .get("stop")
                .filter(|value| !value.is_null())
                .map(crate::anthropic::normalize_stop_sequences);
            let frequency_penalty = request.get("frequency_penalty");
            let presence_penalty = request.get("presence_penalty");
            let mut out: Value = match vendor {
                InvokeVendor::Titan => {
                    let prompt = messages_to_prompt(messages);
                    let mut tgc = Map::new();
                    if let Some(v) = max_tokens {
                        tgc.insert("maxTokenCount".into(), v.clone());
                    }
                    if let Some(v) = temperature {
                        tgc.insert("temperature".into(), v.clone());
                    }
                    if let Some(v) = top_p {
                        tgc.insert("topP".into(), v.clone());
                    }
                    if let Some(v) = stop {
                        tgc.insert("stopSequences".into(), v.clone());
                    }
                    let mut body = Map::new();
                    body.insert("inputText".into(), Value::String(prompt));
                    body.insert("textGenerationConfig".into(), Value::Object(tgc));
                    Value::Object(body)
                }
                InvokeVendor::Llama => {
                    // Meta Llama needs its vendor-specific chat template —
                    // plain concatenation makes the model echo the prompt.
                    // Llama 3 (every `meta.llama*` InvokeModel id on Bedrock
                    // today) uses the official Llama-3 special-token layout;
                    // Llama 2 would need the `[INST]<<SYS>>...` template but
                    // has no InvokeModel id in current use, so we apply the
                    // Llama-3 template unconditionally. See
                    // [`messages_to_llama3_prompt`].
                    let prompt = messages_to_llama3_prompt(messages);
                    let mut body = Map::new();
                    body.insert("prompt".into(), Value::String(prompt));
                    if let Some(v) = max_tokens {
                        body.insert("max_gen_len".into(), v.clone());
                    }
                    if let Some(v) = temperature {
                        body.insert("temperature".into(), v.clone());
                    }
                    if let Some(v) = top_p {
                        body.insert("top_p".into(), v.clone());
                    }
                    Value::Object(body)
                }
                InvokeVendor::Mistral => {
                    let prompt = messages_to_prompt(messages);
                    let mut body = Map::new();
                    body.insert("prompt".into(), Value::String(prompt));
                    if let Some(v) = max_tokens {
                        body.insert("max_tokens".into(), v.clone());
                    }
                    if let Some(v) = temperature {
                        body.insert("temperature".into(), v.clone());
                    }
                    if let Some(v) = top_p {
                        body.insert("top_p".into(), v.clone());
                    }
                    if let Some(v) = top_k {
                        body.insert("top_k".into(), v.clone());
                    }
                    if let Some(v) = stop {
                        body.insert("stop".into(), v.clone());
                    }
                    Value::Object(body)
                }
                InvokeVendor::AI21 => {
                    let prompt = messages_to_prompt(messages);
                    let mut body = Map::new();
                    body.insert("prompt".into(), Value::String(prompt));
                    if let Some(v) = max_tokens {
                        body.insert("maxTokens".into(), v.clone());
                    }
                    if let Some(v) = temperature {
                        body.insert("temperature".into(), v.clone());
                    }
                    if let Some(v) = top_p {
                        body.insert("topP".into(), v.clone());
                    }
                    if let Some(v) = frequency_penalty {
                        body.insert("frequencyPenalty".into(), v.clone());
                    }
                    if let Some(v) = presence_penalty {
                        body.insert("presencePenalty".into(), v.clone());
                    }
                    Value::Object(body)
                }
                InvokeVendor::Cohere => {
                    let prompt = messages_to_prompt(messages);
                    let mut body = Map::new();
                    body.insert("prompt".into(), Value::String(prompt));
                    if let Some(v) = max_tokens {
                        body.insert("max_tokens".into(), v.clone());
                    }
                    if let Some(v) = temperature {
                        body.insert("temperature".into(), v.clone());
                    }
                    if let Some(v) = top_p {
                        body.insert("p".into(), v.clone());
                    }
                    if let Some(v) = top_k {
                        body.insert("k".into(), v.clone());
                    }
                    Value::Object(body)
                }
                InvokeVendor::DeepSeek => {
                    let prompt = messages_to_prompt(messages);
                    let mut body = Map::new();
                    body.insert("prompt".into(), Value::String(prompt));
                    if let Some(v) = max_tokens {
                        body.insert("max_tokens".into(), v.clone());
                    }
                    if let Some(v) = temperature {
                        body.insert("temperature".into(), v.clone());
                    }
                    if let Some(v) = top_p {
                        body.insert("top_p".into(), v.clone());
                    }
                    Value::Object(body)
                }
                InvokeVendor::Qwen => {
                    let prompt = messages_to_qwen_prompt(messages);
                    let mut body = Map::new();
                    body.insert("prompt".into(), Value::String(prompt));
                    if let Some(v) = max_tokens {
                        body.insert("max_tokens".into(), v.clone());
                    }
                    if let Some(v) = temperature {
                        body.insert("temperature".into(), v.clone());
                    }
                    if let Some(v) = top_p {
                        body.insert("top_p".into(), v.clone());
                    }
                    Value::Object(body)
                }
                InvokeVendor::Unknown => passthrough_invoke_request(request, model),
            };
            // Bedrock InvokeModel honours a `performanceConfig` hint for
            // latency optimisation on supported model families. Inject it
            // after the vendor body is built so it applies to both the
            // vendor-specific and the passthrough paths. (Anthropic-on-
            // Bedrock uses its own body shape where this field is not
            // applicable.)
            if let Some(pc) = Self::performance_config_value(deployment) {
                if let Some(map) = out.as_object_mut() {
                    map.insert("performanceConfig".to_string(), pc);
                }
            }
            Ok(out)
        }
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        // Converse responses carry `output.message.content[]` — a uniquely
        // Converse shape. Dispatch on it before falling back to the legacy
        // Anthropic / vendor paths so a Converse-enabled deployment always
        // decodes correctly even when `transform_response` lacks the
        // deployment context.
        if converse::looks_like_converse_response(body) {
            return converse::transform_response_converse(body, model);
        }

        if Self::is_anthropic_model(model)
            || body.get("type").and_then(Value::as_str) == Some("message")
        {
            // Bedrock returns the native Anthropic Messages response shape.
            let mut out = AnthropicProvider.transform_response(body, model)?;
            // The Anthropic transform discards `thinking` content blocks. For
            // Bedrock-hosted reasoning models (Claude extended thinking),
            // surface them as `reasoning_content` (concatenated text, the
            // OpenAI convention) and `thinking_blocks` (raw blocks, for
            // round-trip into subsequent tool messages).
            if let Some((reasoning, blocks)) = extract_thinking(&body["content"]) {
                if let Some(message) = out
                    .get_mut("choices")
                    .and_then(|c| c.as_array_mut())
                    .and_then(|arr| arr.first_mut())
                    .and_then(|choice| choice.get_mut("message"))
                    .and_then(|m| m.as_object_mut())
                {
                    if !reasoning.is_empty() {
                        message.insert("reasoning_content".to_string(), json!(reasoning));
                    }
                    message.insert("thinking_blocks".to_string(), json!(blocks));
                }
            }
            Ok(out)
        } else {
            Ok(transform_vendor_response(body, model))
        }
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        // Bedrock's `invoke-with-response-stream` returns binary
        // `application/vnd.amazon.eventstream` frames. Each frame's payload is
        // a JSON object in the native provider's event shape. The stream pump
        // feeds the provider complete event bodies (after the framing layer is
        // stripped by the host) — but it also passes through any line that
        // looks like SSE. We accept either:
        //
        //   1. a bare JSON object (e.g. `{"type":"message_start",...}`), or
        //   2. an SSE-style `data: {...}` line.
        //
        // For Anthropic-on-Bedrock we reuse the Anthropic chunk transform; for
        // other vendors we map the vendor-native JSON into an OpenAI chunk.
        let trimmed = chunk.trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return Ok(None);
        }
        let payload = extract_payload(trimmed);
        if payload.is_empty() {
            return Ok(None);
        }
        let value: Value = match serde_json::from_str(&payload) {
            Ok(v) => v,
            Err(_) => {
                // Not parseable JSON — likely an eventstream framing artifact
                // we do not recognise. Skip rather than abort the stream.
                return Ok(None);
            }
        };
        if let Some(error) = bedrock_stream_error(&value) {
            return Err(error);
        }

        // Converse stream events have characteristic top-level keys
        // (`messageStart`, `contentBlockDelta`, `stopReason`, …). Dispatch
        // to the Converse SSE parser before falling through to the legacy
        // Anthropic / vendor eventstream decoders. We re-pass the original
        // chunk string (the converse parser owns its own SSE/JSON handling)
        // so the trait contract for this method stays "string in, chunk out".
        if converse::looks_like_converse_event(&value) {
            return converse::transform_stream_chunk_converse(chunk);
        }

        if value.get("type").is_some() || value.get("message").is_some() {
            // Looks like an Anthropic event — reuse the Anthropic transform.
            // Wrap as an SSE `data:` line because the Anthropic transform
            // expects to extract the data line first.
            let wrapped = format!("data: {}", value);
            AnthropicProvider.transform_stream_chunk(&wrapped)
        } else {
            Ok(transform_vendor_stream_chunk(&value))
        }
    }

    /// Stateful streaming override: threads a per-stream reverse tool-name map
    /// (rebuilt once from the original request) into the Converse parser so
    /// tool-name restoration survives tokio task migration across worker
    /// threads. A `thread_local!` (or worse, a process-wide cell) cannot — a
    /// spawned stream task may be rescheduled onto a different worker at every
    /// `.await`, orphaning any thread-local it wrote. The host owns the
    /// `state` slot and passes it through every chunk.
    fn transform_stream_chunk_stateful(
        &self,
        chunk: &str,
        state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
        original_request: &Value,
    ) -> Result<Option<Value>> {
        // Fast path: if this isn't a Converse event, it is either an
        // Anthropic Messages event (Claude on InvokeModel) or a vendor
        // eventstream payload. Vendor events are stateless; Anthropic events
        // must go through the STATEFUL Anthropic transform — the stateless
        // `transform_stream_chunk` leans on `AnthropicProvider`'s
        // `thread_local!` cells, which are orphaned whenever tokio migrates
        // this stream task across worker threads (losing `id`/`model`/
        // `created` injection, tool-name restoration, and tool-block
        // correlation). Mirrors the vertex_ai delegation.
        let trimmed = chunk.trim();
        if !trimmed.is_empty() && trimmed != "[DONE]" {
            let payload = extract_payload(trimmed);
            if !payload.is_empty() {
                if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                    if let Some(error) = bedrock_stream_error(&value) {
                        return Err(error);
                    }
                    if !converse::looks_like_converse_event(&value) {
                        // Same anthropic-vs-vendor dispatch as the stateless
                        // `transform_stream_chunk` above.
                        let is_anthropic_event =
                            value.get("type").is_some() || value.get("message").is_some();
                        // Defensive: if a stray earlier chunk already put a
                        // `BedrockStreamState` in the host slot, the Anthropic
                        // downcast would panic — fall back to the stateless
                        // parser instead (valid chunks, no state carry).
                        let slot_is_foreign = state
                            .as_deref()
                            .map(|s| {
                                s.downcast_ref::<crate::anthropic::AnthropicStreamState>()
                                    .is_none()
                            })
                            .unwrap_or(false);
                        if is_anthropic_event && !slot_is_foreign {
                            return AnthropicProvider.transform_stream_chunk_stateful(
                                &format!("data: {value}"),
                                state,
                                original_request,
                            );
                        }
                        return self.transform_stream_chunk(chunk);
                    }
                }
            }
        }

        // Converse path: lazily build the per-stream reverse map once.
        let s = state.get_or_insert_with(|| {
            Box::new(BedrockStreamState {
                reverse_map: converse::reverse_map_from_request(original_request),
                converse: converse::ConverseStreamState::new(
                    original_request
                        .get("model")
                        .and_then(Value::as_str)
                        .unwrap_or("bedrock"),
                    converse::uses_synthetic_json_tool(original_request),
                ),
            })
        });
        // If some foreign code path wrote a different `Any` into the
        // host-owned state slot, fall back to the stateless parser rather
        // than panicking. The stateless path still emits valid OpenAI-shaped
        // chunks — it just skips tool-name restoration.
        let Some(s) = s.downcast_mut::<BedrockStreamState>() else {
            return self.transform_stream_chunk(chunk);
        };
        converse::transform_stream_chunk_converse_with_state(chunk, &s.reverse_map, &mut s.converse)
    }

    /// Stateful response override: rebuild the reverse tool-name map from the
    /// original request and seed the per-thread cell right here on the calling
    /// thread. There is no `.await` between this seed and the read inside
    /// [`Self::transform_response`] (both run synchronously on this thread),
    /// so the cell is safe for this one-shot non-streaming call — unlike the
    /// streaming path, which uses [`BedrockStreamState`] instead.
    fn transform_response_with_request(
        &self,
        body: &Value,
        model: &str,
        original_request: &Value,
    ) -> Result<Value> {
        if !converse::looks_like_converse_response(body)
            && (Self::is_anthropic_model(model)
                || body.get("type").and_then(Value::as_str) == Some("message"))
        {
            return AnthropicProvider.transform_response_with_request(
                body,
                model,
                original_request,
            );
        }
        converse::seed_tool_name_map(converse::reverse_map_from_request(original_request));
        if converse::looks_like_converse_response(body) {
            return converse::transform_response_converse_with_request(
                body,
                model,
                original_request,
            );
        }
        self.transform_response(body, model)
    }
}

/// Per-stream state for the Bedrock Converse streaming path. Carries the
/// reverse tool-name map (sanitized → original) rebuilt once from the
/// original request, so tool-call names can be restored on the way back to
/// the client without depending on a process-wide or thread-local cell that
/// would be clobbered by concurrent requests or orphaned by tokio task
/// migration.
pub struct BedrockStreamState {
    reverse_map: HashMap<String, String>,
    converse: converse::ConverseStreamState,
}

impl Provider for BedrockProvider {
    fn id(&self) -> ProviderId {
        "bedrock"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(endpoint, oxllm_core::Endpoint::ChatCompletions)
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

/// Strip a cross-region inference prefix from a Bedrock model id.
///
/// Handles geo prefixes (`us.`, `eu.`, `apac.`, `global.`) and
/// region-specific ids (`us-east-1.`, `eu-west-1.`). `us-east-1.` is not
/// treated as `us.` — that would leave `east-1.anthropic…` and miss the
/// Anthropic detector.
pub(super) fn strip_cross_region_prefix(model: &str) -> &str {
    for prefix in ["us.", "eu.", "apac.", "global."] {
        if let Some(rest) = model.strip_prefix(prefix) {
            if !rest.starts_with('-') {
                return rest;
            }
        }
    }
    if let Some(dot) = model.find('.') {
        let head = &model[..dot];
        if head.contains('-') && head.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return &model[dot + 1..];
        }
    }
    model
}

/// Strip LiteLLM-style throughput and context-window suffixes from a
/// Bedrock model id. Removes trailing `:Nk` throughput tiers (e.g.
/// `:51k`, `:18k`) and a single trailing `[...]` context-window tag (e.g.
/// `[1m]`, `[200k]`). The base model id — including its `:version` tag —
/// is always preserved.
pub fn strip_model_suffixes(model: &str) -> String {
    let mut m = model.to_string();
    // Strip throughput suffixes like `:51k`, `:18k`. Loop so a stacked
    // suffix can be removed, but stop at the first non-throughput `:`
    // (typically the version tag) so a real version is never stripped.
    while let Some(pos) = m.rfind(':') {
        let suffix = &m[pos + 1..];
        if suffix.ends_with('k') && suffix[..suffix.len() - 1].parse::<u32>().is_ok() {
            m.truncate(pos);
        } else {
            break;
        }
    }
    // Strip a trailing context-window bracket like `[1m]` or `[200k]`.
    if let Some(start) = m.rfind('[') {
        if m.ends_with(']') {
            m.truncate(start);
        }
    }
    m
}

/// Detect the underlying vendor of a Bedrock model id. Strips the `bedrock/`
/// routing prefix and the cross-region prefix (`us.` / `eu.` / `apac.`)
/// before matching. Foundation-model and system inference-profile ARNs expose
/// their model id; application/custom/provisioned ARNs remain opaque. Amazon
/// models are split into `Nova`, `Titan`, and the generic `Amazon` fallback.
/// Models with no recognised vendor prefix yield
/// [`BedrockVendor::Unknown`].
pub fn detect_bedrock_provider(model: &str) -> BedrockVendor {
    let Some(base) = model_id_for_family(model) else {
        return BedrockVendor::Unknown;
    };
    if base.starts_with("anthropic.") {
        return BedrockVendor::Anthropic;
    }
    if base.starts_with("amazon.") {
        if base.starts_with("amazon.nova") {
            return BedrockVendor::Nova;
        }
        if base.contains("titan") {
            return BedrockVendor::Titan;
        }
        return BedrockVendor::Amazon;
    }
    if base.starts_with("meta.") {
        return BedrockVendor::Llama;
    }
    if base.starts_with("mistral.") {
        return BedrockVendor::Mistral;
    }
    if base.starts_with("ai21.") {
        return BedrockVendor::AI21;
    }
    if base.starts_with("cohere.") {
        return BedrockVendor::Cohere;
    }
    if base.contains("deepseek") {
        return BedrockVendor::DeepSeek;
    }
    if base.contains("qwen") {
        return BedrockVendor::Qwen;
    }
    BedrockVendor::Unknown
}

/// Return only model identifiers whose resource shape exposes a known family.
/// An application profile's chosen name is not evidence of its source model.
pub(super) fn model_id_for_family(model: &str) -> Option<&str> {
    let model = model.strip_prefix("bedrock/").unwrap_or(model);
    let model = if model.starts_with("arn:") {
        let mut parts = model.splitn(6, ':');
        parts.next()?;
        parts.next()?;
        if parts.next()? != "bedrock" {
            return None;
        }
        parts.next()?;
        parts.next()?;
        let resource = parts.next()?;
        resource
            .strip_prefix("foundation-model/")
            .or_else(|| resource.strip_prefix("inference-profile/"))?
    } else {
        model
    };
    Some(strip_cross_region_prefix(model))
}

/// Decide whether a request should use the Converse API. The explicit
/// deployment flag (`use_converse_api` / `use_converse`) always wins.
/// When the flag is unset, auto-route to Converse for:
///
/// - **Application inference profile ARNs** — the model name contains
///   `:application-inference-profile/`. InvokeModel cannot resolve a
///   provider from the opaque ARN, but Converse is provider-agnostic.
/// - **Models in `MUST_USE_CONVERSE`** — InvokeModel does not support
///   them (e.g. `ai21.jamba`, `qwen.qwen3`).
pub fn should_use_converse(model: &str, explicit: bool) -> bool {
    if explicit {
        return true;
    }
    if model.contains(":application-inference-profile/") {
        return true;
    }
    let Some(base) = model_id_for_family(model) else {
        return false;
    };
    let base = strip_model_suffixes(base);
    let base = strip_cross_region_prefix(&base);
    MUST_USE_CONVERSE.iter().any(|m| base.starts_with(m))
}

/// Extract the JSON payload from a chunk that may be either a bare JSON
/// object or an SSE-style `data:` line. Returns the empty string when no
/// payload can be located.
fn extract_payload(raw: &str) -> String {
    crate::util::extract_sse_data(raw).unwrap_or_else(|| raw.trim().to_string())
}

/// SDK-style named exceptions retain their status. The host currently strips
/// eventstream headers, so wire exceptions may arrive with only `message`:
/// fail the stream in that case too, without guessing the lost exception type.
fn bedrock_stream_error(value: &Value) -> Option<oxllm_core::Error> {
    for (name, status) in [
        ("internalServerException", 500),
        ("modelStreamErrorException", 424),
        ("modelTimeoutException", 408),
        ("throttlingException", 429),
        ("serviceUnavailableException", 503),
        ("validationException", 400),
    ] {
        if let Some(error) = value.get(name) {
            return Some(oxllm_core::Error::Upstream {
                status,
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or(name)
                    .to_string(),
            });
        }
    }
    if value.get("type").is_none() && value.get("choices").is_none() {
        if let Some(message) = value.get("message").and_then(Value::as_str) {
            return Some(oxllm_core::Error::Upstream {
                status: if value.get("originalStatusCode").is_some() {
                    424
                } else {
                    500
                },
                message: message.to_string(),
            });
        }
    }
    None
}

/// Pull `thinking` / `redacted_thinking` blocks out of an Anthropic Messages
/// `content` array. Returns `(reasoning_text, raw_blocks)`:
///
/// - `reasoning_text` concatenates the `thinking` field of every visible
///   thinking block (redacted blocks contribute no text).
/// - `raw_blocks` preserves the original block objects (including
///   `signature`) so callers can replay them in a subsequent tool-call turn.
///
/// Returns `None` when the content has no thinking blocks at all.
fn extract_thinking(content: &Value) -> Option<(String, Vec<Value>)> {
    let arr = content.as_array()?;
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut blocks: Vec<Value> = Vec::new();
    for block in arr {
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => {
                if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                    reasoning_parts.push(t.to_string());
                }
                blocks.push(block.clone());
            }
            Some("redacted_thinking") => {
                blocks.push(block.clone());
            }
            _ => {}
        }
    }
    if blocks.is_empty() {
        None
    } else {
        Some((reasoning_parts.join(""), blocks))
    }
}

/// Walk every message in an Anthropic Messages body and rewrite OpenAI-style
/// `image_url` content blocks (with `data:` URLs) into the native Anthropic
/// `{"type":"image","source":{"type":"base64",...}}` shape that Bedrock-hosted
/// Claude requires. HTTP(S) URLs are left in place — Bedrock does not fetch
/// remote URLs, so the server layer is responsible for inlining them into
/// `data:` URLs before this transform runs (see
/// `oxllm_server::pre_call_checks::image_inlining`).
///
/// Because the Bedrock path delegates to the Anthropic transform first, the
/// Anthropic transform may have already converted HTTP(S) `image_url` blocks
/// into `{"type":"image","source":{"type":"url",...}}` blocks (which Bedrock
/// does NOT accept). We revert those back to OpenAI `image_url` blocks so the
/// host pipeline can inline or reject them, matching the pre-existing
/// contract.
fn convert_image_url_blocks(body: &mut Value) {
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(m) => m,
        None => return,
    };
    for msg in messages.iter_mut() {
        let content = match msg.get_mut("content").and_then(|c| c.as_array_mut()) {
            Some(c) => c,
            None => continue,
        };
        for block in content.iter_mut() {
            // Revert URL-source image blocks (produced by the Anthropic
            // transform's HTTP URL conversion) back to OpenAI image_url
            // blocks — Bedrock cannot fetch remote URLs.
            if block.get("type").and_then(Value::as_str) == Some("image")
                && block
                    .get("source")
                    .and_then(|s| s.get("type"))
                    .and_then(Value::as_str)
                    == Some("url")
            {
                let url = block
                    .get("source")
                    .and_then(|s| s.get("url"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                *block = json!({"type": "image_url", "image_url": {"url": url}});
                continue;
            }
            if block.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            let url = block
                .get("image_url")
                .and_then(|iu| iu.get("url"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if let Some(converted) = data_url_to_anthropic_image(url) {
                *block = converted;
            }
        }
    }
}

/// Map a non-Anthropic Bedrock response body into the OpenAI Chat Completion
/// shape. Recognises the vendor-native envelopes each InvokeModel vendor
/// returns (Meta Llama `generation`, Mistral `outputs[].text`, Amazon Titan
/// `results[].outputText`, AI21 `completions[0].data.text`, …); vendor-
/// specific post-processing then runs (DeepSeek `<think>` extraction, Qwen
/// ChatML token stripping). For any unrecognised shape we fall back to the
/// body verbatim so newer vendors still surface something useful rather than
/// being silently mangled.
fn transform_vendor_response(body: &Value, model: &str) -> Value {
    let vendor = detect_invoke_vendor(model);
    let Some(raw_text) = extract_vendor_text(body) else {
        // Unrecognised shape: pass through unchanged so callers see the raw
        // vendor body instead of a half-translation.
        return body.clone();
    };
    // Apply vendor-specific text post-processing.
    let (reasoning, content) = match vendor {
        InvokeVendor::DeepSeek => split_deepseek_reasoning(&raw_text),
        InvokeVendor::Qwen => (None, strip_qwen_tokens(&raw_text)),
        _ => (None, raw_text),
    };
    let prompt = body.get("prompt_token_count").and_then(|v| v.as_u64());
    let completion = body
        .get("generation_token_count")
        .or_else(|| body.get("generated_token_count"))
        .and_then(|v| v.as_u64());
    let usage = match (prompt, completion) {
        (Some(p), Some(c)) => json!({
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p.saturating_add(c),
        }),
        _ => Value::Null,
    };
    let mut message = Map::new();
    message.insert("role".to_string(), json!("assistant"));
    message.insert("content".to_string(), json!(content));
    if let Some(r) = reasoning {
        message.insert("reasoning_content".to_string(), json!(r));
    }
    let mut out = Map::new();
    out.insert("object".to_string(), json!("chat.completion"));
    out.insert("id".to_string(), json!(format!("chatcmpl-bedrock-{model}")));
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
        "message": Value::Object(message),
        "finish_reason": extract_vendor_finish_reason(body).unwrap_or_else(|| json!("stop")),
        }]),
    );
    if !usage.is_null() {
        out.insert("usage".to_string(), usage);
    }
    Value::Object(out)
}

/// Map a non-Anthropic Bedrock streaming chunk into an OpenAI-shaped delta
/// chunk. Meta Llama / Mistral streaming emit
/// `{"outputs":[{"text":"..."}]}`; we treat that as the canonical shape and
/// fall back to a handful of common text fields for any other vendor.
///
/// When the chunk carries tool-call data (Meta `tool_calls`, Mistral
/// `outputs[].tool_calls`, generic top-level `tool_calls`), we instead emit
/// an OpenAI-shaped `delta.tool_calls` so non-Anthropic Bedrock models can
/// stream tool calls back to the client.
fn transform_vendor_stream_chunk(value: &Value) -> Option<Value> {
    // OpenAI-compatible models on Bedrock (DeepSeek V3.2, etc.) already
    // emit OpenAI-shaped streaming chunks. Pass them through unchanged.
    if value.get("choices").is_some() {
        return Some(value.clone());
    }
    // Bedrock may wrap model output in a `bytes` field (JSON-encoded string).
    if let Some(b) = value.get("bytes").and_then(Value::as_str) {
        if let Ok(inner) = serde_json::from_str::<Value>(b) {
            if inner.get("choices").is_some() {
                return Some(inner);
            }
            // Recurse with the unwrapped payload for vendor-format models.
            return transform_vendor_stream_chunk(&inner);
        }
    }
    if let Some(tool_calls) = extract_vendor_stream_tool_calls(value) {
        if !tool_calls.is_empty() {
            return Some(json!({
            "object": "chat.completion.chunk",
            "choices": [{
            "index": 0,
            "delta": {"tool_calls": tool_calls},
            "finish_reason": null,
            }],
            }));
        }
    }
    let text = extract_vendor_text(value)?;
    if text.is_empty() {
        return None;
    }
    Some(json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {"content": text},
    "finish_reason": null,
    }],
    }))
}

/// Extract tool-call deltas from any of the streaming shapes Bedrock's
/// non-Anthropic vendors emit. Looks for:
///
/// - top-level `tool_calls: [...]` (Meta Llama 3.1+)
/// - `outputs[].tool_calls: [...]` (Mistral)
/// - `generations[].tool_calls: [...]` (generic)
///
/// Each entry is normalised to the OpenAI streaming tool-call delta shape:
/// `{index, id, type:"function", function:{name, arguments}}`. Field names
/// inside the vendor entries vary (`name`/`function.name`,
/// `arguments`/`input`/`parameters`); we try each in turn.
fn extract_vendor_stream_tool_calls(value: &Value) -> Option<Vec<Value>> {
    let raw: &Vec<Value> = value
        .get("tool_calls")
        .and_then(Value::as_array)
        .or_else(|| {
            value
                .get("outputs")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(|elem| elem.get("tool_calls"))
                .and_then(Value::as_array)
        })
        .or_else(|| {
            value
                .get("generations")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(|elem| elem.get("tool_calls"))
                .and_then(Value::as_array)
        })?;
    let mut out: Vec<Value> = Vec::with_capacity(raw.len());
    for (idx, tc) in raw.iter().enumerate() {
        let id = tc
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("call_bedrock")
            .to_string();
        let name = tc
            .get("name")
            .and_then(Value::as_str)
            .map(String::from)
            .or_else(|| {
                tc.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .map(String::from)
            })
            .unwrap_or_default();
        let arguments = tc
            .get("arguments")
            .and_then(Value::as_str)
            .map(String::from)
            .or_else(|| {
                // Some vendors ship structured args (`input`/`parameters`)
                // instead of a pre-serialised string. Serialise them so the
                // OpenAI-shaped delta always carries a string `arguments`.
                tc.get("input")
                    .or_else(|| tc.get("parameters"))
                    .or_else(|| tc.get("function").and_then(|f| f.get("arguments")))
                    .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
            })
            .unwrap_or_default();
        out.push(json!({
            "index": idx,
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }));
    }
    Some(out)
}

/// Pull a single text string out of any of the vendor-native response shapes
/// Bedrock serves. Recognises:
///
/// - Meta Llama / DeepSeek: `{"generation": "..."}`
/// - Mistral: `{"outputs":[{"text":"..."}]}`
/// - Titan: `{"results":[{"outputText":"..."}]}`
/// - AI21: `{"completions":[{"data":{"text":"..."}}]}`
/// - Generic: first non-empty of `generation` / `outputText` / `text` /
///   `completion` (top-level), or the `text` field of any element in
///   `outputs`/`results`/`generations`.
fn extract_vendor_text(body: &Value) -> Option<String> {
    // Meta Llama / DeepSeek non-streaming response shape.
    if let Some(s) = body.get("generation").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    // AI21 non-streaming response shape.
    if let Some(s) = body
        .get("completions")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("data"))
        .and_then(|d| d.get("text"))
        .and_then(Value::as_str)
    {
        return Some(s.to_string());
    }
    // Generic top-level text-bearing fields (Titan uses `outputText`,
    // some vendors use `text` / `completion`).
    for key in ["outputText", "text", "completion"] {
        if let Some(s) = body.get(key).and_then(Value::as_str) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    // Array-of-objects shapes (Mistral `outputs`, Titan `results`,
    // generic `generations`). Each element may carry `text` or `outputText`.
    for arr_key in ["outputs", "results", "generations"] {
        if let Some(arr) = body.get(arr_key).and_then(Value::as_array) {
            let mut parts: Vec<String> = Vec::new();
            for elem in arr {
                for text_key in ["text", "outputText", "completion", "generation"] {
                    if let Some(s) = elem.get(text_key).and_then(Value::as_str) {
                        if !s.is_empty() {
                            parts.push(s.to_string());
                        }
                        break;
                    }
                }
            }
            if !parts.is_empty() {
                return Some(parts.join(""));
            }
        }
    }
    None
}

/// Best-effort extraction of a vendor finish reason, mapped onto the OpenAI
/// vocabulary. Returns `None` when no reason field is present.
fn extract_vendor_finish_reason(body: &Value) -> Option<Value> {
    let reason = body
        .get("stop_reason")
        .and_then(Value::as_str)
        .or_else(|| body.get("finish_reason").and_then(Value::as_str))
        .or_else(|| {
            body.get("outputs")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(|elem| elem.get("stop_reason"))
                .and_then(Value::as_str)
        })?;
    let mapped = match reason {
        "stop" | "end_turn" | "STOP" => "stop",
        "length" | "max_tokens" | "MAX_TOKENS" => "length",
        "tool_calls" | "TOOL_CALLS" => "tool_calls",
        "content_filter" | "FILTERED" => "content_filter",
        _ => "stop",
    };
    Some(json!(mapped))
}

/// Categorise a Bedrock model id by its underlying vendor for InvokeModel
/// request/response shaping. Non-Anthropic Bedrock models expose a uniform
/// `InvokeModel` surface but each vendor accepts its own native request body
/// shape; this enum dispatches the transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvokeVendor {
    Titan,
    Llama,
    Mistral,
    AI21,
    Cohere,
    DeepSeek,
    Qwen,
    Unknown,
}

/// Detect the InvokeModel vendor from a Bedrock model id. Looks for the
/// standard Bedrock vendor prefixes (`amazon.`, `meta.`, `mistral.`, …) in
/// any path segment, so cross-region inference prefixes (`us.`, `eu.`,
/// `apac.`) are handled transparently. Amazon Nova models have their own
/// (Converse-based) handler upstream and are classified `Unknown` here so
/// the InvokeModel path leaves them on the passthrough fallback. Qwen
/// imported models may carry no vendor prefix at all and are recognised by
/// a substring search.
fn detect_invoke_vendor(model: &str) -> InvokeVendor {
    let lower = model.to_ascii_lowercase();
    // Nova has its own dedicated handler — leave it on the passthrough
    // fallback for InvokeModel so it isn't mangled into Titan's shape.
    if lower.contains("nova") {
        return InvokeVendor::Unknown;
    }
    for seg in lower.split('.') {
        match seg {
            "amazon" => return InvokeVendor::Titan,
            "meta" => return InvokeVendor::Llama,
            "mistral" => return InvokeVendor::Mistral,
            "ai21" => return InvokeVendor::AI21,
            "cohere" => return InvokeVendor::Cohere,
            "deepseek" => return InvokeVendor::DeepSeek,
            _ => {}
        }
    }
    // Qwen imported models on Bedrock often have no `qwen.` prefix.
    if lower.contains("qwen") {
        return InvokeVendor::Qwen;
    }
    InvokeVendor::Unknown
}

/// Legacy InvokeModel prompt templates can express text turns, but not a
/// tool protocol or multimodal blocks. Reject those requests instead of
/// silently sending a truncated prompt; Converse is the chat/tool surface.
fn normalize_invoke_prompt_messages(request: &Value, messages: &[Value]) -> Result<Vec<Value>> {
    if request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
        || request
            .get("functions")
            .and_then(Value::as_array)
            .is_some_and(|tools| !tools.is_empty())
        || request
            .get("response_format")
            .and_then(|format| format.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|kind| kind != "text")
    {
        return Err(oxllm_core::Error::BadRequest(
            "this Bedrock InvokeModel prompt adapter cannot translate tools or structured outputs; configure use_converse_api for a supporting model".into(),
        ));
    }
    messages.iter().map(|message| {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("user");
        if !matches!(role, "system" | "developer" | "user" | "assistant")
            || message.get("tool_calls").and_then(Value::as_array).is_some_and(|calls| !calls.is_empty())
            || message.get("function_call").is_some_and(|call| !call.is_null())
            || message.get("thinking_blocks").and_then(Value::as_array).is_some_and(|blocks| !blocks.is_empty())
            || message.get("reasoning_content").and_then(Value::as_str).is_some_and(|text| !text.is_empty())
        {
            return Err(oxllm_core::Error::BadRequest(
                "this Bedrock InvokeModel prompt adapter supports text-only system/developer/user/assistant turns; tool and reasoning history requires a supporting Converse model".into(),
            ));
        }
        let text = match message.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(blocks)) => {
                let mut text = String::new();
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) != Some("text") {
                        return Err(oxllm_core::Error::BadRequest(
                            "this Bedrock InvokeModel prompt adapter cannot translate non-text content; use a multimodal Converse model".into(),
                        ));
                    }
                    text.push_str(block.get("text").and_then(Value::as_str).ok_or_else(|| {
                        oxllm_core::Error::BadRequest("text content blocks require a string text field".into())
                    })?);
                }
                text
            }
            None | Some(Value::Null) => String::new(),
            _ => return Err(oxllm_core::Error::BadRequest("message content must be text or an array of text blocks".into())),
        };
        Ok(json!({"role": if role == "developer" { "system" } else { role }, "content": text}))
    }).collect()
}

/// Convert OpenAI messages into a flat prompt string for the vendors that
/// take a single `prompt` field on InvokeModel (Llama, Mistral, Titan,
/// AI21, Cohere, DeepSeek). Each vendor accepts its own chat-template
/// format; we use a minimal "concatenate by role" layout that surfaces
/// every message in arrival order. System messages get a blank-line
/// separator; user/assistant messages get a single trailing newline. Qwen
/// uses a dedicated ChatML template (see [`messages_to_qwen_prompt`]).
fn messages_to_prompt(messages: &[Value]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        match role {
            "system" => {
                prompt.push_str(content);
                prompt.push_str("\n\n");
            }
            "user" => {
                prompt.push_str(content);
                prompt.push('\n');
            }
            "assistant" => {
                prompt.push_str(content);
                prompt.push('\n');
            }
            _ => {}
        }
    }
    prompt
}

/// Convert OpenAI messages into the Meta Llama-3 chat template — the
/// format Bedrock's `meta.llama3-*` InvokeModel ids expect in the
/// `prompt` field. Without this template the model echoes the input
/// instead of answering, because the raw text has no role signal it can
/// latch onto.
///
/// Mirrors the official `meta-llama/Meta-Llama-3-8B-Instruct` Jinja
/// template LiteLLM applies via `hf_chat_template` (see LiteLLM's
/// `prompt_factory` bedrock arm). Tokens are concatenated with no
/// inter-message separators — only the `\n\n` between a header and its
/// content — and the prompt is terminated with an empty
/// `<|start_header_id|>assistant<|end_header_id|>\n\n` block to elicit
/// the reply.
///
/// Unsupported roles (e.g. `tool`) are treated as `user`, matching the
/// Llama-3 template's fallthrough.
fn messages_to_llama3_prompt(messages: &[Value]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let header = match role {
            "system" | "user" | "assistant" => role,
            _ => "user",
        };
        prompt.push_str(&format!(
            "<|start_header_id|>{header}<|end_header_id|>\n\n{content}<|eot_id|>"
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Convert OpenAI messages into the Qwen ChatML prompt format. Qwen
/// imported models on Bedrock take a single `prompt` field built from
/// `<|im_start|>{role}\n{content}<|im_end|>` blocks joined by newlines,
/// terminated by an empty `<|im_start|>assistant\n` block to elicit the
/// reply.
fn messages_to_qwen_prompt(messages: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(messages.len() + 1);
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let block = match role {
            "system" | "user" | "assistant" | "tool" => {
                format!("<|im_start|>{role}\n{content}<|im_end|>")
            }
            _ => continue,
        };
        parts.push(block);
    }
    parts.push("<|im_start|>assistant\n".to_string());
    parts.join("\n")
}

/// Split a DeepSeek `<think>...</think>` block out of a generated response.
/// DeepSeek R1 emits the reasoning inside a `<think>` tag, followed by the
/// final answer; the OpenAI-shaped response carries the reasoning as a
/// separate `reasoning_content` field (the convention used by
/// reasoning-aware clients). When no `<think>...</think>` block is
/// present, the whole text becomes `content` and `reasoning_content` is
/// `None`.
fn split_deepseek_reasoning(text: &str) -> (Option<String>, String) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let open_pos = match text.find(OPEN) {
        Some(p) => p,
        None => return (None, text.to_string()),
    };
    let after_open = &text[open_pos + OPEN.len()..];
    let close_pos = match after_open.find(CLOSE) {
        Some(p) => p,
        None => return (None, text.to_string()),
    };
    let reasoning = after_open[..close_pos].to_string();
    let rest = after_open[close_pos + CLOSE.len()..].to_string();
    // Defensive: preserve any text that appeared before `<think>` (rare in
    // practice — the tag is usually at offset 0).
    let prefix = &text[..open_pos];
    let content = format!("{prefix}{rest}");
    (Some(reasoning), content)
}

/// Strip the Qwen ChatML framing tokens a Qwen-vendor response echoes back
/// at the boundaries: the leading `<|im_start|>assistant\n` prefix and the
/// trailing `<|im_end|>` suffix. Either or both may be absent depending on
/// the imported-model configuration; only present tokens are stripped.
fn strip_qwen_tokens(text: &str) -> String {
    let mut s = text.to_string();
    if let Some(rest) = s.strip_prefix("<|im_start|>assistant\n") {
        s = rest.to_string();
    }
    if let Some(rest) = s.strip_suffix("<|im_end|>") {
        s = rest.to_string();
    }
    s
}

/// Fallback body builder for vendors we don't model explicitly: forward the
/// OpenAI-shaped request body verbatim, swapping in the resolved Bedrock
/// model id. Bedrock accepts the OpenAI Chat Completions shape directly
/// for some imported OpenAI-compatible models (e.g. DeepSeek V3.2), so
/// leaving the body intact keeps those calls working.
fn passthrough_invoke_request(request: &Value, model: &str) -> Value {
    let mut out = Map::new();
    if let Value::Object(obj) = request {
        for (k, v) in obj {
            if k == "model" {
                continue;
            }
            out.insert(k.clone(), v.clone());
        }
    }
    out.insert("model".to_string(), Value::String(model.to_string()));
    Value::Object(out)
}

/// Encode `payload` as a single AWS eventstream frame — the inverse of
/// [`decode_eventstream_frame`].
///
/// Realtime frames carry no headers, so the produced frame is:
///
/// ```text
/// [4 bytes total_len, big-endian u32]   ← full frame size incl. itself + CRC
/// [4 bytes headers_len = 0, big-endian]
/// [4 bytes prelude CRC32, big-endian]   ← CRC32 of the first 8 bytes
/// [payload bytes]
/// [4 bytes CRC32, big-endian]           ← CRC32 of total_len..=payload
/// ```
///
/// The CRC32 is the standard reflected variant (polynomial `0xEDB88320`,
/// init `0xFFFFFFFF`, final XOR `0xFFFFFFFF`) — identical to zlib and to what
/// the AWS eventstream wire format expects.
pub fn encode_eventstream_frame(payload: &[u8]) -> Vec<u8> {
    const PREFIX_LEN: usize = 12;
    const CRC_LEN: usize = 4;
    let total_len = PREFIX_LEN + payload.len() + CRC_LEN;
    let mut buf = Vec::with_capacity(total_len);
    buf.extend_from_slice(&(total_len as u32).to_be_bytes());
    buf.extend_from_slice(&0u32.to_be_bytes()); // headers_len = 0
    let prelude_crc = crc32(&buf);
    buf.extend_from_slice(&prelude_crc.to_be_bytes());
    buf.extend_from_slice(payload);
    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_be_bytes());
    buf
}

/// Standard CRC32 (reflected polynomial `0xEDB88320`, init `0xFFFFFFFF`,
/// final XOR `0xFFFFFFFF`) — identical to zlib / `crc32fast`. AWS eventstream
/// frames carry this CRC over everything from `total_len` through the end of
/// the payload.
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, entry) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 == 1 {
                    0xEDB88320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *entry = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Decode the first complete AWS eventstream frame from `buf`.
///
/// Bedrock's `invoke-with-response-stream` returns binary
/// `application/vnd.amazon.eventstream` frames, not SSE text. Each frame is:
///
/// ```text
/// [4 bytes total_len, big-endian u32]
/// [4 bytes headers_len, big-endian u32]
/// [4 bytes prelude CRC32]
/// [headers_len bytes of typed headers]
/// [payload bytes — a JSON object in the vendor's native event shape]
/// [4 bytes CRC32]
/// ```
///
/// `total_len` is the full frame size including the prefix and the trailing
/// CRC. The payload sits at offset `12 + headers_len` and ends `4` bytes
/// before the end of the frame (the CRC).
///
/// Returns `Some((bytes_consumed, payload_slice))` for the first complete
/// frame, or `None` when the buffer does not yet hold a full frame (the
/// caller should buffer more bytes and retry). Length-field malformations
/// (total_len too small, headers_len overflowing the frame) also yield
/// `None` since the frame boundary itself is untrustworthy and the caller
/// cannot safely advance.
///
/// A **complete** frame whose trailing CRC32 does not match is a different
/// case: the frame boundary is known (`total_len` is present and the buffer
/// holds the whole frame), but the bytes were corrupted in transit (e.g. by
/// a proxy). Returning `None` here would make the caller buffer and retry
/// forever — a complete corrupt frame never becomes valid. Instead we
/// advance past it by returning `Some((total_len, &[]))`: the empty payload
/// is skipped by the caller, the corrupt frame is dropped from the buffer,
/// and the stream continues. The drop is logged at `warn` so it is
/// diagnosable.
///
/// CRC validation mirrors the encoder (`encode_eventstream_frame`), which
/// writes a CRC over `total_len..=payload`.
pub fn decode_eventstream_frame(buf: &[u8]) -> Option<(usize, &[u8])> {
    const PREFIX_LEN: usize = 12;
    const CRC_LEN: usize = 4;
    const MIN_FRAME_LEN: usize = PREFIX_LEN + CRC_LEN;
    if buf.len() < PREFIX_LEN {
        return None;
    }
    let prelude_crc = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
    if crc32(&buf[..8]) != prelude_crc {
        // Length fields are untrusted until the prelude checksum validates.
        return None;
    }
    let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if total_len < MIN_FRAME_LEN || buf.len() < total_len {
        return None;
    }
    let headers_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    if headers_len > total_len - MIN_FRAME_LEN {
        return None;
    }
    let payload_start = PREFIX_LEN + headers_len;
    let payload_end = total_len - CRC_LEN;
    // Integrity check: the trailing 4 bytes are a CRC32 over
    // `total_len..=payload` (everything except the CRC itself). A complete
    // frame whose CRC mismatches was corrupted in transit — advance past it
    // (empty payload) so the caller drops it and continues, rather than
    // retrying forever on bytes that will never validate.
    let expected_crc = crc32(&buf[..payload_end]);
    let stored_crc = u32::from_be_bytes([
        buf[payload_end],
        buf[payload_end + 1],
        buf[payload_end + 2],
        buf[payload_end + 3],
    ]);
    if expected_crc != stored_crc {
        tracing::warn!(
            total_len,
            "bedrock eventstream: complete frame failed CRC32; advancing past corrupt frame"
        );
        return Some((total_len, &[]));
    }
    Some((total_len, &buf[payload_start..payload_end]))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod validation_tests;
