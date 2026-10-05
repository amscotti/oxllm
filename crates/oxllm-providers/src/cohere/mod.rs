//! Cohere provider — translates OpenAI ⇄ Cohere Chat API v2.
//!
//! Wire targets:
//! - chat: `https://api.cohere.com/v2/chat`
//! - embeddings: `https://api.cohere.com/v2/embed`
//! - rerank: `https://api.cohere.com/v1/rerank`
//!
//! Cohere v2 is NOT OpenAI-native. This module performs real transforms:
//! system messages stay in the v2 `messages` array, response
//! `message.content` arrays are concatenated back into a single string,
//! `finish_reason` is mapped (`COMPLETE` → `stop`, `MAX_TOKENS` → `length`,
//! `TOOL_CALL` → `tool_calls`), and `usage.billed_units` feeds the OpenAI
//! `usage` block. Streaming maps Cohere `content-delta` / `message-end`
//! events to OpenAI chunks. See `docs/adding-a-provider.md`.
//!
//! An `api_base` may be a resource root, a versioned/custom path prefix, or
//! a full chat/embed/rerank endpoint. Explicit path prefixes and versions are
//! preserved. Bare hosted Cohere roots use v2 for chat/embed and v1 for
//! rerank. Custom roots keep unversioned chat/embed paths and the historic
//! /v1/rerank path; explicit prefixes or full endpoints override those defaults.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, RerankProvider, ResolvedCredentials,
};
use serde_json::{Map, Value, json};

/// The default Cohere v2 base URL.
pub const DEFAULT_COHERE_BASE: &str = "https://api.cohere.com/v2";

/// The default Cohere v1 base URL (used for the rerank endpoint).
pub const DEFAULT_COHERE_V1_BASE: &str = "https://api.cohere.com/v1";

/// The default Cohere embedding input type.
///
/// Cohere v2 `/embed` only accepts `search_document`, `search_query`,
/// `classification`, `clustering`, or `image`. `search_document` is the
/// correct default for embedding documents.
pub const DEFAULT_EMBED_INPUT_TYPE: &str = "search_document";

/// Cohere v2 `/embed` `input_type` allow-list.
const VALID_EMBED_INPUT_TYPES: [&str; 5] = [
    "search_document",
    "search_query",
    "classification",
    "clustering",
    "image",
];

/// True when `s` is an accepted Cohere v2 `/embed` `input_type`.
fn is_valid_embed_input_type(s: &str) -> bool {
    VALID_EMBED_INPUT_TYPES.contains(&s)
}

/// Cached stream metadata extracted from the first chunk of a Cohere stream.
///
/// OpenAI clients expect every SSE chunk to carry `id`, `created`, and
/// `model`. Cohere only emits the conversation `id`/`model` on the
/// `message-start` event (and never emits a `created` timestamp). The host's
/// request supplies the model when the native event omits it. We stash that
/// metadata and inject it into every subsequent chunk.
///
/// The production path carries this in a host-owned [`CohereStreamState`]
/// (see [`ChatProvider::transform_stream_chunk_stateful`]); the
/// process-wide [`stream_meta_cell`] is kept only for the single-threaded
/// `#[cfg(test)]` [`ChatProvider::transform_stream_chunk`] path, mirroring
/// the Anthropic split.
#[derive(Debug, Clone)]
struct StreamMeta {
    id: String,
    model: String,
    created: u64,
}

/// Per-stream state threaded through
/// [`ChatProvider::transform_stream_chunk_stateful`] by the host. Carries
/// the `message-start` metadata so the production stream pump does not
/// depend on any process-wide cell (which would corrupt `id`/`model`/
/// `created` across concurrent streams). Mirrors `AnthropicStreamState`.
#[derive(Default)]
pub struct CohereStreamState {
    meta: Option<StreamMeta>,
}

/// Process-wide cache of the most recent Cohere stream metadata. Used only
/// by the stateless [`ChatProvider::transform_stream_chunk`] path that
/// single-threaded unit tests call directly; the production host calls
/// [`ChatProvider::transform_stream_chunk_stateful`] instead, which carries
/// a host-owned [`CohereStreamState`] per stream.
fn stream_meta_cell() -> &'static Mutex<Option<StreamMeta>> {
    static CELL: OnceLock<Mutex<Option<StreamMeta>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// Clear the stream-meta cell. Used by tests to prevent cross-test
/// contamination when running in parallel.
#[cfg(test)]
pub(crate) fn clear_stream_meta_for_test() {
    if let Ok(mut guard) = stream_meta_cell().lock() {
        *guard = None;
    }
}

/// Extract `id`/`model` from a Cohere `message-start` event body. Cohere
/// surfaces them at the top level of the event object (unlike Anthropic,
/// which nests them under `message`). Falls back to empty strings when
/// the event is sparse (the spec does not guarantee either field).
fn extract_stream_meta(value: &Value) -> StreamMeta {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-cohere")
        .to_string();
    let model = value
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    StreamMeta {
        id,
        model,
        created: unix_now(),
    }
}

/// Inject `id` / `created` / `model` from a stashed [`StreamMeta`] into an
/// already-built chunk. Only mutates objects; leaves arrays/scalars untouched.
fn inject_stream_meta(chunk: &mut Value, meta: &StreamMeta) {
    if let Value::Object(map) = chunk {
        map.insert("id".to_string(), Value::String(meta.id.clone()));
        map.insert("created".to_string(), json!(meta.created));
        map.insert("model".to_string(), Value::String(meta.model.clone()));
    }
}

/// Core chunk transform, parameterised over the slot that holds the stream's
/// stashed [`StreamMeta`]. The slot is the process-wide cell for the
/// stateless test path ([`ChatProvider::transform_stream_chunk`]) and the
/// host-owned [`CohereStreamState::meta`] for the production path
/// ([`ChatProvider::transform_stream_chunk_stateful`]) — so concurrent
/// streams no longer corrupt each other's `id`/`model`/`created`.
fn transform_stream_chunk_impl(
    chunk: &str,
    meta_slot: &mut Option<StreamMeta>,
) -> Result<Option<Value>> {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let data_str = match extract_data_line(trimmed) {
        Some(s) => s,
        None => return Ok(None),
    };
    if data_str.trim() == "[DONE]" || data_str.is_empty() {
        return Ok(None);
    }

    let value: Value = serde_json::from_str(&data_str)
        .map_err(|e| oxllm_core::Error::Transform(format!("invalid cohere stream chunk: {e}")))?;

    // Check for an in-body `error` object before event-type matching, so
    // an upstream error embedded in a `data:` payload surfaces as an
    // `Upstream` error instead of falling through to the unknown-event
    // arm (which would silently drop it).
    if let Some(error) = value
        .pointer("/delta/error")
        .filter(|e| !e.is_null() && e.as_str() != Some(""))
        .or_else(|| value.get("error").filter(|e| !e.is_null()))
    {
        let message = error_message(error);
        return Err(oxllm_core::Error::Upstream {
            status: upstream_status_for_error(error),
            message: message.to_string(),
        });
    }
    // A failed terminal event must not masquerade as a successful stop even
    // when the upstream omits its optional error message.
    if let Some(reason @ ("ERROR" | "TIMEOUT")) = value
        .pointer("/delta/finish_reason")
        .and_then(Value::as_str)
    {
        return Err(oxllm_core::Error::Upstream {
            status: if reason == "TIMEOUT" { 504 } else { 500 },
            message: format!("cohere generation ended with {reason}"),
        });
    }

    let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let is_message_start = event_type == "message-start";
    let result = match event_type {
        "message-start" => {
            // Stash id/model/created before building the chunk so every
            // subsequent chunk in this stream can inherit them.
            let meta = extract_stream_meta(&value);
            let chunk = build_role_chunk(&meta);
            *meta_slot = Some(meta);
            Ok(Some(chunk))
        }
        "content-delta" => Ok(build_content_delta_chunk(&value)),
        "tool-plan-delta" => Ok(build_tool_plan_delta_chunk(&value)),
        "tool-call-start" | "tool-call-delta" => Ok(build_tool_call_delta_chunk(&value)),
        // The end event only carries an index. Arguments have already been
        // emitted as deltas; repeating a complete call would duplicate them.
        "tool-call-end" => Ok(None),
        "tool-call" => Ok(Some(build_tool_call_chunk(&value))),
        // Citations are emitted inline on the `citation-start` event so
        // the chunk is a pure function of its input (no cross-event
        // state) — this keeps streaming deterministic under parallel
        // execution. The OpenAI SSE shape has no standard citation
        // field, so they ride as a provider-specific top-level
        // `citations` array, matching the non-streaming response.
        "citation-start" => Ok(build_citation_start_chunk(&value)),
        "message-end" => Ok(Some(build_message_end_chunk(&value))),
        _ => Ok(None),
    };

    // OpenAI clients expect every stream chunk to carry `id`, `created`,
    // and `model`. `message-start` populates them directly; for every
    // other event we inject the stashed meta.
    let mut chunk = result?;
    if let Some(chunk_value) = &mut chunk {
        if !is_message_start {
            if let Some(meta) = meta_slot.as_ref() {
                inject_stream_meta(chunk_value, meta);
            }
        }
    }
    Ok(chunk)
}

/// Pick the HTTP status to report for an in-body Cohere stream `error`
/// object. Cohere surfaces capacity / overload failures inside the SSE
/// stream (rather than as a non-200 response), so a naive 500 default would
/// hide them from the orchestrator's retry policy (which keys off 429/5xx).
/// Map known overload / rate-limit signals to 503 so the retry policy
/// engages; everything else stays 500.
fn upstream_status_for_error(error: &Value) -> u16 {
    let hay = error_message(error).to_ascii_lowercase();
    let code = error
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let needle = |s: &str| hay.contains(s) || code.contains(s);
    if needle("overload")
        || needle("overloaded")
        || needle("capacity")
        || needle("rate limit")
        || needle("rate_limit")
        || needle("too many requests")
        || needle("throttl")
        || needle("busy")
        || needle("service unavailable")
    {
        503
    } else {
        500
    }
}

fn error_message(error: &Value) -> &str {
    error
        .as_str()
        .or_else(|| error.get("message").and_then(Value::as_str))
        .unwrap_or("cohere stream error")
}

/// The Cohere provider.
#[derive(Debug, Clone, Default)]
pub struct CohereProvider;

impl ChatProvider for CohereProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        cohere_endpoint_url(deployment, DEFAULT_COHERE_BASE, "v2", "chat")
    }

    fn apply_auth(
        &self,
        _deployment: &Deployment,
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
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let mut out = Map::new();

        let model = resolve_model(request, deployment);
        out.insert("model".to_string(), Value::String(model.to_string()));

        let messages = request["messages"]
            .as_array()
            .map(|messages| messages.iter().map(normalize_message).collect())
            .unwrap_or_default();
        out.insert("messages".to_string(), Value::Array(messages));

        // M2 (Cohere twin): Cohere v2 reads `max_tokens`, not
        // `max_completion_tokens`. Prefer the OpenAI alias when present so
        // the newer param wins (matches the Mistral precedence rule).
        if let Some(v) = request_param(request, "max_completion_tokens")
            .or_else(|| request_param(request, "max_tokens"))
        {
            out.insert("max_tokens".to_string(), v.clone());
        }
        // Standard OpenAI sampling params. Route through `request_param`
        // (top-level then `extra_body`) for parity with the Cohere-native
        // params below, so a caller using LiteLLM's `extra_body` pass-through
        // convention is not silently ignored.
        for key in [
            "temperature",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        ] {
            if let Some(v) = request_param(request, key) {
                out.insert(key.to_string(), v.clone());
            }
        }
        // Bug #40: Cohere v2 Chat uses `p` and `k` (not `top_p` / `top_k`).
        // Forward these under their Cohere names so the values reach the
        // upstream instead of being silently ignored.
        if let Some(v) = request_param(request, "top_p") {
            out.insert("p".to_string(), v.clone());
        }
        if let Some(v) = request_param(request, "top_k") {
            out.insert("k".to_string(), v.clone());
        }
        if let Some(stop) = request_param(request, "stop") {
            out.insert("stop_sequences".to_string(), normalize_stop_sequences(stop));
        }
        if let Some(stream) = request.get("stream").and_then(|v| v.as_bool()) {
            out.insert("stream".to_string(), json!(stream));
        }

        // v2 retains documents, but not v1's preamble, connectors,
        // search_queries_only, conversation_id, prompt_truncation,
        // num_generations or force_single_step.
        if let Some(v) = request_param(request, "documents") {
            out.insert("documents".to_string(), v.clone());
        }

        let mut tools: Vec<Value> = request_param(request, "tools")
            .and_then(Value::as_array)
            .map(|tools| tools.iter().map(map_tool_to_cohere).collect())
            .unwrap_or_default();
        if let Some(tool_choice) = request_param(request, "tool_choice") {
            if let Some(mapped) = map_tool_choice_to_cohere(tool_choice, &mut tools)? {
                out.insert("tool_choice".to_string(), mapped);
            }
        }
        if !tools.is_empty() {
            out.insert("tools".to_string(), Value::Array(tools));
        }

        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let id = body
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("chatcmpl-cohere")
            .to_string();

        let (content, tool_calls) = extract_assistant_content(&body["message"]);
        let mut message = Map::new();
        message.insert("role".to_string(), json!("assistant"));
        // Match OpenAI/LiteLLM: emit `content: null` for a tool-call message
        // with no text so strict clients can branch on `content.is_null()`.
        let content_value = if tool_calls.is_empty() || !content.is_empty() {
            Value::String(content)
        } else {
            Value::Null
        };
        message.insert("content".to_string(), content_value);
        if !tool_calls.is_empty() {
            message.insert("tool_calls".to_string(), Value::Array(tool_calls));
        }

        let finish_reason = body
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .map(map_finish_reason_to_openai)
            .unwrap_or_else(|| json!("stop"));

        let (prompt_tokens, completion_tokens) = extract_usage_tokens(body);
        let total_tokens = prompt_tokens + completion_tokens;

        let mut choice = Map::new();
        choice.insert("index".to_string(), json!(0));
        choice.insert("message".to_string(), Value::Object(message));
        choice.insert("finish_reason".to_string(), finish_reason);

        let mut out = Map::new();
        out.insert("id".to_string(), Value::String(id));
        out.insert("object".to_string(), json!("chat.completion"));
        out.insert("created".to_string(), json!(unix_now()));
        out.insert("model".to_string(), Value::String(model.to_string()));
        out.insert("choices".to_string(), json!([Value::Object(choice)]));
        out.insert(
            "usage".to_string(),
            json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            }),
        );

        // Cohere v2 RAG responses carry citations under `message.citations`
        // (per the v2 Chat API docs and LiteLLM's `CohereV2ChatResponse`
        // type / `v2_transformation.transform_response`). Surface them as a
        // provider-specific extension on the OpenAI response. Fall back to a
        // top-level `body.citations` defensively for non-standard upstreams.
        // Each entry is `{start, end, text, document_ids}`.
        let citations = body
            .get("message")
            .and_then(|m| m.get("citations"))
            .or_else(|| body.get("citations"));
        if let Some(citations) = citations {
            if citations.as_array().is_some_and(|a| !a.is_empty()) {
                out.insert("citations".to_string(), citations.clone());
            }
        }

        Ok(Value::Object(out))
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        // Global-cell-backed path. Used by single-threaded unit tests that
        // call this trait method directly. The production host calls
        // `transform_stream_chunk_stateful` instead (see below) so the meta
        // travels with the tokio task across worker-thread migrations and is
        // not shared across concurrent streams.
        let mut guard = match stream_meta_cell().lock() {
            Ok(g) => g,
            Err(_) => return Ok(None),
        };
        transform_stream_chunk_impl(chunk, &mut guard)
    }

    fn transform_stream_chunk_stateful(
        &self,
        chunk: &str,
        state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
        original_request: &Value,
    ) -> Result<Option<Value>> {
        // Host-owned state path: the per-stream accumulator lives in a
        // `Box<dyn Any>` the host created and threads through every chunk, so
        // it follows the tokio task across worker-thread migrations and is
        // never shared between concurrent streams (unlike the global cell).
        let s = state.get_or_insert_with(|| Box::new(CohereStreamState::default()));
        // A foreign state type means the host mixed accumulators across
        // providers — a wiring bug, not a poisoned stream. Return an error
        // (drops this stream) instead of panicking the worker thread.
        let Some(s) = s.downcast_mut::<CohereStreamState>() else {
            return Err(oxllm_core::Error::Transform(
                "cohere stream state type mismatch".into(),
            ));
        };
        let mut out = transform_stream_chunk_impl(chunk, &mut s.meta)?;
        if let Some(meta) = s.meta.as_mut() {
            if meta.model.is_empty() {
                meta.model = original_request
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
            }
            if let Some(chunk) = out.as_mut() {
                inject_stream_meta(chunk, meta);
            }
        }
        Ok(out)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        Some(vec![
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "stream",
            "tools",
            "tool_choice",
            "seed",
            // Cohere-native v2 pass-through parameter.
            "documents",
            "extra_body",
        ])
    }
}

impl EmbeddingProvider for CohereProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        cohere_endpoint_url(deployment, DEFAULT_COHERE_BASE, "v2", "embed")
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
        let mut out = Map::new();
        let model = resolve_model(request, deployment);
        out.insert("model".to_string(), Value::String(model.to_string()));

        let texts = collect_embedding_texts(&request["input"]);
        out.insert("texts".to_string(), json!(texts));
        // Cohere embed models are asymmetric: `search_document` for indexing,
        // `search_query` for retrieval queries. Honour a caller-supplied
        // `input_type` (top-level or `extra_body`, the LiteLLM pass-through
        // convention), validated against Cohere's allow-list; otherwise default
        // to `search_document`. An invalid value is dropped so we never send
        // a value Cohere would 422 on.
        let input_type = request_param(request, "input_type")
            .and_then(|v| v.as_str())
            .filter(|s| is_valid_embed_input_type(s))
            .unwrap_or(DEFAULT_EMBED_INPUT_TYPE);
        out.insert(
            "input_type".to_string(),
            Value::String(input_type.to_string()),
        );

        if let Some(encoding_format) = request.get("encoding_format") {
            if !encoding_format.is_null() {
                let types = match encoding_format {
                    Value::Array(arr) => Value::Array(arr.clone()),
                    Value::String(s) => json!([s]),
                    other => json!([other]),
                };
                out.insert("embedding_types".to_string(), types);
            }
        }
        if let Some(dimensions) = request.get("dimensions") {
            if !dimensions.is_null() {
                out.insert("output_dimension".to_string(), dimensions.clone());
            }
        }

        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let mut data: Vec<Value> = Vec::new();
        if let Some(embeddings) = body.get("embeddings") {
            if let Some(arr) = embeddings.as_array() {
                for (idx, vec) in arr.iter().enumerate() {
                    data.push(json!({
                    "object": "embedding",
                    "index": idx,
                    "embedding": vec,
                    }));
                }
            } else if let Some(map) = embeddings.as_object() {
                // When the request sets `embedding_types`, Cohere returns
                // `embeddings` keyed by type (`float`, `int8`, …). The
                // OpenAI shape has a single `embedding` per input, so pick
                // one canonical type — prefer `float` (the OpenAI default),
                // else the first key deterministically — and emit one entry
                // per input with globally-unique sequential indices.
                // Iterating every (`kind`, list) pair with per-pair
                // `enumerate()` previously produced colliding indices and a
                // multiplied `data` array, violating `data[i].index == i`.
                let canonical = map.get("float").or_else(|| map.values().next());
                if let Some(arr) = canonical.and_then(|v| v.as_array()) {
                    for (idx, vec) in arr.iter().enumerate() {
                        data.push(json!({
                        "object": "embedding",
                        "index": idx,
                        "embedding": vec,
                        }));
                    }
                }
            }
        }

        // LiteLLM's embed `_calculate_usage` sums `billed_units.input_tokens`
        // (text) and `billed_units.images` (image-embedding token count) into
        // `prompt_tokens`. For an image-embedding call Cohere reports tokens
        // under `images`, not `input_tokens`, so both must be considered.
        let billed = body.get("meta").and_then(|m| m.get("billed_units"));
        let prompt_tokens = billed
            .map(|b| {
                let text = b.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let images = b.get("images").and_then(|v| v.as_u64()).unwrap_or(0);
                text + images
            })
            .unwrap_or(0);

        let mut out = Map::new();
        out.insert("object".to_string(), json!("list"));
        out.insert("data".to_string(), Value::Array(data));
        out.insert("model".to_string(), Value::String(model.to_string()));
        out.insert(
            "usage".to_string(),
            json!({
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            }),
        );

        Ok(Value::Object(out))
    }
}

impl RerankProvider for CohereProvider {
    fn rerank_url(&self, deployment: &Deployment) -> String {
        cohere_endpoint_url(deployment, DEFAULT_COHERE_V1_BASE, "v1", "rerank")
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
        let mut out = Map::new();
        let model = resolve_model(request, deployment);
        out.insert("model".to_string(), Value::String(model.to_string()));

        if let Some(query) = request.get("query") {
            if !query.is_null() {
                out.insert("query".to_string(), query.clone());
            }
        }
        if let Some(documents) = request.get("documents") {
            if !documents.is_null() {
                out.insert("documents".to_string(), documents.clone());
            }
        }
        if let Some(top_n) = request.get("top_n") {
            if !top_n.is_null() {
                out.insert("top_n".to_string(), top_n.clone());
            }
        } else if let Some(top_n) = request.get("top_n_override") {
            if !top_n.is_null() {
                out.insert("top_n".to_string(), top_n.clone());
            }
        }
        for key in ["rank_fields", "return_documents", "max_chunks_per_doc"] {
            if let Some(v) = request.get(key) {
                if !v.is_null() {
                    out.insert(key.to_string(), v.clone());
                }
            }
        }

        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let mut out = Map::new();
        out.insert(
            "id".to_string(),
            body.get("id").cloned().unwrap_or(Value::Null),
        );
        out.insert(
            "results".to_string(),
            body.get("results").cloned().unwrap_or(Value::Null),
        );
        if let Some(meta) = body.get("meta") {
            let billed = meta.get("billed_units").cloned().unwrap_or(Value::Null);
            if !billed.is_null() {
                out.insert("usage".to_string(), billed);
            }
        }
        Ok(Value::Object(out))
    }
}

impl Provider for CohereProvider {
    fn id(&self) -> ProviderId {
        "cohere"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(
            endpoint,
            oxllm_core::Endpoint::ChatCompletions
                | oxllm_core::Endpoint::Embeddings
                | oxllm_core::Endpoint::Rerank
        )
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        Some(self)
    }

    fn rerank(&self) -> Option<&dyn RerankProvider> {
        Some(self)
    }
}

fn strip_cohere_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("cohere", rest)) => rest,
        _ => model,
    }
}

/// Resolve the model id for a Cohere chat / embeddings / rerank request from
/// a configured deployment.
///
/// For non-wildcard deployments the configured (prefix-stripped) model id
/// always wins. For wildcard deployments (`*` or `cohere/*`), the actual
/// model id is taken from the request body's `model` field (after stripping
/// the provider prefix) — otherwise the literal `*` would be forwarded
/// upstream as the model id and rejected; if the request carries no `model`,
/// the wildcard literal is forwarded as-is. Mirrors
/// `OpenAiCompatProvider::resolve_model` and the Voyage/Jina rerank helpers.
fn resolve_model<'a>(request: &'a Value, deployment: &'a Deployment) -> &'a str {
    let configured = strip_cohere_prefix(&deployment.oxllm_params.model);
    let is_wildcard = configured == "*" || configured.ends_with("/*");
    if is_wildcard {
        request
            .get("model")
            .and_then(Value::as_str)
            .map(strip_cohere_prefix)
            .unwrap_or(configured)
    } else {
        configured
    }
}

/// Look up a request parameter, preferring the top-level key and falling back
/// to `extra_body.<key>` (the LiteLLM pass-through convention). Null values
/// are treated as absent.
fn request_param<'a>(request: &'a Value, key: &str) -> Option<&'a Value> {
    request.get(key).filter(|v| !v.is_null()).or_else(|| {
        request
            .get("extra_body")
            .and_then(|eb| eb.get(key))
            .filter(|v| !v.is_null())
    })
}

fn cohere_endpoint_url(
    deployment: &Deployment,
    default_base: &str,
    default_version: &str,
    endpoint: &str,
) -> String {
    let base = deployment
        .oxllm_params
        .api_base
        .as_deref()
        .unwrap_or(default_base);
    let (base, fragment) = base
        .split_once('#')
        .map_or((base, None), |(base, fragment)| (base, Some(fragment)));
    let (path, query) = base
        .split_once('?')
        .map_or((base, None), |(path, query)| (path, Some(query)));
    let mut path = path.trim_end_matches('/');
    // A full endpoint is also a usable base for the other Cohere operations.
    // Inspect only the URL path, never an authority named chat/embed/rerank.
    let has_path = path
        .split_once("://")
        .is_some_and(|(_, rest)| rest.contains('/'));
    let mut full_endpoint = false;
    if has_path
        && let Some((prefix, tail)) = path.rsplit_once('/')
        && matches!(tail, "chat" | "embed" | "rerank")
    {
        path = prefix;
        full_endpoint = true;
    }
    let is_resource_root = path
        .split_once("://")
        .is_some_and(|(_, rest)| !rest.contains('/'));
    // Rerank's established custom-root contract includes /v1; chat/embed
    // custom roots do not. An explicit path or endpoint supplies its own route.
    let infer_version = !full_endpoint
        && (is_hosted_cohere_root(path) || (endpoint == "rerank" && is_resource_root));
    let mut url = if infer_version {
        format!("{path}/{default_version}/{endpoint}")
    } else {
        format!("{path}/{endpoint}")
    };
    if let Some(query) = query {
        url.push('?');
        url.push_str(query);
    }
    if let Some(fragment) = fragment {
        url.push('#');
        url.push_str(fragment);
    }
    url
}

/// Infer a hosted version only for the API's exact public authorities.
/// Custom roots, alternate ports, and proxy path prefixes are explicit routes.
fn is_hosted_cohere_root(base: &str) -> bool {
    let Some((scheme, authority)) = base.split_once("://") else {
        return false;
    };
    let authority = if scheme.eq_ignore_ascii_case("https") {
        authority.strip_suffix(":443").unwrap_or(authority)
    } else if scheme.eq_ignore_ascii_case("http") {
        authority.strip_suffix(":80").unwrap_or(authority)
    } else {
        return false;
    };
    let authority = authority.trim_end_matches('.');
    authority.eq_ignore_ascii_case("api.cohere.com")
        || authority.eq_ignore_ascii_case("api.cohere.ai")
}

fn extract_message_text(msg: &Value) -> Option<String> {
    match msg.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(arr)) => {
            let mut parts: Vec<String> = Vec::new();
            for block in arr {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    parts.push(t.to_string());
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(""))
            }
        }
        _ => None,
    }
}

fn normalize_message(msg: &Value) -> Value {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
    if role == "system" || role == "developer" {
        return json!({"role": "system", "content": extract_message_text(msg).unwrap_or_default()});
    }
    if role == "tool" || role == "function" {
        return map_tool_response_message(msg);
    }
    let mut out = Map::new();
    out.insert("role".to_string(), Value::String(role.to_string()));
    if let Some(content) = msg.get("content") {
        out.insert("content".to_string(), content.clone());
    }
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        let mapped: Vec<Value> = tool_calls.iter().map(map_tool_call_to_cohere).collect();
        if !mapped.is_empty() {
            out.insert("tool_calls".to_string(), Value::Array(mapped));
        }
    }
    Value::Object(out)
}

/// Map OpenAI `role: "tool"` messages to Cohere Chat API **v2** tool results:
/// `role: "tool"`, top-level `tool_call_id`, and `content` as document blocks.
fn map_tool_response_message(msg: &Value) -> Value {
    let tool_call_id = msg
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let content = match msg.get("content") {
        // Already v2-shaped document/text blocks — pass through.
        Some(Value::Array(arr))
            if arr.iter().any(|b| {
                matches!(
                    b.get("type").and_then(|t| t.as_str()),
                    Some("document" | "text")
                )
            }) =>
        {
            Value::Array(arr.clone())
        }
        Some(Value::Array(arr)) => {
            let joined = arr
                .iter()
                .filter_map(|b| b.get("text").and_then(|v| v.as_str()).map(String::from))
                .collect::<Vec<_>>()
                .join("");
            json!([{
            "type": "document",
            "document": {"data": joined},
            }])
        }
        Some(Value::String(s)) => json!([{
        "type": "document",
        "document": {"data": s},
        }]),
        Some(other) => json!([{
        "type": "document",
        "document": {"data": other.to_string()},
        }]),
        None => json!([{
        "type": "document",
        "document": {"data": ""},
        }]),
    };
    json!({
    "role": "tool",
    "tool_call_id": tool_call_id,
    "content": content,
    })
}

fn map_tool_call_to_cohere(tc: &Value) -> Value {
    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let function = tc.get("function").cloned().unwrap_or_else(|| json!({}));
    let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
    // Cohere v2 expects `arguments` as a JSON-encoded string. Accept both the
    // spec-mandated string form and an object form (stringify it) for
    // symmetry with the response-side mapping, rather than silently dropping
    // object-form arguments to `"{}"`.
    let args = match function.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => "{}".to_string(),
        Some(other) => other.to_string(),
    };
    json!({
    "id": id,
    "type": "function",
    "function": {"name": name, "arguments": args},
    })
}

/// Map OpenAI tool definitions to Cohere Chat API **v2** shape: keep
/// `function.parameters` as JSON Schema (not v1 `parameter_definitions`).
fn map_tool_to_cohere(tool: &Value) -> Value {
    let function = tool.get("function").cloned().unwrap_or_else(|| json!({}));
    let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let description = function.get("description").cloned();
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));

    let mut fn_obj = Map::new();
    fn_obj.insert("name".to_string(), Value::String(name.to_string()));
    fn_obj.insert("parameters".to_string(), parameters);
    if let Some(desc) = description {
        if !desc.is_null() {
            fn_obj.insert("description".to_string(), desc);
        }
    }

    json!({
    "type": "function",
    "function": Value::Object(fn_obj),
    })
}

fn map_tool_choice_to_cohere(tool_choice: &Value, tools: &mut Vec<Value>) -> Result<Option<Value>> {
    if let Some(s) = tool_choice.as_str() {
        return Ok(match s {
            "auto" => None,
            "required" | "any" | "REQUIRED" => Some(json!("REQUIRED")),
            "none" | "NONE" => Some(json!("NONE")),
            _ => {
                return Err(oxllm_core::Error::Transform(
                    "unsupported cohere tool_choice".into(),
                ));
            }
        });
    }
    if let Some(name) = tool_choice
        .as_object()
        .and_then(|o| o.get("function"))
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
    {
        // Cohere has no named-function choice. Restrict the available tools
        // and require a call to preserve OpenAI's named-choice semantics.
        tools.retain(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(name));
        if tools.is_empty() {
            return Err(oxllm_core::Error::Transform(
                "cohere named tool_choice must match a supplied tool".into(),
            ));
        }
        return Ok(Some(json!("REQUIRED")));
    }
    Err(oxllm_core::Error::Transform(
        "unsupported cohere tool_choice".into(),
    ))
}

fn normalize_stop_sequences(stop: &Value) -> Value {
    match stop {
        Value::String(s) => json!([s]),
        Value::Array(arr) => Value::Array(arr.clone()),
        other => json!([other]),
    }
}

fn extract_assistant_content(message: &Value) -> (String, Vec<Value>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    if let Some(content) = message.get("content").and_then(|v| v.as_array()) {
        for block in content {
            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                text_parts.push(t.to_string());
            }
        }
    } else if let Some(text) = message.get("content").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            text_parts.push(text.to_string());
        }
    }

    if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for (idx, call) in calls.iter().enumerate() {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let function = call.get("function").cloned().unwrap_or_else(|| json!({}));
            let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args_value = function
                .get("parameters")
                .cloned()
                .or_else(|| function.get("arguments").cloned());
            let args = match args_value {
                Some(Value::String(s)) => s,
                Some(other) => other.to_string(),
                None => "{}".to_string(),
            };
            tool_calls.push(json!({
            "id": id,
            "type": "function",
            "index": idx,
            "function": {"name": name, "arguments": args},
            }));
        }
    }

    (text_parts.join(""), tool_calls)
}

/// Derive `(prompt, completion)` token counts for the non-streaming path.
///
/// Bills on `usage.billed_units` (what Cohere actually charges) and falls back
/// to `usage.tokens` per-field when `billed_units` — or its inner field — is
/// absent. This mirrors `build_message_end_chunk` on the streaming path so the
/// same Cohere call is metered identically regardless of `stream:`. The golden
/// `chat_response_001.json` shows the two bases diverge on the wire
/// (billed_units 12/8 vs tokens 14/9).
fn extract_usage_tokens(body: &Value) -> (u64, u64) {
    let usage = body.get("usage");
    let prompt = usage
        .and_then(|u| u.get("billed_units"))
        .and_then(|b| b.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            usage
                .and_then(|u| u.get("tokens"))
                .and_then(|t| t.get("input_tokens"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0);
    let completion = usage
        .and_then(|u| u.get("billed_units"))
        .and_then(|b| b.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            usage
                .and_then(|u| u.get("tokens"))
                .and_then(|t| t.get("output_tokens"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0);
    (prompt, completion)
}

fn map_finish_reason_to_openai(reason: &str) -> Value {
    match reason {
        "COMPLETE" => json!("stop"),
        "MAX_TOKENS" => json!("length"),
        "TOOL_CALL" | "TOOL_CALLS" | "tool_call" | "tool_calls" => json!("tool_calls"),
        // Generic `error` / Cohere `ERROR` are not OpenAI finish reasons.
        // OpenAI-compat clients treat an unknown value as a failure; map
        // to `stop`. `ERROR_TOXIC` stays a content filter.
        "ERROR" | "error" => json!("stop"),
        "ERROR_TOXIC" => json!("content_filter"),
        _ => json!("stop"),
    }
}

fn collect_embedding_texts(input: &Value) -> Vec<String> {
    match input {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                Value::Array(inner) => Some(
                    inner
                        .iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect::<Vec<_>>()
                        .join(""),
                ),
                Value::Number(n) => Some(n.to_string()),
                Value::Object(o) => {
                    if let Some(t) = o.get("text").and_then(|v| v.as_str()) {
                        Some(t.to_string())
                    } else if let Some(t) = o.get("input").and_then(|v| v.as_str()) {
                        Some(t.to_string())
                    } else {
                        serde_json::to_string(o).ok()
                    }
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Extract the `data:` payload from an SSE chunk.
///
/// Per the SSE spec, an event with multiple `data:` lines must have them
/// joined with `\n` before dispatch, so this collects every `data:` line
/// and concatenates them. A bare JSON object line (no `data:` prefix) is
/// also accepted as a payload. Comment (`:`), `event:`/`id:`/`retry:` and
/// unknown field lines are ignored.
fn extract_data_line(raw: &str) -> Option<String> {
    crate::util::extract_sse_data(raw).or_else(|| {
        let trimmed = raw.trim();
        trimmed.starts_with('{').then(|| trimmed.to_string())
    })
}

fn build_role_chunk(meta: &StreamMeta) -> Value {
    json!({
    "object": "chat.completion.chunk",
    "id": meta.id,
    "created": meta.created,
    "model": meta.model,
    "choices": [{
    "index": 0,
    "delta": {"role": "assistant", "content": ""},
    "finish_reason": null,
    }],
    })
}

fn build_content_delta_chunk(value: &Value) -> Option<Value> {
    let text = value
        .get("delta")
        .and_then(|d| d.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| match c {
            Value::String(s) => Some(s.clone()),
            Value::Object(o) => o.get("text").and_then(|t| t.as_str()).map(String::from),
            _ => None,
        })?;
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

/// Map a Cohere v2 `tool-plan-delta` event (the model's reasoning about
/// which tool to call, emitted before `tool-call-delta`) to an OpenAI delta
/// carrying `reasoning_content` — the same field DeepSeek/Anthropic thinking
/// streams use. Cohere ships the text at `delta.message.tool_plan` (a bare
/// string); an object form `{"text": ...}` is accepted defensively.
fn build_tool_plan_delta_chunk(value: &Value) -> Option<Value> {
    let text = value
        .get("delta")
        .and_then(|d| d.get("message"))
        .and_then(|m| m.get("tool_plan"))
        .and_then(|tp| match tp {
            Value::String(s) => Some(s.clone()),
            Value::Object(o) => o.get("text").and_then(|t| t.as_str()).map(String::from),
            _ => None,
        })?;
    if text.is_empty() {
        return None;
    }
    Some(json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {"reasoning_content": text},
    "finish_reason": null,
    }],
    }))
}

/// Map a Cohere v2 `citation-start` event to an OpenAI chunk that carries
/// the citations inline as a top-level `citations` array (provider-specific
/// extension, same shape as the non-streaming response).
///
/// Emitting inline — rather than buffering across events for `message-end` —
/// keeps [`ChatProvider::transform_stream_chunk`] a pure function of its
/// input: the output for a given event never depends on what other events
/// have been processed. This is what makes streaming deterministic even when
/// the trait has no per-stream handle and multiple streams share a process.
/// Cohere ships a single citation at `delta.message.citations`, including
/// its sources. An array from older compatible upstreams is also accepted.
fn build_citation_start_chunk(value: &Value) -> Option<Value> {
    let citations = value
        .get("delta")
        .and_then(|d| d.get("message"))
        .and_then(|m| m.get("citations"))?;
    let citations = match citations {
        Value::Object(_) => vec![citations.clone()],
        Value::Array(citations) => citations.clone(),
        _ => return None,
    };
    if citations.is_empty() {
        return None;
    }
    Some(json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {},
    "finish_reason": null,
    }],
    "citations": citations,
    }))
}

fn build_tool_call_delta_chunk(value: &Value) -> Option<Value> {
    let tool_calls = value
        .pointer("/delta/message/tool_calls")
        .or_else(|| value.pointer("/delta/tool_calls"))?;
    // v2 emits one tool-call object per event, with the index on the event,
    // not on the call. Accept legacy arrays without changing the v2 path.
    let mapped: Vec<Value> = match tool_calls {
        Value::Object(_) => vec![tool_call_delta_entry(
            value.get("index").and_then(Value::as_u64).unwrap_or(0),
            tool_calls,
        )],
        Value::Array(calls) if !calls.is_empty() => calls
            .iter()
            .enumerate()
            .map(|(i, tc)| {
                tool_call_delta_entry(
                    tc.get("index").and_then(Value::as_u64).unwrap_or(i as u64),
                    tc,
                )
            })
            .collect(),
        _ => return None,
    };
    Some(json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {
    "tool_calls": mapped,
    },
    "finish_reason": null,
    }],
    }))
}

/// Extract one OpenAI-shaped `delta.tool_calls[i]` entry from a single
/// Cohere tool-call object. Tolerates both the legacy flat shape
/// (`name`/`arguments` at the call root) and the OpenAI-nested shape
/// (`function.name`/`function.arguments`).
fn tool_call_delta_entry(index: u64, tc: &Value) -> Value {
    let name = tc
        .get("name")
        .or_else(|| tc.get("function").and_then(|f| f.get("name")))
        .and_then(|v| v.as_str());
    let args_value = tc
        .get("arguments")
        .or_else(|| tc.get("function").and_then(|f| f.get("arguments")))
        .or_else(|| tc.get("function").and_then(|f| f.get("parameters")));
    let mut entry = json!({"index": index, "function": {}});
    if let Some(id) = tc.get("id").and_then(Value::as_str) {
        entry["id"] = json!(id);
        entry["type"] = json!("function");
    }
    if let Some(name) = name {
        entry["function"]["name"] = json!(name);
    }
    if let Some(args) = args_value.filter(|v| !v.is_null()) {
        entry["function"]["arguments"] = match args {
            Value::String(_) => args.clone(),
            other => json!(other.to_string()),
        };
    }
    entry
}

/// Compatibility for legacy `tool-call` fixtures/upstreams. The official v2
/// protocol instead uses start/delta/end with `delta.message.tool_calls`.
fn build_tool_call_chunk(value: &Value) -> Value {
    let tool_calls = value
        .get("delta")
        .and_then(|d| d.get("tool_calls"))
        .and_then(|v| v.as_array());
    let entries: Vec<Value> = if let Some(arr) = tool_calls {
        if arr.is_empty() {
            vec![tool_call_delta_entry(0, &json!({}))]
        } else {
            arr.iter()
                .enumerate()
                .map(|(i, tc)| tool_call_delta_entry(i as u64, tc))
                .collect()
        }
    } else {
        // Defensive fallback: a variant that places the call at the top
        // level instead of under `delta`. Reuse the same extraction.
        vec![tool_call_delta_entry(0, value)]
    };
    json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {
    "tool_calls": entries,
    },
    "finish_reason": null,
    }],
    })
}

fn build_message_end_chunk(value: &Value) -> Value {
    let finish_reason = value
        .get("delta")
        .and_then(|d| d.get("finish_reason"))
        .and_then(|v| v.as_str())
        .map(map_finish_reason_to_openai);

    let mut chunk = json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {},
    "finish_reason": finish_reason,
    }],
    });

    // Bill on `billed_units` (what Cohere actually charges, same basis as
    // the non-streaming path) and fall back to raw `tokens` when the stream
    // omits `billed_units`. Previously this read `tokens` unconditionally,
    // so the same Cohere call was metered differently depending on `stream:`
    // (the golden `chat_response_001.json` shows billed_units 12/8 vs tokens
    // 14/9 — they diverge on the wire).
    let usage = value
        .get("delta")
        .and_then(|d| d.get("usage"))
        .and_then(|u| u.get("billed_units"))
        .or_else(|| {
            value
                .get("delta")
                .and_then(|d| d.get("usage"))
                .and_then(|u| u.get("tokens"))
        });
    if let Some(tokens) = usage {
        let prompt = tokens
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let completion = tokens
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        chunk["usage"] = json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        });
    }
    chunk
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod v2_contract_tests;
