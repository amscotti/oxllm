//! Anthropic provider — translates OpenAI ⇄ Anthropic Messages API.
//!
//! Wire target: `POST https://api.anthropic.com/v1/messages`.
//!
//! Unlike the OpenAI identity transform, this provider rewrites the request
//! shape: OpenAI `messages` (where system messages live inline) become the
//! Anthropic top-level `system` + `messages` split. Response/stream chunks
//! are mapped back to the OpenAI shape. See `docs/adding-a-provider.md`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use serde_json::{Map, Value, json};

/// Default `max_tokens` value Anthropic requires when the client did not set one.
pub const DEFAULT_MAX_TOKENS: u64 = 8192;

/// The Anthropic API version header value.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// `anthropic-beta` header values for gated features. Selected dynamically by
/// [`collect_beta_headers`] based on which features the request actually uses —
/// Anthropic only accepts a beta tag when the corresponding feature is wired
/// in, so we add them on demand rather than unconditionally.
///
/// # Tags
///
/// - [`BETA_STRUCTURED_OUTPUTS`]: historical structured-output beta, retained for
///   API compatibility; stable format forwarding does not select this tag.
/// - [`BETA_FAST_MODE`]: `speed: "fast"` low-latency mode.
/// - [`BETA_WEB_FETCH`]: hosted `web_search` tool.
/// - [`BETA_CONTEXT_MANAGEMENT`]: automatic context compaction.
pub const BETA_STRUCTURED_OUTPUTS: &str = "structured-outputs-2025-11-13";
/// Beta tag for `speed: "fast"`, selected by [`collect_beta_headers`].
pub const BETA_FAST_MODE: &str = "fast-mode-2026-02-01";
/// Beta tag for the hosted `web_search` tool.
pub const BETA_WEB_FETCH: &str = "web-fetch-2025-09-10";
/// Beta tag for automatic context management / compaction.
pub const BETA_CONTEXT_MANAGEMENT: &str = "context-management-2025-06-27";
/// Beta tag for advanced tool-use features (`tool_search_*` tools,
/// `input_examples`, `allowed_callers`). Anthropic gates these behind a
/// single beta — the same tag covers all three triggers.
pub const BETA_ADVANCED_TOOL_USE: &str = "advanced-tool-use-2025-11-20";
/// Beta tag for the hosted `advisor` tool (`advisor_20260301`).
pub const BETA_ADVISOR_TOOL: &str = "advisor-tool-2026-03-01";
/// Beta tag for the `output_config.effort` parameter on pre-4.6 models.
/// Claude 4.6+ models accept `effort` as a stable API feature and do NOT
/// require this beta.
pub const BETA_EFFORT: &str = "effort-2025-11-24";
/// Beta tag for output configuration attached to individual conversation messages.
pub const BETA_MID_CONVERSATION_OUTPUT_CONFIG: &str = "mid-conversation-output-config-2026-07-01";

/// Historical JSON-helper name. It is no longer reserved: explicitly supplied
/// tools with this name remain ordinary tool calls.
pub const JSON_TOOL_CALL_NAME: &str = "json_tool_call";

/// Cached stream metadata extracted from the `message_start` event.
///
/// Anthropic only sends `id` / `model` on the first event of a stream, but
/// OpenAI clients expect every chunk to carry them. We stash the values here
/// on the first chunk and inject them into every subsequent chunk we emit.
///
/// `cache_*` fields hold the cache token counts reported on `message_start`
/// (where they typically appear). Some Bedrock regions (GovCloud) emit them
/// ONLY on `message_start` and omit them on `message_delta`, so the finish
/// chunk must merge them back in when the delta's usage lacks them.
#[derive(Debug, Clone)]
struct StreamMeta {
    id: String,
    model: String,
    created: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    ephemeral_5m_input_tokens: Option<u64>,
    ephemeral_1h_input_tokens: Option<u64>,
    // M2: prompt token count computed from `message_start` usage
    // (input_tokens + cache_read_input_tokens + cache_creation_input_tokens).
    // Stashed here so the finish chunk (`message_delta`) can emit the real
    // value instead of 0 — the delta's own usage object only carries
    // `output_tokens`.
    prompt_tokens: u64,
    speed: Option<String>,
    service_tier: Option<String>,
}

// Per-thread cache of the most recent Anthropic stream metadata.
//
// `AnthropicProvider` is a unit struct shared via `Arc` and constructed as
// `AnthropicProvider` in other crates, so per-instance state is not an
// option. We keep the latest `message_start` meta here so subsequent chunks
// in the same stream can inherit `id` / `model` / `created`.
//
// # ⚠️ Broken under the multi-threaded tokio runtime
//
// This `thread_local!` is **not safe** for production streaming: the host's
// stream pump (`oxllm-server::orchestration::stream`) runs inside a
// `tokio::spawn` task on the default multi-threaded executor, and tokio is
// free to migrate that task to a different worker thread at every `.await`
// (`byte_stream.next()`, `tx.send().await`, `tokio::time::timeout`). A chunk
// that lands on a new thread observes an *empty* cell, so the OpenAI-shape
// chunk it emits is missing `id` / `model` / `created`.
//
// The production code path therefore calls
// [`AnthropicProvider::transform_stream_chunk_stateful`], which carries the
// per-stream state in a host-owned slot that travels with the task. This
// cell is kept ONLY so the trait-required `transform_stream_chunk` (used by
// unit tests, which run single-threaded) keeps working unchanged.
thread_local! {
    static STREAM_META: RefCell<Option<StreamMeta>> = const { RefCell::new(None) };
}

// Per-thread reverse map (sanitized → original) for tool names.
//
// Anthropic requires tool names to match `^[a-zA-Z0-9_-]{1,128}$`. OpenAI
// callers may pass arbitrary tool names; `transform_request` sanitizes them
// and stashes the reverse mapping here so `transform_response` /
// `transform_stream_chunk` can restore the caller's original name on the
// way back.
//
// # ⚠️ Broken under the multi-threaded tokio runtime
//
// Same hazard as `STREAM_META`: `transform_request` and the response/stream
// transforms run on potentially different worker threads (separated by
// `.await` points), so a thread-local cell can be empty when the reader
// needs it. The production paths use
// [`AnthropicProvider::transform_response_with_request`] /
// [`AnthropicProvider::transform_stream_chunk_stateful`], which rebuild the
// reverse map from the original request into a host-owned state slot. This
// cell is kept for the trait-required methods exercised by single-threaded
// unit tests.
//
// It IS overwritten on every `transform_request` (even when empty — see the
// HIGH issue fix below) so a stale map from a prior request never leaks.
thread_local! {
    static TOOL_NAME_MAP: RefCell<HashMap<String, String>> = RefCell::new(HashMap::new());
}

// Per-thread tracker for the tool_use content blocks currently mid-stream.
//
// Hosted server-tool fragments stay separate from client tools. A client
// tool with no argument deltas still needs a final `arguments: "{}"` delta.
//
// Same scope/lifecycle rationale as `STREAM_META` / `TOOL_NAME_MAP`: a
// `thread_local!` cell, cleared on every `message_start` so a new stream
// never inherits a prior stream's block state.
#[derive(Default, Clone)]
struct ToolBlockState {
    /// True for `server_tool_use` and `*_tool_result` blocks (web_search,
    /// code_execution, etc.). These blocks emit `input_json_delta` fragments
    /// that must NOT be forwarded as OpenAI `tool_calls` argument deltas —
    /// doing so would surface a phantom tool call to OpenAI streaming
    /// clients. The start chunk is also suppressed for these blocks.
    is_server_tool: bool,
    accumulated_args: String,
}

thread_local! {
    static STREAM_TOOL_BLOCKS: RefCell<HashMap<u64, ToolBlockState>> =
        RefCell::new(HashMap::new());
}

/// Host-owned per-stream accumulator for the Anthropic provider.
///
/// Replaces the three `thread_local!` cells (`STREAM_META`,
/// `TOOL_NAME_MAP`, `STREAM_TOOL_BLOCKS`) for the production code path.
/// The host creates one of these per stream (it is `Default`) and threads a
/// `&mut` reference through every `transform_stream_chunk_stateful` call, so
/// the state travels with the stream's `tokio` task across worker-thread
/// migrations instead of being orphaned on the thread that wrote it.
///
/// `reverse_map` is rebuilt lazily from the original request on the first
/// chunk, so the streaming transform does not depend on `transform_request`
/// having run on the same thread.
#[derive(Default)]
pub struct AnthropicStreamState {
    meta: Option<StreamMeta>,
    tool_blocks: HashMap<u64, ToolBlockState>,
    reverse_map: HashMap<String, String>,
}

/// The Anthropic provider. Implements chat completion transforms for the Anthropic Messages API.
#[derive(Debug, Clone, Default)]
pub struct AnthropicProvider;

impl ChatProvider for AnthropicProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        crate::util::append_operation_path(base, "messages")
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        if let ResolvedCredentials::PreSignedHeaders(signed) = creds {
            headers.extend(
                signed
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone())),
            );
            headers.insert("anthropic-version".into(), ANTHROPIC_VERSION.into());
            return;
        }
        let key = match creds {
            ResolvedCredentials::ApiKey { bearer } => bearer.clone(),
            other => match other.best_effort_bearer() {
                Some(b) => b.to_string(),
                None => {
                    tracing::warn!(
                        credential_kind = match other {
                            ResolvedCredentials::AwsSigV4 { .. } => "aws_sigv4",
                            ResolvedCredentials::PreSignedHeaders(_) => "pre_signed_headers",
                            // best_effort_bearer returns Some for every other
                            // variant, so this arm is unreachable in practice.
                            _ => "other",
                        },
                        "anthropic deployment resolved a credential with no \
                         bearer token; request will go upstream without \
                         x-api-key and likely be rejected as 401"
                    );
                    return;
                }
            },
        };
        headers.insert("x-api-key".to_string(), key);
        oxllm_core::provider::apply_auth_overrides(deployment, creds, headers);
        headers.insert(
            "anthropic-version".to_string(),
            ANTHROPIC_VERSION.to_string(),
        );
        // NOTE: dynamic `anthropic-beta` headers (feature #1) are computed by
        // [`collect_beta_headers`] from the request body. The `ChatProvider`
        // trait's `apply_auth` signature does not receive the request, so the
        // beta header cannot be injected here — the host must call
        // `collect_beta_headers(request)` and merge the result into the header
        // set alongside the version set above. The function is exposed and
        // unit-tested; the host merges these tags with configured/client betas.
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let cache_request = inject_configured_cache_breakpoints(request, deployment)?;
        let request = cache_request.as_ref().unwrap_or(request);
        let normalized = normalize_legacy_function_request(request)?;
        let request = normalized.as_ref().unwrap_or(request);
        let mut out = Map::new();

        let model = if deployment.oxllm_params.model.contains('*') {
            request
                .get("model")
                .and_then(|m| m.as_str())
                .map(strip_provider_prefix)
                .unwrap_or_else(|| strip_provider_prefix(&deployment.oxllm_params.model))
        } else {
            strip_provider_prefix(&deployment.oxllm_params.model)
        };
        out.insert("model".to_string(), json!(model));

        let (system, messages) = split_system_messages(&request["messages"], model)?;
        if let Some(system) = system {
            out.insert("system".to_string(), system);
        }
        out.insert("messages".to_string(), Value::Array(messages));

        // `max_tokens` / `max_completion_tokens`. Accept JSON integers and
        // floats — some serializers emit integer values as floats (e.g.
        // `4096.0`); `as_u64()` alone drops those silently and falls back to
        // the per-model default, which can inflate the output window. Round
        // floats like the reference (`transformation.py:1427-1430`:
        // `int(round(value))`), saturating to a minimum of 1.
        let max_tokens = request
            .get("max_tokens")
            .and_then(json_number_as_u64)
            .or_else(|| {
                request
                    .get("max_completion_tokens")
                    .and_then(json_number_as_u64)
            })
            .unwrap_or_else(|| default_max_tokens_for_model(model));
        out.insert("max_tokens".to_string(), json!(max_tokens));

        if let Some(temp) = request.get("temperature") {
            if !temp.is_null() {
                out.insert("temperature".to_string(), temp.clone());
            }
        }
        if let Some(top_p) = request.get("top_p") {
            if !top_p.is_null() {
                out.insert("top_p".to_string(), top_p.clone());
            }
        }
        if let Some(stop) = request.get("stop") {
            if !stop.is_null() {
                out.insert("stop_sequences".to_string(), normalize_stop_sequences(stop));
            }
        }
        if let Some(stream) = request.get("stream").and_then(|v| v.as_bool()) {
            out.insert("stream".to_string(), json!(stream));
        }

        // --- speed (feature #2) ---
        //
        // Anthropic accepts `speed: "fast"` (gated behind the `fast-mode`
        // beta header — see `collect_beta_headers`). Any other value is
        // unsupported and silently dropped rather than forwarded (forwarding
        // an unknown value would 400 upstream).
        if request.get("speed").and_then(|v| v.as_str()) == Some("fast") {
            out.insert("speed".to_string(), json!("fast"));
        }

        // Only caller-supplied tools belong in the tool list.
        if request
            .get("tools")
            .is_some_and(|tools| !tools.is_null() && !tools.is_array())
        {
            return Err(oxllm_core::Error::BadRequest(
                "tools must be an array".into(),
            ));
        }
        let raw_tools = tools_from_request(request);

        // --- MCP server tools ---
        //
        // Anthropic connects to MCP servers directly: the client describes
        // them as `type: "url"` (Anthropic-shape) or `type: "mcp"`
        // (OpenAI-shape) entries inside `tools[]`, and we lift them out into
        // a top-level `mcp_servers` array on the Anthropic body. The remaining
        // (function) tools stay in `tools[]` and flow through the standard
        // `map_tool_to_anthropic` path below. Mirrors the reference
        // `_map_tool_helper` / `_map_openai_mcp_server_tool`.
        let (mut tools, mut mcp_servers) = extract_mcp_servers(&raw_tools);

        // A top-level `mcp_servers` field on the request body is an
        // alternative way to specify MCP servers — pass it through directly,
        // appended after any servers extracted from `tools[]`.
        if let Some(top) = request.get("mcp_servers").and_then(|v| v.as_array()) {
            mcp_servers.extend(top.iter().cloned());
        }

        // `x-mcp-authorization` is a bearer token the host may place onto the
        // request body when forwarding the like-named HTTP header (the
        // `ChatProvider::transform_request` signature has no header bag).
        // Inject it into every MCP server entry that does not already carry
        // its own `authorization_token`.
        if let Some(auth) = request.get("x-mcp-authorization").and_then(|v| v.as_str()) {
            for server in mcp_servers.iter_mut() {
                if let Value::Object(map) = server {
                    map.entry("authorization_token".to_string())
                        .or_insert_with(|| json!(auth));
                }
            }
        }

        // Forward schemas natively without a model-name gate or repairs. The
        // selected provider owns model support and schema-subset validation.
        let native_output_format = crate::util::native_output_format(
            request,
            &[
                ("/output_config/format", "output_config.format"),
                ("/output_format", "output_format"),
            ],
        )?;

        // --- web_search_options → hosted web_search tool (feature #3) ---
        //
        // OpenAI callers describe web search via `web_search_options`; Anthropic
        // exposes it as a hosted tool in `tools[]`. Convert and append it to
        // the tool list (it is removed from the body implicitly — `out` is
        // built field-by-field, so `web_search_options` is never copied over).
        // `search_context_size` ("low"/"medium"/"high") maps to `max_uses`
        // (1/3/5); `user_location.approximate` is forwarded when present.
        if let Some(wso) = request
            .get("web_search_options")
            .and_then(|v| v.as_object())
        {
            let mut web_tool = Map::new();
            web_tool.insert("type".to_string(), json!("web_search_20250305"));
            web_tool.insert("name".to_string(), json!("web_search"));
            let max_uses =
                wso.get("search_context_size")
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "low" => 1u64,
                        "medium" => 3,
                        "high" => 5,
                        _ => 3,
                    });
            if let Some(mu) = max_uses {
                web_tool.insert("max_uses".to_string(), json!(mu));
            }
            if let Some(loc) = wso.get("user_location").and_then(|v| v.as_object()) {
                if let Some(approx) = loc.get("approximate").and_then(|v| v.as_object()) {
                    let mut anthropic_loc = Map::new();
                    anthropic_loc.insert("type".to_string(), json!("approximate"));
                    for (k, v) in approx {
                        if k != "type" {
                            anthropic_loc.insert(k.clone(), v.clone());
                        }
                    }
                    web_tool.insert("user_location".to_string(), Value::Object(anthropic_loc));
                }
            }
            tools.push(Value::Object(web_tool));
        }

        // --- Tool name sanitization (Anthropic: ^[a-zA-Z0-9_-]{1,128}$) ---
        //
        // Build forward (original → sanitized) and reverse (sanitized →
        // original) maps. The forward map is applied to `tools[]`,
        // `tool_choice.name`, and `tool_use.name` blocks in assistant
        // messages below; the reverse map is stashed for `transform_response`
        // and `transform_stream_chunk` to restore caller names on the way back.
        let (forward, reverse) = build_tool_name_maps(&tools);
        // Always overwrite the cell, even when empty. A prior request on this
        // thread may have stashed a reverse map; if we skipped the write when
        // `reverse` was empty (a request with no tools), that stale map would
        // be reused by the response transform and could rewrite an unrelated
        // tool name. One HashMap allocation per request is negligible.
        TOOL_NAME_MAP.with(|cell| {
            *cell.borrow_mut() = reverse;
        });

        if !tools.is_empty() {
            let mapped: Vec<Value> = tools
                .iter()
                .map(map_tool_to_anthropic)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            if !mapped.is_empty() {
                out.insert("tools".to_string(), Value::Array(mapped));
            }
        }

        // --- mcp_servers (MCP server tools) ---
        //
        // Only emitted when at least one MCP server was collected from
        // `tools[]` or the top-level `mcp_servers` field — Anthropic rejects
        // an empty array, and omitting the field entirely keeps the request
        // body clean for callers not using MCP.
        if !mcp_servers.is_empty() {
            out.insert("mcp_servers".to_string(), Value::Array(mcp_servers));
        }

        // --- tool_choice ---
        //
        // Preserve the caller's choice; translate API shape and tool names.
        let mut tool_choice_value: Option<Value> = request
            .get("tool_choice")
            .and_then(map_tool_choice_to_anthropic);
        if request
            .get("tool_choice")
            .is_some_and(|value| !value.is_null())
            && tool_choice_value.is_none()
        {
            return Err(oxllm_core::Error::BadRequest("tool choice has no lossless native Anthropic mapping; use auto, required, none or a named function".into()));
        }

        // Names are encoded together once in the final forward-map pass.
        // Encoding a choice here too can select another colliding tool name.
        // The flag belongs to tool_choice.auto/any/tool, never the request
        // root or tool_choice.none. Omitting tool_choice with tools means auto.
        if request.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false)
            && out.get("tools").is_some()
        {
            let choice = tool_choice_value.get_or_insert_with(|| json!({"type": "auto"}));
            if let Some(tc_map) = choice.as_object_mut() {
                if matches!(
                    tc_map.get("type").and_then(Value::as_str),
                    Some("auto" | "any" | "tool")
                ) {
                    if tc_map
                        .get("disable_parallel_tool_use")
                        .is_some_and(|value| !value.is_null() && value.as_bool() != Some(true))
                    {
                        return Err(oxllm_core::Error::BadRequest("parallel_tool_calls false conflicts with native tool_choice.disable_parallel_tool_use".into()));
                    }
                    tc_map.insert("disable_parallel_tool_use".to_string(), json!(true));
                }
            }
        }

        if let Some(tc) = tool_choice_value {
            out.insert("tool_choice".to_string(), tc);
        }

        // A caller-supplied top-level `output_config.effort` is preserved
        // here too, normalised per-model (e.g. Opus 4.5 `max`→`high`). When
        // both `native_output_format` and a caller `output_config.effort` are
        // present, the format and effort are merged into a single
        // `output_config` object so neither is lost.
        let caller_output_config_effort: Option<Value> = request
            .get("output_config")
            .and_then(|oc| oc.get("effort"))
            .and_then(|e| e.as_str())
            .and_then(|e| normalize_effort_for_model(model, e))
            .map(Value::String);
        // Preserve independent caller keys. Format aliases have already been
        // checked for conflicts; effort goes through model normalization.
        let mut output_config = request
            .get("output_config")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        output_config.remove("effort");
        output_config.remove("format");
        if let Some(format) = native_output_format {
            output_config.insert("format".into(), format);
        }
        if let Some(effort) = caller_output_config_effort {
            output_config.insert("effort".into(), effort);
        }
        if !output_config.is_empty() {
            out.insert("output_config".into(), Value::Object(output_config));
        }

        // --- context_management → edits (M3) ---
        //
        // OpenAI callers describe context compaction as a list:
        // `[{"type": "compaction", "compact_threshold": N}]`. Anthropic's
        // native format is an `edits` array: `[{"type": "compact_20260112",
        // "trigger": {"type": "input_tokens", "value": N}}]`. Convert each
        // entry and set the `edits` field on the Anthropic body. The beta
        // header `context-management-2025-06-27` is added via
        // `collect_beta_headers`.
        if let Some(cm) = request.get("context_management") {
            if !cm.is_null() {
                if let Some(edits) = map_context_management_to_edits(cm) {
                    out.insert("edits".to_string(), Value::Array(edits));
                }
            }
        }

        if let Some(thinking) = request.get("thinking").filter(|value| !value.is_null()) {
            // Retain budget clamping when a valid window exists, but never
            // silently discard an explicit request for thinking.
            let clamped =
                clamp_thinking_budget(thinking, max_tokens).ok_or_else(thinking_budget_error)?;
            out.insert("thinking".to_string(), clamped);
        } else if let Some(effort) = request.get("reasoning_effort").and_then(|v| v.as_str()) {
            // Map OpenAI `reasoning_effort` to Anthropic thinking config when
            // the caller didn't set `thinking` directly (feature #4). For
            // pre-4.6 models this emits `{"type": "enabled", "budget_tokens":
            // N}` with budgets matching the reference
            // `DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET` constants
            // (`constants.py:183-189`); `none` skips thinking entirely.
            // For adaptive-thinking models (4.6+/5+) the model self-sizes its
            // budget, so we emit `{"type": "adaptive"}` and additionally set
            // `output_config.effort` (the stable API) — mirroring the reference
            // `_map_reasoning_effort` + `REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT`.
            if is_known_effort(effort) && effort != "none" {
                if is_adaptive_thinking_model(model) {
                    let thinking = json!({"type": "adaptive"});
                    out.insert("thinking".to_string(), thinking);
                    if let Some(mapped) = reasoning_effort_to_output_config_effort(effort) {
                        let cfg = out
                            .entry("output_config".to_string())
                            .or_insert_with(|| Value::Object(Map::new()));
                        if let Value::Object(m) = cfg {
                            m.insert("effort".to_string(), json!(mapped));
                        }
                    }
                } else {
                    let budget = reasoning_effort_budget(effort);
                    let thinking = json!({"type": "enabled", "budget_tokens": budget});
                    // Clamp the derived budget too (e.g. `max` with a small
                    // `max_tokens`).
                    if let Some(clamped) = clamp_thinking_budget(&thinking, max_tokens) {
                        out.insert("thinking".to_string(), clamped);
                    }
                }
            }
        }

        for key in &["top_k", "metadata"] {
            if let Some(val) = request.get(*key) {
                if !val.is_null() {
                    out.insert(key.to_string(), val.clone());
                }
            }
        }

        // `service_tier`: mapped onto Anthropic's tier for the routes that
        // accept it (direct Anthropic and Foundry Claude — reference
        // `SERVICE_TIER_PROVIDERS`). OpenAI's `default` becomes
        // `standard_only`; unmappable values are client errors with a
        // drop_params hint — and because the field is not advertised in
        // `supported_params`, enabling drop_params strips it before this
        // transform, mirroring the reference's drop-with-warning behavior.
        // Vertex Claude and Bedrock InvokeModel drop it — only the Anthropic
        // API routes accept it.
        if let Some(tier) = request
            .get("service_tier")
            .filter(|value| !value.is_null())
            .and_then(Value::as_str)
        {
            match deployment.provider_slug() {
                "anthropic" | "azure_ai" => {
                    let mapped = match tier {
                        "auto" => "auto",
                        "default" | "standard_only" => "standard_only",
                        other => {
                            return Err(oxllm_core::Error::BadRequest(format!(
                                "{model} does not support service_tier='{other}'. Anthropic accepts \
                                 'auto', 'standard_only', or OpenAI's 'default' (sent as \
                                 'standard_only'). To drop unsupported params, set drop_params = true."
                            )));
                        }
                    };
                    out.insert("service_tier".to_string(), json!(mapped));
                }
                _ => {}
            }
        }

        // `user` → `metadata.user_id` (merge into whatever metadata block we
        // already have, so caller-supplied metadata keys survive alongside
        // the OpenAI-standard user id).
        if let Some(user) = request.get("user").and_then(|v| v.as_str()) {
            let metadata = out
                .entry("metadata".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(map) = metadata {
                map.insert("user_id".to_string(), json!(user));
            }
        }

        // Apply the forward map post-hoc to tools/tool_choice/assistant
        // tool_use names. Identity-only maps (the common case — every tool
        // name already valid) short-circuit here so we don't churn `out`.
        if forward.iter().any(|(k, v)| k != v) {
            apply_tool_name_forward_map(&mut out, &forward);
        }

        // Extended thinking, including adaptive thinking, accepts only
        // temperature 1. Any other value is a 400, so drop it and let the
        // API default apply. temperature 1 is kept.
        drop_temperature_incompatible_with_thinking(&mut out);

        Ok(Value::Object(out))
    }

    fn transform_response_with_request(
        &self,
        body: &Value,
        model: &str,
        original_request: &Value,
    ) -> Result<Value> {
        // Rebuild the reverse tool-name map from the original request and seed
        // it into the per-thread cell right here on the calling thread. There
        // is no `.await` between this write and the read inside
        // `transform_response` (both run synchronously on this thread), so the
        // cell is safe for this one-shot non-streaming call — unlike the
        // streaming path, which cannot rely on it across tokio task
        // migrations and therefore uses `AnthropicStreamState`.
        let reverse_map = reverse_map_from_request(original_request);
        TOOL_NAME_MAP.with(|cell| *cell.borrow_mut() = reverse_map);
        self.transform_response(body, model)
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let reverse_map = TOOL_NAME_MAP.with(|c| c.borrow().clone());
        let id = body
            .get("id")
            .cloned()
            .unwrap_or_else(|| json!("chatcmpl-anthropic"));
        let model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let (
            content,
            mut tool_calls,
            reasoning_content,
            thinking_blocks,
            citations,
            guardrail_blocks,
            web_search_results,
            tool_results,
            compaction_blocks,
        ) = extract_assistant_content(&body["content"]);

        // Restore caller tool names: `transform_request` may have sanitized
        // them to fit Anthropic's `[a-zA-Z0-9_-]{1,128}` rule. Look each name
        // up in the reverse map and rewrite in place.
        if !tool_calls.is_empty() {
            reverse_map_tool_call_names_in(&mut tool_calls, &reverse_map);
        }

        let mut message = Map::new();
        message.insert("role".to_string(), json!("assistant"));

        message.insert("content".to_string(), Value::String(content));

        if !reasoning_content.is_empty() {
            // DeepSeek/Grok use `reasoning_content` for the model's chain of
            // thought text; clients that consume that field get thinking
            // content transparently across providers.
            message.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content),
            );
        }
        if !thinking_blocks.is_empty() {
            // Preserve the raw Anthropic thinking blocks (with `signature`
            // and any other fields) so callers that want the full structured
            // data — e.g. to feed back into a multi-turn conversation — can
            // access them without losing information.
            message.insert("thinking_blocks".to_string(), Value::Array(thinking_blocks));
        }
        if !tool_calls.is_empty() {
            // Text and thinking blocks are not members of the tool-call array.
            for (index, call) in tool_calls.iter_mut().enumerate() {
                if let Some(call) = call.as_object_mut() {
                    call.insert("index".into(), json!(index));
                }
            }
            message.insert("tool_calls".to_string(), Value::Array(tool_calls));
        }
        if !citations.is_empty() {
            // Surface the collected citations (from all text blocks) as a
            // top-level field on the message so callers consuming the OpenAI
            // shape can access them without re-parsing Anthropic content
            // blocks.
            message.insert("citations".to_string(), Value::Array(citations));
        }
        // M11: Guardrail intervention — preserve the raw guarded_content blocks
        // (with permissions/metadata) and flag the intervention so callers can
        // detect content-filtered responses.
        if !guardrail_blocks.is_empty() {
            message.insert("guardrail_intervention".to_string(), json!(true));
            message.insert(
                "guardrail_blocks".to_string(),
                Value::Array(guardrail_blocks),
            );
        }
        // Hosted web-search results: preserve the raw `web_search_tool_result`
        // / `web_fetch_tool_result` blocks so callers can replay the
        // `encrypted_content` of each result in the next turn. The results are
        // opaque to us, so we surface them verbatim under a dedicated
        // provider-extension field.
        if !web_search_results.is_empty() {
            message.insert(
                "web_search_results".to_string(),
                Value::Array(web_search_results),
            );
        }
        // Code-execution tool results (`bash_code_execution_tool_result`,
        // `text_editor_code_execution_tool_result`, …): surface verbatim as
        // `code_interpreter_results` so callers can read stdout / stderr /
        // exit_code without re-parsing the Anthropic-specific shape.
        if !tool_results.is_empty() {
            message.insert(
                "code_interpreter_results".to_string(),
                Value::Array(tool_results),
            );
        }
        // Compaction blocks: emitted when automatic context management
        // triggers. Preserve the raw compacted context for potential replay
        // in follow-up requests.
        if !compaction_blocks.is_empty() {
            message.insert(
                "compaction_blocks".to_string(),
                Value::Array(compaction_blocks),
            );
        }

        let finish_reason = body
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(map_stop_reason_to_openai)
            .unwrap_or_else(|| json!("stop"));
        let usage = body.get("usage");
        let input_tokens = usage
            .and_then(|u| u.get("input_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let completion_tokens = usage
            .and_then(|u| u.get("output_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_read = usage
            .and_then(|u| u.get("cache_read_input_tokens"))
            .and_then(|v| v.as_u64());
        let cache_creation = usage
            .and_then(|u| u.get("cache_creation_input_tokens"))
            .and_then(|v| v.as_u64());
        // Reasoning (thinking) tokens for thinking models. Anthropic reports
        // these under `usage.completion_tokens_details.reasoning_tokens`
        // (OpenAI-compatible location); mirror them into the OpenAI output
        // usage so downstream extractors see them.
        let reasoning_tokens = usage.and_then(anthropic_reasoning_tokens);
        // Anthropic's `input_tokens` EXCLUDES both cache-read hits and
        // cache-creation (write) tokens, whereas the OpenAI usage shape (and
        // `calculate_cost`) expects cache tokens to be a subset of
        // `prompt_tokens`. Fold BOTH `cache_read_input_tokens` and
        // `cache_creation_input_tokens` into `prompt_tokens` (the reference
        // convention — and the convention the Bedrock Converse transform
        // already uses) so the normalised response obeys the
        // `cache_read ⊆ prompt_tokens` invariant and the same model reports
        // identical `prompt_tokens` regardless of which surface served it;
        // the mirrors into `prompt_tokens_details.cached_tokens` /
        // `cache_creation_tokens` below keep downstream extraction on the
        // OpenAI branch so the fold never happens twice.
        // Saturating add: a compromised/buggy upstream or test fixture could
        // return values near `u64::MAX`; a plain `+` would panic in debug
        // (denial of service) and silently wrap in release, emitting a bogus
        // `prompt_tokens` on the OpenAI-shape body. Mirror the saturation the
        // host-side extractor (`usage.rs`) already applies.
        let prompt_tokens = input_tokens
            .saturating_add(cache_read.unwrap_or(0))
            .saturating_add(cache_creation.unwrap_or(0));
        let total_tokens = prompt_tokens.saturating_add(completion_tokens);

        let mut choice = Map::new();
        choice.insert("index".to_string(), json!(0));
        choice.insert("message".to_string(), Value::Object(message));
        choice.insert("finish_reason".to_string(), finish_reason);

        let mut out = Map::new();
        out.insert("id".to_string(), id);
        out.insert("object".to_string(), json!("chat.completion"));
        out.insert("created".to_string(), json!(unix_now()));
        out.insert("model".to_string(), Value::String(model));
        out.insert("choices".to_string(), json!([Value::Object(choice)]));
        let mut usage_obj = Map::new();
        usage_obj.insert("prompt_tokens".to_string(), json!(prompt_tokens));
        usage_obj.insert("completion_tokens".to_string(), json!(completion_tokens));
        usage_obj.insert("total_tokens".to_string(), json!(total_tokens));
        if let Some(v) = cache_read {
            usage_obj.insert("cache_read_input_tokens".to_string(), json!(v));
        }
        // NOTE: cache-creation is deliberately NOT mirrored at the top level
        // (unlike cache-read above). The host usage extractor
        // (`orchestration/usage.rs`) treats a top-level
        // `cache_creation_input_tokens` as ADDITIVE — i.e. outside
        // `prompt_tokens` — and bills it at the cache-write rate on top of
        // the prompt tokens. Since `prompt_tokens` here already folds the
        // cache-creation tokens in, also emitting the top-level form would
        // bill every cache-write token twice (once at the input rate inside
        // `prompt_tokens`, once at the cache-creation rate). The
        // `prompt_tokens_details.cache_creation_tokens` subset mirror below
        // is the only form emitted, so the extractor subtracts it and the
        // fold happens exactly once.
        // Mirror reasoning_tokens into the OpenAI-standard
        // `completion_tokens_details.reasoning_tokens` path so downstream
        // consumers (cost, observability) can read them without parsing the
        // Anthropic-specific shape. Also enrich with `text_tokens`
        // (feature #5) = completion - reasoning, the non-reasoning output
        // token count.
        if let Some(v) = reasoning_tokens {
            let details = usage_obj
                .entry("completion_tokens_details".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(details_map) = details {
                details_map.insert("reasoning_tokens".to_string(), json!(v));
                let text = completion_tokens.saturating_sub(v);
                details_map.insert("text_tokens".to_string(), json!(text));
            }
        }
        // P4: Always populate `prompt_tokens_details.text_tokens` with the
        // fresh (non-cached, non-audio) input token count. Anthropic's
        // `input_tokens` already excludes cache reads, so `text_tokens =
        // input_tokens`. Then add cache counters under the same details
        // object when present. This matches the reference `raw_input_tokens =
        // prompt_tokens - cache_read - cache_creation` which equals our
        // `input_tokens`.
        let cache_creation_buckets = usage
            .and_then(|u| u.get("cache_creation"))
            .and_then(|c| c.as_object());
        // OpenAI clients expect the cached-prompt token count under the
        // standard `prompt_tokens_details` path. Mirror both
        // `cache_read_input_tokens` → `cached_tokens` (OpenAI standard) and
        // `cache_creation_input_tokens` → `cache_creation_tokens`
        // (provider-extension) so downstream extractors can read either
        // number from the OpenAI-shape location while the Anthropic-specific
        // top-level fields stay for backward compatibility.
        {
            let details = usage_obj
                .entry("prompt_tokens_details".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(details_map) = details {
                details_map.insert("text_tokens".to_string(), json!(input_tokens));
                if let Some(v) = cache_read {
                    details_map.insert("cached_tokens".to_string(), json!(v));
                }
                if let Some(v) = cache_creation {
                    details_map.insert("cache_creation_tokens".to_string(), json!(v));
                }
                if let Some(buckets) = cache_creation_buckets {
                    for k in ["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"] {
                        if let Some(v) = buckets.get(k).and_then(|v| v.as_u64()) {
                            details_map.insert(k.to_string(), json!(v));
                        }
                    }
                }
            }
        }
        // Server tool use usage: Anthropic reports hosted-tool invocation
        // counts under `usage.server_tool_use` (web_search_requests,
        // tool_search_requests). Mirror the structure verbatim so downstream
        // observability can read the same shape as the Anthropic-native
        // response without re-parsing.
        if let Some(Value::Object(stu_map)) = usage
            .and_then(|u| u.get("server_tool_use"))
            .filter(|v| v.is_object())
        {
            let mut copy = Map::new();
            for (k, v) in stu_map {
                if !v.is_null() {
                    copy.insert(k.clone(), v.clone());
                }
            }
            if !copy.is_empty() {
                usage_obj.insert("server_tool_use".to_string(), Value::Object(copy));
            }
        }
        out.insert("usage".to_string(), Value::Object(usage_obj));
        for field in ["speed", "service_tier"] {
            if let Some(value) = body.get(field).filter(|value| value.is_string()) {
                out.insert(field.into(), value.clone());
            }
        }

        Ok(Value::Object(out))
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        // `thread_local!`-backed path. Used by the trait-required method that
        // single-threaded unit tests call directly. The production host calls
        // `transform_stream_chunk_stateful` instead (see below) so the state
        // travels with the tokio task across worker-thread migrations.
        STREAM_META.with(|meta_cell| {
            STREAM_TOOL_BLOCKS.with(|blocks_cell| {
                TOOL_NAME_MAP.with(|map_cell| {
                    let mut meta_guard = meta_cell.borrow_mut();
                    let mut blocks_guard = blocks_cell.borrow_mut();
                    let map_guard = map_cell.borrow();
                    transform_stream_chunk_impl(
                        chunk,
                        &mut meta_guard,
                        &mut blocks_guard,
                        &map_guard,
                    )
                })
            })
        })
    }

    fn transform_stream_chunk_stateful(
        &self,
        chunk: &str,
        state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
        original_request: &Value,
    ) -> Result<Option<Value>> {
        // Host-owned state path: the per-stream accumulator lives in a
        // `Box<dyn Any>` the host created and threads through every chunk, so
        // it follows the tokio task across worker-thread migrations (unlike a
        // `thread_local!`). The reverse map is rebuilt once from the original
        // request on the first chunk, so this does not depend on
        // `transform_request` having run on the same thread.
        let s = state.get_or_insert_with(|| {
            Box::new(AnthropicStreamState {
                reverse_map: reverse_map_from_request(original_request),
                ..Default::default()
            })
        });
        // A foreign state type means the host mixed accumulators across
        // providers — a wiring bug, not a poisoned stream. Return an error
        // (drops this stream) instead of panicking the worker thread.
        let Some(s) = s.downcast_mut::<AnthropicStreamState>() else {
            return Err(oxllm_core::Error::Transform(
                "anthropic stream state type mismatch".into(),
            ));
        };
        transform_stream_chunk_impl(chunk, &mut s.meta, &mut s.tool_blocks, &s.reverse_map)
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
            "stop",
            "stream",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "thinking",
            "metadata",
            // OpenAI-standard knobs the host would otherwise strip via
            // `drop_params` before the provider transform runs. Each one is
            // translated to its Anthropic equivalent inside `transform_request`.
            "response_format",
            "output_config",
            "output_format",
            "parallel_tool_calls",
            "user",
            "reasoning_effort",
            // Feature #2/#3: speed passthrough and web_search_options → hosted
            // tool are honored by `transform_request`, so advertise them so
            // `drop_params` doesn't strip them before the transform runs.
            // `service_tier` is deliberately NOT listed: the host's
            // drop_params strip then removes unmappable values before the
            // transform (the reference's drop-with-warning behavior), while
            // a kept value is mapped or rejected below.
            "speed",
            "web_search_options",
            "context_management",
            // MCP server tools: `tools[]` already covers the `type: "url"` /
            // `type: "mcp"` entries; advertising the top-level field keeps
            // `drop_params` from stripping a caller-supplied `mcp_servers`.
            "mcp_servers",
        ])
    }
}

impl Provider for AnthropicProvider {
    fn id(&self) -> ProviderId {
        "anthropic"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(endpoint, oxllm_core::Endpoint::ChatCompletions)
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

/// Strip a leading `anthropic/` (case-insensitive) provider prefix.
///
/// Operators sometimes copy-paste model ids from cloud-provider portals that
/// title-case the slug (`Anthropic/claude-...`). A case-sensitive match would
/// forward the prefixed id to the Anthropic API, which 400s. Match the prefix
/// case-insensitively, mirroring `oxllm-core::cost`'s handling of `openai/`.
fn strip_provider_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some((prefix, rest)) if prefix.eq_ignore_ascii_case("anthropic") => rest,
        _ => model,
    }
}

/// Detect Anthropic/Bedrock/Vertex 400s caused by a missing or invalid
/// thinking `signature`.
pub fn is_invalid_thinking_signature_error(error_text: &str) -> bool {
    if error_text.is_empty() {
        return false;
    }
    let lower = error_text.to_ascii_lowercase();
    lower.contains("thinking")
        && lower.contains("signature")
        && (lower.contains("invalid") || lower.contains("valid string"))
}

/// Strip `thinking` / `redacted_thinking` content blocks and the top-level
/// `thinking` param so a retry can recover from an invalid signature.
pub fn strip_thinking_blocks_from_request(body: &mut Value) {
    if let Some(m) = body.as_object_mut() {
        m.remove("thinking");
    }
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    let mut kept = Vec::with_capacity(messages.len());
    for msg in messages.drain(..) {
        match msg.get("content") {
            Some(Value::Array(blocks)) => {
                let filtered: Vec<Value> = blocks
                    .iter()
                    .filter(|b| {
                        !matches!(
                            b.get("type").and_then(Value::as_str),
                            Some("thinking") | Some("redacted_thinking")
                        )
                    })
                    .cloned()
                    .collect();
                if filtered.is_empty() {
                    continue;
                }
                let mut msg = msg;
                if let Some(obj) = msg.as_object_mut() {
                    obj.insert("content".to_string(), Value::Array(filtered));
                }
                kept.push(msg);
            }
            _ => kept.push(msg),
        }
    }
    *messages = kept;
}

/// Claude 4.8+ and the 5 family accept a mid-conversation `role: "system"`
/// message in place (Claude Code reminders). Hoisting those mutates the
/// `system` prefix and busts the prompt cache. Older models reject the
/// role in every position, so without the flag every system entry is
/// hoisted.
fn supports_mid_conversation_system(model: &str) -> bool {
    let m = model.to_lowercase();
    if m.contains("claude-5")
        || m.contains("claude_5")
        || m.contains("opus-5")
        || m.contains("sonnet-5")
        || m.contains("haiku-5")
    {
        return true;
    }
    let is_48 = m.contains("4-8") || m.contains("4.8") || m.contains("4_8");
    is_48
        && (m.contains("claude")
            || m.contains("opus")
            || m.contains("sonnet")
            || m.contains("haiku"))
}

fn is_system_role(msg: &Value) -> bool {
    matches!(
        msg.get("role").and_then(Value::as_str),
        Some("system") | Some("developer")
    )
}

struct PendingFunctionCall {
    name: String,
    id: String,
    legacy: bool,
}

/// Normalize legacy definitions, selectors, and history before the native
/// Messages mapping. Modern requests borrow their original body unchanged.
fn normalize_legacy_function_request(request: &Value) -> Result<Option<Value>> {
    let legacy_history = request
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message.get("role").and_then(Value::as_str) == Some("function")
                    || message
                        .get("function_call")
                        .is_some_and(|value| !value.is_null())
            })
        });
    if !legacy_history
        && request.get("functions").is_none_or(Value::is_null)
        && request.get("function_call").is_none_or(Value::is_null)
    {
        return Ok(None);
    }
    let mut normalized = request.clone();
    let body = normalized.as_object_mut().ok_or_else(|| {
        oxllm_core::Error::BadRequest("Anthropic request body must be an object".into())
    })?;
    if let Some(functions) = body.remove("functions").filter(|value| !value.is_null()) {
        let functions = functions.as_array().ok_or_else(|| {
            oxllm_core::Error::BadRequest("legacy functions must be an array".into())
        })?;
        if body
            .get("tools")
            .is_some_and(|tools| !tools.is_null() && tools != &json!([]))
        {
            return Err(oxllm_core::Error::BadRequest(
                "cannot combine legacy functions with tools; use one definition format".into(),
            ));
        }
        let mut tools = Vec::with_capacity(functions.len());
        for function in functions {
            if function
                .get("name")
                .and_then(Value::as_str)
                .is_none_or(|name| name.trim().is_empty())
                || !function.is_object()
                || function
                    .get("parameters")
                    .is_some_and(|schema| !schema.is_object())
            {
                return Err(oxllm_core::Error::BadRequest(
                    "legacy functions require a non-empty name and an object parameter schema"
                        .into(),
                ));
            }
            tools.push(json!({"type": "function", "function": function}));
        }
        body.insert("tools".into(), Value::Array(tools));
    }
    if let Some(choice) = body
        .remove("function_call")
        .filter(|value| !value.is_null())
    {
        if body
            .get("tool_choice")
            .is_some_and(|value| !value.is_null())
        {
            return Err(oxllm_core::Error::BadRequest(
                "cannot combine legacy function_call with tool_choice".into(),
            ));
        }
        let choice = match &choice {
            Value::String(kind) if matches!(kind.as_str(), "auto" | "none") => choice.clone(),
            Value::Object(function) => {
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.trim().is_empty())
                    .ok_or_else(|| {
                        oxllm_core::Error::BadRequest(
                            "legacy function_call requires a non-empty name".into(),
                        )
                    })?;
                if !body
                    .get("tools")
                    .and_then(Value::as_array)
                    .is_some_and(|tools| {
                        tools.iter().any(|tool| {
                            tool.pointer("/function/name").and_then(Value::as_str) == Some(name)
                        })
                    })
                {
                    return Err(oxllm_core::Error::BadRequest(
                        "legacy function_call must name a supplied function".into(),
                    ));
                }
                json!({"type": "function", "function": {"name": name}})
            }
            _ => {
                return Err(oxllm_core::Error::BadRequest(
                    "unsupported legacy function_call selector".into(),
                ));
            }
        };
        // With no definitions, auto/none both mean ordinary text generation.
        // Anthropic does not need a tool_choice field for that request.
        if body
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| !tools.is_empty())
        {
            body.insert("tool_choice".into(), choice);
        }
    }
    if legacy_history {
        let messages = body
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                oxllm_core::Error::BadRequest(
                    "legacy function history requires a messages array".into(),
                )
            })?;
        let messages = normalize_legacy_function_messages(messages)?;
        body.insert("messages".into(), Value::Array(messages));
    }
    Ok(Some(normalized))
}

fn normalize_legacy_function_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut used_ids = HashSet::new();
    for message in messages {
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    used_ids.insert(id.to_string());
                }
            }
        }
        if let Some(blocks) = message.get("content").and_then(Value::as_array) {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("tool_use")
                    && let Some(id) = block.get("id").and_then(Value::as_str)
                {
                    used_ids.insert(id.to_string());
                }
            }
        }
    }
    let mut pending: Vec<PendingFunctionCall> = Vec::new();
    let mut out = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        if role != "assistant"
            && message
                .get("function_call")
                .is_some_and(|value| !value.is_null())
        {
            return Err(oxllm_core::Error::BadRequest(
                "legacy function_call is only valid on assistant messages".into(),
            ));
        }
        let mut normalized = message.clone();
        match role {
            "assistant" => {
                if pending.iter().any(|call| call.legacy) {
                    return Err(oxllm_core::Error::BadRequest(
                        "legacy assistant function_call has no following matching function result"
                            .into(),
                    ));
                }
                pending.clear();
                if let Some(function) = message
                    .get("function_call")
                    .filter(|value| !value.is_null())
                {
                    if message
                        .get("tool_calls")
                        .is_some_and(|calls| !calls.is_null() && calls != &json!([]))
                        || message
                            .get("content")
                            .and_then(Value::as_array)
                            .is_some_and(|blocks| {
                                blocks.iter().any(|block| {
                                    block.get("type").and_then(Value::as_str) == Some("tool_use")
                                })
                            })
                    {
                        return Err(oxllm_core::Error::BadRequest(
                            "assistant cannot combine legacy function_call with tool_calls/tool_use".into(),
                        ));
                    }
                    let name = function
                        .get("name")
                        .and_then(Value::as_str)
                        .filter(|name| !name.trim().is_empty())
                        .ok_or_else(|| {
                            oxllm_core::Error::BadRequest(
                                "legacy assistant function_call requires a non-empty name".into(),
                            )
                        })?;
                    let input = match function.get("arguments") {
                        None | Some(Value::Null) => json!({}),
                        Some(Value::String(arguments)) if arguments.trim().is_empty() => json!({}),
                        Some(Value::String(arguments)) => serde_json::from_str::<Value>(arguments)
                            .map_err(|_| {
                                oxllm_core::Error::BadRequest(
                                    "legacy function_call arguments must be a JSON object".into(),
                                )
                            })?,
                        Some(Value::Object(input)) => Value::Object(input.clone()),
                        _ => {
                            return Err(oxllm_core::Error::BadRequest(
                                "legacy function_call arguments must be a JSON object".into(),
                            ));
                        }
                    };
                    if !input.is_object() {
                        return Err(oxllm_core::Error::BadRequest(
                            "legacy function_call arguments must be a JSON object".into(),
                        ));
                    }
                    let id = if let Some(id) = function.get("id") {
                        let id =
                            id.as_str()
                                .filter(|id| !id.trim().is_empty())
                                .ok_or_else(|| {
                                    oxllm_core::Error::BadRequest(
                                        "legacy function_call id must be a non-empty string".into(),
                                    )
                                })?;
                        if !used_ids.insert(id.to_string()) {
                            return Err(oxllm_core::Error::BadRequest(
                                "legacy function_call id conflicts with another call".into(),
                            ));
                        }
                        id.to_string()
                    } else {
                        // Stable across retries; reserve all modern/native ids
                        // first so a generated id cannot alias another call.
                        let mut id = format!("toolu_legacy_{index}");
                        while !used_ids.insert(id.clone()) {
                            id.push('_');
                        }
                        id
                    };
                    normalized["tool_calls"] = json!([{
                        "id": id, "type": "function",
                        "function": {"name": name, "arguments": input.to_string()}
                    }]);
                    if let Some(object) = normalized.as_object_mut() {
                        object.remove("function_call");
                    }
                    pending.push(PendingFunctionCall {
                        name: name.into(),
                        id,
                        legacy: true,
                    });
                } else {
                    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                        for call in calls {
                            if let (Some(name), Some(id)) = (
                                call.pointer("/function/name").and_then(Value::as_str),
                                call.get("id")
                                    .and_then(Value::as_str)
                                    .filter(|id| !id.is_empty()),
                            ) {
                                pending.push(PendingFunctionCall {
                                    name: name.into(),
                                    id: id.into(),
                                    legacy: false,
                                });
                            }
                        }
                    }
                    if let Some(blocks) = message.get("content").and_then(Value::as_array) {
                        for block in blocks {
                            if block.get("type").and_then(Value::as_str) == Some("tool_use")
                                && let (Some(name), Some(id)) = (
                                    block.get("name").and_then(Value::as_str),
                                    block
                                        .get("id")
                                        .and_then(Value::as_str)
                                        .filter(|id| !id.is_empty()),
                                )
                            {
                                pending.push(PendingFunctionCall {
                                    name: name.into(),
                                    id: id.into(),
                                    legacy: false,
                                });
                            }
                        }
                    }
                }
            }
            "function" => {
                let name = message
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.trim().is_empty())
                    .ok_or_else(|| {
                        oxllm_core::Error::BadRequest(
                            "legacy function result requires a non-empty name".into(),
                        )
                    })?;
                let explicit_id = match message.get("tool_call_id").filter(|value| !value.is_null())
                {
                    Some(value) => {
                        Some(value.as_str().filter(|id| !id.is_empty()).ok_or_else(|| {
                            oxllm_core::Error::BadRequest(
                                "legacy function result tool_call_id must be a non-empty string"
                                    .into(),
                            )
                        })?)
                    }
                    None => None,
                };
                let matches: Vec<usize> = pending
                    .iter()
                    .enumerate()
                    .filter_map(|(index, call)| {
                        (call.name == name && explicit_id.is_none_or(|id| id == call.id))
                            .then_some(index)
                    })
                    .collect();
                if matches.len() != 1 {
                    return Err(oxllm_core::Error::BadRequest(format!(
                        "legacy function result `{name}` must match exactly one preceding assistant call"
                    )));
                }
                match message.get("content") {
                    None | Some(Value::Null) => normalized["content"] = json!(""),
                    Some(Value::String(_) | Value::Array(_)) => {}
                    Some(_) => {
                        return Err(oxllm_core::Error::BadRequest(
                            "legacy function result content must be text or content blocks".into(),
                        ));
                    }
                }
                let call = pending.remove(matches[0]);
                normalized["role"] = json!("tool");
                normalized["tool_call_id"] = json!(call.id);
                if let Some(object) = normalized.as_object_mut() {
                    object.remove("name");
                }
            }
            "tool" => {
                if let Some(index) = pending.iter().position(|call| {
                    message.get("tool_call_id").and_then(Value::as_str) == Some(call.id.as_str())
                }) {
                    pending.remove(index);
                } else if pending.iter().any(|call| call.legacy) {
                    return Err(oxllm_core::Error::BadRequest(
                        "tool result does not match the legacy function_call id".into(),
                    ));
                }
            }
            _ => {
                if pending.iter().any(|call| call.legacy) {
                    return Err(oxllm_core::Error::BadRequest(
                        "legacy assistant function_call must be followed by its matching function result".into(),
                    ));
                }
                pending.clear();
            }
        }
        out.push(normalized);
    }
    if pending.iter().any(|call| call.legacy) {
        return Err(oxllm_core::Error::BadRequest(
            "legacy assistant function_call has no matching function result".into(),
        ));
    }
    Ok(out)
}

fn tools_from_request(request: &Value) -> Vec<Value> {
    if let Some(tools) = request.get("tools").and_then(Value::as_array)
        && !tools.is_empty()
    {
        return tools.clone();
    }
    request
        .get("functions")
        .and_then(Value::as_array)
        .map(|functions| {
            functions
                .iter()
                .map(|function| json!({"type": "function", "function": function}))
                .collect()
        })
        .unwrap_or_default()
}

const EPHEMERAL_CACHE_CONTROL: fn() -> Value = || json!({"type": "ephemeral"});

/// Mark source messages before normalization changes their indices or splits
/// out system blocks. Preserve caller controls, skip missing targets, and
/// count existing tool/system/message marks against the four-block limit.
fn inject_configured_cache_breakpoints(
    request: &Value,
    deployment: &Deployment,
) -> Result<Option<Value>> {
    let Some(points) = deployment
        .oxllm_params
        .cache_control_injection_points
        .as_deref()
    else {
        return Ok(None);
    };
    if points.is_empty() {
        return Ok(None);
    }
    let mut marked = request.clone();
    let mut used = count_cache_breakpoints(&marked);
    for point in points {
        point
            .validate()
            .map_err(|err| oxllm_core::Error::BadRequest(err.to_string()))?;
        if used >= 4 {
            break;
        }
        let Some(messages) = marked.get_mut("messages").and_then(Value::as_array_mut) else {
            continue;
        };
        let target = point
            .index
            .and_then(|index| resolve_message_index(messages.len(), index));
        for (index, message) in messages.iter_mut().enumerate() {
            let role = message.get("role").and_then(Value::as_str);
            let matches = if point.index.is_some() {
                target == Some(index)
            } else {
                point.role.as_deref() == role
                    || (point.role.as_deref() == Some("system") && role == Some("developer"))
            };
            if !matches
                || used >= 4
                || has_cache_control(message)
                || content_has_cache_control(&message["content"])
            {
                continue;
            }
            if let Some(content) = message.get("content").cloned() {
                if role == Some("tool") && content.is_string() {
                    message["cache_control"] = EPHEMERAL_CACHE_CONTROL();
                    used += 1;
                } else {
                    let content = with_cache_control(content);
                    used += cache_control_count(&content);
                    message["content"] = content;
                }
            }
        }
    }
    Ok(Some(marked))
}

fn resolve_message_index(len: usize, index: i64) -> Option<usize> {
    if index >= 0 {
        usize::try_from(index).ok().filter(|index| *index < len)
    } else {
        len.checked_sub(usize::try_from(index.unsigned_abs()).ok()?)
    }
}

fn content_has_cache_control(content: &Value) -> bool {
    cache_control_count(content) != 0
}

fn has_cache_control(block: &Value) -> bool {
    block
        .get("cache_control")
        .is_some_and(|value| !value.is_null())
}

fn cache_control_count(content: &Value) -> usize {
    content
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|block| has_cache_control(block))
                .count()
        })
        .unwrap_or_else(|| usize::from(has_cache_control(content)))
}

fn count_cache_breakpoints(request: &Value) -> usize {
    usize::from(has_cache_control(request))
        + cache_control_count(&request["system"])
        + request["messages"]
            .as_array()
            .map(|messages| {
                messages
                    .iter()
                    .map(|message| {
                        usize::from(has_cache_control(message))
                            + cache_control_count(&message["content"])
                    })
                    .sum::<usize>()
            })
            .unwrap_or(0)
        + request["tools"]
            .as_array()
            .map(|tools| {
                tools
                    .iter()
                    .filter(|tool| has_cache_control(tool) || has_cache_control(&tool["function"]))
                    .count()
            })
            .unwrap_or(0)
}

fn with_cache_control(content: Value) -> Value {
    match content {
        Value::String(text) => json!([{
            "type": "text",
            "text": text,
            "cache_control": EPHEMERAL_CACHE_CONTROL(),
        }]),
        Value::Array(mut blocks) => {
            if let Some(Value::Object(last)) = blocks.last_mut() {
                last.entry("cache_control".to_string())
                    .or_insert_with(EPHEMERAL_CACHE_CONTROL);
            }
            Value::Array(blocks)
        }
        other => other,
    }
}

fn split_system_messages(messages: &Value, model: &str) -> Result<(Option<Value>, Vec<Value>)> {
    let Some(arr) = messages.as_array() else {
        return Ok((None, Vec::new()));
    };
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut out: Vec<Value> = Vec::new();
    let keep_midturn = supports_mid_conversation_system(model);
    let mut past_leading = false;
    // Consecutive user/tool messages form one conversational turn. An empty
    // user message beside siblings that render sends nothing extra; only a
    // run that renders nothing is kept alive with one placeholder turn,
    // carrying the LAST empty user message's cache_control (reference
    // `_empty_user_turn_placeholder`). Runs end at assistant or kept
    // mid-conversation system messages.
    let mut run: Vec<(&Value, Option<Value>)> = Vec::new();
    for msg in arr {
        let in_user_run = matches!(
            msg.get("role").and_then(Value::as_str),
            Some("user" | "tool")
        );
        if !in_user_run && !run.is_empty() {
            flush_user_run(&mut run, &mut out);
        }
        if is_system_role(msg) && !(keep_midturn && past_leading) {
            if let Some(content) = coerce_message_content(msg) {
                system_blocks.push(content);
            }
            continue;
        }
        past_leading = true;
        if in_user_run {
            run.push((msg, normalize_message(msg)?));
        } else {
            if let Some(message) = normalize_message(msg)? {
                out.push(message);
            }
        }
    }
    flush_user_run(&mut run, &mut out);
    if system_blocks.is_empty() {
        return Ok((None, out));
    }
    let any_structured = system_blocks.iter().any(|v| v.is_array());
    if system_blocks.len() == 1 {
        let block = system_blocks.into_iter().next().unwrap_or(Value::Null);
        if any_structured {
            Ok((Some(block), out))
        } else {
            // `coerce_message_content` only returns `String` or `Array`; the
            // single-string system message collapses back to a bare string.
            let text = match block {
                Value::String(s) => s,
                _ => String::new(),
            };
            Ok((Some(Value::String(text)), out))
        }
    } else {
        let blocks: Vec<Value> = system_blocks
            .into_iter()
            .flat_map(|v| match v {
                Value::String(s) => vec![json!({"type": "text", "text": s})],
                Value::Array(items) => items,
                other => vec![other],
            })
            .collect();
        Ok((Some(Value::Array(blocks)), out))
    }
}

/// Flush one conversational run of user/tool messages. Messages that render
/// nothing are dropped when a sibling renders; a run that renders nothing at
/// all keeps exactly one placeholder user turn.
fn flush_user_run(run: &mut Vec<(&Value, Option<Value>)>, out: &mut Vec<Value>) {
    if run.is_empty() {
        return;
    }
    if run.iter().any(|(_, normalized)| normalized.is_some()) {
        for (_, normalized) in run.drain(..) {
            if let Some(message) = normalized {
                out.push(message);
            }
        }
        return;
    }
    let cache_control = run
        .iter()
        .rev()
        .find(|(message, _)| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|(message, _)| message.get("cache_control"))
        .cloned();
    for _ in run.drain(..) {}
    out.push(json!({
        "role": "user",
        "content": [placeholder_text_block(cache_control)],
    }));
}

fn coerce_message_content(msg: &Value) -> Option<Value> {
    if let Some(control) = msg.get("cache_control").filter(|value| !value.is_null()) {
        if let Some(Value::String(text)) = msg.get("content") {
            return Some(json!([{"type": "text", "text": text, "cache_control": control}]));
        }
    }
    match msg.get("content") {
        Some(Value::String(s)) => Some(Value::String(s.clone())),
        Some(Value::Array(arr)) => {
            let has_cache_control = arr.iter().any(|b| b.get("cache_control").is_some());
            if has_cache_control {
                Some(Value::Array(arr.clone()))
            } else {
                let joined: Vec<String> = arr
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()).map(String::from))
                    .collect();
                if joined.is_empty() {
                    None
                } else {
                    Some(Value::String(joined.join("")))
                }
            }
        }
        _ => None,
    }
}

/// Placeholder text for empty message content. Anthropic rejects messages
/// whose content renders nothing, so empty or whitespace-only text is
/// rewritten to this marker instead of silently dropping the turn (reference
/// `_EMPTY_TEXT_PLACEHOLDER`).
const EMPTY_TEXT_PLACEHOLDER: &str =
    "[System: Empty message content sanitised to satisfy protocol]";

/// A placeholder text block, optionally preserving the original block's
/// `cache_control`.
fn placeholder_text_block(cache_control: Option<Value>) -> Value {
    let mut block = json!({"type": "text", "text": EMPTY_TEXT_PLACEHOLDER});
    if let Some(control) = cache_control.filter(|value| !value.is_null()) {
        block["cache_control"] = control;
    }
    block
}

/// Convert a single OpenAI content block into the Anthropic Messages shape.
///
/// `image_url` blocks are rewritten into Anthropic `image` blocks:
///
/// * `data:<media-type>;base64,<payload>` →
///   `{"type":"image","source":{"type":"base64","media_type":…,"data":…}}`
///   (reuses the shared Bedrock helper so both paths agree on the shape).
/// * `http(s)://…` →
///   `{"type":"image","source":{"type":"url","url":…}}` (Anthropic accepts
///   URL sources directly).
///
/// Any other block type (or an `image_url` whose URL we could not convert)
/// is passed through unchanged so non-image content flows unaffected.
fn convert_content_block_to_anthropic(block: &Value) -> Value {
    if block.get("type").and_then(|v| v.as_str()) == Some("image_url") {
        let url = block
            .get("image_url")
            .and_then(|iu| iu.get("url"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if let Some(converted) = crate::util::data_url_to_anthropic_image(url) {
            return converted;
        }
        if is_http_url(url) {
            return http_url_to_anthropic_image(url);
        }
    }
    block.clone()
}

/// Build an Anthropic Messages image block from an HTTP(S) URL. Anthropic
/// accepts URL sources directly (no server-side fetch required).
fn http_url_to_anthropic_image(url: &str) -> Value {
    json!({
        "type": "image",
        "source": {
            "type": "url",
            "url": url
        }
    })
}

/// Does this URL use an `http` or `https` scheme?
fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Extended thinking, including adaptive thinking, accepts only temperature 1.
/// Drop any other value when thinking is enabled or adaptive, or when
/// `output_config.effort` is set. Disabled or absent thinking keeps the
/// caller's temperature.
fn drop_temperature_incompatible_with_thinking(out: &mut Map<String, Value>) {
    let temperature_ok = out
        .get("temperature")
        .is_none_or(|temp| temp.as_f64() == Some(1.0));
    if temperature_ok {
        return;
    }
    let thinking_on = out
        .get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "enabled" || kind == "adaptive");
    let effort_set = out
        .get("output_config")
        .and_then(|cfg| cfg.get("effort"))
        .is_some_and(|effort| !effort.is_null());
    if thinking_on || effort_set {
        out.remove("temperature");
    }
}

fn normalize_message(msg: &Value) -> Result<Option<Value>> {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
    let role = if role == "developer" { "system" } else { role };

    if role == "function" {
        return Err(oxllm_core::Error::BadRequest(
            "legacy function results require correlation with their preceding assistant call"
                .into(),
        ));
    }
    // Legacy results have already been correlated and rewritten to tool.
    if role == "tool" {
        let tool_use_id = msg
            .get("tool_call_id")
            .cloned()
            .unwrap_or_else(|| json!(""));
        let content = match msg.get("content") {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(Value::Array(arr)) => {
                Value::Array(arr.iter().map(convert_content_block_to_anthropic).collect())
            }
            Some(other) => other.clone(),
            None => Value::String(String::new()),
        };
        let mut native = json!({
        "role": "user",
        "content": [{
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        }]
        });
        if let Some(control) = msg.get("cache_control").filter(|value| !value.is_null()) {
            native["content"][0]["cache_control"] = control.clone();
        }
        if let Some(config) = msg.get("output_config").filter(|value| !value.is_null()) {
            native["output_config"] = config.clone();
        }
        return Ok(Some(native));
    }

    let mut out = Map::new();
    out.insert("role".to_string(), Value::String(role.to_string()));
    if let Some(config) = msg.get("output_config").filter(|value| !value.is_null()) {
        out.insert("output_config".to_string(), config.clone());
    }

    let mut content_blocks = crate::util::thinking_blocks_for_replay(msg, true)?;
    // Anthropic rejects messages whose content renders nothing. Empty or
    // whitespace-only text is rewritten to a placeholder (reference
    // `_sanitize_empty_text_content`); other keys on a rewritten block are
    // preserved. A turn that still renders nothing is dropped by the caller
    // — except an all-empty user/tool run, which keeps one placeholder turn
    // (reference `_empty_user_turn_placeholder`).
    if let Some(content) = msg.get("content") {
        match content {
            Value::String(s) if !s.trim().is_empty() => {
                content_blocks.push(json!({"type": "text", "text": s}));
            }
            Value::String(_) => {
                content_blocks.push(placeholder_text_block(None));
            }
            Value::Array(arr) => {
                for block in arr {
                    // An unsigned thinking block with no text is rejected
                    // ("each thinking block must contain thinking" applies
                    // without a signature); a signed blank block is the
                    // `display: omitted` replay shape and stays. Signature-only
                    // blocks (oxllm's streaming spelling) normalize to the
                    // empty-text form the API accepts.
                    let mut block = block.clone();
                    if block.get("type").and_then(Value::as_str) == Some("thinking")
                        && block.get("thinking").and_then(Value::as_str).is_none()
                        && block
                            .get("signature")
                            .and_then(Value::as_str)
                            .is_some_and(|signature| !signature.is_empty())
                    {
                        block["thinking"] = json!("");
                    }
                    if crate::util::is_blank_unsigned_thinking_block(&block) {
                        continue;
                    }
                    // Rewrite empty/whitespace/non-string text blocks in
                    // place (non-text blocks are untouched) — a content list
                    // that renders nothing is rejected just like an empty
                    // string.
                    if block.get("type").and_then(Value::as_str) == Some("text")
                        && block
                            .get("text")
                            .and_then(Value::as_str)
                            .is_none_or(|text| text.trim().is_empty())
                    {
                        block["text"] = json!(EMPTY_TEXT_PLACEHOLDER);
                        content_blocks.push(block);
                        continue;
                    }
                    content_blocks.push(convert_content_block_to_anthropic(&block));
                }
            }
            other if !other.is_null() => {
                content_blocks.push(other.clone());
            }
            _ => {}
        }
    }
    if let Some(tc) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for call in tc {
            content_blocks.push(map_tool_call_to_anthropic(call));
        }
    }

    // A turn that renders nothing (missing/null/[] content, no tool calls,
    // no thinking) is dropped rather than sent as `content: ""`.
    if content_blocks.is_empty() {
        return Ok(None);
    }

    if let Some(control) = msg.get("cache_control").filter(|value| !value.is_null()) {
        if let Some(Value::Object(block)) = content_blocks.last_mut() {
            block
                .entry("cache_control".to_string())
                .or_insert_with(|| control.clone());
        }
    }

    if content_blocks.len() == 1
        && content_blocks[0].get("type").and_then(|t| t.as_str()) == Some("text")
        && content_blocks[0].get("cache_control").is_none()
        && msg.get("tool_calls").is_none()
    {
        // Keep plain-string content for simple text-only messages.
        let text = content_blocks[0]
            .get("text")
            .cloned()
            .unwrap_or(Value::String(String::new()));
        out.insert("content".to_string(), text);
    } else {
        out.insert("content".to_string(), Value::Array(content_blocks));
    }

    Ok(Some(Value::Object(out)))
}

fn map_tool_call_to_anthropic(tc: &Value) -> Value {
    let id = tc.get("id").cloned().unwrap_or_else(|| json!(""));
    let function = tc.get("function").cloned().unwrap_or_else(|| json!({}));
    let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = function
        .get("arguments")
        .and_then(|v| v.as_str())
        .unwrap_or("{}");
    let input: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
    json!({
    "type": "tool_use",
    "id": id,
    "name": name,
    "input": input,
    })
}

/// Recursively strip JSON Schema constructs Anthropic rejects on tool
/// `input_schema`, rewriting others into the form it accepts.
///
/// Anthropic's tool schema validator is stricter than the OpenAI / JSON Schema
/// spec. Two cases it rejects, mirroring the reference schema-walk in
/// `AnthropicConfig.filter_anthropic_output_schema` (drops unsupported keys
/// while recursing through `properties` / `items` / `$defs` / `anyOf` /
/// `allOf` / `oneOf`):
///
/// - `uniqueItems`: not supported on array schemas — dropped entirely.
/// - `oneOf`: not supported — rewritten to `anyOf`, moving the branch array
///   verbatim. `anyOf` is a safe rewrite because Anthropic validates tool
///   inputs leniently and `oneOf` (exactly one match) is the stricter
///   relation. If both `oneOf` and `anyOf` are present, the `oneOf` branches
///   are appended to the existing `anyOf` so no alternatives are lost.
///
/// It also normalizes the non-standard `type:"custom"` marker that Claude
/// Code emits on custom-tool schemas to `type:"object"`. Bedrock rejects any
/// type string that isn't a valid JSON Schema type with
/// `tools.N.input_schema.type: Input should be 'object'`; the reference applies the
/// same rewrite in `normalize_json_schema_custom_types_to_object`. The
/// rewrite is safe for plain Anthropic too — the API ignores the `type` field
/// on tool `input_schema` — so a single shared sanitizer covers both paths.
///
/// The walk recurses into every standard JSON Schema container location so
/// nested schemas (a `properties` entry whose own `items` carries
/// `uniqueItems`, etc.) are sanitized too. Schemas without any unsupported
/// fields are returned unchanged.
/// This compatibility pass is only for non-strict tools. Strict tools and
/// native JSON outputs preserve the caller's schema without these rewrites.
pub(crate) fn sanitize_tool_input_schema(schema: Value) -> Value {
    let mut map = match schema {
        Value::Object(map) => map,
        other => return other,
    };

    // `type:"custom"` (Claude Code custom-tool schema marker) → `"object"`.
    // Bedrock rejects non-standard JSON Schema type strings; Anthropic ignores
    // the field entirely so rewriting is harmless on both paths.
    if map.get("type").and_then(|v| v.as_str()) == Some("custom") {
        map.insert("type".to_string(), json!("object"));
    }

    // `oneOf` → `anyOf`. If `anyOf` already exists, append `oneOf`'s branches
    // to it so the union of alternatives is preserved.
    if let Some(one) = map.remove("oneOf") {
        match map.entry("anyOf".to_string()) {
            serde_json::map::Entry::Vacant(v) => {
                v.insert(one);
            }
            serde_json::map::Entry::Occupied(mut o) => {
                if let (Value::Array(existing), Value::Array(ones)) = (o.get_mut(), one) {
                    existing.extend(ones);
                }
            }
        }
    }

    // `uniqueItems` is unsupported on array schemas → drop it.
    map.remove("uniqueItems");

    // Recurse into the standard JSON Schema container locations so nested
    // schemas are cleaned too. `items` is a single schema; `properties` /
    // `$defs` are maps of schemas; `anyOf` / `allOf` are arrays of schemas
    // (including the `oneOf` we just renamed to `anyOf`).
    if let Some(items) = map.get_mut("items") {
        *items = sanitize_tool_input_schema(std::mem::take(items));
    }
    for key in ["properties", "$defs", "anyOf", "allOf"] {
        if let Some(value) = map.get_mut(key) {
            match value {
                Value::Object(obj) => {
                    for (_, v) in obj.iter_mut() {
                        *v = sanitize_tool_input_schema(std::mem::take(v));
                    }
                }
                Value::Array(arr) => {
                    for item in arr.iter_mut() {
                        *item = sanitize_tool_input_schema(std::mem::take(item));
                    }
                }
                _ => {}
            }
        }
    }

    Value::Object(map)
}

/// Map an Anthropic hosted (non-`function`) tool to its wire shape (feature #11).
///
/// Hosted tools — `computer_*`, `bash_*`, `text_editor_*`, `web_search_*` —
/// are passed through with their reserved `type`/`name` pair and do NOT get
/// wrapped in `input_schema`. The caller may supply computer display
/// dimensions under `function.parameters`; lift them to the top level where
/// Anthropic expects them. Returns `None` for a tool type we don't recognise
/// as hosted (caller should fall back to the function-tool path).
fn map_hosted_tool(tool: &Value) -> Option<Value> {
    let tool_type = tool.get("type").and_then(|v| v.as_str())?;
    let params = tool.get("function").and_then(|f| f.get("parameters"));
    let pick_name = || {
        tool.get("name").and_then(|v| v.as_str()).or_else(|| {
            tool.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
        })
    };
    // Advisor tool (`advisor_20260301`): Anthropic requires a `model`
    // field; optional `max_uses` and `caching` are forwarded when present.
    // Returns `None` (caller drops the tool) when `model` is missing — the
    // request would 400 upstream otherwise.
    if tool_type.starts_with("advisor_") {
        let advisor_model = tool.get("model").and_then(|v| v.as_str())?;
        let mut out = Map::new();
        out.insert("type".to_string(), json!(tool_type));
        out.insert("name".to_string(), json!("advisor"));
        out.insert("model".to_string(), json!(advisor_model));
        for k in ["max_uses", "caching"] {
            if let Some(v) = tool.get(k) {
                if !v.is_null() {
                    out.insert(k.to_string(), v.clone());
                }
            }
        }
        return Some(Value::Object(out));
    }
    // Tool search tools (regex / BM25 variants): pass through with type +
    // name only. The caller may have supplied a custom name; otherwise use
    // the conventional default.
    if tool_type.starts_with("tool_search_tool_regex_")
        || tool_type.starts_with("tool_search_tool_bm25_")
    {
        let default_name = if tool_type.starts_with("tool_search_tool_regex_") {
            "tool_search_tool_regex"
        } else {
            "tool_search_tool_bm25"
        };
        let mut out = Map::new();
        out.insert("type".to_string(), json!(tool_type));
        out.insert(
            "name".to_string(),
            json!(pick_name().unwrap_or(default_name)),
        );
        return Some(Value::Object(out));
    }
    if tool_type.starts_with("computer_") {
        let mut out = Map::new();
        out.insert("type".to_string(), json!(tool_type));
        out.insert("name".to_string(), json!(pick_name().unwrap_or("computer")));
        if let Some(p) = params {
            for k in ["display_width_px", "display_height_px", "display_number"] {
                if let Some(v) = p.get(k) {
                    out.insert(k.to_string(), v.clone());
                }
            }
        }
        return Some(Value::Object(out));
    }
    if tool_type.starts_with("bash_") {
        let mut out = Map::new();
        out.insert("type".to_string(), json!(tool_type));
        out.insert("name".to_string(), json!(pick_name().unwrap_or("bash")));
        return Some(Value::Object(out));
    }
    if tool_type.starts_with("text_editor_") {
        let mut out = Map::new();
        out.insert("type".to_string(), json!(tool_type));
        out.insert(
            "name".to_string(),
            json!(pick_name().unwrap_or("str_replace_editor")),
        );
        return Some(Value::Object(out));
    }
    // web_search_* and any other hosted type the caller already shaped: pass
    // through verbatim so the request keeps whatever fields it carried.
    if tool_type.starts_with("web_search_") {
        return Some(tool.clone());
    }
    None
}

/// Split an OpenAI-shape `tools[]` array into `(remaining_tools, mcp_servers)`.
///
/// Anthropic connects to MCP (Model Context Protocol) servers directly: the
/// client describes each server as an entry inside `tools[]`, and we lift them
/// out into the top-level `mcp_servers` array on the Anthropic body (they must
/// NOT be passed through `map_tool_to_anthropic`, which expects function
/// tools). Mirrors the reference tool mapping (`type: "url"`) and
/// `_map_openai_mcp_server_tool` (`type: "mcp"`).
///
/// # Input formats
///
/// - `type: "url"` (Anthropic-shape): carried over, keeping `url` / `name` /
///   `authorization_token` plus any `tool_configuration` / `allowed_tools` /
///   `disallowed_tools` knobs the caller supplied.
/// - `type: "mcp"` (OpenAI-shape): rewritten to `type: "url"` with
///   `server_url` → `url`, `server_label` → `name`, and
///   `headers.Authorization` (minus the `Bearer ` prefix) →
///   `authorization_token`. `allowed_tools` / `disallowed_tools` /
///   `tool_configuration` are forwarded when present.
///
/// Any other tool type is returned in `remaining_tools` untouched so the
/// standard function-tool path still processes it.
fn extract_mcp_servers(tools: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut remaining: Vec<Value> = Vec::with_capacity(tools.len());
    let mut mcp_servers: Vec<Value> = Vec::new();

    for tool in tools {
        let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match tool_type {
            "url" => {
                let mut server = Map::new();
                server.insert("type".to_string(), json!("url"));
                for k in ["url", "name", "authorization_token"] {
                    if let Some(v) = tool.get(k) {
                        if !v.is_null() {
                            server.insert(k.to_string(), v.clone());
                        }
                    }
                }
                // Forward optional Anthropic MCP knobs verbatim.
                for k in ["tool_configuration", "allowed_tools", "disallowed_tools"] {
                    if let Some(v) = tool.get(k) {
                        if !v.is_null() {
                            server.insert(k.to_string(), v.clone());
                        }
                    }
                }
                mcp_servers.push(Value::Object(server));
            }
            "mcp" => {
                let mut server = Map::new();
                server.insert("type".to_string(), json!("url"));
                if let Some(v) = tool.get("server_url") {
                    server.insert("url".to_string(), v.clone());
                }
                if let Some(v) = tool.get("server_label") {
                    server.insert("name".to_string(), v.clone());
                }
                // OpenAI mcp tools carry the bearer token under
                // `headers.Authorization` as `Bearer <token>`; Anthropic wants
                // a bare `authorization_token`. Fall back to a top-level
                // `authorization_token` if the caller already supplied one.
                let bearer = tool
                    .get("headers")
                    .and_then(|h| h.get("Authorization"))
                    .and_then(|v| v.as_str());
                if let Some(b) = bearer {
                    let token = b.strip_prefix("Bearer ").unwrap_or(b);
                    server.insert("authorization_token".to_string(), json!(token));
                } else if let Some(v) = tool.get("authorization_token") {
                    if !v.is_null() {
                        server.insert("authorization_token".to_string(), v.clone());
                    }
                }
                for k in ["allowed_tools", "disallowed_tools", "tool_configuration"] {
                    if let Some(v) = tool.get(k) {
                        if !v.is_null() {
                            server.insert(k.to_string(), v.clone());
                        }
                    }
                }
                mcp_servers.push(Value::Object(server));
            }
            _ => remaining.push(tool.clone()),
        }
    }

    (remaining, mcp_servers)
}

fn map_tool_to_anthropic(tool: &Value) -> Result<Option<Value>> {
    let strict = crate::util::strict_tool_definition(tool)?;
    // Hosted tools (computer / bash / text_editor / web_search / advisor /
    // tool_search) are passed through with their reserved shape — they must
    // NOT be wrapped in `input_schema` like a regular function tool. Returns
    // `None` when a hosted tool is malformed in a way the caller should drop
    // it (e.g. advisor without the required `model`).
    if tool
        .get("type")
        .and_then(|v| v.as_str())
        .map(|s| s != "function")
        .unwrap_or(false)
    {
        if let Some(mut hosted) = map_hosted_tool(tool) {
            if let Some(object) = hosted.as_object_mut() {
                object.remove("function");
                if object.get("cache_control").is_some_and(Value::is_null) {
                    object.remove("cache_control");
                }
                // Native fields take precedence over defaults/lifted legacy
                // carrier values. Unknown native options are provider-owned.
                if let Some(native) = tool.as_object() {
                    for (field, value) in native {
                        if field != "function" && !(field == "cache_control" && value.is_null()) {
                            object.insert(field.clone(), value.clone());
                        }
                    }
                }
                // Explicit native null clears a legacy annotation. Do not
                // enable provider caching by reviving that legacy value.
                if tool.get("cache_control").is_none()
                    && let Some(control) = tool
                        .pointer("/function/cache_control")
                        .filter(|value| !value.is_null())
                {
                    object
                        .entry("cache_control".to_string())
                        .or_insert_with(|| control.clone());
                }
                if strict {
                    object.insert("strict".into(), json!(true));
                } else {
                    object.remove("strict");
                }
            }
            return Ok(Some(hosted));
        }
        if strict {
            if tool
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.starts_with("advisor_"))
            {
                return Err(oxllm_core::Error::BadRequest(
                    "advisor tool requires a model field".into(),
                ));
            }
            return Err(oxllm_core::Error::BadRequest(
                "strict tool use has no native mapping for this tool type".into(),
            ));
        }
        // Malformed hosted tool (e.g. advisor without model): drop entirely
        // rather than fall through to the function path, which would
        // synthesise an empty-name tool that 400s upstream.
        return Ok(None);
    }

    let function = tool.get("function").cloned().unwrap_or_else(|| json!({}));
    let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let description = function.get("description").cloned();
    let input_schema = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));

    let mut out = Map::new();
    out.insert("name".to_string(), Value::String(name.to_string()));
    // Keep legacy normalization for advisory tools only. Strict schemas must
    // reach the provider intact, including constraints it might reject.
    out.insert(
        "input_schema".to_string(),
        if strict {
            input_schema
        } else {
            sanitize_tool_input_schema(input_schema)
        },
    );
    if strict {
        out.insert("strict".into(), json!(true));
    }
    if let Some(desc) = description {
        out.insert("description".to_string(), desc);
    }
    // Preserve cache_control: accept it from either the function object or
    // the top-level tool object, matching the reference tool-mapping behavior.
    let cache_control = function
        .get("cache_control")
        .or_else(|| tool.get("cache_control"));
    if let Some(cc) = cache_control {
        if !cc.is_null() {
            out.insert("cache_control".to_string(), cc.clone());
        }
    }
    // A1: `defer_loading` — accept from either the function object or the
    // top-level tool object. Only forwarded for tools that support it
    // (function tools); hosted tools handle their own pass-through above.
    let defer_loading = function
        .get("defer_loading")
        .or_else(|| tool.get("defer_loading"));
    if let Some(dl) = defer_loading {
        if !dl.is_null() {
            out.insert("defer_loading".to_string(), dl.clone());
        }
    }
    // A2: `allowed_callers` — list of strings, accept from function or top.
    let allowed_callers = function
        .get("allowed_callers")
        .or_else(|| tool.get("allowed_callers"));
    if let Some(ac) = allowed_callers {
        if !ac.is_null() {
            out.insert("allowed_callers".to_string(), ac.clone());
        }
    }
    // A3: `input_examples` — only forwarded when it's a non-empty array.
    let input_examples = function
        .get("input_examples")
        .or_else(|| tool.get("input_examples"));
    if let Some(ie) = input_examples {
        if let Some(arr) = ie.as_array() {
            if !arr.is_empty() {
                out.insert("input_examples".to_string(), ie.clone());
            }
        }
    }
    Ok(Some(Value::Object(out)))
}

fn map_tool_choice_to_anthropic(tool_choice: &Value) -> Option<Value> {
    if let Some(s) = tool_choice.as_str() {
        return Some(match s {
            "auto" => json!({"type": "auto"}),
            "required" | "any" => json!({"type": "any"}),
            "none" => json!({"type": "none"}),
            _ => return None,
        });
    }
    if let Some(obj) = tool_choice.as_object() {
        if matches!(
            obj.get("type").and_then(Value::as_str),
            Some("auto" | "any" | "none" | "tool")
        ) {
            // Already-native choices keep all provider-owned options.
            return Some(tool_choice.clone());
        }
        if obj.get("type").and_then(Value::as_str) == Some("required")
            && obj
                .keys()
                .all(|key| matches!(key.as_str(), "type" | "disable_parallel_tool_use"))
        {
            let mut choice = obj.clone();
            choice.insert("type".into(), json!("any"));
            return Some(Value::Object(choice));
        }
        if obj
            .keys()
            .any(|key| !matches!(key.as_str(), "type" | "function"))
            || obj
                .get("type")
                .is_some_and(|kind| kind.as_str() != Some("function"))
            || obj
                .get("function")
                .and_then(Value::as_object)
                .is_some_and(|function| function.keys().any(|key| key != "name"))
        {
            return None;
        }
        if let Some(name) = obj
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
        {
            return Some(json!({"type": "tool", "name": name}));
        }
    }
    None
}

/// Whether a character is allowed in an Anthropic tool name
/// (`^[a-zA-Z0-9_-]{1,128}$`).
fn is_valid_tool_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Truncate a string to at most `max` chars (by Unicode scalar value, not
/// by byte count, so non-ASCII names never panic on slicing).
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Build `(forward, reverse)` tool-name maps for a list of OpenAI-format
/// tools.
///
/// - `forward`: original caller name → Anthropic-safe name to send upstream.
/// - `reverse`: Anthropic-safe name → original caller name, stashed for
///   `transform_response` / `transform_stream_chunk` to restore on the way
///   back.
///
/// Replacement rule: invalid chars → `_`. Names longer than 128 chars are
/// truncated. Collisions (two distinct originals collapsing to the same
/// sanitized form) are broken by appending `_2`, `_3`, … — the suffix is
/// inserted before truncation so the result still fits the 128-char limit.
/// Names that are already valid and unique are recorded as identity mappings
/// so the reverse lookup table is exhaustive over everything sent upstream.
fn build_tool_name_maps(tools: &[Value]) -> (HashMap<String, String>, HashMap<String, String>) {
    let mut forward: HashMap<String, String> = HashMap::new();
    let mut reverse: HashMap<String, String> = HashMap::new();
    // Per-base count of tools already assigned, so N tools that all sanitise
    // to the same `base` pick distinct candidates in O(1) each (candidate k =
    // `base` for k=1, `base_k` for k≥2) instead of an O(N²) linear probe. The
    // k-th same-base tool's candidate is unique by construction; only rare
    // cross-base collisions (two different originals whose truncated forms
    // coincide) need a probe, capped below to bound adversarial CPU burn.
    let mut per_base_count: HashMap<String, u32> = HashMap::new();

    for tool in tools {
        // OpenAI-format caller tools carry the name under `function.name`.
        // Hosted/native definitions may carry their name at the tool root.
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
            name.len() > 128 || name.chars().any(|c| !is_valid_tool_name_char(c));

        // An already-valid name is only an identity mapping when nothing has
        // claimed that name yet. If an earlier tool's SANITIZED form equals
        // this name (e.g. ["my.tool", "my_tool"]), recording the identity
        // mapping would clobber the earlier reverse entry and send BOTH
        // tools upstream under the same name — treat it as a collision and
        // fall through to the suffix-disambiguation loop below.
        if !needs_sanitization && !reverse.contains_key(name) {
            // Identity mapping keeps the reverse table exhaustive so the
            // response side can look up any name we'd have sent.
            forward.insert(name.to_string(), name.to_string());
            reverse.insert(name.to_string(), name.to_string());
            continue;
        }

        let base: String = name
            .chars()
            .map(|c| if is_valid_tool_name_char(c) { c } else { '_' })
            .collect();

        // Assign the next candidate for this base. Bare base for the first
        // tool, `_k` for the k-th (k≥2). Same-base candidates are distinct by
        // construction, so only cross-base collisions need a bounded probe.
        let count = per_base_count.entry(base.clone()).or_insert(0);
        *count += 1;
        let mut suffix_num = if *count == 1 { 0 } else { *count };
        let candidate = loop {
            let c = if suffix_num == 0 {
                truncate_chars(&base, 128)
            } else {
                let suffix = format!("_{suffix_num}");
                let max_base = 128usize.saturating_sub(suffix.len());
                format!("{}{}", truncate_chars(&base, max_base), suffix)
            };
            match reverse.get(&c) {
                None => break c,
                Some(v) if v == name => break c,
                _ => {
                    // Cross-base collision (rare): advance the suffix. Cap the
                    // probe so a pathological cross-base storm cannot pin the
                    // proxy thread; drop the tool if unresolvable.
                    suffix_num += 1;
                    if suffix_num > *count + 1000 {
                        break String::new();
                    }
                }
            }
        };
        if candidate.is_empty() || candidate.len() > 128 {
            // Pathological cross-base collision storm: drop the tool rather
            // than loop further (forward dedup at the top already drops exact
            // duplicate names).
            continue;
        }

        forward.insert(name.to_string(), candidate.clone());
        reverse.insert(candidate, name.to_string());
    }

    (forward, reverse)
}

/// Walk the Anthropic-format request body and rewrite tool names in place
/// using the forward map: `tools[].name`, `tool_choice.name` (when a
/// specific tool is selected), and `tool_use.name` blocks inside assistant
/// messages. Used by `transform_request` after the rest of the body is
/// built. Identity mappings (the common case) leave `out` untouched.
fn apply_tool_name_forward_map(out: &mut Map<String, Value>, forward: &HashMap<String, String>) {
    let remap = |name: &str| -> Option<String> {
        forward.get(name).filter(|mapped| *mapped != name).cloned()
    };

    if let Some(Value::Array(tools)) = out.get_mut("tools") {
        for tool in tools.iter_mut() {
            if let Some(Value::String(name)) = tool.get_mut("name") {
                if let Some(renamed) = remap(name) {
                    *name = renamed;
                }
            }
        }
    }

    if let Some(Value::Object(tc)) = out.get_mut("tool_choice") {
        if tc.get("type").and_then(|v| v.as_str()) == Some("tool") {
            if let Some(Value::String(name)) = tc.get_mut("name") {
                if let Some(renamed) = remap(name) {
                    *name = renamed;
                }
            }
        }
    }

    if let Some(Value::Array(messages)) = out.get_mut("messages") {
        for msg in messages.iter_mut() {
            let Some(Value::Array(blocks)) = msg.get_mut("content") else {
                continue;
            };
            for block in blocks.iter_mut() {
                if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                    if let Some(Value::String(name)) = block.get_mut("name") {
                        if let Some(renamed) = remap(name) {
                            *name = renamed;
                        }
                    }
                }
            }
        }
    }
}

/// Look up the original (caller-side) name for a sanitized tool name. Falls
/// back to the input verbatim when no mapping exists (e.g. a response from a
/// tool the request never registered, or a name that never needed
/// sanitization in the first place — both cases are safe to return as-is).
fn reverse_lookup_in(name: &str, reverse_map: &HashMap<String, String>) -> String {
    reverse_map
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

/// Build the reverse (sanitized → original) tool-name map from an OpenAI-shape
/// request. Used by the stateful response/stream transforms to recover caller
/// tool names without depending on the `thread_local!` `TOOL_NAME_MAP` (which
/// is broken across tokio worker threads). Replays the same
/// `extract_mcp_servers` + `build_tool_name_maps` pipeline `transform_request`
/// uses, so the sanitized forms agree with whatever was sent upstream.
fn reverse_map_from_request(request: &Value) -> HashMap<String, String> {
    let raw_tools = tools_from_request(request);
    let (tools, _mcp) = extract_mcp_servers(&raw_tools);
    build_tool_name_maps(&tools).1
}

/// In-place reverse-map of `tool_calls[].function.name` on an OpenAI-format
/// response, using an explicit reverse map.
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

pub(crate) fn normalize_stop_sequences(stop: &Value) -> Value {
    match stop {
        Value::String(s) => json!([s]),
        Value::Array(arr) => Value::Array(arr.clone()),
        other => json!([other]),
    }
}

/// Clamp a `thinking` block's `budget_tokens` below `max_tokens` (feature #4).
///
/// Anthropic requires `budget_tokens < max_tokens` AND `budget_tokens >= 1024`.
/// The budget is clamped into the valid window `[MIN_BUDGET, max_tokens - 1]`:
/// a budget at/above `max_tokens` is lowered to `max_tokens - 1`, and a budget below
/// Anthropic's 1024 minimum is raised to 1024 rather than being silently
/// rewritten to `max_tokens - 1` (which would inflate a small budget ~16x).
/// When the window is unsatisfiable (`max_tokens <= 1024`), return `None`:
/// explicit thinking must error, while inferred thinking may be omitted.
/// Integer- and float-encoded budgets are both honored (matching
/// `json_number_as_u64`). A `type: "disabled"` configuration never uses the
/// budget window: it is returned unchanged, so any stray `budget_tokens` is
/// provider-validated instead of failing this local clamp. Thinking without a
/// numeric budget is likewise returned unchanged.
pub fn clamp_thinking_budget(thinking: &Value, max_tokens: u64) -> Option<Value> {
    const MIN_BUDGET: u64 = 1024;
    let map = match thinking.as_object() {
        Some(m) => m,
        None => return Some(thinking.clone()),
    };
    if map.get("type").and_then(Value::as_str) == Some("disabled") {
        return Some(thinking.clone());
    }
    let budget = match map.get("budget_tokens").and_then(json_number_as_u64) {
        Some(b) => b,
        None => return Some(thinking.clone()),
    };
    // Clamp into [MIN_BUDGET, max_tokens - 1], but only when that window is
    // non-empty; otherwise the caller must distinguish explicit from inferred
    // intent. A float-encoded budget is normalized to the integer the API
    // expects even when it sits inside the window.
    let upper = max_tokens.saturating_sub(1);
    if upper < MIN_BUDGET {
        return None;
    }
    let clamped = budget.clamp(MIN_BUDGET, upper);
    let mut out = map.clone();
    out.insert("budget_tokens".to_string(), json!(clamped));
    Some(Value::Object(out))
}

pub(crate) fn thinking_budget_error() -> oxllm_core::Error {
    oxllm_core::Error::BadRequest(
        "`max_tokens` must be greater than `thinking.budget_tokens`; extended thinking requires a budget of at least 1024 tokens".into(),
    )
}

/// Compute the dynamic `anthropic-beta` header tags a request needs (feature #1).
///
/// Anthropic gates newer features behind comma-separated `anthropic-beta`
/// header values; a tag is only accepted when the corresponding feature is
/// actually used. Inspect the OpenAI-shape request body and return the set of
/// beta tags that should accompany it:
///
/// Stable schema formats, including the translated legacy `output_format`
/// alias, and strict tools do not require the structured-outputs beta.
///
/// - [`BETA_FAST_MODE`] when `speed == "fast"`.
/// - [`BETA_WEB_FETCH`] when the request carries `web_search_options` or a
///   hosted `web_search_*` tool.
/// - [`BETA_CONTEXT_MANAGEMENT`] when `context_management` is set, or when a
///   `memory_*` hosted tool is present.
/// - [`BETA_ADVANCED_TOOL_USE`] when any tool carries `input_examples` or
///   `allowed_callers`, or when a `tool_search_tool_*` hosted tool is present.
/// - [`BETA_ADVISOR_TOOL`] when an `advisor_*` hosted tool is present.
/// - [`BETA_EFFORT`] when `reasoning_effort` is set on a model that requires
///   the effort beta (pre-4.6 Claude models). 4.6+ models accept `effort` as
///   a stable API feature and do not need the tag.
///
/// The `ChatProvider::apply_auth` signature does not receive the request, so
/// the host must call this function and join the result with `, ` into the
/// `anthropic-beta` header alongside the `anthropic-version` set in
/// [`AnthropicProvider::apply_auth`].
pub fn collect_beta_headers(request: &Value, model: &str) -> Vec<String> {
    let mut betas: Vec<String> = Vec::new();
    if uses_message_output_config(request) {
        betas.push(BETA_MID_CONVERSATION_OUTPUT_CONFIG.to_string());
    }

    // Fast mode.
    if request.get("speed").and_then(|v| v.as_str()) == Some("fast") {
        betas.push(BETA_FAST_MODE.to_string());
    }

    // Hosted web search: either the OpenAI `web_search_options` knob or a
    // hosted `web_search_*` tool in the tool list.
    let tools_arr = request.get("tools").and_then(|v| v.as_array());
    let has_web_search = request.get("web_search_options").is_some()
        || tools_arr
            .map(|arr| {
                arr.iter().any(|t| {
                    t.get("type")
                        .and_then(|v| v.as_str())
                        .map(|s| s.starts_with("web_search"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
    if has_web_search {
        betas.push(BETA_WEB_FETCH.to_string());
    }

    // Context management (explicit field) OR a hosted `memory_*` tool (B3).
    let has_memory_tool = tools_arr
        .map(|arr| {
            arr.iter().any(|t| {
                t.get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.starts_with("memory_"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if request.get("context_management").is_some() || has_memory_tool {
        betas.push(BETA_CONTEXT_MANAGEMENT.to_string());
    }

    // B1: advanced tool-use beta fires when any tool carries `input_examples`
    // or `allowed_callers`, or when a `tool_search_tool_*` hosted tool is
    // present.
    let has_tool_search = tools_arr
        .map(|arr| {
            arr.iter().any(|t| {
                t.get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| {
                        s.starts_with("tool_search_tool_regex_")
                            || s.starts_with("tool_search_tool_bm25_")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    let has_advanced_tool_field = tools_arr
        .map(|arr| {
            arr.iter().any(|t| {
                let ie = t
                    .get("input_examples")
                    .or_else(|| t.get("function").and_then(|f| f.get("input_examples")));
                let ac = t
                    .get("allowed_callers")
                    .or_else(|| t.get("function").and_then(|f| f.get("allowed_callers")));
                let deferred = t
                    .get("defer_loading")
                    .or_else(|| t.get("function").and_then(|f| f.get("defer_loading")))
                    .and_then(Value::as_bool)
                    == Some(true);
                ie.map(|v| v.is_array()).unwrap_or(false)
                    || ac.map(|v| v.is_array()).unwrap_or(false)
                    || deferred
            })
        })
        .unwrap_or(false);
    if has_tool_search || has_advanced_tool_field {
        betas.push(BETA_ADVANCED_TOOL_USE.to_string());
    }

    // B2: advisor-tool beta fires when an `advisor_*` hosted tool is present.
    let has_advisor = tools_arr
        .map(|arr| {
            arr.iter().any(|t| {
                t.get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.starts_with("advisor_"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if has_advisor {
        betas.push(BETA_ADVISOR_TOOL.to_string());
    }

    // C1: effort beta fires when `reasoning_effort` is set AND the model is
    // pre-4.6 (4.6+ models accept `output_config.effort` as a stable API and
    // do not need the beta). Unknown effort values are dropped by the
    // request transform rather than triggering the beta.
    if let Some(effort) = request
        .pointer("/output_config/effort")
        .and_then(Value::as_str)
        .or_else(|| request.get("reasoning_effort").and_then(Value::as_str))
    {
        if is_known_effort(effort) && model_requires_effort_beta(model) {
            betas.push(BETA_EFFORT.to_string());
        }
    }

    betas
}

/// Whether any conversation message carries an active output configuration.
pub fn uses_message_output_config(request: &Value) -> bool {
    request
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message
                    .get("output_config")
                    .is_some_and(|config| !config.is_null())
            })
        })
}

/// Whether a `reasoning_effort` value is one Anthropic accepts.
fn is_known_effort(effort: &str) -> bool {
    matches!(
        effort,
        "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "none"
    )
}

/// Whether a model requires the `effort-2025-11-24` beta header to use
/// `reasoning_effort` / `output_config.effort`.
///
/// Claude 4.6 and later (incl. Opus 4.7/4.8 and any future 4.6+ / 5+ model)
/// accept `effort` as a stable API feature and do NOT need the beta tag.
/// Everything else (3.x, 4.0–4.5) needs it. Detection is a substring scan
/// over the version pair so provider-prefixed, family-prefixed
/// (`claude-opus-*`, `claude-sonnet-*`, `claude-haiku-*`) and date-suffixed
/// ids all resolve the same way.
fn model_requires_effort_beta(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // claude-3-5-* is the legacy 3.5 family — needs the beta. Check first
    // so the major-5 patterns below don't false-match the trailing `-5`.
    if m.contains("3-5") || m.contains("3_5") || m.contains("3.5") {
        return true;
    }
    // For each (major, minor) pair, accept either the direct form
    // (`claude-4-6`) or the family-prefixed form
    // (`claude-{opus,sonnet,haiku}-4-6`); both contain the literal `-4-6`
    // substring. We allow `-`/`_`/`.` separators to cover non-standard ids.
    let stable_pairs = [
        "4-6", "4_6", "4.6", "4-7", "4_7", "4.7", "4-8", "4_8", "4.8",
    ];
    if stable_pairs.iter().any(|p| m.contains(&format!("-{p}"))) {
        return false;
    }
    // Major version 5+: `claude-5*`, `claude-{opus,sonnet,haiku}-5*`. All
    // contain `claude` + (optional family) + `-5` followed by either `-`,
    // `_`, `.`, or end-of-string.
    let major_5_markers = [
        "claude-5",
        "claude_5",
        "claude.5",
        "claude-opus-5",
        "claude-opus_5",
        "claude-opus.5",
        "claude-sonnet-5",
        "claude-sonnet_5",
        "claude-sonnet.5",
        "claude-haiku-5",
        "claude-haiku_5",
        "claude-haiku.5",
    ];
    !major_5_markers.iter().any(|p| m.contains(p))
}

/// Whether a model uses Anthropic's adaptive-thinking interface
/// (`output_config.effort` + `thinking: {type: "adaptive"}`) rather than the
/// legacy budget-based `thinking: {type: "enabled", budget_tokens: N}` form.
///
/// This is the exact complement of [`model_requires_effort_beta`]: models that
/// do NOT need the `effort` beta (4.6+/4.7+/4.8+ and major-version 5+) accept
/// `effort` as a stable API feature and self-size their thinking budget.
/// Matches the reference adaptive-thinking capability probe
/// (`llms/anthropic/common_utils.py:412`), resolved via the version
/// substring rules instead of a cost-map lookup.
fn is_adaptive_thinking_model(model: &str) -> bool {
    !model_requires_effort_beta(model)
}

/// Map an OpenAI `reasoning_effort` string to a thinking `budget_tokens`
/// value, mirroring the reference `DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET`
/// constants (`constants.py:183-189`). `minimal` is clamped up to
/// `ANTHROPIC_MIN_THINKING_BUDGET_TOKENS` (1024) per
/// `_map_reasoning_effort` (`transformation.py:1226-1233`). Unknown/`none`
/// efforts return `0` (the caller drops the thinking block in that case).
fn reasoning_effort_budget(effort: &str) -> u64 {
    match effort {
        "minimal" => 1024,
        "low" => 1024,
        "medium" => 2048,
        "high" => 4096,
        "xhigh" => 8192,
        "max" => 16384,
        _ => 0,
    }
}

/// Map an OpenAI `reasoning_effort` to an Anthropic `output_config.effort`
/// value for adaptive-thinking models. Mirrors the reference
/// `REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT`
/// (`transformation.py:217-224`): `minimal` collapses to `low`; `none` and
/// unknown values return `None` (no effort field emitted).
fn reasoning_effort_to_output_config_effort(effort: &str) -> Option<&'static str> {
    match effort {
        "minimal" | "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("xhigh"),
        "max" => Some("max"),
        _ => None,
    }
}

/// Per-model ceiling for the `effort` value. Returns the (possibly clamped)
/// effort string, or `None` to drop the field entirely.
///
/// Claude Opus 4.5 does not support `max` (only `low`/`medium`/`high`); we
/// clamp `max`→`high` rather than rejecting the request. Other models pass
/// known effort values through unchanged; unknown values are dropped.
pub fn normalize_effort_for_model(model: &str, effort: &str) -> Option<String> {
    if !is_known_effort(effort) {
        return None;
    }
    if effort == "none" {
        // `none` is a sentinel meaning "no thinking" — it is not a valid
        // `output_config.effort` value, so drop it (the request transform
        // already handles `none` by skipping thinking entirely).
        return None;
    }
    let m = model.to_ascii_lowercase();
    // Opus 4.5 (and its dated / provider-prefixed variants) does not support
    // `max` (or `xhigh`) — clamp down to `high` rather than rejecting.
    let is_opus_4_5 = m.contains("opus-4-5")
        || m.contains("opus_4_5")
        || m.contains("opus-4.5")
        || m.contains("opus_4.5");
    if is_opus_4_5 && (effort == "max" || effort == "xhigh") {
        return Some("high".to_string());
    }
    Some(effort.to_string())
}

/// Read a JSON number as a `u64`, accepting both integer and float encodings.
///
/// `serde_json::Value::as_u64()` returns `None` for floats, so a caller whose
/// serializer emits `4096.0` would otherwise be silently dropped. Floats are
/// rounded and clamped to a minimum of 1, matching the reference
/// (`transformation.py:1427-1430`: `max(1, int(round(value)))`).
pub(crate) fn json_number_as_u64(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    let f = v.as_f64()?;
    if !f.is_finite() {
        return None;
    }
    Some(f.round().max(1.0) as u64)
}

/// Per-model default `max_tokens` (C3).
///
/// Anthropic requires `max_tokens` on every request; when the caller omits
/// it, pick a sensible default keyed off the model family:
/// - Haiku-class → 4096 (smaller output window)
/// - Opus 4/5-class models → 64000 (large output window)
/// - Sonnet-class / default → 8192 (the historical constant)
///
/// Model name substring match: anything containing `haiku` → 4096, the
/// 64k-window `opus-4`/`opus-5` families → 64000, otherwise (including the
/// older `claude-3-opus*` models whose real cap is 4096) → 8192.
pub fn default_max_tokens_for_model(model: &str) -> u64 {
    let m = model.to_ascii_lowercase();
    if m.contains("haiku") {
        4096
    } else if m.contains("opus-4") || m.contains("opus-5") {
        64000
    } else {
        DEFAULT_MAX_TOKENS
    }
}

/// Legacy informational direct-API family hint, retained for API compatibility.
/// Not used by request forwarding and not a promise of hosted-platform support:
/// actual model/schema acceptance is provider-owned.
pub fn supports_native_output_format(model: &str) -> bool {
    let model = model.to_ascii_lowercase().replace(['.', '_'], "-");
    [
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "claude-haiku-5-5",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
    ]
    .iter()
    .any(|family| model.contains(family))
}

/// Convert OpenAI `context_management` to Anthropic `edits` (M3).
///
/// - **OpenAI format** (list): `[{"type": "compaction", "compact_threshold": N}]`
///   → each entry becomes `{"type": "compact_20260112", "trigger": {"type":
///   "input_tokens", "value": N}}`.
/// - **Anthropic-native format** (dict with `"edits"`): the `edits` array is
///   passed through verbatim.
///
/// Returns `None` when the input is empty or has no convertible entries.
fn map_context_management_to_edits(cm: &Value) -> Option<Vec<Value>> {
    // Anthropic-native: dict with "edits" key → pass edits through.
    if let Some(obj) = cm.as_object() {
        if obj.contains_key("edits") {
            return cm
                .get("edits")
                .and_then(|v| v.as_array())
                .cloned()
                .filter(|a| !a.is_empty());
        }
    }
    // OpenAI format: list of entries. A bare object (a single compaction
    // entry) is normalised to a one-element list so callers don't have to
    // wrap it themselves.
    let arr: Vec<&Value> = match cm {
        Value::Array(a) => a.iter().collect(),
        Value::Object(_) => vec![cm],
        _ => return None,
    };
    let edits: Vec<Value> = arr
        .iter()
        .filter_map(|entry| {
            let e = entry.as_object()?;
            let entry_type = e.get("type").and_then(|v| v.as_str())?;
            if entry_type != "compaction" {
                return None;
            }
            let mut edit = Map::new();
            edit.insert("type".to_string(), json!("compact_20260112"));
            if let Some(threshold) = e.get("compact_threshold").and_then(json_number_as_u64) {
                edit.insert(
                    "trigger".to_string(),
                    json!({"type": "input_tokens", "value": threshold}),
                );
            }
            Some(Value::Object(edit))
        })
        .collect();
    if edits.is_empty() {
        return None;
    }
    Some(edits)
}

/// Walk Anthropic `content[]` blocks and split them into the OpenAI response
/// shape: `(text, tool_calls, reasoning_content, thinking_blocks, citations,
/// guardrail_blocks, web_search_results, tool_results, compaction_blocks)`.
///
/// - `text`: concatenation of `type: "text"` blocks → `message.content`.
/// - `tool_calls`: `type: "tool_use"` blocks → `message.tool_calls`. The
///   Anthropic `caller` field (programmatic tool calling) is preserved on
///   each OpenAI tool_call when present.
/// - `reasoning_content`: concatenation of `thinking` text from
///   `type: "thinking"` blocks → `message.reasoning_content` (DeepSeek/Grok
///   convention). Redacted thinking blocks contribute no text.
/// - `thinking_blocks`: the raw thinking blocks (including `signature` and
///   any other Anthropic-specific fields) preserved verbatim, so callers can
///   replay them in a multi-turn conversation. Goes to
///   `message.thinking_blocks`.
/// - `citations`: every citation object from the `citations` array on every
///   `type: "text"` block, concatenated in order. Goes to
///   `message.citations`.
/// - `guardrail_blocks`: raw `guarded_content` blocks preserved verbatim.
/// - `web_search_results`: `web_search_tool_result` and
///   `web_fetch_tool_result` blocks (Anthropic hosted web search results —
///   kept for multi-turn reconstruction). Goes to
///   `message.web_search_results`.
/// - `tool_results`: other `*_tool_result` blocks (code execution: bash,
///   text_editor, etc.) — kept verbatim so callers can inspect stdout /
///   stderr / exit_code. Goes to `message.code_interpreter_results`.
/// - `compaction_blocks`: `type: "compaction"` blocks carrying compacted
///   conversation context. Goes to `message.compaction_blocks`.
#[allow(clippy::type_complexity)]
fn extract_assistant_content(
    content: &Value,
) -> (
    String,
    Vec<Value>,
    String,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
    Vec<Value>,
) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut citations: Vec<Value> = Vec::new();
    let mut guardrail_blocks: Vec<Value> = Vec::new();
    let mut web_search_results: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();
    let mut compaction_blocks: Vec<Value> = Vec::new();
    if let Some(arr) = content.as_array() {
        for block in arr {
            let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match block_type {
                "text" => {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(t.to_string());
                    }
                    // Collect citations from text blocks. Anthropic attaches a
                    // `citations` array to each text block when the citations
                    // feature is enabled; gather every citation across all
                    // text blocks so callers see the complete set.
                    if let Some(cites) = block.get("citations").and_then(|v| v.as_array()) {
                        citations.extend(cites.iter().cloned());
                    }
                }
                "thinking" => {
                    if let Some(t) = block.get("thinking").and_then(|v| v.as_str()) {
                        reasoning_parts.push(t.to_string());
                    }
                    thinking_blocks.push(block.clone());
                }
                // Feature #9: `redacted_thinking` carries opaque `data`
                // (encrypted/signed reasoning the API won't expose as text).
                // It must NOT contribute to `reasoning_content` (no text to
                // show), but it IS preserved verbatim in `thinking_blocks` so
                // multi-turn replay works — mirroring the Bedrock transform.
                "redacted_thinking" => {
                    thinking_blocks.push(block.clone());
                }
                // M11: `guarded_content` — guardrails intervened on the
                // response. Extract any inner text the guardrail permits
                // (Anthropic nests the permitted content under `content[]`)
                // so the caller still sees the (possibly redacted) text, and
                // preserve the raw block so callers can inspect the
                // permissions/metadata.
                "guarded_content" => {
                    if let Some(inner) = block.get("content").and_then(|v| v.as_array()) {
                        for inner_block in inner {
                            if let Some(t) = inner_block.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(t.to_string());
                            }
                        }
                    }
                    guardrail_blocks.push(block.clone());
                }
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Normalise `input: null` (redacted-thinking / guardrail
                    // / Bedrock wrapping paths can emit it) to `"{}"`. The
                    // OpenAI tool-call `arguments` field must be a JSON object
                    // string; `"null"` parses to JSON null and breaks tool
                    // dispatch in openai-python / openai-node (which do
                    // `json.loads(arguments).get(...)`).
                    let args = match block.get("input") {
                        Some(Value::Null) | None => "{}".to_string(),
                        Some(input) => serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                    };
                    // Preserve the Anthropic `caller` field (programmatic
                    // tool calling) onto the OpenAI tool_call object so the
                    // caller-visible identity is not lost in translation.
                    let caller = block.get("caller").cloned();
                    let mut tc = json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                    });
                    if let Some(caller_val) = caller {
                        if let Value::Object(map) = &mut tc {
                            map.insert("caller".to_string(), caller_val);
                        }
                    }
                    tool_calls.push(tc);
                }
                // Compaction blocks carry compacted conversation context
                // emitted when automatic context management triggers. Preserve
                // the raw block so callers can replay it in follow-up
                // requests (the compacted summary is opaque to us).
                "compaction" => {
                    compaction_blocks.push(block.clone());
                }
                _ if block_type.ends_with("_tool_result") => {
                    // `tool_search_tool_result` is internal metadata the
                    // Anthropic API uses for its tool-search dispatch; skip it
                    // so it doesn't leak into caller-visible results.
                    if block_type == "tool_search_tool_result" {
                        continue;
                    }
                    // `web_search_tool_result` and `web_fetch_tool_result` are
                    // the hosted web-search results coming back from a server
                    // tool call. Keep them in a dedicated bucket for
                    // multi-turn reconstruction (the `encrypted_content`
                    // field on each result must be replayed verbatim on the
                    // next request).
                    if block_type == "web_search_tool_result"
                        || block_type == "web_fetch_tool_result"
                    {
                        web_search_results.push(block.clone());
                    } else {
                        // All other `*_tool_result` blocks (code execution:
                        // `bash_code_execution_tool_result`,
                        // `text_editor_code_execution_tool_result`, …) carry
                        // stdout / stderr / exit_code we want to surface as
                        // code-interpreter results. Preserve verbatim.
                        tool_results.push(block.clone());
                    }
                }
                _ => {}
            }
        }
    }
    (
        text_parts.join(""),
        tool_calls,
        reasoning_parts.join(""),
        thinking_blocks,
        citations,
        guardrail_blocks,
        web_search_results,
        tool_results,
        compaction_blocks,
    )
}

fn map_stop_reason_to_openai(reason: &str) -> Value {
    match reason {
        "end_turn" | "stop_sequence" => json!("stop"),
        "max_tokens" | "model_context_window_exceeded" => json!("length"),
        "tool_use" => json!("tool_calls"),
        "guardrail_intervention" | "refusal" => json!("content_filter"),
        _ => json!("stop"),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Extract the JSON payload from an SSE event.
///
/// Per the SSE spec (HTML5 §9.2.6) an event may carry multiple `data:` lines
/// that the client must concatenate with `\n` between them. Anthropic does
/// not currently emit multi-line data events, but the host's SSE framer
/// delivers whole events to the provider, so a future beta (or a Bedrock
/// wrapping layer) that splits a payload across `data:` lines would otherwise
/// have its tail silently truncated. Returns `None` when no `data:` line is
/// present so the caller can skip cleanly instead of feeding the raw event
/// (which may be just an `event:` line) to `serde_json` and surfacing a
/// confusing parse error.
fn extract_data_line(raw: &str) -> Option<String> {
    crate::util::extract_sse_data(raw)
}

/// Map an Anthropic error `type` string (from a streamed `{"type":"error"}`
/// event) to the matching HTTP status code. Mirrors Anthropic's documented
/// error taxonomy; unknown types fall back to 500.
fn anthropic_error_status(error_type: Option<&str>) -> u16 {
    match error_type {
        Some("overloaded_error") => 529,
        Some("rate_limit_error") => 429,
        Some("authentication_error") => 401,
        Some("permission_error") => 403,
        Some("invalid_request_error") => 400,
        Some("not_found_error") => 404,
        Some("timeout_error") => 504,
        // `api_error` and any unmapped type → 500.
        _ => 500,
    }
}

/// Core streaming-chunk transform shared by the `thread_local!`-backed trait
/// method and the host-state-backed `transform_stream_chunk_stateful`.
///
/// All per-stream state (`meta`, `tool_blocks`, `reverse_map`) is passed in
/// explicitly by the caller, so this function has no hidden affinity to any
/// particular thread and is safe to drive from a tokio task that migrates
/// across worker threads.
fn transform_stream_chunk_impl(
    chunk: &str,
    meta: &mut Option<StreamMeta>,
    tool_blocks: &mut HashMap<u64, ToolBlockState>,
    reverse_map: &HashMap<String, String>,
) -> Result<Option<Value>> {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let data_str = match extract_data_line(trimmed) {
        Some(s) => s,
        None => return Ok(None),
    };
    if data_str.trim() == "[DONE]" {
        return Ok(None);
    }
    if data_str.is_empty() {
        return Ok(None);
    }

    let value: Value = serde_json::from_str(&data_str).map_err(|e| {
        oxllm_core::Error::Transform(format!("invalid anthropic stream chunk: {e}"))
    })?;

    let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let result = match event_type {
        "message_start" => {
            let new_meta = extract_stream_meta(&value);
            let role_chunk = build_role_chunk(&value, &new_meta);
            // Clear before writing: a new stream's `message_start` supersedes
            // whatever a previous stream stashed.
            *meta = None;
            *meta = Some(new_meta);
            // Clear the per-stream tool-block tracker so a new stream never
            // inherits a prior stream's blocks.
            tool_blocks.clear();
            Ok(Some(role_chunk))
        }
        "content_block_start" => Ok(build_content_block_start_chunk(
            &value,
            tool_blocks,
            reverse_map,
        )),
        "content_block_delta" => Ok(build_delta_chunk(&value, tool_blocks)),
        "content_block_stop" => Ok(build_content_block_stop_chunk(&value, tool_blocks)),
        "message_delta" => Ok(Some(build_finish_chunk(&value, meta))),
        "error" => {
            // Anthropic emits mid-stream error events, e.g. when overloaded:
            // `{"type":"error","error":{"type":"overloaded_error","message":"..."}}`.
            // These must surface as `Err` so the stream pump aborts and
            // `post_failure` fires, rather than being silently dropped and
            // making the stream appear to complete cleanly.
            let err_obj = value.get("error");
            let message = err_obj
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("anthropic stream error")
                .to_string();
            let error_type = err_obj.and_then(|e| e.get("type")).and_then(|v| v.as_str());
            let status = anthropic_error_status(error_type);
            Err(oxllm_core::Error::Upstream { status, message })
        }
        _ => Ok(None),
    };

    // OpenAI clients expect every stream chunk to carry `id`, `created`, and
    // `model`. The `message_start` event already populates them; for every
    // other event we inject the values stashed from message_start.
    let mut chunk_value = result?;
    if let Some(out) = &mut chunk_value {
        if let Some(m) = meta.as_ref() {
            inject_stream_meta(out, m);
        }
    }
    Ok(chunk_value)
}

fn extract_stream_meta(value: &Value) -> StreamMeta {
    let message = value.get("message");
    let id = message
        .and_then(|m| m.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = message
        .and_then(|m| m.get("model"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let usage = message.and_then(|m| m.get("usage"));
    let cache_read_input_tokens = usage
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64());
    let cache_creation_input_tokens = usage
        .and_then(|u| u.get("cache_creation_input_tokens"))
        .and_then(|v| v.as_u64());
    let cache_creation_buckets = usage
        .and_then(|u| u.get("cache_creation"))
        .and_then(|c| c.as_object());
    let ephemeral_5m_input_tokens = cache_creation_buckets
        .and_then(|b| b.get("ephemeral_5m_input_tokens"))
        .and_then(|v| v.as_u64());
    let ephemeral_1h_input_tokens = cache_creation_buckets
        .and_then(|b| b.get("ephemeral_1h_input_tokens"))
        .and_then(|v| v.as_u64());
    // M2: compute the prompt token count (input_tokens + cache_read +
    // cache_creation, matching the non-streaming transform and the Bedrock
    // Converse path) once on `message_start` and stash it in StreamMeta so
    // `build_finish_chunk` can emit the real value instead of a hardcoded 0.
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let prompt_tokens = input_tokens
        .saturating_add(cache_read_input_tokens.unwrap_or(0))
        .saturating_add(cache_creation_input_tokens.unwrap_or(0));
    StreamMeta {
        id,
        model,
        created: unix_now(),
        cache_read_input_tokens,
        cache_creation_input_tokens,
        ephemeral_5m_input_tokens,
        ephemeral_1h_input_tokens,
        prompt_tokens,
        speed: message
            .and_then(|m| m.get("speed"))
            .and_then(Value::as_str)
            .map(str::to_string),
        service_tier: message
            .and_then(|m| m.get("service_tier"))
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

/// Inject `id` / `created` / `model` from a stashed [`StreamMeta`] into an
/// already-built chunk. Only mutates objects; leaves arrays/scalars untouched.
fn inject_stream_meta(chunk: &mut Value, meta: &StreamMeta) {
    if let Value::Object(map) = chunk {
        map.insert("id".to_string(), Value::String(meta.id.clone()));
        map.insert("created".to_string(), json!(meta.created));
        map.insert("model".to_string(), Value::String(meta.model.clone()));
        if let Some(speed) = &meta.speed {
            map.insert("speed".into(), json!(speed));
        }
        if let Some(tier) = &meta.service_tier {
            map.insert("service_tier".into(), json!(tier));
        }
    }
}

fn build_role_chunk(value: &Value, meta: &StreamMeta) -> Value {
    let usage = value.get("message").and_then(|m| m.get("usage"));
    // `text_tokens` mirrors Anthropic's `input_tokens` (cache hits excluded),
    // so read it separately from the role-chunk usage.
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_read = usage
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64());
    let cache_creation = usage
        .and_then(|u| u.get("cache_creation_input_tokens"))
        .and_then(|v| v.as_u64());
    // M2: the prompt token count was already computed in `extract_stream_meta`
    // (input_tokens + cache_read + cache_creation, matching the non-streaming
    // transform) and stashed in `meta.prompt_tokens`. Use it directly so the
    // role chunk and finish chunk agree.
    let prompt_tokens = meta.prompt_tokens;

    let mut usage_obj = Map::new();
    usage_obj.insert("prompt_tokens".to_string(), json!(prompt_tokens));
    usage_obj.insert("completion_tokens".to_string(), json!(0));
    usage_obj.insert("total_tokens".to_string(), json!(prompt_tokens));
    if let Some(v) = cache_read {
        usage_obj.insert("cache_read_input_tokens".to_string(), json!(v));
    }
    // No top-level `cache_creation_input_tokens` mirror: the host extractor
    // treats that form as additive (outside `prompt_tokens`) and would
    // double-bill the cache-write tokens that `meta.prompt_tokens` already
    // folds in. The `prompt_tokens_details.cache_creation_tokens` subset
    // mirror below is the only form emitted.
    // P4: Always populate `prompt_tokens_details.text_tokens` in the stream
    // role chunk, matching the non-streaming transform. Then mirror cache
    // counters under the same details object when present.
    let cache_creation_buckets = usage
        .and_then(|u| u.get("cache_creation"))
        .and_then(|c| c.as_object());
    // Same mirror as `transform_response`: stash both cache counters under
    // their OpenAI-standard `prompt_tokens_details` slot so downstream
    // extractors take the OpenAI branch and never re-fold the top-level
    // fields (no double-count).
    {
        let details = usage_obj
            .entry("prompt_tokens_details".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(details_map) = details {
            details_map.insert("text_tokens".to_string(), json!(input_tokens));
            if let Some(v) = cache_read {
                details_map.insert("cached_tokens".to_string(), json!(v));
            }
            if let Some(v) = cache_creation {
                details_map.insert("cache_creation_tokens".to_string(), json!(v));
            }
            if let Some(buckets) = cache_creation_buckets {
                for k in ["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"] {
                    if let Some(v) = buckets.get(k).and_then(|v| v.as_u64()) {
                        details_map.insert(k.to_string(), json!(v));
                    }
                }
            }
        }
    }

    json!({
    "id": meta.id,
    "object": "chat.completion.chunk",
    "created": meta.created,
    "model": meta.model,
    "choices": [{
    "index": 0,
    "delta": {"role": "assistant", "content": ""},
    "finish_reason": null,
    }],
    "usage": Value::Object(usage_obj),
    })
}

/// Map Anthropic `content_block_start` for `tool_use` blocks to an OpenAI
/// streaming `tool_calls` start chunk (id + name + empty arguments). Text
/// and `thinking` block starts are silent — text arrives via `text_delta`
/// and thinking arrives via `thinking_delta`. Tool names are run through the
/// reverse map so callers see the original (pre-sanitization) name they
/// registered, even though Anthropic saw the sanitized form.
///
/// `server_tool_use` and `*_tool_result` blocks (web_search, code execution,
/// etc.) are also silent: their argument fragments (`input_json_delta`) are
/// not OpenAI tool calls and would surface as phantom tool_calls if
/// forwarded. We register the index with `is_server_tool: true` so
/// [`build_delta_chunk`] can suppress them.
fn build_content_block_start_chunk(
    value: &Value,
    tool_blocks: &mut HashMap<u64, ToolBlockState>,
    reverse_map: &HashMap<String, String>,
) -> Option<Value> {
    let block = value.get("content_block")?;
    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0);

    if block_type == "redacted_thinking" {
        return Some(json!({
            "object":"chat.completion.chunk",
            "choices":[{"index":0,"delta":{"thinking_blocks":[block]},"finish_reason":null}]
        }));
    }

    // server_tool_use (web_search, web_fetch, code execution server tools) and
    // *_tool_result blocks (the results coming back from those server tools)
    // are NOT OpenAI tool_calls — they would emit phantom argument deltas via
    // input_json_delta. Register the index so the delta handler can suppress
    // them, then bail without emitting a chunk.
    if block_type == "server_tool_use" || block_type.ends_with("_tool_result") {
        tool_blocks.insert(
            index,
            ToolBlockState {
                is_server_tool: true,
                accumulated_args: String::new(),
            },
        );
        return None;
    }
    if block_type != "tool_use" {
        return None;
    }
    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let raw_name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let name = reverse_lookup_in(raw_name, reverse_map);

    tool_blocks.insert(
        index,
        ToolBlockState {
            is_server_tool: false,
            accumulated_args: String::new(),
        },
    );
    // Preserve the Anthropic `caller` field (programmatic tool calling) on
    // the OpenAI tool_call start chunk so the caller-visible identity is
    // consistent with the non-streaming response.
    let caller = block.get("caller").cloned();
    let mut tool_call = json!({
    "index": index,
    "id": id,
    "type": "function",
    "function": {
    "name": name,
    "arguments": "",
    },
    });
    if let Some(caller_val) = caller {
        if let Value::Object(map) = &mut tool_call {
            map.insert("caller".to_string(), caller_val);
        }
    }

    Some(json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {
    "tool_calls": [tool_call],
    },
    "finish_reason": null,
    }],
    }))
}

fn build_delta_chunk(
    value: &Value,
    tool_blocks: &mut HashMap<u64, ToolBlockState>,
) -> Option<Value> {
    let delta = value.get("delta")?;
    let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // Tool argument fragments: accumulate client-side via tool_calls[].index.
    if delta_type == "input_json_delta" || delta.get("partial_json").is_some() {
        let partial = delta
            .get("partial_json")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0);

        // `input_json_delta` for `server_tool_use` / `*_tool_result`
        // blocks is the server-side argument assembly for hosted tools (web
        // search queries, code execution payloads) — not an OpenAI tool_call.
        // Suppress it entirely so streaming clients don't see phantom calls.
        let is_server_tool = if let Some(state) = tool_blocks.get_mut(&index) {
            state.accumulated_args.push_str(partial);
            state.is_server_tool
        } else {
            false
        };
        if is_server_tool {
            return None;
        }
        if partial.is_empty() {
            return None;
        }
        return Some(json!({
        "object": "chat.completion.chunk",
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{
        "index": index,
        "function": {
        "arguments": partial,
        },
        }],
        },
        "finish_reason": null,
        }],
        }));
    }

    // Thinking deltas (extended reasoning): Anthropic emits
    // `delta.type == "thinking_delta"` with `delta.thinking` carrying the
    // next reasoning fragment. Surface as `reasoning_content` in the OpenAI
    // delta, matching DeepSeek/Grok streaming so reasoning clients work
    // unchanged across providers.
    //
    // U24: Anthropic occasionally emits a
    // single `content_block_delta` event carrying BOTH a `thinking_delta`
    // and a `signature_delta` (the digital signature for the thinking
    // block). The previous code dispatched on `delta.type` as a single
    // string, so only one of the two pieces was surfaced and the other
    // was silently dropped. We now check for both fields regardless of
    // `delta.type` and emit a single delta carrying whichever pieces are
    // present (one, the other, or both). This keeps the stream from
    // losing the signature (or the thinking text) without erroring.
    if delta_type == "thinking_delta" || delta_type == "signature_delta" {
        let thinking = delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
        let signature = delta
            .get("signature")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if thinking.is_empty() && signature.is_empty() {
            return None;
        }
        let mut delta_obj = Map::new();
        if !thinking.is_empty() {
            delta_obj.insert("reasoning_content".to_string(), json!(thinking));
        }
        if !signature.is_empty() {
            // Feature #8: preserve the signature on
            // `delta.thinking_blocks[].signature` so multi-turn replay
            // clients can reassemble it.
            delta_obj.insert(
                "thinking_blocks".to_string(),
                json!([{"type": "thinking", "signature": signature}]),
            );
        }
        return Some(json!({
        "object": "chat.completion.chunk",
        "choices": [{
        "index": 0,
        "delta": Value::Object(delta_obj),
        "finish_reason": null,
        }],
        }));
    }

    // Citation deltas: Anthropic streams citations one at a time via
    // `delta.type == "citations_delta"` with a `delta.citation` object
    // (document_location / char_location / page_location / etc.). Forward
    // each one under `delta.citations` so streaming clients can render cited
    // responses incrementally.
    if delta_type == "citations_delta" {
        if let Some(citation) = delta.get("citation") {
            return Some(json!({
            "object": "chat.completion.chunk",
            "choices": [{
            "index": 0,
            "delta": {"citations": [citation]},
            "finish_reason": null,
            }],
            }));
        }
        return None;
    }

    // Text deltas (type text_delta, or bare `text` for forward compat).
    let text = delta.get("text").and_then(|v| v.as_str())?;
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

/// Handle `content_block_stop` (feature #7).
///
/// When a `tool_use` block stops with no accumulated `input_json_delta`
/// arguments, Anthropic still means "call the tool with an empty object".
/// Without a final arguments delta, downstream streaming accumulators would
/// see a malformed empty-arguments tool call, so emit one closing
/// `arguments: "{}"` delta for that index. Non-empty tool blocks produce
/// nothing here. The block's tracked state is dropped once it stops.
fn build_content_block_stop_chunk(
    value: &Value,
    tool_blocks: &mut HashMap<u64, ToolBlockState>,
) -> Option<Value> {
    let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
    let state = tool_blocks.remove(&index);
    match state {
        // Only emit a closing `arguments: "{}"` for real tool_use blocks
        // (is_server_tool=false) that received no
        // argument fragments. server_tool / *_tool_result blocks never
        // produced a tool-call start chunk, so they must NOT emit a closing
        // arguments delta either.
        Some(ToolBlockState {
            is_server_tool: false,
            accumulated_args,
        }) if accumulated_args.is_empty() => Some(json!({
        "object": "chat.completion.chunk",
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{
        "index": index,
        "function": {"arguments": "{}"},
        }],
        },
        "finish_reason": null,
        }],
        })),
        _ => None,
    }
}

fn build_finish_chunk(value: &Value, meta: &Option<StreamMeta>) -> Value {
    let stop_reason = value
        .get("delta")
        .and_then(|d| d.get("stop_reason"))
        .and_then(|v| v.as_str());
    let finish_reason = stop_reason.map(map_stop_reason_to_openai);

    let mut chunk = json!({
    "object": "chat.completion.chunk",
    "choices": [{
    "index": 0,
    "delta": {},
    "finish_reason": finish_reason,
    }],
    });

    // GovCloud fix: some Bedrock regions emit cache token counts ONLY on
    // `message_start` (stashed in the stream state), not on `message_delta`.
    // When the finish event's usage lacks them, fall back to the stashed
    // values so cache accounting isn't lost on the final chunk.
    let stashed = meta.clone();
    // M2: the finish event's usage carries only `output_tokens`; the prompt
    // token count lives on `message_start` (stashed in the stream state).
    // Emit the real value when present so the finish chunk's `prompt_tokens`
    // isn't 0.
    let stashed_prompt_tokens = stashed.as_ref().map(|m| m.prompt_tokens).unwrap_or(0);
    let usage = value.get("usage").map(|u| {
        let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        // Anthropic reports reasoning tokens on the `message_delta` usage via
        // `completion_tokens_details.reasoning_tokens` (thinking models).
        // Forward them into the OpenAI-standard location so streaming clients
        // see the same shape as non-streaming ones.
        let reasoning = anthropic_reasoning_tokens(u);
        let mut usage_obj = json!({
        "prompt_tokens": stashed_prompt_tokens,
        "completion_tokens": output,
        "total_tokens": stashed_prompt_tokens.saturating_add(output),
        });
        if let Some(r) = reasoning {
            if let Value::Object(map) = &mut usage_obj {
                map.insert(
                    "completion_tokens_details".to_string(),
                    json!({"reasoning_tokens": r}),
                );
            }
        }
        // Mirror hosted server-tool usage (web search) so streaming requests
        // bill the same per-query fees as non-streaming ones; the final counts
        // ride on the `message_delta` usage object.
        if let Some(server_tool_use) = u.get("server_tool_use").filter(|v| !v.is_null()) {
            if let Value::Object(map) = &mut usage_obj {
                map.insert("server_tool_use".to_string(), server_tool_use.clone());
            }
        }
        // Merge stashed cache tokens when the finish event lacks them
        // (GovCloud path). When present on the finish event itself, the
        // event's value wins.
        if let Value::Object(map) = &mut usage_obj {
            let cache_read = u
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64())
                .or(stashed.as_ref().and_then(|m| m.cache_read_input_tokens));
            let cache_creation = u
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_u64())
                .or(stashed.as_ref().and_then(|m| m.cache_creation_input_tokens));
            if let Some(v) = cache_read {
                map.insert("cache_read_input_tokens".to_string(), json!(v));
            }
            // No top-level `cache_creation_input_tokens` mirror (same
            // double-bill rationale as `transform_response` /
            // `build_role_chunk`): the subset
            // `prompt_tokens_details.cache_creation_tokens` form below is
            // the only one emitted.
            // Mirror into prompt_tokens_details (OpenAI-standard path), only
            // when the finish event didn't already provide a details object.
            let has_ptd = map
                .get("prompt_tokens_details")
                .and_then(|v| v.as_object())
                .map(|o| {
                    o.contains_key("cached_tokens")
                        || o.contains_key("cache_creation_tokens")
                        || o.contains_key("ephemeral_5m_input_tokens")
                        || o.contains_key("ephemeral_1h_input_tokens")
                })
                .unwrap_or(false);
            let stashed_has_buckets = stashed
                .as_ref()
                .map(|m| {
                    m.ephemeral_5m_input_tokens.is_some() || m.ephemeral_1h_input_tokens.is_some()
                })
                .unwrap_or(false);
            if !has_ptd && (cache_read.is_some() || cache_creation.is_some() || stashed_has_buckets)
            {
                let mut details_map = Map::new();
                if let Some(v) = cache_read {
                    details_map.insert("cached_tokens".to_string(), json!(v));
                }
                if let Some(v) = cache_creation {
                    details_map.insert("cache_creation_tokens".to_string(), json!(v));
                }
                // Forward the tiered cache-creation buckets too when present.
                if let Some(meta) = &stashed {
                    if let Some(v) = meta.ephemeral_5m_input_tokens {
                        details_map.insert("ephemeral_5m_input_tokens".to_string(), json!(v));
                    }
                    if let Some(v) = meta.ephemeral_1h_input_tokens {
                        details_map.insert("ephemeral_1h_input_tokens".to_string(), json!(v));
                    }
                }
                map.insert(
                    "prompt_tokens_details".to_string(),
                    Value::Object(details_map),
                );
            }
        }
        usage_obj
    });
    if let Some(usage) = usage {
        chunk["usage"] = usage;
    }
    chunk
}

fn anthropic_reasoning_tokens(usage: &Value) -> Option<u64> {
    usage
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .pointer("/output_tokens_details/thinking_tokens")
                .and_then(Value::as_u64)
        })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod native_output_tests;

pub mod count_tokens;

#[cfg(test)]
mod validation_tests;
