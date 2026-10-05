//! DeepSeek provider — uses DeepSeek's OpenAI-compatible Chat Completions endpoint.
//!
//! Wire target: `https://api.deepseek.com/beta/chat/completions`.
//!
//! The default base URL mirrors LiteLLM's `DeepSeekChatConfig` (`/beta`),
//! which is the endpoint that accepts the native `thinking` parameter this
//! module maps from `reasoning_effort`. DeepSeek has since promoted most
//! features to `/v1`, but `/beta` remains the safe default for thinking-mode
//! parity; operators can override via `api_base`.
//!
//! DeepSeek exposes an OpenAI-compatible surface, so URL building, auth, stream
//! parsing and the supported-params list are delegated to the shared
//! [`OpenAiCompatProvider`] engine. DeepSeek diverges on three request-shape
//! concerns that [`ChatProvider::transform_request`] applies on top of the
//! shared body builder (parity with LiteLLM's `DeepSeekChatConfig`):
//!
//! - **`thinking` parameter mapping**: DeepSeek's native thinking control is
//!   `{"type": "enabled" | "disabled"}`. An OpenAI-style `reasoning_effort` is
//!   mapped to it (`"none"` → disabled, anything else → enabled); an explicit
//!   `thinking` dict always wins. `reasoning_effort` is consumed and never
//!   forwarded verbatim.
//! - **Non-function tools dropped**: DeepSeek only accepts `type: "function"`
//!   tools and rejects any other variant with a 400. Requests bridged from
//!   `/v1/responses` carrying other tool types are filtered to the function
//!   entries before sending.
//! - **Multi-turn thinking-mode `reasoning_content`**: when thinking mode is
//!   active, every assistant message in the history must carry a
//!   `reasoning_content` field or the API rejects the request. Assistant
//!   messages missing it are patched — promoting
//!   `provider_specific_fields.reasoning_content` when available, otherwise
//!   injecting a single-space placeholder.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use serde_json::{Map, Value};

/// The default DeepSeek OpenAI-compatible base URL. Uses `/beta` for LiteLLM
/// parity and thinking-mode support (`thinking={"type":"enabled"}`); the
/// `/v1` endpoint does not reliably accept the `thinking` parameter.
pub const DEFAULT_DEEPSEEK_BASE: &str = "https://api.deepseek.com/beta";

/// The provider prefix stripped from model ids (`deepseek/<model>`).
const STRIP: &str = "deepseek";

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the supported-params list. Only
/// [`ChatProvider::transform_request`] is overridden on the wrapper to apply
/// DeepSeek's thinking/tool/reasoning_content transforms on top of the shared
/// body builder.
const COMPAT: OpenAiCompatProvider =
    OpenAiCompatProvider::new("deepseek", DEFAULT_DEEPSEEK_BASE, STRIP, false, true, true);

/// The DeepSeek provider. Supports chat completions only.
#[derive(Debug, Clone, Copy)]
pub struct DeepSeekProvider;

/// Configured DeepSeek provider handle.
pub const PROVIDER: DeepSeekProvider = DeepSeekProvider;

/// Strip the `deepseek/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_deepseek_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Apply DeepSeek-specific request transforms to an already-built OpenAI chat
/// request body (the output of the shared compat `transform_request`).
/// `request` is the original client body; it is consulted for fields the
/// shared `build_chat_request` does not copy (notably the native `thinking`
/// parameter).
fn apply_deepseek_request_transforms(out: &mut Value, request: &Value) {
    let Some(obj) = out.as_object_mut() else {
        return;
    };

    map_thinking_param(obj, request);
    drop_non_function_tools(obj);

    let thinking_enabled = thinking_type(obj) == Some("enabled");
    let model = obj
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if thinking_mode_active(&model, thinking_enabled) {
        fill_reasoning_content(obj);
    }
}

/// Resolve DeepSeek's native `thinking` parameter, mapping OpenAI-style
/// `reasoning_effort` when no explicit `thinking` was sent (parity with
/// LiteLLM's `DeepSeekChatConfig.map_openai_params`). An explicit `thinking`
/// dict with a recognized `type` always wins; otherwise `reasoning_effort`
/// (`"none"` → disabled, anything else → enabled) is consumed.
/// `reasoning_effort` is never forwarded verbatim.
fn map_thinking_param(obj: &mut Map<String, Value>, request: &Value) {
    let explicit = request
        .get("thinking")
        .filter(|v| !v.is_null())
        .and_then(|v| v.as_object())
        .and_then(|t| t.get("type"))
        .and_then(|ty| ty.as_str())
        .filter(|ty| *ty == "enabled" || *ty == "disabled");

    let reasoning_effort = obj.remove("reasoning_effort").filter(|v| !v.is_null());
    let mapped = reasoning_effort.and_then(|eff| {
        eff.as_str()
            .map(|s| if s == "none" { "disabled" } else { "enabled" })
    });

    let thinking_type = explicit.or(mapped);
    if let Some(ty) = thinking_type {
        let mut t = Map::new();
        t.insert("type".to_string(), Value::String(ty.to_string()));
        obj.insert("thinking".to_string(), Value::Object(t));
    }
}

/// Read the resolved `thinking.type` value from the built body, if present.
fn thinking_type(obj: &Map<String, Value>) -> Option<&str> {
    obj.get("thinking")
        .and_then(|t| t.as_object())
        .and_then(|t| t.get("type"))
        .and_then(|ty| ty.as_str())
}

/// Filter the `tools` array to only `type: "function"` entries (parity with
/// LiteLLM's `DeepSeekChatConfig._drop_unsupported_tools`). DeepSeek rejects any
/// other tool variant with `unknown variant '<type>', expected 'function'`.
/// When no function tool survives, `tools`, `tool_choice`, and
/// `parallel_tool_calls` are all dropped; a `tool_choice` that named a dropped
/// tool is cleared.
fn drop_non_function_tools(obj: &mut Map<String, Value>) {
    let Some(tools) = obj.get("tools").and_then(|v| v.as_array()) else {
        return;
    };
    if tools.iter().all(is_function_tool) {
        return;
    }
    let function_tools: Vec<Value> = tools
        .iter()
        .filter(|t| is_function_tool(t))
        .cloned()
        .collect();
    obj.remove("tools");
    if function_tools.is_empty() {
        obj.remove("tool_choice");
        obj.remove("parallel_tool_calls");
        return;
    }
    let function_names: Vec<String> = function_tools
        .iter()
        .filter_map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(str::to_string)
        })
        .collect();
    let tool_choice_names_dropped = obj
        .get("tool_choice")
        .and_then(|tc| tc.as_object())
        .and_then(|tc| tc.get("function"))
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .is_some_and(|n| !function_names.iter().any(|fn_| fn_ == n));
    if tool_choice_names_dropped {
        obj.remove("tool_choice");
    }
    obj.insert("tools".to_string(), Value::Array(function_tools));
}

/// Whether a tool entry is a function tool (`type: "function"`).
fn is_function_tool(tool: &Value) -> bool {
    tool.as_object()
        .is_some_and(|t| t.get("type").and_then(|ty| ty.as_str()) == Some("function"))
}

/// Whether thinking mode is active for this request. True when the model is one
/// of DeepSeek's always-on reasoning models (`deepseek-reasoner`, `deepseek-r1`)
/// or the caller explicitly enabled thinking (`thinking.type == "enabled"`).
fn thinking_mode_active(model: &str, thinking_enabled: bool) -> bool {
    thinking_enabled || is_always_on_reasoning_model(model)
}

/// DeepSeek models whose reasoning is always on (so multi-turn history always
/// requires `reasoning_content` on prior assistant turns). Covers
/// `deepseek-reasoner` and `deepseek-r1` (and prefixed variants after the
/// `deepseek/` strip).
fn is_always_on_reasoning_model(model: &str) -> bool {
    model.contains("reasoner") || model.ends_with("-r1")
}

/// Inject `reasoning_content` on assistant messages that are missing it, for
/// thinking-mode multi-turn conversations (parity with LiteLLM's
/// `DeepSeekChatConfig._fill_reasoning_content`). DeepSeek's API rejects
/// requests where a prior assistant turn lacks `reasoning_content` with
/// *"The reasoning_content in the thinking mode must be passed back to the API."*
///
/// For each assistant message missing `reasoning_content`: promote it from
/// `provider_specific_fields.reasoning_content` if present, otherwise inject a
/// single-space placeholder (the minimum the API accepts).
fn fill_reasoning_content(obj: &mut Map<String, Value>) {
    let Some(messages) = obj.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for message in messages.iter_mut() {
        let Some(map) = message.as_object_mut() else {
            continue;
        };
        let is_assistant = map
            .get("role")
            .and_then(|r| r.as_str())
            .is_some_and(|r| r == "assistant");
        if !is_assistant {
            continue;
        }
        let already_has = map
            .get("reasoning_content")
            .and_then(|rc| rc.as_str())
            .is_some_and(|s| !s.is_empty());
        if already_has {
            continue;
        }
        let promoted = map
            .get("provider_specific_fields")
            .and_then(|f| f.as_object())
            .and_then(|f| f.get("reasoning_content"))
            .filter(|v| !v.is_null())
            .cloned();
        match promoted {
            Some(rc) => {
                map.insert("reasoning_content".to_string(), rc);
            }
            None => {
                map.insert(
                    "reasoning_content".to_string(),
                    Value::String(" ".to_string()),
                );
            }
        }
    }
}

impl ChatProvider for DeepSeekProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        ChatProvider::endpoint_url(&COMPAT, deployment)
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        ChatProvider::apply_auth(&COMPAT, deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let mut out = ChatProvider::transform_request(&COMPAT, request, deployment)?;
        apply_deepseek_request_transforms(&mut out, request);
        Ok(out)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        ChatProvider::transform_response(&COMPAT, body, model)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        ChatProvider::transform_stream_chunk(&COMPAT, chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        let mut params = ChatProvider::supported_params(&COMPAT).unwrap_or_default();
        params.push("thinking");
        Some(params)
    }
}

impl Provider for DeepSeekProvider {
    fn id(&self) -> ProviderId {
        "deepseek"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(endpoint, Endpoint::ChatCompletions)
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;
