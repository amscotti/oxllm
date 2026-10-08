//! OpenRouter provider — uses OpenRouter's OpenAI-compatible endpoints.
//!
//! Wire targets:
//! - chat: `https://openrouter.ai/api/v1/chat/completions`
//! - embeddings: `https://openrouter.ai/api/v1/embeddings`
//!
//! Because OpenRouter exposes an OpenAI-compatible surface, the transform is
//! a near-identity. Two OpenRouter-specific behaviors:
//!
//! 1. **Optional ranking headers.** OpenRouter ranks apps by `HTTP-Referer`
//!    and displays app names from `X-Title`. These are forwarded only when an
//!    operator supplies them via the deployment's `extra` block
//!    (`http_referer` / `x_title` keys, written in the deployment YAML). oxllm
//!    does not populate them from environment variables; if absent, they are
//!    omitted.
//! 2. **Model string.** A model may be `openrouter/openai/gpt-4o` — only the
//!    first segment (`openrouter/`) is stripped; the remainder is the
//!    upstream id. We never strip further.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use serde_json::{Map, Value};

/// The default OpenRouter OpenAI-compatible base URL.
pub const DEFAULT_OPENROUTER_BASE: &str = "https://openrouter.ai/api/v1";

/// The OpenRouter provider.
#[derive(Debug, Clone, Default)]
pub struct OpenRouterProvider;

impl ChatProvider for OpenRouterProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_OPENROUTER_BASE)
            .trim_end_matches('/');
        // Defensive guard mirroring the reference URL builder (and the
        // shared `openai_compat` sibling): a caller who configures `api_base`
        // as the full endpoint URL must not get `/chat/completions` appended
        // twice.
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
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
                None => {
                    tracing::warn!(
                        target: "oxllm-providers",
                        model = %deployment.model_name,
                        "openrouter: no bearer credential available; \
                         request will be sent unauthenticated",
                    );
                    return;
                }
            },
        };
        headers.insert("authorization".to_string(), format!("Bearer {bearer}"));
        if let Some(referer) = deployment
            .oxllm_params
            .extra
            .get("http_referer")
            .and_then(|v| v.as_str())
        {
            headers.insert("http-referer".to_string(), referer.to_string());
        }
        if let Some(title) = deployment
            .oxllm_params
            .extra
            .get("x_title")
            .and_then(|v| v.as_str())
        {
            headers.insert("x-title".to_string(), title.to_string());
        }
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let mut out = Map::new();
        let model = resolve_openrouter_model(request, &deployment.oxllm_params.model);
        if let Some(messages) = request.get("messages") {
            if !messages.is_null() {
                out.insert("messages".to_string(), messages.clone());
            }
        }
        for key in crate::openai_compat::SUPPORTED_CHAT_PARAMS {
            // Always use deployment model (after openrouter/ strip), never the
            // client alias that may appear on the request body.
            if *key == "model" || *key == "messages" {
                continue;
            }
            if let Some(v) = request.get(*key) {
                if !v.is_null() {
                    out.insert((*key).to_string(), v.clone());
                }
            }
        }
        // M4: OpenRouter's reasoning ladder uses `"xhigh"`, not OpenAI's `"max"`.
        let needs_effort_remap = out
            .get("reasoning_effort")
            .and_then(|v| v.as_str())
            .is_some_and(|s| s == "max");
        if needs_effort_remap {
            out.insert(
                "reasoning_effort".to_string(),
                Value::String("xhigh".to_string()),
            );
        }
        // M3: OpenRouter-only top-level body fields. They are not part of the
        // shared OpenAI param list, so the loop above drops them — re-inject
        // them straight from the original request body.
        for key in ["transforms", "models", "route"] {
            if let Some(v) = request.get(key) {
                if !v.is_null() {
                    out.insert(key.to_string(), v.clone());
                }
            }
        }
        // M5: always ask OpenRouter to return cost/usage data.
        out.insert("usage".to_string(), serde_json::json!({"include": true}));
        out.insert("model".to_string(), Value::String(model));
        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let mut out = body.clone();
        // M6: OpenRouter emits assistant reasoning under `message.reasoning`
        // rather than the OpenAI `message.reasoning_content`. Normalize so
        // downstream consumers see one field name regardless of upstream, and
        // so the non-streaming path matches the streaming path.
        if let Some(choices) = out.get_mut("choices").and_then(|c| c.as_array_mut()) {
            for choice in choices.iter_mut() {
                let Some(message) = choice.get_mut("message").and_then(|m| m.as_object_mut())
                else {
                    continue;
                };
                rename_reasoning_to_reasoning_content(message);
            }
        }
        crate::openai::sanitize_null_tokens(&mut out);
        crate::openai_compat::map_generic_error_finish_reason(&mut out);
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        let mut value = match crate::openai_compat::sse::parse_openai_stream_chunk(chunk)? {
            Some(v) => v,
            None => return Ok(None),
        };
        // M6: OpenRouter emits assistant reasoning under `delta.reasoning`
        // rather than the OpenAI `delta.reasoning_content`. Normalize so
        // downstream consumers see one field name regardless of upstream.
        if let Some(choices) = value.get_mut("choices").and_then(|c| c.as_array_mut()) {
            for choice in choices.iter_mut() {
                let Some(delta) = choice.get_mut("delta").and_then(|d| d.as_object_mut()) else {
                    continue;
                };
                rename_reasoning_to_reasoning_content(delta);
            }
        }
        crate::openai::sanitize_null_tokens(&mut value);
        crate::openai_compat::map_generic_error_finish_reason(&mut value);
        Ok(Some(value))
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        let mut params = crate::openai_compat::SUPPORTED_CHAT_PARAMS.to_vec();
        params.extend_from_slice(&["transforms", "models", "route", "usage"]);
        Some(params)
    }
}

impl EmbeddingProvider for OpenRouterProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_OPENROUTER_BASE)
            .trim_end_matches('/');
        // Defensive guard mirroring `endpoint_url` above: a caller who
        // configures `api_base` as the full embeddings endpoint URL must not
        // get `/embeddings` appended twice.
        if base.ends_with("/embeddings") {
            base.to_string()
        } else {
            format!("{base}/embeddings")
        }
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
        let model = resolve_openrouter_model(request, &deployment.oxllm_params.model);
        out.insert("model".to_string(), Value::String(model));
        if let Some(v) = request.get("input").filter(|value| !value.is_null()) {
            out.insert("input".to_string(), v.clone());
        }
        for key in ["encoding_format", "dimensions", "user"] {
            if let Some(v) = request.get(key) {
                if !v.is_null() {
                    out.insert(key.to_string(), v.clone());
                }
            }
        }
        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let mut out = body.clone();
        crate::openai::sanitize_null_tokens(&mut out);
        Ok(out)
    }
}

impl Provider for OpenRouterProvider {
    fn id(&self) -> ProviderId {
        "openrouter"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(
            endpoint,
            oxllm_core::Endpoint::ChatCompletions | oxllm_core::Endpoint::Embeddings
        )
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        Some(self)
    }
}

/// Strip the first leading `openrouter/` segment from a model id. The
/// remainder is the upstream id OpenRouter routes (which itself may contain
/// a slash, e.g. `openai/gpt-4o`), so we never strip further.
fn strip_openrouter_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("openrouter", rest)) => rest,
        _ => model,
    }
}

/// Rename a `reasoning` key to the OpenAI `reasoning_content` convention.
///
/// OpenRouter emits assistant reasoning under `reasoning` in both streaming
/// deltas and non-streaming messages; oxllm normalises to `reasoning_content`
/// (the OpenAI/DeepSeek field) so downstream sees one name. An existing
/// non-null `reasoning_content` is preserved (clients/providers that already
/// populate it win), but an explicit `reasoning_content: null` is overwritten
/// — otherwise the removed `reasoning` payload would be silently dropped.
fn rename_reasoning_to_reasoning_content(obj: &mut Map<String, Value>) {
    let Some(reasoning) = obj.remove("reasoning") else {
        return;
    };
    let keep_existing = obj.get("reasoning_content").is_some_and(|v| !v.is_null());
    if !keep_existing {
        obj.insert("reasoning_content".to_string(), reasoning);
    }
}

/// Resolve the upstream model id for an OpenRouter request.
///
/// Honours wildcard deployments — `*` and `openrouter/*` mean "route whatever
/// the client sent". In that case the actual model id is read from the request
/// body (with only the first `openrouter/` prefix segment stripped, so
/// `openrouter/openai/gpt-4o` becomes `openai/gpt-4o`). For non-wildcard
/// deployments the configured value (after `openrouter/` strip) is used and
/// the client body is ignored, matching the OpenAI-compat behaviour.
fn resolve_openrouter_model(request: &Value, configured: &str) -> String {
    let stripped = strip_openrouter_prefix(configured);
    let is_wildcard = stripped == "*" || stripped.ends_with("/*");
    if is_wildcard {
        request
            .get("model")
            .and_then(Value::as_str)
            .map(|m| strip_openrouter_prefix(m).to_string())
            .unwrap_or_else(|| stripped.to_string())
    } else {
        stripped.to_string()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod swarm_tests {
    //! Coverage for the swarm-review fixes (M3 / M4 / M5 / M6). Lives in
    //! `mod.rs` because `tests.rs` is reserved for the pre-existing suite.
    use super::*;
    use oxllm_core::deployment::{Deployment, OxllmParams};
    use oxllm_core::provider::ChatProvider;
    use serde_json::json;

    fn deployment() -> Deployment {
        Deployment {
            model_name: "openai/gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openrouter/openai/gpt-4o".to_string(),
                api_key: Some("openrouter-key".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn openrouter_forwards_transforms_models_route_body_fields() {
        // M3: these three OpenRouter-only top-level fields must survive the
        // transform rather than being stripped by SUPPORTED_CHAT_PARAMS.
        let p = OpenRouterProvider;
        let req = json!({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "transforms": ["middle-out"],
            "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
            "route": "fallback",
        });
        let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
        assert_eq!(out["transforms"][0], "middle-out");
        assert_eq!(out["models"][0], "openai/gpt-4o");
        assert_eq!(out["models"][1], "anthropic/claude-3.5-sonnet");
        assert_eq!(out["route"], "fallback");
    }

    #[test]
    fn openrouter_omits_transforms_models_route_when_absent() {
        // M3 complement: no spurious nulls injected when the client omits them.
        let p = OpenRouterProvider;
        let req = json!({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        });
        let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
        assert!(out.get("transforms").is_none());
        assert!(out.get("models").is_none());
        assert!(out.get("route").is_none());
    }

    #[test]
    fn openrouter_maps_reasoning_effort_max_to_xhigh() {
        // M4: OpenRouter's top of the reasoning ladder is "xhigh", not "max".
        let p = OpenRouterProvider;
        let req = json!({
            "model": "openai/o1",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "max",
        });
        let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
        assert_eq!(out["reasoning_effort"], "xhigh");
    }

    #[test]
    fn openrouter_leaves_other_reasoning_effort_values_untouched() {
        let p = OpenRouterProvider;
        for value in ["low", "medium", "high", "xhigh"] {
            let req = json!({
                "model": "openai/o1",
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": value,
            });
            let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
            assert_eq!(out["reasoning_effort"], value);
        }
    }

    #[test]
    fn openrouter_always_injects_usage_include() {
        // M5: cost data must always be requested, even when the client sent no
        // `usage` field.
        let p = OpenRouterProvider;
        let req = json!({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        });
        let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
        assert_eq!(out["usage"]["include"], true);
    }

    #[test]
    fn openrouter_overwrites_client_usage_with_include_true() {
        // M5: the injected `usage` is authoritative — a client that sent a
        // different `usage` block is normalised.
        let p = OpenRouterProvider;
        let req = json!({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "usage": {"include": false},
        });
        let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
        assert_eq!(out["usage"]["include"], true);
    }

    #[test]
    fn openrouter_normalizes_delta_reasoning_to_reasoning_content() {
        // M6: OpenRouter streams reasoning under `delta.reasoning`; normalize
        // to `delta.reasoning_content` so callers see one field name.
        let p = OpenRouterProvider;
        let raw =
            r#"data: {"id":"x","choices":[{"index":0,"delta":{"reasoning":"thinking hard"}}]}"#;
        let out = p
            .transform_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        let delta = &out["choices"][0]["delta"];
        assert!(
            delta.get("reasoning").is_none(),
            "raw `reasoning` must be removed"
        );
        assert_eq!(delta["reasoning_content"], "thinking hard");
    }

    #[test]
    fn openrouter_does_not_overwrite_existing_reasoning_content() {
        // M6: if a chunk already carries `reasoning_content` (and a stray
        // `reasoning`), the existing value wins — we only fill the gap.
        let p = OpenRouterProvider;
        let raw = r#"data: {"id":"x","choices":[{"index":0,"delta":{"reasoning":"raw","reasoning_content":"kept"}}]}"#;
        let out = p
            .transform_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        let delta = &out["choices"][0]["delta"];
        assert!(delta.get("reasoning").is_none());
        assert_eq!(delta["reasoning_content"], "kept");
    }

    #[test]
    fn openrouter_leaves_content_delta_untouched() {
        // M6 complement: ordinary content deltas parse unchanged.
        let p = OpenRouterProvider;
        let raw = r#"data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        let out = p
            .transform_stream_chunk(raw)
            .expect("ok")
            .expect("should produce value");
        assert_eq!(out["choices"][0]["delta"]["content"], "hi");
        assert!(
            out["choices"][0]["delta"]
                .get("reasoning_content")
                .is_none()
        );
    }
}
