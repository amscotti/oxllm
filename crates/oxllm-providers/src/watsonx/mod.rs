//! IBM WatsonX provider — chat completions.
//!
//! Wire target: `https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=…`
//!
//! WatsonX is NOT OpenAI-compatible. Its `/text/chat` endpoint expects a
//! `model_id` (not `model`) plus a `project_id` (or `space_id`). The response
//! is OpenAI-shaped and passes through. Authentication is a Bearer token,
//! normally exchanged from an IBM API key via IAM by the host (which hands the
//! resolved token to the transform via [`ResolvedCredentials`]). See
//! `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use percent_encoding::{CONTROLS, utf8_percent_encode};
use serde_json::Value;

use crate::openai_compat::parse_openai_stream_chunk;
use crate::util::apply_bearer_auth;

/// The default WatsonX base URL (US-South region).
pub const DEFAULT_WATSONX_BASE: &str = "https://us-south.ml.cloud.ibm.com";

/// The default WatsonX API version query parameter.
pub const DEFAULT_API_VERSION: &str = "2024-03-13";

/// The WatsonX provider.
#[derive(Debug, Clone, Default)]
pub struct WatsonXProvider;

/// Characters that must be percent-encoded inside a query-string value, so
/// an `api_version` containing `&` or `#` cannot inject extra query parameters
/// or a fragment. Matches the Azure provider's `QUERY_VALUE` set.
const QUERY_VALUE: &percent_encoding::AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'<')
    .add(b'>')
    .add(b'&')
    .add(b'=')
    .add(b'+')
    .add(b'%');

/// Strip the `watsonx/` or `watsonx_text/` provider prefix from a model id.
pub fn strip_watsonx_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("watsonx", rest)) => rest,
        Some(("watsonx_text", rest)) => rest,
        _ => model,
    }
}

/// Resolve the API version to send on the URL. Honors an `api_version` set on
/// the deployment's litellm params (extra), falling back to the default.
fn api_version(deployment: &Deployment) -> String {
    deployment
        .oxllm_params
        .api_version
        .clone()
        .or_else(|| {
            deployment
                .oxllm_params
                .extra
                .get("api_version")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| DEFAULT_API_VERSION.to_string())
}

/// Read `project_id` from the deployment's litellm params (named field or
/// `extra`), mirroring LiteLLM's `_get_api_params`.
fn project_id(deployment: &Deployment) -> Option<String> {
    deployment
        .oxllm_params
        .extra
        .get("project_id")
        .or_else(|| deployment.oxllm_params.extra.get("watsonx_project"))
        .and_then(value_to_string)
}

/// Read `space_id` from the deployment's litellm params.
fn space_id(deployment: &Deployment) -> Option<String> {
    deployment
        .oxllm_params
        .extra
        .get("space_id")
        .and_then(value_to_string)
}

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

/// Whether the resolved model is a WatsonX deployment reference
/// (`deployment/<id>`). Deployment models do not carry `project_id`/`space_id`.
fn is_deployment_model(model: &str) -> bool {
    model.starts_with("deployment/")
}

impl ChatProvider for WatsonXProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_WATSONX_BASE)
            .trim_end_matches('/');
        let model = strip_watsonx_prefix(&deployment.oxllm_params.model);
        // Percent-encode the version so a custom `api_version` containing
        // URL-special characters (`&`, `#`, `=`, spaces) cannot malform the
        // query string. The default (`2024-03-13`) encodes to itself.
        let version_raw = api_version(deployment);
        let version = utf8_percent_encode(&version_raw, QUERY_VALUE);
        if is_deployment_model(model) {
            // `deployment/<id>` → `/ml/v1/deployments/<id>/text/chat`
            let deployment_id = model.strip_prefix("deployment/").unwrap_or(model);
            format!("{base}/ml/v1/deployments/{deployment_id}/text/chat?version={version}")
        } else {
            format!("{base}/ml/v1/text/chat?version={version}")
        }
    }

    fn apply_auth(
        &self,
        _deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        apply_bearer_auth(creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let mut out = request.clone();
        let out_map = match &mut out {
            Value::Object(map) => map,
            _ => return Ok(out),
        };

        let configured = strip_watsonx_prefix(&deployment.oxllm_params.model);

        if is_deployment_model(configured) {
            // Deployment models: no model_id / project_id / space_id in payload.
            // The deployment id is encoded in the URL, not the body.
            out_map.remove("model");
        } else {
            // WatsonX names the model field `model_id` (not `model`). For a
            // wildcard deployment (`watsonx/*`), resolve the actual model from
            // the request body (prefix-stripped) so a wildcard routes whatever
            // the client sent — mirroring `OpenAiCompatProvider::resolve_model`.
            let is_wildcard = configured == "*" || configured.ends_with("/*");
            let model_id = if is_wildcard {
                request
                    .get("model")
                    .and_then(Value::as_str)
                    .map(|m| strip_watsonx_prefix(m).to_string())
                    .unwrap_or_else(|| configured.to_string())
            } else {
                configured.to_string()
            };
            out_map.remove("model");
            out_map.insert("model_id".to_string(), Value::String(model_id));
            // `project_id` wins over `space_id`; only one is sent.
            if let Some(project_id) = project_id(deployment) {
                out_map.insert("project_id".to_string(), Value::String(project_id));
            } else if let Some(space_id) = space_id(deployment) {
                out_map.insert("space_id".to_string(), Value::String(space_id));
            }
        }
        Ok(out)
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        // WatsonX `/text/chat` returns an OpenAI-shaped response.
        Ok(body.clone())
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        // WatsonX's `/text/chat` emits the standard OpenAI SSE shape, so the
        // shared parser handles multi-line event blocks, empty `data:` lines,
        // `[DONE]` variants, and inline upstream errors uniformly with every
        // other OpenAI-compat provider.
        parse_openai_stream_chunk(chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        Some(vec![
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "stream",
            "n",
            "tools",
            "tool_choice",
            "logprobs",
            "top_logprobs",
            "response_format",
            "reasoning_effort",
        ])
    }
}

impl Provider for WatsonXProvider {
    fn id(&self) -> ProviderId {
        "watsonx"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(endpoint, oxllm_core::Endpoint::ChatCompletions)
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;
