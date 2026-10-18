//! AWS Bedrock CountTokens API (`POST /model/{modelId}/count-tokens`).
//!
//! The utility token counter (`/utils/token_counter?call_endpoint=true`)
//! previously had no Bedrock hosted counter: a Bedrock deployment fell
//! through to the local estimator with HTTP 200 even when the provider
//! would have rejected the request, silently masking bad credentials and
//! malformed input. This module owns the wire pieces — endpoint URL and
//! request-body construction — mirroring the reference Bedrock token
//! counter (`bedrock/count_tokens/`): Converse input for Converse-routed
//! deployments, base64 InvokeModel input otherwise, response
//! `{"inputTokens": n}`.
//!
//! SigV4 signing stays with the host (it signs the serialized body in the
//! upstream pipeline), exactly like chat. The reference's bedrock-mantle
//! fallback for Claude models the runtime CountTokens API rejects with
//! 400/404 is not wired: oxllm's mantle surface is the OpenAI-compatible
//! endpoint, which has no count-tokens operation to fall back to.

use base64::Engine;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::ChatProvider;
use oxllm_core::{Error, Result};
use percent_encoding::utf8_percent_encode;
use serde_json::{Value, json};

use super::{
    BedrockProvider, MODEL_PATH_SEGMENT, converse, should_use_converse, strip_model_suffixes,
};

/// Tokenizer label reported for counts served by this API (the reference
/// reports `bedrock_api` for the runtime CountTokens counter).
pub const TOKENIZER_TYPE: &str = "bedrock_api";

/// The CountTokens endpoint URL for a deployment. Base/region resolution
/// mirrors [`BedrockProvider::endpoint_url`] so the host matches the
/// SigV4 region; the model segment uses `requested_model` when the
/// deployment is a wildcard, exactly like chat resolution.
pub fn count_tokens_url(deployment: &Deployment, requested_model: Option<&str>) -> Result<String> {
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
    // A configured api_base may already carry the operation path (the
    // shared count-tokens URL tests do this for partner hosts); never
    // double it.
    if base.ends_with("/count-tokens") {
        return Ok(base.to_string());
    }
    let model = resolve_count_model(deployment, requested_model);
    let model = strip_model_suffixes(&model);
    let model_encoded = utf8_percent_encode(&model, MODEL_PATH_SEGMENT);
    Ok(format!("{base}/model/{model_encoded}/count-tokens"))
}

/// Resolve the upstream model id for counting: the deployment's model,
/// or the requested model for wildcard deployments (chat semantics).
fn resolve_count_model(deployment: &Deployment, requested_model: Option<&str>) -> String {
    let request = match requested_model {
        Some(model) => json!({"model": model}),
        None => Value::Null,
    };
    BedrockProvider::resolve_model(&request, deployment).to_string()
}

/// Build the CountTokens request body from an OpenAI-shaped chat body.
///
/// Converse deployments reuse the chat Converse transform and keep only
/// the members the CountTokens converse input carries (`messages`,
/// `system`, `toolConfig` — generation-shaped members such as
/// `inferenceConfig` are dropped, matching the reference transform).
/// InvokeModel deployments forward the exact invoke body the chat path
/// would send, base64-encoded per the API contract.
pub fn build_count_tokens_body(request: &Value, deployment: &Deployment) -> Result<Value> {
    let model = BedrockProvider::resolve_model(request, deployment);
    if should_use_converse(model, BedrockProvider::use_converse(deployment)) {
        let converse_body =
            converse::transform_request_converse(request, model, &deployment.oxllm_params.extra)?;
        let mut converse_input = serde_json::Map::new();
        for key in ["messages", "system", "toolConfig"] {
            if let Some(value) = converse_body.get(key) {
                converse_input.insert(key.to_string(), value.clone());
            }
        }
        if !converse_input.contains_key("messages") {
            return Err(Error::BadRequest(
                "Bedrock CountTokens requires messages".into(),
            ));
        }
        Ok(json!({"input": {"converse": Value::Object(converse_input)}}))
    } else {
        let invoke_body = ChatProvider::transform_request(&BedrockProvider, request, deployment)?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(invoke_body.to_string());
        Ok(json!({"input": {"invokeModel": {"body": encoded}}}))
    }
}

/// Extract `inputTokens` from a CountTokens response body.
pub fn parse_count_tokens_response(body: &Value) -> Option<u64> {
    body.get("inputTokens").and_then(Value::as_u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;

    fn deployment(model: &str) -> Deployment {
        Deployment {
            model_name: "counted".to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn count_tokens_url_uses_runtime_host_region_and_encoded_model() {
        let mut dep = deployment("bedrock/anthropic.claude-sonnet-4-20250514-v1:0");
        dep.oxllm_params
            .extra
            .insert("aws_region".to_string(), json!("eu-west-1"));
        let url = count_tokens_url(&dep, None).unwrap();
        assert_eq!(
            url,
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-sonnet-4-20250514-v1:0/count-tokens"
        );
    }

    #[test]
    fn count_tokens_url_honors_api_base_and_wildcard_model() {
        let mut dep = deployment("bedrock/*");
        dep.oxllm_params.api_base = Some("https://example.test/runtime/".to_string());
        let url = count_tokens_url(&dep, Some("bedrock/meta.llama3-70b-instruct-v1:0")).unwrap();
        assert_eq!(
            url,
            "https://example.test/runtime/model/meta.llama3-70b-instruct-v1:0/count-tokens"
        );
    }

    #[test]
    fn converse_count_body_keeps_only_countable_members() {
        // Counting follows the deployment's chat routing: Converse when
        // the deployment opted in (or the model family requires it).
        let mut dep = deployment("bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0");
        dep.oxllm_params
            .extra
            .insert("use_converse_api".to_string(), json!(true));
        let request = json!({
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "look up",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "max_tokens": 64,
            "temperature": 0.2
        });
        let body = build_count_tokens_body(&request, &dep).unwrap();
        let converse = &body["input"]["converse"];
        assert_eq!(converse["messages"][0]["role"], "user");
        assert_eq!(converse["messages"][0]["content"][0]["text"], "hello");
        assert!(
            converse.get("system").is_some(),
            "system must count: {body}"
        );
        assert!(
            converse.get("toolConfig").is_some(),
            "tools must count: {body}"
        );
        assert!(
            converse.get("inferenceConfig").is_none(),
            "generation config is not part of the count input: {body}"
        );
    }

    #[test]
    fn invoke_count_body_wraps_exact_invoke_payload() {
        let dep = deployment("bedrock/anthropic.claude-sonnet-4-20250514-v1:0");
        let request = json!({
            "model": "anthropic.claude-sonnet-4-20250514-v1:0",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64
        });
        let body = build_count_tokens_body(&request, &dep).unwrap();
        let encoded = body["input"]["invokeModel"]["body"].as_str().unwrap();
        let decoded: Value = serde_json::from_slice(
            &base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap(),
        )
        .unwrap();
        let direct = ChatProvider::transform_request(&BedrockProvider, &request, &dep).unwrap();
        assert_eq!(
            decoded, direct,
            "invoke body must be the chat invoke payload"
        );
    }

    #[test]
    fn parse_response_reads_input_tokens() {
        assert_eq!(
            parse_count_tokens_response(&json!({"inputTokens": 42})),
            Some(42)
        );
        assert_eq!(
            parse_count_tokens_response(&json!({"input_tokens": 42})),
            None
        );
        assert_eq!(parse_count_tokens_response(&json!({})), None);
    }
}
