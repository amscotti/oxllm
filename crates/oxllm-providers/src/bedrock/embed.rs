//! Bedrock InvokeModel embeddings.
//!
//! Cohere Embed and Amazon Titan Embed are the two vendors this path
//! translates. Other model ids are rejected instead of being forwarded as
//! an OpenAI-shaped body Bedrock would reject.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{EmbeddingProvider, ResolvedCredentials};
use percent_encoding::utf8_percent_encode;
use serde_json::{Map, Value, json};

use super::{BedrockProvider, MODEL_PATH_SEGMENT, strip_model_suffixes};

/// Embeddings adapter. Kept off [`BedrockProvider`] so chat and embed
/// transforms, which share method names, stay unambiguous.
pub(super) struct Embeddings;

pub(super) static EMBEDDINGS: Embeddings = Embeddings;

impl EmbeddingProvider for Embeddings {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        embedding_url(deployment)
    }

    fn embedding_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        let mut resolved = deployment.clone();
        resolved.oxllm_params.model = format!(
            "bedrock/{}",
            BedrockProvider::resolve_model(request, deployment)
        );
        embedding_url(&resolved)
    }

    fn apply_auth(
        &self,
        _deployment: &Deployment,
        _creds: &ResolvedCredentials,
        _headers: &mut HashMap<String, String>,
    ) {
        // SigV4 is applied by the embeddings handler after the body bytes
        // exist, same as the chat path. There is no header to set here.
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        transform_request(request, deployment)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        transform_response(body, model)
    }
}

const COHERE_INPUT_TYPES: [&str; 4] = [
    "search_document",
    "search_query",
    "classification",
    "clustering",
];

/// InvokeModel URL for an embeddings deployment. Always `/invoke`.
/// Embeddings do not use the Converse path.
pub(super) fn embedding_url(deployment: &Deployment) -> String {
    let region = deployment
        .oxllm_params
        .extra
        .get("aws_region_name")
        .or_else(|| deployment.oxllm_params.extra.get("aws_region"))
        .and_then(Value::as_str)
        .filter(|region| !region.is_empty())
        .unwrap_or("us-east-1");
    let default_base = format!("https://bedrock-runtime.{region}.amazonaws.com");
    let base = deployment
        .oxllm_params
        .api_base
        .as_deref()
        .unwrap_or(&default_base)
        .trim_end_matches('/');
    let model = configured_model_id(deployment);
    let model_encoded = utf8_percent_encode(&model, MODEL_PATH_SEGMENT);
    format!("{base}/model/{model_encoded}/invoke")
}

pub(super) fn transform_request(request: &Value, deployment: &Deployment) -> Result<Value> {
    let model = strip_model_suffixes(BedrockProvider::resolve_model(request, deployment));
    let texts = embedding_texts(request.get("input").unwrap_or(&Value::Null))?;
    if texts.is_empty() {
        return Err(oxllm_core::Error::BadRequest(
            "bedrock embeddings require a text input".into(),
        ));
    }
    // `encoding_format` is an OpenAI field. Bedrock rejects it, so it is
    // never copied, even when the proxy-wide drop list is off.
    if is_cohere_embed(&model) {
        Ok(cohere_body(request, &texts))
    } else if is_titan_embed(&model) {
        if texts.len() != 1 {
            return Err(oxllm_core::Error::BadRequest(
                "amazon titan embed accepts a single text input".into(),
            ));
        }
        Ok(titan_body(request, &texts[0]))
    } else {
        Err(oxllm_core::Error::BadRequest(format!(
            "bedrock model '{model}' does not support embeddings"
        )))
    }
}

pub(super) fn transform_response(body: &Value, model: &str) -> Result<Value> {
    let vectors = vectors_from_body(body)?;
    if vectors.is_empty() {
        return Err(oxllm_core::Error::Transform(
            "bedrock embeddings response did not include vectors".into(),
        ));
    }
    if vectors.iter().any(|vector| {
        vector.as_array().is_none_or(|values| {
            values.is_empty() || values.iter().any(|value| value.as_f64().is_none())
        })
    }) {
        return Err(oxllm_core::Error::Transform(
            "bedrock embeddings response contains an invalid vector".into(),
        ));
    }
    let data: Vec<Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding,
            })
        })
        .collect();
    let prompt_tokens = body
        .get("inputTextTokenCount")
        .and_then(Value::as_u64)
        .or_else(|| {
            body.get("meta")
                .and_then(|meta| meta.get("billed_units"))
                .map(|units| {
                    units
                        .get("input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        .saturating_add(units.get("images").and_then(Value::as_u64).unwrap_or(0))
                })
        })
        .unwrap_or(0);
    Ok(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }))
}

fn configured_model_id(deployment: &Deployment) -> String {
    let model = deployment
        .oxllm_params
        .model
        .strip_prefix("bedrock/")
        .unwrap_or(&deployment.oxllm_params.model);
    strip_model_suffixes(model)
}

fn is_cohere_embed(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    super::model_id_for_family(&lower).is_some_and(|model| model.starts_with("cohere.embed-"))
}

fn is_titan_embed(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    super::model_id_for_family(&lower).is_some_and(|model| {
        model.starts_with("amazon.titan-embed-text-") || model == "amazon.titan-embed-g1-text-02"
    })
}

fn cohere_body(request: &Value, texts: &[String]) -> Value {
    let input_type = request
        .get("input_type")
        .and_then(Value::as_str)
        .filter(|value| COHERE_INPUT_TYPES.contains(value))
        .unwrap_or("search_document");
    let truncate = request
        .get("truncate")
        .and_then(Value::as_str)
        .filter(|value| matches!(*value, "NONE" | "START" | "END"))
        .unwrap_or("END");
    let mut out = Map::new();
    out.insert("texts".to_string(), json!(texts));
    out.insert("input_type".to_string(), json!(input_type));
    out.insert("truncate".to_string(), json!(truncate));
    if let Some(dimensions) = request.get("dimensions").filter(|value| !value.is_null()) {
        out.insert("output_dimension".to_string(), dimensions.clone());
    }
    Value::Object(out)
}

fn titan_body(request: &Value, text: &str) -> Value {
    let mut out = Map::new();
    out.insert("inputText".to_string(), json!(text));
    if let Some(dimensions) = request.get("dimensions").filter(|value| !value.is_null()) {
        out.insert("dimensions".to_string(), dimensions.clone());
    }
    Value::Object(out)
}

fn embedding_texts(input: &Value) -> Result<Vec<String>> {
    match input {
        Value::String(text) if !text.is_empty() => Ok(vec![text.clone()]),
        Value::Array(items) => {
            let mut texts = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(text) if !text.is_empty() => texts.push(text.clone()),
                    _ => {
                        return Err(oxllm_core::Error::BadRequest(
                            "bedrock embeddings require text input".into(),
                        ));
                    }
                }
            }
            Ok(texts)
        }
        _ => Err(oxllm_core::Error::BadRequest(
            "bedrock embeddings require text input".into(),
        )),
    }
}

fn vectors_from_body(body: &Value) -> Result<Vec<Value>> {
    if let Some(vector) = body.get("embedding").filter(|value| value.is_array()) {
        return Ok(vec![vector.clone()]);
    }
    let Some(embeddings) = body.get("embeddings") else {
        return Ok(Vec::new());
    };
    if let Some(rows) = embeddings.as_array() {
        return Ok(rows.clone());
    }
    if let Some(map) = embeddings.as_object() {
        let canonical = map.get("float").or_else(|| map.values().next());
        if let Some(rows) = canonical.and_then(Value::as_array) {
            return Ok(rows.clone());
        }
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;

    fn deployment(model: &str) -> Deployment {
        Deployment {
            model_name: "embed".to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn embedding_url_uses_us_east_1_and_invoke() {
        let dep = deployment("bedrock/us.cohere.embed-v4:0");
        assert_eq!(
            embedding_url(&dep),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.cohere.embed-v4:0/invoke"
        );
    }

    #[test]
    fn embedding_url_uses_aws_region_override() {
        let mut dep = deployment("bedrock/cohere.embed-v4:0");
        dep.oxllm_params
            .extra
            .insert("aws_region".into(), json!("us-west-2"));
        assert_eq!(
            embedding_url(&dep),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/cohere.embed-v4:0/invoke"
        );
    }

    #[test]
    fn cohere_body_drops_encoding_format_and_defaults_input_type() {
        let dep = deployment("bedrock/us.cohere.embed-v4:0");
        let request = json!({
            "model": "external-cohere-embed-v4",
            "input": ["hello"],
            "encoding_format": "float"
        });
        let out = transform_request(&request, &dep).expect("cohere body");
        assert_eq!(out["texts"], json!(["hello"]));
        assert_eq!(out["input_type"], "search_document");
        assert_eq!(out["truncate"], "END");
        assert!(out.get("encoding_format").is_none());
    }

    #[test]
    fn cohere_body_honors_input_type() {
        let dep = deployment("bedrock/cohere.embed-english-v3");
        let request = json!({"input": "query text", "input_type": "search_query"});
        let out = transform_request(&request, &dep).expect("cohere body");
        assert_eq!(out["texts"], json!(["query text"]));
        assert_eq!(out["input_type"], "search_query");
    }

    #[test]
    fn embedding_family_dispatch_preserves_foundation_and_inference_profile_arns() {
        for model in [
            "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-v4:0",
            "arn:aws:bedrock:us-west-2:123456789012:inference-profile/us.cohere.embed-v4:0",
        ] {
            let dep = deployment(&format!("bedrock/{model}"));
            let body = transform_request(&json!({"input": ["hello"]}), &dep).unwrap();
            assert_eq!(body["texts"], json!(["hello"]));
            assert!(embedding_url(&dep).contains("%2F"));
            assert!(embedding_url(&dep).ends_with("/invoke"));
        }
        let dep = deployment(
            "bedrock/arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
        );
        assert_eq!(
            transform_request(&json!({"input": "hello"}), &dep).unwrap()["inputText"],
            "hello"
        );
    }

    #[test]
    fn titan_body_uses_input_text() {
        let dep = deployment("bedrock/amazon.titan-embed-text-v2:0");
        let request = json!({"input": "hello", "dimensions": 256, "encoding_format": "float"});
        let out = transform_request(&request, &dep).expect("titan body");
        assert_eq!(out["inputText"], "hello");
        assert_eq!(out["dimensions"], 256);
        assert!(out.get("encoding_format").is_none());
    }

    #[test]
    fn unknown_embed_model_is_rejected() {
        let dep = deployment("bedrock/anthropic.claude-3-haiku");
        let request = json!({"input": "hello"});
        let err = transform_request(&request, &dep).expect_err("not an embed model");
        assert!(err.to_string().contains("does not support embeddings"));
    }

    #[test]
    fn cohere_typed_response_prefers_float_vectors() {
        let body = json!({
            "embeddings": {"float": [[0.1, 0.2]], "int8": [[1, 2]]},
            "response_type": "embeddings_by_type"
        });
        let out = transform_response(&body, "bedrock/cohere.embed-v4:0").expect("response");
        assert_eq!(out["data"][0]["embedding"], json!([0.1, 0.2]));
        assert_eq!(out["data"][0]["index"], 0);
        assert_eq!(out["usage"]["prompt_tokens"], 0);
    }

    #[test]
    fn titan_response_maps_token_count() {
        let body = json!({"embedding": [0.5, 0.25], "inputTextTokenCount": 3});
        let out = transform_response(&body, "amazon.titan-embed-text-v2:0").expect("response");
        assert_eq!(out["data"][0]["embedding"], json!([0.5, 0.25]));
        assert_eq!(out["usage"]["prompt_tokens"], 3);
        assert_eq!(out["usage"]["total_tokens"], 3);
    }
}
