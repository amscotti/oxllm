use std::borrow::Cow;

use oxllm_core::provider::{
    EmbeddingRequest, EmbeddingRequestBatch, EmbeddingRequestPlan, is_base64_data_uri,
};
use oxllm_core::{Deployment, Error, Result};
use serde_json::{Value, json};

fn is_cohere_model(model: &str) -> bool {
    let model = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    model.strip_prefix("embed-v").is_some_and(|suffix| {
        suffix
            .trim_start_matches('-')
            .starts_with(|c: char| c.is_ascii_digit())
    }) || [
        "embed-english-v",
        "embed-multilingual-v",
        "cohere-embed-",
        "offer-cohere-embed-",
    ]
    .iter()
    .any(|prefix| model.starts_with(prefix))
}

fn is_cohere(deployment: &Deployment) -> bool {
    is_cohere_model(&deployment.oxllm_params.model)
        || deployment
            .model_info
            .as_ref()
            .and_then(|info| info.base_model.as_deref())
            .or_else(|| {
                deployment
                    .oxllm_params
                    .extra
                    .get("base_model")
                    .and_then(Value::as_str)
            })
            .is_some_and(is_cohere_model)
}

/// All well-formed migration inputs agree; malformed multi-comma payloads are
/// deliberately stricter, validating the entire remainder rather than a prefix.
fn is_image(value: &Value) -> bool {
    value.as_str().is_some_and(is_base64_data_uri)
}

fn normalize_owned_input(request: &mut Value) {
    if let Some(input) = request.get_mut("input").filter(|input| input.is_string()) {
        *input = Value::Array(vec![std::mem::take(input)]);
    }
}

pub(super) fn normalize_input(request: &Value) -> Cow<'_, Value> {
    if request.get("input").is_some_and(Value::is_string) {
        let mut request = request.clone();
        normalize_owned_input(&mut request);
        Cow::Owned(request)
    } else {
        Cow::Borrowed(request)
    }
}

pub(super) fn has_image_input(request: &Value, deployment: &Deployment) -> bool {
    is_cohere(deployment)
        && match request.get("input") {
            Some(Value::Array(items)) => items.iter().any(is_image),
            Some(input) => is_image(input),
            None => false,
        }
}

pub(super) fn is_image_request(request: &Value, deployment: &Deployment) -> bool {
    is_cohere(deployment)
        && match request.get("input") {
            Some(Value::Array(items)) => !items.is_empty() && items.iter().all(is_image),
            Some(input) => is_image(input),
            None => false,
        }
}

pub(super) fn prepare_requests(
    mut request: Value,
    deployment: &Deployment,
) -> Result<EmbeddingRequestPlan> {
    normalize_owned_input(&mut request);
    let classified = if is_cohere(deployment) {
        request
            .get("input")
            .and_then(Value::as_array)
            .map(|inputs| inputs.iter().map(is_image).collect::<Vec<_>>())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    if !classified.iter().any(|&image| image) {
        let body = crate::openai_compat::build_embedding_request(&request, wire_model(deployment));
        return Ok(EmbeddingRequestPlan::Single(EmbeddingRequest {
            url: crate::util::foundry_endpoint_url(deployment, "embeddings", false),
            body,
            params: request,
            allowed_fields: None,
        }));
    }
    let inputs = request["input"]
        .as_array_mut()
        .expect("image inputs were normalized");
    if !inputs.iter().all(Value::is_string) {
        return Err(Error::BadRequest(
            "image embedding batches require string inputs".into(),
        ));
    }
    let mut text = Vec::new();
    let mut images = Vec::new();
    let mut text_indices = Vec::new();
    let mut image_indices = Vec::new();
    for (index, (input, image)) in std::mem::take(inputs)
        .into_iter()
        .zip(classified)
        .enumerate()
    {
        if image {
            images.push(input);
            image_indices.push(index);
        } else {
            text.push(input);
            text_indices.push(index);
        }
    }
    request
        .as_object_mut()
        .expect("embedding request is an object")
        .remove("input");
    let image_body = image_body_from_inputs(images, &request, deployment);
    let mut batches = Vec::with_capacity(2);
    if !text.is_empty() {
        let mut body =
            crate::openai_compat::build_embedding_request(&request, wire_model(deployment));
        body["input"] = Value::Array(text);
        batches.push(EmbeddingRequestBatch {
            upstream: EmbeddingRequest {
                url: crate::util::foundry_endpoint_url(deployment, "embeddings", false),
                body,
                params: request,
                allowed_fields: None,
            },
            input_indices: text_indices,
        });
    }
    batches.push(EmbeddingRequestBatch {
        upstream: EmbeddingRequest {
            url: image_url(deployment),
            body: image_body,
            params: Value::Null,
            allowed_fields: Some(&["model", "input", "dimensions", "encoding_format"]),
        },
        input_indices: image_indices,
    });
    Ok(EmbeddingRequestPlan::Batched(batches))
}

fn wire_model(deployment: &Deployment) -> &str {
    // Requests already resolved the optional router wrapper. Strip only the
    // provider prefix here, so a selector is never normalized twice.
    deployment
        .oxllm_params
        .model
        .strip_prefix("azure_ai/")
        .unwrap_or(&deployment.oxllm_params.model)
}

pub(super) fn model_selector(model: &str) -> &str {
    let model = model.strip_prefix("azure_ai/").unwrap_or(model);
    model
        .strip_prefix("model_router/")
        .or_else(|| model.strip_prefix("model-router/"))
        .unwrap_or(model)
}

pub(super) fn image_url(deployment: &Deployment) -> String {
    if deployment
        .oxllm_params
        .api_version
        .as_deref()
        .is_none_or(str::is_empty)
    {
        let mut deployment = deployment.clone();
        deployment.oxllm_params.api_version = Some("2024-05-01-preview".into());
        crate::util::foundry_endpoint_url(&deployment, "images/embeddings", false)
    } else {
        crate::util::foundry_endpoint_url(deployment, "images/embeddings", false)
    }
}

fn image_body_from_inputs(input: Vec<Value>, request: &Value, deployment: &Deployment) -> Value {
    let input = input
        .into_iter()
        .map(|image| Value::Object(serde_json::Map::from_iter([("image".into(), image)])))
        .collect();
    let mut body = json!({"model":wire_model(deployment)});
    body["input"] = Value::Array(input);
    for key in ["dimensions", "encoding_format"] {
        if let Some(value) = request.get(key).filter(|value| !value.is_null()) {
            body[key] = value.clone();
        }
    }
    body
}

pub(super) fn image_body(request: &Value, deployment: &Deployment) -> Value {
    image_body_from_inputs(
        request["input"]
            .as_array()
            .expect("image inputs were normalized")
            .clone(),
        request,
        deployment,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD};

    #[test]
    fn allocation_free_classifier_matches_the_standard_base64_engine() {
        for len in 0..128 {
            let encoded = STANDARD.encode((0..len).map(|i| (i * 17) as u8).collect::<Vec<_>>());
            for payload in [
                &encoded[..],
                encoded.trim_end_matches('='),
                &format!("{encoded}="),
                &format!("{encoded},extra"),
            ] {
                assert_eq!(
                    is_base64_data_uri(&format!("data:image/png;base64,{payload}")),
                    STANDARD.decode(payload).is_ok(),
                    "{payload}"
                );
            }
        }
        for a in b"Aa09+/=" {
            for b in b"Aa09+/=" {
                for c in b"Aa09+/=" {
                    for d in b"Aa09+/=" {
                        let bytes = [*a, *b, *c, *d];
                        let payload = std::str::from_utf8(&bytes).unwrap();
                        assert_eq!(
                            is_base64_data_uri(&format!("data:,{payload}")),
                            STANDARD.decode(payload).is_ok(),
                            "{payload}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn image_classifier_requires_a_data_prefix_and_canonical_base64() {
        for (input, expected) in [
            ("data:image/png;base64,aGVsbG8=", true),
            ("data:image/jpeg;base64,/9j/", true),
            ("data:application/octet-stream;base64,YWJj", true),
            ("data:image/png,aGVsbG8=", true),
            ("data:;base64,YWJj", true),
            ("data:image/png;base64,", true),
            ("aGVsbG8=", false),
            ("Dog", false),
            ("", false),
            ("https://example.test/image.png", false),
            ("DATA:image/png;base64,YWJj", false),
            ("data:image/png;base64", false),
            ("data:image/png;base64,invalid!", false),
            ("data:image/png;base64,aGVsbG8", false),
            ("data:image/png;base64,aGVsbG9=", false),
            ("data:image/png;base64,aGVsbG8=,extra", false),
            ("data:image/png;base64,aGVs\nbG8=", false),
            ("data:text/plain,hello%20world", false),
        ] {
            assert_eq!(is_image(&json!(input)), expected, "{input:?}");
        }
        assert!(!is_image(&Value::Null));
        assert!(!is_image(&json!(123)));
    }
}
