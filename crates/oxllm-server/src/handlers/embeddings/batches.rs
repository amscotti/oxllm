use oxllm_core::UsageSummary;
use oxllm_core::provider::EmbeddingRequestBatch;
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::orchestration::unary::UnaryUpstreamResponse;

type BatchResponse = (Vec<usize>, UnaryUpstreamResponse);

/// Validate provider mappings before any upstream call can consume inputs.
pub(super) fn validate_plan(
    batches: &[EmbeddingRequestBatch],
    count: usize,
) -> Result<(), ApiError> {
    let mut mapped = vec![false; count];
    for batch in batches {
        let indices = &batch.input_indices;
        let local_count = match batch.upstream.body.get("input") {
            Some(Value::Array(items)) => items.len(),
            Some(Value::String(_)) => 1,
            _ => 0,
        };
        if indices.is_empty() || indices.len() != local_count {
            return Err(ApiError::internal("invalid embedding batch cardinality"));
        }
        for &index in indices {
            let slot = mapped
                .get_mut(index)
                .ok_or_else(|| ApiError::internal("embedding batch index is out of bounds"))?;
            if std::mem::replace(slot, true) {
                return Err(ApiError::internal(
                    "embedding batch input is mapped more than once",
                ));
            }
        }
    }
    if batches.is_empty() || !mapped.into_iter().all(|mapped| mapped) {
        return Err(ApiError::internal(
            "embedding batches do not cover all inputs",
        ));
    }
    Ok(())
}

pub(super) fn combine(
    responses: Vec<BatchResponse>,
    input_count: usize,
) -> Result<UnaryUpstreamResponse, ApiError> {
    combine_with_limit(
        responses,
        input_count,
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )
}

fn combine_with_limit(
    responses: Vec<BatchResponse>,
    input_count: usize,
    cap: usize,
) -> Result<UnaryUpstreamResponse, ApiError> {
    let mut ordered = vec![None; input_count];
    let mut usage: Option<UsageSummary> = None;
    let mut cost = Some(0.0f64);
    let mut combined = None;
    let mut response_headers = Default::default();
    for (indices, (mut body, batch_usage, headers)) in responses {
        let data = body
            .get_mut("data")
            .and_then(Value::as_array_mut)
            .filter(|data| data.len() == indices.len())
            .ok_or_else(invalid_response)?;
        for mut item in data.drain(..) {
            let index = item
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .and_then(|index| indices.get(index).copied())
                .ok_or_else(invalid_response)?;
            if item
                .get("embedding")
                .is_none_or(|value| !value.is_array() && !value.is_string())
            {
                return Err(invalid_response());
            }
            let slot = ordered.get_mut(index).ok_or_else(invalid_response)?;
            if slot.is_some() {
                return Err(invalid_response());
            }
            item["index"] = json!(index);
            *slot = Some(item);
        }
        if let Some(batch_usage) = batch_usage {
            let total = usage.get_or_insert_with(UsageSummary::default);
            total.prompt_tokens = total
                .prompt_tokens
                .saturating_add(batch_usage.prompt_tokens);
            total.total_tokens = total.total_tokens.saturating_add(batch_usage.total_tokens);
        }
        cost = cost
            .zip(
                body.pointer("/usage/cost")
                    .and_then(Value::as_f64)
                    .filter(|cost| cost.is_finite() && *cost >= 0.0),
            )
            .map(|(total, batch)| total + batch)
            .filter(|cost| cost.is_finite());
        combined.get_or_insert(body);
        // Sequential batches share a resource, so its last quota snapshot is
        // the relevant one. Token counts were already read per response.
        response_headers = headers;
    }
    let data = ordered
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(invalid_response)?;
    let mut body = combined.ok_or_else(invalid_response)?;
    body["data"] = Value::Array(data);
    body.as_object_mut()
        .ok_or_else(invalid_response)?
        .remove("usage");
    if let Some(usage) = &usage {
        body["usage"] =
            json!({"prompt_tokens":usage.prompt_tokens, "total_tokens":usage.total_tokens});
    }
    if let Some(cost) = cost {
        if body.get("usage").is_none() {
            body["usage"] = json!({});
        }
        body["usage"]["cost"] = json!(cost);
    }
    struct LimitedWriter {
        len: usize,
        cap: usize,
    }
    impl std::io::Write for LimitedWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.len = self.len.saturating_add(bytes.len());
            if self.len > self.cap {
                return Err(std::io::Error::other("embedding response limit"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(LimitedWriter { len: 0, cap }, &body).map_err(|_| {
        ApiError::upstream(
            502,
            format!("combined embedding response exceeds {cap} byte limit"),
        )
    })?;
    Ok((body, usage, response_headers))
}

fn invalid_response() -> ApiError {
    ApiError::upstream(
        502,
        "upstream embedding response does not match its input batch",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    fn vector(index: usize, value: Value) -> Value {
        json!({"index":index, "object":"embedding", "embedding":value})
    }

    fn batch(indices: Vec<usize>, data: Value, tokens: u64) -> BatchResponse {
        (
            indices,
            (
                json!({"model":"test-model", "object":"list", "data":data}),
                (tokens > 0).then_some(UsageSummary {
                    prompt_tokens: tokens,
                    total_tokens: tokens,
                    ..Default::default()
                }),
                HeaderMap::new(),
            ),
        )
    }

    #[test]
    fn combined_embeddings_restore_original_indices_and_sum_known_usage() {
        let text = batch(
            vec![1, 3],
            json!([vector(1, json!([4.0])), vector(0, json!([2.0]))]),
            7,
        );
        let mut images = batch(
            vec![0, 2],
            json!([vector(1, json!("encoded-3")), vector(0, json!("encoded-1"))]),
            11,
        );
        images.1.2.insert(
            "x-ratelimit-remaining-requests",
            HeaderValue::from_static("8"),
        );
        let (body, usage, headers) = combine(vec![text, images], 4).unwrap();
        assert_eq!(
            body["data"],
            json!([
                vector(0, json!("encoded-1")),
                vector(1, json!([2.0])),
                vector(2, json!("encoded-3")),
                vector(3, json!([4.0]))
            ])
        );
        assert_eq!(
            body["usage"],
            json!({"prompt_tokens":18, "total_tokens":18})
        );
        assert_eq!(usage.unwrap().prompt_tokens, 18);
        assert_eq!(headers["x-ratelimit-remaining-requests"], "8");
    }

    #[test]
    fn combined_receipts_are_complete_or_omitted_never_a_single_legs_cost() {
        for second_cost in [Some(0.02), None] {
            let mut first = batch(vec![0], json!([vector(0, json!([1]))]), 2);
            first.1.0["usage"] = json!({"cost":0.01});
            let mut second = batch(vec![1], json!([vector(0, json!([2]))]), 3);
            if let Some(cost) = second_cost {
                second.1.0["usage"] = json!({"cost":cost});
            }
            let (body, _, _) = combine(vec![first, second], 2).unwrap();
            assert_eq!(
                body.pointer("/usage/cost").and_then(Value::as_f64),
                second_cost.map(|cost| cost + 0.01)
            );
        }
    }

    #[test]
    fn unmetered_batches_do_not_invent_a_zero_usage_signal() {
        let (body, usage, _) =
            combine(vec![batch(vec![0], json!([vector(0, json!([1]))]), 0)], 1).unwrap();
        assert!(usage.is_none());
        assert!(body.get("usage").is_none());
    }

    #[test]
    fn batch_usage_addition_saturates_instead_of_overflowing() {
        let (_, usage, _) = combine(
            vec![
                batch(vec![0], json!([vector(0, json!([1]))]), u64::MAX),
                batch(vec![1], json!([vector(0, json!([2]))]), 10),
            ],
            2,
        )
        .unwrap();
        assert_eq!(usage.unwrap().prompt_tokens, u64::MAX);
    }

    #[test]
    fn combined_response_size_is_bounded_even_when_each_part_fits() {
        let part = json!([vector(0, json!("x".repeat(512)))]);
        let error = combine_with_limit(
            vec![batch(vec![0], part.clone(), 2), batch(vec![1], part, 3)],
            2,
            1000,
        )
        .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert!(error.message.contains("1000 byte limit"));
    }

    #[test]
    fn malformed_batch_responses_cannot_misattribute_embeddings() {
        for data in [
            json!(null),
            json!([]),
            json!([vector(0, json!([1]))]),
            json!([vector(0, json!([1])), vector(0, json!([2]))]),
            json!([vector(0, json!([1])), vector(2, json!([2]))]),
            json!([vector(0, json!([1])), {"index":-1, "embedding":[2]}]),
            json!([vector(0, json!([1])), {"embedding":[2]}]),
            json!([vector(0, json!([1])), {"index":1}]),
            json!([vector(0, json!([1])), vector(1, json!(null))]),
        ] {
            let error = combine(vec![batch(vec![0, 1], data.clone(), 1)], 2).unwrap_err();
            assert_eq!(error.status, StatusCode::BAD_GATEWAY, "{data}");
        }
    }

    #[test]
    fn input_mappings_are_a_partition_before_any_http_call() {
        fn prepared(body: Value) -> oxllm_core::EmbeddingRequest {
            oxllm_core::EmbeddingRequest {
                url: "https://example.test/embeddings".into(),
                body,
                params: Value::Null,
                allowed_fields: None,
            }
        }
        for indices in [
            vec![],
            vec![vec![]],
            vec![vec![0]],
            vec![vec![0, 0]],
            vec![vec![0, 2]],
        ] {
            let batches = indices
                .into_iter()
                .map(|indices| EmbeddingRequestBatch {
                    upstream: prepared(json!({"input":vec!["item"; indices.len()]})),
                    input_indices: indices,
                })
                .collect::<Vec<_>>();
            assert!(validate_plan(&batches, 2).is_err());
        }
        let batches = vec![EmbeddingRequestBatch {
            upstream: prepared(json!({"input":["second", "first"]})),
            input_indices: vec![1, 0],
        }];
        validate_plan(&batches, 2).unwrap();
    }
}
