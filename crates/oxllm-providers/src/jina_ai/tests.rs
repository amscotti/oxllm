use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{EmbeddingProvider, Provider, RerankProvider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "jina-embeddings-v3".to_string(),
        oxllm_params: OxllmParams {
            model: "jina_ai/jina-embeddings-v3".to_string(),
            api_key: Some("jina-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

// =============================================================================
// Ported from the reference `tests/llm_translation/test_jina_ai.py`.
//
// The reference Jina suite is tiny and live-call based (it hits the network for a
// base embedding/rerank). The ports below adapt the observable transform
// behaviors to `transform_request` / `transform_response`: the embedding
// request carries the resolved model, input, `task`, and `dimensions`; the
// rerank request forwards supported params.
// =============================================================================

/// Ported from `test_jina_ai_embedding`: the embedding request must carry the
/// resolved model, the input (wrapped if scalar), `task`, and `dimensions`.
#[test]
fn embedding_request_forwards_task_and_dimensions() {
    let p = JinaAIProvider;
    let req = json!({
        "model": "jina_ai/jina-embeddings-v3",
        "input": ["a"],
        "task": "separation",
        "dimensions": 1024,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "jina-embeddings-v3");
    assert_eq!(out["input"], json!(["a"]));
    assert_eq!(out["task"], "separation");
    assert_eq!(out["dimensions"], 1024);
}

/// Ported from `TestJinaAI.get_base_rerank_call_args` / base rerank: the rerank
/// request must carry the resolved model, query, documents, and top_n.
#[test]
fn rerank_request_forwards_supported_params() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "jina_ai/jina-reranker-v2-base-multilingual".to_string();
    let req = json!({
        "model": "jina-reranker-v2-base-multilingual",
        "query": "Hello there!",
        "documents": ["Hello there!", "How are you?", "How do you do?"],
        "top_n": 2,
        "return_documents": true,
        "rank_fields": ["title"],
        "max_chunks_per_doc": 3,
    });
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "jina-reranker-v2-base-multilingual");
    assert_eq!(out["query"], "Hello there!");
    assert_eq!(
        out["documents"],
        json!(["Hello there!", "How are you?", "How do you do?"])
    );
    assert_eq!(out["top_n"], 2);
    assert_eq!(out["return_documents"], true);
    assert!(
        out.get("rank_fields").is_none(),
        "rank_fields is not a supported Jina rerank param"
    );
    assert!(
        out.get("max_chunks_per_doc").is_none(),
        "max_chunks_per_doc is not a supported Jina rerank param"
    );
}

// =============================================================================
// Core transform tests (URL construction, auth, input normalization, response).
// =============================================================================

#[test]
fn embedding_url_uses_default_jina_base() {
    let p = JinaAIProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.jina.ai/v1/embeddings"
    );
}

#[test]
fn embedding_url_respects_custom_base() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://custom.example.com/v1/embeddings"
    );
}

#[test]
fn rerank_url_uses_default_jina_base() {
    let p = JinaAIProvider;
    assert_eq!(p.rerank_url(&deployment()), "https://api.jina.ai/v1/rerank");
}

#[test]
fn rerank_url_appends_v1_rerank_to_bare_custom_base() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/rerank");
}

#[test]
fn rerank_url_appends_rerank_to_v1_base() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/rerank");
}

#[test]
fn rerank_url_passes_through_full_endpoint() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/rerank".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/rerank");
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = JinaAIProvider;
    let creds = ResolvedCredentials::from_key("jina-secret");
    let mut headers = std::collections::HashMap::new();
    EmbeddingProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer jina-secret")
    );
}

#[test]
fn embedding_request_wraps_scalar_input_in_array() {
    let p = JinaAIProvider;
    let req = json!({"model": "jina-embeddings-v3", "input": "hello"});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["input"], json!(["hello"]));
}

#[test]
fn embedding_request_passes_array_input_through() {
    let p = JinaAIProvider;
    let req = json!({"model": "jina-embeddings-v3", "input": ["hello", "world"]});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["input"], json!(["hello", "world"]));
}

/// A missing or null `input` must not produce `input: []` in the upstream
/// body — matching the Voyage transform. Jina rejects an empty array with a
/// less helpful error than omitting the field entirely.
#[test]
fn embedding_request_omits_input_when_null() {
    let p = JinaAIProvider;
    let req = json!({"model": "jina-embeddings-v3"});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("input").is_none(),
        "missing/null input must not be forwarded"
    );
}

#[test]
fn transform_request_overrides_request_model_with_deployment_model() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "jina_ai/jina-embeddings-v2".to_string();
    let req = json!({
        "model": "caller-supplied-ignored",
        "input": "hello",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "jina-embeddings-v2");
}

#[test]
fn embedding_response_passes_through_with_resolved_model() {
    let p = JinaAIProvider;
    let resp = json!({
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "ignored-by-transform",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "jina-embeddings-v3").expect("ok");
    assert_eq!(out["object"], "list");
    assert_eq!(out["model"], "jina-embeddings-v3");
    let data = out["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(out["usage"]["prompt_tokens"], 5);
}

#[test]
fn rerank_request_omits_top_n_when_null() {
    let p = JinaAIProvider;
    let req = json!({
        "model": "jina-reranker-v2",
        "query": "q",
        "documents": ["a"],
        "top_n": null,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("top_n").is_none(),
        "null top_n must not be forwarded"
    );
}

/// A wildcard rerank deployment (`jina_ai/*`) must take the model from the
/// request body (prefix-stripped) rather than sending the literal `*`
/// upstream, mirroring `OpenAiCompatProvider::resolve_model`.
#[test]
fn rerank_request_with_wildcard_deployment_uses_request_model() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "jina_ai/*".to_string();
    let req = json!({
        "model": "jina-reranker-v2",
        "query": "q",
        "documents": ["a"],
    });
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "jina-reranker-v2");
}

/// A wildcard rerank deployment whose request carries no `model` falls back
/// to the wildcard literal.
#[test]
fn rerank_request_with_wildcard_deployment_without_model_falls_back() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "jina_ai/*".to_string();
    let req = json!({"query": "q", "documents": ["a"]});
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "*");
}

/// A wildcard embeddings deployment (`jina_ai/*`) must take the model from
/// the request body (prefix-stripped) rather than sending the literal `*`
/// upstream as the model id — same rule as the rerank path.
#[test]
fn embedding_request_with_wildcard_deployment_uses_request_model() {
    let p = JinaAIProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "jina_ai/*".to_string();
    let req = json!({
        "model": "jina_ai/jina-clip-v2",
        "input": ["hello"],
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "jina-clip-v2");

    // No request model → wildcard literal forwarded (same fallback as
    // rerank), so the misconfiguration surfaces upstream.
    let req = json!({"input": ["hello"]});
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "*");
}

/// Ported from Jina rerank `transform_rerank_response`: Jina returns results
/// under a `results` key with `document` as a string; wrap it in
/// `{ "text": … }`. Dicts pass through unchanged.
#[test]
fn rerank_response_wraps_string_documents() {
    let p = JinaAIProvider;
    let resp = json!({
        "id": "jina-1",
        "results": [
            {"index": 0, "relevance_score": 0.72, "document": "hello"},
            {"index": 1, "relevance_score": 0.30, "document": {"text": "world"}},
        ],
        "usage": {"total_tokens": 10},
    });
    let out = RerankProvider::transform_response(&p, &resp, "jina-reranker-v2").expect("ok");
    assert_eq!(out["id"], "jina-1");
    let results = out["results"].as_array().expect("results");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["index"], 0);
    assert_eq!(results[0]["relevance_score"], 0.72);
    assert_eq!(results[0]["document"]["text"], "hello");
    assert_eq!(results[1]["document"]["text"], "world");
    assert_eq!(out["usage"]["total_tokens"], 10);
}

#[test]
fn rerank_response_synthesizes_id_when_absent() {
    let p = JinaAIProvider;
    let resp = json!({
        "results": [{"index": 0, "relevance_score": 0.5}],
    });
    let out = RerankProvider::transform_response(&p, &resp, "jina-reranker-v2").expect("ok");
    let id = out["id"].as_str().expect("id");
    assert!(id.starts_with("jina-rerank-"));
}

#[test]
fn rerank_response_synthesized_ids_are_unique_and_non_sequential() {
    // Two synthesized ids (when upstream omits `id`) must differ and not be
    // sequential — a v4 UUID is random, unlike the old process-global counter
    // which collided across replicas starting at 0.
    let p = JinaAIProvider;
    let resp = json!({"results": [{"index": 0, "relevance_score": 0.5}]});
    let id1 = RerankProvider::transform_response(&p, &resp, "m").expect("ok")["id"]
        .as_str()
        .expect("id")
        .to_string();
    let id2 = RerankProvider::transform_response(&p, &resp, "m").expect("ok")["id"]
        .as_str()
        .expect("id")
        .to_string();
    assert_ne!(id1, id2);
}

#[test]
fn rerank_response_missing_results_surfaces_error() {
    // A response missing the `results` array is malformed/error and must
    // surface as an error rather than a fake empty success.
    let p = JinaAIProvider;
    let resp = json!({"usage": {"total_tokens": 0}});
    let err = RerankProvider::transform_response(&p, &resp, "jina-reranker-v2")
        .expect_err("should error");
    assert!(matches!(err, oxllm_core::Error::Transform(_)));
}

#[test]
fn provider_id_and_supports() {
    let p = JinaAIProvider;
    assert_eq!(p.id(), "jina_ai");
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.supports(Endpoint::Rerank));
    assert!(!p.supports(Endpoint::ChatCompletions));
    assert!(p.embeddings().is_some());
    assert!(p.rerank().is_some());
    assert!(p.chat().is_none());
}

#[test]
fn strip_jina_prefix_handles_unknown() {
    assert_eq!(
        strip_jina_prefix("jina_ai/jina-embeddings-v3"),
        "jina-embeddings-v3"
    );
    assert_eq!(
        strip_jina_prefix("jina-embeddings-v3"),
        "jina-embeddings-v3"
    );
}
