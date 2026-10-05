use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{EmbeddingProvider, Provider, RerankProvider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "voyage-3-lite".to_string(),
        oxllm_params: OxllmParams {
            model: "voyage/voyage-3-lite".to_string(),
            api_key: Some("voyage-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

// =============================================================================
// Ported from LiteLLM `tests/llm_translation/test_voyage_ai.py`.
//
// LiteLLM's Voyage suite is heavily mock-based (it patches `litellm.embedding`
// and asserts the Python call surface) and covers contextual embeddings, which
// are out of scope for the oxllm transform surface. The ports below adapt the
// transformable behaviors to `transform_request` / `transform_response`:
// embedding request shape (model override, param renaming), response token
// mapping, and rerank request/response translation.
// =============================================================================

/// Ported from `test_voyage_ai_embedding_extra_params`: the embedding request
/// must carry the resolved model, the input, and forward `dimensions`. Voyage
/// renames `dimensions` to `output_dimension`.
#[test]
fn embedding_request_maps_dimensions_to_output_dimension() {
    let p = VoyageProvider;
    let req = json!({
        "model": "voyage/voyage-3-lite",
        "input": ["a"],
        "dimensions": 512,
        "input_type": "document",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "voyage-3-lite");
    assert_eq!(out["input"], json!(["a"]));
    assert_eq!(out["output_dimension"], 512);
    assert_eq!(
        out["input_type"], "document",
        "input_type must be forwarded to Voyage (asymmetric retrieval depends on it)"
    );
    assert!(
        out.get("dimensions").is_none(),
        "dimensions must be renamed to output_dimension"
    );
}

/// Voyage-specific embeddings params (`input_type`, `truncation`) are forwarded
/// when present and non-null.
#[test]
fn embedding_request_forwards_input_type_and_truncation() {
    let p = VoyageProvider;
    let req = json!({
        "input": ["a", "b"],
        "input_type": "query",
        "truncation": true,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["input_type"], "query");
    assert_eq!(out["truncation"], true);
}

/// A missing `input` field must not produce `input: null` in the upstream body:
/// the upstream would reject it with a less helpful error than forwarding no
/// field at all.
#[test]
fn embedding_request_omits_input_when_null() {
    let p = VoyageProvider;
    let req = json!({"model": "voyage-3"});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("input").is_none(),
        "missing/null input must not be forwarded as null"
    );
}

/// Ported from `test_voyage_ai_embedding_prompt_token_mapping`: Voyage returns
/// `usage.total_tokens` only; the response must map it to both `prompt_tokens`
/// and `total_tokens`.
#[test]
fn embedding_response_maps_total_tokens_to_prompt_tokens() {
    let p = VoyageProvider;
    let resp = json!({
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "voyage-3-lite",
        "usage": {"total_tokens": 120},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "voyage-3-lite").expect("ok");
    assert_eq!(out["object"], "list");
    assert_eq!(out["model"], "voyage-3-lite");
    let data = out["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(out["usage"]["prompt_tokens"], 120);
    assert_eq!(out["usage"]["total_tokens"], 120);
}

/// A whole-number float `total_tokens` (some JSON encoders emit token counts
/// as floats) must still map to the integer value, not be silently zeroed by
/// `Value::as_u64` (which returns `None` for any JSON float). This is a
/// billing-accuracy guard: the cost hook reads `prompt_tokens` from the
/// mapped response.
#[test]
fn embedding_response_handles_float_total_tokens() {
    let p = VoyageProvider;
    let resp = json!({
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
        "usage": {"total_tokens": 120.0},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "voyage-3-lite").expect("ok");
    assert_eq!(
        out["usage"]["prompt_tokens"], 120,
        "float total_tokens must coerce to 120, not 0"
    );
    assert_eq!(out["usage"]["total_tokens"], 120);
}

/// Same float-coercion guard for the rerank response path.
#[test]
fn rerank_response_handles_float_total_tokens() {
    let p = VoyageProvider;
    let resp = json!({
        "data": [{"index": 0, "relevance_score": 0.5}],
        "usage": {"total_tokens": 42.0},
    });
    let out = RerankProvider::transform_response(&p, &resp, "rerank-2").expect("ok");
    assert_eq!(out["usage"]["total_tokens"], 42);
}

// =============================================================================
// Core transform tests (request shape, URL construction, auth, rerank).
// =============================================================================

#[test]
fn embedding_url_uses_default_voyage_base() {
    let p = VoyageProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.voyageai.com/v1/embeddings"
    );
}

#[test]
fn embedding_url_appends_endpoint_to_custom_base() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://custom.example.com/v1/embeddings"
    );
}

#[test]
fn embedding_url_passes_through_base_already_ending_with_endpoint() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/embeddings".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://custom.example.com/v1/embeddings"
    );
}

#[test]
fn embedding_url_trims_trailing_slash() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://custom.example.com/v1/embeddings"
    );
}

/// A bare custom base (no `/v1` path) is normalised to `/v1/embeddings`,
/// matching `rerank_url`'s convention so the two endpoints stay symmetric.
#[test]
fn embedding_url_appends_v1_to_bare_custom_base() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://custom.example.com/v1/embeddings"
    );
}

#[test]
fn rerank_url_uses_default_voyage_base() {
    let p = VoyageProvider;
    assert_eq!(
        p.rerank_url(&deployment()),
        "https://api.voyageai.com/v1/rerank"
    );
}

#[test]
fn rerank_url_appends_v1_rerank_to_bare_custom_base() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/rerank");
}

#[test]
fn rerank_url_appends_rerank_to_v1_base() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/rerank");
}

#[test]
fn rerank_url_passes_through_full_endpoint() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/rerank".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/rerank");
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = VoyageProvider;
    let creds = ResolvedCredentials::from_key("voyage-secret");
    let mut headers = std::collections::HashMap::new();
    EmbeddingProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer voyage-secret")
    );
}

#[test]
fn request_strips_voyage_prefix_from_model() {
    let p = VoyageProvider;
    let req = json!({"model": "voyage-3-lite", "input": ["hello", "world"]});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "voyage-3-lite");
    assert_eq!(out["input"], json!(["hello", "world"]));
}

#[test]
fn embedding_request_passes_through_encoding_format() {
    let p = VoyageProvider;
    let req = json!({
        "model": "voyage-3-lite",
        "input": ["a"],
        "encoding_format": "float",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["encoding_format"], "float");
}

#[test]
fn transform_request_overrides_request_model_with_deployment_model() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "voyage/voyage-3".to_string();
    let req = json!({
        "model": "caller-supplied-ignored",
        "input": "hello",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "voyage-3");
}

/// Ported from Voyage rerank `map_cohere_rerank_params`: Voyage renames
/// `top_n` to `top_k` and only forwards `query`, `documents`, `top_n`, and
/// `return_documents`.
#[test]
fn rerank_request_renames_top_n_to_top_k() {
    let p = VoyageProvider;
    let req = json!({
        "model": "rerank-2",
        "query": "What is AI?",
        "documents": ["AI is ...", "Cats are nice"],
        "top_n": 2,
        "return_documents": true,
        "rank_fields": ["title"],
        "max_chunks_per_doc": 3,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "voyage-3-lite");
    assert_eq!(out["query"], "What is AI?");
    assert_eq!(out["documents"], json!(["AI is ...", "Cats are nice"]));
    assert_eq!(out["top_k"], 2);
    assert!(out.get("top_n").is_none(), "top_n must be renamed to top_k");
    assert_eq!(out["return_documents"], true);
    assert!(
        out.get("rank_fields").is_none(),
        "rank_fields is not a supported Voyage rerank param"
    );
    assert!(
        out.get("max_chunks_per_doc").is_none(),
        "max_chunks_per_doc is not a supported Voyage rerank param"
    );
}

#[test]
fn rerank_request_top_n_override_used_when_top_n_absent() {
    let p = VoyageProvider;
    let req = json!({
        "model": "rerank-2",
        "query": "q",
        "documents": ["a"],
        "top_n_override": 5,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["top_k"], 5);
    assert!(out.get("top_n").is_none());
}

#[test]
fn rerank_request_omits_top_k_when_null() {
    let p = VoyageProvider;
    let req = json!({
        "model": "rerank-2",
        "query": "q",
        "documents": ["a"],
        "top_n": null,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("top_k").is_none(),
        "null top_n must not be forwarded"
    );
}

/// A wildcard rerank deployment (`voyage/*`) must take the model from the
/// request body (prefix-stripped) rather than sending the literal `*`
/// upstream, mirroring `OpenAiCompatProvider::resolve_model`.
#[test]
fn rerank_request_with_wildcard_deployment_uses_request_model() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "voyage/*".to_string();
    let req = json!({
        "model": "rerank-2",
        "query": "q",
        "documents": ["a"],
    });
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "rerank-2");
}

/// A wildcard rerank deployment whose request carries no `model` falls back
/// to the wildcard literal — matching the shared compat layer's behaviour.
#[test]
fn rerank_request_with_wildcard_deployment_without_model_falls_back() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "voyage/*".to_string();
    let req = json!({"query": "q", "documents": ["a"]});
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "*");
}

/// A wildcard embeddings deployment (`voyage/*`) must take the model from
/// the request body (prefix-stripped) rather than sending the literal `*`
/// upstream as the model id — same rule as the rerank path.
#[test]
fn embedding_request_with_wildcard_deployment_uses_request_model() {
    let p = VoyageProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "voyage/*".to_string();
    let req = json!({
        "model": "voyage/voyage-3-large",
        "input": ["hello"],
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "voyage-3-large");

    // No request model → wildcard literal forwarded (same fallback as
    // rerank), so the misconfiguration surfaces upstream.
    let req = json!({"input": ["hello"]});
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "*");
}

/// Ported from Voyage rerank `transform_rerank_response`: Voyage returns
/// results under a `data` key (not `results`); each document string is wrapped
/// in `{ "text": … }`, dicts pass through.
#[test]
fn rerank_response_maps_data_key_to_results_and_wraps_documents() {
    let p = VoyageProvider;
    let resp = json!({
        "id": "voyage-1",
        "data": [
            {"index": 0, "relevance_score": 0.9, "document": "hello"},
            {"index": 1, "relevance_score": 0.2, "document": {"text": "world"}},
        ],
        "usage": {"total_tokens": 42},
    });
    let out = RerankProvider::transform_response(&p, &resp, "rerank-2").expect("ok");
    assert_eq!(out["id"], "voyage-1");
    let results = out["results"].as_array().expect("results");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["index"], 0);
    assert_eq!(results[0]["relevance_score"], 0.9);
    // string document wrapped in {"text": ...}
    assert_eq!(results[0]["document"]["text"], "hello");
    // dict document passes through unchanged
    assert_eq!(results[1]["document"]["text"], "world");
    assert_eq!(out["usage"]["total_tokens"], 42);
}

#[test]
fn rerank_response_synthesizes_id_when_absent() {
    let p = VoyageProvider;
    let resp = json!({
        "data": [{"index": 0, "relevance_score": 0.5}],
        "usage": {"total_tokens": 1},
    });
    let out = RerankProvider::transform_response(&p, &resp, "rerank-2").expect("ok");
    assert_eq!(out["id"], "voyage-rerank-rerank-2");
}

#[test]
fn rerank_response_missing_data_surfaces_error() {
    // A response missing the `data` array is malformed/error and must
    // surface as an error rather than a fake empty success.
    let p = VoyageProvider;
    let resp = json!({"usage": {"total_tokens": 0}});
    let err = RerankProvider::transform_response(&p, &resp, "rerank-2").expect_err("should error");
    assert!(matches!(err, oxllm_core::Error::Transform(_)));
}

#[test]
fn provider_id_and_supports() {
    let p = VoyageProvider;
    assert_eq!(p.id(), "voyage");
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.supports(Endpoint::Rerank));
    assert!(!p.supports(Endpoint::ChatCompletions));
    assert!(p.embeddings().is_some());
    assert!(p.rerank().is_some());
    assert!(p.chat().is_none());
}

#[test]
fn strip_voyage_prefix_handles_unknown() {
    assert_eq!(strip_voyage_prefix("voyage/voyage-3-lite"), "voyage-3-lite");
    assert_eq!(strip_voyage_prefix("voyage-3-lite"), "voyage-3-lite");
}
