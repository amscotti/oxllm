use super::*;

#[test]
fn validation_vertex_additive_reasoning_is_normalized_once_for_json_and_sse() {
    for (usage, expected_completion) in [
        (
            json!({"prompt_tokens":5, "completion_tokens":1, "total_tokens":11,
            "completion_tokens_details":{"reasoning_tokens":5}}),
            6,
        ),
        (
            json!({"prompt_tokens":4, "completion_tokens":0, "total_tokens":7,
            "completion_tokens_details":{"reasoning_tokens":3}}),
            3,
        ),
        (
            json!({"prompt_tokens":4, "total_tokens":7,
            "completion_tokens_details":{"reasoning_tokens":3}}),
            3,
        ),
        (
            json!({"prompt_tokens":5, "completion_tokens":6, "total_tokens":11,
            "completion_tokens_details":{"reasoning_tokens":5}}),
            6,
        ),
        (json!({"prompt_tokens":21, "total_tokens":21}), 0),
    ] {
        let body = json!({"model":"gemini-3-flash", "choices":[], "usage":usage});
        let out =
            ChatProvider::transform_response(&VertexProvider, &body, "vertex_ai/gemini-3-flash")
                .unwrap();
        assert_eq!(
            out["usage"]["completion_tokens"], expected_completion,
            "{usage}"
        );
        assert_eq!(out["usage"]["total_tokens"], usage["total_tokens"]);
        assert_eq!(
            out["usage"]["completion_tokens_details"],
            usage["completion_tokens_details"]
        );
        assert_eq!(
            ChatProvider::transform_response(&VertexProvider, &out, "vertex_ai/gemini-3-flash")
                .unwrap(),
            out,
            "an already-normalized endpoint must not bill reasoning twice"
        );
        let raw = format!("data: {body}\n\n");
        let streamed = VertexProvider
            .transform_stream_chunk_stateful(&raw, &mut None, &json!({"model":"gemini-3-flash"}))
            .unwrap()
            .unwrap();
        assert_eq!(streamed["usage"], out["usage"]);
    }
}

#[test]
fn validation_vertex_unknown_or_overflowing_usage_is_not_folded() {
    for usage in [
        json!({"service_tier":"default"}),
        json!({"total_tokens":21}),
        json!({"prompt_tokens":21, "total_tokens":21, "completion_tokens_details":{"reasoning_tokens":5}}),
        json!({"prompt_tokens":21, "total_tokens":21, "completion_tokens_details":{"reasoning_tokens":"unknown"}}),
        json!({"prompt_tokens":5, "completion_tokens":1, "total_tokens":99, "completion_tokens_details":{"reasoning_tokens":5}}),
        json!({"prompt_tokens":u64::MAX, "completion_tokens":1, "total_tokens":u64::MAX, "completion_tokens_details":{"reasoning_tokens":1}}),
    ] {
        let body = json!({"usage":usage});
        let out =
            ChatProvider::transform_response(&VertexProvider, &body, "vertex_ai/gemini-3-flash")
                .unwrap();
        assert_eq!(out["usage"], usage);
        let raw = format!("data: {body}\n\n");
        let streamed = VertexProvider
            .transform_stream_chunk(&raw)
            .unwrap()
            .unwrap();
        assert_eq!(streamed["usage"], usage);
    }
}

#[test]
fn validation_vertex_locations_remain_lowercase_path_safe_names_with_custom_bases() {
    let request = json!({"messages":[{"role":"user", "content":"hello"}], "max_tokens":32});
    for model in ["vertex_ai/gemini-3.8-flash", "vertex_ai/claude-haiku-5-5"] {
        for api_base in [None, Some("https://gateway.example.test")] {
            for location in ["us", "eu", "global", "us-east5", "europe-west4"] {
                let deployment: Deployment = serde_json::from_value(json!({"model_name":"alias", "oxllm_params":{
                    "model":model, "api_base":api_base, "vertex_project":"project", "vertex_location":location
                }})).unwrap();
                assert!(
                    ChatProvider::transform_request(&VertexProvider, &request, &deployment).is_ok()
                );
            }
            for location in [
                "US",
                "EU",
                "Us-East5",
                "",
                "../other",
                "us-east5?other",
                "us_east5",
            ] {
                let deployment: Deployment = serde_json::from_value(json!({"model_name":"alias", "oxllm_params":{
                    "model":model, "api_base":api_base, "vertex_project":"project", "vertex_location":location
                }})).unwrap();
                let error = ChatProvider::transform_request(&VertexProvider, &request, &deployment)
                    .unwrap_err();
                assert!(matches!(error, oxllm_core::Error::Config(_)));
                assert!(error.to_string().contains("vertex_location"));
            }
        }
    }
}

#[test]
fn validation_vertex_multiline_anthropic_sse_uses_stateful_messages_transform() {
    let request = json!({"model": "vertex_ai/claude-sonnet-4-5"});
    let mut state = None;
    let start = VertexProvider.transform_stream_chunk_stateful(
        "event: message_start\rdata: {\"type\":\"message_start\",\rdata: \"message\":{\"id\":\"vertex_message\",\"model\":\"claude-sonnet-4-5\",\"usage\":{}}}\r\r",
        &mut state, &request,
    ).unwrap().unwrap();
    let delta = VertexProvider.transform_stream_chunk_stateful(
        "event: content_block_delta\rdata: {\"type\":\"content_block_delta\",\rdata: \"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\r\r",
        &mut state, &request,
    ).unwrap().unwrap();
    assert_eq!(start["id"], "vertex_message");
    assert_eq!(delta["id"], start["id"]);
    assert_eq!(delta["created"], start["created"]);
    assert_eq!(delta["choices"][0]["delta"]["content"], "hello");
    assert_eq!(delta["object"], "chat.completion.chunk");
    assert!(delta.get("type").is_none());
}

#[test]
fn validation_vertex_multiline_gemini_sse_keeps_openai_shape() {
    let chunk = VertexProvider.transform_stream_chunk(
        "event: message\rdata: {\"id\":\"gemini_chunk\",\rdata: \"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\r\r",
    ).unwrap().unwrap();
    assert_eq!(chunk["id"], "gemini_chunk");
    assert_eq!(chunk["choices"][0]["delta"]["content"], "hello");
}

#[test]
fn validation_vertex_google_models_are_qualified_once_for_fixed_and_wildcard_routes() {
    use oxllm_core::deployment::OxllmParams;
    for selector in [
        "gemini-3-flash-preview",
        "google/gemini-3-flash-preview",
        "publishers/google/models/gemini-3-flash-preview",
    ] {
        for wildcard in [false, true] {
            let mut deployment = Deployment {
                oxllm_params: OxllmParams {
                    model: if wildcard {
                        "vertex_ai/*".into()
                    } else {
                        format!("vertex_ai/{selector}")
                    },
                    ..Default::default()
                },
                ..Default::default()
            };
            deployment
                .oxllm_params
                .extra
                .insert("vertex_project".into(), json!("project"));
            let request = json!({"model": format!("vertex_ai/{selector}"), "messages": [{"role": "user", "content": "hello"}]});
            let body =
                ChatProvider::transform_request(&VertexProvider, &request, &deployment).unwrap();
            assert_eq!(body["model"], "google/gemini-3-flash-preview");
            assert!(
                VertexProvider
                    .endpoint_url_for_request(&deployment, &request)
                    .ends_with("/endpoints/openapi/chat/completions")
            );
        }
    }
}

#[test]
fn validation_qualified_vertex_claude_keeps_a_single_safe_model_path_segment() {
    use oxllm_core::deployment::OxllmParams;
    for selector in [
        "anthropic/claude-sonnet-4-5@20250929",
        "publishers/anthropic/models/claude-sonnet-4-5@20250929",
    ] {
        let mut deployment = Deployment {
            oxllm_params: OxllmParams {
                model: format!("vertex_ai/{selector}"),
                ..Default::default()
            },
            ..Default::default()
        };
        deployment
            .oxllm_params
            .extra
            .insert("vertex_project".into(), json!("project"));
        let request = json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32});
        let body = ChatProvider::transform_request(&VertexProvider, &request, &deployment).unwrap();
        assert!(body.get("model").is_none());
        assert_eq!(body["anthropic_version"], VERTEX_ANTHROPIC_VERSION);
        assert!(
            VertexProvider
                .endpoint_url_for_request(&deployment, &request)
                .ends_with("/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict")
        );
    }
    let mut wildcard = Deployment {
        oxllm_params: OxllmParams {
            model: "vertex_ai/*".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    wildcard
        .oxllm_params
        .extra
        .insert("vertex_project".into(), json!("project"));
    for selector in [
        "google/../other",
        "publishers/google/models/gemini/other",
        "anthropic/claude?injected=1",
    ] {
        assert!(
            ChatProvider::transform_request(
                &VertexProvider,
                &json!({"model": format!("vertex_ai/{selector}"), "messages": []}),
                &wildcard
            )
            .is_err()
        );
    }
}
